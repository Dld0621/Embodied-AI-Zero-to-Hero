"""
Unified PushCube — VLA Track (Dual-Cube, Language-Conditioned)
==============================================================
Train a tiny vision-language-action policy on the dual-cube PushCube task.

  Input:  128x128 RGB image + tokenized language instruction
  Output: 2-D action (arm movement)

Because two colored cubes are on the table and only the *active* cube
(identified by language) must be pushed to the target, a vision-only
policy cannot disambiguate which cube to push.

Ablation design (single-model, same-episode evaluation)
-------------------------------------------------------
Instead of training a *separate* model on shuffled language (which
confounds training data, random init, and optimisation trajectory),
we train ONE Full-VLA model and evaluate it under three language
conditions on the *same* set of evaluation episodes:

  (a) Full VLA + correct language  — the model should push the correct cube.
  (b) Full VLA + swapped language  — the model should push the *wrong* cube
      (proves it actually reads the colour word, not just memorises positions).
  (c) Full VLA + zeroed language   — language-dropout test at inference time.

A separately *trained* Vision-Only baseline (d) is also included: it is
trained from scratch with all language tokens zeroed, so it never sees
language during training.  This is a stronger control than (c) because
it cannot rely on any language-conditional feature learned during training.

For every condition we report three metrics:
  - correct_success  : active cube ended in the target zone.
  - wrong_success    : the *other* cube ended in the target zone.
  - selection_accuracy: active cube closer to target than the other cube.

A `--smoke-test` flag shrinks the run (2 episodes / 2 epochs / 2 eval) for CI.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from unified_pushcube_env import (
    PushCubeEnv,
    expert_action,
    COLOR_NAMES,
    CUBE_COLORS,
)


# ----------------------------------------------------------------------
# Vocabulary & tokenization
# ----------------------------------------------------------------------
# Only the words the environment can produce are included. "red" and "green"
# are the two cube color words; both must be present so the policy can learn
# the color -> action mapping.
VOCAB = {
    "<pad>": 0, "push": 1, "the": 2, "red": 3, "green": 4,
    "cube": 5, "to": 6, "right": 7, "left": 8, "top": 9,
    "bottom": 10, "and": 11, "center": 12,
}
MAX_LEN = 10

# Sanity: every color the environment can emit must be in the vocab.
for _name in COLOR_NAMES:
    assert _name in VOCAB, f"Vocab missing color word: {_name}"


def tokenize(text):
    """Lowercase, split, map to ids, pad/truncate to MAX_LEN."""
    words = text.lower().replace(".", "").split()
    toks = [VOCAB.get(w, 0) for w in words]
    toks = toks[:MAX_LEN] + [0] * (MAX_LEN - len(toks))
    return toks


def zero_tokens():
    """All-pad tokens — used for the vision-only (no-language) condition."""
    return [0] * MAX_LEN


# ----------------------------------------------------------------------
# Tiny policy network: CNN (vision) + word-embedding (language) -> MLP
# ----------------------------------------------------------------------
def build_policy(device):
    import torch
    import torch.nn as nn

    class TinyVLAPolicy(nn.Module):
        def __init__(self, vocab_size=len(VOCAB), embed_dim=16, action_dim=2):
            super().__init__()
            # Vision encoder: 128x128 -> 8x8x8 feature map
            self.conv = nn.Sequential(
                nn.Conv2d(3, 8, 5, stride=2, padding=2),   # 64x64
                nn.ReLU(),
                nn.Conv2d(8, 16, 5, stride=2, padding=2),  # 32x32
                nn.ReLU(),
                nn.Conv2d(16, 16, 5, stride=2, padding=2), # 16x16
                nn.ReLU(),
                nn.Conv2d(16, 8, 5, stride=2, padding=2),  # 8x8
                nn.ReLU(),
            )
            self.vision_fc = nn.Linear(8 * 8 * 8, 32)

            # Language encoder: word embeddings averaged over the sentence
            self.word_embed = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
            self.lang_fc = nn.Linear(embed_dim, 16)

            # Fusion + action head
            self.fusion = nn.Sequential(
                nn.Linear(32 + 16, 32),
                nn.ReLU(),
                nn.Linear(32, action_dim),
                nn.Tanh(),  # action in [-1, 1]
            )

        def forward(self, image, text_tokens):
            # image: (B, 3, 128, 128)
            x = self.conv(image)
            x = x.reshape(x.size(0), -1)   # .reshape (not .view) for non-contiguous
            v = self.vision_fc(x)          # (B, 32)

            # text_tokens: (B, seq_len)
            w = self.word_embed(text_tokens).mean(dim=1)  # (B, embed_dim)
            l = self.lang_fc(w)                            # (B, 16)

            fused = torch.cat([v, l], dim=-1)
            return self.fusion(fused)

    return TinyVLAPolicy().to(device)


# ----------------------------------------------------------------------
# Demonstration collection using the shared expert_action(env)
# ----------------------------------------------------------------------
def collect_episodes(n_episodes, shuffled=False, vision_only=False, verbose=True):
    """
    Roll out expert_action(env) and record (image, tokens, action) frames.

    Parameters
    ----------
    shuffled : bool
        If True, the language token comes from env.get_shuffled_language()
        (the distractor cube's instruction).  The expert action still pushes
        the *active* cube, so the shuffled data teaches a deliberately wrong
        color->action map.  (Kept for backward compatibility.)
    vision_only : bool
        If True, language tokens are zeroed — used to train the independently
        trained Vision-Only baseline.
    """
    images, tokens, actions = [], [], []
    for ep in range(n_episodes):
        env = PushCubeEnv()
        env.reset(seed=ep)
        if vision_only:
            tok = zero_tokens()
        elif shuffled:
            tok = tokenize(env.get_shuffled_language())
        else:
            tok = tokenize(env.get_language_instruction())
        for _ in range(env.max_steps):
            img = env.render(size=128).transpose(2, 0, 1)  # (3, 128, 128)
            action = expert_action(env)                     # (2,) float32
            images.append(img)
            tokens.append(tok)
            actions.append(action)
            obs, reward, done, truncated, info = env.step(action)
            if done or truncated:
                break
        if verbose and (ep + 1) % max(1, n_episodes // 5) == 0:
            print(f"      collected {ep+1}/{n_episodes} episodes, {len(images)} frames")
    return images, tokens, actions


# ----------------------------------------------------------------------
# Behavior-cloning training
# ----------------------------------------------------------------------
def train_policy(images, tokens, actions, epochs, batch_size, device, tag=""):
    import torch
    import torch.nn.functional as F

    policy = build_policy(device)
    optimizer = torch.optim.Adam(policy.parameters(), lr=1e-3)

    images_t = torch.tensor(np.stack(images), dtype=torch.float32).to(device)
    tokens_t = torch.tensor(np.array(tokens), dtype=torch.long).to(device)
    actions_t = torch.tensor(np.stack(actions), dtype=torch.float32).to(device)

    policy.train()
    for epoch in range(epochs):
        perm = torch.randperm(len(images_t))
        epoch_loss, n_batches = 0.0, 0
        for i in range(0, len(images_t), batch_size):
            idx = perm[i:i + batch_size]
            pred = policy(images_t[idx], tokens_t[idx])
            loss = F.mse_loss(pred, actions_t[idx])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            n_batches += 1
        if (epoch + 1) % max(1, epochs // 5) == 0 or epoch == 0:
            print(f"    [{tag}] epoch {epoch+1}/{epochs}: "
                  f"loss={epoch_loss / max(1, n_batches):.4f}")
    return policy


# ----------------------------------------------------------------------
# Evaluation (single-model, same-episode, multi-condition)
# ----------------------------------------------------------------------
def evaluate(policy, n_eval, device, lang_mode="correct", seed_offset=1000):
    """
    Roll out `policy` for `n_eval` fresh episodes and return a dict of
    three metrics computed at episode end.

    lang_mode:
      "correct"  -> tokenize(env.get_language_instruction())
      "shuffled" -> tokenize(env.get_shuffled_language())
      "none"     -> zero_tokens()  (vision-only / language-dropout)

    Returns
    -------
    dict with keys:
      correct_success   — % of episodes where the *active* cube is in target
      wrong_success     — % of episodes where the *other* cube is in target
      selection_accuracy— % of episodes where active cube is closer to target
    """
    import torch
    policy.eval()
    correct_count = 0
    wrong_count = 0
    selection_count = 0

    with torch.no_grad():
        for ep in range(n_eval):
            env = PushCubeEnv()
            env.reset(seed=seed_offset + ep)

            if lang_mode == "correct":
                tok = tokenize(env.get_language_instruction())
            elif lang_mode == "shuffled":
                tok = tokenize(env.get_shuffled_language())
            else:
                tok = zero_tokens()
            tok_t = torch.tensor([tok], dtype=torch.long).to(device)

            for _ in range(env.max_steps):
                img = torch.tensor(
                    env.render(size=128).transpose(2, 0, 1),
                    dtype=torch.float32,
                ).unsqueeze(0).to(device)
                action = policy(img, tok_t).cpu().numpy()[0]
                obs, reward, done, truncated, info = env.step(action)
                if done or truncated:
                    break

            # Measure both cubes at episode end
            active_cube = env.cube_positions[env.active_idx]
            other_cube = env.cube_positions[1 - env.active_idx]
            target = env.target_pos

            active_dist = float(np.linalg.norm(active_cube - target))
            other_dist = float(np.linalg.norm(other_cube - target))

            if active_dist < env.goal_threshold:
                correct_count += 1
            if other_dist < env.goal_threshold:
                wrong_count += 1
            if active_dist < other_dist:
                selection_count += 1

    n = max(1, n_eval)
    return {
        "correct_success": round(correct_count / n * 100, 1),
        "wrong_success": round(wrong_count / n * 100, 1),
        "selection_accuracy": round(selection_count / n * 100, 1),
    }


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Unified PushCube — VLA Track (dual-cube, language-conditioned)"
    )
    parser.add_argument("--n-episodes", type=int, default=100,
                        help="Number of expert demo episodes for BC")
    parser.add_argument("--epochs", type=int, default=30, help="Training epochs")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--n-eval", type=int, default=20, help="Eval episodes per condition")
    parser.add_argument("--output-dir", type=str,
                        default="../results/unified_pushcube/vla",
                        help="Output directory")
    parser.add_argument("--smoke-test", action="store_true",
                        help="CI mode: 2 episodes, 2 epochs, 2 eval episodes")
    parser.add_argument("--ablation", action=argparse.BooleanOptionalAction,
                        default=True,
                        help="Run language ablation (default True; "
                             "use --no-ablation to skip vision-only baseline)")
    args = parser.parse_args()

    if args.smoke_test:
        args.n_episodes = 2
        args.epochs = 2
        args.n_eval = 2

    print("=" * 72)
    print(" Unified PushCube — VLA Track (dual-cube, language-conditioned)")
    print("=" * 72)
    print(f"Cube colors: {COLOR_NAMES} -> {CUBE_COLORS}")

    try:
        import torch  # noqa: F401
    except ImportError:
        print("[Error] PyTorch required. Install: pip install torch")
        sys.exit(1)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}  | smoke_test={args.smoke_test}  | ablation={args.ablation}")

    # --- 1. Collect expert demonstrations (correct language) ---
    print(f"\n[1/5] Collecting {args.n_episodes} expert demos (correct language)...")
    imgs, toks, acts = collect_episodes(args.n_episodes, shuffled=False)
    print(f"      {len(imgs)} frames collected")

    # --- 2. Train Full-VLA policy ---
    print(f"\n[2/5] Training Full-VLA policy ({args.epochs} epochs)...")
    main_policy = train_policy(imgs, toks, acts, args.epochs,
                               args.batch_size, device, tag="full")

    # --- 3. (Optional) Train Vision-Only baseline (independently trained) ---
    vision_policy = None
    if args.ablation:
        print(f"\n[3/5] Training Vision-Only baseline "
              f"({args.n_episodes} demos, {args.epochs} epochs, zeroed tokens)...")
        imgs_v, toks_v, acts_v = collect_episodes(
            args.n_episodes, vision_only=True
        )
        print(f"      {len(imgs_v)} vision-only frames collected")
        vision_policy = train_policy(imgs_v, toks_v, acts_v, args.epochs,
                                     args.batch_size, device, tag="vision-only")
    else:
        print("\n[3/5] Vision-Only baseline skipped (--no-ablation)")

    # --- 4. Evaluate all conditions (single model, same episodes) ---
    print(f"\n[4/5] Evaluating Full-VLA under 3 language conditions "
          f"({args.n_eval} episodes each)...")

    # (a) Full VLA — correct language
    res_correct = evaluate(main_policy, args.n_eval, device, lang_mode="correct")
    print(f"  (a) Full VLA + correct lang:  "
          f"correct={res_correct['correct_success']:5.1f}%  "
          f"wrong={res_correct['wrong_success']:5.1f}%  "
          f"select={res_correct['selection_accuracy']:5.1f}%")

    # (b) Full VLA — swapped language (SAME model, SAME episodes)
    #     If the policy truly uses language, swapping the colour word should
    #     make it push the *wrong* cube: wrong_success rises, correct drops.
    res_shuffled = evaluate(main_policy, args.n_eval, device, lang_mode="shuffled")
    print(f"  (b) Full VLA + swapped lang:  "
          f"correct={res_shuffled['correct_success']:5.1f}%  "
          f"wrong={res_shuffled['wrong_success']:5.1f}%  "
          f"select={res_shuffled['selection_accuracy']:5.1f}%")

    # (c) Full VLA — zeroed language (language-dropout at inference)
    res_none = evaluate(main_policy, args.n_eval, device, lang_mode="none")
    print(f"  (c) Full VLA + zeroed lang:   "
          f"correct={res_none['correct_success']:5.1f}%  "
          f"wrong={res_none['wrong_success']:5.1f}%  "
          f"select={res_none['selection_accuracy']:5.1f}%")

    # --- 5. Evaluate Vision-Only trained baseline ---
    if vision_policy is not None:
        print(f"\n[5/5] Evaluating Vision-Only trained baseline "
              f"({args.n_eval} episodes)...")
        res_vision = evaluate(vision_policy, args.n_eval, device, lang_mode="none")
        print(f"  (d) Vision-only trained:      "
              f"correct={res_vision['correct_success']:5.1f}%  "
              f"wrong={res_vision['wrong_success']:5.1f}%  "
              f"select={res_vision['selection_accuracy']:5.1f}%")
    else:
        res_vision = None

    # --- Save model + results ---
    save_dir = Path(args.output_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    import torch
    torch.save(main_policy.state_dict(), save_dir / "pushcube_vla_policy.pt")
    if vision_policy is not None:
        torch.save(vision_policy.state_dict(),
                   save_dir / "pushcube_vla_policy_vision_only.pt")

    results = {
        "task": "PushCube VLA (dual-cube, language-conditioned)",
        "ablation_design": (
            "Single Full-VLA model evaluated with correct/swapped/zeroed "
            "language on identical episodes. Vision-only baseline is "
            "independently trained with zeroed language tokens."
        ),
        "n_episodes": args.n_episodes,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "n_eval": args.n_eval,
        "smoke_test": args.smoke_test,
        "conditions": {
            "full_vla_correct_lang": res_correct,
            "full_vla_swapped_lang": res_shuffled,
            "full_vla_zeroed_lang": res_none,
            "vision_only_trained": res_vision,
        },
    }
    with open(save_dir / "vla_results.json", "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nModel  -> {save_dir / 'pushcube_vla_policy.pt'}")
    print(f"Results-> {save_dir / 'vla_results.json'}")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
