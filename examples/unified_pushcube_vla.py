"""
Unified PushCube — VLA Track (Dual-Cube, Language-Conditioned)
==============================================================
Train a tiny vision-language-action policy on the dual-cube PushCube task.

  Input:  128x128 RGB image + tokenized language instruction
  Output: 2-D action (arm movement)

Because two colored cubes are on the table and only the *active* cube
(identified by language) must be pushed to the target, a vision-only
policy cannot disambiguate which cube to push.

Three evaluation conditions are reported:
  (a) Full VLA          — trained on correct language, eval with correct language
  (b) Language-shuffled — trained on distractor language, eval with correct language
                          (demonstrates the policy actually uses the color word)
  (c) Vision-only       — main policy evaluated with the language token zeroed out
                          (language-dropout ablation at eval time)

A `--smoke-test` flag shrinks the run (2 episodes / 2 epochs / 2 eval) for CI.
`--ablation` (default True) toggles the language-shuffled ablation experiment.
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
    """All-pad tokens — used for the vision-only (no-language) ablation."""
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
def collect_episodes(n_episodes, shuffled=False, verbose=True):
    """
    Roll out expert_action(env) and record (image, tokens, action) frames.

    If `shuffled` is True, the language token comes from
    env.get_shuffled_language() (the distractor cube's instruction) instead of
    env.get_language_instruction(). The expert action still pushes the *active*
    cube, so the shuffled data teaches a deliberately wrong color->action map.
    """
    images, tokens, actions = [], [], []
    for ep in range(n_episodes):
        env = PushCubeEnv()
        env.reset(seed=ep)
        lang = env.get_shuffled_language() if shuffled else env.get_language_instruction()
        tok = tokenize(lang)
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
# Evaluation
# ----------------------------------------------------------------------
def evaluate(policy, n_eval, device, lang_mode="correct", seed_offset=1000):
    """
    Roll out `policy` for `n_eval` fresh episodes and return success rate.

    lang_mode:
      "correct"  -> tokenize(env.get_language_instruction())
      "shuffled" -> tokenize(env.get_shuffled_language())
      "none"     -> zero_tokens()  (vision-only / language-dropout ablation)
    """
    import torch
    policy.eval()
    success = 0
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
                if done:
                    success += 1
                    break
                if truncated:
                    break
    return success / max(1, n_eval)


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
                        help="Run language-shuffled ablation (default True; "
                             "use --no-ablation to skip)")
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
    print(f"\n[1/4] Collecting {args.n_episodes} expert demos (correct language)...")
    imgs, toks, acts = collect_episodes(args.n_episodes, shuffled=False)
    print(f"      {len(imgs)} frames collected")

    # --- 2. Train Full-VLA policy ---
    print(f"\n[2/4] Training Full-VLA policy ({args.epochs} epochs)...")
    main_policy = train_policy(imgs, toks, acts, args.epochs,
                               args.batch_size, device, tag="full")

    # --- 3. (Optional) Language-shuffled ablation policy ---
    ablation_policy = None
    if args.ablation:
        print(f"\n[3/4] Training Language-Shuffled ablation policy "
              f"({args.n_episodes} demos, {args.epochs} epochs)...")
        imgs_s, toks_s, acts_s = collect_episodes(args.n_episodes, shuffled=True)
        print(f"      {len(imgs_s)} shuffled-language frames collected")
        ablation_policy = train_policy(imgs_s, toks_s, acts_s, args.epochs,
                                       args.batch_size, device, tag="shuffled")
    else:
        print("\n[3/4] Language-shuffled ablation skipped (--no-ablation)")

    # --- 4. Evaluate all conditions ---
    print(f"\n[4/4] Evaluating ({args.n_eval} episodes each)...")
    full_rate = evaluate(main_policy, args.n_eval, device, lang_mode="correct")
    print(f"  (a) Full VLA success rate:          {full_rate * 100:5.1f}%")

    if ablation_policy is not None:
        # Trained on distractor language, evaluated with the *correct* language.
        # A policy that truly uses the color word will push the wrong cube here.
        shuffled_rate = evaluate(ablation_policy, args.n_eval, device,
                                 lang_mode="correct")
        print(f"  (b) Language-shuffled success rate: {shuffled_rate * 100:5.1f}%")
    else:
        shuffled_rate = None

    vision_rate = evaluate(main_policy, args.n_eval, device, lang_mode="none")
    print(f"  (c) Vision-only success rate:        {vision_rate * 100:5.1f}%")

    # --- Save model + results ---
    save_dir = Path(args.output_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    torch.save(main_policy.state_dict(), save_dir / "pushcube_vla_policy.pt")
    if ablation_policy is not None:
        torch.save(ablation_policy.state_dict(),
                   save_dir / "pushcube_vla_policy_shuffled.pt")

    results = {
        "task": "PushCube VLA (dual-cube, language-conditioned)",
        "n_episodes": args.n_episodes,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "n_eval": args.n_eval,
        "smoke_test": args.smoke_test,
        "ablation_run": args.ablation,
        "success_rate_full_vla": round(full_rate * 100, 1),
        "success_rate_language_shuffled": (round(shuffled_rate * 100, 1)
                                           if shuffled_rate is not None else None),
        "success_rate_vision_only": round(vision_rate * 100, 1),
    }
    with open(save_dir / "vla_results.json", "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nModel  -> {save_dir / 'pushcube_vla_policy.pt'}")
    print(f"Results-> {save_dir / 'vla_results.json'}")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
