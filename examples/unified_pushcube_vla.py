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
            # Vision encoder: 128x128 -> 8x8x16 feature map
            self.conv = nn.Sequential(
                nn.Conv2d(3, 16, 5, stride=2, padding=2),   # 64x64
                nn.BatchNorm2d(16),
                nn.ReLU(),
                nn.Conv2d(16, 32, 5, stride=2, padding=2),  # 32x32
                nn.BatchNorm2d(32),
                nn.ReLU(),
                nn.Conv2d(32, 32, 5, stride=2, padding=2),  # 16x16
                nn.BatchNorm2d(32),
                nn.ReLU(),
                nn.Conv2d(32, 16, 5, stride=2, padding=2), # 8x8
                nn.BatchNorm2d(16),
                nn.ReLU(),
            )
            self.vision_fc = nn.Linear(16 * 8 * 8, 64)
            self.vision_bn = nn.BatchNorm1d(64)

            # Language encoder: word embeddings averaged over the sentence
            self.word_embed = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
            self.lang_fc = nn.Linear(embed_dim, 16)

            # Fusion + action head
            self.fusion = nn.Sequential(
                nn.Linear(64 + 16, 64),
                nn.ReLU(),
                nn.Linear(64, action_dim),
                nn.Tanh(),  # action in [-1, 1]
            )

        def forward(self, image, text_tokens):
            # image: (B, 3, 128, 128)
            x = self.conv(image)
            x = x.reshape(x.size(0), -1)   # .reshape (not .view) for non-contiguous
            v = self.vision_fc(x)          # (B, 64)
            v = self.vision_bn(v)          # BatchNorm stabilizes vision features

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
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    images_t = torch.tensor(np.stack(images), dtype=torch.float32).to(device)
    tokens_t = torch.tensor(np.array(tokens), dtype=torch.long).to(device)
    actions_t = torch.tensor(np.stack(actions), dtype=torch.float32).to(device)

    policy.train()
    for epoch in range(epochs):
        perm = torch.randperm(len(images_t))
        epoch_loss, n_batches = 0.0, 0
        for i in range(0, len(images_t), batch_size):
            idx = perm[i:i + batch_size]
            if len(idx) < 2:
                continue  # BatchNorm needs >= 2 samples in train mode
            pred = policy(images_t[idx], tokens_t[idx])
            loss = F.mse_loss(pred, actions_t[idx])
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), max_norm=1.0)
            optimizer.step()
            epoch_loss += loss.item()
            n_batches += 1
        scheduler.step()
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
# State-BC baseline: state vector + language tokens -> action
# ----------------------------------------------------------------------
def build_state_policy(device):
    """State-BC policy: MLP with geometric feature engineering.

    Architecture: state(14) + computed_features(13) + lang_embed(16)
                  -> 128 -> 128 -> 2 -> tanh

    The model internally computes the expert's key decision variables
    (behind_proj, lateral, approach distance) from the raw 14-D state,
    making it much easier for the MLP to learn the three-phase pushing
    policy (flank, approach, push).
    """
    import torch
    import torch.nn as nn

    class StateBCPolicy(nn.Module):
        def __init__(self, state_dim=14, vocab_size=len(VOCAB),
                     embed_dim=16, action_dim=2):
            super().__init__()
            # Input normalization (state features have mixed scales)
            self.state_bn = nn.BatchNorm1d(state_dim)

            # Language encoder (same architecture as VLA)
            self.word_embed = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
            self.lang_fc = nn.Linear(embed_dim, 16)

            # Computed geometric features:
            #   arm_to_active(2), active_to_target(2), dist_arm(1),
            #   dist_target(1), push_dir(2) = 8
            # Plus expert decision variables:
            #   behind_proj(1), lateral(1), dist_arm_to_approach(1),
            #   approach_dir(2) = 5
            # Total extra_dim = 13
            extra_dim = 13

            # State + features + language fusion -> action
            self.net = nn.Sequential(
                nn.Linear(state_dim + extra_dim + 16, 128),
                nn.ReLU(),
                nn.Linear(128, 128),
                nn.ReLU(),
                nn.Linear(128, action_dim),
                nn.Tanh(),  # action in [-1, 1]
            )

        def forward(self, state, text_tokens):
            # state: (B, 14), text_tokens: (B, seq_len)
            # Extract state components
            arm = state[:, 0:2]
            cube0 = state[:, 2:4]
            cube1 = state[:, 4:6]
            target = state[:, 6:8]
            cube0_color = state[:, 8:10]
            cube1_color = state[:, 10:12]
            onehot = state[:, 12:14]

            # Differentiable active-cube selection via color matching.
            # cube0_diff > 0 means red, < 0 means green.
            # onehot_diff > 0 means active color is red.
            # Product is positive when cube0 matches the active color.
            cube0_diff = cube0_color[:, 0:1] - cube0_color[:, 1:2]
            onehot_diff = onehot[:, 0:1] - onehot[:, 1:2]
            cube0_match = torch.clamp((cube0_diff * onehot_diff + 0.7) / 1.4, 0, 1)
            cube1_match = 1.0 - cube0_match
            active_cube = cube0_match * cube0 + cube1_match * cube1

            # Push geometry (same as expert_action)
            active_to_target = active_cube - target
            dist_target = torch.norm(active_to_target, dim=-1, keepdim=True)
            push_dir = active_to_target / (dist_target + 1e-6)
            behind_dir = -push_dir
            perp = torch.stack([-push_dir[:, 1], push_dir[:, 0]], dim=-1)

            arm_to_active = arm - active_cube
            dist_arm = torch.norm(arm_to_active, dim=-1, keepdim=True)

            # Expert decision variables (these are what the expert uses to
            # decide its three phases):
            behind_proj = (arm_to_active * behind_dir).sum(dim=-1, keepdim=True)
            lateral = (arm_to_active * perp).sum(dim=-1, keepdim=True)

            # Approach point and distance to it
            cube_size = 0.08
            approach_offset = cube_size / 2 + 0.06
            approach_point = active_cube + behind_dir * approach_offset
            arm_to_approach = approach_point - arm
            dist_arm_to_approach = torch.norm(
                arm_to_approach, dim=-1, keepdim=True)
            approach_dir = arm_to_approach / (dist_arm_to_approach + 1e-6)

            s = self.state_bn(state)                       # normalized state
            w = self.word_embed(text_tokens).mean(dim=1)  # (B, embed_dim)
            l = self.lang_fc(w)                            # (B, 16)

            extra = torch.cat([
                arm_to_active,        # (2)
                active_to_target,    # (2)
                dist_arm,            # (1)
                dist_target,         # (1)
                push_dir,            # (2)
                behind_proj,         # (1)
                lateral,             # (1)
                dist_arm_to_approach,  # (1)
                approach_dir,        # (2)
            ], dim=-1)
            fused = torch.cat([s, extra, l], dim=-1)
            return self.net(fused)

    return StateBCPolicy().to(device)


def collect_state_episodes(n_episodes, shuffled=False, verbose=True):
    """Roll out expert_action(env) and record (state_vector, tokens, action) frames."""
    states, tokens, actions = [], [], []
    for ep in range(n_episodes):
        env = PushCubeEnv()
        env.reset(seed=ep)
        if shuffled:
            tok = tokenize(env.get_shuffled_language())
        else:
            tok = tokenize(env.get_language_instruction())
        for _ in range(env.max_steps):
            state = env.get_state_vector()
            action = expert_action(env)
            states.append(state)
            tokens.append(tok)
            actions.append(action)
            obs, reward, done, truncated, info = env.step(action)
            if done or truncated:
                break
        if verbose and (ep + 1) % max(1, n_episodes // 5) == 0:
            print(f"      collected {ep+1}/{n_episodes} state episodes, "
                  f"{len(states)} frames")
    return states, tokens, actions


def train_state_policy(states, tokens, actions, epochs, batch_size, device, tag=""):
    """Train the State-BC policy with cosine LR scheduling and gradient clipping."""
    import torch
    import torch.nn.functional as F

    policy = build_state_policy(device)
    optimizer = torch.optim.Adam(policy.parameters(), lr=3e-3)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs)

    states_t = torch.tensor(np.stack(states), dtype=torch.float32).to(device)
    tokens_t = torch.tensor(np.array(tokens), dtype=torch.long).to(device)
    actions_t = torch.tensor(np.stack(actions), dtype=torch.float32).to(device)

    policy.train()
    for epoch in range(epochs):
        perm = torch.randperm(len(states_t))
        epoch_loss, n_batches = 0.0, 0
        for i in range(0, len(states_t), batch_size):
            idx = perm[i:i + batch_size]
            if len(idx) < 2:
                continue  # BatchNorm needs >= 2 samples in train mode
            pred = policy(states_t[idx], tokens_t[idx])
            loss = F.mse_loss(pred, actions_t[idx])
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), max_norm=1.0)
            optimizer.step()
            epoch_loss += loss.item()
            n_batches += 1
        scheduler.step()
        if (epoch + 1) % max(1, epochs // 5) == 0 or epoch == 0:
            print(f"    [{tag}] epoch {epoch+1}/{epochs}: "
                  f"loss={epoch_loss / max(1, n_batches):.4f}")
    return policy


def evaluate_state(policy, n_eval, device, seed_offset=1000):
    """Roll out the State-BC policy and return success rate."""
    import torch
    policy.eval()
    success_count = 0

    with torch.no_grad():
        for ep in range(n_eval):
            env = PushCubeEnv()
            env.reset(seed=seed_offset + ep)
            tok = tokenize(env.get_language_instruction())
            tok_t = torch.tensor([tok], dtype=torch.long).to(device)

            for _ in range(env.max_steps):
                state = torch.tensor(
                    env.get_state_vector(), dtype=torch.float32
                ).unsqueeze(0).to(device)
                action = policy(state, tok_t).cpu().numpy()[0]
                obs, reward, done, truncated, info = env.step(action)
                if done or truncated:
                    break

            active_cube = env.cube_positions[env.active_idx]
            target = env.target_pos
            if np.linalg.norm(active_cube - target) < env.goal_threshold:
                success_count += 1

    n = max(1, n_eval)
    return {
        "success_rate": round(success_count / n * 100, 1),
    }


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Unified PushCube — VLA Track (dual-cube, language-conditioned)"
    )
    parser.add_argument("--n-episodes", type=int, default=200,
                        help="Number of expert demo episodes for BC")
    parser.add_argument("--epochs", type=int, default=50, help="Training epochs")
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

    # --- 1. Collect expert demonstrations (images, correct language) ---
    print(f"\n[1/8] Collecting {args.n_episodes} expert demos "
          f"(images, correct language)...")
    imgs, toks, acts = collect_episodes(args.n_episodes, shuffled=False)
    print(f"      {len(imgs)} image frames collected")

    # --- 2. Collect state episodes for State-BC ---
    print(f"\n[2/8] Collecting {args.n_episodes} state episodes for State-BC...")
    s_states, s_tokens, s_actions = collect_state_episodes(args.n_episodes)
    print(f"      {len(s_states)} state frames collected")

    # --- 3. Train Full-VLA policy ---
    print(f"\n[3/8] Training Full-VLA policy "
          f"({args.epochs} epochs, cosine LR, grad clip)...")
    main_policy = train_policy(imgs, toks, acts, args.epochs,
                               args.batch_size, device, tag="full")

    # --- 4. (Optional) Train Vision-Only baseline (independently trained) ---
    vision_policy = None
    if args.ablation:
        print(f"\n[4/8] Training Vision-Only baseline "
              f"({args.n_episodes} demos, {args.epochs} epochs, zeroed tokens)...")
        imgs_v, toks_v, acts_v = collect_episodes(
            args.n_episodes, vision_only=True
        )
        print(f"      {len(imgs_v)} vision-only frames collected")
        vision_policy = train_policy(imgs_v, toks_v, acts_v, args.epochs,
                                     args.batch_size, device, tag="vision-only")
    else:
        print("\n[4/8] Vision-Only baseline skipped (--no-ablation)")

    # --- 5. Train State-BC policy ---
    print(f"\n[5/8] Training State-BC policy "
          f"({args.epochs} epochs, cosine LR, grad clip)...")
    state_policy = train_state_policy(s_states, s_tokens, s_actions,
                                      args.epochs, args.batch_size, device,
                                      tag="state-bc")

    # --- 6. Evaluate Full-VLA under 3 language conditions ---
    print(f"\n[6/8] Evaluating Full-VLA under 3 language conditions "
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

    # --- 7. Evaluate Vision-Only trained baseline ---
    if vision_policy is not None:
        print(f"\n[7/8] Evaluating Vision-Only trained baseline "
              f"({args.n_eval} episodes)...")
        res_vision = evaluate(vision_policy, args.n_eval, device, lang_mode="none")
        print(f"  (d) Vision-only trained:      "
              f"correct={res_vision['correct_success']:5.1f}%  "
              f"wrong={res_vision['wrong_success']:5.1f}%  "
              f"select={res_vision['selection_accuracy']:5.1f}%")
    else:
        res_vision = None
        print("\n[7/8] Vision-Only evaluation skipped (--no-ablation)")

    # --- 8. Evaluate State-BC policy ---
    print(f"\n[8/8] Evaluating State-BC policy ({args.n_eval} episodes)...")
    res_state = evaluate_state(state_policy, args.n_eval, device)
    print(f"  (e) State-BC:                 "
          f"success={res_state['success_rate']:5.1f}%")

    # --- Save model + results ---
    save_dir = Path(args.output_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    import torch
    torch.save(main_policy.state_dict(), save_dir / "pushcube_vla_policy.pt")
    torch.save(state_policy.state_dict(),
               save_dir / "pushcube_state_bc_policy.pt")
    if vision_policy is not None:
        torch.save(vision_policy.state_dict(),
                   save_dir / "pushcube_vla_policy_vision_only.pt")

    results = {
        "task": "PushCube VLA (dual-cube, language-conditioned)",
        "ablation_design": (
            "Single Full-VLA model evaluated with correct/swapped/zeroed "
            "language on identical episodes. Vision-only baseline is "
            "independently trained with zeroed language tokens. State-BC "
            "baseline uses 14-D state vector + language tokens."
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
            "state_bc": res_state,
        },
    }
    with open(save_dir / "vla_results.json", "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nModel  -> {save_dir / 'pushcube_vla_policy.pt'}")
    print(f"State  -> {save_dir / 'pushcube_state_bc_policy.pt'}")
    print(f"Results-> {save_dir / 'vla_results.json'}")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
