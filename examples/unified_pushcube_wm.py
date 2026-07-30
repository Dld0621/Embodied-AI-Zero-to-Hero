"""
Unified PushCube — World Model Track
=====================================
Train a tiny world model on PushCube:
  Input:  current state (13-D) + action (2-D)
  Output: next state (13-D) + reward (scalar)

State (13-D): [arm_x, arm_y,
               cube1_x, cube1_y, cube2_x, cube2_y,
               target_x, target_y,
               cube1_r, cube1_g, cube2_r, cube2_g,
               active_idx]

Uses an MLP dynamics model (teaching implementation).
Data collection mixes expert demonstrations and random actions
for better coverage of the state-action space.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from unified_pushcube_env import PushCubeEnv, expert_action


def train_wm(args):
    """Train a tiny MLP world model."""
    print("=" * 70)
    print(" Unified PushCube — World Model Training")
    print("=" * 70)

    try:
        import torch
        import torch.nn as nn
        import torch.nn.functional as F
    except ImportError:
        print("[Error] PyTorch required. Install: pip install torch")
        sys.exit(1)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ------------------------------------------------------------------
    # Get dimensions from environment
    # ------------------------------------------------------------------
    _env_tmp = PushCubeEnv()
    state_dim = _env_tmp.state_dim   # 13
    action_dim = _env_tmp.action_dim  # 2
    print(f"\nState dim: {state_dim}, Action dim: {action_dim}")

    # ------------------------------------------------------------------
    # Collect rollout data (mix of expert and random for better coverage)
    # ------------------------------------------------------------------
    def collect_rollouts(n_episodes, seed=42):
        data = []
        rng = np.random.RandomState(seed)
        n_expert = 0
        n_random = 0

        for ep in range(n_episodes):
            env = PushCubeEnv()
            obs = env.reset(seed=ep)
            state = env.get_state_vector()

            # Alternate between expert and random episodes (50/50 mix)
            use_expert = (ep % 2 == 0)

            for _ in range(env.max_steps):
                if use_expert:
                    action = expert_action(env)
                    n_expert += 1
                else:
                    action = rng.uniform(-1, 1, size=2).astype(np.float32)
                    n_random += 1

                next_obs, reward, done, truncated, info = env.step(action)
                next_state = env.get_state_vector()

                data.append({
                    "state": state,
                    "action": action,
                    "next_state": next_state,
                    "reward": reward,
                    "done": float(done),
                })

                state = next_state
                if done or truncated:
                    break

        print(f"  Expert transitions: {n_expert}, Random transitions: {n_random}")
        return data

    print(f"\nCollecting {args.n_episodes} rollout episodes "
          f"(expert + random mix)...")
    data = collect_rollouts(args.n_episodes, seed=args.seed)
    print(f"  Collected {len(data)} transitions")

    # Split train/val
    split = int(len(data) * 0.8)
    train_data = data[:split]
    val_data = data[split:]

    # Convert to tensors
    def to_tensor(batch, key):
        return torch.tensor(
            np.stack([d[key] for d in batch]), dtype=torch.float32
        ).to(device)

    train_s = to_tensor(train_data, "state")
    train_a = to_tensor(train_data, "action")
    train_ns = to_tensor(train_data, "next_state")
    train_r = to_tensor(train_data, "reward").unsqueeze(-1)
    train_d = to_tensor(train_data, "done").unsqueeze(-1)

    val_s = to_tensor(val_data, "state")
    val_a = to_tensor(val_data, "action")
    val_ns = to_tensor(val_data, "next_state")
    val_r = to_tensor(val_data, "reward").unsqueeze(-1)

    # ------------------------------------------------------------------
    # MLP World Model
    # ------------------------------------------------------------------
    class TinyWorldModel(nn.Module):
        def __init__(self, state_dim=13, action_dim=2):
            super().__init__()
            self.encoder = nn.Sequential(
                nn.Linear(state_dim + action_dim, 64),
                nn.ReLU(),
                nn.Linear(64, 64),
                nn.ReLU(),
            )
            self.next_state_head = nn.Linear(64, state_dim)
            self.reward_head = nn.Linear(64, 1)

        def forward(self, state, action):
            x = torch.cat([state, action], dim=-1)
            h = self.encoder(x)
            next_state = self.next_state_head(h)
            reward = self.reward_head(h)
            return next_state, reward

    model = TinyWorldModel(state_dim=state_dim, action_dim=action_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    # ------------------------------------------------------------------
    # Train
    # ------------------------------------------------------------------
    print(f"\nTraining for {args.epochs} epochs...")
    best_val_loss = float("inf")

    for epoch in range(args.epochs):
        # Train
        model.train()
        perm = torch.randperm(len(train_s))
        epoch_loss = 0.0
        n_batches = 0

        for i in range(0, len(train_s), args.batch_size):
            idx = perm[i:i + args.batch_size]
            s, a, ns, r = train_s[idx], train_a[idx], train_ns[idx], train_r[idx]

            pred_ns, pred_r = model(s, a)
            loss = F.mse_loss(pred_ns, ns) + F.mse_loss(pred_r, r)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            n_batches += 1

        avg_train_loss = epoch_loss / max(n_batches, 1)

        # Val
        model.eval()
        with torch.no_grad():
            pred_ns, pred_r = model(val_s, val_a)
            val_loss = (
                F.mse_loss(pred_ns, val_ns).item()
                + F.mse_loss(pred_r, val_r).item()
            )

        # Print roughly 10 progress lines
        print_interval = max(1, args.epochs // 10)
        if (epoch + 1) % print_interval == 0 or epoch == 0:
            print(f"  Epoch {epoch+1}/{args.epochs}: "
                  f"train_loss={avg_train_loss:.4f}, val_loss={val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss

    print(f"\nTraining complete. Best val loss: {best_val_loss:.4f}")

    # ------------------------------------------------------------------
    # Multi-step prediction evaluation
    # ------------------------------------------------------------------
    print(f"\nMulti-step prediction evaluation ({args.n_test} test episodes)...")
    model.eval()

    # First 6 state dims: [arm_x, arm_y, cube1_x, cube1_y, cube2_x, cube2_y]
    POS_SLICE = 6
    multistep_errors = {}

    for horizon in [1, 5, 10]:
        errors = []

        for ep in range(args.n_test):
            env = PushCubeEnv()
            obs = env.reset(seed=2000 + ep)
            state = env.get_state_vector()

            for step in range(horizon):
                action = np.random.uniform(-1, 1, size=2).astype(np.float32)

                with torch.no_grad():
                    s_t = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(device)
                    a_t = torch.tensor(action, dtype=torch.float32).unsqueeze(0).to(device)
                    pred_ns, _ = model(s_t, a_t)
                    pred_state = pred_ns.cpu().numpy()[0]

                # Step true environment
                _, _, done, truncated, _ = env.step(action)
                true_state = env.get_state_vector()

                # Open-loop: feed predicted state back for next step
                state = pred_state if step < horizon - 1 else true_state

                if step == horizon - 1:
                    # Position error on first 6 dims (arm + both cubes)
                    err = np.linalg.norm(
                        pred_state[:POS_SLICE] - true_state[:POS_SLICE]
                    )
                    errors.append(err)

                if done or truncated:
                    break

        mean_err = float(np.mean(errors)) if errors else 0.0
        multistep_errors[f"H{horizon}"] = round(mean_err, 4)
        print(f"  H={horizon}: mean position error (first {POS_SLICE} dims) "
              f"= {mean_err:.4f}")

    # ------------------------------------------------------------------
    # Save model and results
    # ------------------------------------------------------------------
    save_dir = Path(args.output_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), save_dir / "pushcube_wm.pt")
    print(f"\nModel saved to {save_dir / 'pushcube_wm.pt'}")

    results = {
        "task": "PushCube World Model",
        "state_dim": state_dim,
        "n_episodes": args.n_episodes,
        "epochs": args.epochs,
        "n_test": args.n_test,
        "seed": args.seed,
        "smoke_test": args.smoke_test,
        "best_val_loss": round(best_val_loss, 4),
        "multistep_errors": multistep_errors,
    }
    results_path = save_dir / "wm_results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to {results_path}")

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Unified PushCube — World Model Track"
    )
    parser.add_argument("--n-episodes", type=int, default=200,
                        help="Number of rollout episodes (default: 200)")
    parser.add_argument("--epochs", type=int, default=50,
                        help="Training epochs (default: 50)")
    parser.add_argument("--batch-size", type=int, default=64,
                        help="Batch size (default: 64)")
    parser.add_argument("--n-test", type=int, default=50,
                        help="Test episodes for multi-step prediction (default: 50)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed (default: 42)")
    parser.add_argument("--output-dir", type=str,
                        default="../results/unified_pushcube/wm",
                        help="Output directory")
    parser.add_argument("--smoke-test", action="store_true",
                        help="Run smoke test (5 episodes, 5 epochs, 10 test) for CI")
    args = parser.parse_args()

    if args.smoke_test:
        args.n_episodes = 5
        args.epochs = 5
        args.n_test = 10
        print("[SMOKE TEST MODE] 5 episodes, 5 epochs, 10 test episodes")

    train_wm(args)


if __name__ == "__main__":
    main()
