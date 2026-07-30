"""
Unified PushCube — World Model Track
=====================================
Train a tiny world model on PushCube:
  Input:  current state (8-D) + action (2-D)
  Output: next state (8-D) + reward (scalar)

Uses an MLP dynamics model (teaching implementation).
"""

import argparse
import sys
from pathlib import Path

import numpy as np

from unified_pushcube_env import PushCubeEnv


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
    # Collect rollout data
    # ------------------------------------------------------------------
    def collect_rollouts(n_episodes):
        data = []
        for ep in range(n_episodes):
            env = PushCubeEnv()
            obs = env.reset(seed=ep)
            state = env.get_state_vector()

            for _ in range(env.max_steps):
                action = np.random.uniform(-1, 1, size=2).astype(np.float32)
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
        return data

    print(f"\nCollecting {args.n_episodes} rollout episodes...")
    data = collect_rollouts(args.n_episodes)
    print(f"  Collected {len(data)} transitions")

    # Split train/val
    split = int(len(data) * 0.8)
    train_data = data[:split]
    val_data = data[split:]

    # Convert to tensors
    def to_tensor(batch, key):
        return torch.tensor(np.stack([d[key] for d in batch]), dtype=torch.float32).to(device)

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
        def __init__(self, state_dim=8, action_dim=2):
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

    model = TinyWorldModel().to(device)
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
            idx = perm[i:i+args.batch_size]
            s, a, ns, r = train_s[idx], train_a[idx], train_ns[idx], train_r[idx]

            pred_ns, pred_r = model(s, a)
            loss = F.mse_loss(pred_ns, ns) + F.mse_loss(pred_r, r)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            n_batches += 1

        avg_train_loss = epoch_loss / n_batches

        # Val
        model.eval()
        with torch.no_grad():
            pred_ns, pred_r = model(val_s, val_a)
            val_loss = F.mse_loss(pred_ns, val_ns).item() + F.mse_loss(pred_r, val_r).item()

        if (epoch + 1) % 5 == 0:
            print(f"  Epoch {epoch+1}/{args.epochs}: train_loss={avg_train_loss:.4f}, val_loss={val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss

    print(f"\nTraining complete. Best val loss: {best_val_loss:.4f}")

    # ------------------------------------------------------------------
    # Multi-step prediction evaluation
    # ------------------------------------------------------------------
    print("\nMulti-step prediction evaluation...")
    model.eval()

    for horizon in [1, 5, 10]:
        errors = []
        n_test = 50

        for ep in range(n_test):
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

                state = pred_state if step < horizon - 1 else true_state  # open-loop

                if step == horizon - 1:
                    err = np.linalg.norm(pred_state[:4] - true_state[:4])  # pos error
                    errors.append(err)

                if done or truncated:
                    break

        mean_err = np.mean(errors)
        print(f"  H={horizon}: mean position error = {mean_err:.4f}")

    # ------------------------------------------------------------------
    # Save
    # ------------------------------------------------------------------
    save_dir = Path(args.output_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), save_dir / "pushcube_wm.pt")
    print(f"\nModel saved to {save_dir / 'pushcube_wm.pt'}")

    import json
    with open(save_dir / "wm_config.json", "w") as f:
        json.dump({
            "task": "PushCube World Model",
            "n_episodes": args.n_episodes,
            "epochs": args.epochs,
            "best_val_loss": round(best_val_loss, 4),
        }, f, indent=2)


def main():
    parser = argparse.ArgumentParser(description="Unified PushCube — World Model Track")
    parser.add_argument("--n-episodes", type=int, default=200, help="Number of rollouts")
    parser.add_argument("--epochs", type=int, default=50, help="Training epochs")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size")
    parser.add_argument("--output-dir", type=str, default="../results/unified_pushcube/wm", help="Output directory")
    args = parser.parse_args()
    train_wm(args)


if __name__ == "__main__":
    main()
