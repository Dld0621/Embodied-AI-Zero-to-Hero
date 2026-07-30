"""
Unified PushCube — ACT (Action Chunking with Transformers) Track
=================================================================
Minimal ACT implementation on the shared PushCube task.

Input:  128x128 image
Output: action chunk (T steps of 2-D arm movement)

This is a teaching implementation, not a production policy.
"""

import argparse
import sys
from collections import deque
from pathlib import Path

import numpy as np

from unified_pushcube_env import PushCubeEnv


def train_act(args):
    """Train a minimal ACT-style policy with action chunking."""
    print("=" * 70)
    print(" Unified PushCube — ACT Training")
    print("=" * 70)

    try:
        import torch
        import torch.nn as nn
        import torch.nn.functional as F
    except ImportError:
        print("[Error] PyTorch required. Install: pip install torch")
        sys.exit(1)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ------------------------------------------------------------------
    # Minimal ACT Policy
    # ------------------------------------------------------------------
    class MinimalACT(nn.Module):
        def __init__(self, action_dim=2, chunk_size=10, hidden_dim=64):
            super().__init__()
            self.chunk_size = chunk_size

            # Vision encoder: 128x128 -> hidden_dim
            self.cnn = nn.Sequential(
                nn.Conv2d(3, 8, 5, stride=2, padding=2),   # 64x64
                nn.ReLU(),
                nn.Conv2d(8, 16, 5, stride=2, padding=2),  # 32x32
                nn.ReLU(),
                nn.Conv2d(16, 16, 5, stride=2, padding=2),  # 16x16
                nn.ReLU(),
                nn.Conv2d(16, 8, 5, stride=2, padding=2),   # 8x8
                nn.ReLU(),
            )
            self.vision_fc = nn.Linear(8 * 8 * 8, hidden_dim)

            # Transformer encoder
            enc_layer = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=4, batch_first=True)
            self.encoder = nn.TransformerEncoder(enc_layer, num_layers=2)

            # Action chunk head
            self.action_head = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, action_dim * chunk_size),
                nn.Tanh(),
            )

        def forward(self, image):
            # image: (B, 3, 128, 128)
            x = self.cnn(image)
            x = x.reshape(x.size(0), -1)
            v = self.vision_fc(x)  # (B, hidden_dim)
            v = v.unsqueeze(1)  # (B, 1, hidden_dim)
            encoded = self.encoder(v)  # (B, 1, hidden_dim)
            actions = self.action_head(encoded[:, 0])  # (B, action_dim * chunk_size)
            return actions.view(-1, self.chunk_size, action_dim)

    policy = MinimalACT(chunk_size=args.chunk_size).to(device)
    optimizer = torch.optim.Adam(policy.parameters(), lr=1e-3)

    # ------------------------------------------------------------------
    # Data Collection with chunking
    # ------------------------------------------------------------------
    def collect_episodes(n_episodes, chunk_size):
        """Collect (image, action_chunk) pairs from heuristic demos."""
        data_images = []
        data_chunks = []

        for ep in range(n_episodes):
            env = PushCubeEnv()
            obs = env.reset(seed=ep)

            images = []
            actions = []

            for _ in range(env.max_steps):
                img = env.render(size=128).transpose(2, 0, 1)
                images.append(img)

                arm = obs["arm_pos"]
                cube = obs["cube_pos"]
                target = obs["target_pos"]
                dist = np.linalg.norm(cube - arm)
                if dist > 0.08:
                    dir_vec = cube - arm
                    dir_vec /= np.linalg.norm(dir_vec) + 1e-6
                    action = dir_vec * 0.8
                else:
                    dir_vec = target - cube
                    dir_vec /= np.linalg.norm(dir_vec) + 1e-6
                    action = dir_vec * 0.8
                action = np.clip(action, -1.0, 1.0)
                actions.append(action)

                obs, _, done, truncated, _ = env.step(action)
                if done or truncated:
                    break

            # Sliding window: every K steps predict next chunk_size actions
            K = args.pred_interval
            for i in range(0, len(actions) - chunk_size, K):
                chunk = np.stack(actions[i:i + chunk_size])  # (chunk_size, action_dim)
                data_images.append(images[i])
                data_chunks.append(chunk)

            if (ep + 1) % 20 == 0:
                print(f"  Collected {ep+1}/{n_episodes} episodes, {len(data_images)} chunks")

        return data_images, data_chunks

    print(f"\nCollecting {args.n_episodes} demonstration episodes...")
    images, chunks = collect_episodes(args.n_episodes, args.chunk_size)
    print(f"  Total chunks: {len(chunks)}")

    images_t = torch.tensor(np.stack(images), dtype=torch.float32).to(device)
    chunks_t = torch.tensor(np.stack(chunks), dtype=torch.float32).to(device)

    # ------------------------------------------------------------------
    # Train
    # ------------------------------------------------------------------
    print(f"\nTraining for {args.epochs} epochs...")
    best_loss = float("inf")

    for epoch in range(args.epochs):
        perm = torch.randperm(len(images_t))
        epoch_loss = 0.0
        n_batches = 0

        for i in range(0, len(images_t), args.batch_size):
            idx = perm[i:i + args.batch_size]
            pred = policy(images_t[idx])
            loss = F.mse_loss(pred, chunks_t[idx])

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            n_batches += 1

        avg_loss = epoch_loss / n_batches
        if (epoch + 1) % 5 == 0:
            print(f"  Epoch {epoch+1}/{args.epochs}: loss={avg_loss:.4f}")
        if avg_loss < best_loss:
            best_loss = avg_loss

    print(f"\nTraining complete. Best loss: {best_loss:.4f}")

    # ------------------------------------------------------------------
    # Evaluate with action queue (temporal ensembling)
    # ------------------------------------------------------------------
    print(f"\nEvaluating with action chunking (deterministic, 20 episodes)...")
    policy.eval()
    success_count = 0

    with torch.no_grad():
        for ep in range(20):
            env = PushCubeEnv()
            obs = env.reset(seed=5000 + ep)
            action_queue = deque()

            for step in range(env.max_steps):
                if not action_queue:
                    img = torch.tensor(
                        env.render(size=128).transpose(2, 0, 1),
                        dtype=torch.float32
                    ).unsqueeze(0).to(device)
                    chunk = policy(img).cpu().numpy()[0]  # (chunk_size, action_dim)
                    action_queue.extend(chunk)

                action = action_queue.popleft()
                obs, _, done, truncated, info = env.step(action)
                if done:
                    success_count += 1
                    break
                if truncated:
                    break

    print(f"Success rate: {success_count}/20 = {success_count/20*100:.1f}%")

    # ------------------------------------------------------------------
    # Save
    # ------------------------------------------------------------------
    save_dir = Path(args.output_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    torch.save(policy.state_dict(), save_dir / "pushcube_act.pt")
    print(f"Model saved to {save_dir / 'pushcube_act.pt'}")

    import json
    with open(save_dir / "act_config.json", "w") as f:
        json.dump({
            "task": "PushCube ACT",
            "n_episodes": args.n_episodes,
            "chunk_size": args.chunk_size,
            "epochs": args.epochs,
            "best_loss": round(best_loss, 4),
            "success_rate": round(success_count / 20 * 100, 1),
        }, f, indent=2)


def main():
    parser = argparse.ArgumentParser(description="Unified PushCube — ACT Track")
    parser.add_argument("--n-episodes", type=int, default=100, help="Demo episodes")
    parser.add_argument("--epochs", type=int, default=30, help="Training epochs")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--chunk-size", type=int, default=10, help="Action chunk size")
    parser.add_argument("--pred-interval", type=int, default=5, help="Prediction interval")
    parser.add_argument("--output-dir", type=str, default="../results/unified_pushcube/act",
                        help="Output directory")
    args = parser.parse_args()
    train_act(args)


if __name__ == "__main__":
    main()
