"""
Unified PushCube — VLA Track
==============================
Train a tiny vision-language-action policy on PushCube:
  Input:  128x128 image + language instruction
  Output: 2-D action (arm movement)

This is a teaching implementation, not a pretrained policy.
"""

import argparse
import sys
from pathlib import Path

import numpy as np

# Import our shared environment
from unified_pushcube_env import PushCubeEnv


def train_vla(args):
    """Train a tiny CNN + Language Embedding -> MLP policy."""
    print("=" * 70)
    print(" Unified PushCube — VLA Training")
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

    # Vocabulary for language instructions
    VOCAB = {
        "<pad>": 0, "push": 1, "the": 2, "red": 3, "green": 4, "yellow": 5,
        "cube": 6, "to": 7, "right": 8, "left": 9, "top": 10, "bottom": 11,
        "and": 12, "center": 13,
    }

    def tokenize(text):
        words = text.lower().replace(".", "").split()
        return [VOCAB.get(w, 0) for w in words]

    # ------------------------------------------------------------------
    # Tiny Policy Network
    # ------------------------------------------------------------------
    class TinyVLAPolicy(nn.Module):
        def __init__(self, vocab_size=len(VOCAB), embed_dim=16, action_dim=2):
            super().__init__()
            # Vision encoder: 128x128 -> small feature
            self.conv = nn.Sequential(
                nn.Conv2d(3, 8, 5, stride=2, padding=2),  # 64x64
                nn.ReLU(),
                nn.Conv2d(8, 16, 5, stride=2, padding=2),  # 32x32
                nn.ReLU(),
                nn.Conv2d(16, 16, 5, stride=2, padding=2),  # 16x16
                nn.ReLU(),
                nn.Conv2d(16, 8, 5, stride=2, padding=2),  # 8x8
                nn.ReLU(),
            )
            self.vision_fc = nn.Linear(8 * 8 * 8, 32)

            # Language encoder
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
            # Image: (B, 3, 128, 128)
            x = self.conv(image)
            x = x.reshape(x.size(0), -1)
            v = self.vision_fc(x)  # (B, 32)

            # Language: (B, seq_len)
            w = self.word_embed(text_tokens)  # (B, seq_len, embed_dim)
            w = w.mean(dim=1)  # (B, embed_dim)
            l = self.lang_fc(w)  # (B, 16)

            fused = torch.cat([v, l], dim=-1)
            action = self.fusion(fused)
            return action

    policy = TinyVLAPolicy().to(device)
    optimizer = torch.optim.Adam(policy.parameters(), lr=1e-3)

    # ------------------------------------------------------------------
    # Data Collection (Behavior Cloning from random + heuristic)
    # ------------------------------------------------------------------
    def collect_episode(env, seed):
        obs = env.reset(seed=seed)
        lang = env.get_language_instruction()
        images = []
        tokens = []
        actions = []

        for _ in range(env.max_steps):
            img = env.render(size=128).transpose(2, 0, 1)  # (3, 128, 128)
            tok = tokenize(lang)
            tok += [0] * (10 - len(tok))  # pad to 10
            tok = tok[:10]

            # Simple heuristic: move toward cube, then push toward target
            arm = obs["arm_pos"]
            cube = obs["cube_pos"]
            target = obs["target_pos"]
            dist_to_cube = np.linalg.norm(cube - arm)

            if dist_to_cube > 0.08:
                dir_to_cube = cube - arm
                dir_to_cube = dir_to_cube / (np.linalg.norm(dir_to_cube) + 1e-6)
                action = dir_to_cube * 0.8
            else:
                dir_to_target = target - cube
                dir_to_target = dir_to_target / (np.linalg.norm(dir_to_target) + 1e-6)
                action = dir_to_target * 0.8

            action = np.clip(action, -1.0, 1.0)

            images.append(img)
            tokens.append(tok)
            actions.append(action)

            obs, reward, done, truncated, info = env.step(action)
            if done or truncated:
                break

        return images, tokens, actions

    print(f"\nCollecting {args.n_episodes} demonstration episodes...")
    all_images = []
    all_tokens = []
    all_actions = []

    for ep in range(args.n_episodes):
        imgs, toks, acts = collect_episode(PushCubeEnv(), seed=ep)
        all_images.extend(imgs)
        all_tokens.extend(toks)
        all_actions.extend(acts)
        if (ep + 1) % 10 == 0:
            print(f"  Collected {ep+1}/{args.n_episodes} episodes, {len(all_images)} frames")

    # Convert to tensors
    images = torch.tensor(np.stack(all_images), dtype=torch.float32).to(device)
    tokens = torch.tensor(np.array(all_tokens), dtype=torch.long).to(device)
    actions = torch.tensor(np.stack(all_actions), dtype=torch.float32).to(device)

    print(f"\nDataset: {len(images)} frames")

    # ------------------------------------------------------------------
    # Train (Behavior Cloning)
    # ------------------------------------------------------------------
    print(f"\nTraining for {args.epochs} epochs...")
    policy.train()

    best_loss = float("inf")
    for epoch in range(args.epochs):
        # Shuffle
        perm = torch.randperm(len(images))
        images_shuffled = images[perm]
        tokens_shuffled = tokens[perm]
        actions_shuffled = actions[perm]

        epoch_loss = 0.0
        n_batches = 0
        for i in range(0, len(images), args.batch_size):
            batch_img = images_shuffled[i:i+args.batch_size]
            batch_tok = tokens_shuffled[i:i+args.batch_size]
            batch_act = actions_shuffled[i:i+args.batch_size]

            pred = policy(batch_img, batch_tok)
            loss = F.mse_loss(pred, batch_act)

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
    # Evaluate
    # ------------------------------------------------------------------
    print("\nEvaluating trained policy...")
    policy.eval()
    success_count = 0
    n_eval = 20

    with torch.no_grad():
        for ep in range(n_eval):
            env = PushCubeEnv()
            obs = env.reset(seed=1000 + ep)
            lang = env.get_language_instruction()
            tok = torch.tensor([tokenize(lang) + [0]*(10-len(tokenize(lang)))], dtype=torch.long).to(device)
            tok = tok[:, :10]

            for step in range(env.max_steps):
                img = torch.tensor(env.render(size=128).transpose(2, 0, 1), dtype=torch.float32).unsqueeze(0).to(device)
                action = policy(img, tok).cpu().numpy()[0]
                obs, reward, done, truncated, info = env.step(action)
                if done:
                    success_count += 1
                    break
                if truncated:
                    break

    print(f"Success rate: {success_count}/{n_eval} = {success_count/n_eval*100:.1f}%")

    # ------------------------------------------------------------------
    # Save
    # ------------------------------------------------------------------
    save_dir = Path(args.output_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    torch.save(policy.state_dict(), save_dir / "pushcube_vla_policy.pt")
    print(f"Model saved to {save_dir / 'pushcube_vla_policy.pt'}")

    # Save config
    import json
    with open(save_dir / "vla_config.json", "w") as f:
        json.dump({
            "task": "PushCube VLA",
            "n_episodes": args.n_episodes,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "best_loss": round(best_loss, 4),
            "success_rate": round(success_count / n_eval * 100, 1),
        }, f, indent=2)


def main():
    parser = argparse.ArgumentParser(description="Unified PushCube — VLA Track")
    parser.add_argument("--n-episodes", type=int, default=100, help="Number of demo episodes for BC")
    parser.add_argument("--epochs", type=int, default=30, help="Training epochs")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--output-dir", type=str, default="../results/unified_pushcube/vla", help="Output directory")
    args = parser.parse_args()
    train_vla(args)


if __name__ == "__main__":
    main()
