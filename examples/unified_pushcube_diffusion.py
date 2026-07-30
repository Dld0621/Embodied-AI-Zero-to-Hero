"""
Unified PushCube — Diffusion Policy Track
==========================================
Minimal Diffusion Policy implementation on the shared PushCube task.

Input:  128x128 image
Output: 2-D action (sampled from learned distribution via denoising)

This is a teaching implementation, not a production policy.
"""

import argparse
import sys
from pathlib import Path

import numpy as np

from unified_pushcube_env import PushCubeEnv


def train_diffusion(args):
    """Train a minimal diffusion policy."""
    print("=" * 70)
    print(" Unified PushCube — Diffusion Policy Training")
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

    action_dim = 2
    obs_dim = 32
    hidden_dim = 64
    n_steps = args.diffusion_steps

    # ------------------------------------------------------------------
    # Minimal Diffusion Policy
    # ------------------------------------------------------------------
    class MinimalDiffusionPolicy(nn.Module):
        def __init__(self):
            super().__init__()
            self.time_embed = nn.Embedding(n_steps, hidden_dim)

            # Vision encoder -> obs feature
            self.cnn = nn.Sequential(
                nn.Conv2d(3, 8, 5, stride=2, padding=2),
                nn.ReLU(),
                nn.Conv2d(8, 16, 5, stride=2, padding=2),
                nn.ReLU(),
                nn.Conv2d(16, 16, 5, stride=2, padding=2),
                nn.ReLU(),
                nn.Conv2d(16, 8, 5, stride=2, padding=2),
                nn.ReLU(),
            )
            self.vision_fc = nn.Linear(8 * 8 * 8, obs_dim)

            # Noise prediction network
            self.noise_pred = nn.Sequential(
                nn.Linear(action_dim + hidden_dim + obs_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, action_dim),
            )

            # Diffusion schedule
            self.register_buffer("betas", torch.linspace(1e-4, 0.02, n_steps))
            alphas = 1.0 - self.betas
            self.register_buffer("alphas_cumprod", torch.cumprod(alphas, dim=0))
            self.register_buffer("sqrt_alphas_cumprod", torch.sqrt(self.alphas_cumprod))
            self.register_buffer("sqrt_one_minus_alphas_cumprod", torch.sqrt(1.0 - self.alphas_cumprod))

        def encode_obs(self, image):
            x = self.cnn(image)
            x = x.reshape(x.size(0), -1)
            return self.vision_fc(x)

        def forward(self, noisy_action, t, obs_feat):
            t_emb = self.time_embed(t)  # (B, hidden_dim)
            inp = torch.cat([noisy_action, t_emb, obs_feat], dim=-1)
            return self.noise_pred(inp)  # (B, action_dim)

        def sample(self, image):
            obs_feat = self.encode_obs(image)
            action = torch.randn(image.size(0), action_dim, device=device)
            for t in reversed(range(n_steps)):
                t_batch = torch.full((image.size(0),), t, device=device, dtype=torch.long)
                epsilon_pred = self.forward(action, t_batch, obs_feat)
                alpha_t = self.alphas_cumprod[t]
                beta_t = self.betas[t]
                # Denoise step
                action = (action - self.sqrt_one_minus_alphas_cumprod[t] * epsilon_pred) / torch.sqrt(alpha_t)
                if t > 0:
                    action += torch.sqrt(beta_t) * torch.randn_like(action)
            return action

    model = MinimalDiffusionPolicy().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    # ------------------------------------------------------------------
    # Data Collection
    # ------------------------------------------------------------------
    def collect_data(n_episodes):
        images = []
        actions = []
        for ep in range(n_episodes):
            env = PushCubeEnv()
            obs = env.reset(seed=ep)
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
                    act = dir_vec * 0.8
                else:
                    dir_vec = target - cube
                    dir_vec /= np.linalg.norm(dir_vec) + 1e-6
                    act = dir_vec * 0.8
                act = np.clip(act, -1.0, 1.0)
                actions.append(act)

                obs, _, done, truncated, _ = env.step(act)
                if done or truncated:
                    break
            if (ep + 1) % 50 == 0:
                print(f"  Collected {ep+1}/{n_episodes} episodes")
        return images, actions

    print(f"\nCollecting {args.n_episodes} demonstration episodes...")
    images, actions = collect_data(args.n_episodes)
    print(f"  Total frames: {len(actions)}")

    images_t = torch.tensor(np.stack(images), dtype=torch.float32).to(device)
    actions_t = torch.tensor(np.stack(actions), dtype=torch.float32).to(device)

    # ------------------------------------------------------------------
    # Train (DDPM-style)
    # ------------------------------------------------------------------
    print(f"\nTraining for {args.epochs} epochs...")
    best_loss = float("inf")

    for epoch in range(args.epochs):
        perm = torch.randperm(len(actions_t))
        epoch_loss = 0.0
        n_batches = 0

        for i in range(0, len(actions_t), args.batch_size):
            idx = perm[i:i + args.batch_size]
            batch_img = images_t[idx]
            batch_act = actions_t[idx]
            batch_size_actual = batch_act.size(0)

            # Sample random timestep
            t = torch.randint(0, n_steps, (batch_size_actual,), device=device)

            # Forward diffusion: add noise
            epsilon = torch.randn_like(batch_act)
            a_noisy = model.sqrt_alphas_cumprod[t].view(-1, 1) * batch_act + \
                      model.sqrt_one_minus_alphas_cumprod[t].view(-1, 1) * epsilon

            # Predict noise
            obs_feat = model.encode_obs(batch_img)
            epsilon_pred = model(a_noisy, t, obs_feat)

            loss = F.mse_loss(epsilon_pred, epsilon)

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
    print(f"\nEvaluating (deterministic sampling, 20 episodes)...")
    model.eval()
    success_count = 0

    with torch.no_grad():
        for ep in range(20):
            env = PushCubeEnv()
            obs = env.reset(seed=6000 + ep)

            for step in range(env.max_steps):
                img = torch.tensor(
                    env.render(size=128).transpose(2, 0, 1),
                    dtype=torch.float32
                ).unsqueeze(0).to(device)
                action = model.sample(img).cpu().numpy()[0]
                action = np.clip(action, -1.0, 1.0)

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
    torch.save(model.state_dict(), save_dir / "pushcube_diffusion.pt")
    print(f"Model saved to {save_dir / 'pushcube_diffusion.pt'}")

    import json
    with open(save_dir / "diffusion_config.json", "w") as f:
        json.dump({
            "task": "PushCube Diffusion Policy",
            "n_episodes": args.n_episodes,
            "diffusion_steps": args.diffusion_steps,
            "epochs": args.epochs,
            "best_loss": round(best_loss, 4),
            "success_rate": round(success_count / 20 * 100, 1),
        }, f, indent=2)


def main():
    parser = argparse.ArgumentParser(description="Unified PushCube — Diffusion Policy Track")
    parser.add_argument("--n-episodes", type=int, default=200, help="Demo episodes")
    parser.add_argument("--epochs", type=int, default=50, help="Training epochs")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size")
    parser.add_argument("--diffusion-steps", type=int, default=20, help="Diffusion denoising steps")
    parser.add_argument("--output-dir", type=str, default="../results/unified_pushcube/diffusion",
                        help="Output directory")
    args = parser.parse_args()
    train_diffusion(args)


if __name__ == "__main__":
    main()
