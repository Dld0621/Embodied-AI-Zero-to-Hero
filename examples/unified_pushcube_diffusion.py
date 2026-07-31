"""
Unified PushCube — Diffusion Policy Track (Language-Conditioned)
================================================================
Minimal Diffusion Policy implementation on the shared PushCube task.

Input:  128x128 image + tokenized language instruction
Output: action HORIZON (T steps of 2-D arm movement) sampled via DDPM
        denoising.

This version fixes the critical DDPM bugs identified in review:
  1. The reverse (denoising) step now uses the standard DDPM formula
     that distinguishes the *single-step* alpha_t (= 1 - beta_t) from
     the *cumulative* alpha_bar_t (= alphas_cumprod[t]).
        x_{t-1} = (1/sqrt(alpha_t)) * (x_t - (1-alpha_t)/sqrt(1-alpha_bar_t) * eps_pred)
                  + sigma_t * z
     where sigma_t = sqrt(beta_t) during training-style sampling and
     sigma_t = 0 for deterministic evaluation.
  2. The model now predicts an *action horizon* (a sequence of T
     actions at once) instead of a single-step action.
  3. Evaluation is truly deterministic: a fixed eval seed is set with
     torch.manual_seed(eval_seed) and NO noise is added during the
     reverse process (sigma_t = 0).
  4. Language conditioning: the noise prediction network now receives
     language token features alongside vision features, so the policy
     can disambiguate which cube to push.
  5. Demonstration actions come from the shared expert_action(env)
     heuristic.

This is a teaching implementation, not a production policy.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from unified_pushcube_env import PushCubeEnv, expert_action, COLOR_NAMES, CUBE_COLORS


# ----------------------------------------------------------------------
# Vocabulary & tokenization (shared with VLA track)
# ----------------------------------------------------------------------
VOCAB = {
    "<pad>": 0, "push": 1, "the": 2, "red": 3, "green": 4,
    "cube": 5, "to": 6, "right": 7, "left": 8, "top": 9,
    "bottom": 10, "and": 11, "center": 12,
}
MAX_LEN = 10

for _name in COLOR_NAMES:
    assert _name in VOCAB, f"Vocab missing color word: {_name}"


def tokenize(text):
    """Lowercase, split, map to ids, pad/truncate to MAX_LEN."""
    words = text.lower().replace(".", "").split()
    toks = [VOCAB.get(w, 0) for w in words]
    toks = toks[:MAX_LEN] + [0] * (MAX_LEN - len(toks))
    return toks


def train_diffusion(args):
    """Train a minimal diffusion policy that predicts an action horizon."""
    print("=" * 70)
    print(" Unified PushCube — Diffusion Policy Training (lang-conditioned)")
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
    horizon = args.horizon            # action horizon T (predict T actions)
    obs_dim = 32
    lang_dim = 16
    hidden_dim = 64
    n_steps = args.diffusion_steps
    vocab_size = len(VOCAB)
    embed_dim = 16

    # ------------------------------------------------------------------
    # Minimal Diffusion Policy (predicts an action horizon, lang-conditioned)
    # ------------------------------------------------------------------
    class MinimalDiffusionPolicy(nn.Module):
        def __init__(self):
            super().__init__()
            self.horizon = horizon
            self.action_dim = action_dim
            self.flat_dim = horizon * action_dim

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

            # Language encoder: word embeddings averaged over the sentence
            self.word_embed = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
            self.lang_fc = nn.Linear(embed_dim, lang_dim)

            # Noise prediction network: predicts noise over the ENTIRE
            # action horizon (flat_dim = horizon * action_dim).
            # Input: noisy_action + time_emb + obs_feat + lang_feat
            self.noise_pred = nn.Sequential(
                nn.Linear(self.flat_dim + hidden_dim + obs_dim + lang_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, self.flat_dim),
            )

            # ---- DDPM noise schedule ----
            # betas[t], alpha_t = 1 - beta_t (single step),
            # alpha_bar_t = alphas_cumprod[t] (cumulative).
            self.register_buffer(
                "betas", torch.linspace(1e-4, 0.02, n_steps)
            )
            alphas = 1.0 - self.betas                       # single-step alpha_t
            self.register_buffer("alphas", alphas)
            self.register_buffer("alphas_cumprod", torch.cumprod(alphas, dim=0))
            self.register_buffer(
                "sqrt_alphas_cumprod", torch.sqrt(self.alphas_cumprod)
            )
            self.register_buffer(
                "sqrt_one_minus_alphas_cumprod",
                torch.sqrt(1.0 - self.alphas_cumprod),
            )
            # sigma_t = sqrt(beta_t) (used only for stochastic sampling)
            self.register_buffer("sqrt_betas", torch.sqrt(self.betas))

        def encode_obs(self, image, text_tokens):
            """Encode image + language -> (obs_feat, lang_feat)."""
            x = self.cnn(image)
            x = x.reshape(x.size(0), -1)
            obs_feat = self.vision_fc(x)

            w = self.word_embed(text_tokens).mean(dim=1)  # (B, embed_dim)
            lang_feat = self.lang_fc(w)                    # (B, lang_dim)

            return obs_feat, lang_feat

        def forward(self, noisy_action, t, obs_feat, lang_feat):
            # noisy_action: (B, horizon*action_dim)
            t_emb = self.time_embed(t)                      # (B, hidden_dim)
            inp = torch.cat([noisy_action, t_emb, obs_feat, lang_feat], dim=-1)
            return self.noise_pred(inp)                     # (B, horizon*action_dim)

        @torch.no_grad()
        def sample(self, image, text_tokens, deterministic=False):
            """Reverse DDPM sampling.

            Standard reverse step (Ho et al. 2020):
                mean = (1/sqrt(alpha_t)) *
                       (x_t - (1-alpha_t)/sqrt(1-alpha_bar_t) * eps_pred)
                x_{t-1} = mean + sigma_t * z
            where
                alpha_t     = alphas[t]            (single step: 1 - beta_t)
                alpha_bar_t = alphas_cumprod[t]     (cumulative)
                beta_t      = betas[t]
                sigma_t     = sqrt(beta_t)          (training / stochastic)
                          = 0                       (deterministic eval)

            With `deterministic=True`, NO noise is added -> the sampling
            is fully reproducible given a fixed RNG seed.
            """
            obs_feat, lang_feat = self.encode_obs(image, text_tokens)
            B = image.size(0)
            # Start from pure Gaussian noise over the whole horizon.
            x = torch.randn(B, self.flat_dim, device=device)

            for t in reversed(range(n_steps)):
                t_batch = torch.full(
                    (B,), t, device=device, dtype=torch.long
                )
                eps_pred = self.forward(x, t_batch, obs_feat, lang_feat)

                beta_t = self.betas[t]
                alpha_t = self.alphas[t]                 # single-step alpha_t
                alpha_bar_t = self.alphas_cumprod[t]     # cumulative

                mean = (1.0 / torch.sqrt(alpha_t)) * (
                    x
                    - (1.0 - alpha_t) / torch.sqrt(1.0 - alpha_bar_t) * eps_pred
                )

                if t > 0 and not deterministic:
                    # Stochastic: add Gaussian noise (sigma_t = sqrt(beta_t))
                    x = mean + self.sqrt_betas[t] * torch.randn_like(x)
                else:
                    # Deterministic eval (or last step): no noise added.
                    x = mean

            return x.reshape(B, self.horizon, self.action_dim)

    model = MinimalDiffusionPolicy().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    # ------------------------------------------------------------------
    # Data Collection (uses the shared expert_action heuristic)
    # ------------------------------------------------------------------
    def collect_data(n_episodes):
        """Collect (image, language, action_horizon) pairs using expert_action(env)."""
        images = []
        tokens = []
        horizons = []
        for ep in range(n_episodes):
            env = PushCubeEnv()
            obs = env.reset(seed=ep)

            lang = env.get_language_instruction()
            tok = tokenize(lang)

            imgs = []
            acts = []
            for _ in range(env.max_steps):
                img = env.render(size=128).transpose(2, 0, 1)
                imgs.append(img)

                # Use the shared expert policy for demonstrations.
                act = expert_action(env)
                acts.append(act)

                obs, _, done, truncated, _ = env.step(act)
                if done or truncated:
                    break

            imgs = np.stack(imgs).astype(np.float32)    # (T, 3, 128, 128)
            acts = np.stack(acts).astype(np.float32)    # (T, action_dim)

            # Build action-horizon targets: at frame i predict the next
            # `horizon` actions.
            for i in range(len(acts) - horizon):
                images.append(imgs[i])
                tokens.append(tok)
                horizons.append(acts[i:i + horizon])    # (horizon, action_dim)

            if (ep + 1) % 50 == 0:
                print(f"  Collected {ep+1}/{n_episodes} episodes")
        return images, tokens, horizons

    print(f"\nCollecting {args.n_episodes} demonstration episodes "
          f"(expert_action)...")
    images, tokens, horizons = collect_data(args.n_episodes)
    print(f"  Total (image, horizon) pairs: {len(horizons)}")

    images_t = torch.tensor(np.stack(images), dtype=torch.float32).to(device)
    tokens_t = torch.tensor(np.array(tokens), dtype=torch.long).to(device)
    horizons_t = torch.tensor(np.stack(horizons), dtype=torch.float32).to(device)

    # ------------------------------------------------------------------
    # Train (DDPM forward + noise prediction over the action horizon)
    # ------------------------------------------------------------------
    print(f"\nTraining for {args.epochs} epochs...")
    best_loss = float("inf")

    for epoch in range(args.epochs):
        perm = torch.randperm(len(horizons_t))
        epoch_loss = 0.0
        n_batches = 0

        for i in range(0, len(horizons_t), args.batch_size):
            idx = perm[i:i + args.batch_size]
            batch_img = images_t[idx]                       # (B, 3, 128, 128)
            batch_tok = tokens_t[idx]                       # (B, MAX_LEN)
            batch_hor = horizons_t[idx]                     # (B, horizon, action_dim)
            B = batch_hor.size(0)

            # Flatten the action horizon for diffusion.
            batch_hor_flat = batch_hor.reshape(B, model.flat_dim)

            # Sample random timestep per sample.
            t = torch.randint(0, n_steps, (B,), device=device)

            # Forward diffusion: x_t = sqrt(a_bar_t) * x_0 + sqrt(1-a_bar_t) * eps
            epsilon = torch.randn_like(batch_hor_flat)
            a_noisy = (
                model.sqrt_alphas_cumprod[t].reshape(-1, 1) * batch_hor_flat
                + model.sqrt_one_minus_alphas_cumprod[t].reshape(-1, 1) * epsilon
            )

            # Predict noise.
            obs_feat, lang_feat = model.encode_obs(batch_img, batch_tok)
            epsilon_pred = model(a_noisy, t, obs_feat, lang_feat)

            loss = F.mse_loss(epsilon_pred, epsilon)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            n_batches += 1

        avg_loss = epoch_loss / max(n_batches, 1)
        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"  Epoch {epoch+1}/{args.epochs}: loss={avg_loss:.4f}")
        if avg_loss < best_loss:
            best_loss = avg_loss

    print(f"\nTraining complete. Best loss: {best_loss:.4f}")

    # ------------------------------------------------------------------
    # Evaluate (truly deterministic)
    # ------------------------------------------------------------------
    n_eval = args.n_eval
    print(f"\nEvaluating (deterministic sampling, {n_eval} episodes)...")

    model.eval()

    # Set a fixed eval seed and reset the PyTorch RNG so that the
    # initial noise x_T and (absent) sampling noise are reproducible.
    # Because deterministic=True below, NO noise is added during the
    # reverse process, so eval is fully deterministic.
    torch.manual_seed(args.eval_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.eval_seed)

    success_count = 0

    with torch.no_grad():
        for ep in range(n_eval):
            env = PushCubeEnv()
            obs = env.reset(seed=6000 + ep)

            # Language instruction for this episode
            lang = env.get_language_instruction()
            tok = tokenize(lang)
            tok_t = torch.tensor([tok], dtype=torch.long).to(device)

            # Receding-horizon execution: sample a horizon of actions and
            # execute the first `pred_interval` of them, then re-sample.
            pred_interval = args.pred_interval

            for step in range(env.max_steps):
                if step % pred_interval == 0:
                    img = torch.tensor(
                        env.render(size=128).transpose(2, 0, 1),
                        dtype=torch.float32,
                    ).unsqueeze(0).to(device)
                    # deterministic=True -> no noise added in reverse process
                    horizon_actions = model.sample(img, tok_t, deterministic=True)
                    horizon_actions = horizon_actions.cpu().numpy()[0]
                    # (horizon, action_dim)

                # Execute the action for the current step within the horizon.
                h_idx = step % pred_interval
                if h_idx >= len(horizon_actions):
                    h_idx = 0
                action = horizon_actions[h_idx]
                action = np.clip(action, -1.0, 1.0)

                obs, _, terminated, truncated, info = env.step(action)
                if terminated:
                    success_count += 1
                    break
                if truncated:
                    break

    success_rate = success_count / n_eval * 100 if n_eval > 0 else 0.0
    print(f"Success rate: {success_count}/{n_eval} = {success_rate:.1f}%")

    # ------------------------------------------------------------------
    # Save
    # ------------------------------------------------------------------
    save_dir = Path(args.output_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), save_dir / "pushcube_diffusion.pt")
    print(f"Model saved to {save_dir / 'pushcube_diffusion.pt'}")

    results = {
        "task": "PushCube Diffusion Policy (lang-conditioned)",
        "note": ("Predicts an action horizon via DDPM. Fixed DDPM reverse "
                 "step (alpha_t vs alpha_bar_t distinction), deterministic "
                 "eval (no noise, fixed seed), language conditioning, "
                 "expert_action demos."),
        "method": "Diffusion Policy (action horizon, lang-conditioned)",
        "action_horizon": horizon,
        "has_language_conditioning": True,
        "n_episodes": args.n_episodes,
        "diffusion_steps": args.diffusion_steps,
        "epochs": args.epochs,
        "best_loss": round(best_loss, 4),
        "n_eval": n_eval,
        "eval_seed": args.eval_seed,
        "deterministic_eval": True,
        "success_rate": round(success_rate, 1),
        "success_count": success_count,
        "smoke_test": args.smoke_test,
    }
    from benchmark_provenance import build_provenance
    results["provenance"] = build_provenance(
        command=f"python {__file__} --n_episodes {args.n_episodes} --epochs {args.epochs} --eval_seed {args.eval_seed}",
        result_generated_by=__file__,
    )
    with open(save_dir / "diffusion_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to {save_dir / 'diffusion_results.json'}")


def main():
    parser = argparse.ArgumentParser(
        description="Unified PushCube — Diffusion Policy Track"
    )
    parser.add_argument("--n-episodes", type=int, default=200,
                        help="Demo episodes")
    parser.add_argument("--epochs", type=int, default=50,
                        help="Training epochs")
    parser.add_argument("--batch-size", type=int, default=64,
                        help="Batch size")
    parser.add_argument("--diffusion-steps", type=int, default=20,
                        help="Diffusion denoising steps")
    parser.add_argument("--horizon", type=int, default=10,
                        help="Action horizon T (predict T actions at once)")
    parser.add_argument("--pred-interval", type=int, default=5,
                        help="Receding-horizon execution interval")
    parser.add_argument("--eval-seed", type=int, default=1234,
                        help="Fixed RNG seed for deterministic evaluation")
    parser.add_argument("--n-eval", type=int, default=20,
                        help="Number of eval episodes")
    parser.add_argument("--output-dir", type=str,
                        default="../results/unified_pushcube/diffusion",
                        help="Output directory")
    parser.add_argument("--smoke-test", action="store_true",
                        help="CI smoke test: 2 episodes, 2 epochs, 2 eval")
    args = parser.parse_args()

    if args.smoke_test:
        args.n_episodes = 2
        args.epochs = 2
        args.n_eval = 2
        print("[SMOKE TEST] n_episodes=2, epochs=2, n_eval=2")

    train_diffusion(args)


if __name__ == "__main__":
    main()
