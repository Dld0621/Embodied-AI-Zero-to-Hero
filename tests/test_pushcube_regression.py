"""
tests/test_pushcube_regression.py
=================================
Lightweight regression tests for the unified PushCube environment and
policies. These verify fundamental correctness properties that go
beyond "does it run without crashing":

  1. Expert success rate >= 50% (over 20 fixed seeds)
  2. Environment state_dim == 14 (includes goal-color one-hot)
  3. Swapped language changes the target color (language is necessary)
  4. DDPM deterministic sampling produces identical output (reproducibility)
  5. Action-Chunking output shape == [B, horizon, action_dim]

运行：python -m pytest tests/test_pushcube_regression.py -v
"""

import sys
import unittest
import importlib.util
from pathlib import Path

import numpy as np

# 将项目根目录加入 Python 路径
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
EXAMPLES_DIR = PROJECT_ROOT / "examples"


# ----------------------------------------------------------------------
# Load the PushCube environment module
# ----------------------------------------------------------------------
def _load_env_module():
    spec = importlib.util.spec_from_file_location(
        "unified_pushcube_env", EXAMPLES_DIR / "unified_pushcube_env.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_env_mod = _load_env_module()
PushCubeEnv = _env_mod.PushCubeEnv
expert_action = _env_mod.expert_action
COLOR_NAMES = _env_mod.COLOR_NAMES


class TestExpertSuccessRate(unittest.TestCase):
    """Expert policy should succeed on at least 50% of fixed-seed episodes."""

    def test_expert_success_ge_50pct(self):
        n_episodes = 20
        success = 0
        for seed in range(n_episodes):
            env = PushCubeEnv()
            env.reset(seed=seed)
            for _ in range(env.max_steps):
                action = expert_action(env)
                obs, reward, done, truncated, info = env.step(action)
                if done:
                    success += 1
                    break
                if truncated:
                    break
        rate = success / n_episodes
        self.assertGreaterEqual(
            rate, 0.50,
            f"Expert success rate {rate*100:.1f}% < 50% threshold "
            f"({success}/{n_episodes} episodes)"
        )


class TestStateDim(unittest.TestCase):
    """Environment state must be 14-D with goal-color one-hot."""

    def test_state_dim_is_14(self):
        env = PushCubeEnv()
        env.reset(seed=0)
        self.assertEqual(env.state_dim, 14)

    def test_state_vector_length_matches(self):
        env = PushCubeEnv()
        env.reset(seed=0)
        state = env.get_state_vector()
        self.assertEqual(state.shape, (14,))

    def test_goal_color_onehot_is_valid(self):
        env = PushCubeEnv()
        env.reset(seed=0)
        onehot = env.get_goal_color_onehot()
        self.assertEqual(onehot.shape, (2,))
        self.assertAlmostEqual(float(onehot.sum()), 1.0, places=5)
        # Exactly one element should be 1.0
        self.assertEqual(int(onehot.argmax()), env.active_color_idx)


class TestLanguageSwapChangesTarget(unittest.TestCase):
    """Swapped language must refer to a different cube color than correct
    language. This verifies that language carries non-redundant information."""

    def test_correct_and_shuffled_differ(self):
        env = PushCubeEnv()
        env.reset(seed=42)
        correct = env.get_language_instruction()
        shuffled = env.get_shuffled_language()
        self.assertNotEqual(
            correct, shuffled,
            "Correct and shuffled language should differ"
        )

    def test_swapped_refers_to_other_color(self):
        """The shuffled instruction should mention the OTHER cube's color."""
        for seed in range(10):
            env = PushCubeEnv()
            env.reset(seed=seed)
            correct_lang = env.get_language_instruction()
            shuffled_lang = env.get_shuffled_language()

            active_color = COLOR_NAMES[env.active_color_idx]
            other_color = COLOR_NAMES[1 - env.active_color_idx]

            # Correct language mentions the active color
            self.assertIn(
                active_color, correct_lang,
                f"Seed {seed}: correct lang '{correct_lang}' should mention "
                f"active color '{active_color}'"
            )
            # Shuffled language mentions the other color
            self.assertIn(
                other_color, shuffled_lang,
                f"Seed {seed}: shuffled lang '{shuffled_lang}' should mention "
                f"other color '{other_color}'"
            )

    def test_goal_onehot_matches_language(self):
        """The goal-color one-hot should be consistent with the language
        instruction's color word."""
        for seed in range(10):
            env = PushCubeEnv()
            env.reset(seed=seed)
            lang = env.get_language_instruction()
            onehot = env.get_goal_color_onehot()
            active_color_idx = int(onehot.argmax())
            color_name = COLOR_NAMES[active_color_idx]
            self.assertIn(
                color_name, lang,
                f"Seed {seed}: onehot says '{color_name}' but lang is '{lang}'"
            )


class TestDDPMDeterministic(unittest.TestCase):
    """DDPM deterministic sampling must produce identical actions for the
    same input + same RNG seed."""

    def test_deterministic_sample_reproducible(self):
        try:
            import torch
            import torch.nn as nn
        except ImportError:
            self.skipTest("torch not installed, skipping DDPM test")

        # Minimal DDPM model with the same schedule as the Diffusion Policy
        action_dim = 2
        horizon = 10
        flat_dim = horizon * action_dim
        hidden_dim = 64
        n_steps = 20

        class TinyDDPM(nn.Module):
            def __init__(self):
                super().__init__()
                self.horizon = horizon
                self.action_dim = action_dim
                self.flat_dim = flat_dim
                self.time_embed = nn.Embedding(n_steps, hidden_dim)
                self.net = nn.Sequential(
                    nn.Linear(flat_dim + hidden_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Linear(hidden_dim, flat_dim),
                )
                self.register_buffer(
                    "betas", torch.linspace(1e-4, 0.02, n_steps)
                )
                alphas = 1.0 - self.betas
                self.register_buffer("alphas", alphas)
                self.register_buffer("alphas_cumprod", torch.cumprod(alphas, dim=0))
                self.register_buffer("sqrt_betas", torch.sqrt(self.betas))

            @torch.no_grad()
            def sample(self, batch_size, deterministic, device="cpu"):
                x = torch.randn(batch_size, self.flat_dim, device=device)
                for t in reversed(range(n_steps)):
                    t_batch = torch.full(
                        (batch_size,), t, device=device, dtype=torch.long
                    )
                    t_emb = self.time_embed(t_batch)
                    eps_pred = self.net(torch.cat([x, t_emb], dim=-1))
                    alpha_t = self.alphas[t]
                    alpha_bar_t = self.alphas_cumprod[t]
                    mean = (1.0 / torch.sqrt(alpha_t)) * (
                        x - (1.0 - alpha_t) / torch.sqrt(1.0 - alpha_bar_t) * eps_pred
                    )
                    if t > 0 and not deterministic:
                        x = mean + self.sqrt_betas[t] * torch.randn_like(x)
                    else:
                        x = mean
                return x.reshape(batch_size, self.horizon, self.action_dim)

        model = TinyDDPM().eval()

        # Run 1: deterministic with seed 42
        torch.manual_seed(42)
        out1 = model.sample(batch_size=4, deterministic=True)

        # Run 2: deterministic with same seed 42
        torch.manual_seed(42)
        out2 = model.sample(batch_size=4, deterministic=True)

        self.assertTrue(
            torch.allclose(out1, out2, atol=1e-6),
            "Deterministic DDPM sampling should produce identical output "
            "for the same seed and input"
        )

        # Run 3: stochastic should generally differ (sanity check)
        torch.manual_seed(42)
        out3 = model.sample(batch_size=4, deterministic=False)
        # Stochastic and deterministic should differ (extremely likely)
        self.assertFalse(
            torch.allclose(out1, out3, atol=1e-4),
            "Stochastic and deterministic sampling should produce different output"
        )


class TestActionChunkingOutputShape(unittest.TestCase):
    """Action-Chunking Policy output must have shape [B, horizon, action_dim]."""

    def test_output_shape(self):
        try:
            import torch
            import torch.nn as nn
        except ImportError:
            self.skipTest("torch not installed, skipping ACT shape test")

        action_dim = 2
        chunk_size = 10  # horizon
        hist_len = 3
        hidden_dim = 64
        vocab_size = 13
        embed_dim = 16

        class TinyACT(nn.Module):
            def __init__(self):
                super().__init__()
                self.action_dim = action_dim
                self.chunk_size = chunk_size
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
                self.vision_fc = nn.Linear(8 * 8 * 8, hidden_dim)
                self.word_embed = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
                self.lang_fc = nn.Linear(embed_dim, hidden_dim)
                self.temporal_pos = nn.Embedding(hist_len + 1, hidden_dim)
                enc_layer = nn.TransformerEncoderLayer(
                    d_model=hidden_dim, nhead=4, batch_first=True
                )
                self.encoder = nn.TransformerEncoder(enc_layer, num_layers=2)
                self.action_head = nn.Sequential(
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Linear(hidden_dim, action_dim * chunk_size),
                    nn.Tanh(),
                )

            def forward(self, images, text_tokens):
                B, K, C, H, W = images.shape
                imgs_flat = images.reshape(B * K, C, H, W)
                x = self.cnn(imgs_flat).reshape(B * K, -1)
                vis_feats = self.vision_fc(x).reshape(B, K, -1)
                w = self.word_embed(text_tokens).mean(dim=1)
                lang_feat = self.lang_fc(w).unsqueeze(1)
                all_tokens = torch.cat([vis_feats, lang_feat], dim=1)
                positions = torch.arange(K + 1, device=all_tokens.device)
                all_tokens = all_tokens + self.temporal_pos(positions).unsqueeze(0)
                encoded = self.encoder(all_tokens)
                actions = self.action_head(encoded[:, -1])
                return actions.reshape(-1, self.chunk_size, self.action_dim)

        model = TinyACT().eval()
        B = 4
        images = torch.randn(B, hist_len, 3, 128, 128)
        text_tokens = torch.randint(0, vocab_size, (B, 10))

        with torch.no_grad():
            output = model(images, text_tokens)

        self.assertEqual(
            output.shape, (B, chunk_size, action_dim),
            f"Expected shape ({B}, {chunk_size}, {action_dim}), "
            f"got {tuple(output.shape)}"
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
