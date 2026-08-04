"""
Lightweight VLA Training Script
================================
Trains a small CNN + language + state policy on real PushCube expert
demonstrations.  This is NOT the full 450M SmolVLA model (which requires
LeRobot + GPU), but a lightweight VLA that:

1. Takes 128x128 RGB image + 12-D state (no goal-color one-hot) + language
2. Outputs 2-D action [dx, dy]
3. Trains on real expert data (50 episodes, ~1788 frames)
4. Splits by episode (not frames) to prevent validation leakage
5. Can run closed-loop evaluation on CPU

The resulting checkpoint can be loaded by ``SmolVLAAdapter`` when
``pretrained_name_or_path`` points to the ``.pt`` file.

Key design decisions (P0 fixes from 89/100 review):
- **Episode-level split**: train/val/test split by .pkl episode files, not
  random frame sampling. Prevents same-episode frames leaking across splits.
- **No goal-color one-hot in state**: state is sliced to first 12 dims
  (arm pos, cube pos, target pos, cube colors). The model MUST use language
  to know which cube to push. state_dim=12 (not 14).
- **Proper tokenizer**: padding_idx=0 in Embedding, masked mean pooling
  so padding tokens don't dilute the language signal.

Usage
-----
.. code-block:: bash

    cd examples/robot_foundation_models/smolvla
    python train_lightweight_vla.py --epochs 100 --batch_size 64

    # Quick smoke test (5 epochs)
    python train_lightweight_vla.py --epochs 5 --smoke-test
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
import sys
import time
from pathlib import Path
from typing import List, Tuple, Dict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# Add project root for imports
_PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "..")
)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)


# ------------------------------------------------------------------
# Constants
# ------------------------------------------------------------------

# Full PushCube state is 14-D:
#   [arm_x, arm_y, cube1_x, cube1_y, cube2_x, cube2_y,
#    target_x, target_y, cube1_r, cube1_g, cube2_r, cube2_g,
#    goal_red, goal_green]
#
# VLA state is 12-D: first 12 dims only (excludes goal-color one-hot).
# This forces the model to use language to identify the target cube.
VLA_STATE_DIM = 12
FULL_STATE_DIM = 14
ACTION_DIM = 2
PAD_TOKEN = 0  # Reserved for padding in tokenizer


# ------------------------------------------------------------------
# Model
# ------------------------------------------------------------------

class LightweightVLA(nn.Module):
    """Small CNN + language + state -> action policy.

    Architecture:
    - Image encoder: 4-layer CNN -> 128-D feature
    - Language encoder: Embedding(padding_idx=0) -> masked mean pool -> 32-D
    - State encoder: 2-layer MLP -> 64-D
    - Policy head: 224-D -> 128 -> 64 -> 2 (action)

    The state input is 12-D (no goal-color one-hot), so the model
    MUST use the language instruction to identify the target cube.
    """

    def __init__(
        self,
        state_dim: int = VLA_STATE_DIM,
        action_dim: int = ACTION_DIM,
        img_size: int = 128,
        vocab_size: int = 200,
        lang_dim: int = 32,
        img_feat_dim: int = 128,
        state_hidden: int = 64,
    ):
        super().__init__()

        # Image encoder: (3, 128, 128) -> (128,)
        self.image_encoder = nn.Sequential(
            nn.Conv2d(3, 32, 3, stride=2, padding=1),   # 64x64
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),   # 32x32
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, stride=2, padding=1),    # 16x16
            nn.ReLU(),
            nn.Conv2d(64, 128, 3, stride=2, padding=1),   # 8x8
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(128, img_feat_dim),
            nn.ReLU(),
        )

        # Language encoder: Embedding with padding_idx=0
        # Token 0 is <pad>; its embedding is zeroed and excluded from mean.
        self.language_encoder = nn.Embedding(vocab_size, lang_dim, padding_idx=PAD_TOKEN)

        # State encoder
        self.state_encoder = nn.Sequential(
            nn.Linear(state_dim, state_hidden),
            nn.ReLU(),
            nn.Linear(state_hidden, state_hidden),
            nn.ReLU(),
        )

        # Policy head
        combined_dim = img_feat_dim + lang_dim + state_hidden
        self.policy_head = nn.Sequential(
            nn.Linear(combined_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, action_dim),
        )

    def forward(
        self,
        image: torch.Tensor,
        state: torch.Tensor,
        lang_tokens: torch.Tensor,
    ) -> torch.Tensor:
        """Forward pass.

        Parameters
        ----------
        image : (B, 3, H, W) float32, normalized to [0, 1]
        state : (B, state_dim) float32 — first 12 dims of PushCube state
        lang_tokens : (B, max_len) long, token indices (0 = padding)
        """
        img_feat = self.image_encoder(image)

        # Masked mean pooling for language
        embeddings = self.language_encoder(lang_tokens)  # (B, L, D)
        mask = lang_tokens.ne(PAD_TOKEN).float().unsqueeze(-1)  # (B, L, 1)
        lang_feat = (embeddings * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)  # (B, D)

        state_feat = self.state_encoder(state)
        combined = torch.cat([img_feat, lang_feat, state_feat], dim=-1)
        return self.policy_head(combined)


# ------------------------------------------------------------------
# Tokenizer
# ------------------------------------------------------------------

class SimpleTokenizer:
    """Word-level tokenizer with hashing for compact vocabulary.

    Token 0 is reserved as <pad>. Real words are hashed to indices
    1..vocab_size-1 to avoid collision with padding.
    """

    def __init__(self, vocab_size: int = 200):
        self.vocab_size = vocab_size
        self.word2idx: Dict[str, int] = {}

    def _hash(self, word: str) -> int:
        """Deterministic hash to index 1..vocab_size-1 (0 is reserved for pad)."""
        h = 0
        for ch in word:
            h = (h * 31 + ord(ch)) % (self.vocab_size - 1)
        return h + 1  # Shift to range [1, vocab_size-1]

    def encode(self, text: str, max_len: int = 20) -> np.ndarray:
        """Encode text to token indices, padded to max_len with 0 (pad)."""
        words = text.lower().strip().split()
        tokens = []
        for w in words:
            if w not in self.word2idx:
                idx = self._hash(w)
                self.word2idx[w] = idx
            tokens.append(self.word2idx[w])
        # Pad or truncate
        if len(tokens) < max_len:
            tokens.extend([PAD_TOKEN] * (max_len - len(tokens)))
        else:
            tokens = tokens[:max_len]
        return np.array(tokens, dtype=np.int64)


# ------------------------------------------------------------------
# Dataset
# ------------------------------------------------------------------

class PushCubeEpisodeLoader:
    """Loads .pkl episode files and splits by episode (not by frame).

    This prevents validation set leakage where adjacent frames from the
    same episode appear in both train and validation sets.
    """

    @staticmethod
    def load_episodes(episodes_dir: str) -> List[Dict]:
        """Load all .pkl episodes from a directory.

        Returns a list of episode dicts, each containing:
        - 'images': list of (H, W, 3) uint8 arrays
        - 'states': list of (14,) float32 arrays
        - 'actions': list of (2,) float32 arrays
        - 'languages': list of strings
        - 'filename': source .pkl filename
        """
        episodes_dir = Path(episodes_dir)
        pkl_files = sorted(episodes_dir.glob("*.pkl"))
        if not pkl_files:
            raise FileNotFoundError(f"No .pkl files in {episodes_dir}")

        episodes = []
        for pkl_path in pkl_files:
            with open(pkl_path, "rb") as f:
                ep = pickle.load(f)

            if hasattr(ep, "action"):
                actions = ep.action
                images_front = ep.observation["images"]["front"]
                states = ep.observation["state"]
                languages = ep.language
            elif isinstance(ep, dict):
                actions = ep["action"]
                images_front = ep["observation"]["images"]["front"]
                states = ep["observation"]["state"]
                languages = ep["language"]
            else:
                print(f"  WARNING: Unknown episode format in {pkl_path.name}, skipping")
                continue

            episodes.append({
                "images": images_front,
                "states": states,
                "actions": actions,
                "languages": languages,
                "filename": pkl_path.name,
            })

        print(f"[EpisodeLoader] Loaded {len(episodes)} episodes from {episodes_dir}")
        return episodes

    @staticmethod
    def split_episodes(
        episodes: List[Dict],
        n_train: int = 40,
        n_val: int = 5,
        n_test: int = 5,
        seed: int = 42,
    ) -> Tuple[List[Dict], List[Dict], List[Dict]]:
        """Split episodes into train/val/test by episode (not by frame).

        Default: 40 train / 5 val / 5 test (out of 50 episodes).
        """
        rng = np.random.RandomState(seed)
        indices = rng.permutation(len(episodes))

        train_idx = indices[:n_train]
        val_idx = indices[n_train:n_train + n_val]
        test_idx = indices[n_train + n_val:n_train + n_val + n_test]

        train_eps = [episodes[i] for i in train_idx]
        val_eps = [episodes[i] for i in val_idx]
        test_eps = [episodes[i] for i in test_idx]

        print(f"[Split] Train: {len(train_eps)} eps, Val: {len(val_eps)} eps, Test: {len(test_eps)} eps")

        # Print episode filenames for reproducibility
        print(f"  Train episodes: {[e['filename'] for e in train_eps]}")
        print(f"  Val episodes: {[e['filename'] for e in val_eps]}")
        print(f"  Test episodes: {[e['filename'] for e in test_eps]}")

        return train_eps, val_eps, test_eps


class PushCubeFrameDataset(Dataset):
    """Expands a list of episodes into individual frame samples.

    Each sample is (image, state[:12], lang_tokens, action).
    State is sliced to first 12 dims to exclude goal-color one-hot,
    forcing the model to use language for target identification.
    """

    def __init__(
        self,
        episodes: List[Dict],
        tokenizer: SimpleTokenizer,
        max_lang_len: int = 20,
    ):
        self.tokenizer = tokenizer
        self.max_lang_len = max_lang_len
        self.samples: List[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = []

        for ep in episodes:
            n = len(ep["actions"])
            for t in range(n):
                img = ep["images"][t]
                state = ep["states"][t]
                action = ep["actions"][t]
                lang = ep["languages"][t]

                # Slice state to first 12 dims (remove goal_red, goal_green)
                vla_state = state[:VLA_STATE_DIM].copy()

                lang_tokens = tokenizer.encode(lang, self.max_lang_len)
                self.samples.append((img, vla_state, lang_tokens, action))

        print(f"[FrameDataset] {len(self.samples)} frames from {len(episodes)} episodes")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        img, state, lang_tokens, action = self.samples[idx]
        # Normalize image to [0, 1] and convert to (C, H, W)
        img_f = img.astype(np.float32) / 255.0
        img_chw = np.transpose(img_f, (2, 0, 1))
        return (
            torch.from_numpy(img_chw),
            torch.from_numpy(state.astype(np.float32)),
            torch.from_numpy(lang_tokens),
            torch.from_numpy(action.astype(np.float32)),
        )


# ------------------------------------------------------------------
# Training
# ------------------------------------------------------------------

def train(
    data_dir: str,
    output_dir: str,
    epochs: int = 100,
    batch_size: int = 64,
    lr: float = 1e-3,
    smoke_test: bool = False,
    seed: int = 42,
):
    """Train lightweight VLA on PushCube expert data.

    Uses episode-level train/val/test split to prevent frame leakage.
    State is 12-D (no goal-color one-hot) to force language dependency.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Tokenizer
    tokenizer = SimpleTokenizer(vocab_size=200)

    # Load and split episodes (not frames!)
    all_episodes = PushCubeEpisodeLoader.load_episodes(data_dir)

    n_total_eps = len(all_episodes)
    n_test = max(1, n_total_eps // 10)   # ~10% test
    n_val = max(1, n_total_eps // 10)    # ~10% val
    n_train = n_total_eps - n_val - n_test

    train_eps, val_eps, test_eps = PushCubeEpisodeLoader.split_episodes(
        all_episodes, n_train, n_val, n_test, seed=seed
    )

    # Create frame-level datasets from episode splits
    train_dataset = PushCubeFrameDataset(train_eps, tokenizer)
    val_dataset = PushCubeFrameDataset(val_eps, tokenizer)
    test_dataset = PushCubeFrameDataset(test_eps, tokenizer)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    n_train_frames = len(train_dataset)
    n_val_frames = len(val_dataset)

    # Model — state_dim=12 (no goal-color one-hot)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = LightweightVLA(state_dim=VLA_STATE_DIM, action_dim=ACTION_DIM).to(device)
    print(f"[Model] LightweightVLA on {device}")
    print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"  State dim: {VLA_STATE_DIM} (excludes goal-color one-hot)")
    print(f"  Train: {n_train_frames} frames ({n_train} eps), Val: {n_val_frames} frames ({n_val} eps)")
    print(f"  Test: {len(test_dataset)} frames ({n_test} eps)")

    # Optimizer + scheduler
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    # Training loop
    best_val_loss = float("inf")
    best_epoch = 0
    history = []

    for epoch in range(1, epochs + 1):
        # Train
        model.train()
        train_loss = 0.0
        n_batches = 0
        for imgs, states, lang_tokens, actions in train_loader:
            imgs, states = imgs.to(device), states.to(device)
            lang_tokens, actions = lang_tokens.to(device), actions.to(device)

            pred = model(imgs, states, lang_tokens)
            loss = F.mse_loss(pred, actions)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            n_batches += 1

        train_loss /= max(1, n_batches)

        # Validate
        model.eval()
        val_loss = 0.0
        val_mae = 0.0
        n_val_batches = 0
        with torch.no_grad():
            for imgs, states, lang_tokens, actions in val_loader:
                imgs, states = imgs.to(device), states.to(device)
                lang_tokens, actions = lang_tokens.to(device), actions.to(device)

                pred = model(imgs, states, lang_tokens)
                val_loss += F.mse_loss(pred, actions).item()
                val_mae += (pred - actions).abs().mean().item()
                n_val_batches += 1

        val_loss /= max(1, n_val_batches)
        val_mae /= max(1, n_val_batches)

        scheduler.step()

        history.append({
            "epoch": epoch,
            "train_loss": round(train_loss, 6),
            "val_loss": round(val_loss, 6),
            "val_mae": round(val_mae, 6),
            "lr": round(optimizer.param_groups[0]["lr"], 6),
        })

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            checkpoint = {
                "model_state_dict": model.state_dict(),
                "model_config": {
                    "state_dim": VLA_STATE_DIM,
                    "action_dim": ACTION_DIM,
                    "img_size": 128,
                    "vocab_size": 200,
                    "lang_dim": 32,
                    "img_feat_dim": 128,
                    "state_hidden": 64,
                },
                "epoch": epoch,
                "val_loss": val_loss,
                "val_mae": val_mae,
                "train_samples": n_train_frames,
                "val_samples": n_val_frames,
                "test_samples": len(test_dataset),
                "training_info": {
                    "epochs": epochs,
                    "batch_size": batch_size,
                    "lr": lr,
                    "device": str(device),
                    "data_source": f"pushcube_canonical ({n_total_eps} expert episodes)",
                    "split_method": "episode-level (no frame leakage)",
                    "state_dim_explanation": "12-D (excludes goal_red, goal_green to force language dependency)",
                    "tokenizer": "padding_idx=0, masked mean pooling",
                    "seed": seed,
                },
                "split_info": {
                    "train_episodes": [e["filename"] for e in train_eps],
                    "val_episodes": [e["filename"] for e in val_eps],
                    "test_episodes": [e["filename"] for e in test_eps],
                },
            }
            torch.save(checkpoint, output_dir / "lightweight_vla_pushcube.pt")

        if epoch % 10 == 0 or epoch == 1 or epoch == epochs:
            print(
                f"  Epoch {epoch:3d}/{epochs} | "
                f"train_loss={train_loss:.6f} | "
                f"val_loss={val_loss:.6f} | "
                f"val_mae={val_mae:.6f} | "
                f"best={best_val_loss:.6f} (ep {best_epoch})"
            )

        if smoke_test and epoch >= 5:
            print("  [smoke-test] Stopping after 5 epochs")
            break

    # Final test evaluation
    model.eval()
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    test_loss = 0.0
    test_mae = 0.0
    n_test_batches = 0
    with torch.no_grad():
        for imgs, states, lang_tokens, actions in test_loader:
            imgs, states = imgs.to(device), states.to(device)
            lang_tokens, actions = lang_tokens.to(device), actions.to(device)
            pred = model(imgs, states, lang_tokens)
            test_loss += F.mse_loss(pred, actions).item()
            test_mae += (pred - actions).abs().mean().item()
            n_test_batches += 1
    test_loss /= max(1, n_test_batches)
    test_mae /= max(1, n_test_batches)
    print(f"\n[Test] test_loss={test_loss:.6f}, test_mae={test_mae:.6f} ({len(test_dataset)} frames)")

    # Save training history
    with open(output_dir / "training_history.json", "w") as f:
        json.dump({
            "best_epoch": best_epoch,
            "best_val_loss": best_val_loss,
            "test_loss": test_loss,
            "test_mae": test_mae,
            "history": history,
            "model_type": "LightweightVLA",
            "parameters": sum(p.numel() for p in model.parameters()),
            "data_source": f"pushcube_canonical ({n_total_eps} expert episodes)",
            "split_method": "episode-level (40 train / 5 val / 5 test)",
            "state_dim": VLA_STATE_DIM,
            "state_excludes": "goal_red, goal_green (forces language dependency)",
            "tokenizer_fix": "padding_idx=0, masked mean pooling",
            "seed": seed,
        }, f, indent=2)

    print(f"\n{'='*60}")
    print(f"Training complete.")
    print(f"  Best epoch: {best_epoch}")
    print(f"  Best val_loss: {best_val_loss:.6f}")
    print(f"  Test loss: {test_loss:.6f}, Test MAE: {test_mae:.6f}")
    print(f"  Checkpoint: {output_dir / 'lightweight_vla_pushcube.pt'}")
    print(f"  History: {output_dir / 'training_history.json'}")
    print(f"{'='*60}")

    return best_val_loss


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Train Lightweight VLA on PushCube")
    parser.add_argument(
        "--data_dir",
        default="datasets/pushcube_canonical",
        help="Path to .pkl episodes directory",
    )
    parser.add_argument(
        "--output_dir",
        default="models/lightweight_vla",
        help="Output directory for checkpoint",
    )
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42, help="Random seed for episode split")
    parser.add_argument("--smoke-test", action="store_true", help="Quick 5-epoch test")
    args = parser.parse_args()

    # Resolve data dir relative to script location
    script_dir = Path(__file__).parent
    data_dir = script_dir / args.data_dir
    output_dir = script_dir / args.output_dir

    if not data_dir.exists():
        print(f"ERROR: Data directory not found: {data_dir}")
        sys.exit(1)

    train(
        data_dir=str(data_dir),
        output_dir=str(output_dir),
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        smoke_test=args.smoke_test,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
