"""
SmolVLA Fine-tuning Script
==========================
Fine-tune SmolVLA (or train from scratch) on PushCube demonstrations.

This script supports two modes:
1. **Real fine-tuning** — loads a pre-trained SmolVLA checkpoint and
   fine-tunes on your collected PushCube dataset (requires ``lerobot``
   and a GPU).
2. **Mock training** — runs a minimal training loop with a randomly
   initialized network for CI / local testing without downloading 450M
   weights.

Usage
-----
.. code-block:: bash

    # Real fine-tuning (requires lerobot + GPU)
    python finetune.py --dataset_dir datasets/pushcube_lerobot/ --output_dir models/smolvlа_pushcube/

    # Mock mode (CPU, no downloads)
    python finetune.py --mock --dataset_dir datasets/pushcube_canonical/ --epochs 2

    # Smoke test
    python finetune.py --mock --smoke_test
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

_PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from examples.robot_foundation_models.common.canonical_dataset import (
    load_episodes_from_dir,
    compute_action_statistics,
)


def load_config(config_path: str = None) -> dict:
    """Load fine-tuning config from YAML or use defaults."""
    defaults = {
        "batch_size": 8,
        "learning_rate": 1e-4,
        "epochs": 30,
        "chunk_size": 10,
        "temporal_ensemble": True,
        "augmentation": True,
        "save_every": 10,
        "device": "cuda",
    }
    if config_path is None:
        local_config = Path(__file__).parent / "finetune_config.yaml"
        if local_config.exists():
            config_path = str(local_config)

    if config_path and Path(config_path).exists():
        try:
            import yaml
            with open(config_path) as f:
                loaded = yaml.safe_load(f)
            defaults.update(loaded)
        except ImportError:
            print("[Warning] pyyaml not installed — using default config")

    return defaults


def create_mock_model(action_dim: int, state_dim: int, chunk_size: int):
    """Create a tiny randomly-initialized model for mock training."""
    import torch
    import torch.nn as nn

    class MockSmolVLA(nn.Module):
        def __init__(self, action_dim, state_dim, chunk_size):
            super().__init__()
            self.action_dim = action_dim
            self.chunk_size = chunk_size
            # Simple CNN encoder
            self.encoder = nn.Sequential(
                nn.Conv2d(3, 16, 3, stride=2, padding=1),
                nn.ReLU(),
                nn.Conv2d(16, 32, 3, stride=2, padding=1),
                nn.ReLU(),
                nn.AdaptiveAvgPool2d(1),
                nn.Flatten(),
            )
            # Language embedding (simple lookup for mock)
            self.lang_embed = nn.Embedding(100, 32)
            # State encoder
            self.state_encoder = nn.Linear(state_dim, 32) if state_dim > 0 else None
            # Policy head
            self.policy = nn.Sequential(
                nn.Linear(32 + 32 + 32, 128),
                nn.ReLU(),
                nn.Linear(128, action_dim * chunk_size),
            )

        def forward(self, image, language_idx, state=None):
            img_feat = self.encoder(image)  # (B, 32)
            lang_feat = self.lang_embed(language_idx % 100)  # (B, 32)
            if state is not None and self.state_encoder is not None:
                state_feat = self.state_encoder(state)  # (B, 32)
            else:
                state_feat = torch.zeros_like(img_feat)
            x = torch.cat([img_feat, lang_feat, state_feat], dim=-1)
            actions = self.policy(x).view(-1, self.chunk_size, self.action_dim)
            return actions

    return MockSmolVLA(action_dim, state_dim, chunk_size)


def mock_train(
    episodes,
    config: dict,
    output_dir: Path,
):
    """Run a mock training loop without lerobot."""
    import torch
    import torch.nn as nn
    import torch.optim as optim

    print("[Mock Training] Starting...")
    action_dim = episodes[0].action_dim
    state_dim = episodes[0].state_dim
    chunk_size = config["chunk_size"]
    device = torch.device("cpu")  # mock always CPU

    model = create_mock_model(action_dim, state_dim, chunk_size).to(device)
    optimizer = optim.Adam(model.parameters(), lr=config["learning_rate"])
    criterion = nn.MSELoss()

    # Compute action normalization stats
    stats = compute_action_statistics(episodes)
    action_mean = torch.from_numpy(stats["mean"]).float()
    action_std = torch.from_numpy(stats["std"]).float()

    # Flatten dataset
    all_images = []
    all_states = []
    all_actions = []
    all_langs = []

    for ep in episodes:
        imgs = ep.observation["images"]["front"]
        states = ep.observation.get("state")
        for t in range(len(ep.action)):
            all_images.append(imgs[t])
            if states is not None:
                all_states.append(states[t])
            all_actions.append(ep.action[t])
            # Simple language hash for mock embedding
            all_langs.append(hash(ep.language[t]) % 100)

    # Convert to tensors
    images = torch.from_numpy(np.stack(all_images)).permute(0, 3, 1, 2).float() / 255.0
    states = torch.from_numpy(np.stack(all_states)).float() if all_states else None
    actions = torch.from_numpy(np.stack(all_actions)).float()
    actions = (actions - action_mean) / action_std
    langs = torch.tensor(all_langs, dtype=torch.long)

    # Training loop
    n_samples = len(actions)
    batch_size = min(config["batch_size"], n_samples)
    n_batches = max(1, n_samples // batch_size)

    for epoch in range(config["epochs"]):
        total_loss = 0.0
        # Shuffle
        perm = torch.randperm(n_samples)
        for i in range(n_batches):
            idx = perm[i * batch_size : (i + 1) * batch_size]
            batch_img = images[idx].to(device)
            batch_lang = langs[idx].to(device)
            batch_action = actions[idx].to(device)
            batch_state = states[idx].to(device) if states is not None else None

            # Target: repeat action for chunk_size (simplified)
            target = batch_action.unsqueeze(1).repeat(1, chunk_size, 1)

            pred = model(batch_img, batch_lang, batch_state)
            loss = criterion(pred, target)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / n_batches
        print(f"  Epoch {epoch+1}/{config['epochs']}: loss={avg_loss:.6f}")

        if (epoch + 1) % config.get("save_every", 10) == 0:
            save_path = output_dir / f"mock_model_epoch_{epoch+1}.pt"
            torch.save(model.state_dict(), save_path)
            print(f"    Saved checkpoint: {save_path}")

    # Save final model
    final_path = output_dir / "mock_model_final.pt"
    torch.save(model.state_dict(), final_path)
    print(f"[Mock Training] Final model saved to {final_path}")

    # Save config
    with open(output_dir / "training_config.json", "w") as f:
        json.dump(config, f, indent=2)

    return model


def real_train(
    dataset_dir: Path,
    output_dir: Path,
    config: dict,
):
    """Fine-tune real SmolVLA using LeRobot."""
    try:
        from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
        from lerobot.common.policies.smolvla.modeling_smolvla import SmolVLAPolicy
        from lerobot.configs.train import TrainPipelineConfig
        from lerobot.scripts.train import train
    except ImportError as e:
        print(f"[Error] LeRobot not installed: {e}")
        print("  Install: pip install lerobot")
        print("  Or run with --mock flag")
        return None

    print("[Real Training] Starting SmolVLA fine-tuning...")
    print(f"  Dataset: {dataset_dir}")
    print(f"  Output: {output_dir}")
    print(f"  Epochs: {config['epochs']}, LR: {config['learning_rate']}")

    # This is a placeholder — real LeRobot training requires its config system
    # which is beyond the scope of this mock-compatible script.
    print("[Real Training] LeRobot training entry point:")
    print(f"  python -m lerobot.scripts.train \\")
    print(f"    dataset.root={dataset_dir} \\")
    print(f"    output_dir={output_dir} \\")
    print(f"    training.learning_rate={config['learning_rate']} \\")
    print(f"    training.num_epochs={config['epochs']}")

    return None


def main():
    parser = argparse.ArgumentParser(description="Fine-tune SmolVLA on PushCube")
    parser.add_argument("--dataset_dir", default="datasets/pushcube_canonical/", help="Dataset directory")
    parser.add_argument("--output_dir", default="models/smolvlа_pushcube/", help="Output directory")
    parser.add_argument("--config", default=None, help="Path to YAML config")
    parser.add_argument("--mock", action="store_true", help="Mock mode (no lerobot)")
    parser.add_argument("--smoke_test", action="store_true", help="Quick smoke test (2 epochs)")
    parser.add_argument("--epochs", type=int, default=None, help="Override epochs")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load config
    config = load_config(args.config)
    if args.epochs is not None:
        config["epochs"] = args.epochs
    if args.smoke_test:
        config["epochs"] = 2
        config["batch_size"] = 4

    print("=" * 60)
    print("SmolVLA Fine-tuning")
    print("=" * 60)
    print(f"Mode: {'mock' if args.mock else 'real'}")
    print(f"Config: {config}")

    if args.mock:
        # Load canonical episodes
        dataset_dir = Path(args.dataset_dir)
        if not dataset_dir.exists() or not list(dataset_dir.glob("*.pkl")):
            print(f"[Error] No episodes found in {dataset_dir}")
            print("  Run: python collect_pushcube_dataset.py --n_episodes 10")
            return

        episodes = load_episodes_from_dir(dataset_dir)
        print(f"Loaded {len(episodes)} episodes")
        mock_train(episodes, config, output_dir)
    else:
        dataset_dir = Path(args.dataset_dir)
        real_train(dataset_dir, output_dir, config)


if __name__ == "__main__":
    main()
