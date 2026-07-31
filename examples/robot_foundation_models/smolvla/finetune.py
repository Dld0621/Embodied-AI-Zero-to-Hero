"""
SmolVLA Fine-tuning Script
==========================
Fine-tune SmolVLA (or train from scratch) on PushCube demonstrations.

This script supports two modes:
1. **Real fine-tuning** — loads a pre-trained SmolVLA checkpoint and
   fine-tunes on your collected PushCube dataset via the LeRobot training
   pipeline. Requires ``lerobot`` installed and a GPU. The training
   command is constructed with PushCube-specific settings (``action_dim=2``,
   ``chunk_size`` from config) and executed via ``lerobot.scripts.train``.
2. **Mock training** — runs a minimal training loop with a randomly
   initialized network for CI / local testing without downloading 450M
   weights.

Usage
-----
.. code-block:: bash

    # Real fine-tuning (requires lerobot + GPU + LeRobot-format dataset)
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
    """Fine-tune real SmolVLA using LeRobot.

    This function attempts to set up and launch a real LeRobot training
    pipeline for SmolVLA on the PushCube dataset. It:

    1. Checks that ``lerobot`` is installed and a GPU is available.
    2. Locates the LeRobot-format dataset (``dataset_dir`` should contain
       a ``dataset_info.json`` or be a HuggingFace repo ID).
    3. Constructs a ``SmolVLAConfig`` with PushCube-specific settings
       (``action_dim=2``, ``chunk_size`` from config).
    4. Launches training via ``lerobot.scripts.train`` or the LeRobot CLI.

    If any step fails, a clear error is raised with instructions.

    Parameters
    ----------
    dataset_dir : Path
        Path to a LeRobot-format dataset directory (must contain
        ``dataset_info.json``), or a HuggingFace repo ID string.
    output_dir : Path
        Directory to save checkpoints and training logs.
    config : dict
        Fine-tuning configuration (batch_size, learning_rate, epochs,
        chunk_size, etc.).
    """
    import subprocess
    import shutil

    # ---- Step 1: Check lerobot installation ----
    try:
        import lerobot  # noqa: F401
    except ImportError:
        raise RuntimeError(
            "lerobot is not installed. Install it first:\n"
            "  pip install lerobot\n"
            "Or from source:\n"
            "  git clone https://github.com/huggingface/lerobot.git\n"
            "  cd lerobot && pip install -e .\n"
        )

    # ---- Step 2: Check for GPU ----
    try:
        import torch
        if not torch.cuda.is_available():
            print("[Warning] No GPU detected. Real fine-tuning is extremely "
                  "slow on CPU. Consider using a GPU instance.")
    except ImportError:
        raise RuntimeError("PyTorch is not installed. Install: pip install torch")

    # ---- Step 3: Locate dataset ----
    dataset_path = str(dataset_dir)
    is_hf_repo = "/" in dataset_path and not Path(dataset_path).exists()

    if not is_hf_repo:
        # Local directory: must contain dataset_info.json (LeRobot format)
        info_file = Path(dataset_path) / "dataset_info.json"
        if not info_file.exists():
            # Try meta/ subdirectory (newer LeRobot convention)
            meta_info = Path(dataset_path) / "meta" / "info.json"
            if not meta_info.exists():
                raise FileNotFoundError(
                    f"Dataset directory '{dataset_path}' does not contain "
                    "dataset_info.json or meta/info.json. "
                    "Run collect_pushcube_dataset.py --format lerobot first "
                    "to create a LeRobot-format dataset."
                )

    # ---- Step 4: Build training command ----
    # LeRobot uses Hydra config system. We construct a CLI command that
    # overrides the SmolVLA config with PushCube-specific settings.
    epochs = config.get("epochs", 30)
    batch_size = config.get("batch_size", 8)
    learning_rate = config.get("learning_rate", 1e-4)
    chunk_size = config.get("chunk_size", 10)

    cmd = [
        sys.executable, "-m", "lerobot.scripts.train",
        f"policy=smolvla",
        f"dataset_repo_id={dataset_path}",
        f"output_dir={output_dir}",
        f"policy.chunk_size={chunk_size}",
        f"batch_size={batch_size}",
        f"lr={learning_rate}",
        f"epochs={epochs}",
        # PushCube-specific: 2-D action space [dx, dy]
        "policy.action_dim=2",
        "policy.num_motors=2",
    ]

    print("=" * 60)
    print("[Real Training] Launching LeRobot training pipeline")
    print("=" * 60)
    print(f"  Dataset:  {dataset_path}")
    print(f"  Output:   {output_dir}")
    print(f"  Epochs:   {epochs}")
    print(f"  Batch:    {batch_size}")
    print(f"  LR:       {learning_rate}")
    print(f"  Chunk:    {chunk_size}")
    print(f"  Action:   2-D [dx, dy] (PushCube)")
    print(f"\n  Command:")
    print(f"  {' '.join(cmd)}")
    print()

    # ---- Step 5: Execute training ----
    try:
        result = subprocess.run(cmd, check=True)
        print(f"\n[Real Training] Training completed successfully.")
        print(f"  Checkpoint saved to: {output_dir}")

        # Save training metadata
        training_meta = {
            "dataset": dataset_path,
            "output_dir": str(output_dir),
            "config": config,
            "action_dim": 2,
            "action_type": "ee_delta_2d",
            "command": " ".join(cmd),
            "exit_code": result.returncode,
        }
        with open(output_dir / "real_training_config.json", "w") as f:
            json.dump(training_meta, f, indent=2)

    except subprocess.CalledProcessError as e:
        raise RuntimeError(
            f"LeRobot training failed with exit code {e.returncode}.\n"
            f"Common issues:\n"
            f"  1. Dataset format mismatch — ensure collect_pushcube_dataset.py "
            f"was run with --format lerobot\n"
            f"  2. GPU out of memory — reduce batch_size in config\n"
            f"  3. LeRobot version mismatch — check that your lerobot version "
            f"supports SmolVLA (src/lerobot/policies/smolvla/)\n"
            f"\n"
            f"Try running the command manually to see full error output:\n"
            f"  {' '.join(cmd)}"
        ) from e



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
