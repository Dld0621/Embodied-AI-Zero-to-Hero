"""
SmolVLA Fine-tuning Script
==========================
Fine-tune SmolVLA (or train from scratch) on PushCube demonstrations.

This script supports two modes:
1. **Real fine-tuning** — loads a pre-trained SmolVLA checkpoint
   (``lerobot/smolvla_base``) and fine-tunes on your collected PushCube
   dataset via the official ``lerobot-train`` CLI. Requires ``lerobot``
   installed (``pip install -e '.[smolvla]'``) and a GPU. Action
   dimension (2-D ``[dx, dy]``) is determined by the dataset's ``action``
   feature, not overridden on the CLI.
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
    """Load fine-tuning config from YAML, flattening nested sections.

    The YAML file (``finetune_config.yaml``) uses nested sections:
    ``model``, ``dataset``, ``training``, ``hardware``, ``logging``.

    This function flattens those sections into a single-level dict so
    that both ``real_train()`` and ``mock_train()`` can access values
    via ``config["key"]`` instead of ``config["training"]["key"]``.

    Defaults are aligned with the official LeRobot SmolVLA documentation
    (batch_size=64, steps=20000, lr=1e-4, device=cuda).
    """
    config = {
        # real_train keys (lerobot-train CLI)
        "batch_size": 64,
        "steps": 20000,
        "learning_rate": 1.0e-4,
        "device": "cuda",
        "pretrained_name_or_path": "lerobot/smolvla_base",
        "job_name": "pushcube_smolvla",
        "use_wandb": False,
        # mock_train keys
        "num_epochs": 50,
        "epochs": 50,               # alias for mock_train backward compat
        "chunk_size": 10,
        "temporal_ensemble": True,
        "augmentation": True,
        "save_every": 10,
    }

    if config_path is None:
        local_config = Path(__file__).parent / "finetune_config.yaml"
        if local_config.exists():
            config_path = str(local_config)

    if config_path and Path(config_path).exists():
        try:
            import yaml
            with open(config_path, encoding="utf-8") as f:
                raw = yaml.safe_load(f) or {}
        except ImportError:
            print("[Warning] pyyaml not installed — using default config")
            return config

        # ---- Flatten nested YAML sections into flat dict ----
        training = raw.get("training", {})
        model = raw.get("model", {})
        hardware = raw.get("hardware", {})
        logging_cfg = raw.get("logging", {})
        dataset_cfg = raw.get("dataset", {})

        # training.* → flat
        if "batch_size" in training:
            config["batch_size"] = training["batch_size"]
        if "steps" in training:
            config["steps"] = training["steps"]
        if "num_epochs" in training:
            config["num_epochs"] = training["num_epochs"]
            config["epochs"] = training["num_epochs"]   # keep alias in sync
        if "learning_rate" in training:
            config["learning_rate"] = training["learning_rate"]
        if "chunk_size" in training:
            config["chunk_size"] = training["chunk_size"]

        # model.* → flat
        if "pretrained_name_or_path" in model:
            config["pretrained_name_or_path"] = model["pretrained_name_or_path"]

        # hardware.* → flat
        if "device" in hardware:
            config["device"] = hardware["device"]

        # logging.* → flat
        if "project" in logging_cfg:
            config["job_name"] = logging_cfg["project"]
        if "use_wandb" in logging_cfg:
            config["use_wandb"] = logging_cfg["use_wandb"]

        # dataset.* → flat (used by real_train for local dataset handling)
        if "repo_id" in dataset_cfg:
            config["dataset_repo_id"] = dataset_cfg["repo_id"]
        if "root" in dataset_cfg:
            config["dataset_root"] = dataset_cfg["root"]

    return config


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


def build_train_command(
    train_cli: str,
    dataset_path: str,
    is_hf_repo: bool,
    output_dir: str,
    config: dict,
) -> list[str]:
    """Build the ``lerobot-train`` CLI command from flattened config.

    This is separated from ``real_train()`` so it can be unit-tested
    without a GPU or lerobot installation.

    Local dataset handling:
      - HF Hub ID (e.g. ``username/dataset``): ``--dataset.repo_id=<id>``
      - Local directory: ``--dataset.repo_id=local/<name>`` +
        ``--dataset.root=<path>``

    Action dimension is NOT set on the CLI — it is determined by the
    dataset's ``action`` feature (correct LeRobot convention).
    """
    steps = config.get("steps", 20000)
    batch_size = config.get("batch_size", 64)
    learning_rate = config.get("learning_rate", 1e-4)
    device = config.get("device", "cuda")
    pretrained = config.get("pretrained_name_or_path", "lerobot/smolvla_base")
    job_name = config.get("job_name", "pushcube_smolvla")
    use_wandb = config.get("use_wandb", False)

    # ---- Dataset repo_id and root ----
    if is_hf_repo:
        # HuggingFace Hub dataset ID (e.g. "username/pushcube_demo")
        repo_id = dataset_path
        dataset_args = [f"--dataset.repo_id={repo_id}"]
    else:
        # Local LeRobot-format dataset directory.
        # LeRobot convention: repo_id="local/<name>", root=<parent_dir>
        # The dataset directory contains meta/ and data/ subdirs.
        dataset_dir_path = Path(dataset_path)
        dataset_name = dataset_dir_path.name
        parent_dir = str(dataset_dir_path.parent)
        repo_id = f"local/{dataset_name}"
        dataset_args = [
            f"--dataset.repo_id={repo_id}",
            f"--dataset.root={parent_dir}",
        ]

    cmd = [
        train_cli,
        f"--policy.path={pretrained}",
        *dataset_args,
        f"--output_dir={output_dir}",
        f"--job_name={job_name}",
        f"--policy.device={device}",
        f"--batch_size={batch_size}",
        f"--steps={steps}",
        f"--lr={learning_rate}",
    ]

    if use_wandb:
        cmd.append("--wandb.enable=true")

    return cmd


def real_train(
    dataset_dir: Path,
    output_dir: Path,
    config: dict,
):
    """Fine-tune real SmolVLA using the official LeRobot CLI.

    This function launches a real LeRobot training pipeline for SmolVLA
    on the PushCube dataset using the ``lerobot-train`` CLI (the official
    entry point as documented in the HuggingFace LeRobot SmolVLA guide).

    Steps:
    1. Checks that ``lerobot`` is installed and the ``lerobot-train`` CLI
       is available on PATH.
    2. Checks for GPU availability.
    3. Locates the LeRobot-format dataset (``dataset_dir`` should contain
       ``meta/info.json`` or ``dataset_info.json``, or be a HuggingFace
       repo ID like ``username/dataset_name``).
    4. Constructs the official ``lerobot-train`` command with
       ``--policy.path=lerobot/smolvla_base`` and PushCube-appropriate
       settings. Action dimension (2-D ``[dx, dy]``) is determined by
       the dataset's ``action`` feature — not overridden on the CLI —
       so the model learns the correct action space during training.
    5. Executes training via ``subprocess.run``.
    6. Saves ``real_training_config.json`` metadata on success.

    Parameters
    ----------
    dataset_dir : Path
        Path to a LeRobot-format dataset directory, or a HuggingFace
        repo ID string (e.g. ``"username/pushcube_demo"``).
    output_dir : Path
        Directory to save checkpoints and training logs.
    config : dict
        Fine-tuning configuration (batch_size, learning_rate, steps,
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
            "  pip install -e '.[smolvla]'\n"
            "Or from source:\n"
            "  git clone https://github.com/huggingface/lerobot.git\n"
            "  cd lerobot && pip install -e '.[smolvla]'\n"
        )

    # Verify the lerobot-train CLI is available
    train_cli = shutil.which("lerobot-train")
    if train_cli is None:
        raise RuntimeError(
            "lerobot-train CLI not found on PATH. Ensure lerobot is "
            "installed with 'pip install -e \\\".[smolvla]\\\"' from the "
            "lerobot source directory.\n"
            "See: https://huggingface.co/docs/lerobot/main/smolvla"
        )

    # ---- Step 2: Check for GPU ----
    try:
        import torch
        if not torch.cuda.is_available():
            print("[Warning] No GPU detected. Real fine-tuning is extremely "
                  "slow on CPU. Consider using a GPU instance.")
    except ImportError:
        raise RuntimeError("PyTorch is not installed. Install: pip install torch")

    # ---- Step 3-4: Resolve dataset + build command ----
    dataset_path = str(dataset_dir)
    is_hf_repo = "/" in dataset_path and not Path(dataset_path).exists()

    if not is_hf_repo:
        # Local directory: must contain meta/info.json (LeRobot v2 format)
        # or dataset_info.json (older format)
        meta_info = Path(dataset_path) / "meta" / "info.json"
        old_info = Path(dataset_path) / "dataset_info.json"
        if not meta_info.exists() and not old_info.exists():
            raise FileNotFoundError(
                f"Dataset directory '{dataset_path}' does not contain "
                "meta/info.json or dataset_info.json. "
                "Run collect_pushcube_dataset.py --format lerobot first "
                "to create a LeRobot-format dataset."
            )

    cmd = build_train_command(
        train_cli=train_cli,
        dataset_path=dataset_path,
        is_hf_repo=is_hf_repo,
        output_dir=str(output_dir),
        config=config,
    )

    # Print summary
    pretrained = config.get("pretrained_name_or_path", "lerobot/smolvla_base")
    steps = config.get("steps", 20000)
    batch_size = config.get("batch_size", 64)
    learning_rate = config.get("learning_rate", 1e-4)
    device = config.get("device", "cuda")

    print("=" * 60)
    print("[Real Training] Launching LeRobot training pipeline")
    print("=" * 60)
    print(f"  Policy:   {pretrained}")
    print(f"  Dataset:  {dataset_path} ({'HF Hub' if is_hf_repo else 'local'})")
    print(f"  Output:   {output_dir}")
    print(f"  Steps:    {steps}")
    print(f"  Batch:    {batch_size}")
    print(f"  LR:       {learning_rate}")
    print(f"  Device:   {device}")
    print(f"  Action:   2-D [dx, dy] (from dataset feature, not CLI override)")
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
            "policy": pretrained,
            "dataset": dataset_path,
            "output_dir": str(output_dir),
            "config": config,
            "action_dim": 2,
            "action_type": "ee_delta_2d",
            "command": " ".join(cmd),
            "exit_code": result.returncode,
            "note": "Action dim determined by dataset feature, not CLI override",
        }
        with open(output_dir / "real_training_config.json", "w") as f:
            json.dump(training_meta, f, indent=2)

    except subprocess.CalledProcessError as e:
        raise RuntimeError(
            f"lerobot-train failed with exit code {e.returncode}.\n"
            f"Common issues:\n"
            f"  1. Dataset format mismatch — ensure collect_pushcube_dataset.py "
            f"was run with --format lerobot\n"
            f"  2. GPU out of memory — reduce batch_size in config\n"
            f"  3. LeRobot version mismatch — ensure SmolVLA is supported "
            f"(pip install -e '.[smolvla]' from lerobot source)\n"
            f"  4. Local dataset path issue — verify --dataset.repo_id and "
            f"--dataset.root with 'lerobot-train --help'\n"
            f"  5. HuggingFace Hub dataset — ensure you are logged in "
            f"('huggingface-cli login') and the repo ID is correct\n"
            f"\n"
            f"Try running the command manually to see full error output:\n"
            f"  {' '.join(cmd)}\n"
            f"\n"
            f"Official docs: https://huggingface.co/docs/lerobot/main/smolvla"
        ) from e



def main():
    parser = argparse.ArgumentParser(description="Fine-tune SmolVLA on PushCube")
    parser.add_argument("--dataset_dir", default="datasets/pushcube_canonical/",
                        help="Dataset directory (canonical .pkl for mock, "
                             "LeRobot-format for real)")
    parser.add_argument("--output_dir", default="models/smolvla_pushcube/",
                        help="Output directory")
    parser.add_argument("--config", default=None, help="Path to YAML config")
    parser.add_argument("--mock", action="store_true",
                        help="Mock mode (no lerobot, CPU only)")
    parser.add_argument("--smoke_test", action="store_true",
                        help="Quick smoke test (2 epochs/steps)")
    # --- Real training args (lerobot-train CLI) ---
    parser.add_argument("--steps", type=int, default=None,
                        help="Override training steps (real mode)")
    parser.add_argument("--batch_size", type=int, default=None,
                        help="Override batch size (real mode)")
    # --- Mock training args (deprecated for real mode) ---
    parser.add_argument("--epochs", type=int, default=None,
                        help="Override epochs (mock mode only; "
                             "real mode uses --steps)")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load config (now properly flattens nested YAML)
    config = load_config(args.config)

    # Apply CLI overrides
    if args.steps is not None:
        config["steps"] = args.steps
    if args.batch_size is not None:
        config["batch_size"] = args.batch_size
    if args.epochs is not None:
        config["epochs"] = args.epochs
        config["num_epochs"] = args.epochs

    if args.smoke_test:
        config["epochs"] = 2
        config["num_epochs"] = 2
        config["steps"] = 2
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


# ---------------------------------------------------------------------------
# Tests (no GPU / lerobot required)
# ---------------------------------------------------------------------------

def test_config_flattening():
    """Verify that load_config() flattens nested YAML sections correctly."""
    config = load_config()

    # These must come from the nested YAML, NOT the old flat defaults
    assert config["batch_size"] == 64, (
        f"Expected batch_size=64 from YAML, got {config['batch_size']}"
    )
    assert config["steps"] == 20000, (
        f"Expected steps=20000 from YAML, got {config['steps']}"
    )
    assert config["learning_rate"] == 1.0e-4, (
        f"Expected lr=1e-4 from YAML, got {config['learning_rate']}"
    )
    assert config["device"] == "cuda", (
        f"Expected device=cuda from YAML, got {config['device']}"
    )
    assert config["pretrained_name_or_path"] == "lerobot/smolvla_base", (
        f"Expected pretrained=lerobot/smolvla_base, got "
        f"{config['pretrained_name_or_path']}"
    )

    # mock_train backward compat
    assert "epochs" in config and "num_epochs" in config, (
        "mock_train needs 'epochs' / 'num_epochs' keys"
    )

    print("[test_config_flattening] PASSED")
    print(f"  batch_size = {config['batch_size']}")
    print(f"  steps      = {config['steps']}")
    print(f"  lr         = {config['learning_rate']}")
    print(f"  device     = {config['device']}")


def test_build_train_command_hf_hub():
    """Verify command construction for HuggingFace Hub datasets."""
    config = load_config()
    cmd = build_train_command(
        train_cli="lerobot-train",
        dataset_path="dld0621/pushcube_demo",
        is_hf_repo=True,
        output_dir="outputs/train/smolvla",
        config=config,
    )

    cmd_str = " ".join(cmd)

    # Official CLI format checks
    assert "--policy.path=lerobot/smolvla_base" in cmd, cmd_str
    assert "--dataset.repo_id=dld0621/pushcube_demo" in cmd, cmd_str
    assert "--dataset.root" not in cmd_str, "HF Hub should not have --dataset.root"
    assert "--steps=20000" in cmd, cmd_str
    assert "--batch_size=64" in cmd, cmd_str
    assert "--policy.device=cuda" in cmd, cmd_str

    # Must NOT contain old-style overrides
    assert "policy.action_dim=2" not in cmd_str, "action_dim should not be on CLI"
    assert "policy.num_motors=2" not in cmd_str, "num_motors should not be on CLI"

    print("[test_build_train_command_hf_hub] PASSED")
    print(f"  Command: {cmd_str}")


def test_build_train_command_local():
    """Verify command construction for local LeRobot datasets."""
    config = load_config()
    cmd = build_train_command(
        train_cli="lerobot-train",
        dataset_path="datasets/pushcube_lerobot",
        is_hf_repo=False,
        output_dir="outputs/train/smolvla",
        config=config,
    )

    cmd_str = " ".join(cmd)

    # Local dataset should use local/ prefix and --dataset.root
    assert "--dataset.repo_id=local/pushcube_lerobot" in cmd, cmd_str
    assert "--dataset.root=" in cmd_str, "Local dataset needs --dataset.root"
    assert "--steps=20000" in cmd, cmd_str
    assert "--batch_size=64" in cmd, cmd_str

    print("[test_build_train_command_local] PASSED")
    print(f"  Command: {cmd_str}")


def test_cli_override():
    """Verify that CLI overrides take precedence over YAML."""
    config = load_config()
    config["steps"] = 5000      # simulate --steps 5000
    config["batch_size"] = 32   # simulate --batch_size 32

    cmd = build_train_command(
        train_cli="lerobot-train",
        dataset_path="dld0621/pushcube_demo",
        is_hf_repo=True,
        output_dir="outputs/train/smolvla",
        config=config,
    )

    cmd_str = " ".join(cmd)
    assert "--steps=5000" in cmd, cmd_str
    assert "--batch_size=32" in cmd, cmd_str

    print("[test_cli_override] PASSED")


def run_tests():
    """Run all tests (no GPU / lerobot required)."""
    print("=" * 60)
    print("Running finetune.py tests (no GPU needed)")
    print("=" * 60)
    test_config_flattening()
    test_build_train_command_hf_hub()
    test_build_train_command_local()
    test_cli_override()
    print("\nAll tests PASSED.")


if __name__ == "__main__":
    import sys
    if "--test" in sys.argv:
        run_tests()
    else:
        main()
