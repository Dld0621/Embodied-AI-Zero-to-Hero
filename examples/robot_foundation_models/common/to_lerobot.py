"""
Canonical → LeRobot Dataset Converter
=====================================
Convert ``CanonicalEpisode`` objects to `LeRobot dataset format`_.

LeRobot uses `HuggingFace datasets`_ under the hood with a specific
Parquet schema.  This converter writes:

1. ``data/`` — Parquet files with all timesteps
2. ``meta/`` — JSON metadata (features, stats, info)

.. _LeRobot dataset format: https://github.com/huggingface/lerobot/blob/main/lerobot/common/datasets/README.md
.. _HuggingFace datasets: https://huggingface.co/docs/datasets

Usage
-----
.. code-block:: python

    from canonical_dataset import load_episodes_from_dir
    from to_lerobot import convert_to_lerobot

    episodes = load_episodes_from_dir("datasets/pushcube_canonical/")
    convert_to_lerobot(
        episodes,
        output_dir="datasets/pushcube_lerobot/",
        dataset_name="pushcube_dual_cube",
    )

Then train with LeRobot::

    python lerobot/scripts/train.py \
        dataset.repo_id=pushcube_dual_cube \
        dataset.root=datasets/pushcube_lerobot/

Dependencies
------------
- ``pip install lerobot`` (for real conversion)
- This module has a ``mock`` mode that writes Parquet directly via
  ``pyarrow`` when LeRobot is not installed.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from canonical_dataset import CanonicalEpisode, compute_action_statistics


def convert_to_lerobot(
    episodes: List[CanonicalEpisode],
    output_dir: str | Path,
    dataset_name: str = "canonical_dataset",
    fps: Optional[float] = None,
    mock: bool = False,
):
    """Convert canonical episodes to LeRobot dataset.

    Parameters
    ----------
    episodes : list[CanonicalEpisode]
    output_dir : str or Path
        Root directory for the LeRobot dataset.
    dataset_name : str
        Human-readable dataset identifier.
    fps : float or None
        Frames per second.  If None, uses ``episodes[0].control_frequency``.
    mock : bool
        If True (or ``lerobot`` is unavailable), write Parquet via pyarrow
        without importing LeRobot.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if fps is None and episodes:
        fps = episodes[0].control_frequency

    # ------------------------------------------------------------------
    # Try real LeRobot converter
    # ------------------------------------------------------------------
    if not mock:
        try:
            _convert_with_lerobot(episodes, output_dir, dataset_name, fps)
            print(f"[LeRobot] Dataset written to {output_dir}")
            return
        except ImportError:
            print("[LeRobot] lerobot not installed — falling back to mock Parquet writer.")

    # ------------------------------------------------------------------
    # Fallback: write Parquet + JSON metadata manually
    # ------------------------------------------------------------------
    _convert_with_pyarrow(episodes, output_dir, dataset_name, fps)


def _convert_with_lerobot(
    episodes: List[CanonicalEpisode],
    output_dir: Path,
    dataset_name: str,
    fps: float,
):
    """Real LeRobot conversion (requires ``pip install lerobot``)."""
    from lerobot.common.datasets.lerobot_dataset import LeRobotDataset

    # Infer feature shapes from first episode
    ep0 = episodes[0]
    cam_names = ep0.camera_names

    # Build feature dictionary
    features = {}
    for cam in cam_names:
        h, w, c = ep0.observation["images"][cam][0].shape
        features[f"observation.images.{cam}"] = {"shape": (c, h, w), "dtype": "uint8"}

    if ep0.state_dim > 0:
        features["observation.state"] = {"shape": (ep0.state_dim,), "dtype": "float32"}

    features["action"] = {"shape": (ep0.action_dim,), "dtype": "float32"}

    # Create dataset
    dataset = LeRobotDataset.create(
        repo_id=dataset_name,
        root=output_dir,
        fps=fps,
        features=features,
        use_videos=False,  # store frames as individual images
    )

    # Add episodes
    for ep in episodes:
        for t in range(ep.length):
            frame = {
                "action": ep.action[t],
                "timestamp": ep.timestamps[t],
            }
            for cam in cam_names:
                img = ep.observation["images"][cam][t]
                # LeRobot expects (C, H, W)
                frame[f"observation.images.{cam}"] = img.transpose(2, 0, 1)
            if ep.state_dim > 0:
                frame["observation.state"] = ep.observation["state"][t]

            dataset.add_frame(frame)

        dataset.save_episode(task=ep.task)

    # Consolidate
    dataset.consolidate()


def _convert_with_pyarrow(
    episodes: List[CanonicalEpisode],
    output_dir: Path,
    dataset_name: str,
    fps: float,
):
    """Mock conversion using pyarrow / pandas (no LeRobot dependency)."""
    try:
        import pandas as pd
        import pyarrow as pa
        import pyarrow.parquet as pq
    except ImportError:
        print("[to_lerobot] pyarrow not installed — cannot write Parquet.")
        print("  Install: pip install pyarrow pandas")
        print("  Or run with mock=False and lerobot installed.")
        return

    data_dir = output_dir / "data"
    data_dir.mkdir(parents=True, exist_ok=True)

    # Flatten all timesteps into a DataFrame
    rows = []
    episode_index = []
    frame_index = []

    for ep_idx, ep in enumerate(episodes):
        for t in range(ep.length):
            row: Dict[str, any] = {
                "episode_index": ep_idx,
                "frame_index": t,
                "timestamp": ep.timestamps[t],
                "action": ep.action[t].tolist(),
                "language": ep.language[t],
                "reward": ep.reward[t],
                "success": ep.success[t],
            }
            # Images: store as bytes (simplified — real LeRobot uses video files)
            for cam_name, imgs in ep.observation.get("images", {}).items():
                row[f"image_{cam_name}"] = imgs[t].tobytes()
                row[f"image_{cam_name}_shape"] = list(imgs[t].shape)

            if ep.state_dim > 0:
                row["state"] = ep.observation["state"][t].tolist()

            rows.append(row)
            episode_index.append(ep_idx)
            frame_index.append(t)

    df = pd.DataFrame(rows)

    # Write Parquet
    parquet_path = data_dir / "train.parquet"
    pq.write_table(pa.Table.from_pandas(df), parquet_path)

    # Write metadata
    stats = compute_action_statistics(episodes)
    meta = {
        "dataset_name": dataset_name,
        "fps": fps,
        "n_episodes": len(episodes),
        "total_frames": len(rows),
        "action_dim": episodes[0].action_dim if episodes else 0,
        "state_dim": episodes[0].state_dim if episodes else 0,
        "camera_names": episodes[0].camera_names if episodes else [],
        "action_stats": {
            k: v.tolist() for k, v in stats.items()
        },
    }

    meta_dir = output_dir / "meta"
    meta_dir.mkdir(parents=True, exist_ok=True)
    with open(meta_dir / "info.json", "w") as f:
        json.dump(meta, f, indent=2)

    print(f"[to_lerobot] Mock dataset written to {output_dir}")
    print(f"  Episodes: {meta['n_episodes']}, Frames: {meta['total_frames']}")
    print(f"  Parquet: {parquet_path}")
    print(f"  Meta: {meta_dir / 'info.json'}")


# ------------------------------------------------------------------
# Smoke test
# ------------------------------------------------------------------
if __name__ == "__main__":
    import tempfile

    from canonical_dataset import EpisodeBuilder

    print("=" * 60)
    print("to_leobot Converter Smoke Test")
    print("=" * 60)

    # Build a tiny dataset
    episodes = []
    for ep_idx in range(3):
        builder = EpisodeBuilder(
            task="push the red cube",
            robot_type="pushcube_2d",
            control_frequency=20.0,
            action_type="ee_delta",
        )
        for t in range(10):
            img = np.random.randint(0, 255, (128, 128, 3), dtype=np.uint8)
            state = np.random.randn(14).astype(np.float32)
            action = np.random.randn(2).astype(np.float32)
            builder.add_step(
                observation={"images": {"front": img}, "state": state},
                action=action,
                language="push the red cube",
                reward=-float(t),
                success=(t == 9),
            )
        episodes.append(builder.to_episode())

    # Convert with mock mode
    with tempfile.TemporaryDirectory() as tmpdir:
        convert_to_lerobot(episodes, tmpdir, dataset_name="smoke_test", mock=True)

    print("\n✓ to_lerobot smoke test passed")
