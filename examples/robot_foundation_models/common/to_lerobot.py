"""
Canonical → LeRobot Dataset Converter
=====================================
Convert ``CanonicalEpisode`` objects using the pinned LeRobot v0.4.1 API.
The official path still requires a real loader round trip before training;
the explicit mock path writes teaching Parquet, not a compatible dataset.

LeRobot uses `HuggingFace datasets`_ under the hood with a specific
Parquet schema.  This converter writes:

1. ``data/`` — Parquet files with all timesteps
2. ``meta/`` — metadata and episode/task tables (official mode)

.. _LeRobot dataset format: https://github.com/huggingface/lerobot/blob/v0.4.1/src/lerobot/datasets/lerobot_dataset.py
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

After validating an official-loader round trip, a candidate training command is::

    lerobot-train \
        --policy.path=lerobot/smolvla_base \
        --dataset.repo_id=local/pushcube_dual_cube \
        --dataset.root=datasets/pushcube_lerobot/ \
        --policy.push_to_hub=false \
        --output_dir=outputs/train/new_smolvla_run

Dependencies
------------
- ``pip install 'lerobot[smolvla]==0.4.1'`` (official conversion/training API)
- Explicit ``mock=True`` writes teaching Parquet through ``pyarrow`` and
  ``pandas``. It does not implement the official LeRobot v3 schema and must
  not be supplied to ``lerobot-train``. Missing dependencies are errors.
- Existing output paths are never removed or overwritten in either mode.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

if __package__:
    from .canonical_dataset import CanonicalEpisode, compute_action_statistics
else:  # Preserve the direct-script smoke-test entry point.
    from canonical_dataset import CanonicalEpisode, compute_action_statistics


def convert_to_lerobot(
    episodes: List[CanonicalEpisode],
    output_dir: str | Path,
    dataset_name: str = "canonical_dataset",
    fps: Optional[float] = None,
    mock: bool = False,
):
    """Convert canonical episodes without replacing existing files.

    Parameters
    ----------
    episodes : list[CanonicalEpisode]
    output_dir : str or Path
        Root directory for the LeRobot dataset.
    dataset_name : str
        Human-readable dataset identifier.
    fps : float or None
        Positive integer frame rate. If None, uses the first episode's
        control_frequency. Align/resample source episodes explicitly first.
    mock : bool
        Only explicit True selects a teaching Parquet writer. It does not
        create an official LeRobot-compatible dataset. Missing LeRobot in
        official mode raises ImportError; there is no implicit fallback.
    """
    if not episodes or any(ep.length == 0 for ep in episodes):
        raise ValueError("Conversion requires at least one nonempty episode; empty episodes are unsupported.")
    output_dir = Path(output_dir).expanduser()
    if output_dir.exists() or output_dir.is_symlink():
        raise FileExistsError(
            f"Output path '{output_dir}' already exists. Choose a new path; "
            "the converter does not delete or overwrite prior output."
        )

    if fps is None:
        fps = episodes[0].control_frequency
    if not np.isfinite(fps) or fps <= 0 or int(fps) != fps:
        raise ValueError("The pinned converter requires a finite positive integer fps; resample explicitly first.")

    # ------------------------------------------------------------------
    # Official mode is explicit and never silently becomes a mock dataset.
    # ------------------------------------------------------------------
    if not mock:
        try:
            _convert_with_lerobot(episodes, output_dir, dataset_name, fps)
            return
        except ImportError as error:
            raise ImportError(
                "Official conversion requires LeRobot v0.4.1 and its dependencies. "
                "No mock fallback was written. Use mock=True only for an explicitly "
                "selected teaching serialization, not for official training."
            ) from error

    # ------------------------------------------------------------------
    # Explicit teaching serialization; the writer checks its own dependencies.
    # ------------------------------------------------------------------
    _convert_with_pyarrow(episodes, output_dir, dataset_name, fps)


def _convert_with_lerobot(
    episodes: List[CanonicalEpisode],
    output_dir: Path,
    dataset_name: str,
    fps: float,
):
    """Official API conversion (requires pinned LeRobot v0.4.1).

    Uses LeRobot 0.4.1 API: ``LeRobotDataset.create()``, ``add_frame()``,
    ``save_episode()`` and ``finalize()``. Images are stored as image features
    (``use_videos=False``) with ``dtype="image"``.
    """
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    # Infer feature shapes from first episode
    ep0 = episodes[0]
    cam_names = ep0.camera_names

    # Build feature dictionary — LeRobot 0.4.1 requires "image" dtype
    # for per-frame images (not "uint8").
    features = {}
    for cam in cam_names:
        h, w, c = ep0.observation["images"][cam][0].shape
        features[f"observation.images.{cam}"] = {
            "shape": (c, h, w),
            "dtype": "image",
            "names": None,
        }

    if ep0.state_dim > 0:
        features["observation.state"] = {
            "shape": (ep0.state_dim,),
            "dtype": "float32",
            "names": None,
        }

    features["action"] = {
        "shape": (ep0.action_dim,),
        "dtype": "float32",
        "names": None,
    }

    # Create dataset (root must not exist yet — LeRobotDataset.create
    # calls mkdir with exist_ok=False)
    dataset = LeRobotDataset.create(
        repo_id=dataset_name,
        root=output_dir,
        fps=int(fps),
        features=features,
        use_videos=False,  # store frames as individual images
    )

    # Add episodes — each frame MUST include a "task" key (string) which
    # add_frame() pops and stores.  Images are passed as (H, W, C) uint8
    # arrays; image_array_to_pil_image() handles the conversion.
    # This wrapper lets add_frame() compute a uniform frame_index/fps timeline;
    # original timestamps are not preserved. Align/resample inputs first.
    try:
        for ep_idx, ep in enumerate(episodes):
            for t in range(ep.length):
                frame = {
                    "action": ep.action[t],
                    "task": ep.language[t] if ep.language[t] else ep.task,
                }
                for cam in cam_names:
                    img = ep.observation["images"][cam][t]
                    # Pass (H, W, C) directly — write_image handles conversion
                    frame[f"observation.images.{cam}"] = img
                if ep.state_dim > 0:
                    frame["observation.state"] = ep.observation["state"][t]

                dataset.add_frame(frame)

            dataset.save_episode()
            if (ep_idx + 1) % 10 == 0:
                print(f"  Saved episode {ep_idx + 1}/{len(episodes)}")
    except BaseException as write_error:
        # Retain the primary write failure even if closing the partial output
        # also fails. Python 3.11+ can attach that secondary failure as a note.
        try:
            dataset.finalize()
        except BaseException as finalize_error:
            if hasattr(write_error, "add_note"):
                write_error.add_note(f"Dataset finalization also failed: {finalize_error!r}")
        raise
    else:
        # v0.4.1 requires this to close data AND metadata parquet writers.
        # A close failure after successful writes must propagate as a failure.
        dataset.finalize()

    print(f"[LeRobot] Dataset written to {output_dir}")


def _convert_with_pyarrow(
    episodes: List[CanonicalEpisode],
    output_dir: Path,
    dataset_name: str,
    fps: float,
):
    """Teaching serialization; NOT the official LeRobot v3 dataset schema."""
    try:
        import pandas as pd
        import pyarrow as pa
        import pyarrow.parquet as pq
    except ImportError as error:
        raise ImportError(
            "Explicit mock serialization requires pandas and pyarrow. "
            "Install them or select official conversion with LeRobot installed."
        ) from error

    output_dir.mkdir(parents=True, exist_ok=False)
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
        "format": "embodied-ai-teaching-parquet/v1",
        "official_lerobot_compatible": False,
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

    print(f"[to_lerobot] Teaching mock written to {output_dir}; not an official LeRobot dataset")
    print(f"  Episodes: {meta['n_episodes']}, Frames: {meta['total_frames']}")
    print(f"  Parquet: {parquet_path}")
    print(f"  Meta: {meta_dir / 'info.json'}")


# ------------------------------------------------------------------
# Smoke test
# ------------------------------------------------------------------
if __name__ == "__main__":
    import tempfile

    if __package__:
        from .canonical_dataset import EpisodeBuilder
    else:
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
        convert_to_lerobot(episodes, Path(tmpdir) / "mock_dataset", dataset_name="smoke_test", mock=True)

    print("\n✓ to_lerobot smoke test passed")
