"""
Canonical → RLDS Dataset Converter
==================================
Convert ``CanonicalEpisode`` objects to `RLDS (Reinforcement Learning Datasets)`_
TFRecord format.

RLDS is the standard format used by OpenVLA, Octo, and many robotics
benchmarks (LIBERO, Bridge, etc.).  It stores episodes as TensorFlow
``tf.train.Example`` protos with the following structure::

    {
        "steps/observation/image": bytes (JPEG-encoded),
        "steps/observation/state": float32[],
        "steps/action": float32[],
        "steps/language_instruction": string,
        "steps/reward": float32,
        "steps/is_terminal": bool,
        "steps/is_last": bool,
        "steps/is_first": bool,
    }

.. _RLDS: https://github.com/google-research/rlds

Usage
-----
.. code-block:: python

    from canonical_dataset import load_episodes_from_dir
    from to_rlds import convert_to_rlds

    episodes = load_episodes_from_dir("datasets/pushcube_canonical/")
    convert_to_rlds(
        episodes,
        output_dir="datasets/pushcube_rlds/",
        dataset_name="pushcube_dual_cube",
    )

Dependencies
------------
- ``pip install tensorflow rlds`` (for real conversion)
- This module has a ``mock`` mode that writes NumPy NPZ files when TF is
  unavailable, plus a conversion guide.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

import numpy as np

from canonical_dataset import CanonicalEpisode


def convert_to_rlds(
    episodes: List[CanonicalEpisode],
    output_dir: str | Path,
    dataset_name: str = "canonical_dataset",
    mock: bool = False,
):
    """Convert canonical episodes to RLDS TFRecord format.

    Parameters
    ----------
    episodes : list[CanonicalEpisode]
    output_dir : str or Path
        Root directory for the RLDS dataset.
    dataset_name : str
        Human-readable dataset identifier.
    mock : bool
        If True (or TensorFlow is unavailable), write NumPy NPZ files
        instead of TFRecords, plus a README explaining the conversion.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not mock:
        try:
            _convert_with_tf(episodes, output_dir, dataset_name)
            print(f"[RLDS] Dataset written to {output_dir}")
            return
        except ImportError:
            print("[RLDS] tensorflow not installed — falling back to NPZ writer.")

    _convert_with_npz(episodes, output_dir, dataset_name)


def _convert_with_tf(
    episodes: List[CanonicalEpisode],
    output_dir: Path,
    dataset_name: str,
):
    """Real RLDS conversion (requires ``pip install tensorflow rlds``)."""
    import tensorflow as tf

    # RLDS feature specification
    def _bytes_feature(value):
        return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

    def _float_feature(value):
        return tf.train.Feature(float_list=tf.train.FloatList(value=value))

    def _int64_feature(value):
        return tf.train.Feature(int64_list=tf.train.Int64List(value=value))

    data_dir = output_dir / "train"
    data_dir.mkdir(parents=True, exist_ok=True)

    writer = tf.io.TFRecordWriter(str(data_dir / "episode.tfrecord"))

    for ep in episodes:
        cam_name = ep.camera_names[0] if ep.camera_names else "front"
        for t in range(ep.length):
            # Encode image as JPEG bytes
            img = ep.observation["images"][cam_name][t]
            img_bytes = tf.io.encode_jpeg(img).numpy()

            # Build feature dict
            features = {
                "steps/observation/image": _bytes_feature(img_bytes),
                "steps/action": _float_feature(ep.action[t].tolist()),
                "steps/language_instruction": _bytes_feature(ep.language[t].encode("utf-8")),
                "steps/reward": _float_feature([ep.reward[t]]),
                "steps/is_terminal": _int64_feature([1 if ep.success[t] else 0]),
                "steps/is_last": _int64_feature([1 if t == ep.length - 1 else 0]),
                "steps/is_first": _int64_feature([1 if t == 0 else 0]),
            }

            if ep.state_dim > 0:
                features["steps/observation/state"] = _float_feature(
                    ep.observation["state"][t].tolist()
                )

            example = tf.train.Example(
                features=tf.train.Features(feature=features)
            )
            writer.write(example.SerializeToString())

    writer.close()

    # Write metadata
    meta = {
        "dataset_name": dataset_name,
        "n_episodes": len(episodes),
        "total_frames": sum(ep.length for ep in episodes),
        "action_dim": episodes[0].action_dim if episodes else 0,
        "state_dim": episodes[0].state_dim if episodes else 0,
        "camera_names": episodes[0].camera_names if episodes else [],
    }
    with open(output_dir / "dataset_info.json", "w") as f:
        json.dump(meta, f, indent=2)


def _convert_with_npz(
    episodes: List[CanonicalEpisode],
    output_dir: Path,
    dataset_name: str,
):
    """Fallback: write NPZ files + README when TensorFlow is unavailable."""
    data_dir = output_dir / "train"
    data_dir.mkdir(parents=True, exist_ok=True)

    for ep_idx, ep in enumerate(episodes):
        cam_name = ep.camera_names[0] if ep.camera_names else "front"
        images = np.stack(ep.observation["images"][cam_name])
        actions = np.stack(ep.action)
        rewards = np.array(ep.reward, dtype=np.float32)
        successes = np.array(ep.success, dtype=bool)
        languages = np.array(ep.language)
        timestamps = np.array(ep.timestamps, dtype=np.float32)

        save_dict = {
            "images": images,
            "actions": actions,
            "rewards": rewards,
            "successes": successes,
            "languages": languages,
            "timestamps": timestamps,
        }

        if ep.state_dim > 0:
            save_dict["states"] = np.stack(ep.observation["state"])

        np.savez_compressed(data_dir / f"episode_{ep_idx:04d}.npz", **save_dict)

    # Metadata
    meta = {
        "dataset_name": dataset_name,
        "n_episodes": len(episodes),
        "total_frames": sum(ep.length for ep in episodes),
        "action_dim": episodes[0].action_dim if episodes else 0,
        "state_dim": episodes[0].state_dim if episodes else 0,
        "camera_names": episodes[0].camera_names if episodes else [],
        "format": "npz (numpy-compressed)",
        "note": "Install tensorflow+rlds for true TFRecord conversion",
    }
    with open(output_dir / "dataset_info.json", "w") as f:
        json.dump(meta, f, indent=2)

    # README
    readme = output_dir / "README.md"
    readme.write_text(
        """# RLDS Dataset (Mock Format)

This directory contains episodes in NumPy NPZ format as a fallback when
TensorFlow / RLDS is not installed.

## To convert to real RLDS TFRecords

```bash
pip install tensorflow rlds
python -c "
from to_rlds import convert_to_rlds
from canonical_dataset import load_episodes_from_dir
episodes = load_episodes_from_dir('canonical/')
convert_to_rlds(episodes, 'rlds_output/', mock=False)
"
```

## NPZ Structure

Each ``episode_XXXX.npz`` contains:
- ``images``: (T, H, W, 3) uint8
- ``actions``: (T, action_dim) float32
- ``rewards``: (T,) float32
- ``successes``: (T,) bool
- ``languages``: (T,) str
- ``timestamps``: (T,) float32
- ``states``: (T, state_dim) float32 (optional)
"""
    )

    print(f"[RLDS] Mock dataset written to {output_dir}")
    print(f"  Episodes: {meta['n_episodes']}, Frames: {meta['total_frames']}")
    print(f"  Format: NPZ (install tensorflow for TFRecord output)")


# ------------------------------------------------------------------
# Smoke test
# ------------------------------------------------------------------
if __name__ == "__main__":
    import tempfile

    from canonical_dataset import EpisodeBuilder

    print("=" * 60)
    print("to_rlds Converter Smoke Test")
    print("=" * 60)

    episodes = []
    for ep_idx in range(2):
        builder = EpisodeBuilder(
            task="push the red cube",
            robot_type="pushcube_2d",
            control_frequency=20.0,
            action_type="ee_delta",
        )
        for t in range(5):
            img = np.random.randint(0, 255, (128, 128, 3), dtype=np.uint8)
            state = np.random.randn(14).astype(np.float32)
            action = np.random.randn(2).astype(np.float32)
            builder.add_step(
                observation={"images": {"front": img}, "state": state},
                action=action,
                language="push the red cube",
                reward=-float(t),
                success=(t == 4),
            )
        episodes.append(builder.to_episode())

    with tempfile.TemporaryDirectory() as tmpdir:
        convert_to_rlds(episodes, tmpdir, dataset_name="smoke_test", mock=True)

    print("\n✓ to_rlds smoke test passed")
