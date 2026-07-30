"""
Canonical Dataset Standard
==========================
Unified episode structure for all robot foundation models.

This module defines the **canonical internal format** used throughout the
repository.  Every data collector, converter, and model adapter works with
this format.  Converters (``to_lerobot.py``, ``to_rlds.py``) translate it
to framework-specific formats without touching model code.

Canonical Episode Schema
------------------------
.. code-block:: python

    {
        "task": "push the red cube to the target",
        "robot_type": "pushcube_2d",
        "control_frequency": 20,
        "action_type": "ee_delta",
        "timestamps": [0.0, 0.05, 0.10, ...],          # (T,)
        "observation": {
            "images": {
                "front": [(H, W, 3), ...],             # list of uint8 arrays
                "wrist_left": [...],
            },
            "state": [(state_dim,), ...],              # list of float32 arrays
        },
        "action": [(action_dim,), ...],                # list of float32 arrays
        "language": ["push the red cube...", ...],     # (T,) — repeated per frame
        "reward": [0.0, ...],                          # (T,)
        "success": [False, ..., True],                 # (T,)
        "metadata": {},                                # episode-level extras
    }

Design Decisions
----------------
1. **List-of-arrays over stacked array**: episodes may have variable length,
   and images are not easily stacked without padding.
2. **Per-frame language**: keeps the format uniform even when language is
   constant across the episode (just repeat the string).
3. **State is optional**: ``observation["state"]`` may be ``None`` for
   vision-only policies.
4. **Metadata dict**: stores environment-specific info (e.g., MuJoCo model
   name, random seed, object poses) without polluting the core schema.

Usage
-----
Collect data::

    from canonical_dataset import CanonicalEpisode, EpisodeBuilder
    builder = EpisodeBuilder(task="push the red cube", robot_type="pushcube_2d")
    builder.add_step(obs, action, language, reward, success)
    episode = builder.to_episode()

Save / load::

    episode.save("episode_000.pkl")
    ep = CanonicalEpisode.load("episode_000.pkl")

Convert::

    from to_lerobot import convert_to_lerobot
    convert_to_lerobot([episode], output_dir="lerobot_dataset/")
"""

from __future__ import annotations

import pickle
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np


# ------------------------------------------------------------------
# Episode dataclass
# ------------------------------------------------------------------

@dataclass
class CanonicalEpisode:
    """A single robot episode in canonical format.

    Attributes match the schema documented in the module docstring.
    All array-like fields are **lists** of ``np.ndarray`` to support
    variable-length episodes and heterogeneous image sizes.
    """

    task: str
    robot_type: str
    control_frequency: float
    action_type: str
    timestamps: List[float]
    observation: Dict[str, Optional[List[np.ndarray]]]
    action: List[np.ndarray]
    language: List[str]
    reward: List[float]
    success: List[bool]
    metadata: dict = field(default_factory=dict)

    def __post_init__(self):
        # Basic consistency check
        n = len(self.timestamps)
        assert len(self.action) == n, "action length mismatch"
        assert len(self.language) == n, "language length mismatch"
        assert len(self.reward) == n, "reward length mismatch"
        assert len(self.success) == n, "success length mismatch"
        if self.observation.get("state") is not None:
            assert len(self.observation["state"]) == n, "state length mismatch"
        if self.observation.get("images") is not None:
            for cam_name, imgs in self.observation["images"].items():
                assert len(imgs) == n, f"image list length mismatch for {cam_name}"

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------
    @property
    def length(self) -> int:
        return len(self.timestamps)

    @property
    def action_dim(self) -> int:
        return self.action[0].shape[0] if self.action else 0

    @property
    def state_dim(self) -> int:
        s = self.observation.get("state")
        return s[0].shape[0] if s is not None and len(s) > 0 else 0

    @property
    def camera_names(self) -> List[str]:
        imgs = self.observation.get("images", {})
        return sorted(imgs.keys())

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------
    def save(self, path: str | Path):
        """Serialize to pickle."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @staticmethod
    def load(path: str | Path) -> "CanonicalEpisode":
        """Load from pickle."""
        with open(path, "rb") as f:
            return pickle.load(f)

    def __repr__(self) -> str:
        cams = ", ".join(self.camera_names)
        return (
            f"CanonicalEpisode(task='{self.task[:30]}...', "
            f"robot={self.robot_type}, length={self.length}, "
            f"action_dim={self.action_dim}, cameras=[{cams}])"
        )


# ------------------------------------------------------------------
# Episode builder (incremental construction)
# ------------------------------------------------------------------

class EpisodeBuilder:
    """Incrementally build a ``CanonicalEpisode``.

    Example::

        builder = EpisodeBuilder(
            task="push the red cube",
            robot_type="pushcube_2d",
            control_frequency=20.0,
            action_type="ee_delta",
        )
        for t in range(T):
            builder.add_step(
                observation={"images": {"front": img}, "state": state_vec},
                action=action_vec,
                language="push the red cube",
                reward=-dist,
                success=(dist < 0.05),
                timestamp=t * 0.05,
            )
        episode = builder.to_episode()
    """

    def __init__(
        self,
        task: str,
        robot_type: str,
        control_frequency: float,
        action_type: str,
        metadata: Optional[dict] = None,
    ):
        self.task = task
        self.robot_type = robot_type
        self.control_frequency = control_frequency
        self.action_type = action_type
        self.metadata = metadata or {}

        self._timestamps: List[float] = []
        self._images: Dict[str, List[np.ndarray]] = {}
        self._state: List[Optional[np.ndarray]] = []
        self._action: List[np.ndarray] = []
        self._language: List[str] = []
        self._reward: List[float] = []
        self._success: List[bool] = []

    def add_step(
        self,
        observation: Dict[str, Optional[Dict[str, np.ndarray] | np.ndarray]],
        action: np.ndarray,
        language: str,
        reward: float = 0.0,
        success: bool = False,
        timestamp: Optional[float] = None,
    ):
        """Add one timestep.

        Parameters
        ----------
        observation : dict
            Must have keys ``"images"`` (dict[str, np.ndarray]) and/or
            ``"state"`` (np.ndarray or None).
        """
        if timestamp is None:
            timestamp = len(self._timestamps) / self.control_frequency

        self._timestamps.append(timestamp)
        self._action.append(np.asarray(action, dtype=np.float32))
        self._language.append(language)
        self._reward.append(float(reward))
        self._success.append(bool(success))

        # Images
        imgs = observation.get("images", {})
        for cam_name, img in imgs.items():
            if cam_name not in self._images:
                self._images[cam_name] = []
            self._images[cam_name].append(np.asarray(img, dtype=np.uint8))

        # State
        state = observation.get("state")
        if state is not None:
            self._state.append(np.asarray(state, dtype=np.float32))
        else:
            self._state.append(None)

    def to_episode(self) -> CanonicalEpisode:
        """Finalize and return the episode."""
        # Filter out None states
        has_state = any(s is not None for s in self._state)
        obs = {
            "images": {k: v for k, v in self._images.items()},
            "state": [s for s in self._state if s is not None] if has_state else None,
        }
        return CanonicalEpisode(
            task=self.task,
            robot_type=self.robot_type,
            control_frequency=self.control_frequency,
            action_type=self.action_type,
            timestamps=self._timestamps,
            observation=obs,
            action=self._action,
            language=self._language,
            reward=self._reward,
            success=self._success,
            metadata=self.metadata,
        )

    def __len__(self) -> int:
        return len(self._timestamps)


# ------------------------------------------------------------------
# Dataset helpers
# ------------------------------------------------------------------

def load_episodes_from_dir(directory: str | Path) -> List[CanonicalEpisode]:
    """Load all ``*.pkl`` episodes from a directory."""
    directory = Path(directory)
    episodes = []
    for p in sorted(directory.glob("*.pkl")):
        episodes.append(CanonicalEpisode.load(p))
    return episodes


def compute_action_statistics(episodes: List[CanonicalEpisode]) -> Dict[str, np.ndarray]:
    """Compute mean / std / min / max over all actions in a dataset.

    Useful for action normalization before feeding to a policy.
    """
    all_actions = np.concatenate([np.stack(ep.action) for ep in episodes], axis=0)
    return {
        "mean": np.mean(all_actions, axis=0),
        "std": np.std(all_actions, axis=0) + 1e-8,
        "min": np.min(all_actions, axis=0),
        "max": np.max(all_actions, axis=0),
    }


# ------------------------------------------------------------------
# Smoke test
# ------------------------------------------------------------------
if __name__ == "__main__":
    print("=" * 60)
    print("Canonical Dataset Smoke Test")
    print("=" * 60)

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

    ep = builder.to_episode()
    print(f"\nCreated: {ep}")
    print(f"  length={ep.length}, action_dim={ep.action_dim}, state_dim={ep.state_dim}")

    # Save / load round-trip
    ep.save("/tmp/canonical_episode_smoke.pkl")
    ep2 = CanonicalEpisode.load("/tmp/canonical_episode_smoke.pkl")
    print(f"  Round-trip OK: length={ep2.length}")

    stats = compute_action_statistics([ep])
    print(f"  Action stats keys: {list(stats.keys())}")

    print("\n✓ Canonical dataset smoke test passed")
