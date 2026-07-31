"""
tests/test_rfm_dataset_regression.py
====================================
Regression test for the tiny PushCube dataset committed to the repo.

Verifies that the committed ``pushcube_lerobot_tiny/`` dataset:
  1. Has valid ``meta/info.json`` with correct dimensions
  2. Each episode JSON loads and has consistent frame counts
  3. State dim == 14, action dim == 2
  4. Action type == "ee_delta_2d"
  5. Language instructions are non-empty
  6. At least one episode ends in success

This test runs in CI with **no extra dependencies** (only stdlib json/os).
It does NOT require lerobot, pyarrow, or GPU.

运行：python -m pytest tests/test_rfm_dataset_regression.py -v
"""

import json
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
TINY_DIR = (
    PROJECT_ROOT
    / "examples"
    / "robot_foundation_models"
    / "smolvla"
    / "datasets"
    / "pushcube_lerobot_tiny"
)


class TestTinyDatasetExists(unittest.TestCase):
    """The tiny dataset directory and its core files must exist."""

    def test_directory_exists(self):
        self.assertTrue(
            TINY_DIR.exists(),
            f"Tiny dataset directory not found: {TINY_DIR}",
        )

    def test_info_json_exists(self):
        info_path = TINY_DIR / "meta" / "info.json"
        self.assertTrue(info_path.exists(), "meta/info.json not found")

    def test_episode_files_exist(self):
        data_dir = TINY_DIR / "data"
        episodes = sorted(data_dir.glob("episode_*.json"))
        self.assertGreaterEqual(
            len(episodes), 2,
            "Expected at least 2 episode JSON files",
        )


class TestTinyDatasetMetadata(unittest.TestCase):
    """info.json must declare correct dimensions and format."""

    @classmethod
    def setUpClass(cls):
        info_path = TINY_DIR / "meta" / "info.json"
        with open(info_path, encoding="utf-8") as f:
            cls.info = json.load(f)

    def test_state_dim_is_14(self):
        self.assertEqual(self.info["state_dim"], 14)

    def test_action_dim_is_2(self):
        self.assertEqual(self.info["action_dim"], 2)

    def test_action_type_is_ee_delta_2d(self):
        self.assertEqual(self.info["action_type"], "ee_delta_2d")

    def test_fps_is_20(self):
        self.assertEqual(self.info["fps"], 20.0)

    def test_n_episodes_matches_files(self):
        data_dir = TINY_DIR / "data"
        n_files = len(list(data_dir.glob("episode_*.json")))
        self.assertEqual(self.info["n_episodes"], n_files)

    def test_total_frames_positive(self):
        self.assertGreater(self.info["total_frames"], 0)


class TestTinyDatasetEpisodes(unittest.TestCase):
    """Each episode must have consistent, well-formed frame data."""

    @classmethod
    def setUpClass(cls):
        data_dir = TINY_DIR / "data"
        cls.episodes = []
        for p in sorted(data_dir.glob("episode_*.json")):
            with open(p, encoding="utf-8") as f:
                cls.episodes.append(json.load(f))

    def test_at_least_two_episodes(self):
        self.assertGreaterEqual(len(self.episodes), 2)

    def test_episode_frame_counts_match_info(self):
        info_path = TINY_DIR / "meta" / "info.json"
        with open(info_path, encoding="utf-8") as f:
            info = json.load(f)
        total = sum(ep["n_frames"] for ep in self.episodes)
        self.assertEqual(total, info["total_frames"])

    def test_state_dim_per_frame(self):
        for ep in self.episodes:
            for frame in ep["frames"]:
                self.assertEqual(
                    len(frame["state"]), 14,
                    f"State dim != 14 in episode {ep['episode_index']}",
                )

    def test_action_dim_per_frame(self):
        for ep in self.episodes:
            for frame in ep["frames"]:
                self.assertEqual(
                    len(frame["action"]), 2,
                    f"Action dim != 2 in episode {ep['episode_index']}",
                )

    def test_language_non_empty(self):
        for ep in self.episodes:
            for frame in ep["frames"]:
                self.assertTrue(
                    frame["language"],
                    f"Empty language in episode {ep['episode_index']}",
                )

    def test_at_least_one_success(self):
        has_success = any(ep["success"] for ep in self.episodes)
        self.assertTrue(
            has_success,
            "At least one episode should end in success",
        )

    def test_action_type_in_episode(self):
        for ep in self.episodes:
            self.assertEqual(ep["action_type"], "ee_delta_2d")

    def test_timestamps_monotonic(self):
        for ep in self.episodes:
            timestamps = [f["timestamp"] for f in ep["frames"]]
            for i in range(1, len(timestamps)):
                self.assertGreater(
                    timestamps[i], timestamps[i - 1],
                    f"Timestamps not strictly increasing in episode {ep['episode_index']}",
                )


if __name__ == "__main__":
    unittest.main(verbosity=2)
