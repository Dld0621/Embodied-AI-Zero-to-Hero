"""
SmolVLA Inference Adapter
=========================
Wraps the HuggingFace LeRobot SmolVLA model to conform to the
``RobotFoundationModel`` protocol.

SmolVLA is a 450M-parameter lightweight VLA model that takes:
- Multi-camera RGB images
- Robot proprioceptive state
- Language instruction

and outputs a chunk of continuous actions.

This adapter handles:
1. Converting ``RobotObservation`` → LeRobot input dict
2. Calling ``SmolVLAPolicy.select_action``
3. Converting the output → ``ActionChunk``

Usage
-----
.. code-block:: python

    from examples.robot_foundation_models.smolvla.inference import SmolVLAAdapter
    from examples.robot_foundation_models.common import RobotObservation

    model = SmolVLAAdapter(device="cuda")
    model.reset()

    obs = RobotObservation(
        images={"front": front_img, "wrist_left": wrist_img},
        state=robot_state,
        language_instruction="pick up the red block",
        timestamp=0.0,
    )
    chunk = model.predict_action(obs)
    # chunk.actions: (chunk_size, action_dim) numpy array

Fallback
--------
If ``lerobot`` is not installed, the adapter runs in ``mock`` mode,
outputting zero actions with the correct shape.  This allows CI to test
the interface without downloading the 450M model.
"""

from __future__ import annotations

import sys
import os
import time
from typing import Optional, Dict, Any
from collections import deque

import numpy as np

# Add project root to path for common imports
_PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "..")
)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from examples.robot_foundation_models.common.observation_schema import RobotObservation
from examples.robot_foundation_models.common.action_schema import ActionChunk


class SmolVLAAdapter:
    """Adapts SmolVLA to the ``RobotFoundationModel`` protocol.

    Parameters
    ----------
    device : str
        Torch device ("cuda", "cpu").
    pretrained_name_or_path : str
        HuggingFace repo or local path.  Default: official SmolVLA checkpoint.
    action_type : str
        Output action type (SmolVLA outputs joint-position deltas by default).
    control_frequency : float
        Control frequency in Hz (SmolVLA default: 20 Hz).
    chunk_size : int or None
        Override the model's default action chunk size.
    mock : bool
        If True (or if lerobot is unavailable), run in mock mode.
    """

    def __init__(
        self,
        device: str = "cpu",
        pretrained_name_or_path: str = "lerobot/smolvla_base",
        action_type: str = "joint_delta",
        control_frequency: float = 20.0,
        chunk_size: Optional[int] = None,
        mock: bool = False,
    ):
        self.device = device
        self.pretrained_name_or_path = pretrained_name_or_path
        self.action_type = action_type
        self.control_frequency = control_frequency
        self._chunk_size = chunk_size
        self._mock = mock
        self._policy = None
        self._action_queue: deque = deque()
        self._step = 0

        if not mock:
            self._try_load_model()
        else:
            if self._chunk_size is None:
                self._chunk_size = 10

    def _try_load_model(self):
        """Attempt to load the SmolVLA policy from LeRobot."""
        try:
            import torch
            from lerobot.common.policies.smolvla.modeling_smolvla import SmolVLAPolicy

            print(f"[SmolVLA] Loading from {self.pretrained_name_or_path}...")
            self._policy = SmolVLAPolicy.from_pretrained(self.pretrained_name_or_path)
            self._policy = self._policy.to(self.device)
            self._policy.eval()
            self._torch = torch

            # Determine chunk size from model config
            if self._chunk_size is None:
                cfg = getattr(self._policy.config, "chunk_size", 10)
                self._chunk_size = cfg
            print(f"[SmolVLA] Loaded. chunk_size={self._chunk_size}")
        except ImportError:
            print("[SmolVLA] lerobot not installed — running in mock mode.")
            self._mock = True
            if self._chunk_size is None:
                self._chunk_size = 10
        except Exception as e:
            print(f"[SmolVLA] Failed to load model: {e}")
            print("[SmolVLA] Running in mock mode.")
            self._mock = True
            if self._chunk_size is None:
                self._chunk_size = 10

    # ------------------------------------------------------------------
    # RobotFoundationModel protocol
    # ------------------------------------------------------------------
    def reset(self) -> None:
        """Clear action queue and reset step counter."""
        self._action_queue.clear()
        self._step = 0
        if self._policy is not None and hasattr(self._policy, "reset"):
            self._policy.reset()

    def predict_action(self, observation: RobotObservation) -> ActionChunk:
        """Predict action chunk from canonical observation."""
        if self._mock or self._policy is None:
            return self._mock_predict(observation)

        return self._real_predict(observation)

    # ------------------------------------------------------------------
    # Real inference
    # ------------------------------------------------------------------
    def _real_predict(self, obs: RobotObservation) -> ActionChunk:
        """Call the actual SmolVLA model."""
        torch = self._torch

        # Build LeRobot-format observation dict
        policy_config = self._policy.config
        image_features = getattr(policy_config, "image_features", {})
        state_feature = getattr(policy_config, "state_feature", None)

        lerobot_obs: Dict[str, Any] = {}

        # Images
        for cam_name in image_features:
            if cam_name in obs.images:
                img = obs.images[cam_name]
                if img.dtype == np.uint8:
                    img = img.astype(np.float32) / 255.0
                # LeRobot expects (C, H, W)
                img_tensor = torch.from_numpy(img).permute(2, 0, 1).float()
                lerobot_obs[cam_name] = img_tensor.unsqueeze(0).to(self.device)

        # State
        if state_feature is not None and obs.state is not None:
            lerobot_obs[state_feature] = (
                torch.from_numpy(obs.state).float().unsqueeze(0).to(self.device)
            )

        # Language
        task_key = getattr(policy_config, "language_feature", "task")
        lerobot_obs[task_key] = [obs.language_instruction]

        with torch.no_grad():
            action_tensor = self._policy.select_action(lerobot_obs)

        actions = action_tensor.cpu().numpy()
        if actions.ndim == 1:
            actions = actions.reshape(1, -1)

        self._step += 1
        return ActionChunk(
            actions=actions,
            action_type=self.action_type,
            control_frequency=self.control_frequency,
        )

    # ------------------------------------------------------------------
    # Mock inference (for CI / CPU-only environments)
    # ------------------------------------------------------------------
    def _mock_predict(self, obs: RobotObservation) -> ActionChunk:
        """Generate zero actions with correct shape for interface testing."""
        action_dim = 7  # default: 7-DOF arm
        if obs.state is not None:
            action_dim = obs.state.shape[0]

        actions = np.zeros((self._chunk_size, action_dim), dtype=np.float32)
        self._step += 1
        return ActionChunk(
            actions=actions,
            action_type=self.action_type,
            control_frequency=self.control_frequency,
            confidence=0.0,
        )

    def __repr__(self) -> str:
        mode = "mock" if self._mock else "loaded"
        return f"SmolVLAAdapter(mode={mode}, chunk_size={self._chunk_size}, device={self.device})"


# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    print("=" * 60)
    print("SmolVLA Adapter Smoke Test")
    print("=" * 60)

    adapter = SmolVLAAdapter(mock=True)
    adapter.reset()

    # Create a fake observation
    fake_img = np.zeros((128, 128, 3), dtype=np.uint8)
    obs = RobotObservation(
        images={"front": fake_img},
        state=np.zeros(7, dtype=np.float32),
        language_instruction="push the red cube to the target",
        timestamp=time.time(),
    )

    print(f"\nObservation: {obs}")

    chunk = adapter.predict_action(obs)
    print(f"Action chunk: {chunk}")
    print(f"  actions shape: {chunk.actions.shape}")
    print(f"  first action: {chunk.first_action()}")

    print("\n✓ SmolVLA adapter smoke test passed (mock mode)")
