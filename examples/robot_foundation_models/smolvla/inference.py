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
1. Converting ``RobotObservation`` -> LeRobot input dict, including an
   explicit camera-key mapping between the *canonical* observation keys
   (``"front"``, ``"wrist_left"`` ...) and the LeRobot/SmolVLA config
   keys (``"observation.images.front"`` ...).
2. Calling ``SmolVLAPolicy.select_action`` and turning its single-step
   output into a proper ``(chunk_size, action_dim)`` ``ActionChunk`` by
   draining the policy's internal action queue. If the policy exposes no
   queue, the adapter falls back to ``chunk_size=1``.
3. Converting the output -> ``ActionChunk``.

Action chunking
---------------
LeRobot policies (SmolVLA, ACT, Diffusion Policy) follow a common
convention: ``select_action`` returns a SINGLE action step that it pops
from an internal ``_action_queue``; the queue is refilled by a full
``forward`` pass whenever it runs empty. To honor the
``(chunk_size, action_dim)`` contract of ``ActionChunk``, this adapter
drains the remaining queue after each ``select_action`` call and
prepends the just-returned action.

- If the policy exposes an action queue, the returned chunk has up to
  ``chunk_size`` steps (fewer if the queue is exhausted first).
- If the policy exposes no queue (or returns a single action with no
  way to recover the rest), the adapter returns a chunk with
  ``chunk_size=1`` and documents this limitation.

Usage
-----
.. code-block:: python

    from examples.robot_foundation_models.smolvla.inference import SmolVLAAdapter
    from examples.robot_foundation_models.common import RobotObservation

    model = SmolVLAAdapter(device="cuda")       # real SmolVLA, chunk_size=10
    model = SmolVLAAdapter(mock=True)           # mock, chunk_size=1
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
the interface without downloading the 450M model. In mock mode the
default ``chunk_size`` is 1 (representing a single-action policy); pass
``chunk_size=10`` explicitly to exercise multi-step chunking in tests.
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
        Desired action-chunk length returned by :meth:`predict_action`.
        If ``None``, the default depends on the mode:

        - ``mock=True`` (or real load failure) -> ``1``
        - real SmolVLA -> the model config's ``chunk_size`` (typically 10)

        The returned chunk may be shorter than ``chunk_size`` if the
        policy's internal action queue is exhausted first.
    mock : bool
        If True (or if lerobot is unavailable), run in mock mode.
    """

    #: Mapping from LeRobot/SmolVLA config camera keys to canonical
    #: ``RobotObservation.images`` keys. Used to translate between the two
    #: naming conventions so the policy is fed images under the keys its
    #: config expects.
    CAMERA_MAPPING: Dict[str, str] = {
        "observation.images.front": "front",
        "observation.images.left_wrist": "wrist_left",
        "observation.images.right_wrist": "wrist_right",
        "observation.images.wrist": "wrist_left",  # generic wrist fallback
    }

    def __init__(
        self,
        device: str = "cpu",
        pretrained_name_or_path: str = "lerobot/smolvla_base",
        action_type: str = "ee_delta_2d",
        action_dim: int = 2,
        control_frequency: float = 20.0,
        chunk_size: Optional[int] = None,
        mock: bool = False,
    ):
        self.device = device
        self.pretrained_name_or_path = pretrained_name_or_path
        self.action_type = action_type
        self.action_dim = action_dim  # PushCube: 2-D [dx, dy]
        self.control_frequency = control_frequency
        self._chunk_size = chunk_size
        self._mock = mock
        self._policy = None
        self._torch = None
        self._action_queue: deque = deque()
        self._step = 0

        if not mock:
            self._try_load_model()
        else:
            # Mock emulates a single-action policy -> chunk_size defaults
            # to 1. Pass chunk_size explicitly to test multi-step chunks.
            if self._chunk_size is None:
                self._chunk_size = 1

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

            # Determine chunk size from model config (SmolVLA default ~ 10)
            if self._chunk_size is None:
                cfg = getattr(self._policy.config, "chunk_size", 10)
                self._chunk_size = cfg
            print(f"[SmolVLA] Loaded. chunk_size={self._chunk_size}")
        except ImportError:
            print("[SmolVLA] lerobot not installed -- running in mock mode.")
            self._mock = True
            self._torch = None
            if self._chunk_size is None:
                self._chunk_size = 1
        except Exception as e:
            print(f"[SmolVLA] Failed to load model: {e}")
            print("[SmolVLA] Running in mock mode.")
            self._mock = True
            self._torch = None
            if self._chunk_size is None:
                self._chunk_size = 1

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
    def _build_lerobot_obs(self, obs: RobotObservation) -> Dict[str, Any]:
        """Convert a canonical ``RobotObservation`` to a LeRobot obs dict.

        Uses :attr:`CAMERA_MAPPING` to translate canonical camera keys
        (``"front"``, ``"wrist_left"`` ...) into the LeRobot config keys
        (``"observation.images.front"`` ...). Raises a clear error if no
        cameras can be mapped.
        """
        torch = self._torch
        policy_config = self._policy.config
        image_features = getattr(policy_config, "image_features", {}) or {}
        state_feature = getattr(policy_config, "state_feature", None)

        lerobot_obs: Dict[str, Any] = {}

        # ---- Images (with explicit camera key mapping) ----
        mapped_cameras: list = []
        for cam_name in image_features:
            canonical_key = self.CAMERA_MAPPING.get(cam_name)
            if canonical_key is None:
                # Fall back to a direct key match (covers configs that
                # already use canonical names).
                canonical_key = cam_name
            if canonical_key in obs.images:
                img = obs.images[canonical_key]
                if img.dtype == np.uint8:
                    img = img.astype(np.float32) / 255.0
                # LeRobot expects (C, H, W)
                img_tensor = torch.from_numpy(
                    np.ascontiguousarray(img)
                ).permute(2, 0, 1).float()
                lerobot_obs[cam_name] = img_tensor.unsqueeze(0).to(self.device)
                mapped_cameras.append(cam_name)

        if not mapped_cameras:
            available = list(obs.images.keys())
            if isinstance(image_features, dict):
                expected = list(image_features.keys())
            else:
                expected = list(image_features)
            raise RuntimeError(
                "SmolVLA could not map any cameras from the canonical "
                "observation to the model config. "
                f"Model expects image features: {expected}. "
                f"Observation has cameras: {available}. "
                f"Translation table (CAMERA_MAPPING): {self.CAMERA_MAPPING}. "
                "Either rename the observation image keys to match the "
                "canonical names (front, wrist_left, wrist_right) or extend "
                "SmolVLAAdapter.CAMERA_MAPPING."
            )

        # ---- State ----
        if state_feature is not None and obs.state is not None:
            state_key = (
                state_feature if isinstance(state_feature, str)
                else "observation.state"
            )
            lerobot_obs[state_key] = (
                torch.from_numpy(np.ascontiguousarray(obs.state))
                .float()
                .unsqueeze(0)
                .to(self.device)
            )

        # ---- Language ----
        task_key = getattr(policy_config, "language_feature", "task")
        lerobot_obs[task_key] = [obs.language_instruction]

        return lerobot_obs

    def _real_predict(self, obs: RobotObservation) -> ActionChunk:
        """Call the actual SmolVLA model."""
        torch = self._torch
        lerobot_obs = self._build_lerobot_obs(obs)

        with torch.no_grad():
            action_tensor = self._policy.select_action(lerobot_obs)

        actions = self._select_action_to_chunk(action_tensor)

        self._step += 1
        return ActionChunk(
            actions=actions,
            action_type=self.action_type,
            control_frequency=self.control_frequency,
        )

    # ------------------------------------------------------------------
    # Action-chunk reconstruction
    # ------------------------------------------------------------------
    @staticmethod
    def _to_numpy(x) -> np.ndarray:
        """Best-effort conversion of a torch/numpy scalar to ``np.ndarray``."""
        if hasattr(x, "detach"):
            return x.detach().cpu().numpy()
        return np.asarray(x)

    def _get_policy_action_queue(self):
        """Return the policy's internal action queue, or ``None``.

        LeRobot policies (SmolVLA, ACT, DiffusionPolicy) store pending
        action steps in ``_action_queue`` (a ``deque`` in recent versions,
        a ``list`` in older ones).
        """
        for attr in ("_action_queue", "action_queue"):
            queue = getattr(self._policy, attr, None)
            if queue is not None:
                return queue
        return None

    @staticmethod
    def _pop_from_queue(queue):
        """Pop one item from a deque/list queue, or ``None`` if empty."""
        if hasattr(queue, "popleft"):
            try:
                return queue.popleft()
            except IndexError:
                return None
        if hasattr(queue, "pop"):
            try:
                return queue.pop(0)
            except IndexError:
                return None
        return None

    def _select_action_to_chunk(self, action_tensor) -> np.ndarray:
        """Turn a ``select_action`` return value into ``(chunk_size, action_dim)``.

        ``select_action`` returns a SINGLE action step (popped from the
        policy's internal queue). To honor the ``(chunk_size, action_dim)``
        contract we drain the remaining queue and prepend the returned
        action.

        - If the policy exposes an action queue, the chunk has up to
          ``self._chunk_size`` steps.
        - If the policy exposes no queue (or it is empty), we treat it as
          a single-action policy and return ``(1, action_dim)`` -- i.e.
          ``chunk_size`` is effectively 1 for that call.
        """
        first_np = self._to_numpy(action_tensor)
        if first_np.ndim == 1:
            first_np = first_np.reshape(1, -1)
        first_np = first_np[:1]  # keep only the first row if 2-D

        # Fast path: single-step chunking requested
        if self._chunk_size is not None and self._chunk_size <= 1:
            return first_np.astype(np.float32)

        queue = self._get_policy_action_queue()
        if queue is None or len(queue) == 0:
            # No queue available -> single-action policy -> chunk_size = 1
            return first_np.astype(np.float32)

        collected = [first_np[0]]
        n_remaining = max(0, (self._chunk_size or 1) - 1)
        while n_remaining > 0 and len(queue) > 0:
            queued = self._pop_from_queue(queue)
            if queued is None:
                break
            q_np = self._to_numpy(queued)
            if q_np.ndim == 1:
                q_np = q_np.reshape(1, -1)
            collected.append(q_np[0])
            n_remaining -= 1

        return np.stack(collected, axis=0).astype(np.float32)

    # ------------------------------------------------------------------
    # Mock inference (for CI / CPU-only environments)
    # ------------------------------------------------------------------
    def _mock_predict(self, obs: RobotObservation) -> ActionChunk:
        """Generate zero actions with correct shape for interface testing.

        The mock emulates a policy that returns a single action per call,
        so its default ``chunk_size`` is 1. Pass ``chunk_size`` to the
        constructor to produce longer zero-filled chunks for testing.

        Action dimension is always ``self.action_dim`` (2 for PushCube),
        independent of the observation state dimension.
        """
        action_dim = self.action_dim  # PushCube: 2-D [dx, dy]

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
        state=np.zeros(2, dtype=np.float32),  # PushCube: 2-D state
        language_instruction="push the red cube to the target",
        timestamp=time.time(),
    )

    print(f"\nObservation: {obs}")

    chunk = adapter.predict_action(obs)
    print(f"Action chunk: {chunk}")
    print(f"  actions shape: {chunk.actions.shape}")
    print(f"  first action: {chunk.first_action()}")

    # Verify the chunk honors chunk_size and is 2-D
    assert chunk.actions.shape[0] == adapter._chunk_size, (
        f"chunk_size mismatch: {chunk.actions.shape[0]} != {adapter._chunk_size}"
    )
    assert chunk.actions.ndim == 2, f"actions must be 2-D, got {chunk.actions.shape}"
    # PushCube action_dim = 2
    assert chunk.actions.shape[1] == 2, (
        f"action_dim should be 2 for PushCube, got {chunk.actions.shape[1]}"
    )

    # Multi-step mock chunking (explicit chunk_size)
    adapter_multi = SmolVLAAdapter(mock=True, chunk_size=10)
    adapter_multi.reset()
    chunk_multi = adapter_multi.predict_action(obs)
    print(f"\nMulti-step mock (chunk_size=10): {chunk_multi}")
    print(f"  actions shape: {chunk_multi.actions.shape}")
    assert chunk_multi.actions.shape == (10, 2), chunk_multi.actions.shape

    print("\nSmolVLA adapter smoke test passed (mock mode)")
