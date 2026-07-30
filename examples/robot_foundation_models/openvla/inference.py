"""
OpenVLA Inference Adapter (Stub)
================================
Wraps the OpenVLA-7B model to conform to the ``RobotFoundationModel``
protocol.

OpenVLA is a 7B-parameter generalist VLA model fine-tuned from Prismatic-7B.
It supports:
- Single RGB image input
- Language instruction
- LoRA fine-tuning
- RLDS dataset format
- LIBERO / BridgeData V2 evaluation

Repository: https://github.com/openvla/openvla

Status: 🟡 Adapter — interface defined, model loading requires GPU + 7B download.
CI tests the interface with mock mode only.

Usage
-----
.. code-block:: python

    from examples.robot_foundation_models.openvla.inference import OpenVLAAdapter

    model = OpenVLAAdapter(device="cuda", mock=False)
    model.reset()
    chunk = model.predict_action(obs)
"""

from __future__ import annotations

import os
import sys
import time
from typing import Optional

import numpy as np

_PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "..")
)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from examples.robot_foundation_models.common.observation_schema import RobotObservation
from examples.robot_foundation_models.common.action_schema import ActionChunk


class OpenVLAAdapter:
    """Adapts OpenVLA-7B to the ``RobotFoundationModel`` protocol.

    Parameters
    ----------
    device : str
        "cuda" recommended (7B model requires ~16GB VRAM).
    pretrained_name_or_path : str
        HuggingFace repo or local path.
    action_type : str
        OpenVLA outputs tokenized actions (discretized joint positions).
    control_frequency : float
        Default 5 Hz (OpenVLA runs at ~5 Hz on A100).
    mock : bool
        If True, skip model loading and output zeros.
    """

    def __init__(
        self,
        device: str = "cuda",
        pretrained_name_or_path: str = "openvla/openvla-7b",
        action_type: str = "joint_position",
        control_frequency: float = 5.0,
        chunk_size: int = 1,
        mock: bool = False,
    ):
        self.device = device
        self.pretrained_name_or_path = pretrained_name_or_path
        self.action_type = action_type
        self.control_frequency = control_frequency
        self.chunk_size = chunk_size
        self._mock = mock
        self._policy = None
        self._processor = None
        self._step = 0

        if not mock:
            self._try_load_model()

    def _try_load_model(self):
        try:
            from transformers import AutoModelForVision2Seq, AutoProcessor
            import torch

            print(f"[OpenVLA] Loading {self.pretrained_name_or_path}...")
            self._processor = AutoProcessor.from_pretrained(
                self.pretrained_name_or_path,
                trust_remote_code=True,
            )
            self._policy = AutoModelForVision2Seq.from_pretrained(
                self.pretrained_name_or_path,
                torch_dtype=torch.bfloat16,
                trust_remote_code=True,
            ).to(self.device)
            self._policy.eval()
            self._torch = torch
            print("[OpenVLA] Loaded successfully.")
        except ImportError:
            print("[OpenVLA] transformers or torch not installed — mock mode.")
            self._mock = True
        except Exception as e:
            print(f"[OpenVLA] Load failed: {e} — mock mode.")
            self._mock = True

    def reset(self) -> None:
        self._step = 0

    def predict_action(self, observation: RobotObservation) -> ActionChunk:
        if self._mock or self._policy is None:
            return self._mock_predict(observation)
        return self._real_predict(observation)

    def _real_predict(self, obs: RobotObservation) -> ActionChunk:
        torch = self._torch

        # OpenVLA uses a single front image
        img = obs.front_image
        if img.dtype == np.uint8:
            pass  # OpenVLA processor expects uint8 PIL-like images
        else:
            img = (img * 255).astype(np.uint8)

        # Build prompt
        prompt = f"In: What action should the robot take to {obs.language_instruction}?\nOut:"

        inputs = self._processor(prompt, img).to(self.device, dtype=torch.bfloat16)

        with torch.no_grad():
            action = self._policy.predict_action(**inputs, unnorm_key=None, do_sample=False)

        # action is a numpy array of shape (action_dim,)
        action = np.asarray(action, dtype=np.float32)
        actions = action.reshape(1, -1)  # horizon=1 for OpenVLA

        self._step += 1
        return ActionChunk(
            actions=actions,
            action_type=self.action_type,
            control_frequency=self.control_frequency,
        )

    def _mock_predict(self, obs: RobotObservation) -> ActionChunk:
        action_dim = 7
        if obs.state is not None:
            action_dim = obs.state.shape[0]
        actions = np.zeros((self.chunk_size, action_dim), dtype=np.float32)
        self._step += 1
        return ActionChunk(
            actions=actions,
            action_type=self.action_type,
            control_frequency=self.control_frequency,
            confidence=0.0,
        )


if __name__ == "__main__":
    print("OpenVLA Adapter Smoke Test (mock mode)")
    adapter = OpenVLAAdapter(mock=True)
    adapter.reset()
    obs = RobotObservation(
        images={"front": np.zeros((256, 256, 3), dtype=np.uint8)},
        state=np.zeros(7, dtype=np.float32),
        language_instruction="pick up the block",
        timestamp=0.0,
    )
    chunk = adapter.predict_action(obs)
    print(f"Action: {chunk}")
    print("✓ Passed")
