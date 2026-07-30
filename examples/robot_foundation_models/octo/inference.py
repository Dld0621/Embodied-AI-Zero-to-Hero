"""
Octo Inference Adapter
======================
Wraps the Octo generalist robot policy to conform to the
``RobotFoundationModel`` protocol.

Octo is a transformer-based diffusion policy trained on ~800k robot
episodes across multiple robots, cameras, and action spaces.  Its key
feature is **cross-embodiment**: the same model can be adapted to new
robots by providing a small amount of target-domain data.

Architecture highlights
-----------------------
- **Input**: task token (language or goal image) + observation tokens
  (image patches + proprioception)
- **Output**: diffusion-denoised action chunk
- **Cross-embodiment**: action readout head is retrained per robot;
  transformer backbone is frozen

Repository status
-----------------
This adapter is a **tutorial stub** showing how Octo would fit into the
RFM interface.  Full integration requires:

1. ``pip install octo`` (JAX/Flax-based, distinct PyTorch stack)
2. Download Octo weights (27M or 93M)
3. Retrain action readout head on target robot data

Because Octo uses JAX/Flax (not PyTorch), integrating it fully would
introduce significant maintenance overhead.  This module is provided for
architectural reference and cross-embodiment study.

Usage
-----
.. code-block:: python

    from examples.robot_foundation_models.octo.inference import OctoAdapter
    from examples.robot_foundation_models.common import RobotObservation

    adapter = OctoAdapter(mock=True)
    adapter.reset()
    chunk = adapter.predict_action(obs)

References
----------
- Octo paper: https://arxiv.org/abs/2405.12213
- Octo models: https://octo-models.github.io/
"""

from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Optional

import numpy as np

_PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from examples.robot_foundation_models.common.observation_schema import RobotObservation
from examples.robot_foundation_models.common.action_schema import ActionChunk


class OctoAdapter:
    """Adapts Octo to the ``RobotFoundationModel`` protocol.

    Parameters
    ----------
    model_type : str
        ``"octo-small"`` (27M) or ``"octo-base"`` (93M).
    action_horizon : int
        Number of future actions to predict.
    mock : bool
        If True, run without loading JAX/Flax weights.
    """

    def __init__(
        self,
        model_type: str = "octo-small",
        action_horizon: int = 10,
        mock: bool = False,
    ):
        self.model_type = model_type
        self.action_horizon = action_horizon
        self._mock = mock
        self._model = None
        self._step = 0

        if not mock:
            self._try_load_model()

    def _try_load_model(self):
        """Attempt to load Octo via the official API."""
        try:
            # Octo is JAX-based; this import will fail without JAX
            from octo.model.octo_model import OctoModel

            print(f"[Octo] Loading {self.model_type}...")
            self._model = OctoModel.load_pretrained("hf://rail-berkeley/" + self.model_type)
            print(f"[Octo] Loaded {self.model_type}")
        except ImportError:
            print("[Octo] octo/jax not installed — running in mock mode.")
            print("  Install: pip install octo")
            self._mock = True
        except Exception as e:
            print(f"[Octo] Failed to load: {e}")
            self._mock = True

    # ------------------------------------------------------------------
    # RobotFoundationModel protocol
    # ------------------------------------------------------------------
    def reset(self) -> None:
        self._step = 0

    def predict_action(self, observation: RobotObservation) -> ActionChunk:
        if self._mock or self._model is None:
            return self._mock_predict(observation)
        return self._real_predict(observation)

    # ------------------------------------------------------------------
    # Real inference (JAX/Flax)
    # ------------------------------------------------------------------
    def _real_predict(self, obs: RobotObservation) -> ActionChunk:
        """Call the actual Octo model.

        This is a simplified sketch — real Octo inference involves:
        1. Tokenizing images into patches
        2. Adding proprioception tokens
        3. Running transformer forward pass
        4. Diffusion denoising for action generation
        5. Un-normalizing actions with dataset statistics
        """
        # Placeholder: real implementation would use Octo's API
        raise NotImplementedError(
            "Real Octo inference not yet implemented. "
            "Use mock=True for interface testing."
        )

    # ------------------------------------------------------------------
    # Mock inference
    # ------------------------------------------------------------------
    def _mock_predict(self, obs: RobotObservation) -> ActionChunk:
        action_dim = 7  # default
        if obs.state is not None:
            action_dim = obs.state.shape[0]

        actions = np.zeros((self.action_horizon, action_dim), dtype=np.float32)
        self._step += 1
        return ActionChunk(
            actions=actions,
            action_type="joint_delta",
            control_frequency=20.0,
            confidence=0.0,
        )

    def __repr__(self) -> str:
        mode = "mock" if self._mock else "loaded"
        return f"OctoAdapter(model={self.model_type}, mode={mode})"


# ------------------------------------------------------------------
# Cross-embodiment tutorial
# ------------------------------------------------------------------
class OctoCrossEmbodimentTutorial:
    """Tutorial code showing how Octo adapts to new robots.

    Octo's cross-embodiment works by:
    1. Keeping the transformer backbone frozen
    2. Adding a new *action readout head* for the target robot
    3. Fine-tuning only the readout head on ~50-100 target episodes

    This class documents the conceptual steps without requiring JAX.
    """

    STEPS = """
    Cross-Embodiment Adaptation with Octo
    =====================================

    1. Prepare target robot data in RLDS or canonical format.
       Episodes should include: images, state, action, language.

    2. Load pre-trained Octo (backbone frozen):
       model = OctoModel.load_pretrained("octo-small")

    3. Add new action readout head:
       model = model.add_action_head(
           action_dim=target_action_dim,
           action_horizon=10,
       )

    4. Fine-tune readout head only:
       for batch in target_data:
           loss = model(batch, train_head_only=True)
           loss.backward()
           optimizer.step()

    5. Evaluate on target robot tasks.

    Key insight: The transformer backbone handles visual-language understanding
    generically; only the action readout needs to know about the specific
    robot's morphology and control space.
    """

    @classmethod
    def print_tutorial(cls):
        print(cls.STEPS)


# ------------------------------------------------------------------
# Smoke test
# ------------------------------------------------------------------
if __name__ == "__main__":
    print("=" * 60)
    print("Octo Adapter Smoke Test")
    print("=" * 60)

    adapter = OctoAdapter(mock=True)
    adapter.reset()

    fake_img = np.zeros((128, 128, 3), dtype=np.uint8)
    obs = RobotObservation(
        images={"front": fake_img},
        state=np.zeros(7, dtype=np.float32),
        language_instruction="pick up the red block",
        timestamp=time.time(),
    )

    chunk = adapter.predict_action(obs)
    print(f"\nObservation: {obs}")
    print(f"Action chunk: {chunk}")
    print(f"  actions shape: {chunk.actions.shape}")

    print("\n--- Cross-Embodiment Tutorial ---")
    OctoCrossEmbodimentTutorial.print_tutorial()

    print("\n✓ Octo adapter smoke test passed (mock mode)")
