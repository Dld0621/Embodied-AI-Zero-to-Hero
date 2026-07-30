"""
Observation Schema
==================
Standardized observation format for all robot foundation models.

Every model in the RFM pipeline — SmolVLA, OpenVLA, Octo, GR00T — receives
the same ``RobotObservation`` dataclass.  Model-specific adapters are
responsible for converting this canonical format into whatever tensor
layout the underlying model expects (e.g., LeRobot dict, RLDS tf.Tensor).

Design principles
-----------------
1. **Camera-agnostic**: ``images`` is a ``dict[str, np.ndarray]`` keyed by
   camera name (``"front"``, ``"wrist_left"``, …).  Models that only use
   one camera simply read ``images["front"]``.
2. **State is optional**: some models (e.g., Diffusion Policy) may not use
   proprioceptive state.  ``state`` can be ``None``.
3. **Language is always present**: even if a model doesn't use language,
   the instruction is carried for logging and ablation.
4. **Timestamp for sync**: enables multi-camera time-alignment and
   action-chunk replay at the correct control frequency.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional

import numpy as np


@dataclass
class RobotObservation:
    """Canonical observation passed to ``RobotFoundationModel.predict_action``.

    Attributes
    ----------
    images : dict[str, np.ndarray]
        Camera name → RGB image of shape ``(H, W, 3)`` in ``uint8`` or
        ``float32`` (0–1).  At least one camera ("front") should be present.
    state : np.ndarray or None
        Robot proprioceptive state (joint positions, velocities, etc.).
        Shape ``(state_dim,)``.  ``None`` if the model doesn't use state.
    language_instruction : str
        Natural-language task description, e.g.,
        ``"push the red cube to the target"``.
    timestamp : float
        Observation time in seconds (wall-clock or episode-relative).
    extras : dict
        Optional model-specific metadata (e.g., ``{"goal_image": ...}``).
    """

    images: Dict[str, np.ndarray]
    state: Optional[np.ndarray]
    language_instruction: str
    timestamp: float
    extras: dict = field(default_factory=dict)

    # ------------------------------------------------------------------
    # Convenience helpers
    # ------------------------------------------------------------------
    @property
    def front_image(self) -> np.ndarray:
        """Shortcut for the front camera image."""
        return self.images["front"]

    @property
    def has_state(self) -> bool:
        return self.state is not None

    def camera_names(self) -> list:
        """Return sorted list of available camera names."""
        return sorted(self.images.keys())

    def __repr__(self) -> str:
        cams = ", ".join(
            f"{k}{v.shape}" for k, v in self.images.items()
        )
        s_dim = self.state.shape[0] if self.state is not None else 0
        return (
            f"RobotObservation(cameras=[{cams}], state_dim={s_dim}, "
            f'lang="{self.language_instruction[:40]}...", t={self.timestamp:.3f})'
        )
