"""
Action Schema
=============
Standardized action format returned by all robot foundation models.

``ActionChunk`` represents a *sequence* of actions predicted by the model
(e.g., 10 steps of joint-position deltas).  The ``action_type`` field tells
downstream adapters how to interpret the raw numbers.

Supported action types
----------------------
- ``"joint_position"``: absolute joint angles (rad)
- ``"joint_velocity"``: joint velocity commands (rad/s)
- ``"ee_pose"``: end-effector pose ``[x, y, z, qx, qy, qz, qw]``
- ``"ee_delta"``: end-effector delta ``[dx, dy, dz, droll, dpitch, dyaw]``
- ``"joint_delta"``: joint angle deltas (rad)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np


# Valid action type strings
VALID_ACTION_TYPES = frozenset({
    "joint_position",
    "joint_velocity",
    "ee_pose",
    "ee_delta",
    "joint_delta",
})


@dataclass
class ActionChunk:
    """Canonical action output from ``RobotFoundationModel.predict_action``.

    Attributes
    ----------
    actions : np.ndarray
        Action sequence of shape ``(horizon, action_dim)``.
    action_type : str
        How to interpret each row (see module docstring).
    control_frequency : float
        Hz at which actions should be executed on the robot.
    confidence : float or None
        Optional model confidence / log-probability, useful for safety
        gating and ensemble methods.
    """

    actions: np.ndarray
    action_type: str
    control_frequency: float
    confidence: Optional[float] = None

    def __post_init__(self):
        if self.action_type not in VALID_ACTION_TYPES:
            raise ValueError(
                f"Unknown action_type '{self.action_type}'. "
                f"Valid: {sorted(VALID_ACTION_TYPES)}"
            )
        if self.actions.ndim != 2:
            raise ValueError(
                f"actions must be 2-D (horizon, action_dim), "
                f"got shape {self.actions.shape}"
            )

    # ------------------------------------------------------------------
    # Convenience
    # ------------------------------------------------------------------
    @property
    def horizon(self) -> int:
        return self.actions.shape[0]

    @property
    def action_dim(self) -> int:
        return self.actions.shape[1]

    def first_action(self) -> np.ndarray:
        """Return only the first step — useful for receding-horizon control."""
        return self.actions[0]

    def __repr__(self) -> str:
        return (
            f"ActionChunk(horizon={self.horizon}, dim={self.action_dim}, "
            f"type={self.action_type}, freq={self.control_frequency}Hz)"
        )


@dataclass
class ActionResult:
    """Feedback after executing an action on the robot/sim.

    Used for closed-loop evaluation and world-model training.
    """

    success: bool
    collision: bool = False
    timeout: bool = False
    steps_executed: int = 0
    final_reward: float = 0.0
    info: dict = None

    def __post_init__(self):
        if self.info is None:
            self.info = {}
