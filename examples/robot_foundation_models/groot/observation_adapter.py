"""
GR00T Observation Adapter
=========================
Shows how humanoid robot observations map to GR00T N1.6 input format.

GR00T expects:
- Multi-camera RGB (front, left, right, wrist)
- Proprioceptive state (joint positions, velocities, forces)
- Language instruction or goal image

This module is a reference mapping, not a runnable converter.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np


@dataclass
class HumanoidState:
    """Typical humanoid proprioceptive state."""

    # Body
    pelvis_orientation: np.ndarray  # (4,) quaternion
    pelvis_angular_velocity: np.ndarray  # (3,)

    # Arms
    left_arm_joint_pos: np.ndarray  # (7,) — shoulder3 + elbow + wrist3
    right_arm_joint_pos: np.ndarray  # (7,)
    left_arm_joint_vel: np.ndarray  # (7,)
    right_arm_joint_vel: np.ndarray  # (7,)

    # Hands (optional, dexterous)
    left_hand_joint_pos: Optional[np.ndarray]  # (10,) — e.g. OmniHand O10
    right_hand_joint_pos: Optional[np.ndarray]  # (10,)

    # Legs (if whole-body)
    left_leg_joint_pos: Optional[np.ndarray]  # (6,)
    right_leg_joint_pos: Optional[np.ndarray]  # (6,)

    @property
    def total_dim(self) -> int:
        dims = [
            4, 3, 7, 7, 7, 7,  # body + arms
        ]
        if self.left_hand_joint_pos is not None:
            dims.append(len(self.left_hand_joint_pos))
        if self.right_hand_joint_pos is not None:
            dims.append(len(self.right_hand_joint_pos))
        if self.left_leg_joint_pos is not None:
            dims.append(len(self.left_leg_joint_pos))
        if self.right_leg_joint_pos is not None:
            dims.append(len(self.right_leg_joint_pos))
        return sum(dims)

    def to_vector(self) -> np.ndarray:
        """Flatten to 1-D vector for model input."""
        parts = [
            self.pelvis_orientation,
            self.pelvis_angular_velocity,
            self.left_arm_joint_pos,
            self.right_arm_joint_pos,
            self.left_arm_joint_vel,
            self.right_arm_joint_vel,
        ]
        if self.left_hand_joint_pos is not None:
            parts.append(self.left_hand_joint_pos)
        if self.right_hand_joint_pos is not None:
            parts.append(self.right_hand_joint_pos)
        if self.left_leg_joint_pos is not None:
            parts.append(self.left_leg_joint_pos)
        if self.right_leg_joint_pos is not None:
            parts.append(self.right_leg_joint_pos)
        return np.concatenate(parts).astype(np.float32)


@dataclass
class GR00TInput:
    """Standardized input to GR00T N1.6 model."""

    images: Dict[str, np.ndarray]  # camera_name -> (H, W, 3) uint8
    state: np.ndarray  # flattened proprioception
    language: str
    goal_image: Optional[np.ndarray] = None  # for goal-conditioned tasks
