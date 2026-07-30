"""
Embodiment Adapter
==================
Converts generic ``ActionChunk`` output from a foundation model into
robot-specific commands.

Different robots have different:
- Joint counts (7-DOF arm vs. 6-DOF arm)
- Action spaces (joint position vs. end-effector delta)
- Control frequencies (10 Hz vs. 500 Hz)
- Coordinate frames (base frame vs. world frame)

The ``EmbodimentAdapter`` abstracts these differences so that the same
foundation model can control multiple robots by swapping adapters.

Example hierarchy::

    RobotFoundationModel (generic)
        ↓ ActionChunk (ee_delta, horizon=10)
    EmbodimentAdapter
        ↓ robot-specific joint commands
    MuJoCo / Real Robot
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np

from .action_schema import ActionChunk


@dataclass
class GenericAction:
    """Intermediate representation between model output and robot command.

    This decouples the foundation model's output space from the robot's
    control space.  The adapter translates ``ActionChunk`` → ``GenericAction``
    → robot-specific command.

    For high-level intent (recommended for dexterous manipulation):

    .. code-block:: python

        {
            "arm_target_pose": [x, y, z, qx, qy, qz, qw],
            "hand_intent": "power_grasp",
            "target_object": "cup_handle",
            "contact_regions": ["thumb_pad", "index_pad"],
            "grasp_phase": "approach",
        }
    """

    arm_target_pose: Optional[np.ndarray] = None   # [x,y,z,qx,qy,qz,qw]
    joint_positions: Optional[np.ndarray] = None   # [n_joints]
    joint_velocities: Optional[np.ndarray] = None
    hand_intent: Optional[str] = None               # "power_grasp", "pinch", ...
    target_object: Optional[str] = None
    contact_regions: Optional[list] = None
    grasp_phase: Optional[str] = None               # "approach", "contact", "lift"
    extras: Dict = None

    def __post_init__(self):
        if self.extras is None:
            self.extras = {}


class EmbodimentAdapter(ABC):
    """Abstract base class for robot-specific action translation.

    Subclasses must implement :meth:`adapt` to convert an ``ActionChunk``
    into a ``GenericAction`` (or directly into robot commands).

    Common adapters:
    - ``FrankaAdapter``: 7-DOF arm + parallel-jaw gripper
    - ``OmniHandAdapter``: 7-DOF arm + 10-DOF dexterous hand
    - ``PushCubeAdapter``: 2-DOF planar pusher (for PushCube env)
    """

    def __init__(self, robot_type: str, control_frequency: float):
        self.robot_type = robot_type
        self.control_frequency = control_frequency

    @abstractmethod
    def adapt(self, action_chunk: ActionChunk) -> GenericAction:
        """Convert model output to robot-specific intermediate action.

        Parameters
        ----------
        action_chunk : ActionChunk
            Generic action from the foundation model.

        Returns
        -------
        GenericAction
            Robot-specific intermediate representation.
        """
        ...

    @abstractmethod
    def get_robot_command(self, generic: GenericAction) -> np.ndarray:
        """Convert GenericAction to raw robot command vector.

        This is the final step before sending to MuJoCo / real hardware.
        """
        ...

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(robot={self.robot_type}, "
            f"freq={self.control_frequency}Hz)"
        )
