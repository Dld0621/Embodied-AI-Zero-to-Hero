"""
Safety Filter
=============
Final gate before actions reach the robot or simulator.

The safety filter checks:
1. **Joint limits**: every commanded angle within ``[q_min, q_max]``
2. **Velocity limits**: per-joint speed below ``dq_max``
3. **Collision**: optional callback to a collision checker
4. **NaN / Inf**: reject actions with invalid values
5. **Emergency stop**: immediately zero all commands if triggered

If an action violates a constraint, the filter can:
- **clip** it to the valid range (default),
- **hold** the previous safe action, or
- **abort** (return zeros + set ``emergency_stop=True``).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional
from enum import Enum

import numpy as np

from .action_schema import ActionChunk


class SafetyAction(str, Enum):
    """What the filter does when a constraint is violated."""
    CLIP = "clip"    # clamp to valid range
    HOLD = "hold"    # repeat last safe action
    ABORT = "abort"  # zero out and stop


@dataclass
class SafetyStatus:
    """Result of a safety check."""
    safe: bool
    action: SafetyAction = SafetyAction.CLIP
    reason: str = ""
    clipped_indices: list = field(default_factory=list)
    emergency_stop: bool = False

    def __repr__(self) -> str:
        if self.emergency_stop:
            return "SafetyStatus(EMERGENCY_STOP)"
        return f"SafetyStatus(safe={self.safe}, action={self.action}, reason='{self.reason}')"


class SafetyFilter:
    """Configurable safety filter for robot actions.

    Parameters
    ----------
    joint_lower : np.ndarray
        Minimum joint positions ``(n_joints,)``.
    joint_upper : np.ndarray
        Maximum joint positions ``(n_joints,)``.
    max_velocity : float or np.ndarray
        Maximum per-step joint velocity.  If scalar, applied to all joints.
    violation_action : SafetyAction
        What to do on violation (CLIP / HOLD / ABORT).
    collision_checker : callable or None
        Optional ``Callable[[np.ndarray], bool]`` that returns ``True`` if
        the configuration is collision-free.
    """

    def __init__(
        self,
        joint_lower: np.ndarray,
        joint_upper: np.ndarray,
        max_velocity: float | np.ndarray = 0.5,
        violation_action: SafetyAction = SafetyAction.CLIP,
        collision_checker: Optional[Callable[[np.ndarray], bool]] = None,
    ):
        self.joint_lower = np.asarray(joint_lower, dtype=np.float64)
        self.joint_upper = np.asarray(joint_upper, dtype=np.float64)
        if np.isscalar(max_velocity):
            self.max_velocity = np.full_like(self.joint_lower, max_velocity)
        else:
            self.max_velocity = np.asarray(max_velocity, dtype=np.float64)
        self.violation_action = violation_action
        self.collision_checker = collision_checker

        self._last_safe: Optional[np.ndarray] = None
        self._emergency = False

    # ------------------------------------------------------------------
    # Emergency stop
    # ------------------------------------------------------------------
    def trigger_emergency_stop(self):
        """Manually trigger emergency stop."""
        self._emergency = True

    def reset(self):
        """Clear emergency stop and history."""
        self._emergency = False
        self._last_safe = None

    # ------------------------------------------------------------------
    # Core check
    # ------------------------------------------------------------------
    def check(
        self,
        action: np.ndarray,
        current_state: Optional[np.ndarray] = None,
    ) -> tuple[np.ndarray, SafetyStatus]:
        """Check and optionally correct a single action vector.

        Parameters
        ----------
        action : np.ndarray
            Proposed joint command ``(n_joints,)``.
        current_state : np.ndarray or None
            Current robot state (for velocity check).  If ``None``,
            velocity check is skipped.

        Returns
        -------
        safe_action : np.ndarray
            Corrected action (may be clipped, held, or zeroed).
        status : SafetyStatus
            Detailed safety report.
        """
        n = len(self.joint_lower)
        action = np.asarray(action, dtype=np.float64).copy()

        # Emergency stop
        if self._emergency:
            return np.zeros(n), SafetyStatus(
                safe=False,
                action=SafetyAction.ABORT,
                reason="Emergency stop active",
                emergency_stop=True,
            )

        # NaN / Inf check
        if not np.all(np.isfinite(action)):
            return self._handle_violation(action, "NaN or Inf in action")

        # Joint limit check
        below = action < self.joint_lower
        above = action > self.joint_upper
        violated = below | above
        if np.any(violated):
            clipped_idx = np.where(violated)[0].tolist()
            if self.violation_action == SafetyAction.CLIP:
                action[below] = self.joint_lower[below]
                action[above] = self.joint_upper[above]
                return action, SafetyStatus(
                    safe=True,
                    action=SafetyAction.CLIP,
                    reason="Joint limits clipped",
                    clipped_indices=clipped_idx,
                )
            return self._handle_violation(action, "Joint limit violation")

        # Velocity check
        if current_state is not None:
            delta = np.abs(action - current_state)
            over = delta > self.max_velocity
            if np.any(over):
                if self.violation_action == SafetyAction.CLIP:
                    # Scale down to max velocity
                    scale = np.where(
                        over,
                        self.max_velocity / (delta + 1e-8),
                        1.0,
                    )
                    action = current_state + (action - current_state) * scale
                    return action, SafetyStatus(
                        safe=True,
                        action=SafetyAction.CLIP,
                        reason="Velocity limit clipped",
                        clipped_indices=np.where(over)[0].tolist(),
                    )
                return self._handle_violation(action, "Velocity limit violation")

        # Collision check
        if self.collision_checker is not None:
            if not self.collision_checker(action):
                return self._handle_violation(action, "Collision detected")

        # All checks passed
        self._last_safe = action.copy()
        return action, SafetyStatus(safe=True)

    def check_chunk(
        self,
        chunk: ActionChunk,
        current_state: Optional[np.ndarray] = None,
    ) -> tuple[ActionChunk, list[SafetyStatus]]:
        """Check every step in an action chunk.

        Returns
        -------
        safe_chunk : ActionChunk
            Chunk with corrected actions.
        statuses : list[SafetyStatus]
            Per-step safety reports.
        """
        safe_actions = []
        statuses = []
        state = current_state

        for t in range(chunk.horizon):
            safe_a, status = self.check(chunk.actions[t], state)
            safe_actions.append(safe_a)
            statuses.append(status)
            if status.emergency_stop:
                # Zero out remaining steps
                safe_actions.extend(
                    np.zeros_like(chunk.actions[0])
                    for _ in range(chunk.horizon - t - 1)
                )
                break
            state = safe_a  # propagate for velocity check

        safe_chunk = ActionChunk(
            actions=np.stack(safe_actions),
            action_type=chunk.action_type,
            control_frequency=chunk.control_frequency,
            confidence=chunk.confidence,
        )
        return safe_chunk, statuses

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------
    def _handle_violation(self, action: np.ndarray, reason: str) -> tuple:
        n = len(self.joint_lower)
        if self.violation_action == SafetyAction.HOLD and self._last_safe is not None:
            return self._last_safe.copy(), SafetyStatus(
                safe=False, action=SafetyAction.HOLD, reason=reason
            )
        # ABORT or HOLD without history
        return np.zeros(n), SafetyStatus(
            safe=False, action=SafetyAction.ABORT, reason=reason
        )
