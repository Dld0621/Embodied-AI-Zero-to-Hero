"""
Model Interface
===============
Protocol (interface) that every robot foundation model must implement.

By using ``typing.Protocol``, we get structural subtyping — any class that
has ``reset()`` and ``predict_action()`` with the right signatures
automatically satisfies the protocol, without needing to inherit.

Usage
-----
.. code-block:: python

    from common import RobotObservation, ActionChunk, RobotFoundationModel

    class MyVLA:  # no need to inherit RobotFoundationModel
        def reset(self) -> None:
            self._step = 0

        def predict_action(self, obs: RobotObservation) -> ActionChunk:
            ...

    # Anywhere in the control loop:
    def run_episode(model: RobotFoundationModel, env):
        model.reset()
        obs = env.reset()
        while not done:
            action_chunk = model.predict_action(obs)
            obs, reward, done, info = env.step(action_chunk.first_action())
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from .observation_schema import RobotObservation
from .action_schema import ActionChunk


@runtime_checkable
class RobotFoundationModel(Protocol):
    """Protocol defining the interface for robot foundation models.

    Implementations may wrap SmolVLA, OpenVLA, Octo, GR00T, or a custom
    policy.  The external control loop depends only on this interface,
    so models can be swapped without changing any controller code.

    Required methods
    ----------------
    reset() -> None
        Clear internal state (action queues, hidden states, etc.)
        between episodes.

    predict_action(observation: RobotObservation) -> ActionChunk
        Given a canonical observation, predict an action chunk.
        The chunk may contain a single step (horizon=1) or a multi-step
        sequence for action chunking / receding-horizon control.
    """

    def reset(self) -> None:
        """Reset model state between episodes."""
        ...

    def predict_action(
        self,
        observation: RobotObservation,
    ) -> ActionChunk:
        """Predict action chunk from observation.

        Parameters
        ----------
        observation : RobotObservation
            Canonical observation containing images, state, and language.

        Returns
        -------
        ActionChunk
            Action sequence with type and control frequency metadata.
        """
        ...
