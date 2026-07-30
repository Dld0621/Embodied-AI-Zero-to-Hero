"""
Cross-Embodiment Evaluation
===========================
Evaluate how well a robot foundation model transfers across different
robot morphologies without retraining the full backbone.

This script tests the ``EmbodimentAdapter`` abstraction by:
1. Running the same policy on multiple "virtual" robot configurations
2. Measuring action-space compatibility
3. Reporting per-embodiment metrics

Supported Embodiments (for evaluation)
--------------------------------------
- ``pushcube_2d`` : 2-D planar pusher (this repo's default)
- ``franka_7dof`` : 7-DOF arm + parallel-jaw gripper
- ``omnihand_x1`` : 7-DOF arm + 10-DOF dexterous hand (Agibot)
- ``humanoid_34dof`` : Dual-arm humanoid with hands

Usage
-----
.. code-block:: bash

    # Evaluate all embodiment adapters (mock mode)
    python cross_embodiment_eval.py --mock --embodiments pushcube_2d franka_7dof

    # Smoke test
    python cross_embodiment_eval.py --mock --smoke-test
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from examples.robot_foundation_models.common.observation_schema import RobotObservation
from examples.robot_foundation_models.common.action_schema import ActionChunk
from examples.robot_foundation_models.common.embodiment_adapter import (
    EmbodimentAdapter,
    GenericAction,
)


# ------------------------------------------------------------------
# Mock embodiment adapters
# ------------------------------------------------------------------

class PushCubeAdapter(EmbodimentAdapter):
    """2-D planar pusher."""

    def __init__(self):
        super().__init__("pushcube_2d", 20.0)

    def adapt(self, chunk: ActionChunk) -> GenericAction:
        return GenericAction(
            arm_target_pose=None,
            joint_positions=chunk.first_action()[:2],
        )

    def get_robot_command(self, generic: GenericAction) -> np.ndarray:
        return generic.joint_positions


class FrankaAdapter(EmbodimentAdapter):
    """Franka Emika Panda 7-DOF + gripper."""

    def __init__(self):
        super().__init__("franka_7dof", 20.0)

    def adapt(self, chunk: ActionChunk) -> GenericAction:
        action = chunk.first_action()
        # Pad or truncate to 7-DOF + gripper
        if len(action) >= 7:
            joints = action[:7]
            gripper = action[7] if len(action) > 7 else 0.0
        else:
            joints = np.pad(action, (0, 7 - len(action)))
            gripper = 0.0
        return GenericAction(joint_positions=np.concatenate([joints, [gripper]]))

    def get_robot_command(self, generic: GenericAction) -> np.ndarray:
        return generic.joint_positions


class OmniHandAdapter(EmbodimentAdapter):
    """Agibot X1 + OmniHand O10 dexterous hand."""

    def __init__(self):
        super().__init__("omnihand_x1", 20.0)

    def adapt(self, chunk: ActionChunk) -> GenericAction:
        action = chunk.first_action()
        # Arm: 7-DOF, Hand: 10 active joints
        if len(action) >= 17:
            arm = action[:7]
            hand = action[7:17]
        else:
            arm = np.pad(action, (0, 7 - len(action))) if len(action) < 7 else action[:7]
            hand = np.zeros(10)
        return GenericAction(
            arm_target_pose=None,
            joint_positions=np.concatenate([arm, hand]),
        )

    def get_robot_command(self, generic: GenericAction) -> np.ndarray:
        return generic.joint_positions


class HumanoidAdapter(EmbodimentAdapter):
    """34-DOF humanoid (dual arm + hand + body)."""

    def __init__(self):
        super().__init__("humanoid_34dof", 30.0)

    def adapt(self, chunk: ActionChunk) -> GenericAction:
        action = chunk.first_action()
        # Pad or truncate to 34-DOF
        if len(action) < 34:
            joints = np.pad(action, (0, 34 - len(action)))
        else:
            joints = action[:34]
        return GenericAction(joint_positions=joints)

    def get_robot_command(self, generic: GenericAction) -> np.ndarray:
        return generic.joint_positions


# Registry
ADAPTERS = {
    "pushcube_2d": PushCubeAdapter,
    "franka_7dof": FrankaAdapter,
    "omnihand_x1": OmniHandAdapter,
    "humanoid_34dof": HumanoidAdapter,
}


# ------------------------------------------------------------------
# Evaluation
# ------------------------------------------------------------------

def evaluate_adapter(
    adapter: EmbodimentAdapter,
    action_dim: int = 7,
    n_steps: int = 10,
) -> Dict:
    """Test adapter with random action chunks."""
    results = []
    for _ in range(n_steps):
        chunk = ActionChunk(
            actions=np.random.randn(10, action_dim).astype(np.float32),
            action_type="joint_delta",
            control_frequency=20.0,
        )
        generic = adapter.adapt(chunk)
        command = adapter.get_robot_command(generic)
        results.append({
            "input_dim": action_dim,
            "output_dim": len(command),
            "output_shape_valid": command.ndim == 1,
        })

    return {
        "robot_type": adapter.robot_type,
        "control_frequency": adapter.control_frequency,
        "output_dim": results[0]["output_dim"],
        "shape_valid": all(r["output_shape_valid"] for r in results),
        "n_tested": n_steps,
    }


def main():
    parser = argparse.ArgumentParser(description="Cross-Embodiment Evaluation")
    parser.add_argument("--embodiments", nargs="+", default=list(ADAPTERS.keys()))
    parser.add_argument("--action_dim", type=int, default=7)
    parser.add_argument("--n_steps", type=int, default=10)
    parser.add_argument("--mock", action="store_true", help="Run without real models")
    parser.add_argument("--smoke-test", action="store_true", help="Quick test")
    parser.add_argument("--output", default=None, help="Output JSON path")
    args = parser.parse_args()

    if args.smoke_test:
        args.n_steps = 2
        args.embodiments = ["pushcube_2d", "franka_7dof"]

    print("=" * 60)
    print("Cross-Embodiment Evaluation")
    print("=" * 60)
    print(f"Testing: {', '.join(args.embodiments)}")
    print(f"Action dim: {args.action_dim}, Steps: {args.n_steps}")

    all_results = {}
    for name in args.embodiments:
        if name not in ADAPTERS:
            print(f"[Warning] Unknown embodiment: {name}")
            continue

        adapter = ADAPTERS[name]()
        result = evaluate_adapter(adapter, args.action_dim, args.n_steps)
        all_results[name] = result

        print(f"\n{name}:")
        for k, v in result.items():
            print(f"  {k}: {v}")

    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    all_valid = all(r["shape_valid"] for r in all_results.values())
    print(f"All adapters valid: {all_valid}")

    if args.output:
        with open(args.output, "w") as f:
            json.dump(all_results, f, indent=2)
        print(f"Results saved to {args.output}")


if __name__ == "__main__":
    main()
