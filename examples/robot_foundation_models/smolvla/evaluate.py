"""
SmolVLA Evaluation Script
=========================
Runs offline and closed-loop evaluation for the SmolVLA adapter.

Offline evaluation:
- Compare predicted actions to expert demonstrations
- Report Action MAE, L2 error, direction consistency

Closed-loop evaluation:
- Roll out the model in the PushCube environment
- Report task success rate, wrong-object rate, selection accuracy

Usage
-----
.. code-block:: bash

    # Offline (mock mode, for CI)
    python evaluate.py --mode offline --mock

    # Offline (real model)
    python evaluate.py --mode offline --data results/benchmarks/pushcube_expert.json

    # Closed-loop
    python evaluate.py --mode closed_loop --n_episodes 20
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import List, Dict

import numpy as np

_PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "..")
)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from examples.robot_foundation_models.common.observation_schema import RobotObservation
from examples.robot_foundation_models.common.action_schema import ActionChunk, ActionResult


def run_offline_eval(
    adapter,
    expert_data: List[dict],
    device: str = "cpu",
) -> Dict:
    """Compare model predictions to expert actions.

    Parameters
    ----------
    adapter : SmolVLAAdapter
        The model adapter to evaluate.
    expert_data : list[dict]
        Each dict has keys: "image", "state", "language", "expert_action".
    device : str
        Device for inference.

    Returns
    -------
    dict with metrics:
        - action_mae: mean absolute error per dimension
        - action_l2: L2 distance
        - direction_consistency: fraction of steps where predicted direction
          matches expert direction (cosine > 0)
        - n_samples
    """
    adapter.reset()

    all_mae = []
    all_l2 = []
    direction_matches = 0

    for sample in expert_data:
        obs = RobotObservation(
            images={"front": sample["image"]},
            state=sample.get("state"),
            language_instruction=sample["language"],
            timestamp=0.0,
        )

        chunk = adapter.predict_action(obs)
        pred = chunk.first_action()
        expert = sample["expert_action"]

        # Metrics
        mae = np.mean(np.abs(pred - expert))
        l2 = np.linalg.norm(pred - expert)
        all_mae.append(mae)
        all_l2.append(l2)

        # Direction consistency
        if np.linalg.norm(pred) > 1e-6 and np.linalg.norm(expert) > 1e-6:
            cos_sim = np.dot(pred, expert) / (
                np.linalg.norm(pred) * np.linalg.norm(expert)
            )
            if cos_sim > 0:
                direction_matches += 1

    n = len(expert_data)
    return {
        "action_mae": float(np.mean(all_mae)) if all_mae else 0.0,
        "action_l2": float(np.mean(all_l2)) if all_l2 else 0.0,
        "direction_consistency": direction_matches / max(1, n),
        "n_samples": n,
    }


def run_closed_loop_eval(
    adapter,
    n_episodes: int = 20,
    seed_offset: int = 2000,
) -> Dict:
    """Roll out the model in PushCube environment.

    Returns metrics for correct/wrong cube selection and task success.
    """
    # Import PushCube env
    from examples.unified_pushcube_env import PushCubeEnv

    adapter.reset()

    correct_success = 0
    wrong_success = 0
    selection_accuracy = 0
    total_steps = 0

    for ep in range(n_episodes):
        env = PushCubeEnv()
        env.reset(seed=seed_offset + ep)
        adapter.reset()

        lang = env.get_language_instruction()
        goal_onehot = env.get_goal_color_onehot()

        for step in range(env.max_steps):
            img = env.render(size=128)
            state = env.get_state_vector()

            obs = RobotObservation(
                images={"front": img},
                state=state,
                language_instruction=lang,
                timestamp=step / env.control_frequency if hasattr(env, 'control_frequency') else step * 0.05,
            )

            chunk = adapter.predict_action(obs)
            action = chunk.first_action()  # Already 2-D [dx, dy] for PushCube

            env.step(action)
            total_steps += 1

            if env._check_success():
                break

        # Measure both cubes
        active_cube = env.cube_positions[env.active_idx]
        other_cube = env.cube_positions[1 - env.active_idx]
        target = env.target_pos

        active_dist = float(np.linalg.norm(active_cube - target))
        other_dist = float(np.linalg.norm(other_cube - target))

        if active_dist < env.goal_threshold:
            correct_success += 1
        if other_dist < env.goal_threshold:
            wrong_success += 1
        if active_dist < other_dist:
            selection_accuracy += 1

    n = max(1, n_episodes)
    return {
        "correct_success": round(correct_success / n * 100, 1),
        "wrong_success": round(wrong_success / n * 100, 1),
        "selection_accuracy": round(selection_accuracy / n * 100, 1),
        "avg_steps": total_steps / n,
        "n_episodes": n_episodes,
    }


def main():
    parser = argparse.ArgumentParser(description="SmolVLA Evaluation")
    parser.add_argument("--mode", choices=["offline", "closed_loop"], default="closed_loop")
    parser.add_argument("--mock", action="store_true", help="Use mock model (no lerobot needed)")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--n_episodes", type=int, default=20)
    parser.add_argument("--data", default=None, help="Path to expert data JSON")
    parser.add_argument("--output", default=None, help="Output JSON path")
    args = parser.parse_args()

    from examples.robot_foundation_models.smolvla.inference import SmolVLAAdapter

    adapter = SmolVLAAdapter(device=args.device, mock=args.mock)

    if args.mode == "offline":
        # Load expert data or generate synthetic
        if args.data and os.path.exists(args.data):
            with open(args.data) as f:
                expert_data = json.load(f)
        else:
            print("[Info] No expert data provided, generating synthetic...")
            expert_data = []
            for i in range(10):
                expert_data.append({
                    "image": np.zeros((128, 128, 3), dtype=np.uint8),
                    "state": np.zeros(7, dtype=np.float32),
                    "language": f"push the {'red' if i % 2 == 0 else 'green'} cube",
                    "expert_action": np.zeros(7, dtype=np.float32),
                })

        results = run_offline_eval(adapter, expert_data, args.device)
        print("\n=== Offline Evaluation Results ===")
        for k, v in results.items():
            print(f"  {k}: {v}")
    else:
        results = run_closed_loop_eval(adapter, args.n_episodes)
        print("\n=== Closed-Loop Evaluation Results ===")
        for k, v in results.items():
            print(f"  {k}: {v}")

    if args.output:
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
