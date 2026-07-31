"""
SmolVLA Closed-Loop Evaluation on PushCube
============================================
Roll out the SmolVLA adapter in the dual-cube PushCube environment and
report task success, wrong-object rate, and language ablation metrics.

This script connects the full RFM pipeline::

    PushCubeEnv → RobotObservation → SmolVLAAdapter → ActionChunk
        → PushCube step → metrics

Usage
-----
.. code-block:: bash

    # Mock mode (no model download, zero actions)
    python closed_loop_eval.py --mock --n_episodes 10

    # Real evaluation (requires lerobot + GPU)
    python closed_loop_eval.py --n_episodes 20 --device cuda

    # Strict language ablation (5 conditions)
    python closed_loop_eval.py --mock --ablation --n_episodes 10

Output
------
Prints metrics to stdout and optionally saves JSON::

    {
      "correct_success": 45.0,
      "wrong_success": 5.0,
      "selection_accuracy": 85.0,
      "avg_steps": 42.3,
      "n_episodes": 20
    }
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict, List

import numpy as np

_PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from examples.unified_pushcube_env import PushCubeEnv
from examples.robot_foundation_models.common.observation_schema import RobotObservation
from examples.robot_foundation_models.smolvla.inference import SmolVLAAdapter


def evaluate_episode(
    env: PushCubeEnv,
    adapter: SmolVLAAdapter,
    language: str,
    seed: int,
    max_steps: int = 80,
) -> Dict:
    """Roll out one episode.

    Returns
    -------
    dict with keys: success, wrong_success, selected_correct, steps, trajectory.
    """
    env.reset(seed=seed)
    adapter.reset()

    for step in range(max_steps):
        img = env.render(size=128)
        state = env.get_state_vector()

        obs = RobotObservation(
            images={"front": (img * 255).astype(np.uint8)},
            state=state,
            language_instruction=language,
            timestamp=step / 20.0,
        )

        chunk = adapter.predict_action(obs)
        action = chunk.first_action()  # Already 2-D [dx, dy] for PushCube

        env.step(action)

        if env._check_success():
            break

    # Metrics
    active_cube = env.cube_positions[env.active_idx]
    other_cube = env.cube_positions[1 - env.active_idx]
    target = env.target_pos

    active_dist = float(np.linalg.norm(active_cube - target))
    other_dist = float(np.linalg.norm(other_cube - target))

    return {
        "success": active_dist < env.goal_threshold,
        "wrong_success": other_dist < env.goal_threshold,
        "selected_correct": active_dist < other_dist,
        "steps": env.step_count,
        "active_dist": active_dist,
        "other_dist": other_dist,
    }


def run_closed_loop(
    adapter: SmolVLAAdapter,
    n_episodes: int = 20,
    seed_start: int = 3000,
    lang_mode: str = "correct",
) -> Dict:
    """Run closed-loop evaluation.

    Parameters
    ----------
    adapter : SmolVLAAdapter
    n_episodes : int
    seed_start : int
    lang_mode : str
        One of ``"correct"``, ``"swapped"``, ``"none"``.

    Returns
    -------
    Aggregated metrics dict.
    """
    correct_success = 0
    wrong_success = 0
    selection_accuracy = 0
    total_steps = 0

    for ep in range(n_episodes):
        env = PushCubeEnv()
        env.reset(seed=seed_start + ep)

        if lang_mode == "correct":
            lang = env.get_language_instruction()
        elif lang_mode == "swapped":
            lang = env.get_shuffled_language()
        elif lang_mode == "none":
            lang = ""
        else:
            lang = env.get_language_instruction()

        result = evaluate_episode(env, adapter, lang, seed=seed_start + ep)

        if result["success"]:
            correct_success += 1
        if result["wrong_success"]:
            wrong_success += 1
        if result["selected_correct"]:
            selection_accuracy += 1
        total_steps += result["steps"]

    n = max(1, n_episodes)
    return {
        "correct_success": round(correct_success / n * 100, 1),
        "wrong_success": round(wrong_success / n * 100, 1),
        "selection_accuracy": round(selection_accuracy / n * 100, 1),
        "avg_steps": round(total_steps / n, 1),
        "n_episodes": n_episodes,
        "lang_mode": lang_mode,
    }


def run_ablation(adapter: SmolVLAAdapter, n_episodes: int = 20, seed_start: int = 3000):
    """Run strict language ablation with 3 conditions on the SAME model."""
    print("\n" + "=" * 60)
    print("Language Ablation (Same Model, Same Seeds)")
    print("=" * 60)

    results = {}
    for mode in ["correct", "swapped", "none"]:
        print(f"\n--- Language mode: {mode} ---")
        metrics = run_closed_loop(adapter, n_episodes, seed_start, lang_mode=mode)
        results[mode] = metrics
        for k, v in metrics.items():
            print(f"  {k}: {v}")

    return results


def main():
    parser = argparse.ArgumentParser(description="SmolVLA Closed-Loop Evaluation")
    parser.add_argument("--n_episodes", type=int, default=20, help="Number of episodes")
    parser.add_argument("--seed_start", type=int, default=3000, help="Random seed offset")
    parser.add_argument("--mock", action="store_true", help="Use mock adapter")
    parser.add_argument("--device", default="cpu", help="Device for inference")
    parser.add_argument("--ablation", action="store_true", help="Run language ablation")
    parser.add_argument("--output", default=None, help="Output JSON path")
    parser.add_argument("--lang_mode", default="correct", choices=["correct", "swapped", "none"])
    args = parser.parse_args()

    print("=" * 60)
    print("SmolVLA Closed-Loop Evaluation")
    print("=" * 60)
    print(f"Mode: {'mock' if args.mock else 'real'}, Episodes: {args.n_episodes}")

    adapter = SmolVLAAdapter(device=args.device, mock=args.mock)

    t0 = time.time()
    if args.ablation:
        results = run_ablation(adapter, args.n_episodes, args.seed_start)
    else:
        results = run_closed_loop(adapter, args.n_episodes, args.seed_start, args.lang_mode)
        print("\n=== Results ===")
        for k, v in results.items():
            print(f"  {k}: {v}")
    dt = time.time() - t0

    print(f"\nTotal time: {dt:.1f}s")

    if args.output:
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2)
        print(f"Results saved to {args.output}")


if __name__ == "__main__":
    main()
