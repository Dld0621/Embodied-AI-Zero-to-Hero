"""
Closed-Loop Evaluation Benchmark for Robot Foundation Models
============================================================
Rolls out a model adapter in the dual-cube PushCube environment and
measures task-level performance under multiple language conditions.

The PushCube task places two colored cubes on a table.  A language
instruction identifies which cube to push into the target zone.  Because
only the *active* cube (identified by language) should reach the target,
we measure three outcome metrics:

* **correct_success**  — the *active* cube ended in the target zone.
* **wrong_success**    — the *other* cube ended in the target zone.
* **selection_accuracy** — the active cube is closer to the target than
  the other cube (even if neither is in the goal).

Two additional diagnostic metrics are reported:

* **avg_steps**        — mean number of steps executed per episode.
* **collision_count**  — total number of cube-contact events (a cube
  position changed because the arm pushed it).

Language ablation
-----------------
The same model is evaluated under three language conditions on the same
set of episodes (same seeds):

  * ``correct`` — the true instruction for the active cube.
  * ``swapped`` — the distractor cube's instruction (wrong color word).
  * ``none``    — empty string (language dropout at inference time).

If the model truly uses language, swapping the color word should shift
success from the active cube to the other cube, and removing language
should degrade selection accuracy.

Usage
-----
.. code-block:: bash

    # CI smoke test (mock mode, 2 episodes, no GPU needed)
    python benchmarks/robot_foundation_models/evaluate_closed_loop.py --mock --smoke-test

    # Full run with SmolVLA in mock mode
    python benchmarks/robot_foundation_models/evaluate_closed_loop.py --mock --n-episodes 20

    # Real model
    python benchmarks/robot_foundation_models/evaluate_closed_loop.py --model smolvla

    # Only run the "correct" condition (skip ablation)
    python benchmarks/robot_foundation_models/evaluate_closed_loop.py --mock --conditions correct
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Dict, List

import numpy as np

# ---------------------------------------------------------------------------
# Path setup — allow running as a standalone script from the project root
# ---------------------------------------------------------------------------
_PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..")
)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from examples.robot_foundation_models.common.observation_schema import RobotObservation
from examples.unified_pushcube_env import PushCubeEnv

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
PUSHCUBE_ACTION_DIM = 2   # [dx, dy]
RENDER_SIZE = 128
DEFAULT_MAX_STEPS = 80
DEFAULT_SEED_OFFSET = 2000
ALL_CONDITIONS = ("correct", "swapped", "none")


class _NumpyEncoder(json.JSONEncoder):
    """JSON encoder that handles numpy scalar/array types."""

    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (np.bool_,)):
            return bool(obj)
        return super().default(obj)


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------
def load_model(model_name: str, mock: bool, device: str = "cpu"):
    """Instantiate a robot foundation model adapter.

    Returns an object implementing the ``RobotFoundationModel`` protocol.
    """
    if model_name == "smolvla":
        from examples.robot_foundation_models.smolvla.inference import SmolVLAAdapter
        # In mock mode _try_load_model() is skipped, so chunk_size stays
        # None.  Pass an explicit default so _mock_predict works.
        if mock:
            return SmolVLAAdapter(device=device, mock=True, chunk_size=10)
        return SmolVLAAdapter(device=device, mock=False)
    elif model_name == "openvla":
        from examples.robot_foundation_models.openvla.inference import OpenVLAAdapter
        return OpenVLAAdapter(device=device, mock=mock)
    else:
        raise ValueError(
            f"Unknown model '{model_name}'. Choose 'smolvla' or 'openvla'."
        )


# ---------------------------------------------------------------------------
# Language condition helpers
# ---------------------------------------------------------------------------
def get_language(env: PushCubeEnv, condition: str) -> str:
    """Return the language instruction for a given ablation condition.

    Parameters
    ----------
    env : PushCubeEnv
        The environment (must have been reset).
    condition : str
        One of ``"correct"``, ``"swapped"``, ``"none"``.

    Returns
    -------
    str
        The language string to feed to the model.
    """
    if condition == "correct":
        return env.get_language_instruction()
    elif condition == "swapped":
        return env.get_shuffled_language()
    elif condition == "none":
        return ""
    else:
        raise ValueError(f"Unknown condition '{condition}'.")


# ---------------------------------------------------------------------------
# Rollout
# ---------------------------------------------------------------------------
def rollout_episode(
    adapter,
    env: PushCubeEnv,
    language: str,
    max_steps: int,
    render_size: int = RENDER_SIZE,
) -> dict:
    """Execute a single closed-loop episode.

    The adapter is reset at the start of the episode.  At each step the
    model receives the rendered image, 14-D state, and the (possibly
    ablated) language string.  The predicted action must be 2-D
    ``[dx, dy]`` for PushCube.

    Returns a dict with per-episode outcome metrics.
    """
    adapter.reset()
    steps = 0
    collision_count = 0

    for step in range(max_steps):
        img = env.render(size=render_size)
        state = env.get_state_vector()

        obs = RobotObservation(
            images={"front": img},
            state=state,
            language_instruction=language,
            timestamp=float(step),
        )

        chunk = adapter.predict_action(obs)

        # PushCube requires exactly 2-D action [dx, dy].
        # The adapter must be configured with action_type="ee_delta_2d"
        # and action_dim=2 — truncation is NOT allowed.
        action = chunk.first_action()
        if len(action) != PUSHCUBE_ACTION_DIM:
            raise ValueError(
                f"Expected action dim {PUSHCUBE_ACTION_DIM} for PushCube, "
                f"got {len(action)}. Ensure the model adapter is configured "
                f"with action_type='ee_delta_2d' and action_dim=2."
            )

        # Track collisions: detect cube-position changes caused by the arm.
        old_positions = [c.copy() for c in env.cube_positions]
        env.step(action)
        for i in range(2):
            if np.linalg.norm(env.cube_positions[i] - old_positions[i]) > 1e-6:
                collision_count += 1

        steps += 1
        if env._check_success():
            break

    # End-of-episode measurement
    active_cube = env.cube_positions[env.active_idx]
    other_cube = env.cube_positions[1 - env.active_idx]
    target = env.target_pos

    active_dist = float(np.linalg.norm(active_cube - target))
    other_dist = float(np.linalg.norm(other_cube - target))

    return {
        "steps": steps,
        "collision_count": collision_count,
        "correct_success": active_dist < env.goal_threshold,
        "wrong_success": other_dist < env.goal_threshold,
        "selection_correct": active_dist < other_dist,
        "active_dist": active_dist,
        "other_dist": other_dist,
    }


def evaluate_condition(
    adapter,
    n_episodes: int,
    seed_offset: int,
    condition: str,
    max_steps: int = DEFAULT_MAX_STEPS,
    render_size: int = RENDER_SIZE,
) -> Dict:
    """Evaluate one language condition across *n_episodes* episodes.

    Each episode uses a deterministic seed (``seed_offset + ep``) so that
    different conditions see the same initial states.
    """
    results: List[dict] = []
    for ep in range(n_episodes):
        env = PushCubeEnv()
        env.reset(seed=seed_offset + ep)
        language = get_language(env, condition)
        result = rollout_episode(adapter, env, language, max_steps, render_size)
        results.append(result)

    n = max(1, len(results))
    return {
        "correct_success": round(sum(r["correct_success"] for r in results) / n * 100, 1),
        "wrong_success": round(sum(r["wrong_success"] for r in results) / n * 100, 1),
        "selection_accuracy": round(sum(r["selection_correct"] for r in results) / n * 100, 1),
        "avg_steps": round(sum(r["steps"] for r in results) / n, 1),
        "collision_count": int(sum(r["collision_count"] for r in results)),
        "n_episodes": len(results),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Closed-loop evaluation benchmark for robot foundation models"
    )
    parser.add_argument(
        "--model", choices=["smolvla", "openvla"], default="smolvla",
        help="Model adapter to evaluate (default: smolvla)",
    )
    parser.add_argument(
        "--mock", action="store_true",
        help="Use mock model (no GPU or model download needed)",
    )
    parser.add_argument(
        "--smoke-test", action="store_true",
        help="CI mode: reduce to 2 episodes for fast sanity check",
    )
    parser.add_argument(
        "--n-episodes", type=int, default=20,
        help="Number of episodes per condition (default: 20)",
    )
    parser.add_argument(
        "--device", default="cpu",
        help="Torch device for inference (default: cpu)",
    )
    parser.add_argument(
        "--seed-offset", type=int, default=DEFAULT_SEED_OFFSET,
        help=f"Seed offset for evaluation episodes (default: {DEFAULT_SEED_OFFSET})",
    )
    parser.add_argument(
        "--max-steps", type=int, default=DEFAULT_MAX_STEPS,
        help=f"Max steps per episode (default: {DEFAULT_MAX_STEPS})",
    )
    parser.add_argument(
        "--conditions", default="correct,swapped,none",
        help="Comma-separated language conditions to evaluate (default: correct,swapped,none)",
    )
    parser.add_argument(
        "--output", default=None,
        help="Output JSON path (default: results/benchmarks/rfm/closed_loop_eval_<model>_<tag>.json)",
    )
    args = parser.parse_args()

    # Smoke-test overrides
    if args.smoke_test:
        args.n_episodes = 2

    # Parse conditions
    conditions = [c.strip() for c in args.conditions.split(",") if c.strip()]
    for c in conditions:
        if c not in ALL_CONDITIONS:
            parser.error(
                f"Unknown condition '{c}'. Valid: {', '.join(ALL_CONDITIONS)}"
            )

    # Header
    print("=" * 72)
    print(" RFM Closed-Loop Evaluation Benchmark")
    print("=" * 72)
    print(f"  Model:      {args.model}")
    print(f"  Mock mode:  {args.mock}")
    print(f"  Smoke test: {args.smoke_test}")
    print(f"  Episodes:   {args.n_episodes} per condition")
    print(f"  Conditions: {conditions}")
    print(f"  Device:     {args.device}")
    print()

    # 1. Load model
    print("[1/2] Loading model adapter...")
    adapter = load_model(args.model, args.mock, args.device)
    print(f"      {adapter}")

    # 2. Evaluate each condition
    print(f"\n[2/2] Evaluating {len(conditions)} condition(s)...")
    condition_results: Dict[str, Dict] = {}
    for cond in conditions:
        print(f"\n  --- Condition: {cond} ---")
        res = evaluate_condition(
            adapter,
            n_episodes=args.n_episodes,
            seed_offset=args.seed_offset,
            condition=cond,
            max_steps=args.max_steps,
        )
        condition_results[cond] = res
        print(f"    correct_success:   {res['correct_success']:5.1f}%")
        print(f"    wrong_success:     {res['wrong_success']:5.1f}%")
        print(f"    selection_accuracy:{res['selection_accuracy']:5.1f}%")
        print(f"    avg_steps:         {res['avg_steps']}")
        print(f"    collision_count:   {res['collision_count']}")

    # Summary table
    print("\n" + "-" * 72)
    print(" Closed-Loop Evaluation Summary")
    print("-" * 72)
    header = f"  {'Condition':<14} {'correct%':>9} {'wrong%':>9} {'select%':>9} {'avg_stp':>8} {'collisions':>11}"
    print(header)
    print("  " + "-" * 66)
    for cond in conditions:
        r = condition_results[cond]
        print(
            f"  {cond:<14} {r['correct_success']:>9.1f} {r['wrong_success']:>9.1f} "
            f"{r['selection_accuracy']:>9.1f} {r['avg_steps']:>8.1f} {r['collision_count']:>11}"
        )
    print("-" * 72)

    # Assemble output
    results = {
        "benchmark": "rfm_closed_loop_evaluation",
        "model": {
            "name": args.model,
            "mock": args.mock,
            "device": args.device,
        },
        "config": {
            "n_episodes": args.n_episodes,
            "max_steps": args.max_steps,
            "seed_offset": args.seed_offset,
            "conditions": conditions,
            "smoke_test": args.smoke_test,
        },
        "conditions": condition_results,
    }

    # Determine output path — mock and real results are stored separately
    if args.output is None:
        subdir = "mock" if args.mock else "real"
        tag = "smoke" if args.smoke_test else "full"
        args.output = os.path.join(
            _PROJECT_ROOT, "results", "benchmarks", "rfm", subdir,
            f"closed_loop_eval_{args.model}_{tag}.json"
        )

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2, cls=_NumpyEncoder)
    print(f"\nResults saved to: {args.output}")


if __name__ == "__main__":
    main()
