"""
Strict Language Ablation for Robot Foundation Models
=====================================================
Evaluates a **single** trained model under five language conditions on
the **same** set of episodes (same seeds, same initial states).

This is the *proper* ablation design: rather than training separate
models on different language inputs (which confounds training data,
random initialisation, and optimisation trajectory), we load ONE model
and vary only the language string at inference time.  Any difference in
behaviour across conditions is therefore attributable to the language
signal alone.

The five conditions
-------------------
1. **correct**       — the true instruction for the active cube.
                       ``"push the red cube to the right"``
2. **swapped**       — the distractor cube's instruction (wrong colour).
                       ``"push the green cube to the right"``
3. **none**          — empty string (language dropout at inference).
                       ``""``
4. **paraphrased**   — synonym substitution, same meaning.
                       ``"move the red block toward the right"``
5. **contradictory** — direction words reversed, same object.
                       ``"push the red cube to the left"``

For each condition we record:
  * **correct_success**    — % of episodes where the active cube reaches
                             the target zone.
  * **wrong_success**      — % of episodes where the other cube reaches
                             the target zone.
  * **selection_accuracy** — % of episodes where the active cube is
                             closer to the target than the other cube.

Interpretation
--------------
A language-aware model should show:
  * High ``correct_success`` under **correct**.
  * Low ``correct_success`` (and high ``wrong_success``) under **swapped**.
  * Degraded ``selection_accuracy`` under **none**.
  * Performance similar to **correct** under **paraphrased** (robustness
    to surface form).
  * Poor performance under **contradictory** (the model follows the
    wrong direction, proving it reads direction words).

Usage
-----
.. code-block:: bash

    # CI smoke test (mock mode, 2 episodes, no GPU needed)
    python benchmarks/robot_foundation_models/language_ablation.py --mock --smoke-test

    # Full run
    python benchmarks/robot_foundation_models/language_ablation.py --mock --n-episodes 20

    # Real model
    python benchmarks/robot_foundation_models/language_ablation.py --model smolvla

    # Select specific conditions
    python benchmarks/robot_foundation_models/language_ablation.py --mock --conditions correct,contradictory
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
DEFAULT_SEED_OFFSET = 3000
ALL_CONDITIONS = (
    "correct",
    "swapped",
    "none",
    "paraphrased",
    "contradictory",
)

# Opposite direction words for the contradictory condition.
DIRECTION_OPPOSITES = {
    "right": "left",
    "left": "right",
    "top": "bottom",
    "bottom": "top",
    "center": "center",
}


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
    """Instantiate a robot foundation model adapter."""
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
# Language-condition generation
# ---------------------------------------------------------------------------
def paraphrase_instruction(text: str) -> str:
    """Generate a paraphrased version of the instruction.

    Applies synonym substitutions that preserve meaning:
        ``push``  -> ``move``
        ``cube``  -> ``block``
        `` to ``  -> `` toward ``

    Examples
    --------
    >>> paraphrase_instruction("push the red cube to the right")
    'move the red block toward the right'
    """
    result = text
    result = result.replace("push", "move")
    result = result.replace("cube", "block")
    result = result.replace(" to ", " toward ")
    return result


def contradict_instruction(text: str) -> str:
    """Generate a contradictory instruction by reversing direction words.

    Each directional word is replaced with its opposite, keeping the
    object (colour) the same.  This creates an instruction that points
    the model to the wrong spatial goal.

    Examples
    --------
    >>> contradict_instruction("push the red cube to the right")
    'push the red cube to the left'
    >>> contradict_instruction("push the red cube to the right and top")
    'push the red cube to the left and bottom'
    """
    words = text.split()
    new_words = [DIRECTION_OPPOSITES.get(w, w) for w in words]
    return " ".join(new_words)


def get_language_for_condition(env: PushCubeEnv, condition: str) -> str:
    """Return the language instruction for a given ablation condition.

    Parameters
    ----------
    env : PushCubeEnv
        The environment (must have been reset so that the active cube
        and colours are initialised).
    condition : str
        One of the five ablation conditions.

    Returns
    -------
    str
        The language string to feed to the model.
    """
    correct_lang = env.get_language_instruction()
    if condition == "correct":
        return correct_lang
    elif condition == "swapped":
        return env.get_shuffled_language()
    elif condition == "none":
        return ""
    elif condition == "paraphrased":
        return paraphrase_instruction(correct_lang)
    elif condition == "contradictory":
        return contradict_instruction(correct_lang)
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
    """Execute a single closed-loop episode and return outcome metrics.

    The adapter is reset at the start of the episode.  At each step the
    model receives the rendered image, 14-D state, and the language
    string for this condition.  The predicted action must be 2-D
    ``[dx, dy]`` for PushCube.
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
    all five conditions see the **same** initial states — this is the key
    property of the strict ablation design.
    """
    results: List[dict] = []
    for ep in range(n_episodes):
        env = PushCubeEnv()
        env.reset(seed=seed_offset + ep)
        language = get_language_for_condition(env, condition)
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
# Analysis helpers
# ---------------------------------------------------------------------------
def compute_analysis(condition_results: Dict[str, Dict]) -> Dict:
    """Compute summary statistics across ablation conditions.

    Returns
    -------
    dict with:
        * ``language_sensitivity`` — absolute difference in correct_success
          between *correct* and *none* conditions (higher = more language-
          dependent).
        * ``swap_effect`` — difference in wrong_success between *swapped*
          and *correct* (positive = model follows the swapped instruction).
        * ``paraphrase_robustness`` — difference in correct_success between
          *correct* and *paraphrased* (near-zero = robust to paraphrase).
        * ``contradiction_effect`` — difference in correct_success between
          *correct* and *contradictory* (large positive = model follows
          direction words).
    """
    def _get(cond, key, default=0.0):
        return condition_results.get(cond, {}).get(key, default)

    correct_cs = _get("correct", "correct_success")
    none_cs = _get("none", "correct_success")
    swapped_ws = _get("swapped", "wrong_success")
    correct_ws = _get("correct", "wrong_success")
    para_cs = _get("paraphrased", "correct_success")
    contra_cs = _get("contradictory", "correct_success")

    return {
        "language_sensitivity": round(abs(correct_cs - none_cs), 1),
        "swap_effect": round(swapped_ws - correct_ws, 1),
        "paraphrase_robustness": round(abs(correct_cs - para_cs), 1),
        "contradiction_effect": round(correct_cs - contra_cs, 1),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Strict language ablation for robot foundation models"
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
        "--conditions",
        default="correct,swapped,none,paraphrased,contradictory",
        help="Comma-separated conditions (default: all five)",
    )
    parser.add_argument(
        "--output", default=None,
        help="Output JSON path (default: results/benchmarks/rfm/language_ablation_<model>_<tag>.json)",
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
    print(" RFM Strict Language Ablation")
    print("=" * 72)
    print(f"  Model:      {args.model}")
    print(f"  Mock mode:  {args.mock}")
    print(f"  Smoke test: {args.smoke_test}")
    print(f"  Episodes:   {args.n_episodes} per condition")
    print(f"  Conditions: {conditions}")
    print(f"  Device:     {args.device}")
    print()
    print("  Design: ONE model, SAME episodes, varied language at inference.")
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

    # Summary table
    print("\n" + "-" * 72)
    print(" Language Ablation Summary")
    print("-" * 72)
    header = (
        f"  {'Condition':<16} {'correct%':>9} {'wrong%':>9} {'select%':>9}"
    )
    print(header)
    print("  " + "-" * 48)
    for cond in conditions:
        r = condition_results[cond]
        print(
            f"  {cond:<16} {r['correct_success']:>9.1f} "
            f"{r['wrong_success']:>9.1f} {r['selection_accuracy']:>9.1f}"
        )
    print("-" * 72)

    # Analysis
    analysis = compute_analysis(condition_results)
    print("\n  Analysis:")
    print(f"    language_sensitivity     = {analysis['language_sensitivity']:.1f}"
          f"  (correct_success gap: correct vs none)")
    print(f"    swap_effect              = {analysis['swap_effect']:.1f}"
          f"  (wrong_success gap: swapped vs correct)")
    print(f"    paraphrase_robustness    = {analysis['paraphrase_robustness']:.1f}"
          f"  (correct_success gap: correct vs paraphrased)")
    print(f"    contradiction_effect     = {analysis['contradiction_effect']:.1f}"
          f"  (correct_success gap: correct vs contradictory)")
    print("-" * 72)

    # Show example instructions for each condition (from the first episode)
    print("\n  Example instructions (episode 0):")
    demo_env = PushCubeEnv()
    demo_env.reset(seed=args.seed_offset)
    for cond in conditions:
        lang = get_language_for_condition(demo_env, cond)
        display = lang if lang else "(empty)"
        print(f"    {cond:<16} -> \"{display}\"")
    print("-" * 72)

    # Assemble output
    results = {
        "benchmark": "rfm_language_ablation",
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
        "ablation_design": (
            "Single model evaluated under five language conditions on "
            "identical episodes (same seeds, same initial states). "
            "This isolates the effect of language at inference time "
            "without confounding from separate training runs."
        ),
        "condition_descriptions": {
            "correct": "True instruction for the active cube.",
            "swapped": "Distractor cube's instruction (wrong colour word).",
            "none": "Empty string (language dropout at inference).",
            "paraphrased": "Synonym substitution preserving meaning.",
            "contradictory": "Direction words reversed (same object, wrong goal).",
        },
        "results": condition_results,
        "analysis": analysis,
    }

    # Determine output path — mock and real results are stored separately
    if args.output is None:
        subdir = "mock" if args.mock else "real"
        tag = "smoke" if args.smoke_test else "full"
        args.output = os.path.join(
            _PROJECT_ROOT, "results", "benchmarks", "rfm", subdir,
            f"language_ablation_{args.model}_{tag}.json"
        )

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2, cls=_NumpyEncoder)
    print(f"\nResults saved to: {args.output}")


if __name__ == "__main__":
    main()
