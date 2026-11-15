"""CPU-only VLA contract exercises; NumPy, synthetic data, no robot or pretrained model.

Run: python examples/vla_contract_lab.py --json
The goal identifier is hand-coded, not a learned language representation. A scalar
scene coordinate stands in for perception. These results do not measure VLA quality.
"""
from __future__ import annotations

import argparse
import json

import numpy as np


def _bounds(lower: float, upper: float) -> None:
    if not np.isfinite([lower, upper]).all() or not lower < upper:
        raise ValueError("Require finite lower < upper")


def denormalize(value, lower: float, upper: float):
    """Affine [-1, 1] mapping; reject out-of-contract values rather than clip silently."""
    _bounds(lower, upper)
    array = np.asarray(value, dtype=float)
    if not np.isfinite(array).all() or np.any(np.abs(array) > 1):
        raise ValueError("Normalized values must be finite and in [-1, 1]")
    return lower + (array + 1) * 0.5 * (upper - lower)


def quantize_centers(value, bins: int = 256):
    """Uniform bin-center teaching codec, not OpenVLA's exact tokenizer/quantiles."""
    if isinstance(bins, bool) or not isinstance(bins, int) or bins < 2:
        raise ValueError("bins must be an integer >= 2")
    array = np.asarray(value, dtype=float)
    if not np.isfinite(array).all() or np.any(np.abs(array) > 1):
        raise ValueError("Values must be finite and in [-1, 1]")
    indices = np.minimum(np.floor((array + 1) * bins / 2).astype(int), bins - 1)
    decoded = -1 + (indices + 0.5) * 2 / bins
    return indices, decoded


def action_window(actions, start: int, horizon: int):
    """Slice ONE episode; zero-pad and return a loss mask for missing future steps."""
    array = np.asarray(actions, dtype=float)
    if array.ndim != 2 or array.shape[0] == 0 or array.shape[1] == 0 or not np.isfinite(array).all():
        raise ValueError("actions must be a nonempty finite [time, dimension] array")
    if any(isinstance(v, bool) or not isinstance(v, int) for v in (start, horizon)):
        raise ValueError("start and horizon must be integers")
    if not 0 <= start < len(array) or horizon < 1:
        raise ValueError("Invalid start or horizon")
    count = min(horizon, len(array) - start)
    chunk = np.zeros((horizon, array.shape[1]))
    chunk[:count] = array[start:start + count]
    mask = np.arange(horizon) < count
    return chunk, mask


def language_experiment():
    """Fit linear BC on a balanced, hand-coded goal task; test on new positions."""
    positions = np.tile(np.array([-0.4, -0.1, 0.1, 0.4]), 2)
    goals = np.repeat([-1.0, 1.0], 4)
    target = 0.5 * goals - positions  # one-step ideal target-reaching displacement
    features = np.column_stack([np.ones(len(positions)), positions, goals])
    weights = np.linalg.lstsq(features, target, rcond=None)[0]
    visual_weights = np.linalg.lstsq(features[:, :2], target, rcond=None)[0]
    test_positions = np.tile(np.array([-0.25, 0.25]), 2)
    test_goals = np.repeat([-1.0, 1.0], 2)
    expected = 0.5 * test_goals - test_positions

    def measure(goals_in, visual_only=False):
        design = np.column_stack([np.ones(4), test_positions, goals_in])
        prediction = design[:, :2] @ visual_weights if visual_only else design @ weights
        return {
            "predictions": prediction.tolist(),
            "mse_against_original_goal": float(np.mean((prediction - expected) ** 2)),
            "within_0_05_m": int(np.sum(np.abs(prediction - expected) < 0.05)),
            "trials": len(expected),
        }

    return {
        "evidence": "synthetic linear fit; hand-coded goal ID and scalar scene; not a VLA benchmark",
        "weights_bias_position_goal": weights.tolist(),
        "correct": measure(test_goals),
        "swapped": measure(-test_goals),
        "blank": measure(np.zeros(4)),
        "visual_only": measure(test_goals, visual_only=True),
    }


def run_experiments():
    _, decoded = quantize_centers(np.linspace(-1, 1, 1001))
    physical = denormalize(decoded, -0.02, 0.02)
    original = denormalize(np.linspace(-1, 1, 1001), -0.02, 0.02)
    chunk, mask = action_window([[0.01], [0.02], [0.03]], start=1, horizon=4)
    dt, latency, execute = 0.05, 0.1, 4
    cycle = latency + execute * dt
    return {
        "evidence_level": "teaching",
        "contract": {
            "normalized_value": 0.5,
            "base_frame_delta_m": float(denormalize(0.5, -0.02, 0.02)),
            "other_training_bounds_delta_m": float(denormalize(0.5, -0.1, 0.1)),
            "uniform_256_bin_max_error_m": float(np.max(np.abs(physical - original))),
        },
        "chunk_mask": {"actions": chunk.tolist(), "valid": mask.tolist(), "valid_steps": int(mask.sum())},
        "serial_timing": {
            "cycle_s": cycle,
            "replans_per_s": 1 / cycle,
            "commands_per_s": execute / cycle,
            "last_command_start_age_s": latency + (execute - 1) * dt,
        },
        "language": language_experiment(),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", action="store_true", help="Print reproducible numerical exercises")
    args = parser.parse_args()
    result = run_experiments()
    print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))
    if not args.json:
        print("Interpretation: changing the instruction changes the target; swapped/blank inputs are scored against the ORIGINAL task.")
        print("Synthetic one-step fitting is not closed-loop robot evaluation. See docs/01-what-is-vla.md.")


if __name__ == "__main__":
    main()
