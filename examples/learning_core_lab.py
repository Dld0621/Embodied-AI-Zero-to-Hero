"""Small, deterministic NumPy teaching experiments; no hardware or network.

Run from the repository root: python examples/learning_core_lab.py --json
These experiments demonstrate mechanisms, not robot performance.
"""

from __future__ import annotations

import argparse
import json

import numpy as np


def fit_standardizer(train: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Fit only the caller's training partition; constant features use scale 1."""
    train = np.asarray(train, dtype=np.float64)
    if train.ndim != 2 or not train.size or not np.isfinite(train).all():
        raise ValueError("train must be a nonempty finite samples-by-features matrix")
    mean, scale = train.mean(axis=0), train.std(axis=0)
    return mean, np.where(scale > 0, scale, 1.0)


def make_parameters(seed: int = 7, width: int = 8) -> dict[str, np.ndarray]:
    if width < 1:
        raise ValueError("width must be positive")
    rng = np.random.default_rng(seed)
    return {
        "w1": rng.normal(0, 0.4, (1, width)),
        "b1": np.zeros(width),
        "w2": rng.normal(0, 0.4, (width, 1)),
        "b2": np.zeros(1),
    }


def loss_and_gradients(
    params: dict[str, np.ndarray], x: np.ndarray, y: np.ndarray
) -> tuple[float, dict[str, np.ndarray]]:
    """MSE averaged over all outputs, with a manually differentiated tanh MLP."""
    if x.ndim != 2 or y.shape != (len(x), 1) or x.shape[1] != 1 or not len(x):
        raise ValueError("expected nonempty x and y with shape (batch, 1)")
    if not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError("inputs must be finite")
    h = np.tanh(x @ params["w1"] + params["b1"])
    error = h @ params["w2"] + params["b2"] - y
    output_grad = 2 * error / error.size
    hidden_grad = (output_grad @ params["w2"].T) * (1 - h * h)
    grads = {
        "w1": x.T @ hidden_grad,
        "b1": hidden_grad.sum(axis=0),
        "w2": h.T @ output_grad,
        "b2": output_grad.sum(axis=0),
    }
    return float(np.mean(error * error)), grads


def gradient_check(params, x, y, epsilon: float = 1e-5) -> float:
    """Return maximum absolute analytic/central-difference discrepancy."""
    if not np.isfinite(epsilon) or epsilon <= 0:
        raise ValueError("epsilon must be positive and finite")
    _, grads = loss_and_gradients(params, x, y)
    worst = 0.0
    for name, values in params.items():
        for index in np.ndindex(values.shape):
            original = values[index]
            try:
                values[index] = original + epsilon
                plus, _ = loss_and_gradients(params, x, y)
                values[index] = original - epsilon
                minus, _ = loss_and_gradients(params, x, y)
            finally:
                values[index] = original
            numerical = (plus - minus) / (2 * epsilon)
            worst = max(worst, abs(numerical - grads[name][index]))
    return float(worst)


def data_experiment() -> dict:
    # Split episode identities before extracting frames/windows.
    partitions = {"train": [0, 1], "validation": [2], "test": [3]}
    mean, scale = fit_standardizer(np.array([[0.0], [2.0]]))
    held_out = (np.array([[100.0]]) - mean) / scale
    return {
        "episode_ids": partitions,
        "train_mean": mean.tolist(),
        "train_scale": scale.tolist(),
        "held_out_transformed": held_out.tolist(),
        "scope": "Illustrates train-only preprocessing, not new-object generalization.",
    }


def gradient_experiment(seed: int = 7, steps: int = 800) -> dict:
    if steps < 1:
        raise ValueError("steps must be positive")
    params = make_parameters(seed)
    train_x = np.linspace(-1.0, 1.0, 41).reshape(-1, 1)
    train_y = np.sin(2 * train_x)
    # Separate interpolation points; this is not an extrapolation test.
    test_x = ((train_x[:-1] + train_x[1:]) / 2).copy()
    test_y = np.sin(2 * test_x)
    initial, _ = loss_and_gradients(params, train_x, train_y)
    discrepancy = gradient_check(params, train_x[:5], train_y[:5])
    for _ in range(steps):
        _, grads = loss_and_gradients(params, train_x, train_y)
        for name in params:
            params[name] -= 0.1 * grads[name]
    final, _ = loss_and_gradients(params, train_x, train_y)
    test_loss, _ = loss_and_gradients(params, test_x, test_y)
    return {
        "seed": seed,
        "steps": steps,
        "learning_rate": 0.1,
        "max_gradient_absolute_error": discrepancy,
        "initial_train_mse": initial,
        "final_train_mse": final,
        "held_out_interpolation_mse": test_loss,
        "scope": "One fixed function and seed; no robot or out-of-range performance claim.",
    }


def fit_feedback(states: np.ndarray, actions: np.ndarray) -> float:
    """Least-squares a = gain*x; minimum-norm gain 0 for all-zero inputs."""
    states, actions = np.asarray(states), np.asarray(actions)
    if states.ndim != 1 or actions.shape != states.shape or not states.size:
        raise ValueError("states and actions must be equal nonempty vectors")
    if not np.isfinite(states).all() or not np.isfinite(actions).all():
        raise ValueError("states and actions must be finite")
    denominator = float(states @ states)
    return float(states @ actions / denominator) if denominator > 0 else 0.0


def rollout(gain: float, disturbance: float = 0.6, steps: int = 10) -> dict:
    """At step 4 disturb state BEFORE observing it; dynamics x_next = x + a."""
    if steps < 5 or not np.isfinite([gain, disturbance]).all():
        raise ValueError("finite gain/disturbance and at least five steps required")
    state = 0.0
    observed, actions, next_states = [], [], []
    for step in range(steps):
        if step == 4:
            state += disturbance
        observed.append(state)
        action = gain * state
        actions.append(action)
        state += action
        next_states.append(state)
    return {
        "observed_states": observed,
        "actions": actions,
        "next_states": next_states,
        "final_absolute_error": abs(state),
    }


def imitation_experiment() -> dict:
    demo_states = np.zeros(8)
    demo_actions = np.zeros(8)
    bc_gain = fit_feedback(demo_states, demo_actions)
    baseline = rollout(bc_gain)
    visited = np.asarray(baseline["observed_states"])
    expert_labels = -visited
    gain = fit_feedback(
        np.concatenate([demo_states, visited]),
        np.concatenate([demo_actions, expert_labels]),
    )
    return {
        "bc_gain": bc_gain,
        "aggregated_gain": gain,
        "bc": baseline,
        "after_expert_query": rollout(gain),
        "scope": "One-round DAgger-style illustration with exact expert, not full DAgger.",
    }


def multimodal_experiment() -> dict:
    # One-step abstract corridor: abs(action) < 0.5 is blocked.
    expert_actions = np.array([-1.0, 1.0])
    mean = float(expert_actions.mean())
    return {
        "expert_modes": expert_actions.tolist(),
        "mse_optimal_constant": mean,
        "mean_action_blocked": bool(abs(mean) < 0.5),
        "expert_actions_blocked": (np.abs(expert_actions) < 0.5).tolist(),
        "scope": "Abstract collision rule; no generative policy is trained here.",
    }


def run_all() -> dict:
    return {
        "data": data_experiment(),
        "gradients": gradient_experiment(),
        "imitation": imitation_experiment(),
        "multimodal": multimodal_experiment(),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", action="store_true", help="print machine-readable results")
    args = parser.parse_args()
    results = run_all()
    if not args.json:
        print("Offline teaching experiments — numerical mechanisms, not robot validation.")
    print(json.dumps(results, ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
