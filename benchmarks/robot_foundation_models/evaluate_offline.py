"""
Offline Action Evaluation Benchmark for Robot Foundation Models
==============================================================
Compares model-predicted actions to expert demonstrations generated
from the dual-cube PushCube environment.

Workflow
--------
1.  Load a model adapter (SmolVLA or OpenVLA) in real or mock mode.
2.  Generate expert demonstrations by rolling out ``expert_action(env)``
    in ``PushCubeEnv``.  Each sample stores the rendered image, 14-D
    state vector, language instruction, and the 2-D expert action.
3.  For every sample, query the model for an action prediction and
    compare it to the expert action.
4.  Report four metrics:

    * **Action MAE** — mean absolute error between prediction and expert.
    * **Action L2** — Euclidean distance between prediction and expert.
    * **Direction consistency** — fraction of samples where the predicted
      direction aligns with the expert (cosine similarity > 0).
    * **Inference latency** — mean and P99 wall-clock time per
      ``predict_action`` call.

Because PushCube uses a 2-D action space ``[dx, dy]`` while model adapters
may output higher-dimensional vectors (e.g. 14-D in mock mode to match
the state dimension), the comparison is performed on the first
``action_dim`` dimensions, consistent with the closed-loop rollout that
executes ``chunk.first_action()[:2]``.

Usage
-----
.. code-block:: bash

    # CI smoke test (mock mode, 2 episodes, no GPU needed)
    python benchmarks/robot_foundation_models/evaluate_offline.py --mock --smoke-test

    # Full run with SmolVLA in mock mode
    python benchmarks/robot_foundation_models/evaluate_offline.py --mock --n-episodes 50

    # Real model (requires GPU + model download)
    python benchmarks/robot_foundation_models/evaluate_offline.py --model smolvla --n-episodes 100

    # OpenVLA
    python benchmarks/robot_foundation_models/evaluate_offline.py --model openvla --device cuda
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
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
from examples.unified_pushcube_env import PushCubeEnv, expert_action

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
RESULTS_DIR = os.path.join(_PROJECT_ROOT, "results", "benchmarks", "rfm")
PUSHCUBE_ACTION_DIM = 2   # [dx, dy]
RENDER_SIZE = 128


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

    Parameters
    ----------
    model_name : str
        ``"smolvla"`` or ``"openvla"``.
    mock : bool
        Skip real model loading (for CI / CPU-only environments).
    device : str
        Torch device string.

    Returns
    -------
    adapter
        Object implementing the ``RobotFoundationModel`` protocol
        (``reset()`` and ``predict_action()``).
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
# Expert demonstration generation
# ---------------------------------------------------------------------------
def generate_expert_demonstrations(
    n_episodes: int,
    seed_offset: int = 0,
    render_size: int = RENDER_SIZE,
) -> List[dict]:
    """Roll out ``expert_action(env)`` and record per-step demonstrations.

    Each returned dict has keys:
        ``image``         — (render_size, render_size, 3) float32, range [0, 1]
        ``state``         — (14,) float32 state vector
        ``language``      — str, language instruction for the active cube
        ``expert_action`` — (2,) float32 expert action
        ``episode``       — int, episode index
        ``step``          — int, step within episode
    """
    demos: List[dict] = []
    for ep in range(n_episodes):
        env = PushCubeEnv()
        env.reset(seed=seed_offset + ep)
        lang = env.get_language_instruction()

        for step in range(env.max_steps):
            img = env.render(size=render_size)
            state = env.get_state_vector()
            action = expert_action(env)

            demos.append({
                "image": img,
                "state": state,
                "language": lang,
                "expert_action": action,
                "episode": ep,
                "step": step,
            })

            _, _, done, truncated, _ = env.step(action)
            if done or truncated:
                break

    return demos


# ---------------------------------------------------------------------------
# Offline evaluation
# ---------------------------------------------------------------------------
def run_offline_eval(
    adapter,
    demos: List[dict],
    action_dim: int = PUSHCUBE_ACTION_DIM,
) -> Dict:
    """Compare model predictions to expert actions.

    Metrics
    -------
    action_mae : float
        Mean absolute error averaged over all samples.
    action_l2 : float
        Mean L2 (Euclidean) distance averaged over all samples.
    direction_consistency : float
        Fraction of samples where cosine similarity between prediction
        and expert is positive (directions agree).  Samples where either
        vector has near-zero norm are excluded from the numerator but
        kept in the denominator.
    mean_latency_ms : float
        Mean wall-clock inference latency in milliseconds.
    p99_latency_ms : float
        99th-percentile latency in milliseconds.
    n_samples : int
        Total number of demonstration samples evaluated.
    """
    adapter.reset()

    all_mae: List[float] = []
    all_l2: List[float] = []
    direction_matches = 0
    direction_valid = 0
    latencies: List[float] = []
    per_episode_data: Dict[int, dict] = {}

    for sample in demos:
        obs = RobotObservation(
            images={"front": sample["image"]},
            state=sample["state"],
            language_instruction=sample["language"],
            timestamp=float(sample["step"]),
        )

        # Time the inference call
        t0 = time.perf_counter()
        chunk = adapter.predict_action(obs)
        latency = time.perf_counter() - t0
        latencies.append(latency)

        # Extract first action and align to PushCube action dimension.
        # Model adapters may output higher-dim vectors (e.g. 14-D in mock
        # mode); PushCube only uses the first 2 dims [dx, dy].
        pred_full = chunk.first_action()
        pred = pred_full[:action_dim]
        if len(pred) < action_dim:
            pred = np.pad(pred, (0, action_dim - len(pred)))
        expert = np.asarray(sample["expert_action"][:action_dim], dtype=np.float64)

        # Per-sample metrics
        mae = float(np.mean(np.abs(pred - expert)))
        l2 = float(np.linalg.norm(pred - expert))
        all_mae.append(mae)
        all_l2.append(l2)

        # Direction consistency (cosine similarity > 0)
        pred_norm = float(np.linalg.norm(pred))
        expert_norm = float(np.linalg.norm(expert))
        if pred_norm > 1e-6 and expert_norm > 1e-6:
            cos_sim = float(np.dot(pred, expert) / (pred_norm * expert_norm))
            direction_valid += 1
            if cos_sim > 0:
                direction_matches += 1

        # Per-episode accumulation
        ep = sample["episode"]
        if ep not in per_episode_data:
            per_episode_data[ep] = {"mae": [], "l2": [], "n_steps": 0}
        per_episode_data[ep]["mae"].append(mae)
        per_episode_data[ep]["l2"].append(l2)
        per_episode_data[ep]["n_steps"] += 1

    n = len(demos)
    per_episode_summary = [
        {
            "episode": ep,
            "n_steps": per_episode_data[ep]["n_steps"],
            "mae": float(np.mean(per_episode_data[ep]["mae"])),
            "l2": float(np.mean(per_episode_data[ep]["l2"])),
        }
        for ep in sorted(per_episode_data.keys())
    ]

    return {
        "action_mae": float(np.mean(all_mae)) if all_mae else 0.0,
        "action_l2": float(np.mean(all_l2)) if all_l2 else 0.0,
        "direction_consistency": direction_matches / max(1, n),
        "direction_valid_samples": direction_valid,
        "mean_latency_ms": float(np.mean(latencies) * 1000) if latencies else 0.0,
        "p99_latency_ms": float(np.percentile(latencies, 99) * 1000) if latencies else 0.0,
        "n_samples": n,
        "per_episode": per_episode_summary,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Offline action evaluation benchmark for robot foundation models"
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
        "--n-episodes", type=int, default=50,
        help="Number of expert demo episodes to generate (default: 50)",
    )
    parser.add_argument(
        "--device", default="cpu",
        help="Torch device for inference (default: cpu)",
    )
    parser.add_argument(
        "--seed-offset", type=int, default=0,
        help="Seed offset for episode generation (default: 0)",
    )
    parser.add_argument(
        "--output", default=None,
        help="Output JSON path (default: results/benchmarks/rfm/offline_eval_<model>_<tag>.json)",
    )
    args = parser.parse_args()

    # Smoke-test overrides
    if args.smoke_test:
        args.n_episodes = 2

    # Header
    print("=" * 72)
    print(" RFM Offline Action Evaluation Benchmark")
    print("=" * 72)
    print(f"  Model:      {args.model}")
    print(f"  Mock mode:  {args.mock}")
    print(f"  Smoke test: {args.smoke_test}")
    print(f"  Episodes:   {args.n_episodes}")
    print(f"  Device:     {args.device}")
    print()

    # 1. Load model
    print("[1/3] Loading model adapter...")
    adapter = load_model(args.model, args.mock, args.device)
    print(f"      {adapter}")

    # 2. Generate expert demonstrations
    print(f"\n[2/3] Generating {args.n_episodes} expert demo episodes...")
    demos = generate_expert_demonstrations(
        n_episodes=args.n_episodes,
        seed_offset=args.seed_offset,
    )
    print(f"      {len(demos)} samples collected")

    # 3. Run offline evaluation
    print(f"\n[3/3] Running offline evaluation ({len(demos)} samples)...")
    metrics = run_offline_eval(adapter, demos, action_dim=PUSHCUBE_ACTION_DIM)

    # Print results
    print("\n" + "-" * 72)
    print(" Offline Evaluation Results")
    print("-" * 72)
    print(f"  Action MAE:            {metrics['action_mae']:.6f}")
    print(f"  Action L2:             {metrics['action_l2']:.6f}")
    print(f"  Direction consistency: {metrics['direction_consistency']:.4f}"
          f"  ({metrics['direction_valid_samples']}/{metrics['n_samples']} valid)")
    print(f"  Mean latency:          {metrics['mean_latency_ms']:.3f} ms")
    print(f"  P99 latency:           {metrics['p99_latency_ms']:.3f} ms")
    print(f"  Total samples:         {metrics['n_samples']}")
    print("-" * 72)

    # Assemble output
    results = {
        "benchmark": "rfm_offline_evaluation",
        "model": {
            "name": args.model,
            "mock": args.mock,
            "device": args.device,
        },
        "config": {
            "n_episodes": args.n_episodes,
            "action_dim": PUSHCUBE_ACTION_DIM,
            "render_size": RENDER_SIZE,
            "seed_offset": args.seed_offset,
            "smoke_test": args.smoke_test,
        },
        "metrics": {k: v for k, v in metrics.items() if k != "per_episode"},
        "per_episode": metrics["per_episode"],
    }

    # Determine output path
    if args.output is None:
        tag = "smoke" if args.smoke_test else "full"
        args.output = os.path.join(
            RESULTS_DIR, f"offline_eval_{args.model}_{tag}.json"
        )

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2, cls=_NumpyEncoder)
    print(f"\nResults saved to: {args.output}")


if __name__ == "__main__":
    main()
