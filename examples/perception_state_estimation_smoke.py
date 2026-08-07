#!/usr/bin/env python3
"""Deterministic teaching smoke test for perception and state estimation.

The demo checks a small pinhole-calibration fixture, timestamp health, and a
one-dimensional constant-velocity Kalman filter. It proves that the repository
contract is executable; it is not a camera, SLAM, or real-robot benchmark.
"""

from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path
from typing import Any


DEFAULT_OUTPUT = Path("results/pipelines/perception_state/smoke/metrics.json")


def _project(
    point: tuple[float, float, float],
    intrinsics: tuple[float, float, float, float],
) -> tuple[float, float]:
    x, y, z = point
    fx, fy, cx, cy = intrinsics
    if z <= 0.0:
        raise ValueError("calibration points must be in front of the camera")
    return fx * x / z + cx, fy * y / z + cy


def calibration_reprojection_rmse_px() -> float:
    """Return reprojection RMSE for a fixed, slightly perturbed calibration."""

    points = (
        (-0.18, -0.12, 0.8),
        (0.20, -0.10, 0.9),
        (-0.16, 0.14, 1.1),
        (0.22, 0.16, 1.0),
        (0.02, -0.04, 1.3),
        (-0.08, 0.06, 1.5),
    )
    reference = (520.0, 518.0, 320.0, 240.0)
    candidate = (519.0, 518.8, 320.15, 239.9)
    squared_errors = []
    for point in points:
        ref_u, ref_v = _project(point, reference)
        est_u, est_v = _project(point, candidate)
        squared_errors.append((ref_u - est_u) ** 2 + (ref_v - est_v) ** 2)
    return math.sqrt(sum(squared_errors) / len(squared_errors))


def _percentile(values: list[float], fraction: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, math.ceil(fraction * len(ordered)) - 1)
    return ordered[max(index, 0)]


def run_demo(seed: int = 7, steps: int = 120, dt_s: float = 0.05) -> dict[str, Any]:
    if steps < 20:
        raise ValueError("steps must be at least 20")
    if dt_s <= 0.0:
        raise ValueError("dt_s must be positive")

    rng = random.Random(seed)
    position = 0.0
    velocity = 0.0
    covariance = [[1.0, 0.0], [0.0, 1.0]]
    position_errors: list[float] = []
    covered_by_two_sigma = 0
    sync_skews_ms: list[float] = []
    stale_messages = 0
    received_messages = 0

    for step in range(steps):
        time_s = step * dt_s
        true_position = 0.22 * time_s + 0.08 * math.sin(0.7 * time_s)

        # Constant-velocity prediction with a small acceleration-noise model.
        position += velocity * dt_s
        p00, p01 = covariance[0]
        p10, p11 = covariance[1]
        q = 0.18
        covariance = [
            [
                p00 + dt_s * (p10 + p01) + dt_s * dt_s * p11 + q * dt_s**4 / 4.0,
                p01 + dt_s * p11 + q * dt_s**3 / 2.0,
            ],
            [
                p10 + dt_s * p11 + q * dt_s**3 / 2.0,
                p11 + q * dt_s**2,
            ],
        ]

        camera_timestamp = time_s + rng.uniform(-0.004, 0.004)
        proprio_timestamp = time_s + rng.uniform(-0.002, 0.002)
        if step > 0 and step % 23 == 0:
            camera_timestamp -= 0.12  # deterministic delayed packet

        camera = None if step > 0 and step % 19 == 0 else true_position + rng.gauss(0.0, 0.055)
        proprio = None if step > 0 and step % 31 == 0 else true_position + rng.gauss(0.0, 0.025)

        valid_measurements: list[tuple[float, float]] = []
        for measurement, timestamp, variance in (
            (camera, camera_timestamp, 0.055**2),
            (proprio, proprio_timestamp, 0.025**2),
        ):
            if measurement is None:
                continue
            received_messages += 1
            if time_s - timestamp > 0.08:
                stale_messages += 1
                continue
            valid_measurements.append((measurement, variance))

        if camera is not None and proprio is not None:
            sync_skews_ms.append(abs(camera_timestamp - proprio_timestamp) * 1000.0)

        for measurement, variance in valid_measurements:
            p00, p01 = covariance[0]
            p10, p11 = covariance[1]
            innovation_variance = p00 + variance
            gain_position = p00 / innovation_variance
            gain_velocity = p10 / innovation_variance
            innovation = measurement - position
            position += gain_position * innovation
            velocity += gain_velocity * innovation
            covariance = [
                [(1.0 - gain_position) * p00, (1.0 - gain_position) * p01],
                [p10 - gain_velocity * p00, p11 - gain_velocity * p01],
            ]

        error = position - true_position
        position_errors.append(error)
        sigma = math.sqrt(max(covariance[0][0], 0.0))
        if abs(error) <= 2.0 * sigma:
            covered_by_two_sigma += 1

    coverage = covered_by_two_sigma / steps
    metrics = {
        "calibration_reprojection_error_px": calibration_reprojection_rmse_px(),
        "sensor_sync_skew_p95_ms": _percentile(sync_skews_ms, 0.95),
        "state_estimation_rmse": math.sqrt(
            sum(error * error for error in position_errors) / len(position_errors)
        ),
        "uncertainty_coverage_2sigma": coverage,
        "uncertainty_calibration_error": abs(coverage - 0.9545),
        "stale_observation_rate": stale_messages / max(received_messages, 1),
    }
    checks = {
        "reprojection_error_below_1px": metrics["calibration_reprojection_error_px"] < 1.0,
        "sync_skew_p95_below_15ms": metrics["sensor_sync_skew_p95_ms"] < 15.0,
        "state_rmse_below_0_10": metrics["state_estimation_rmse"] < 0.10,
        "two_sigma_coverage_above_0_80": metrics["uncertainty_coverage_2sigma"] >= 0.80,
        "stale_rate_below_0_10": metrics["stale_observation_rate"] < 0.10,
    }
    return {
        "schema_version": 1,
        "pipeline": "perception-state-estimation",
        "evidence": {
            "level": "synthetic-smoke",
            "supports": "calibration, timestamp-health, scalar-fusion, and uncertainty wiring",
            "does_not_support": "camera, SLAM, task-level, or real-robot performance claims",
        },
        "configuration": {"seed": seed, "steps": steps, "dt_s": dt_s},
        "metrics": metrics,
        "metric_units": {
            "calibration_reprojection_error_px": "px_rmse",
            "sensor_sync_skew_p95_ms": "ms_p95",
            "state_estimation_rmse": "m_rmse",
            "uncertainty_coverage_2sigma": "fraction_of_steps",
            "uncertainty_calibration_error": "absolute_fraction_gap",
            "stale_observation_rate": "stale_over_received",
        },
        "checks": checks,
        "passed": all(checks.values()),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--steps", type=int, default=120)
    parser.add_argument("--dt", type=float, default=0.05, dest="dt_s")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--check", action="store_true", help="return non-zero when a gate fails")
    args = parser.parse_args()

    report = run_demo(seed=args.seed, steps=args.steps, dt_s=args.dt_s)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    print(f"Wrote {args.output}")
    return 1 if args.check and not report["passed"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
