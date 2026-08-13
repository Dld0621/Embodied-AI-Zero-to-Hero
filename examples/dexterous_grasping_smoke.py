#!/usr/bin/env python3
"""Run a deterministic MuJoCo approach-contact-lift-hold grasp smoke test.

This is an intentionally small contact-dynamics fixture. It verifies that a
phase controller can establish multi-finger contact, lift an object, and retain
it during a bounded disturbance. It does not represent a learned policy,
in-hand reorientation, a production hand model, or real-robot validation.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "assets" / "simulation" / "dexterous_grasp_smoke.xml"
DEFAULT_OUTPUT = ROOT / "results" / "pipelines" / "dexterous_grasping" / "smoke" / "metrics.json"
FINGER_GEOMS = {
    "finger_left_tip",
    "finger_right_tip",
    "finger_front_tip",
    "finger_back_tip",
}


@dataclass(frozen=True)
class TrialConfig:
    """One deterministic contact-robustness condition."""

    name: str
    friction: float
    disturbance_n: float


TRIALS = (
    TrialConfig("nominal", friction=0.9, disturbance_n=0.08),
    TrialConfig("lower-friction", friction=0.65, disturbance_n=0.08),
    TrialConfig("stronger-disturbance", friction=0.9, disturbance_n=0.12),
)


def _id(model: mujoco.MjModel, object_type: mujoco.mjtObj, name: str) -> int:
    identifier = mujoco.mj_name2id(model, object_type, name)
    if identifier < 0:
        raise ValueError(f"MuJoCo object not found: {name}")
    return identifier


def _finger_contacts(model: mujoco.MjModel, data: mujoco.MjData, object_geom: int) -> set[str]:
    names: set[str] = set()
    for index in range(data.ncon):
        contact = data.contact[index]
        pair = {contact.geom1, contact.geom2}
        if object_geom not in pair:
            continue
        other = contact.geom2 if contact.geom1 == object_geom else contact.geom1
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, other)
        if name in FINGER_GEOMS:
            names.add(name)
    return names


def _object_penetration(model: mujoco.MjModel, data: mujoco.MjData, object_geom: int) -> float:
    penetration = 0.0
    for index in range(data.ncon):
        contact = data.contact[index]
        if object_geom in {contact.geom1, contact.geom2}:
            penetration = max(penetration, max(0.0, -float(contact.dist)))
    return penetration


def _step_phase(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    *,
    steps: int,
    wrist_target: float,
    closure_target: float,
    object_body: int,
    object_geom: int,
    disturbance_n: float = 0.0,
) -> dict[str, float]:
    max_contacts = 0
    max_penetration = 0.0
    min_object_height = float("inf")
    max_object_height = float("-inf")
    for step in range(steps):
        data.ctrl[0] = wrist_target
        data.ctrl[1:] = closure_target
        data.xfrc_applied[object_body] = 0.0
        if disturbance_n and steps // 4 <= step < steps // 2:
            data.xfrc_applied[object_body, 0] = disturbance_n
        mujoco.mj_step(model, data)
        max_contacts = max(max_contacts, len(_finger_contacts(model, data, object_geom)))
        max_penetration = max(max_penetration, _object_penetration(model, data, object_geom))
        object_height = float(data.xpos[object_body, 2])
        min_object_height = min(min_object_height, object_height)
        max_object_height = max(max_object_height, object_height)
    return {
        "max_finger_contacts": float(max_contacts),
        "max_penetration_m": max_penetration,
        "min_object_height_m": min_object_height,
        "max_object_height_m": max_object_height,
    }


def run_trial(config: TrialConfig) -> dict[str, Any]:
    """Execute one approach-close-lift-hold trial and return task evidence."""
    model = mujoco.MjModel.from_xml_path(str(MODEL_PATH))
    data = mujoco.MjData(model)
    object_body = _id(model, mujoco.mjtObj.mjOBJ_BODY, "object")
    object_geom = _id(model, mujoco.mjtObj.mjOBJ_GEOM, "object_geom")
    model.geom_friction[object_geom, 0] = config.friction

    data.qpos[0] = 0.08
    data.ctrl[0] = 0.08
    mujoco.mj_forward(model, data)
    initial_object_position = data.xpos[object_body].copy()

    phases = {
        "approach": _step_phase(
            model,
            data,
            steps=500,
            wrist_target=0.0,
            closure_target=0.0,
            object_body=object_body,
            object_geom=object_geom,
        ),
        "contact": _step_phase(
            model,
            data,
            steps=650,
            wrist_target=0.0,
            closure_target=0.029,
            object_body=object_body,
            object_geom=object_geom,
        ),
        "lift": _step_phase(
            model,
            data,
            steps=600,
            wrist_target=0.075,
            closure_target=0.029,
            object_body=object_body,
            object_geom=object_geom,
        ),
        "hold": _step_phase(
            model,
            data,
            steps=500,
            wrist_target=0.075,
            closure_target=0.029,
            object_body=object_body,
            object_geom=object_geom,
            disturbance_n=config.disturbance_n,
        ),
    }

    final_position = data.xpos[object_body].copy()
    lift_height = float(final_position[2] - initial_object_position[2])
    lateral_slip = float(np.linalg.norm(final_position[:2] - initial_object_position[:2]))
    final_contacts = len(_finger_contacts(model, data, object_geom))
    contact_established = phases["contact"]["max_finger_contacts"] >= 3
    retained = lift_height >= 0.045 and final_contacts >= 2
    passed = contact_established and retained and lateral_slip <= 0.02

    return {
        "name": config.name,
        "friction": config.friction,
        "disturbance_n": config.disturbance_n,
        "passed": passed,
        "contact_established": contact_established,
        "object_retained": retained,
        "final_finger_contacts": final_contacts,
        "lift_height_m": lift_height,
        "lateral_slip_m": lateral_slip,
        "max_penetration_m": max(phase["max_penetration_m"] for phase in phases.values()),
        "phases": phases,
    }


def run_demo() -> dict[str, Any]:
    """Run the reviewed robustness sweep and aggregate task-level metrics."""
    trials = [run_trial(config) for config in TRIALS]
    successes = sum(bool(trial["passed"]) for trial in trials)
    metrics = {
        "grasp_success_rate": successes / len(trials),
        "mean_lift_height_m": float(np.mean([trial["lift_height_m"] for trial in trials])),
        "max_lateral_slip_m": max(float(trial["lateral_slip_m"]) for trial in trials),
        "max_contact_penetration_m": max(float(trial["max_penetration_m"]) for trial in trials),
        "minimum_final_finger_contacts": min(int(trial["final_finger_contacts"]) for trial in trials),
    }
    thresholds = {
        "grasp_success_rate_min": 1.0,
        "mean_lift_height_m_min": 0.045,
        "max_lateral_slip_m_max": 0.02,
        "minimum_final_finger_contacts_min": 2,
    }
    passed = (
        metrics["grasp_success_rate"] >= thresholds["grasp_success_rate_min"]
        and metrics["mean_lift_height_m"] >= thresholds["mean_lift_height_m_min"]
        and metrics["max_lateral_slip_m"] <= thresholds["max_lateral_slip_m_max"]
        and metrics["minimum_final_finger_contacts"]
        >= thresholds["minimum_final_finger_contacts_min"]
    )
    return {
        "pipeline": "dexterous-manipulation",
        "passed": passed,
        "simulator": {
            "name": "MuJoCo",
            "version": mujoco.__version__,
            "model": MODEL_PATH.relative_to(ROOT).as_posix(),
        },
        "task": "approach-contact-lift-hold",
        "metrics": metrics,
        "metric_units": {
            "grasp_success_rate": "ratio",
            "mean_lift_height_m": "m",
            "max_lateral_slip_m": "m",
            "max_contact_penetration_m": "m",
            "minimum_final_finger_contacts": "count",
        },
        "thresholds": thresholds,
        "trials": trials,
        "evidence": {
            "level": "smoke",
            "qualifier": "synthetic-contact-dynamics",
            "supports": [
                "fixture_contact_establishment",
                "fixture_object_lift_and_retention",
                "deterministic_friction_and_disturbance_conditions",
            ],
            "does_not_support": [
                "in_hand_reorientation",
                "learned_policy_quality",
                "production_hand_transfer",
                "real_hardware_performance",
            ],
            "object_contact_evaluated": True,
            "object_retention_evaluated": True,
            "task_success_evaluated": True,
            "in_hand_reorientation_evaluated": False,
            "learned_policy_evaluated": False,
            "real_hardware_allowed": False,
        },
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="return non-zero when an acceptance gate fails")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="machine-readable JSON artifact")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    report = run_demo()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 1 if args.check and not report["passed"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
