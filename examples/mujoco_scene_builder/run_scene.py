#!/usr/bin/env python3
"""Load, inspect, simulate, visualize, render, and export a modular MJCF scene."""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import Any

import mujoco
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
SCENE_PATH = HERE / "scene.xml"
DEFAULT_OUTPUT = ROOT / "results" / "tutorials" / "mujoco_scene_builder" / "report.json"

REQUIRED_NAMES = {
    mujoco.mjtObj.mjOBJ_BODY: ("arm_base", "link1", "link2", "tool", "object"),
    mujoco.mjtObj.mjOBJ_JOINT: ("shoulder", "elbow", "object_freejoint"),
    mujoco.mjtObj.mjOBJ_ACTUATOR: ("shoulder_position", "elbow_position"),
    mujoco.mjtObj.mjOBJ_SITE: ("end_effector", "task_target"),
    mujoco.mjtObj.mjOBJ_CAMERA: ("overview",),
}


def _identifier(model: mujoco.MjModel, object_type: mujoco.mjtObj, name: str) -> int:
    identifier = mujoco.mj_name2id(model, object_type, name)
    if identifier < 0:
        raise ValueError(f"Required MuJoCo element is missing: {name}")
    return identifier


def _named_elements(model: mujoco.MjModel, object_type: mujoco.mjtObj, count: int) -> list[str]:
    return [
        name
        for index in range(count)
        if (name := mujoco.mj_id2name(model, object_type, index)) is not None
    ]


def inspect_model(model: mujoco.MjModel) -> dict[str, Any]:
    """Return dimensions and stable name-based handles for a compiled model."""
    missing: list[str] = []
    resolved: dict[str, int] = {}
    for object_type, names in REQUIRED_NAMES.items():
        for name in names:
            identifier = mujoco.mj_name2id(model, object_type, name)
            if identifier < 0:
                missing.append(name)
            else:
                resolved[name] = identifier

    return {
        "dimensions": {
            "nq": model.nq,
            "nv": model.nv,
            "nu": model.nu,
            "nbody": model.nbody,
            "njnt": model.njnt,
            "ngeom": model.ngeom,
            "nsite": model.nsite,
            "nsensor": model.nsensor,
            "nsensordata": model.nsensordata,
        },
        "names": {
            "bodies": _named_elements(model, mujoco.mjtObj.mjOBJ_BODY, model.nbody),
            "joints": _named_elements(model, mujoco.mjtObj.mjOBJ_JOINT, model.njnt),
            "actuators": _named_elements(model, mujoco.mjtObj.mjOBJ_ACTUATOR, model.nu),
            "sites": _named_elements(model, mujoco.mjtObj.mjOBJ_SITE, model.nsite),
            "sensors": _named_elements(model, mujoco.mjtObj.mjOBJ_SENSOR, model.nsensor),
        },
        "resolved_ids": resolved,
        "missing_required_names": missing,
    }


def _joint_position(model: mujoco.MjModel, data: mujoco.MjData, name: str) -> float:
    joint_id = _identifier(model, mujoco.mjtObj.mjOBJ_JOINT, name)
    return float(data.qpos[model.jnt_qposadr[joint_id]])


def run_scene(*, steps: int = 1200, use_viewer: bool = False) -> tuple[mujoco.MjModel, mujoco.MjData, dict[str, Any]]:
    """Run a deterministic control sweep and return a machine-readable report."""
    if steps <= 0:
        raise ValueError("steps must be positive")

    model = mujoco.MjModel.from_xml_path(str(SCENE_PATH))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    inspection = inspect_model(model)

    shoulder_actuator = _identifier(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "shoulder_position")
    elbow_actuator = _identifier(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "elbow_position")
    object_body = _identifier(model, mujoco.mjtObj.mjOBJ_BODY, "object")
    initial_object_position = data.xpos[object_body].copy()

    max_contacts = 0
    max_joint_limit_violation = 0.0
    states_are_finite = True

    def advance() -> None:
        nonlocal max_contacts, max_joint_limit_violation, states_are_finite
        phase = data.time
        data.ctrl[shoulder_actuator] = 0.55 * math.sin(0.9 * phase)
        data.ctrl[elbow_actuator] = -0.55 + 0.35 * math.cos(1.1 * phase)
        mujoco.mj_step(model, data)

        max_contacts = max(max_contacts, data.ncon)
        states_are_finite = states_are_finite and bool(
            np.isfinite(data.qpos).all() and np.isfinite(data.qvel).all() and np.isfinite(data.sensordata).all()
        )
        for joint_name in ("shoulder", "elbow"):
            joint_id = _identifier(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
            position = _joint_position(model, data, joint_name)
            lower, upper = model.jnt_range[joint_id]
            max_joint_limit_violation = max(
                max_joint_limit_violation,
                float(max(lower - position, position - upper, 0.0)),
            )

    if use_viewer:
        from mujoco import viewer as mujoco_viewer

        with mujoco_viewer.launch_passive(model, data) as viewer:
            for _ in range(steps):
                step_start = time.perf_counter()
                advance()
                viewer.sync()
                remaining = model.opt.timestep - (time.perf_counter() - step_start)
                if remaining > 0:
                    time.sleep(remaining)
    else:
        for _ in range(steps):
            advance()

    final_object_position = data.xpos[object_body].copy()
    report = {
        "tutorial": "mujoco-scene-builder",
        "passed": bool(
            not inspection["missing_required_names"]
            and states_are_finite
            and max_joint_limit_violation <= 1e-9
        ),
        "simulator": {
            "name": "MuJoCo",
            "version": mujoco.__version__,
            "model": SCENE_PATH.relative_to(ROOT).as_posix(),
        },
        "inspection": inspection,
        "simulation": {
            "steps": steps,
            "simulated_time_s": float(data.time),
            "states_are_finite": states_are_finite,
            "max_contacts": max_contacts,
            "max_joint_limit_violation_rad": max_joint_limit_violation,
            "final_joint_positions_rad": {
                "shoulder": _joint_position(model, data, "shoulder"),
                "elbow": _joint_position(model, data, "elbow"),
            },
            "object_displacement_m": float(np.linalg.norm(final_object_position - initial_object_position)),
        },
        "evidence": {
            "level": "smoke",
            "supports": [
                "modular_mjcf_compiles",
                "named_elements_resolve",
                "actuated_scene_steps_with_finite_state",
            ],
            "does_not_support": [
                "task_policy_quality",
                "system_identification_accuracy",
                "real_robot_transfer",
            ],
        },
    }
    return model, data, report


def render_frame(model: mujoco.MjModel, data: mujoco.MjData, output: Path) -> None:
    """Render the named overview camera to a PNG using an optional Matplotlib dependency."""
    import matplotlib.pyplot as plt

    output.parent.mkdir(parents=True, exist_ok=True)
    with mujoco.Renderer(model, height=720, width=960) as renderer:
        renderer.update_scene(data, camera="overview")
        image = renderer.render()
    plt.imsave(output, image)


def export_model(model: mujoco.MjModel, *, canonical_xml: Path | None, binary_mjb: Path | None) -> None:
    """Export canonical MJCF and/or a version-specific compiled MJB."""
    if canonical_xml is not None:
        canonical_xml.parent.mkdir(parents=True, exist_ok=True)
        mujoco.mj_saveLastXML(str(canonical_xml), model)
    if binary_mjb is not None:
        binary_mjb.parent.mkdir(parents=True, exist_ok=True)
        mujoco.mj_saveModel(model, str(binary_mjb))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=1200, help="number of physics steps")
    parser.add_argument("--viewer", action="store_true", help="open the interactive passive viewer")
    parser.add_argument("--render", type=Path, help="optional overview PNG path")
    parser.add_argument("--save-canonical", type=Path, help="optional flattened canonical MJCF path")
    parser.add_argument("--save-mjb", type=Path, help="optional compiled MJB path")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="JSON report path")
    parser.add_argument("--check", action="store_true", help="return non-zero if the smoke gate fails")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    model, data, report = run_scene(steps=args.steps, use_viewer=args.viewer)
    export_model(model, canonical_xml=args.save_canonical, binary_mjb=args.save_mjb)
    if args.render is not None:
        render_frame(model, data, args.render)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 1 if args.check and not report["passed"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
