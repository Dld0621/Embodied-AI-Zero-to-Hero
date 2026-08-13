"""Contracts for the modular MuJoCo scene-building tutorial."""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

mujoco = pytest.importorskip("mujoco")

ROOT = Path(__file__).resolve().parents[1]
EXAMPLE_DIR = ROOT / "examples" / "mujoco_scene_builder"
SCRIPT = EXAMPLE_DIR / "run_scene.py"
SCENE = EXAMPLE_DIR / "scene.xml"


def _load_module():
    spec = importlib.util.spec_from_file_location("mujoco_scene_builder", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_modular_scene_compiles_and_steps_with_named_contracts():
    module = _load_module()
    model, data, report = module.run_scene(steps=300)

    assert report["passed"] is True
    assert report["inspection"]["missing_required_names"] == []
    assert report["inspection"]["dimensions"] == {
        "nq": 9,
        "nv": 8,
        "nu": 2,
        "nbody": 6,
        "njnt": 3,
        "ngeom": 11,
        "nsite": 2,
        "nsensor": 7,
        "nsensordata": 9,
    }
    assert report["simulation"]["states_are_finite"] is True
    assert report["simulation"]["max_joint_limit_violation_rad"] == 0.0
    assert data.time == pytest.approx(300 * model.opt.timestep)
    assert np.isfinite(data.sensordata).all()


def test_visual_and_collision_geometry_are_separated():
    model = mujoco.MjModel.from_xml_path(str(SCENE))
    for visual_name in ("link1_visual", "link2_visual", "tool_visual", "object_visual"):
        geom_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, visual_name)
        assert geom_id >= 0
        assert model.geom_group[geom_id] == 2
        assert model.geom_contype[geom_id] == 0
        assert model.geom_conaffinity[geom_id] == 0

    for collision_name in ("link1_collision", "link2_collision", "tool_collision", "object_collision"):
        geom_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, collision_name)
        assert geom_id >= 0
        assert model.geom_group[geom_id] == 3
        assert model.geom_contype[geom_id] == 1
        assert model.geom_conaffinity[geom_id] == 1


def test_canonical_and_binary_exports_round_trip(tmp_path: Path):
    module = _load_module()
    model, _, _ = module.run_scene(steps=20)
    canonical = tmp_path / "canonical.xml"
    binary = tmp_path / "compiled.mjb"

    module.export_model(model, canonical_xml=canonical, binary_mjb=binary)

    canonical_model = mujoco.MjModel.from_xml_path(str(canonical))
    binary_model = mujoco.MjModel.from_binary_path(str(binary))
    assert (canonical_model.nq, canonical_model.nv, canonical_model.nu) == (model.nq, model.nv, model.nu)
    assert (binary_model.nq, binary_model.nv, binary_model.nu) == (model.nq, model.nv, model.nu)


def test_cli_writes_checked_report(tmp_path: Path):
    output = tmp_path / "report.json"
    completed = subprocess.run(
        [sys.executable, str(SCRIPT), "--steps", "120", "--check", "--output", str(output)],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    report = json.loads(output.read_text(encoding="utf-8"))
    assert report["tutorial"] == "mujoco-scene-builder"
    assert report["passed"] is True
    assert report["simulator"]["model"] == "examples/mujoco_scene_builder/scene.xml"
