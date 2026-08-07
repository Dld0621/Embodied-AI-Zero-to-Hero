"""Regression tests for the lightweight system-level pipeline demos."""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PERCEPTION = ROOT / "examples" / "perception_state_estimation_smoke.py"
NAVIGATION = ROOT / "examples" / "navigation_locomotion_smoke.py"


def _load(path: Path, module_name: str):
    spec = importlib.util.spec_from_file_location(module_name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def test_perception_smoke_is_deterministic_and_passes_gates():
    module = _load(PERCEPTION, "perception_state_estimation_smoke")
    first = module.run_demo(seed=7, steps=120)
    second = module.run_demo(seed=7, steps=120)
    assert first == second
    assert first["passed"] is True
    assert first["evidence"]["level"] == "synthetic-smoke"
    assert first["metrics"]["stale_observation_rate"] > 0.0
    assert set(first["metric_units"]) == set(first["metrics"])


def test_navigation_smoke_replans_without_collision():
    module = _load(NAVIGATION, "navigation_locomotion_smoke")
    report = module.run_demo(seed=11)
    assert report["passed"] is True
    assert report["metrics"]["goal_success_rate"] == 1.0
    assert report["metrics"]["collision_or_fall_rate"] == 0.0
    assert set(report["metric_units"]) == set(report["metrics"])
    dynamic = next(item for item in report["scenarios"] if item["name"] == "dynamic-replan")
    assert dynamic["safety_interventions"] == 1
    assert dynamic["recovery_successes"] == 1


def test_pipeline_clis_write_machine_readable_artifacts(tmp_path: Path):
    for script, pipeline_id in (
        (PERCEPTION, "perception-state-estimation"),
        (NAVIGATION, "navigation-locomotion"),
    ):
        output = tmp_path / f"{pipeline_id}.json"
        completed = subprocess.run(
            [sys.executable, str(script), "--check", "--output", str(output)],
            cwd=ROOT,
            check=False,
            capture_output=True,
            text=True,
        )
        assert completed.returncode == 0, completed.stderr or completed.stdout
        report = json.loads(output.read_text(encoding="utf-8"))
        assert report["pipeline"] == pipeline_id
        assert report["passed"] is True
