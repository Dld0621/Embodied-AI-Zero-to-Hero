"""Regression tests for the abstract MuJoCo dexterous-grasp fixture."""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest

pytest.importorskip("mujoco")

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "examples" / "dexterous_grasping_smoke.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("dexterous_grasping_smoke", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_fixture_reports_contact_retention_and_task_evidence():
    report = _load_module().run_demo()

    assert report["pipeline"] == "dexterous-manipulation"
    assert report["task"] == "approach-contact-lift-hold"
    assert report["passed"] is True
    assert report["metrics"]["grasp_success_rate"] == 1.0
    assert report["metrics"]["mean_lift_height_m"] >= 0.045
    assert report["metrics"]["max_lateral_slip_m"] <= 0.02
    assert report["metrics"]["minimum_final_finger_contacts"] >= 2

    evidence = report["evidence"]
    assert evidence["level"] == "smoke"
    assert evidence["qualifier"] == "synthetic-contact-dynamics"
    assert evidence["object_contact_evaluated"] is True
    assert evidence["object_retention_evaluated"] is True
    assert evidence["task_success_evaluated"] is True
    assert evidence["in_hand_reorientation_evaluated"] is False
    assert evidence["learned_policy_evaluated"] is False
    assert evidence["real_hardware_allowed"] is False


def test_fixture_is_deterministic():
    module = _load_module()
    assert module.run_demo() == module.run_demo()


def test_cli_writes_checked_machine_readable_artifact(tmp_path: Path):
    output = tmp_path / "metrics.json"
    completed = subprocess.run(
        [sys.executable, str(SCRIPT), "--check", "--output", str(output)],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    report = json.loads(output.read_text(encoding="utf-8"))
    assert report["passed"] is True
    assert report["simulator"]["model"] == "assets/simulation/dexterous_grasp_smoke.xml"
