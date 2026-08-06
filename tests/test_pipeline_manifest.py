"""Contract tests for the machine-readable learning pipeline catalog."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "pipelines" / "manifest.json"
RUNNER = ROOT / "scripts" / "run_pipeline.py"


def _load_runner():
    spec = importlib.util.spec_from_file_location("run_pipeline", RUNNER)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_manifest_is_valid_and_covers_core_tracks():
    data = json.loads(MANIFEST.read_text(encoding="utf-8"))
    runner = _load_runner()
    assert runner.validate_manifest(data) == []

    pipeline_ids = {pipeline["id"] for pipeline in data["pipelines"]}
    assert pipeline_ids == {
        "simulation-data",
        "vla-policy",
        "world-model-planning",
        "rl-post-training",
        "rfm-cross-embodiment",
        "embodied-reasoning",
        "sim-to-real",
        "dexterous-retargeting",
    }


def test_every_pipeline_has_learning_contract_fields():
    data = json.loads(MANIFEST.read_text(encoding="utf-8"))
    for pipeline in data["pipelines"]:
        assert pipeline["requires"]
        assert pipeline["artifacts"]
        assert pipeline["metrics"]
        assert (ROOT / pipeline["document"]).is_file()
        assert (ROOT / pipeline["entrypoint"]).is_file()


def test_commands_use_argument_arrays_and_python_placeholder():
    data = json.loads(MANIFEST.read_text(encoding="utf-8"))
    for pipeline in data["pipelines"]:
        for mode in ("smoke", "full"):
            spec = pipeline[mode]
            if spec is None:
                continue
            assert isinstance(spec["command"], list)
            assert spec["command"][0] == "{python}"
            assert (ROOT / spec["cwd"]).is_dir()
