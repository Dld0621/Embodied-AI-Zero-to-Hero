from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "knowledge" / "manifest.json"
PIPELINE_MANIFEST = ROOT / "pipelines" / "manifest.json"
RUNNER = ROOT / "scripts" / "run_knowledge_map.py"


def _load_runner():
    spec = importlib.util.spec_from_file_location("run_knowledge_map", RUNNER)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_knowledge_manifest_is_a_bilingual_acyclic_contract() -> None:
    data = json.loads(MANIFEST.read_text(encoding="utf-8"))
    runner = _load_runner()
    assert runner.validate_manifest(data) == []
    assert len(data["stages"]) == 6
    assert len(data["domains"]) == 9
    assert len(data["nodes"]) == 45


def test_every_pipeline_is_grounded_in_multiple_knowledge_nodes() -> None:
    knowledge = json.loads(MANIFEST.read_text(encoding="utf-8"))
    pipelines = json.loads(PIPELINE_MANIFEST.read_text(encoding="utf-8"))
    counts = {
        pipeline["id"]: sum(
            pipeline["id"] in node["pipelines"] for node in knowledge["nodes"]
        )
        for pipeline in pipelines["pipelines"]
    }
    assert all(count >= 2 for count in counts.values()), counts


def test_every_node_has_bilingual_outcome_and_assessment() -> None:
    data = json.loads(MANIFEST.read_text(encoding="utf-8"))
    for node in data["nodes"]:
        for field in ("title", "title_zh", "outcome", "outcome_zh", "assessment", "assessment_zh"):
            assert node[field].strip(), f"{node['id']}.{field}"
        assert (ROOT / node["document"]).is_file(), node["document"]


def test_dependency_resolution_is_prerequisite_first() -> None:
    data = json.loads(MANIFEST.read_text(encoding="utf-8"))
    runner = _load_runner()
    path = runner.learning_order("learning-vla", data["nodes"])
    ids = [node["id"] for node in path]
    positions = {node_id: index for index, node_id in enumerate(ids)}
    assert ids[-1] == "learning-vla"
    for node in path:
        for prerequisite in node["prerequisites"]:
            assert positions[prerequisite] < positions[node["id"]]


def test_runner_validates_and_localizes() -> None:
    env = {**os.environ, "PYTHONIOENCODING": "utf-8"}
    validated = subprocess.run(
        [sys.executable, str(RUNNER), "--validate"],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
        env=env,
    )
    assert validated.returncode == 0, validated.stderr
    assert "45 knowledge nodes" in validated.stdout

    resolved = subprocess.run(
        [sys.executable, str(RUNNER), "--path-to", "task-dexterity-teleoperation", "--lang", "zh"],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
        env=env,
    )
    assert resolved.returncode == 0, resolved.stderr
    assert "前置学习路径" in resolved.stdout
    assert "灵巧操作、重定向与遥操作" in resolved.stdout


def test_bilingual_guides_expose_all_domains_and_visuals() -> None:
    data = json.loads(MANIFEST.read_text(encoding="utf-8"))
    guides = (
        ROOT / "docs" / "knowledge-system" / "README.md",
        ROOT / "docs" / "knowledge-system" / "README_CN.md",
    )
    for guide in guides:
        text = guide.read_text(encoding="utf-8")
        for domain in data["domains"]:
            assert f'<a id="{domain["id"]}"></a>' in text
    assert (ROOT / "docs" / "assets" / "knowledge-system.svg").is_file()
    assert (ROOT / "docs" / "assets" / "knowledge-system-cn.svg").is_file()
