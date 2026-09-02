"""Regression tests for the beginner-to-expert learner journey."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CURRICULUM = ROOT / "curriculum" / "manifest.json"
RUBRIC = ROOT / "curriculum" / "quality_rubric.json"
KNOWLEDGE = ROOT / "knowledge" / "manifest.json"
RUNNER = ROOT / "scripts" / "run_curriculum.py"


def _run(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(RUNNER), *args],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
        env={**os.environ, "PYTHONIOENCODING": "utf-8"},
    )


def test_curriculum_covers_every_knowledge_node_exactly_once() -> None:
    curriculum = json.loads(CURRICULUM.read_text(encoding="utf-8"))
    knowledge = json.loads(KNOWLEDGE.read_text(encoding="utf-8"))
    mapped = [node for module in curriculum["modules"] for node in module["knowledge_nodes"]]
    expected = {node["id"] for node in knowledge["nodes"]}

    assert [level["id"] for level in curriculum["levels"]] == [f"L{i}" for i in range(6)]
    assert [module["id"] for module in curriculum["modules"]] == [
        f"M{i:02d}" for i in range(12)
    ]
    assert len(mapped) == len(set(mapped)) == 45
    assert set(mapped) == expected
    assert len(curriculum["capstones"]) == 3


def test_quality_rubric_is_transparent_and_totals_100() -> None:
    rubric = json.loads(RUBRIC.read_text(encoding="utf-8"))
    assert len(rubric["criteria"]) == 10
    assert sum(item["before"] for item in rubric["criteria"]) == 85
    assert sum(item["after"] for item in rubric["criteria"]) == 100
    assert all(item["after"] == 10 for item in rubric["criteria"])
    assert "not a universal ranking" in rubric["boundary"]
    for criterion in rubric["criteria"]:
        assert criterion["evidence"]
        for relative in criterion["evidence"]:
            assert (ROOT / relative).is_file(), relative


def test_runner_validates_diagnoses_and_plans_in_chinese() -> None:
    validated = _run("--validate")
    assert validated.returncode == 0, validated.stderr
    assert "45 knowledge nodes" in validated.stdout
    assert "85/100 -> 100/100" in validated.stdout
    assert "not certify expertise" in validated.stdout

    diagnosed = _run("--diagnose", "--lang", "zh")
    assert diagnosed.returncode == 0, diagnosed.stderr
    assert "证据自测" in diagnosed.stdout
    assert "M00 · L0" in diagnosed.stdout
    assert "M11 · L5" in diagnosed.stdout

    planned = _run(
        "--plan",
        "full-stack-expert",
        "--hours-per-week",
        "8",
        "--completed",
        "M00,M01",
        "--lang",
        "zh",
    )
    assert planned.returncode == 0, planned.stderr
    assert "全栈具身智能专家" in planned.stdout
    assert "毕业项目" in planned.stdout
    assert "--completed 只用于规划" in planned.stdout


def test_progress_requires_existing_evidence_and_review(tmp_path: Path) -> None:
    progress = tmp_path / "progress.json"
    initialized = _run(
        "--init-progress",
        str(progress),
        "--goal",
        "full-stack-expert",
        "--learner",
        "test-learner",
    )
    assert initialized.returncode == 0, initialized.stderr

    empty_report = _run("--report-progress", str(progress), "--lang", "zh")
    assert empty_report.returncode == 0, empty_report.stderr
    assert "证据加权进度: 0.0%" in empty_report.stdout
    assert "M00" in empty_report.stdout

    record = json.loads(progress.read_text(encoding="utf-8"))
    record["modules"]["M00"] = {
        "status": "passed",
        "evidence": ["evidence/M00/experiment-card.md"],
        "reviewer": "independent-reviewer",
        "reviewed_on": "2026-09-02",
    }
    progress.write_text(json.dumps(record, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    missing = _run("--report-progress", str(progress))
    assert missing.returncode == 1
    assert "evidence does not exist" in missing.stderr

    artifact = tmp_path / "evidence" / "M00" / "experiment-card.md"
    artifact.parent.mkdir(parents=True)
    artifact.write_text("# reviewed evidence\n", encoding="utf-8")
    passing = _run("--report-progress", str(progress))
    assert passing.returncode == 0, passing.stderr
    assert "Passed modules: 1/12" in passing.stdout
    assert "Next: M01" in passing.stdout


def test_passed_progress_cannot_omit_reviewer_or_date(tmp_path: Path) -> None:
    progress = tmp_path / "progress.json"
    initialized = _run(
        "--init-progress",
        str(progress),
        "--goal",
        "robot-learning-engineer",
    )
    assert initialized.returncode == 0, initialized.stderr
    record = json.loads(progress.read_text(encoding="utf-8"))
    record["modules"]["M00"]["status"] = "passed"
    progress.write_text(json.dumps(record), encoding="utf-8")

    report = _run("--report-progress", str(progress))
    assert report.returncode == 1
    assert "has no evidence" in report.stderr
    assert "has no reviewer" in report.stderr
    assert "reviewed_on is not YYYY-MM-DD" in report.stderr
