#!/usr/bin/env python3
"""Fail on known inaccurate claims and unverifiable evidence labels.

This is a regression guard, not an automatic fact checker. It encodes mistakes
that have already been reviewed against primary sources and verifies that the
benchmark catalog keeps missing evidence visible.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
BENCHMARK = ROOT / "results" / "benchmarks" / "benchmark_v2.json"
ALLOWED_EVIDENCE_LEVELS = {
    "source-backed",
    "reproduced",
    "reported-aggregate",
    "not-evaluated",
    "hardware-validated",
}

# These patterns represent previously identified errors or unsupported
# generalizations. Keep the expressions narrow so legitimate discussions of a
# concept do not become impossible.
FORBIDDEN_PATTERNS = (
    (
        re.compile(r"动力学参数误差.*接触模型误差.*65%"),
        "DexSim2Real does not report this 65% causal decomposition",
    ),
    (
        re.compile(r"无法超越(?:演示者|专家)"),
        "behavior cloning performance must not be bounded by an absolute slogan",
    ),
    (
        re.compile(r"Behavior cloning alone cannot learn robust contact-rich manipulation", re.I),
        "behavior cloning capability depends on policy, data, and evaluation conditions",
    ),
    (
        re.compile(r"This is classic BC overfitting", re.I),
        "low training loss plus task failure does not by itself prove memorization",
    ),
    (
        re.compile(r"10[–-]100× more data", re.I),
        "the repository has no controlled evidence for a universal data multiplier",
    ),
    (
        re.compile(r">\s*1000 episodes", re.I),
        "the repository has no controlled evidence for a universal episode threshold",
    ),
    (
        re.compile(r"100× too small", re.I),
        "the repository has no scaling study supporting this ratio",
    ),
    (
        re.compile(r"推理时，从先验.*采样.*多样"),
        "canonical ACT sets the latent to the prior mean at test time",
    ),
    (
        re.compile(r"直驱（无减速器）.*\|\s*100%"),
        "a real direct-drive system must not be modeled as 100% efficient",
    ),
    (
        re.compile(r"教科书级.*过拟合"),
        "the recorded aggregate does not independently establish overfitting",
    ),
    (
        re.compile(r"OCTO.*pad.*32D", re.I),
        "Octo adapts action spaces through modular output heads, not a universal 32-D claim",
    ),
    (
        re.compile(r"Octo 支持点云输入", re.I),
        "the released Octo base configuration does not natively include a point-cloud tokenizer",
    ),
    (
        re.compile(r"C\+\+ 性能最强"),
        "programming-language performance depends on implementation and workload",
    ),
    (
        re.compile(r"目前最大的开源机器人操作数据集"),
        "time-sensitive dataset superlatives require a dated comparison",
    ),
    (
        re.compile(r"通用性最强.*OpenVLA"),
        "model superlatives require a named, current benchmark",
    ),
    (
        re.compile(r"表达能力最强"),
        "architecture superlatives require a defined metric and comparison",
    ),
    (
        re.compile(r"Smart Motor.*30mN"),
        "sensor resolution must come from the exact Shadow Hand product datasheet",
    ),
    (
        re.compile(r"Allegro.*六轴力矩传感器.*1kHz", re.S),
        "a stock Allegro Hand must not imply a specific third-party fingertip sensor",
    ),
    (
        re.compile(r"关节阻尼通常比理论值高 20-40%"),
        "joint damping requires system identification on the target hand",
    ),
)

REQUIRED_SOURCE_MARKERS = {
    "docs/05-interview-prep.md": (
        "https://arxiv.org/abs/2304.13705",
        "https://github.com/tonyzhaozh/act",
    ),
    "docs/19-sim-to-real-guide.md": (
        "https://arxiv.org/abs/2605.05241",
        "预印本",
    ),
    "docs/CLAIM_REVIEW.md": (
        "reported-aggregate",
        "not-evaluated",
        "不能替代",
    ),
}


def _content_files() -> list[Path]:
    files = [ROOT / "README.md", ROOT / "README_CN.md", ROOT / "BENCHMARK.md"]
    files.extend(sorted((ROOT / "docs").rglob("*.md")))
    return [path for path in files if path.is_file()]


def _tracked_paths(errors: list[str]) -> list[str]:
    try:
        result = subprocess.run(
            [
                "git",
                "-C",
                str(ROOT),
                "ls-files",
                "--cached",
                "--others",
                "--exclude-standard",
                "-z",
            ],
            check=True,
            capture_output=True,
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        errors.append(f"cannot enumerate tracked paths: {exc}")
        return []
    paths = [item.decode("utf-8") for item in result.stdout.split(b"\0") if item]
    return [relative for relative in paths if (ROOT / relative).exists()]


def _load_benchmark(errors: list[str]) -> dict[str, Any]:
    try:
        data = json.loads(BENCHMARK.read_text(encoding="utf-8-sig"))
    except (OSError, json.JSONDecodeError) as exc:
        errors.append(f"cannot load benchmark catalog: {exc}")
        return {}
    if not isinstance(data, dict):
        errors.append("benchmark catalog root must be an object")
        return {}
    return data


def audit_claims() -> dict[str, Any]:
    errors: list[str] = []
    checked_files = 0

    for path in _content_files():
        text = path.read_text(encoding="utf-8")
        relative = path.relative_to(ROOT).as_posix()
        checked_files += 1
        for pattern, reason in FORBIDDEN_PATTERNS:
            for match in pattern.finditer(text):
                line = text.count("\n", 0, match.start()) + 1
                errors.append(f"{relative}:{line}: {reason}")

    for relative, markers in REQUIRED_SOURCE_MARKERS.items():
        path = ROOT / relative
        if not path.is_file():
            errors.append(f"required claim-review document missing: {relative}")
            continue
        text = path.read_text(encoding="utf-8")
        for marker in markers:
            if marker not in text:
                errors.append(f"{relative}: required source/evidence marker missing: {marker}")

    for relative in _tracked_paths(errors):
        if any(ord(character) > 127 for character in relative):
            errors.append(f"tracked path contains non-ASCII/confusable characters: {relative}")

    data = _load_benchmark(errors)
    if not str(data.get("evidence_policy", "")).strip():
        errors.append("benchmark catalog must declare evidence_policy")
    results = data.get("results", {})
    if not isinstance(results, dict):
        errors.append("benchmark catalog results must be an object")
        results = {}

    for method, result in results.items():
        if not isinstance(result, dict):
            errors.append(f"benchmark result must be an object: {method}")
            continue
        evidence_level = result.get("evidence_level")
        if evidence_level not in ALLOWED_EVIDENCE_LEVELS:
            errors.append(f"benchmark {method} has invalid evidence_level: {evidence_level}")
        evaluated = result.get("success_rate_pct") is not None
        if evaluated and evidence_level == "not-evaluated":
            errors.append(f"benchmark {method} is evaluated but labeled not-evaluated")
        if not evaluated and evidence_level != "not-evaluated":
            errors.append(f"benchmark {method} lacks a task metric but is not labeled not-evaluated")

    for method in ("smolvla_500", "smolvla_10k"):
        result = results.get(method, {})
        artifacts = result.get("artifacts", {}) if isinstance(result, dict) else {}
        if artifacts.get("eval_results.json") is not False:
            errors.append(f"{method} must keep the missing per-episode evaluation visible")
        if result.get("evidence_level") != "reported-aggregate":
            errors.append(f"{method} must remain reported-aggregate until raw artifacts are committed")

    provenance = data.get("provenance", {})
    if not isinstance(provenance, dict):
        errors.append("benchmark provenance must be an object")
    elif provenance.get("independent_reaggregation") is not False:
        errors.append("benchmark must not claim independent re-aggregation while raw artifacts are absent")
    elif not str(provenance.get("limitation", "")).strip():
        errors.append("benchmark provenance must explain the re-aggregation limitation")

    return {
        "ok": not errors,
        "errors": errors,
        "stats": {
            "content_files": checked_files,
            "known_invalid_patterns": len(FORBIDDEN_PATTERNS),
            "benchmark_methods": len(results),
        },
        "boundary": "Known-claim regression guard only; primary-source semantic review remains required.",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", action="store_true", help="emit machine-readable JSON")
    args = parser.parse_args()
    report = audit_claims()
    if args.json:
        print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    elif report["ok"]:
        print("Known-claim accuracy gate: OK")
        for key, value in report["stats"].items():
            print(f"- {key}: {value}")
        print(f"Boundary: {report['boundary']}")
    else:
        print("Known-claim accuracy gate: FAILED")
        for error in report["errors"]:
            print(f"- {error}")
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
