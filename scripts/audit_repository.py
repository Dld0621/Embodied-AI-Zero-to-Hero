#!/usr/bin/env python3
"""Audit repository contracts that can be verified without network or hardware.

This script deliberately checks structural and recorded-evidence consistency. It
does not certify semantic correctness of every external source or authorize real
robot motion. See docs/VALIDATION.md for the evidence boundary.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import xml.etree.ElementTree as ET
from datetime import date
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
PIPELINE_MANIFEST = ROOT / "pipelines" / "manifest.json"
ROUTE_MANIFEST = ROOT / "learning_paths" / "manifest.json"
BENCHMARK = ROOT / "results" / "benchmarks" / "benchmark_v2.json"

REQUIRED_FILES = (
    "README.md",
    "README_CN.md",
    "LICENSE",
    "CHANGELOG.md",
    "CONTRIBUTING.md",
    "CODE_OF_CONDUCT.md",
    "SECURITY.md",
    "CITATION.cff",
    "THIRD_PARTY_NOTICES.md",
    "Dockerfile",
    "mkdocs.yml",
    "pyproject.toml",
    "requirements-test-lock.txt",
    ".pre-commit-config.yaml",
    ".github/CODEOWNERS",
    ".github/dependabot.yml",
    ".github/PULL_REQUEST_TEMPLATE.md",
    ".github/repository-metadata.json",
    ".github/workflows/docs-pages.yml",
    "docs/index.md",
    "docs/index_cn.md",
    "docs/field-map.md",
    "docs/field-map-cn.md",
    "docs/learning-paths/README.md",
    "docs/learning-paths/README_CN.md",
    "docs/stylesheets/extra.css",
    "docs/VALIDATION.md",
    "docs/CLAIM_REVIEW.md",
    "docs/SOURCES.md",
    "docs/foundations/README_EN.md",
    "docs/tutorials/mujoco-scene-building.md",
    "examples/mujoco_scene_builder/README.md",
    "examples/mujoco_scene_builder/scene.xml",
    "examples/mujoco_scene_builder/robot.xml",
    "examples/mujoco_scene_builder/run_scene.py",
    "learning_paths/manifest.json",
    "scripts/run_learning_path.py",
    "scripts/check_claims.py",
    "scripts/check_markdown_format.py",
)

ALLOWED_PIPELINE_STATUS = {
    "smoke-tested",
    "interface-tested",
    "documented",
    "hardware-validated",
}

ALLOWED_EVIDENCE_LEVELS = {
    "source-backed",
    "reproduced",
    "reported-aggregate",
    "not-evaluated",
    "hardware-validated",
}


def _load_json(path: Path, errors: list[str]) -> dict[str, Any]:
    try:
        # utf-8-sig accepts both ordinary UTF-8 and historical JSON files
        # carrying a BOM without weakening JSON parsing.
        value = json.loads(path.read_text(encoding="utf-8-sig"))
    except (OSError, json.JSONDecodeError) as exc:
        errors.append(f"cannot load JSON {path.relative_to(ROOT)}: {exc}")
        return {}
    if not isinstance(value, dict):
        errors.append(f"JSON root must be an object: {path.relative_to(ROOT)}")
        return {}
    return value


def _check_required_files(errors: list[str]) -> None:
    for relative in REQUIRED_FILES:
        if not (ROOT / relative).is_file():
            errors.append(f"required file missing: {relative}")


def _check_pipeline_manifest(errors: list[str], stats: dict[str, Any]) -> None:
    data = _load_json(PIPELINE_MANIFEST, errors)
    pipelines = data.get("pipelines", [])
    if not isinstance(pipelines, list):
        errors.append("pipelines/manifest.json: pipelines must be a list")
        return

    ids: list[str] = []
    status_counts: dict[str, int] = {}
    required_fields = {"id", "title", "status", "document", "entrypoint", "requires", "artifacts", "metrics", "smoke", "full"}

    for item in pipelines:
        if not isinstance(item, dict):
            errors.append("pipeline entry must be an object")
            continue
        missing = required_fields.difference(item)
        if missing:
            errors.append(f"pipeline {item.get('id', '<unknown>')} missing fields: {sorted(missing)}")
            continue

        pipeline_id = str(item["id"])
        ids.append(pipeline_id)
        status = str(item["status"])
        status_counts[status] = status_counts.get(status, 0) + 1
        if status not in ALLOWED_PIPELINE_STATUS:
            errors.append(f"pipeline {pipeline_id} has unsupported status: {status}")

        for key in ("document", "entrypoint"):
            target = ROOT / str(item[key])
            if not target.is_file():
                errors.append(f"pipeline {pipeline_id} missing {key}: {item[key]}")

        for requirement in item.get("requires", []):
            if not (ROOT / str(requirement)).is_file():
                errors.append(f"pipeline {pipeline_id} missing prerequisite: {requirement}")

        if not item.get("artifacts") or not item.get("metrics"):
            errors.append(f"pipeline {pipeline_id} must declare artifacts and metrics")

        if status in {"smoke-tested", "interface-tested"} and item.get("smoke") is None:
            errors.append(f"pipeline {pipeline_id} status {status} requires a smoke command")

        for mode in ("smoke", "full"):
            command_spec = item.get(mode)
            if command_spec is None:
                continue
            command = command_spec.get("command")
            if not isinstance(command, list) or not command or command[0] != "{python}":
                errors.append(f"pipeline {pipeline_id} {mode} command must be an argument array beginning with {{python}}")
            if not (ROOT / str(command_spec.get("cwd", ""))).is_dir():
                errors.append(f"pipeline {pipeline_id} {mode} cwd does not exist")

        document = ROOT / str(item["document"])
        if document.is_file():
            text = document.read_text(encoding="utf-8")
            for heading in ("## English contract", "## 目标与边界"):
                if heading not in text:
                    errors.append(f"pipeline {pipeline_id} document missing heading: {heading}")

    if len(ids) != len(set(ids)):
        errors.append("pipeline IDs must be unique")
    if len(ids) != 11:
        errors.append(f"expected 11 registered pipelines, found {len(ids)}")

    expected_status_counts = {
        "smoke-tested": 8,
        "interface-tested": 2,
        "documented": 1,
    }
    if status_counts != expected_status_counts:
        errors.append(
            "pipeline evidence distribution differs from the reviewed baseline: "
            f"expected {expected_status_counts}, found {status_counts}"
        )

    by_id = {
        str(item.get("id")): item
        for item in pipelines
        if isinstance(item, dict)
    }
    for pipeline_id in (
        "perception-state-estimation",
        "navigation-locomotion",
        "dexterous-manipulation",
    ):
        item = by_id.get(pipeline_id, {})
        command = (item.get("smoke") or {}).get("command", [])
        if item.get("status") != "smoke-tested" or "--check" not in command:
            errors.append(
                f"pipeline {pipeline_id} must retain a checked synthetic smoke command"
            )

    stats["pipelines"] = len(ids)
    stats["pipeline_status"] = status_counts
    stats["full_commands"] = sum(item.get("full") is not None for item in pipelines if isinstance(item, dict))


def _check_foundations_and_languages(errors: list[str], stats: dict[str, Any]) -> None:
    foundation_dir = ROOT / "docs" / "foundations"
    numbered = sorted(foundation_dir.glob("[0-9][0-9]-*.md"))
    lessons = [path for path in numbered if path.name != "00-roadmap.md"]
    if len(lessons) != 14:
        errors.append(f"expected 14 foundation lessons, found {len(lessons)}")

    for lesson in lessons:
        text = lesson.read_text(encoding="utf-8")
        if "../SOURCES.md#" not in text:
            errors.append(f"foundation lesson lacks primary-source pointer: {lesson.name}")
        if "检查理解" not in text:
            errors.append(f"foundation lesson lacks understanding check: {lesson.name}")

    english_overview = foundation_dir / "README_EN.md"
    if english_overview.is_file():
        overview = english_overview.read_text(encoding="utf-8")
        for lesson in lessons:
            if lesson.name not in overview:
                errors.append(f"English foundations overview does not link {lesson.name}")

    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    readme_cn = (ROOT / "README_CN.md").read_text(encoding="utf-8")
    for anchor in ("start", "system", "pipelines", "evidence", "docs"):
        marker = f'<a id="{anchor}"></a>'
        if marker not in readme or marker not in readme_cn:
            errors.append(f"bilingual README anchor missing: {anchor}")
    if "README_CN.md" not in readme or "README.md" not in readme_cn:
        errors.append("root READMEs must link to each other")

    sources = (ROOT / "docs" / "SOURCES.md").read_text(encoding="utf-8")
    source_sections = re.findall(r"^## (\d{2}) ", sources, flags=re.MULTILINE)
    if source_sections != [f"{index:02d}" for index in range(1, 17)]:
        errors.append("docs/SOURCES.md must contain ordered sections 01 through 16")
    official_links = re.findall(r"https://[^)\s]+", sources)
    if len(official_links) < 14:
        errors.append("docs/SOURCES.md must include at least one external primary source per lesson")

    stats["foundation_lessons"] = len(lessons)
    stats["primary_source_links"] = len(official_links)


def _check_research_routes(errors: list[str], stats: dict[str, Any]) -> None:
    data = _load_json(ROUTE_MANIFEST, errors)
    routes = data.get("routes", [])
    if not isinstance(routes, list):
        errors.append("learning_paths/manifest.json: routes must be a list")
        return

    pipeline_data = _load_json(PIPELINE_MANIFEST, errors)
    pipeline_ids = {
        str(item.get("id"))
        for item in pipeline_data.get("pipelines", [])
        if isinstance(item, dict)
    }
    route_ids: list[str] = []
    covered_pipelines: set[str] = set()
    required_fields = {
        "id",
        "title",
        "title_zh",
        "question",
        "question_zh",
        "foundations",
        "pipelines",
        "deliverable",
        "deliverable_zh",
        "metrics",
        "promotion_gate",
        "promotion_gate_zh",
        "boundary",
        "boundary_zh",
    }
    bilingual_fields = (
        ("title", "title_zh"),
        ("question", "question_zh"),
        ("deliverable", "deliverable_zh"),
        ("promotion_gate", "promotion_gate_zh"),
        ("boundary", "boundary_zh"),
    )

    for route in routes:
        if not isinstance(route, dict):
            errors.append("research route entry must be an object")
            continue
        missing = required_fields.difference(route)
        if missing:
            errors.append(
                f"research route {route.get('id', '<unknown>')} missing fields: {sorted(missing)}"
            )
            continue

        route_id = str(route["id"])
        route_ids.append(route_id)
        if not re.fullmatch(r"[a-z0-9]+(?:-[a-z0-9]+)*", route_id):
            errors.append(f"invalid research route id: {route_id}")
        for english, chinese in bilingual_fields:
            if not str(route.get(english, "")).strip() or not str(route.get(chinese, "")).strip():
                errors.append(f"research route {route_id} lacks bilingual field: {english}/{chinese}")

        foundations = route.get("foundations", [])
        if not isinstance(foundations, list) or not foundations:
            errors.append(f"research route {route_id} must declare foundations")
        else:
            for relative in foundations:
                if not (ROOT / str(relative)).is_file():
                    errors.append(f"research route {route_id} missing foundation: {relative}")

        linked_pipelines = route.get("pipelines", [])
        if not isinstance(linked_pipelines, list) or not linked_pipelines:
            errors.append(f"research route {route_id} must declare pipelines")
        else:
            for pipeline_id in linked_pipelines:
                pipeline_id = str(pipeline_id)
                if pipeline_id not in pipeline_ids:
                    errors.append(f"research route {route_id} references unknown pipeline: {pipeline_id}")
                covered_pipelines.add(pipeline_id)

        metrics = route.get("metrics", [])
        if not isinstance(metrics, list) or not metrics:
            errors.append(f"research route {route_id} must declare metrics")

    if len(route_ids) != 7:
        errors.append(f"expected 7 research routes, found {len(route_ids)}")
    if len(route_ids) != len(set(route_ids)):
        errors.append("research route IDs must be unique")
    missing_pipeline_coverage = sorted(pipeline_ids.difference(covered_pipelines))
    if missing_pipeline_coverage:
        errors.append(
            "research routes do not cover every pipeline: "
            + ", ".join(missing_pipeline_coverage)
        )

    route_docs = (
        ROOT / "docs" / "learning-paths" / "README.md",
        ROOT / "docs" / "learning-paths" / "README_CN.md",
    )
    for document in route_docs:
        if not document.is_file():
            continue
        text = document.read_text(encoding="utf-8")
        for route_id in route_ids:
            if f'<a id="{route_id}"></a>' not in text:
                errors.append(
                    f"research route document lacks anchor {route_id}: {document.relative_to(ROOT)}"
                )

    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    readme_cn = (ROOT / "README_CN.md").read_text(encoding="utf-8")
    for text, expected in (
        (readme, "docs/learning-paths/README.md"),
        (readme_cn, "docs/learning-paths/README_CN.md"),
    ):
        if expected not in text or "run_learning_path.py" not in text:
            errors.append(f"root README must expose the research-route guide and CLI: {expected}")

    stats["research_routes"] = len(route_ids)
    stats["route_pipeline_coverage"] = len(pipeline_ids.intersection(covered_pipelines))


def _check_benchmark(errors: list[str], stats: dict[str, Any]) -> None:
    data = _load_json(BENCHMARK, errors)
    results = data.get("results", {})
    summary = data.get("summary_table", {})
    if not isinstance(results, dict) or not isinstance(summary, dict):
        errors.append("benchmark results and summary_table must be objects")
        return

    if not str(data.get("evidence_policy", "")).strip():
        errors.append("benchmark catalog must declare its evidence policy")

    for method, value in summary.items():
        if method == "note":
            continue
        if method not in results:
            errors.append(f"benchmark summary references unknown method: {method}")
            continue
        if results[method].get("success_rate_pct") != value:
            errors.append(f"benchmark summary mismatch for {method}")

    for method, result in results.items():
        if not isinstance(result, dict):
            errors.append(f"benchmark result must be an object: {method}")
            continue
        evidence_level = result.get("evidence_level")
        if evidence_level not in ALLOWED_EVIDENCE_LEVELS:
            errors.append(f"benchmark {method} has invalid evidence level: {evidence_level}")
        if result.get("success_rate_pct") is None and evidence_level != "not-evaluated":
            errors.append(f"benchmark {method} lacks a task metric but is not labeled not-evaluated")
        if result.get("success_rate_pct") is not None and evidence_level == "not-evaluated":
            errors.append(f"benchmark {method} is evaluated but labeled not-evaluated")

    for label, relative in data.get("source_files", {}).items():
        if not isinstance(relative, str) or not (ROOT / relative).is_file():
            errors.append(f"benchmark source file missing for {label}: {relative}")

    updated = data.get("last_updated")
    try:
        if updated and date.fromisoformat(str(updated)) > date.today():
            errors.append("benchmark last_updated is in the future")
    except ValueError:
        errors.append("benchmark last_updated must be ISO YYYY-MM-DD")

    wm_raw = _load_json(ROOT / "results" / "benchmarks" / "wm_results.json", errors)
    wm_canonical = results.get("wm_mpc_cem", {})
    expected_pairs = (
        (wm_raw.get("best_val_loss"), wm_canonical.get("wm_val_loss"), "world-model validation loss"),
        (wm_raw.get("multistep_errors", {}).get("H1"), wm_canonical.get("wm_multi_step_error", {}).get("H=1"), "world-model H1 error"),
        (wm_raw.get("multistep_errors", {}).get("H5"), wm_canonical.get("wm_multi_step_error", {}).get("H=5"), "world-model H5 error"),
        (wm_raw.get("multistep_errors", {}).get("H10"), wm_canonical.get("wm_multi_step_error", {}).get("H=10"), "world-model H10 error"),
    )
    for raw, canonical, label in expected_pairs:
        if raw != canonical:
            errors.append(f"{label} differs between wm_results.json and benchmark_v2.json")

    expected_line = "H=1: 0.0708, H=5: 0.2961, H=10: 0.5560"
    for relative in ("BENCHMARK.md", "docs/benchmark_report.md"):
        text = (ROOT / relative).read_text(encoding="utf-8")
        if expected_line not in text or "0.0409" not in text:
            errors.append(f"{relative} does not match canonical world-model metrics")

    for method, directory in (
        ("smolvla_500", ROOT / "results" / "smolvla" / "500_steps"),
        ("smolvla_10k", ROOT / "results" / "smolvla" / "10k_steps"),
    ):
        declared = results.get(method, {}).get("artifacts", {})
        for filename, present in declared.items():
            actual = (directory / filename).is_file()
            if bool(present) != actual:
                errors.append(f"{method} artifact declaration mismatch: {filename}")
        if declared.get("eval_results.json") is False and results.get(method, {}).get("evidence_level") != "reported-aggregate":
            errors.append(f"{method} must remain reported-aggregate while per-episode evidence is absent")

    provenance = data.get("provenance", {})
    if not isinstance(provenance, dict):
        errors.append("benchmark provenance must be an object")
    elif provenance.get("independent_reaggregation") is not False:
        errors.append("benchmark must not claim independent re-aggregation without raw artifacts")
    elif not str(provenance.get("limitation", "")).strip():
        errors.append("benchmark provenance must state its re-aggregation limitation")

    stats["benchmark_methods"] = len(results)
    stats["benchmark_source_files"] = len(data.get("source_files", {}))


def _check_third_party(errors: list[str], stats: dict[str, Any]) -> None:
    licenses = (
        "pretrained/anyteleop/frankmocap/LICENSE",
        "pretrained/urdf/mujoco_menagerie/LICENSE",
        "pretrained/urdf/mujoco_menagerie/franka_fr3/LICENSE",
        "pretrained/urdf/mujoco_menagerie/shadow_hand/LICENSE",
        "pretrained/urdf/leap_hand_sim/LICENSE.txt",
        "pretrained/urdf/orcahand_description/LICENSE",
    )
    for relative in licenses:
        if not (ROOT / relative).is_file():
            errors.append(f"third-party license missing: {relative}")

    notices = (ROOT / "THIRD_PARTY_NOTICES.md").read_text(encoding="utf-8")
    if "Attribution-NonCommercial 4.0" not in notices or "per-model" not in notices:
        errors.append("third-party notices must preserve FrankMocap and Menagerie license boundaries")

    for relative in ("README.md", "README_CN.md"):
        readme = (ROOT / relative).read_text(encoding="utf-8")
        if "original%20content-MIT" not in readme or "third--party%20assets-mixed%20licenses" not in readme:
            errors.append(f"{relative} must distinguish original MIT content from mixed-license assets")

    stats["third_party_license_files"] = len(licenses)


def _check_project_identity(errors: list[str], stats: dict[str, Any]) -> None:
    metadata = _load_json(ROOT / ".github" / "repository-metadata.json", errors)
    description = str(metadata.get("description", ""))
    topics = metadata.get("topics", [])
    if not description or "建议" in description or "rename" in description.lower():
        errors.append("repository metadata must use a publication-ready description")
    if not isinstance(topics, list) or len(set(topics)) < 8:
        errors.append("repository metadata must declare at least eight unique topics")
    required_topics = {"embodied-ai", "robot-learning", "vision-language-action", "sim-to-real"}
    if isinstance(topics, list) and not required_topics.issubset(set(topics)):
        errors.append("repository metadata is missing required discovery topics")

    citation = (ROOT / "CITATION.cff").read_text(encoding="utf-8")
    security = (ROOT / "SECURITY.md").read_text(encoding="utf-8")
    expected_email = "Steven.LI@connect.hku.hk"
    if expected_email not in citation or expected_email not in security:
        errors.append("public contact email must be consistent across citation and security files")
    if not re.search(r'^version:\s*["\']?0\.1\.0-dev["\']?\s*$', citation, flags=re.MULTILINE):
        errors.append("CITATION.cff must retain a pre-release version until a tag is published")

    stats["repository_topics"] = len(set(topics)) if isinstance(topics, list) else 0


def _check_visual_system(errors: list[str], stats: dict[str, Any]) -> None:
    assets = (
        "assets/dof-hero.svg",
        "assets/dof-hero-dark.svg",
        "assets/dof-hero-cn.svg",
        "assets/dof-hero-cn-dark.svg",
        "assets/system_architecture.svg",
        "assets/system_architecture-cn.svg",
        "assets/dof-learning-map.svg",
        "assets/dof-learning-map-cn.svg",
    )
    svg_namespace = {"svg": "http://www.w3.org/2000/svg"}
    for relative in assets:
        path = ROOT / relative
        if not path.is_file():
            errors.append(f"active visual asset missing: {relative}")
            continue
        try:
            root = ET.parse(path).getroot()
        except ET.ParseError as exc:
            errors.append(f"invalid SVG {relative}: {exc}")
            continue
        if root.find("svg:title", svg_namespace) is None or root.find("svg:desc", svg_namespace) is None:
            errors.append(f"active SVG lacks accessible title/description: {relative}")

    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    readme_cn = (ROOT / "README_CN.md").read_text(encoding="utf-8")
    if "dof-logo" in readme or "dof-logo" in readme_cn:
        errors.append("inactive DoF logo drafts must not be displayed on README landing pages")
    if "dof-hero-dark.svg" not in readme or "dof-hero-cn-dark.svg" not in readme_cn:
        errors.append("bilingual README heroes must declare dedicated dark-mode sources")

    mkdocs = (ROOT / "mkdocs.yml").read_text(encoding="utf-8")
    if "stylesheets/extra.css" not in mkdocs:
        errors.append("MkDocs must load the repository interface stylesheet")

    for relative in ("docs/index.md", "docs/index_cn.md"):
        landing = (ROOT / relative).read_text(encoding="utf-8")
        for marker in (
            "dof-loop",
            "dof-route-grid",
            "dof-coverage",
            "dof-status",
            "dof-signal",
            "dof-legend",
        ):
            if marker not in landing:
                errors.append(f"documentation landing page lacks {marker}: {relative}")
    landing_cn = (ROOT / "docs/index_cn.md").read_text(encoding="utf-8")
    if re.search(r'href="(?:foundations|field-map|pipelines|benchmark_report|VALIDATION)/', landing_cn):
        errors.append("Chinese landing raw-HTML links must resolve from the site root via ../")
    if "field-map.md" not in mkdocs or "field-map-cn.md" not in mkdocs:
        errors.append("MkDocs navigation must expose both field-map languages")
    if "learning-paths/README.md" not in mkdocs or "learning-paths/README_CN.md" not in mkdocs:
        errors.append("MkDocs navigation must expose both research-route languages")

    stats["active_visual_assets"] = len(assets)


def audit_repository() -> dict[str, Any]:
    errors: list[str] = []
    stats: dict[str, Any] = {}
    _check_required_files(errors)
    _check_pipeline_manifest(errors, stats)
    _check_foundations_and_languages(errors, stats)
    _check_research_routes(errors, stats)
    _check_benchmark(errors, stats)
    _check_third_party(errors, stats)
    _check_project_identity(errors, stats)
    _check_visual_system(errors, stats)
    return {
        "ok": not errors,
        "errors": errors,
        "stats": stats,
        "boundary": "Repository contracts only; no hardware certification or universal semantic guarantee.",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", action="store_true", help="emit machine-readable JSON")
    args = parser.parse_args()
    report = audit_repository()

    if args.json:
        print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    elif report["ok"]:
        print("Repository contract audit: OK")
        for key, value in report["stats"].items():
            print(f"- {key}: {value}")
        print(f"Boundary: {report['boundary']}")
    else:
        print("Repository contract audit: FAILED")
        for error in report["errors"]:
            print(f"- {error}")

    return 0 if report["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
