#!/usr/bin/env python3
"""Plan and audit a gate-based journey from beginner to Embodied AI expert.

The runner treats completion as an evidence claim, not a time-spent claim. It
can validate the curriculum contract, print a self-diagnostic, build a study
plan, initialize a progress record, and audit an existing record.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from datetime import date
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
CURRICULUM_MANIFEST = ROOT / "curriculum" / "manifest.json"
QUALITY_RUBRIC = ROOT / "curriculum" / "quality_rubric.json"
KNOWLEDGE_MANIFEST = ROOT / "knowledge" / "manifest.json"
ID_PATTERN = re.compile(r"^[A-Z][0-9]{2}$")
LEVEL_PATTERN = re.compile(r"^L[0-5]$")
DATE_PATTERN = re.compile(r"^\d{4}-\d{2}-\d{2}$")
VALID_STATUSES = {"not_started", "in_progress", "passed", "blocked"}


class CurriculumError(ValueError):
    """Raised when a curriculum or progress contract is malformed."""


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8-sig") as handle:
        data = json.load(handle)
    if not isinstance(data, dict):
        raise CurriculumError(f"JSON root must be an object: {path}")
    return data


def load_manifest() -> dict[str, Any]:
    """Load the curriculum contract."""
    return _load_json(CURRICULUM_MANIFEST)


def _text(value: object) -> bool:
    return isinstance(value, str) and bool(value.strip())


def validate_manifest(data: dict[str, Any]) -> list[str]:
    """Return every locally checkable curriculum-contract error."""
    errors: list[str] = []
    if data.get("schema_version") != 1:
        errors.append("schema_version must be 1")
    if not _text(data.get("boundary")):
        errors.append("boundary must be a non-empty string")

    levels = data.get("levels")
    if not isinstance(levels, list):
        return errors + ["levels must be an array"]
    level_ids = [item.get("id") for item in levels if isinstance(item, dict)]
    if level_ids != [f"L{index}" for index in range(6)]:
        errors.append("levels must be ordered exactly from L0 through L5")
    for index, level in enumerate(levels):
        if not isinstance(level, dict):
            errors.append(f"levels[{index}] must be an object")
            continue
        if not LEVEL_PATTERN.fullmatch(str(level.get("id", ""))):
            errors.append(f"levels[{index}].id is invalid")
        for field in ("title", "title_zh", "exit", "exit_zh"):
            if not _text(level.get(field)):
                errors.append(f"{level.get('id', index)}.{field} must be non-empty")

    try:
        knowledge = _load_json(KNOWLEDGE_MANIFEST)
    except (OSError, json.JSONDecodeError, CurriculumError) as exc:
        return errors + [f"cannot load knowledge manifest: {exc}"]
    knowledge_nodes = {
        node.get("id") for node in knowledge.get("nodes", []) if isinstance(node, dict)
    }

    modules = data.get("modules")
    if not isinstance(modules, list):
        return errors + ["modules must be an array"]
    module_ids: list[str] = []
    mapped_nodes: list[str] = []
    for index, module in enumerate(modules):
        if not isinstance(module, dict):
            errors.append(f"modules[{index}] must be an object")
            continue
        module_id = str(module.get("id", ""))
        module_ids.append(module_id)
        if not ID_PATTERN.fullmatch(module_id):
            errors.append(f"modules[{index}].id is invalid: {module_id!r}")
        if module.get("level") not in level_ids:
            errors.append(f"{module_id}.level references an unknown level")
        if not isinstance(module.get("hours"), int) or module["hours"] <= 0:
            errors.append(f"{module_id}.hours must be a positive integer")
        for field in ("title", "title_zh", "artifact", "artifact_zh", "gate", "gate_zh"):
            if not _text(module.get(field)):
                errors.append(f"{module_id}.{field} must be non-empty")

        nodes = module.get("knowledge_nodes")
        if not isinstance(nodes, list) or not nodes:
            errors.append(f"{module_id}.knowledge_nodes must be a non-empty array")
        else:
            for node_id in nodes:
                if node_id not in knowledge_nodes:
                    errors.append(f"{module_id} references unknown knowledge node: {node_id}")
                elif isinstance(node_id, str):
                    mapped_nodes.append(node_id)

        documents = module.get("documents")
        if not isinstance(documents, list) or not documents:
            errors.append(f"{module_id}.documents must be a non-empty array")
        else:
            for relative in documents:
                if not isinstance(relative, str) or not (ROOT / relative).is_file():
                    errors.append(f"{module_id} document does not exist: {relative}")

    if len(module_ids) != len(set(module_ids)):
        errors.append("module IDs must be unique")
    expected_modules = [f"M{index:02d}" for index in range(12)]
    if module_ids != expected_modules:
        errors.append("modules must be ordered exactly from M00 through M11")
    duplicate_nodes = sorted({node for node in mapped_nodes if mapped_nodes.count(node) > 1})
    if duplicate_nodes:
        errors.append("knowledge nodes mapped more than once: " + ", ".join(duplicate_nodes))
    missing_nodes = sorted(knowledge_nodes.difference(mapped_nodes))
    if missing_nodes:
        errors.append("curriculum does not cover knowledge nodes: " + ", ".join(missing_nodes))

    capstones = data.get("capstones")
    if not isinstance(capstones, list) or not capstones:
        return errors + ["capstones must be a non-empty array"]
    capstone_ids: list[str] = []
    module_id_set = set(module_ids)
    for index, capstone in enumerate(capstones):
        if not isinstance(capstone, dict):
            errors.append(f"capstones[{index}] must be an object")
            continue
        capstone_id = str(capstone.get("id", ""))
        capstone_ids.append(capstone_id)
        for field in ("title", "title_zh"):
            if not _text(capstone.get(field)):
                errors.append(f"{capstone_id}.{field} must be non-empty")
        required = capstone.get("required_modules")
        if not isinstance(required, list) or not required:
            errors.append(f"{capstone_id}.required_modules must be non-empty")
        else:
            unknown = sorted(set(required).difference(module_id_set))
            if unknown:
                errors.append(f"{capstone_id} references unknown modules: {unknown}")
        score = capstone.get("pass_score")
        if not isinstance(score, int) or not 0 <= score <= 100:
            errors.append(f"{capstone_id}.pass_score must be an integer from 0 to 100")
        reviewers = capstone.get("independent_reviewers")
        if not isinstance(reviewers, int) or reviewers < 1:
            errors.append(f"{capstone_id}.independent_reviewers must be positive")
    if len(capstone_ids) != len(set(capstone_ids)):
        errors.append("capstone IDs must be unique")

    goals = data.get("goals")
    if not isinstance(goals, list) or not goals:
        return errors + ["goals must be a non-empty array"]
    goal_ids: list[str] = []
    for index, goal in enumerate(goals):
        if not isinstance(goal, dict):
            errors.append(f"goals[{index}] must be an object")
            continue
        goal_id = str(goal.get("id", ""))
        goal_ids.append(goal_id)
        for field in ("title", "title_zh"):
            if not _text(goal.get(field)):
                errors.append(f"{goal_id}.{field} must be non-empty")
        path = goal.get("modules")
        if not isinstance(path, list) or not path:
            errors.append(f"{goal_id}.modules must be a non-empty array")
            continue
        if len(path) != len(set(path)):
            errors.append(f"{goal_id}.modules contains duplicates")
        unknown = sorted(set(path).difference(module_id_set))
        if unknown:
            errors.append(f"{goal_id} references unknown modules: {unknown}")
        capstone_id = goal.get("capstone")
        if capstone_id not in capstone_ids:
            errors.append(f"{goal_id} references unknown capstone: {capstone_id}")
            continue
        capstone = next(item for item in capstones if item.get("id") == capstone_id)
        missing_required = sorted(set(capstone["required_modules"]).difference(path))
        if missing_required:
            errors.append(f"{goal_id} omits capstone requirements: {missing_required}")
    if len(goal_ids) != len(set(goal_ids)):
        errors.append("goal IDs must be unique")

    try:
        rubric = _load_json(QUALITY_RUBRIC)
    except (OSError, json.JSONDecodeError, CurriculumError) as exc:
        return errors + [f"cannot load quality rubric: {exc}"]
    if rubric.get("schema_version") != 1:
        errors.append("quality rubric schema_version must be 1")
    if not isinstance(rubric.get("reviewed_on"), str) or not DATE_PATTERN.fullmatch(
        rubric["reviewed_on"]
    ):
        errors.append("quality rubric reviewed_on must use YYYY-MM-DD")
    if not re.fullmatch(r"[0-9a-f]{40}", str(rubric.get("baseline_commit", ""))):
        errors.append("quality rubric baseline_commit must be a full Git commit hash")
    if not _text(rubric.get("boundary")):
        errors.append("quality rubric boundary must be non-empty")
    criteria = rubric.get("criteria")
    if not isinstance(criteria, list):
        return errors + ["quality rubric criteria must be an array"]
    if len(criteria) != 10:
        errors.append(f"quality rubric must contain 10 criteria, found {len(criteria)}")
    criterion_ids: list[str] = []
    before_total = 0
    after_total = 0
    for index, criterion in enumerate(criteria):
        if not isinstance(criterion, dict):
            errors.append(f"quality criterion {index} must be an object")
            continue
        criterion_id = str(criterion.get("id", ""))
        criterion_ids.append(criterion_id)
        for field in ("name", "name_zh", "gap"):
            if not _text(criterion.get(field)):
                errors.append(f"quality criterion {criterion_id}.{field} must be non-empty")
        for field in ("before", "after"):
            score = criterion.get(field)
            if not isinstance(score, int) or not 0 <= score <= 10:
                errors.append(f"quality criterion {criterion_id}.{field} must be 0..10")
        before_total += criterion.get("before", 0) if isinstance(criterion.get("before"), int) else 0
        after_total += criterion.get("after", 0) if isinstance(criterion.get("after"), int) else 0
        evidence = criterion.get("evidence")
        if not isinstance(evidence, list) or not evidence:
            errors.append(f"quality criterion {criterion_id}.evidence must be non-empty")
        else:
            for relative in evidence:
                if not isinstance(relative, str) or not (ROOT / relative).is_file():
                    errors.append(f"quality criterion {criterion_id} evidence missing: {relative}")
    if len(criterion_ids) != len(set(criterion_ids)):
        errors.append("quality criterion IDs must be unique")
    if before_total != 85:
        errors.append(f"reviewed baseline score must total 85, found {before_total}")
    if after_total != 100 or any(
        isinstance(item, dict) and item.get("after") != 10 for item in criteria
    ):
        errors.append(f"implemented quality score must be 10 per criterion and 100 total, found {after_total}")
    return errors


def _index(data: dict[str, Any], key: str) -> dict[str, dict[str, Any]]:
    return {str(item["id"]): item for item in data[key]}


def _localized(item: dict[str, Any], field: str, lang: str) -> str:
    return str(item[f"{field}_zh" if lang == "zh" else field])


def print_goals(data: dict[str, Any], lang: str) -> None:
    label = "目标" if lang == "zh" else "Goal"
    print(f"{'ID':28} {label}")
    print("-" * 76)
    for goal in data["goals"]:
        print(f"{goal['id']:28} {_localized(goal, 'title', lang)}")


def print_diagnostic(data: dict[str, Any], lang: str) -> None:
    zh = lang == "zh"
    print("证据自测：只有能够展示产物并通过门禁，才标记已完成。" if zh else
          "Evidence diagnostic: mark a module complete only when you can show the artifact and pass the gate.")
    for module in data["modules"]:
        print(f"\n[{module['id']} · {module['level']}] {_localized(module, 'title', lang)}")
        print(("产物: " if zh else "Artifact: ") + _localized(module, "artifact", lang))
        print(("门禁: " if zh else "Gate: ") + _localized(module, "gate", lang))


def _parse_completed(value: str | None) -> set[str]:
    if not value:
        return set()
    return {item.strip() for item in value.split(",") if item.strip()}


def print_plan(
    data: dict[str, Any], goal_id: str, hours_per_week: int, completed: set[str], lang: str
) -> int:
    goals = _index(data, "goals")
    modules = _index(data, "modules")
    capstones = _index(data, "capstones")
    goal = goals.get(goal_id)
    if goal is None:
        print(f"Unknown goal: {goal_id}. Available: {', '.join(goals)}", file=sys.stderr)
        return 2
    unknown = sorted(completed.difference(modules))
    if unknown:
        print(f"Unknown completed modules: {', '.join(unknown)}", file=sys.stderr)
        return 2
    if hours_per_week <= 0:
        print("hours-per-week must be positive", file=sys.stderr)
        return 2

    zh = lang == "zh"
    remaining = [modules[module_id] for module_id in goal["modules"] if module_id not in completed]
    total_hours = sum(module["hours"] for module in remaining)
    weeks = math.ceil(total_hours / hours_per_week) if total_hours else 0
    print(f"{_localized(goal, 'title', lang)} [{goal_id}]")
    print(("节奏: " if zh else "Pace: ") + f"{hours_per_week} h/week")
    print(("剩余: " if zh else "Remaining: ") + f"{total_hours} h, ~{weeks} weeks")
    print("\n" + ("顺序" if zh else "Sequence"))
    for number, module in enumerate(remaining, start=1):
        print(
            f"{number:02d}. {module['id']} · {module['level']} · "
            f"{_localized(module, 'title', lang)} · {module['hours']} h"
        )
        print(("    验收: " if zh else "    Gate: ") + _localized(module, "gate", lang))
    if not remaining:
        print("所有课程模块均已声明完成；请审计证据并进入 Capstone。" if zh else
              "All course modules are declared complete; audit the evidence and enter the capstone.")
    capstone = capstones[goal["capstone"]]
    print("\n" + ("毕业项目: " if zh else "Capstone: ") + _localized(capstone, "title", lang))
    print(
        ("通过条件: " if zh else "Pass: ")
        + f"{capstone['pass_score']}/100, "
        + f"{capstone['independent_reviewers']} "
        + ("名独立评审者" if zh else "independent reviewer(s)")
    )
    print("\n" + ("提示: --completed 只用于规划，不构成通过证据。" if zh else
                    "Note: --completed helps planning; it is not proof of passing."))
    return 0


def initialize_progress(data: dict[str, Any], path: Path, goal_id: str, learner: str) -> int:
    goals = _index(data, "goals")
    if goal_id not in goals:
        print(f"Unknown goal: {goal_id}. Available: {', '.join(goals)}", file=sys.stderr)
        return 2
    if path.exists():
        print(f"Refusing to overwrite existing progress file: {path}", file=sys.stderr)
        return 2
    path.parent.mkdir(parents=True, exist_ok=True)
    record = {
        "schema_version": 1,
        "learner": learner,
        "goal": goal_id,
        "started_on": date.today().isoformat(),
        "modules": {
            module_id: {
                "status": "not_started",
                "evidence": [],
                "reviewer": "",
                "reviewed_on": "",
            }
            for module_id in goals[goal_id]["modules"]
        },
        "capstones": {},
    }
    path.write_text(json.dumps(record, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Initialized learner progress: {path}")
    return 0


def validate_progress(data: dict[str, Any], record: dict[str, Any], path: Path) -> list[str]:
    errors: list[str] = []
    if record.get("schema_version") != 1:
        errors.append("progress schema_version must be 1")
    goals = _index(data, "goals")
    goal_id = record.get("goal")
    if goal_id not in goals:
        return errors + [f"progress references unknown goal: {goal_id}"]
    if not _text(record.get("learner")):
        errors.append("learner must be non-empty")
    started_on = record.get("started_on")
    if not isinstance(started_on, str) or not DATE_PATTERN.fullmatch(started_on):
        errors.append("started_on must use YYYY-MM-DD")

    module_records = record.get("modules")
    if not isinstance(module_records, dict):
        return errors + ["progress modules must be an object"]
    expected = set(goals[str(goal_id)]["modules"])
    found = set(module_records)
    if found != expected:
        errors.append(
            "progress module set differs from goal; missing="
            f"{sorted(expected.difference(found))}, extra={sorted(found.difference(expected))}"
        )
    for module_id, state in module_records.items():
        if not isinstance(state, dict):
            errors.append(f"{module_id} progress must be an object")
            continue
        status = state.get("status")
        if status not in VALID_STATUSES:
            errors.append(f"{module_id}.status is invalid: {status}")
        evidence = state.get("evidence")
        if not isinstance(evidence, list) or not all(_text(item) for item in evidence):
            errors.append(f"{module_id}.evidence must be a string array")
            evidence = []
        if status == "passed":
            if not evidence:
                errors.append(f"{module_id} is passed but has no evidence")
            if not _text(state.get("reviewer")):
                errors.append(f"{module_id} is passed but has no reviewer")
            reviewed_on = state.get("reviewed_on")
            if not isinstance(reviewed_on, str) or not DATE_PATTERN.fullmatch(reviewed_on):
                errors.append(f"{module_id} is passed but reviewed_on is not YYYY-MM-DD")
            for entry in evidence:
                if entry.startswith("https://"):
                    continue
                target = (path.parent / entry).resolve()
                if not target.exists():
                    errors.append(f"{module_id} evidence does not exist: {entry}")
    return errors


def print_progress(data: dict[str, Any], record: dict[str, Any], path: Path, lang: str) -> int:
    errors = validate_progress(data, record, path)
    if errors:
        for error in errors:
            print(f"Progress error: {error}", file=sys.stderr)
        return 1
    goals = _index(data, "goals")
    modules = _index(data, "modules")
    capstones = _index(data, "capstones")
    goal = goals[record["goal"]]
    states = record["modules"]
    passed = {module_id for module_id, state in states.items() if state["status"] == "passed"}
    total_hours = sum(modules[module_id]["hours"] for module_id in goal["modules"])
    passed_hours = sum(modules[module_id]["hours"] for module_id in passed)
    percent = 100.0 * passed_hours / total_hours
    zh = lang == "zh"
    print(("学习者: " if zh else "Learner: ") + record["learner"])
    print(("目标: " if zh else "Goal: ") + _localized(goal, "title", lang))
    print(("证据加权进度: " if zh else "Evidence-weighted progress: ") + f"{percent:.1f}%")
    print(("已通过模块: " if zh else "Passed modules: ") + f"{len(passed)}/{len(goal['modules'])}")
    next_module = next((module_id for module_id in goal["modules"] if module_id not in passed), None)
    if next_module:
        print(("下一项: " if zh else "Next: ") + f"{next_module} · {_localized(modules[next_module], 'title', lang)}")
    capstone = capstones[goal["capstone"]]
    eligible = set(capstone["required_modules"]).issubset(passed)
    print(("Capstone 资格: " if zh else "Capstone eligibility: ") + ("已满足" if zh and eligible else
          "未满足" if zh else "eligible" if eligible else "not yet"))
    print(("边界: " if zh else "Boundary: ") + ("进度记录通过不等于专家认证或真机授权。" if zh else
          "A valid progress record is not expert certification or hardware authorization."))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Plan, track, and validate the evidence-gated Embodied AI curriculum."
    )
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--validate", action="store_true", help="validate curriculum contracts")
    action.add_argument("--list-goals", action="store_true", help="list available learner goals")
    action.add_argument("--diagnose", action="store_true", help="print the evidence self-diagnostic")
    action.add_argument("--plan", metavar="GOAL", help="print an ordered study plan")
    action.add_argument("--init-progress", metavar="PATH", help="create a progress JSON file")
    action.add_argument("--report-progress", metavar="PATH", help="audit and summarize progress JSON")
    parser.add_argument("--goal", default="full-stack-expert", help="goal for --init-progress")
    parser.add_argument("--learner", default="learner", help="learner name for --init-progress")
    parser.add_argument("--hours-per-week", type=int, default=8, help="planning pace")
    parser.add_argument("--completed", help="comma-separated module IDs for planning only")
    parser.add_argument("--lang", choices=("en", "zh"), default="en", help="output language")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    try:
        data = load_manifest()
    except (OSError, json.JSONDecodeError, CurriculumError) as exc:
        print(f"Curriculum error: {exc}", file=sys.stderr)
        return 1
    errors = validate_manifest(data)
    if errors:
        for error in errors:
            print(f"Curriculum error: {error}", file=sys.stderr)
        return 1
    if args.validate:
        node_count = sum(len(module["knowledge_nodes"]) for module in data["modules"])
        rubric = _load_json(QUALITY_RUBRIC)
        before_total = sum(item["before"] for item in rubric["criteria"])
        after_total = sum(item["after"] for item in rubric["criteria"])
        print(
            f"OK: {len(data['modules'])} modules, {len(data['levels'])} levels, "
            f"{node_count} knowledge nodes, and {len(data['capstones'])} capstones validated "
            f"from {CURRICULUM_MANIFEST}"
        )
        print(f"Quality contract: {before_total}/100 -> {after_total}/100 across 10 criteria")
        print(f"Boundary: {data['boundary']}")
        return 0
    if args.list_goals:
        print_goals(data, args.lang)
        return 0
    if args.diagnose:
        print_diagnostic(data, args.lang)
        return 0
    if args.plan:
        return print_plan(
            data,
            args.plan,
            args.hours_per_week,
            _parse_completed(args.completed),
            args.lang,
        )
    if args.init_progress:
        return initialize_progress(data, Path(args.init_progress), args.goal, args.learner)
    try:
        progress_path = Path(args.report_progress)
        record = _load_json(progress_path)
    except (OSError, json.JSONDecodeError, CurriculumError) as exc:
        print(f"Progress error: {exc}", file=sys.stderr)
        return 1
    return print_progress(data, record, progress_path, args.lang)


if __name__ == "__main__":
    raise SystemExit(main())
