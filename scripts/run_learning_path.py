#!/usr/bin/env python3
"""Discover and validate the seven goal-oriented research routes."""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
ROUTE_MANIFEST = REPO_ROOT / "learning_paths" / "manifest.json"
PIPELINE_MANIFEST = REPO_ROOT / "pipelines" / "manifest.json"
ID_PATTERN = re.compile(r"^[a-z0-9]+(?:-[a-z0-9]+)*$")
REQUIRED_TEXT_FIELDS = (
    "title",
    "title_zh",
    "question",
    "question_zh",
    "deliverable",
    "deliverable_zh",
    "promotion_gate",
    "promotion_gate_zh",
    "boundary",
    "boundary_zh",
)


class RouteManifestError(ValueError):
    """Raised when the research-route contract is invalid."""


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8-sig") as handle:
        data = json.load(handle)
    if not isinstance(data, dict):
        raise RouteManifestError(f"JSON root must be an object: {path}")
    return data


def load_manifest() -> dict[str, Any]:
    """Load the route manifest without weakening JSON parsing."""
    return _load_json(ROUTE_MANIFEST)


def validate_manifest(data: dict[str, Any]) -> list[str]:
    """Return all route-contract errors that can be checked locally."""
    errors: list[str] = []
    if data.get("schema_version") != 1:
        errors.append("schema_version must be 1")

    routes = data.get("routes")
    if not isinstance(routes, list):
        return errors + ["routes must be an array"]
    if len(routes) != 7:
        errors.append(f"expected 7 research routes, found {len(routes)}")

    try:
        pipeline_data = _load_json(PIPELINE_MANIFEST)
    except (OSError, json.JSONDecodeError, RouteManifestError) as exc:
        return errors + [f"cannot load pipeline manifest: {exc}"]
    pipelines = pipeline_data.get("pipelines", [])
    pipeline_ids = {
        item.get("id") for item in pipelines if isinstance(item, dict)
    }

    seen: set[str] = set()
    covered_pipelines: set[str] = set()
    for index, route in enumerate(routes):
        if not isinstance(route, dict):
            errors.append(f"routes[{index}] must be an object")
            continue
        route_id = route.get("id")
        if not isinstance(route_id, str) or not ID_PATTERN.fullmatch(route_id):
            errors.append(f"routes[{index}].id is invalid: {route_id!r}")
            continue
        if route_id in seen:
            errors.append(f"duplicate route id: {route_id}")
        seen.add(route_id)

        for field in REQUIRED_TEXT_FIELDS:
            value = route.get(field)
            if not isinstance(value, str) or not value.strip():
                errors.append(f"{route_id}.{field} must be a non-empty string")

        foundations = route.get("foundations")
        if not isinstance(foundations, list) or not foundations:
            errors.append(f"{route_id}.foundations must be a non-empty array")
        else:
            for relative in foundations:
                if not isinstance(relative, str) or not (REPO_ROOT / relative).is_file():
                    errors.append(f"{route_id}.foundations path does not exist: {relative}")

        route_pipelines = route.get("pipelines")
        if not isinstance(route_pipelines, list) or not route_pipelines:
            errors.append(f"{route_id}.pipelines must be a non-empty array")
        else:
            for pipeline_id in route_pipelines:
                if pipeline_id not in pipeline_ids:
                    errors.append(f"{route_id} references unknown pipeline: {pipeline_id}")
                elif isinstance(pipeline_id, str):
                    covered_pipelines.add(pipeline_id)

        metrics = route.get("metrics")
        if not isinstance(metrics, list) or not metrics or not all(
            isinstance(metric, str) and metric for metric in metrics
        ):
            errors.append(f"{route_id}.metrics must be a non-empty string array")

    missing_coverage = sorted(pipeline_ids.difference(covered_pipelines))
    if missing_coverage:
        errors.append(
            "research routes do not cover registered pipelines: "
            + ", ".join(missing_coverage)
        )
    return errors


def _route_title(route: dict[str, Any], lang: str) -> str:
    return str(route["title_zh" if lang == "zh" else "title"])


def print_list(data: dict[str, Any], lang: str) -> None:
    """Print a compact, goal-oriented route index."""
    label = "研究问题" if lang == "zh" else "Research question"
    print(f"{'ID':30} {'ROUTE' if lang == 'en' else '方向'}")
    print("-" * 88)
    for route in data["routes"]:
        question = route["question_zh" if lang == "zh" else "question"]
        print(f"{route['id']:30} {_route_title(route, lang)}")
        print(f"  {label}: {question}")


def print_route(route: dict[str, Any], lang: str) -> None:
    """Print one route as a practical experiment brief."""
    zh = lang == "zh"
    labels = {
        "question": "研究问题" if zh else "Research question",
        "foundations": "基础课程" if zh else "Foundations",
        "pipelines": "工程管线" if zh else "Pipelines",
        "deliverable": "交付物" if zh else "Deliverable",
        "metrics": "核心指标" if zh else "Core metrics",
        "gate": "晋级门槛" if zh else "Promotion gate",
        "boundary": "证据边界" if zh else "Evidence boundary",
    }
    print(f"{_route_title(route, lang)} [{route['id']}]")
    print(f"{labels['question']}: {route['question_zh' if zh else 'question']}")
    print(f"{labels['foundations']}: " + ", ".join(route["foundations"]))
    print(f"{labels['pipelines']}: " + " -> ".join(route["pipelines"]))
    print(f"{labels['deliverable']}: {route['deliverable_zh' if zh else 'deliverable']}")
    print(f"{labels['metrics']}: " + ", ".join(route["metrics"]))
    print(f"{labels['gate']}: {route['promotion_gate_zh' if zh else 'promotion_gate']}")
    print(f"{labels['boundary']}: {route['boundary_zh' if zh else 'boundary']}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="List, inspect, or validate the seven Embodied AI research routes."
    )
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--list", action="store_true", help="list all research routes")
    action.add_argument("--show", metavar="ID", help="show one route as an experiment brief")
    action.add_argument("--validate", action="store_true", help="validate route contracts")
    parser.add_argument("--lang", choices=("en", "zh"), default="en", help="output language")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    try:
        data = load_manifest()
    except (OSError, json.JSONDecodeError, RouteManifestError) as exc:
        print(f"Route manifest error: {exc}", file=sys.stderr)
        return 1

    errors = validate_manifest(data)
    if errors:
        for error in errors:
            print(f"Route manifest error: {error}", file=sys.stderr)
        return 1
    if args.validate:
        print(f"OK: {len(data['routes'])} research routes validated from {ROUTE_MANIFEST}")
        return 0
    if args.list:
        print_list(data, args.lang)
        return 0

    index = {route["id"]: route for route in data["routes"]}
    route = index.get(args.show)
    if route is None:
        print(f"Unknown research route: {args.show}", file=sys.stderr)
        print(f"Available: {', '.join(index)}", file=sys.stderr)
        return 2
    print_route(route, args.lang)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
