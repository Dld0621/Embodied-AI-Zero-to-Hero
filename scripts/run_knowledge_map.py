#!/usr/bin/env python3
"""Inspect and validate the bilingual Embodied AI knowledge graph."""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
KNOWLEDGE_MANIFEST = REPO_ROOT / "knowledge" / "manifest.json"
PIPELINE_MANIFEST = REPO_ROOT / "pipelines" / "manifest.json"
ID_PATTERN = re.compile(r"^[a-z0-9]+(?:-[a-z0-9]+)*$")
DATE_PATTERN = re.compile(r"^\d{4}-\d{2}-\d{2}$")
EVIDENCE_LEVELS = {"explain", "derive", "execute", "evaluate", "deployment-gate"}
NODE_TEXT_FIELDS = (
    "title",
    "title_zh",
    "outcome",
    "outcome_zh",
    "assessment",
    "assessment_zh",
)


class KnowledgeManifestError(ValueError):
    """Raised when the knowledge-system contract cannot be loaded."""


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8-sig") as handle:
        data = json.load(handle)
    if not isinstance(data, dict):
        raise KnowledgeManifestError(f"JSON root must be an object: {path}")
    return data


def load_manifest() -> dict[str, Any]:
    """Load the knowledge graph without weakening JSON parsing."""
    return _load_json(KNOWLEDGE_MANIFEST)


def _require_bilingual_text(
    item: dict[str, Any], fields: tuple[str, ...], label: str, errors: list[str]
) -> None:
    for field in fields:
        value = item.get(field)
        if not isinstance(value, str) or not value.strip():
            errors.append(f"{label}.{field} must be a non-empty string")


def _find_cycle(index: dict[str, dict[str, Any]]) -> list[str] | None:
    state: dict[str, int] = {}
    stack: list[str] = []

    def visit(node_id: str) -> list[str] | None:
        current = state.get(node_id, 0)
        if current == 2:
            return None
        if current == 1:
            start = stack.index(node_id)
            return [*stack[start:], node_id]
        state[node_id] = 1
        stack.append(node_id)
        for prerequisite in index[node_id].get("prerequisites", []):
            if prerequisite in index:
                cycle = visit(prerequisite)
                if cycle:
                    return cycle
        stack.pop()
        state[node_id] = 2
        return None

    for node_id in index:
        cycle = visit(node_id)
        if cycle:
            return cycle
    return None


def validate_manifest(data: dict[str, Any]) -> list[str]:
    """Return all locally checkable knowledge-graph contract errors."""
    errors: list[str] = []
    if data.get("schema_version") != 1:
        errors.append("schema_version must be 1")
    reviewed_on = data.get("reviewed_on")
    if not isinstance(reviewed_on, str) or not DATE_PATTERN.fullmatch(reviewed_on):
        errors.append("reviewed_on must use YYYY-MM-DD")
    for field in ("purpose", "boundary"):
        if not isinstance(data.get(field), str) or not data[field].strip():
            errors.append(f"{field} must be a non-empty string")

    stages = data.get("stages")
    if not isinstance(stages, list):
        return errors + ["stages must be an array"]
    if len(stages) != 6:
        errors.append(f"expected 6 stages, found {len(stages)}")
    stage_ids: list[int] = []
    for index, stage in enumerate(stages):
        if not isinstance(stage, dict):
            errors.append(f"stages[{index}] must be an object")
            continue
        stage_id = stage.get("id")
        if not isinstance(stage_id, int):
            errors.append(f"stages[{index}].id must be an integer")
            continue
        stage_ids.append(stage_id)
        _require_bilingual_text(
            stage, ("title", "title_zh", "exit", "exit_zh"), f"stage {stage_id}", errors
        )
    if sorted(stage_ids) != list(range(6)):
        errors.append("stage IDs must be exactly 0 through 5")

    domains = data.get("domains")
    if not isinstance(domains, list):
        return errors + ["domains must be an array"]
    if len(domains) != 9:
        errors.append(f"expected 9 knowledge domains, found {len(domains)}")
    domain_ids: list[str] = []
    for index, domain in enumerate(domains):
        if not isinstance(domain, dict):
            errors.append(f"domains[{index}] must be an object")
            continue
        domain_id = domain.get("id")
        if not isinstance(domain_id, str) or not ID_PATTERN.fullmatch(domain_id):
            errors.append(f"domains[{index}].id is invalid: {domain_id!r}")
            continue
        domain_ids.append(domain_id)
        _require_bilingual_text(
            domain,
            ("title", "title_zh", "question", "question_zh"),
            f"domain {domain_id}",
            errors,
        )
    if len(domain_ids) != len(set(domain_ids)):
        errors.append("domain IDs must be unique")

    try:
        pipeline_data = _load_json(PIPELINE_MANIFEST)
    except (OSError, json.JSONDecodeError, KnowledgeManifestError) as exc:
        return errors + [f"cannot load pipeline manifest: {exc}"]
    pipeline_ids = {
        pipeline.get("id")
        for pipeline in pipeline_data.get("pipelines", [])
        if isinstance(pipeline, dict) and isinstance(pipeline.get("id"), str)
    }

    nodes = data.get("nodes")
    if not isinstance(nodes, list):
        return errors + ["nodes must be an array"]
    if len(nodes) != 45:
        errors.append(f"expected 45 knowledge nodes, found {len(nodes)}")

    node_index: dict[str, dict[str, Any]] = {}
    covered_domains: set[str] = set()
    covered_pipelines: set[str] = set()
    for index, node in enumerate(nodes):
        if not isinstance(node, dict):
            errors.append(f"nodes[{index}] must be an object")
            continue
        node_id = node.get("id")
        if not isinstance(node_id, str) or not ID_PATTERN.fullmatch(node_id):
            errors.append(f"nodes[{index}].id is invalid: {node_id!r}")
            continue
        if node_id in node_index:
            errors.append(f"duplicate node id: {node_id}")
        node_index[node_id] = node
        _require_bilingual_text(node, NODE_TEXT_FIELDS, f"node {node_id}", errors)

        domain_id = node.get("domain")
        if domain_id not in domain_ids:
            errors.append(f"{node_id} references unknown domain: {domain_id}")
        elif isinstance(domain_id, str):
            covered_domains.add(domain_id)

        stage_id = node.get("stage")
        if stage_id not in stage_ids:
            errors.append(f"{node_id} references unknown stage: {stage_id}")

        document = node.get("document")
        if not isinstance(document, str) or not (REPO_ROOT / document).is_file():
            errors.append(f"{node_id} document does not exist: {document}")

        evidence = node.get("evidence")
        if evidence not in EVIDENCE_LEVELS:
            errors.append(f"{node_id} has unsupported learner evidence: {evidence}")

        prerequisites = node.get("prerequisites")
        if not isinstance(prerequisites, list) or not all(
            isinstance(item, str) for item in prerequisites
        ):
            errors.append(f"{node_id}.prerequisites must be a string array")

        linked_pipelines = node.get("pipelines")
        if not isinstance(linked_pipelines, list) or not linked_pipelines:
            errors.append(f"{node_id}.pipelines must be a non-empty array")
        elif not all(isinstance(item, str) for item in linked_pipelines):
            errors.append(f"{node_id}.pipelines must contain only strings")
        else:
            for pipeline_id in linked_pipelines:
                if pipeline_id not in pipeline_ids:
                    errors.append(f"{node_id} references unknown pipeline: {pipeline_id}")
                else:
                    covered_pipelines.add(pipeline_id)

    for node_id, node in node_index.items():
        node_stage = node.get("stage")
        for prerequisite in node.get("prerequisites", []):
            if prerequisite not in node_index:
                errors.append(f"{node_id} references unknown prerequisite: {prerequisite}")
                continue
            prerequisite_stage = node_index[prerequisite].get("stage")
            if isinstance(node_stage, int) and isinstance(prerequisite_stage, int):
                if prerequisite_stage > node_stage:
                    errors.append(
                        f"{node_id} depends on later-stage prerequisite {prerequisite}"
                    )

    cycle = _find_cycle(node_index)
    if cycle:
        errors.append("knowledge graph contains a cycle: " + " -> ".join(cycle))
    missing_domains = sorted(set(domain_ids).difference(covered_domains))
    if missing_domains:
        errors.append("knowledge domains without nodes: " + ", ".join(missing_domains))
    missing_pipelines = sorted(pipeline_ids.difference(covered_pipelines))
    if missing_pipelines:
        errors.append("pipelines without knowledge coverage: " + ", ".join(missing_pipelines))
    return errors


def _title(item: dict[str, Any], lang: str) -> str:
    return str(item["title_zh" if lang == "zh" else "title"])


def learning_order(target: str, nodes: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Return a stable prerequisite-first path for one target node."""
    index = {node["id"]: node for node in nodes}
    if target not in index:
        raise KeyError(target)
    ordered: list[dict[str, Any]] = []
    seen: set[str] = set()

    def visit(node_id: str) -> None:
        if node_id in seen:
            return
        for prerequisite in index[node_id]["prerequisites"]:
            visit(prerequisite)
        seen.add(node_id)
        ordered.append(index[node_id])

    visit(target)
    return ordered


def print_list(data: dict[str, Any], lang: str) -> None:
    nodes_by_domain: dict[str, list[dict[str, Any]]] = {}
    for node in data["nodes"]:
        nodes_by_domain.setdefault(node["domain"], []).append(node)
    for domain in data["domains"]:
        print(f"\n{_title(domain, lang)} [{domain['id']}]")
        for node in nodes_by_domain.get(domain["id"], []):
            print(f"  L{node['stage']}  {node['id']:38} {_title(node, lang)}")


def print_node(node: dict[str, Any], lang: str) -> None:
    zh = lang == "zh"
    print(f"{_title(node, lang)} [{node['id']}]")
    print(f"{'层级' if zh else 'Stage'}: L{node['stage']}")
    print(f"{'前置' if zh else 'Prerequisites'}: {', '.join(node['prerequisites']) or '—'}")
    print(f"{'文档' if zh else 'Document'}: {node['document']}")
    print(f"{'关联管线' if zh else 'Pipelines'}: {', '.join(node['pipelines'])}")
    print(f"{'学习证据' if zh else 'Learner evidence'}: {node['evidence']}")
    print(f"{'学习结果' if zh else 'Outcome'}: {node['outcome_zh' if zh else 'outcome']}")
    print(f"{'验收方式' if zh else 'Assessment'}: {node['assessment_zh' if zh else 'assessment']}")


def print_path(data: dict[str, Any], target: str, lang: str) -> None:
    path = learning_order(target, data["nodes"])
    label = "前置学习路径" if lang == "zh" else "Prerequisite learning path"
    print(f"{label}: {target} ({len(path)} nodes)")
    for index, node in enumerate(path, start=1):
        print(f"{index:02d}. L{node['stage']} {node['id']} — {_title(node, lang)}")


def print_stats(data: dict[str, Any]) -> None:
    by_stage = Counter(node["stage"] for node in data["nodes"])
    by_evidence = Counter(node["evidence"] for node in data["nodes"])
    print(
        f"{len(data['nodes'])} nodes · {len(data['domains'])} domains · "
        f"{len(data['stages'])} stages"
    )
    print("by stage: " + ", ".join(f"L{key}={value}" for key, value in sorted(by_stage.items())))
    print("by learner evidence: " + ", ".join(f"{key}={value}" for key, value in sorted(by_evidence.items())))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="List, inspect, validate, or resolve the Embodied AI knowledge graph."
    )
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--list", action="store_true", help="list all knowledge nodes")
    action.add_argument("--show", metavar="ID", help="show one knowledge-node contract")
    action.add_argument("--path-to", metavar="ID", help="resolve prerequisites for one node")
    action.add_argument("--stats", action="store_true", help="show graph coverage statistics")
    action.add_argument("--validate", action="store_true", help="validate the knowledge graph")
    parser.add_argument("--lang", choices=("en", "zh"), default="en", help="output language")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    try:
        data = load_manifest()
    except (OSError, json.JSONDecodeError, KnowledgeManifestError) as exc:
        print(f"Knowledge manifest error: {exc}", file=sys.stderr)
        return 1
    errors = validate_manifest(data)
    if errors:
        for error in errors:
            print(f"Knowledge manifest error: {error}", file=sys.stderr)
        return 1
    if args.validate:
        print(
            f"OK: {len(data['nodes'])} knowledge nodes across "
            f"{len(data['domains'])} domains validated from {KNOWLEDGE_MANIFEST}"
        )
        return 0
    if args.list:
        print_list(data, args.lang)
        return 0
    if args.stats:
        print_stats(data)
        return 0

    node_index = {node["id"]: node for node in data["nodes"]}
    target = args.show or args.path_to
    if target not in node_index:
        print(f"Unknown knowledge node: {target}", file=sys.stderr)
        print(f"Available: {', '.join(node_index)}", file=sys.stderr)
        return 2
    if args.show:
        print_node(node_index[target], args.lang)
    else:
        print_path(data, target, args.lang)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
