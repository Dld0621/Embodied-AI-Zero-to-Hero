#!/usr/bin/env python3
"""Discover, validate, and run the repository's learning pipelines.

The manifest is intentionally data-only. Commands are executed as argument
arrays with ``shell=False`` so learners can inspect exactly what will run.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
MANIFEST_PATH = REPO_ROOT / "pipelines" / "manifest.json"
VALID_STATUSES = {
    "smoke-tested",
    "interface-tested",
    "documented",
    "experimental",
}
ID_PATTERN = re.compile(r"^[a-z0-9]+(?:-[a-z0-9]+)*$")


class ManifestError(ValueError):
    """Raised when the pipeline manifest is invalid."""


def load_manifest() -> dict[str, Any]:
    with MANIFEST_PATH.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, dict):
        raise ManifestError("manifest root must be a JSON object")
    return data


def _repo_path(relative_path: str, field: str) -> Path:
    candidate = (REPO_ROOT / relative_path).resolve()
    try:
        candidate.relative_to(REPO_ROOT.resolve())
    except ValueError as exc:
        raise ManifestError(f"{field} escapes repository root: {relative_path}") from exc
    return candidate


def _validate_command(command: Any, pipeline_id: str, mode: str) -> list[str]:
    if command is None:
        return []
    if not isinstance(command, dict):
        raise ManifestError(f"{pipeline_id}.{mode} must be an object or null")
    cwd = command.get("cwd")
    argv = command.get("command")
    if not isinstance(cwd, str) or not cwd:
        raise ManifestError(f"{pipeline_id}.{mode}.cwd must be a non-empty string")
    cwd_path = _repo_path(cwd, f"{pipeline_id}.{mode}.cwd")
    if not cwd_path.is_dir():
        raise ManifestError(f"{pipeline_id}.{mode}.cwd does not exist: {cwd}")
    if not isinstance(argv, list) or not argv or not all(
        isinstance(item, str) and item for item in argv
    ):
        raise ManifestError(
            f"{pipeline_id}.{mode}.command must be a non-empty string array"
        )
    if argv[0] != "{python}":
        raise ManifestError(f"{pipeline_id}.{mode} must start with {{python}}")
    return argv


def validate_manifest(data: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    try:
        if data.get("schema_version") != 1:
            raise ManifestError("schema_version must be 1")
        pipelines = data.get("pipelines")
        if not isinstance(pipelines, list) or not pipelines:
            raise ManifestError("pipelines must be a non-empty array")

        seen: set[str] = set()
        required_list_fields = ("requires", "artifacts", "metrics")
        for index, pipeline in enumerate(pipelines):
            if not isinstance(pipeline, dict):
                raise ManifestError(f"pipelines[{index}] must be an object")
            pipeline_id = pipeline.get("id")
            if not isinstance(pipeline_id, str) or not ID_PATTERN.fullmatch(pipeline_id):
                raise ManifestError(f"pipelines[{index}].id is invalid: {pipeline_id!r}")
            if pipeline_id in seen:
                raise ManifestError(f"duplicate pipeline id: {pipeline_id}")
            seen.add(pipeline_id)

            if pipeline.get("status") not in VALID_STATUSES:
                raise ManifestError(
                    f"{pipeline_id}.status must be one of {sorted(VALID_STATUSES)}"
                )
            for field in ("title", "document", "entrypoint"):
                value = pipeline.get(field)
                if not isinstance(value, str) or not value:
                    raise ManifestError(f"{pipeline_id}.{field} must be a string")
            for field in ("document", "entrypoint"):
                value = pipeline[field]
                if not _repo_path(value, f"{pipeline_id}.{field}").is_file():
                    raise ManifestError(f"{pipeline_id}.{field} does not exist: {value}")

            for field in required_list_fields:
                values = pipeline.get(field)
                if not isinstance(values, list) or not values or not all(
                    isinstance(value, str) and value for value in values
                ):
                    raise ManifestError(f"{pipeline_id}.{field} must be a non-empty string array")
            for required_path in pipeline["requires"]:
                if not _repo_path(required_path, f"{pipeline_id}.requires").is_file():
                    raise ManifestError(
                        f"{pipeline_id}.requires path does not exist: {required_path}"
                    )

            _validate_command(pipeline.get("smoke"), pipeline_id, "smoke")
            _validate_command(pipeline.get("full"), pipeline_id, "full")
    except ManifestError as exc:
        errors.append(str(exc))
    return errors


def pipelines_by_id(data: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {pipeline["id"]: pipeline for pipeline in data["pipelines"]}


def print_list(data: dict[str, Any]) -> None:
    print(f"{'ID':28} {'STATUS':18} TITLE")
    print("-" * 88)
    for pipeline in data["pipelines"]:
        print(f"{pipeline['id']:28} {pipeline['status']:18} {pipeline['title']}")


def print_pipeline(pipeline: dict[str, Any]) -> None:
    print(json.dumps(pipeline, indent=2, ensure_ascii=False))


def run_pipeline(pipeline: dict[str, Any], full: bool, dry_run: bool) -> int:
    mode = "full" if full else "smoke"
    command_spec = pipeline.get(mode)
    if command_spec is None:
        print(
            f"Pipeline '{pipeline['id']}' has no {mode} command. "
            f"Follow {pipeline['document']} manually.",
            file=sys.stderr,
        )
        return 2

    argv = [sys.executable if item == "{python}" else item for item in command_spec["command"]]
    cwd = _repo_path(command_spec["cwd"], f"{pipeline['id']}.{mode}.cwd")
    print(f"Pipeline : {pipeline['id']} ({mode})")
    print(f"Workdir  : {cwd}")
    print("Command  : " + " ".join(argv))
    if dry_run:
        return 0
    completed = subprocess.run(argv, cwd=cwd, check=False, shell=False)
    return completed.returncode


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="List, inspect, validate, or run a DoF learning pipeline."
    )
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--list", action="store_true", help="list all pipelines")
    action.add_argument("--show", metavar="ID", help="show one manifest entry")
    action.add_argument("--validate", action="store_true", help="validate the manifest")
    action.add_argument("--run", metavar="ID", help="run one pipeline")
    parser.add_argument("--full", action="store_true", help="use the full command instead of smoke")
    parser.add_argument("--dry-run", action="store_true", help="print the command without running it")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    try:
        data = load_manifest()
    except (OSError, json.JSONDecodeError, ManifestError) as exc:
        print(f"Manifest error: {exc}", file=sys.stderr)
        return 1

    errors = validate_manifest(data)
    if errors:
        for error in errors:
            print(f"Manifest error: {error}", file=sys.stderr)
        return 1

    if args.validate:
        print(f"OK: {len(data['pipelines'])} pipelines validated from {MANIFEST_PATH}")
        return 0
    if args.list:
        print_list(data)
        return 0

    index = pipelines_by_id(data)
    requested_id = args.show or args.run
    if requested_id not in index:
        print(f"Unknown pipeline: {requested_id}", file=sys.stderr)
        print(f"Available: {', '.join(index)}", file=sys.stderr)
        return 2
    if args.show:
        print_pipeline(index[requested_id])
        return 0
    return run_pipeline(index[requested_id], full=args.full, dry_run=args.dry_run)


if __name__ == "__main__":
    raise SystemExit(main())
