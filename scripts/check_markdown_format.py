#!/usr/bin/env python3
"""Check or normalize first-party Markdown trailing whitespace."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def tracked_markdown_files() -> list[Path]:
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
            "*.md",
        ],
        check=True,
        capture_output=True,
    )
    paths = []
    for item in result.stdout.split(b"\0"):
        if not item:
            continue
        relative = item.decode("utf-8")
        if relative.startswith("pretrained/"):
            continue
        paths.append(ROOT / relative)
    return paths


def normalize_line(line: str) -> str:
    newline = ""
    body = line
    if body.endswith("\r\n"):
        body, newline = body[:-2], "\n"
    elif body.endswith("\n"):
        body, newline = body[:-1], "\n"

    # Preserve the CommonMark two-space hard break, but remove tabs, one
    # trailing space, or runs of three or more spaces.
    if body.endswith("  ") and not body.endswith("   ") and not body.endswith("\t  "):
        normalized = body
    else:
        normalized = body.rstrip(" \t")
    return normalized + newline


def format_markdown(*, write: bool) -> dict[str, object]:
    changed_files: list[str] = []
    changed_lines = 0
    for path in tracked_markdown_files():
        original_lines = path.read_text(encoding="utf-8").splitlines(keepends=True)
        normalized_lines = [normalize_line(line) for line in original_lines]
        if normalized_lines == original_lines:
            continue
        changed_files.append(path.relative_to(ROOT).as_posix())
        changed_lines += sum(
            before != after for before, after in zip(original_lines, normalized_lines)
        )
        if write:
            path.write_text("".join(normalized_lines), encoding="utf-8", newline="\n")
    return {
        "ok": not changed_files or write,
        "changed_files": changed_files,
        "changed_lines": changed_lines,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write", action="store_true", help="normalize files in place")
    args = parser.parse_args()
    report = format_markdown(write=args.write)
    if report["changed_files"]:
        action = "Normalized" if args.write else "Needs normalization"
        print(f"{action}: {len(report['changed_files'])} files, {report['changed_lines']} lines")
        for relative in report["changed_files"]:
            print(f"- {relative}")
    else:
        print("First-party Markdown format: OK")
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
