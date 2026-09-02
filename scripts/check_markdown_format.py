#!/usr/bin/env python3
"""Check first-party Markdown encoding, math delimiters, and whitespace."""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]

SUSPICIOUS_ENCODING = (
    "\ufffd",
    "锟斤拷",
    "Ã©",
    "Ã¨",
    "Ã ",
    "Â ",
    "â€",
    "â€™",
    "â€œ",
    "ðŸ",
)
MATH_TOKEN = re.compile(r"\\\(|\\\)|\\\[|\\\]|(?<!\\)\$\$")
INLINE_CODE = re.compile(r"(`+)(.*?)\1")


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


def _prose_without_fenced_or_inline_code(text: str) -> str:
    """Return Markdown prose while preserving line numbers."""
    lines: list[str] = []
    fence_marker: str | None = None
    for line in text.splitlines(keepends=True):
        stripped = line.lstrip()
        marker = None
        if stripped.startswith("```"):
            marker = "```"
        elif stripped.startswith("~~~"):
            marker = "~~~"

        if marker is not None:
            if fence_marker is None:
                fence_marker = marker
            elif marker == fence_marker:
                fence_marker = None
            lines.append("\n" if line.endswith(("\n", "\r")) else "")
            continue

        if fence_marker is not None:
            lines.append("\n" if line.endswith(("\n", "\r")) else "")
            continue
        lines.append(INLINE_CODE.sub("", line))
    return "".join(lines)


def _line_number(text: str, offset: int) -> int:
    return text.count("\n", 0, offset) + 1


def audit_text(text: str, relative: str) -> list[str]:
    """Find encoding damage and unpaired display/inline math delimiters."""
    errors: list[str] = []
    for marker in SUSPICIOUS_ENCODING:
        start = 0
        while True:
            offset = text.find(marker, start)
            if offset < 0:
                break
            errors.append(
                f"{relative}:{_line_number(text, offset)}: "
                f"suspicious encoding sequence {marker!r}"
            )
            start = offset + len(marker)

    for offset, char in enumerate(text):
        if ord(char) < 32 and char not in "\n\r\t":
            errors.append(
                f"{relative}:{_line_number(text, offset)}: "
                f"unexpected control character U+{ord(char):04X}"
            )

    prose = _prose_without_fenced_or_inline_code(text)
    stack: list[tuple[str, int]] = []
    dollar_offsets: list[int] = []
    expected_close = {"\\(": "\\)", "\\[": "\\]"}
    for match in MATH_TOKEN.finditer(prose):
        token = match.group(0)
        if token == "$$":
            dollar_offsets.append(match.start())
            continue
        if token in expected_close:
            stack.append((token, match.start()))
            continue
        if not stack or expected_close[stack[-1][0]] != token:
            errors.append(
                f"{relative}:{_line_number(prose, match.start())}: "
                f"unexpected math delimiter {token}"
            )
            continue
        stack.pop()

    for token, offset in stack:
        errors.append(
            f"{relative}:{_line_number(prose, offset)}: "
            f"unclosed math delimiter {token}"
        )
    if len(dollar_offsets) % 2:
        offset = dollar_offsets[-1]
        errors.append(
            f"{relative}:{_line_number(prose, offset)}: unpaired display math delimiter $$"
        )
    return errors


def format_markdown(*, write: bool) -> dict[str, object]:
    changed_files: list[str] = []
    changed_lines = 0
    errors: list[str] = []
    for path in tracked_markdown_files():
        original_lines = path.read_text(encoding="utf-8").splitlines(keepends=True)
        relative = path.relative_to(ROOT).as_posix()
        errors.extend(audit_text("".join(original_lines), relative))
        normalized_lines = [normalize_line(line) for line in original_lines]
        if normalized_lines == original_lines:
            continue
        changed_files.append(relative)
        changed_lines += sum(
            before != after for before, after in zip(original_lines, normalized_lines)
        )
        if write:
            path.write_text("".join(normalized_lines), encoding="utf-8", newline="\n")
    return {
        "ok": (not changed_files or write) and not errors,
        "changed_files": changed_files,
        "changed_lines": changed_lines,
        "errors": errors,
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
        print("First-party Markdown whitespace: OK")
    if report["errors"]:
        print(f"Markdown encoding/math errors: {len(report['errors'])}")
        for error in report["errors"]:
            print(f"- {error}")
    else:
        print("First-party Markdown encoding/math: OK")
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
