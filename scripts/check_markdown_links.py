"""Check repository-local Markdown links without making network requests."""

from __future__ import annotations

import re
import sys
from pathlib import Path
from urllib.parse import unquote


LINK = re.compile(r"!?\[[^\]]*\]\(([^)]+)\)")
FENCED_CODE = re.compile(r"```.*?```", re.DOTALL)
SKIP_PARTS = {".git", "pretrained"}


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    broken: list[str] = []

    for document in root.rglob("*.md"):
        if SKIP_PARTS.intersection(document.relative_to(root).parts):
            continue
        text = FENCED_CODE.sub("", document.read_text(encoding="utf-8"))
        for raw_target in LINK.findall(text):
            target = raw_target.strip().split()[0].strip("<>")
            if not target or target.startswith(("#", "http://", "https://", "mailto:")):
                continue
            relative_path = unquote(target.split("#", 1)[0])
            resolved = (document.parent / relative_path).resolve()
            # A few research guides intentionally point to sibling workspaces.
            # CI can only validate paths that belong to this repository.
            if not resolved.is_relative_to(root):
                continue
            if relative_path and not resolved.exists():
                broken.append(f"{document.relative_to(root)}: {target}")

    if broken:
        print("Broken repository-local Markdown links:")
        print("\n".join(f"- {item}" for item in broken))
        return 1

    print("Repository-local Markdown links: OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
