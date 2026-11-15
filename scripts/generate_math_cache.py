#!/usr/bin/env python3
"""Collect equations from a temporary MkDocs build and refresh their static cache.

Ordinary documentation builds use only Python and the committed cache. This
maintainer command needs the pinned development renderer installed by npm ci.
Both collection and verification builds use temporary site directories, never
the currently served site/ preview.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

from mkdocs_math import CACHE_PATH, COLLECT_ENV, ROOT, load_cache


def build(site_dir: Path, collection: Path | None = None) -> None:
    environment = dict(os.environ)
    environment.pop(COLLECT_ENV, None)
    if collection:
        environment[COLLECT_ENV] = str(collection)
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "mkdocs",
            "build",
            "--strict",
            "--clean",
            "--site-dir",
            str(site_dir),
        ],
        cwd=ROOT,
        env=environment,
        text=True,
        capture_output=True,
        timeout=180,
        check=False,
    )
    if completed.returncode:
        raise RuntimeError(completed.stdout + completed.stderr)


def generate(check: bool = False, node: str | None = None) -> int:
    with tempfile.TemporaryDirectory(prefix="embodied-static-math-") as temporary:
        directory = Path(temporary)
        collection_path = directory / "expressions.json"
        build(directory / "collect-site", collection_path)
        collection = json.loads(collection_path.read_text(encoding="utf-8"))
        if check:
            entries = load_cache()
            if set(entries) != set(collection["expressions"]):
                missing = set(collection["expressions"]) - set(entries)
                stale = set(entries) - set(collection["expressions"])
                raise RuntimeError(
                    f"Static math cache is out of date: {len(missing)} missing, {len(stale)} stale. "
                    "Run python scripts/generate_math_cache.py after npm ci."
                )
        else:
            executable = node or os.environ.get("EMBODIED_MATH_NODE") or shutil.which("node")
            if not executable:
                raise RuntimeError(
                    "Node.js is needed only to regenerate formulas; install Node and npm ci."
                )
            rendered = subprocess.run(
                [executable, str(ROOT / "scripts" / "render_math.cjs")],
                input=json.dumps(collection, ensure_ascii=False),
                cwd=ROOT,
                text=True,
                capture_output=True,
                timeout=180,
                check=False,
            )
            if rendered.returncode:
                raise RuntimeError(rendered.stderr)
            candidate = directory / "math-cache.json"
            candidate.write_text(rendered.stdout, encoding="utf-8")
            entries = load_cache(candidate)
            if set(entries) != set(collection["expressions"]):
                raise RuntimeError("Renderer did not return exactly the collected equations.")
            CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
            staging = CACHE_PATH.with_suffix(".json.tmp")
            staging.write_text(rendered.stdout, encoding="utf-8")
            staging.replace(CACHE_PATH)
        # Normal mode, no collector flag: fail rather than accepting raw math.
        build(directory / "verified-site")
        print(
            f"Static mathematics {'verified' if check else 'generated and verified'}: {len(entries)} unique expressions."
        )
        return len(entries)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check",
        action="store_true",
        help="Verify cache coverage without Node or rewriting files.",
    )
    parser.add_argument("--node", help="Optional path to the maintainer Node.js runtime.")
    arguments = parser.parse_args()
    try:
        generate(check=arguments.check, node=arguments.node)
    except (OSError, ValueError, RuntimeError, subprocess.TimeoutExpired) as error:
        print(error, file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
