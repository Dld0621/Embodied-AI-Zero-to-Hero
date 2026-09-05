#!/usr/bin/env python3
"""Audit built HTML, independently of the renderer and without a browser.

This proves static formula coverage, not browser layout or font availability.
Run after ``python -m mkdocs build --strict --clean``.
"""

from __future__ import annotations

import argparse
import logging
import os
import re
from html.parser import HTMLParser
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW_TEX = re.compile(r"\\[()[\]]|\\[A-Za-z]{2,}")
VOID = {
    "area",
    "base",
    "br",
    "col",
    "embed",
    "hr",
    "img",
    "input",
    "link",
    "meta",
    "param",
    "source",
    "track",
    "wbr",
}


class MathAudit(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.stack: list[tuple[str, set[str]]] = []
        self.formula: dict | None = None
        self.count = 0
        self.errors: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        attributes = dict(attrs)
        classes = set((attributes.get("class") or "").split())
        if tag == "script" and any(
            token in (attributes.get("src") or "").lower()
            for token in ("mathjax", "unpkg", "katex")
        ):
            self.errors.append("runtime math dependency remains")
        if "arithmatex" in classes:
            if self.formula is not None:
                self.errors.append("nested formula wrapper")
            self.count += 1
            self.formula = {
                "depth": len(self.stack),
                "svg": False,
                "mathml": False,
                "ink": False,
                "raw": [],
            }
            if attributes.get("data-math-rendered") != "static-svg":
                self.errors.append(f"formula {self.count}: missing static render marker")
        if self.formula is not None:
            inside_visual = any("math-visual" in entry[1] for entry in self.stack)
            inside_assistive = any("math-assistive" in entry[1] for entry in self.stack)
            if tag == "svg" and inside_visual:
                self.formula["svg"] = True
            if tag in {"path", "text", "rect", "line"} and inside_visual:
                self.formula["ink"] = True
            if tag == "math" and inside_assistive:
                self.formula["mathml"] = True
            if tag == "merror" or "data-mjx-error" in attributes:
                self.errors.append(f"formula {self.count}: renderer error element")
        if tag not in VOID:
            self.stack.append((tag, classes))

    def handle_endtag(self, tag: str) -> None:
        if tag in VOID:
            return
        for index in range(len(self.stack) - 1, -1, -1):
            if self.stack[index][0] == tag:
                if self.formula is not None and index <= self.formula["depth"]:
                    for key in ("svg", "mathml", "ink"):
                        if not self.formula[key]:
                            self.errors.append(f"formula {self.count}: missing {key}")
                    if "".join(self.formula["raw"]).strip():
                        self.errors.append(
                            f"formula {self.count}: visible raw content outside rendered math"
                        )
                    self.formula = None
                del self.stack[index:]
                break

    def handle_data(self, data: str) -> None:
        if "\ufffd" in data or "锟斤拷" in data:
            self.errors.append("replacement / mojibake character in visible source")
        if (
            self.formula is None
            and not any(
                tag in {"code", "pre", "script", "style", "textarea", "svg", "math"}
                for tag, _classes in self.stack
            )
            and RAW_TEX.search(data)
        ):
            self.errors.append(
                f"raw TeX outside rendered math (including headings/TOC): {data.strip()[:100]!r}"
            )
        if self.formula is not None and not any(
            {"math-visual", "math-assistive"} & entry[1] for entry in self.stack
        ):
            self.formula["raw"].append(data)

    def close(self) -> None:
        super().close()
        if self.formula is not None:
            self.errors.append("unclosed formula wrapper")


def audit_html(html: str) -> MathAudit:
    audit = MathAudit()
    audit.feed(html)
    audit.close()
    return audit


def audit_site(site: Path) -> tuple[list[str], int, int]:
    errors: list[str] = []
    count = pages = 0
    files = sorted(site.rglob("*.html"))
    if not files:
        return [f"no built HTML found in {site}"], 0, 0
    for path in files:
        relative = path.relative_to(site).as_posix()
        try:
            audit = audit_html(path.read_text(encoding="utf-8"))
        except UnicodeDecodeError:
            errors.append(f"{relative}: invalid UTF-8")
            continue
        errors.extend(f"{relative}: {message}" for message in audit.errors)
        count += audit.count
        pages += int(audit.count > 0)
        if (
            relative in {"learning-lab/index.html", "learning-lab-cn/index.html"}
            and audit.count != 62
        ):
            errors.append(f"{relative}: expected 62 teaching formulas, found {audit.count}")
    if count == 0:
        errors.append("site has no formulas; empty coverage is not a pass")
    for slug in ("learning-lab", "learning-lab-cn"):
        if not (site / slug / "index.html").is_file():
            errors.append(f"missing bilingual laboratory: {slug}")
    return errors, count, pages


def on_post_build(config) -> None:
    """Run the independent audit in existing Python-only MkDocs/Pages CI.

    The first hook validates that collection can only target an isolated
    temporary build; those intentional raw-source pages are never deployed.
    """
    if os.environ.get("EMBODIED_MATH_COLLECT"):
        return
    from mkdocs.exceptions import PluginError

    errors, count, pages = audit_site(Path(config["site_dir"]))
    if errors:
        raise PluginError("Static formula audit failed:\n" + "\n".join(errors))
    logging.getLogger("mkdocs").info(
        "Static formulas verified: %s across %s pages (browser layout checked separately).",
        count,
        pages,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("site", nargs="?", type=Path, default=ROOT / "site")
    args = parser.parse_args()
    errors, count, pages = audit_site(args.site)
    if errors:
        print("Static formula audit FAILED:\n" + "\n".join(errors))
        return 1
    print(
        f"Static formula audit passed: {count} formulas across {pages} pages; SVG + MathML, no raw TeX fallback or runtime math CDN. Browser layout is a separate check."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
