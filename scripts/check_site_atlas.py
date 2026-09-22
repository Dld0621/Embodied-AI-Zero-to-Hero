"""Audit built atlas HTML and image paths without a browser or network."""

from __future__ import annotations

import json
import logging
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import unquote, urlsplit

ROOT = Path(__file__).resolve().parents[1]
LOG = logging.getLogger("mkdocs.plugins.knowledge-atlas")


class AtlasHTML(HTMLParser):
    def __init__(self):
        super().__init__()
        self.figures = []
        self.current = None
        self.answers = []
        self.data = []
        self.ids = set()

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        if attrs.get("id"):
            self.ids.add(attrs["id"])
        classes = attrs.get("class", "").split()
        if tag == "figure" and "atlas-visual" in classes:
            self.current = {"attrs": attrs, "images": [], "sources": []}
            self.figures.append(self.current)
        if self.current is not None:
            if tag == "img":
                self.current["images"].append(attrs)
            if tag == "source":
                self.current["sources"].append(attrs)
        if tag == "details" and "atlas-answer" in classes:
            self.answers.append(attrs)
        if tag == "details" and "atlas-data" in classes:
            self.data.append(attrs)

    def handle_endtag(self, tag):
        if tag == "figure":
            self.current = None


def audit_page(html: str, page: Path, site: Path, atoms: list[dict]) -> list[str]:
    parser = AtlasHTML()
    parser.feed(html)
    errors = []
    if len(parser.figures) != len(atoms):
        errors.append("one figure required for every atom")
    if len(parser.answers) != len(atoms) or any("open" in a for a in parser.answers):
        errors.append("self-check answers must exist and start collapsed")
    expected_data = sum(a["visual"]["kind"] not in {"flow", "compare"} for a in atoms)
    if len(parser.data) != expected_data:
        errors.append("numeric diagrams must expose their source data")
    for figure, atom in zip(parser.figures, atoms):
        if atom["id"] not in parser.ids:
            errors.append(f"missing deep-link anchor: {atom['id']}")
        if figure["attrs"].get("tabindex") != "0" or not figure["attrs"].get("aria-label"):
            errors.append("figure needs an accessible keyboard-scroll region")
        if len(figure["images"]) != 1 or len(figure["sources"]) != 1:
            errors.append("figure needs one fallback image and one mobile source")
            continue
        image, mobile = figure["images"][0], figure["sources"][0]
        if not image.get("alt") or image.get("loading") != "lazy":
            errors.append("image needs descriptive alt text and lazy loading")
        for attr, key, suffix in ((image, "src", ".svg"), (mobile, "srcset", "-mobile.svg")):
            url = urlsplit(attr.get(key, ""))
            target = (page.parent / unquote(url.path)).resolve()
            if url.scheme or url.netloc or not target.is_relative_to(site.resolve()):
                errors.append("diagram must stay inside the local site")
            elif not target.is_file() or target.name != atom["id"] + suffix:
                errors.append(f"missing or mismatched diagram: {target.name}")
        if mobile.get("media") != "(max-width: 600px)":
            errors.append("mobile diagram breakpoint missing")
    return errors


def audit_site(site: Path) -> tuple[list[str], int, int]:
    atoms = []
    for path in sorted((ROOT / "knowledge/atlas").glob("*.json")):
        atoms.extend(json.loads(path.read_text(encoding="utf-8"))["atoms"])
    manifest = json.loads((ROOT / "knowledge/manifest.json").read_text(encoding="utf-8"))
    errors, pages = [], 0
    for node in manifest["nodes"]:
        group = [a for a in atoms if a["node_id"] == node["id"]]
        path = site / "knowledge-atlas" / node["id"] / "index.html"
        if not path.is_file() or len(group) < 4:
            errors.append(f"{node['id']}: missing chapter overview")
            continue
        chapter = AtlasHTML()
        chapter.feed(path.read_text(encoding="utf-8"))
        if chapter.figures or not all(a["id"] in chapter.ids for a in group):
            errors.append(f"{node['id']}: overview must retain old anchors, not full figures")
        for atom in group:
            lesson = path.parent / atom["id"] / "index.html"
            if not lesson.is_file():
                errors.append(f"{atom['id']}: missing individual lesson")
                continue
            pages += 1
            errors.extend(
                f"{atom['id']}: {error}"
                for error in audit_page(lesson.read_text(encoding="utf-8"), lesson, site, [atom])
            )
        complete = path.parent / "complete" / "index.html"
        if not complete.is_file():
            errors.append(f"{node['id']}: missing continuous reading page")
        else:
            errors.extend(
                f"{node['id']}/complete: {error}"
                for error in audit_page(complete.read_text(encoding="utf-8"), complete, site, group)
            )
    if not (site / "stylesheets/knowledge-atlas.css").is_file():
        errors.append("missing atlas stylesheet")
    return errors, pages, len(atoms)


def on_post_build(config, **kwargs):
    errors, pages, atoms = audit_site(Path(config["site_dir"]))
    if errors:
        raise ValueError("Built knowledge atlas failed:\n" + "\n".join(errors))
    LOG.info(
        "Built atlas verified: %s individual lessons, %s concepts, chapter and continuous pages; offline HTML checks, not browser acceptance",
        pages,
        atoms,
    )
