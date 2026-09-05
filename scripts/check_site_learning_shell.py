"""Check generated HTML structure and local navigation; no browser acceptance claim."""

from __future__ import annotations

import json
import logging
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import unquote, urlsplit

ROOT = Path(__file__).resolve().parents[1]
LOG = logging.getLogger("mkdocs.plugins.learning-shell")


class ReadingHTML(HTMLParser):
    def __init__(self):
        super().__init__()
        self.in_article = False
        self.classes = {}
        self.links = []
        self.scripts = []
        self.ids = set()

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        if tag == "article":
            self.in_article = True
        if attrs.get("id"):
            self.ids.add(attrs["id"])
        if tag == "script" and attrs.get("src"):
            self.scripts.append(attrs["src"])
        if self.in_article:
            for name in attrs.get("class", "").split():
                self.classes.setdefault(name, []).append(attrs)
            if tag == "a" and attrs.get("href"):
                self.links.append(attrs["href"])

    def handle_endtag(self, tag):
        if tag == "article":
            self.in_article = False


def read_html(path: Path) -> ReadingHTML:
    parser = ReadingHTML()
    parser.feed(path.read_text(encoding="utf-8"))
    return parser


def audit_site(site: Path) -> tuple[list[str], int]:
    site = site.resolve()
    pages = [site / "index.html", site / "index_cn/index.html"]
    pages.extend(sorted((site / "knowledge-atlas").rglob("index.html")))
    errors, seen = [], {}
    for page in pages:
        if not page.is_file():
            errors.append(f"missing learning page: {page}")
            continue
        parsed = read_html(page)
        seen[page] = parsed
        controls = parsed.classes.get("study-toolbar", [])
        if len(controls) != 1 or "hidden" not in controls[0]:
            errors.append(f"{page}: one progressive reading toolbar required")
        if (
            sum(urlsplit(p).path.endswith("javascripts/learning-shell.js") for p in parsed.scripts)
            != 1
        ):
            errors.append(f"{page}: reading behavior must load exactly once")
        lessons = parsed.classes.get("study-lesson", [])
        if lessons:
            if len(lessons) != 1 or not lessons[0].get("data-study-title"):
                errors.append(f"{page}: lesson bookmark metadata missing")
            if len(parsed.classes.get("study-section", [])) != 5:
                errors.append(f"{page}: five explanation sections required")
            if len(parsed.classes.get("study-pagination", [])) != 1:
                errors.append(f"{page}: adjacent-lesson navigation missing")
    for page, parsed in list(seen.items()):
        for href in parsed.links:
            url = urlsplit(href)
            if url.scheme or url.netloc:
                continue
            target = (page.parent / unquote(url.path)).resolve() if url.path else page
            if target.is_dir():
                target /= "index.html"
            if not target.is_relative_to(site) or not target.is_file():
                errors.append(f"{page.relative_to(site)}: broken learning link {href}")
            elif url.fragment:
                if target not in seen:
                    seen[target] = read_html(target)
                if unquote(url.fragment) not in seen[target].ids:
                    errors.append(f"{page.relative_to(site)}: missing anchor {href}")
    search = site / "search/search_index.json"
    if search.is_file():
        entries = json.loads(search.read_text(encoding="utf-8"))["docs"]
        locations = {entry["location"].split("#")[0] for entry in entries}
        if any(p.startswith("knowledge-atlas/") and "/complete/" in p for p in locations):
            errors.append("continuous-reading copies must not duplicate search results")
        for page, parsed in seen.items():
            if parsed.classes.get("study-lesson"):
                location = page.parent.relative_to(site).as_posix() + "/"
                if location not in locations:
                    errors.append(f"individual lesson missing from search: {location}")
    else:
        errors.append("local search index missing")
    return errors, len(pages)


def on_post_build(config, **kwargs):
    errors, pages = audit_site(Path(config["site_dir"]))
    if errors:
        raise ValueError("Learning layout failed:\n" + "\n".join(errors))
    LOG.info("Learning layout: %s built pages, internal links and search checked offline", pages)
