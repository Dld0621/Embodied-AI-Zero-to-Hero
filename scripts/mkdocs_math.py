"""Replace Arithmatex source with audited, pre-rendered, self-contained mathematics.

The published site never needs MathJax, Node.js, a CDN, or JavaScript to display
equations. Maintainers explicitly regenerate the committed cache after changing
math; a normal MkDocs build fails closed if that cache is absent or incomplete.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from html.parser import HTMLParser
from pathlib import Path

from mkdocs.exceptions import PluginError

ROOT = Path(__file__).resolve().parents[1]
CACHE_PATH = ROOT / "generated" / "math-cache.json"
COLLECT_ENV = "EMBODIED_MATH_COLLECT"
RENDERER = {
    "name": "mathjax-full",
    "version": "3.2.2",
    "output": "svg",
    "fontCache": "none",
}
SVG_ELEMENTS = {"svg", "g", "path", "rect", "text", "tspan", "title", "desc"}
MATHML_ELEMENTS = {
    "math",
    "mi",
    "mn",
    "mo",
    "mtext",
    "mspace",
    "mrow",
    "mfrac",
    "msqrt",
    "mroot",
    "msub",
    "msup",
    "msubsup",
    "munder",
    "mover",
    "munderover",
    "mtable",
    "mtr",
    "mtd",
    "mstyle",
    "mpadded",
    "mphantom",
    "menclose",
    "mmultiscripts",
    "mprescripts",
    "none",
}
_entries: dict = {}
_collected: dict = {}
_collect_path: Path | None = None


def equation_key(tex: str, display: bool) -> str:
    """Keep text and display mode in the key; never share inline/block metrics."""
    return hashlib.sha256((str(int(display)) + "\0" + tex).encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class Formula:
    start: int
    end: int
    opening: str
    closing: str
    tex: str
    display: bool

    @property
    def key(self) -> str:
        return equation_key(self.tex, self.display)


class FormulaParser(HTMLParser):
    """Find generated math wrappers without reserializing unrelated page HTML."""

    def __init__(self, source: str):
        super().__init__(convert_charrefs=True)
        self.source = source
        self.formulas: list[Formula] = []
        self._line_starts = [0]
        self._line_starts.extend(i + 1 for i, char in enumerate(source) if char == "\n")
        self._active: tuple[int, str, str] | None = None
        self._text: list[str] = []
        self.feed(source)
        self.close()
        if self._active:
            raise PluginError("Unclosed Arithmatex wrapper in generated Markdown HTML.")

    def _offset(self) -> int:
        line, column = self.getpos()
        return self._line_starts[line - 1] + column

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if self._active:
            raise PluginError("Unexpected HTML element inside an Arithmatex formula.")
        if "arithmatex" in (dict(attrs).get("class") or "").split():
            if tag not in ("span", "div"):
                raise PluginError(f"Unsupported Arithmatex wrapper: {tag}.")
            self._active = (self._offset(), tag, self.get_starttag_text())
            self._text = []

    def handle_startendtag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if self._active or "arithmatex" in (dict(attrs).get("class") or "").split():
            raise PluginError("Unexpected self-closing HTML element in an Arithmatex formula.")

    def handle_data(self, data: str) -> None:
        if self._active:
            self._text.append(data)

    def handle_comment(self, data: str) -> None:
        if self._active:
            raise PluginError("Unexpected HTML comment inside an Arithmatex formula.")

    def handle_endtag(self, tag: str) -> None:
        if not self._active:
            return
        start, expected, opening = self._active
        if tag != expected:
            raise PluginError(f"Mismatched Arithmatex wrapper: expected {expected}, got {tag}.")
        display = tag == "div"
        raw = "".join(self._text).strip()
        left, right = ("\\[", "\\]") if display else ("\\(", "\\)")
        if not raw.startswith(left) or not raw.endswith(right):
            raise PluginError("Arithmatex formula is missing its expected generated delimiters.")
        tex = raw[len(left) : -len(right)].strip()
        if not tex:
            raise PluginError("Empty mathematical expression in documentation.")
        closing_start = self._offset()
        closing_end = self.source.find(">", closing_start) + 1
        self.formulas.append(
            Formula(
                start, closing_end, opening, self.source[closing_start:closing_end], tex, display
            )
        )
        self._active = None


def validate_entry(key: str, entry: dict) -> None:
    """Reject stale, malformed, error-bearing, or externally dependent cache data."""
    if not isinstance(entry, dict) or not isinstance(entry.get("tex"), str):
        raise PluginError(f"Invalid static math cache entry: {key}.")
    if not isinstance(entry.get("display"), bool):
        raise PluginError(f"Invalid math display mode: {key}.")
    if equation_key(entry["tex"], entry["display"]) != key:
        raise PluginError(f"Static math cache key does not match its formula: {key}.")
    for field, expected_root in (("svg", "svg"), ("mathml", "math")):
        markup = entry.get(field)
        if not isinstance(markup, str) or not markup:
            raise PluginError(f"Missing {field} in static math cache: {key}.")
        try:
            root = ET.fromstring(markup)
        except ET.ParseError as error:
            raise PluginError(f"Malformed {field} in static math cache: {key}: {error}") from error
        if root.tag.rsplit("}", 1)[-1] != expected_root:
            raise PluginError(f"Incorrect {field} root element in static math cache: {key}.")
        allowed = SVG_ELEMENTS if field == "svg" else MATHML_ELEMENTS
        namespace = (
            "http://www.w3.org/2000/svg" if field == "svg" else "http://www.w3.org/1998/Math/MathML"
        )
        for element in root.iter():
            name = element.tag.rsplit("}", 1)[-1]
            if name not in allowed or element.tag != f"{{{namespace}}}{name}":
                raise PluginError(f"Unsafe, unresolved or error-bearing {field} element: {name}.")
            for attr, value in element.attrib.items():
                local = attr.rsplit("}", 1)[-1].lower()
                if local in {"href", "src"} or local.startswith("on") or "url(" in value.lower():
                    raise PluginError(f"External/active dependency in cached {field}: {key}.")
                if local == "data-mjx-error":
                    raise PluginError(f"MathJax reported a rendering error: {key}.")


def load_cache(path: Path = CACHE_PATH) -> dict:
    try:
        cache = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        raise PluginError(
            f"Static math cache is unavailable: {path}. "
            "Run python scripts/generate_math_cache.py after npm ci."
        ) from error
    if not isinstance(cache, dict) or cache.get("schema") != 1 or cache.get("renderer") != RENDERER:
        raise PluginError("Static math cache schema or pinned renderer does not match this build.")
    entries = cache.get("expressions")
    if not isinstance(entries, dict) or not entries:
        raise PluginError("Static math cache contains no expressions.")
    for key, entry in entries.items():
        validate_entry(key, entry)
    return entries


def on_config(config):
    global _entries, _collected, _collect_path
    _collected = {}
    collect = os.environ.get(COLLECT_ENV)
    _collect_path = Path(collect).resolve() if collect else None
    if _collect_path:
        # Collection intentionally leaves source math intact, only inside an
        # isolated temporary build controlled by the explicit generator script.
        site_dir = Path(config["site_dir"]).resolve()
        if site_dir.parent != _collect_path.parent or site_dir == ROOT / "site":
            raise PluginError("Math collection requires an isolated temporary site directory.")
        _entries = {}
    else:
        _entries = load_cache()
    return config


def render_html(html: str, entries: dict, page_name: str = "page") -> str:
    output: list[str] = []
    previous = 0
    for formula in FormulaParser(html).formulas:
        if formula.key not in entries:
            raise PluginError(
                f"Uncached equation in {page_name}: {formula.tex[:100]!r}. "
                "Run python scripts/generate_math_cache.py after npm ci; "
                "raw TeX will not be published."
            )
        entry = entries[formula.key]
        opening = formula.opening
        width = ET.fromstring(entry["svg"]).get("width", "0ex")
        if not formula.display and width.endswith("ex") and float(width[:-2]) > 28:
            opening = re.sub(
                r"(\bclass\s*=\s*)([\"\'])(.*?)\2",
                lambda match: match[1] + match[2] + match[3] + " math-wide" + match[2],
                opening,
            )
        output.extend((html[previous : formula.start], opening[:-1]))
        output.append(' data-math-rendered="static-svg">')
        output.append('<span class="math-visual" aria-hidden="true">')
        output.extend(
            (entry["svg"], '</span><span class="math-assistive">', entry["mathml"], "</span>")
        )
        output.append(formula.closing)
        previous = formula.end
    output.append(html[previous:])
    return "".join(output)


def on_page_content(html, page, config, files):
    if _collect_path:
        for formula in FormulaParser(html).formulas:
            _collected[formula.key] = {"tex": formula.tex, "display": formula.display}
        return html
    return render_html(html, _entries, page.file.src_uri)


def on_post_build(config):
    if _collect_path:
        _collect_path.write_text(
            json.dumps({"expressions": _collected}, ensure_ascii=False, sort_keys=True),
            encoding="utf-8",
        )
