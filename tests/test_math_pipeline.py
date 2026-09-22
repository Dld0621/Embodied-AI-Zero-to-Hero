"""Fail-closed equation cache and HTML-hook contracts (no browser or network)."""

import importlib.util
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

pytest.importorskip("mkdocs", reason="Install requirements-docs.txt for MkDocs hook tests.")

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("static_math_hook", ROOT / "scripts/mkdocs_math.py")
math_hook = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = math_hook
SPEC.loader.exec_module(math_hook)


def sample_entry(tex="x", display=False):
    return {
        "tex": tex,
        "display": display,
        "svg": '<svg xmlns="http://www.w3.org/2000/svg" width="1ex" height="1ex"><path d="M0 0L1 1"/></svg>',
        "mathml": '<math xmlns="http://www.w3.org/1998/Math/MathML"><mi>x</mi></math>',
    }


def test_math_parser_decodes_entities_and_preserves_surrounding_html():
    source = '<p id="原文">before <span class="arithmatex">\\(x &lt; y\\)</span> &amp; after</p>'
    formulas = math_hook.FormulaParser(source).formulas
    assert len(formulas) == 1
    assert formulas[0].tex == "x < y"
    rendered = math_hook.render_html(source, {formulas[0].key: sample_entry("x < y")})
    assert rendered.startswith('<p id="原文">before <span class="arithmatex"')
    assert rendered.endswith("</span> &amp; after</p>")
    assert 'data-math-rendered="static-svg"' in rendered
    assert '<span class="math-visual" aria-hidden="true"><svg' in rendered
    assert '<span class="math-assistive"><math' in rendered
    assert "\\(" not in rendered


def test_inline_and_display_math_have_distinct_keys_and_metrics():
    source = '<span class="arithmatex">\\(x\\)</span><div class="arithmatex">\\[\nx\n\\]</div>'
    inline, block = math_hook.FormulaParser(source).formulas
    assert inline.tex == block.tex == "x"
    assert inline.display is False and block.display is True
    assert inline.key != block.key
    result = math_hook.render_html(
        source, {inline.key: sample_entry(), block.key: sample_entry(display=True)}
    )
    assert result.count('data-math-rendered="static-svg"') == 2


def test_only_wide_inline_equations_get_a_scrollable_class():
    for display, tag, delimiters in (
        (False, "span", ("\\(", "\\)")),
        (True, "div", ("\\[", "\\]")),
    ):
        source = f'<{tag} class="arithmatex">{delimiters[0]}x{delimiters[1]}</{tag}>'
        for width in ("28ex", "28.001ex", "38.4ex"):
            entry = sample_entry(display=display)
            entry["svg"] = entry["svg"].replace('width="1ex"', f'width="{width}"')
            rendered = math_hook.render_html(source, {math_hook.equation_key("x", display): entry})
            assert ('class="arithmatex math-wide"' in rendered) == (
                not display and float(width[:-2]) > 28
            )


@pytest.mark.parametrize(
    "source",
    [
        '<span class="arithmatex">\\(x\\)',
        '<div class="arithmatex">\\(x\\)</div>',
        '<span class="arithmatex">\\(\\)</span>',
        '<span class="arithmatex"><b>x</b></span>',
        '<span class="arithmatex">\\(x<br>y\\)</span>',
        '<span class="arithmatex">\\(x<!--hidden-->\\)</span>',
        '<span class="arithmatex">\\(x\\)</div>',
    ],
)
def test_malformed_generated_wrappers_fail_closed(source):
    with pytest.raises(math_hook.PluginError):
        math_hook.FormulaParser(source)


def test_uncached_formula_fails_with_page_and_regeneration_instruction():
    with pytest.raises(
        math_hook.PluginError, match="lesson.md.*Uncached|Uncached.*lesson.md"
    ) as error:
        math_hook.render_html('<span class="arithmatex">\\(x\\)</span>', {}, "lesson.md")
    assert "generate_math_cache.py" in str(error.value)
    assert "raw TeX will not be published" in str(error.value)


@pytest.mark.parametrize(
    "markup",
    [
        "<svg><merror>invalid</merror></svg>",
        '<svg><g data-mjx-error="oops"/></svg>',
        '<svg><use href="#global-font"/></svg>',
        '<svg><image href="https://example.invalid/a.png"/></svg>',
        "<svg><script>alert(1)</script></svg>",
        '<svg onload="alert(1)"/>',
        '<svg style="fill:url(https://example.invalid/a)"/>',
        "<svg><foreignObject/></svg>",
        '<svg xmlns="http://www.w3.org/2000/svg"><style>@import "https://example.invalid/style.css";</style></svg>',
        '<svg xmlns="http://www.w3.org/2000/svg"><animate attributeName="href" values="https://example.invalid/"/></svg>',
        '<svg xmlns="http://www.w3.org/2000/svg"><set attributeName="href" to="https://example.invalid/"/></svg>',
        '<svg xmlns="http://www.w3.org/2000/svg"><link href="https://example.invalid/style.css"/></svg>',
        '<svg xmlns="http://wrong.example.invalid/namespace"><path d="M0 0L1 1"/></svg>',
        "<svg>",
    ],
)
def test_cache_rejects_errors_global_fonts_and_external_or_active_content(markup):
    entry = sample_entry()
    entry["svg"] = markup
    with pytest.raises(math_hook.PluginError):
        math_hook.validate_entry(math_hook.equation_key("x", False), entry)


def test_cache_key_must_match_source_and_display_mode():
    with pytest.raises(math_hook.PluginError, match="does not match"):
        math_hook.validate_entry(math_hook.equation_key("y", False), sample_entry())


def test_committed_cache_passes_strict_validation_and_pinned_version():
    entries = math_hook.load_cache()
    assert entries, "The teaching corpus must not silently lose all rendered formulas."
    config = (ROOT / "mkdocs.yml").read_text(encoding="utf-8")
    assert "scripts/mkdocs_math.py" in config
    assert "unpkg.com" not in config
    assert "javascripts/mathjax.js" not in config
    package = json.loads((ROOT / "package.json").read_text())
    lock = json.loads((ROOT / "package-lock.json").read_text())
    assert package["devDependencies"]["mathjax-full"] == math_hook.RENDERER["version"]
    assert lock["packages"]["node_modules/mathjax-full"]["version"] == math_hook.RENDERER["version"]
    assert package["overrides"]["@xmldom/xmldom"] == "0.9.12"
    assert lock["packages"]["node_modules/@xmldom/xmldom"]["version"] == "0.9.12"


def test_committed_cache_exactly_covers_current_document_formulas():
    # Content corrections may legitimately remove formulas. Validate exact source
    # coverage in isolated builds instead of requiring an arbitrary minimum count.
    completed = subprocess.run(
        [sys.executable, "scripts/generate_math_cache.py", "--check"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=180,
        check=False,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr


def test_collector_cannot_replace_the_current_preview(monkeypatch, tmp_path):
    monkeypatch.setenv(math_hook.COLLECT_ENV, str(tmp_path / "formulas.json"))
    with pytest.raises(math_hook.PluginError, match="isolated temporary"):
        math_hook.on_config({"site_dir": str(ROOT / "site")})
    monkeypatch.delenv(math_hook.COLLECT_ENV)
    math_hook.on_config({"site_dir": str(ROOT / "site")})


def test_missing_or_wrong_version_cache_fails_closed(tmp_path):
    path = tmp_path / "missing.json"
    with pytest.raises(math_hook.PluginError, match="unavailable"):
        math_hook.load_cache(path)
    path.write_text(json.dumps({"schema": 1, "renderer": {"version": "4"}}))
    with pytest.raises(math_hook.PluginError, match="pinned renderer"):
        math_hook.load_cache(path)


def test_maintainer_renderer_with_available_node_and_pinned_dependency():
    node = (
        os.environ.get("EMBODIED_MATH_NODE")
        or os.environ.get("EMBODIED_LAB_NODE")
        or shutil.which("node")
    )
    if not node or not (ROOT / "node_modules/mathjax-full/package.json").is_file():
        pytest.skip(
            "Maintainer renderer requires Node and npm ci; normal Python site builds do not."
        )
    completed = subprocess.run(
        [node, "--test", "tests/math/math-render.test.cjs"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
