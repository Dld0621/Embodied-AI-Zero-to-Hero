"""The independent built-HTML audit must reject its historical blind spots."""

from scripts.check_site_math import audit_html, audit_site

GOOD = '<span class="arithmatex" data-math-rendered="static-svg"><span class="math-visual" aria-hidden="true"><svg><path d="M0 0"/></svg></span><span class="math-assistive"><math><mi>x</mi></math></span></span>'


def test_static_formula_passes_without_runtime():
    audit = audit_html(GOOD)
    assert audit.count == 1
    assert audit.errors == []


def test_missing_renderer_is_not_a_valid_formula():
    audit = audit_html(r'<span class="arithmatex">\(x\)</span>')
    assert any("static render marker" in error for error in audit.errors)
    assert any("raw content" in error for error in audit.errors)


def test_visible_source_cannot_hide_beside_good_svg():
    audit = audit_html(
        GOOD.replace('<span class="math-visual"', r'\frac{1}{2}<span class="math-visual"')
    )
    assert any("raw content" in error for error in audit.errors)


def test_accessible_math_and_actual_glyphs_are_required():
    assert any(
        "mathml" in error
        for error in audit_html(GOOD.replace("<math>", "<div>").replace("</math>", "</div>")).errors
    )
    assert any("ink" in error for error in audit_html(GOOD.replace('<path d="M0 0"/>', "")).errors)


def test_renderer_errors_and_remote_script_fail():
    audit = audit_html(
        GOOD.replace("<mi>x</mi>", '<merror data-mjx-error="bad TeX">error</merror>')
        + '<script src="https://unpkg.com/mathjax@3/es5/tex-mml-chtml.js"></script>'
    )
    assert any("renderer error" in error for error in audit.errors)
    assert any("runtime math" in error for error in audit.errors)


def test_empty_site_is_not_success(tmp_path):
    errors, count, pages = audit_site(tmp_path)
    assert errors and count == pages == 0


def test_toc_math_stripped_of_its_wrapper_is_not_a_pass():
    audit = audit_html(GOOD + r"<nav><a>阻尼系数 \(\lambda\) 的选择</a></nav>")
    assert any("headings/TOC" in error for error in audit.errors)


def test_intentional_code_examples_are_not_raw_prose_math():
    audit = audit_html(GOOD + r"<pre><code>\frac{1}{2}</code></pre><code>\(x\)</code>")
    assert not audit.errors
