"""Regression tests for scientific-claim and text-quality gates."""

from __future__ import annotations

import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _load_module(name: str, relative: str):
    path = ROOT / relative
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_known_claim_accuracy_gate_passes():
    module = _load_module("check_claims", "scripts/check_claims.py")
    report = module.audit_claims()
    assert report["ok"], "\n".join(report["errors"])


def test_claim_gate_declares_semantic_boundary():
    module = _load_module("check_claims", "scripts/check_claims.py")
    report = module.audit_claims()
    assert "primary-source semantic review" in report["boundary"]


def test_first_party_markdown_format_is_clean():
    module = _load_module("check_markdown_format", "scripts/check_markdown_format.py")
    report = module.format_markdown(write=False)
    details = [*report["changed_files"], *report["errors"]]
    assert report["ok"], "\n".join(details)


def test_markdown_audit_catches_encoding_and_github_math_damage():
    module = _load_module("check_markdown_format", "scripts/check_markdown_format.py")
    clean = "value $x_t$\n$$\nx = 1\n$$\n```python\ntext = '\\\\('\n```\n"
    assert module.audit_text(clean, "clean.md") == []

    broken = (
        "bad \ufffd text\nvalue \\(x_t\\)\nraw \\theta\n$O^+_t$\n"
        "$\\text{bad_name}$\nlabel:\n$$x$$\n$$y$$\n$$\n\nx\n"
    )
    errors = module.audit_text(broken, "broken.md")
    assert any("suspicious encoding" in error for error in errors)
    assert any("GitHub-incompatible math delimiter" in error for error in errors)
    assert any("raw TeX command" in error for error in errors)
    assert any("ambiguous GitHub math script order" in error for error in errors)
    assert any("underscore escapes inside" in error for error in errors)
    assert any("blank line before and after" in error for error in errors)
    assert any("blank line after opening" in error for error in errors)
    assert any("unpaired display math delimiter" in error for error in errors)


def test_markdown_normalizer_spaces_cjk_inline_math_but_preserves_code():
    module = _load_module("check_markdown_format", "scripts/check_markdown_format.py")
    source = "其中 $x$，$y$（$z$）。`中文$u$`\n```text\n中文$w$\n```\n"
    expected = "其中 $x$， $y$（ $z$）。`中文$u$`\n```text\n中文$w$\n```\n"
    assert module.normalize_github_math_spacing(source) == expected


def test_markdown_audit_rejects_currency_ranges_and_prose_dollars():
    module = _load_module("check_markdown_format", "scripts/check_markdown_format.py")
    cases = (
        "价格 $200-$2,000",
        "Price $200 – $300",
        "V1 ~$2,000（BOM）；V2 ~$200-$300（简化版）",
        "~$19,000（V4），V5 ~$20,000+（新增指尖触觉）",
        "Cost $200 and $300",
        "训练成本仅 $30k。",
        "整机 <$2000",
        "Cost ~$100,000+",
    )
    for source in cases:
        errors = module.audit_text("Header\n" + source, "prices.md")
        assert any("prices.md:2: unescaped currency dollar" in error for error in errors), source


def test_markdown_headings_keep_toc_readable():
    module = _load_module("check_markdown_format", "scripts/check_markdown_format.py")
    assert any(
        "table of contents" in error
        for error in module.audit_text("### 阻尼系数 $\\lambda$ 的选择\n", "lesson.md")
    )
    assert module.audit_text("### 阻尼系数 λ 的选择\n正文 $\\lambda$。\n", "lesson.md") == []


def test_markdown_currency_check_preserves_math_escaped_dollars_and_code():
    module = _load_module("check_markdown_format", "scripts/check_markdown_format.py")
    source = (
        "Values $2$, $2k$, $200$, $2.5$, and $1,000$.\n"
        "Subtract $200-300$ or write $200$-$300$.\n"
        "Multiply $2(3)$; note $2\\text{个}$ and $2\\text{and }3$.\n"
        "\n$$200-300$$\n\n$$\n2 + 3 = 5\n$$\n"
        "Prices \\$200-\\$300, ~\\$19,000（V4），V5 ~\\$20,000+.\n"
        "Use USD 200–300 or `${ROS_DISTRO}` or `cost = '$200-$300'`.\n"
        "```sh\necho ${ROS_DISTRO}\necho '$200-$300'\n```\n"
    )
    assert module.audit_text(source, "clean.md") == []
