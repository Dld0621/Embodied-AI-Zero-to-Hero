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
