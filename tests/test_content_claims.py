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
    assert report["ok"], "\n".join(report["changed_files"])
