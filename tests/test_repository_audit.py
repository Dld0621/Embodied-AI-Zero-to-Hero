"""Regression tests for repository-level evidence and governance contracts."""

from __future__ import annotations

import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
AUDIT_SCRIPT = ROOT / "scripts" / "audit_repository.py"


def _load_audit_module():
    spec = importlib.util.spec_from_file_location("audit_repository", AUDIT_SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_repository_contract_audit_passes():
    module = _load_audit_module()
    report = module.audit_repository()
    assert report["ok"], "\n".join(report["errors"])


def test_audit_declares_its_hardware_boundary():
    module = _load_audit_module()
    report = module.audit_repository()
    assert "no hardware certification" in report["boundary"]
