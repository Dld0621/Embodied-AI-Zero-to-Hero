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


def test_audit_reports_the_knowledge_system_contract():
    module = _load_audit_module()
    stats = module.audit_repository()["stats"]
    assert stats["knowledge_nodes"] == 45
    assert stats["knowledge_domains"] == 9
    assert stats["knowledge_stages"] == 6
    assert stats["knowledge_pipeline_coverage"] == 11


def test_content_first_landing_contract():
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    readme_cn = (ROOT / "README_CN.md").read_text(encoding="utf-8")
    index = (ROOT / "docs" / "index.md").read_text(encoding="utf-8")
    index_cn = (ROOT / "docs" / "index_cn.md").read_text(encoding="utf-8")

    assert "dof-hero" not in readme
    assert "dof-hero" not in readme_cn
    assert "docs/curriculum.md" in readme
    assert "docs/curriculum_cn.md" in readme_cn
    assert "dof-intro" in index and "dof-signal" not in index
    assert "dof-intro" in index_cn and "dof-signal" not in index_cn
