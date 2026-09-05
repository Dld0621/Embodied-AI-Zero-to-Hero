"""Static contracts complement numerical and real-browser laboratory tests."""

import os
import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def test_laboratory_assets_are_local_and_loaded_in_dependency_order():
    config = (ROOT / "mkdocs.yml").read_text(encoding="utf-8")
    assert config.index("javascripts/lab-models.js") < config.index(
        "javascripts/learning-lab.js"
    )
    assert "stylesheets/learning-lab.css" in config
    for relative in (
        "docs/javascripts/lab-models.js",
        "docs/javascripts/learning-lab.js",
        "docs/stylesheets/learning-lab.css",
    ):
        assert (ROOT / relative).is_file(), relative


def test_bilingual_laboratory_pages_have_static_lessons():
    for name, language in (("learning-lab-cn.md", "zh"), ("learning-lab.md", "en")):
        lesson = (ROOT / "docs" / name).read_text(encoding="utf-8")
        assert f'data-lab-lang="{language}"' in lesson
        assert 'class="eai-labs"' in lesson
        assert len(lesson) > 5000
        assert "https://" in lesson, "Each lesson needs primary-source references."


def test_proposed_browser_ci_patch_preserves_complete_configuration():
    proposed_patch = (ROOT / "docs/patches/learning-lab-ci.patch").read_text(
        encoding="utf-8"
    )
    for tracked_path in (
        '"docs/javascripts/**"',
        '"tests/interactive/**"',
        '"scripts/test_learning_lab_browser.cjs"',
        '"tests/test_learning_lab.py"',
    ):
        assert proposed_patch.count(tracked_path) == 2, tracked_path
    assert "node --test tests/interactive/lab-models.test.cjs" in proposed_patch
    assert "node scripts/test_learning_lab_browser.cjs" in proposed_patch
    assert "playwright@1.62.1" in proposed_patch
    assert "actions/upload-artifact@v4" in proposed_patch


def test_laboratory_models_with_available_node_runtime():
    node = os.environ.get("EMBODIED_LAB_NODE") or shutil.which("node")
    if not node:
        pytest.skip("Node.js unavailable; set EMBODIED_LAB_NODE to run laboratory model tests.")
    completed = subprocess.run(
        [node, "--test", "tests/interactive/lab-models.test.cjs"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
        timeout=60,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr


def test_laboratory_has_no_telemetry_or_remote_execution():
    for name in ("lab-models.js", "learning-lab.js"):
        source = (ROOT / "docs/javascripts" / name).read_text(encoding="utf-8")
        for forbidden in ("fetch(", "XMLHttpRequest", "sendBeacon", "eval(", "new Function("):
            assert forbidden not in source, f"{name}: unexpected remote execution/telemetry API"
