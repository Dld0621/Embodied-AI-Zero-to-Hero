"""Regression tests for the VLA/WAM specialization contract."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _load_selector():
    path = ROOT / "scripts" / "select_vla_wam_algorithm.py"
    spec = importlib.util.spec_from_file_location("select_vla_wam_algorithm", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_algorithm_catalog_has_primary_sources_and_boundaries():
    data = json.loads(
        (ROOT / "learning_tracks" / "vla_wam_algorithms.json").read_text(encoding="utf-8")
    )
    ids = {item["id"] for item in data["families"]}
    assert len(ids) == len(data["families"]) >= 8
    assert {"vla", "wam", "policy-baseline", "world-model-baseline"}.issubset(
        {item["track"] for item in data["families"]}
    )
    for item in data["families"]:
        assert item["strengths"] and item["risks"]
        assert all(source.startswith("https://") for source in item["primary_sources"])


def test_selector_does_not_jump_to_wam_without_resources():
    module = _load_selector()
    choices, notes = module.recommend(
        goal="future-video-and-action",
        compute="single-gpu",
        data="task-specific",
        latency="soft",
    )
    assert choices[0] in {"direct-chunked-bc", "latent-world-model-mpc"}
    assert any("Do not start" in note for note in notes)


def test_selector_routes_language_and_latency_to_continuous_chunking():
    module = _load_selector()
    choices, notes = module.recommend(
        goal="language-generalization",
        compute="limited",
        data="multi-task",
        latency="hard",
    )
    assert choices[0] == "continuous-chunk-vla"
    assert any("language" in note.lower() for note in notes)


def test_bilingual_specialization_pages_exist_and_cross_link():
    pages = [
        "docs/specializations/README.md",
        "docs/specializations/README_CN.md",
        "docs/specializations/vla-zero-to-one.md",
        "docs/specializations/vla-zero-to-one-cn.md",
        "docs/specializations/wam-zero-to-one.md",
        "docs/specializations/wam-zero-to-one-cn.md",
    ]
    for relative in pages:
        text = (ROOT / relative).read_text(encoding="utf-8")
        assert "algorithm" in text.lower() or "算法" in text
        assert "evidence" in text.lower() or "证据" in text
