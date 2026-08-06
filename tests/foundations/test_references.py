"""Verify that file paths cross-referenced from the foundations docs actually
exist in the project tree.

Two complementary checks:

1. ``test_key_project_references_exist`` -- a curated, parametrized mapping of
   the most important project files each doc tells the reader to open
   (e.g. ``examples/fk_ik_demo.py``). Failing one of these produces a precise,
   human-readable error.

2. ``test_all_markdown_links_resolve`` -- a general scan that parses every
   ``[text](relative/path)`` link in all 15 foundations docs (including the
   roadmap) and asserts the resolved target exists on disk. External URLs and
   in-page anchors are skipped.
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest

from ._helpers import FOUNDATIONS_DIR, PROJECT_ROOT, read_doc

#: Curated mapping: doc number -> list of project files (relative to repo root)
#: that the doc explicitly tells the reader to open / read.
KEY_REFERENCES: dict[str, list[str]] = {
    "01": [
        "examples/fk_ik_demo.py",
        "examples/unified_pushcube_env.py",
    ],
    "02": [
        "examples/unified_pushcube_env.py",
        "examples/fk_ik_demo.py",
    ],
    "03": [
        "examples/unified_pushcube_vla.py",
        "examples/robot_foundation_models/smolvla/models/lightweight_vla/training_history.json",
        "docs/28-smolvla-gpu-finetuning-runbook.md",
    ],
    "04": [
        "examples/unified_pushcube_vla.py",
        "examples/unified_pushcube_act.py",
        "examples/robot_foundation_models/openvla/inference.py",
        "examples/robot_foundation_models/smolvla/inference.py",
        "docs/01-what-is-vla.md",
        "docs/13-vla-zero-to-one.md",
    ],
    "05": [
        "examples/unified_pushcube_env.py",
        "examples/fk_ik_demo.py",
    ],
    "06": [
        "examples/fk_ik_demo.py",
        "examples/finger_chain_3d.py",
    ],
    "07": [
        "examples/fk_ik_demo.py",
        "examples/finger_chain_3d.py",
        "examples/complete_retargeting_pipeline.py",
    ],
    "08": [
        "examples/robot_foundation_models/common/safety_filter.py",
        "tutorials/05-complete-pipeline/README.md",
        "tutorials/03-vector-optimization/README.md",
        "README.md",
    ],
    "09": [
        "pretrained/urdf/README.md",
        "examples/dexmv_style_retargeting/dexmv_retargeting.py",
        "examples/dexmv_style_retargeting/run_pipeline.py",
        "examples/robot_foundation_models/common/safety_filter.py",
        "docs/20-vla-deployment-guide.md",
        "docs/19-sim-to-real-guide.md",
    ],
    "10": [
        "examples/robot_foundation_models/smolvla/collect_pushcube_dataset.py",
        "examples/robot_foundation_models/common/canonical_dataset.py",
        "examples/robot_foundation_models/smolvla/train_lightweight_vla.py",
        "examples/robot_foundation_models/common/to_lerobot.py",
        "examples/robot_foundation_models/smolvla/closed_loop_eval.py",
        "docs/28-smolvla-gpu-finetuning-runbook.md",
        "README.md",
    ],
    "11": [
        "examples/dreamer_rssm.py",
    ],
    "12": [
        "examples/robot_foundation_models/common/observation_schema.py",
    ],
    "13": [
        "examples/robot_foundation_models/common/model_interface.py",
        "examples/robot_foundation_models/common/embodiment_adapter.py",
        "examples/robot_foundation_models/common/safety_filter.py",
    ],
    "14": [
        "benchmarks/run_benchmark.py",
        "BENCHMARK.md",
        "docs/pipelines/README.md",
    ],
}


def _key_ref_params():
    """Flatten the mapping into (doc_number, relative_path) pairs for parametrize."""
    params = []
    for doc_no, files in KEY_REFERENCES.items():
        for rel in files:
            params.append(pytest.param(doc_no, rel, id=f"{doc_no}:{rel}"))
    return params


@pytest.mark.parametrize("doc_no, rel_path", _key_ref_params())
def test_key_project_references_exist(doc_no, rel_path):
    """Each curated cross-referenced project file must exist on disk."""
    target = PROJECT_ROOT / rel_path
    assert target.exists(), (
        f"doc {doc_no} references '{rel_path}' but it does not exist at {target}"
    )


# Regex for Markdown links:  [label](target)
_LINK_RE = re.compile(r"\[(?P<label>[^\]]*)\]\((?P<target>[^)]*)\)")


def _all_foundations_docs() -> list[Path]:
    return sorted(FOUNDATIONS_DIR.glob("*.md"))


def _collect_local_links() -> list[tuple[Path, str, str]]:
    """Return (doc_path, label, target) for every local link in every doc."""
    out = []
    for doc in _all_foundations_docs():
        text = read_doc(doc)
        for m in _LINK_RE.finditer(text):
            target = m.group("target").strip()
            # skip external URLs, emails, and in-page anchors
            if target.startswith(("http://", "https://", "mailto:", "#")):
                continue
            # strip an optional in-page anchor, e.g. "page.md#section"
            target = target.split("#", 1)[0].strip()
            if not target:
                continue
            out.append((doc, m.group("label"), target))
    return out


def test_all_markdown_links_resolve():
    """Every local Markdown link in the foundations docs must resolve to a real file/dir."""
    links = _collect_local_links()
    # sanity: we must have found a non-trivial number of links
    assert len(links) > 30, f"Expected many doc links, only found {len(links)}"

    missing: list[str] = []
    for doc, label, target in links:
        resolved = (doc.parent / target).resolve()
        if not resolved.exists():
            missing.append(f"{doc.name}: [{label}]({target}) -> {resolved}")
    assert not missing, (
        "Broken cross-references in foundations docs:\n" + "\n".join(missing)
    )


#: Basenames of project source files that the foundations docs reference (some
#: docs cite them by bare filename rather than a full ``examples/`` path).
_PROJECT_BASENAMES = {
    "fk_ik_demo.py",
    "unified_pushcube_env.py",
    "unified_pushcube_vla.py",
    "unified_pushcube_act.py",
    "unified_pushcube_rl.py",
    "unified_pushcube_wm.py",
    "unified_pushcube_diffusion.py",
    "finger_chain_3d.py",
    "complete_retargeting_pipeline.py",
    "minimal_retargeting.py",
    "safety_filter.py",
    "canonical_dataset.py",
    "to_lerobot.py",
    "to_rlds.py",
    "collect_pushcube_dataset.py",
    "train_lightweight_vla.py",
    "closed_loop_eval.py",
    "dexmv_retargeting.py",
    "run_pipeline.py",
}


def test_each_numbered_doc_references_project_code():
    """Each of the 14 docs should reference at least one project source file."""
    for doc_no in [f"{i:02d}" for i in range(1, 15)]:
        matches = sorted(FOUNDATIONS_DIR.glob(f"{doc_no}-*.md"))
        assert len(matches) == 1
        text = read_doc(matches[0])
        referenced = "examples/" in text or any(b in text for b in _PROJECT_BASENAMES)
        assert referenced, (
            f"{matches[0].name} does not reference any project source file"
        )
