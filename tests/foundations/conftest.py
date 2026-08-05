"""Shared fixtures and configuration for the Foundations Layer test suite.

These tests verify the documentation under ``docs/foundations/`` of the
Embodied AI project: code snippets run correctly, every doc ships enough
"检查理解" exercises, cross-referenced project files exist, and the roadmap
prerequisite chain is consistent.

Non-fixture constants/helpers live in ``_helpers.py`` so that test modules can
import them with a relative import; this file only holds pytest fixtures.
"""
from __future__ import annotations

import os

# Use a non-interactive Matplotlib backend so that any code snippet calling
# ``plt.show()`` returns immediately instead of blocking the test run. This
# must be set *before* matplotlib.pyplot is imported anywhere.
os.environ.setdefault("MPLBACKEND", "Agg")

from pathlib import Path

import pytest

from ._helpers import DOC_NUMBERS, FOUNDATIONS_DIR, PROJECT_ROOT, read_doc  # noqa: F401,E402


@pytest.fixture(scope="session")
def project_root() -> Path:
    """Absolute path to the repository root."""
    return PROJECT_ROOT


@pytest.fixture(scope="session")
def foundations_dir() -> Path:
    """Absolute path to ``docs/foundations/``."""
    return FOUNDATIONS_DIR


@pytest.fixture(scope="session")
def roadmap_path() -> Path:
    """Absolute path to ``docs/foundations/00-roadmap.md``."""
    return FOUNDATIONS_DIR / "00-roadmap.md"


@pytest.fixture(scope="session", params=DOC_NUMBERS)
def doc_number(request) -> str:
    """Parametrized fixture yielding each doc number '01'..'10'."""
    return request.param


@pytest.fixture(scope="session")
def doc_path_for_number(doc_number: str) -> Path:
    """Resolve the single markdown file matching a doc number."""
    matches = sorted(FOUNDATIONS_DIR.glob(f"{doc_number}-*.md"))
    assert matches, f"No foundations doc found for number {doc_number}"
    assert len(matches) == 1, (
        f"Expected exactly one doc for number {doc_number}, found {[m.name for m in matches]}"
    )
    return matches[0]
