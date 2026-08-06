"""Shared constants and tiny helpers for the Foundations Layer test suite.

Kept separate from ``conftest.py`` so that the test modules can import them
with a plain relative import (``from ._helpers import ...``). ``conftest.py``
is loaded by pytest as a plugin and is not meant to be imported by name.
"""
from __future__ import annotations

from pathlib import Path

#: Project root = <repo>/tests/foundations/_helpers.py -> .parent.parent.parent
PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent.parent

#: Directory holding the foundations markdown docs.
FOUNDATIONS_DIR: Path = PROJECT_ROOT / "docs" / "foundations"

#: The 14 numbered foundations documents (01 .. 14), in order.
DOC_NUMBERS = [f"{i:02d}" for i in range(1, 15)]


def read_doc(path: Path) -> str:
    """Read a markdown doc as UTF-8 text."""
    return path.read_text(encoding="utf-8")
