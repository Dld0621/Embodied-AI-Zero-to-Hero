"""Verify that every foundations document ships a "检查理解" (Check
Understanding) section with a healthy number of exercises.

The docs use two exercise styles:
  * docs 01-02  ->  ``**练习 N（…）**``  (bold "exercise N")
  * docs 03-10  ->  ``N. **…题**``       (numbered bold list items)

Both are recognised by the counting helper below.
"""
from __future__ import annotations

import re

from ._helpers import DOC_NUMBERS, FOUNDATIONS_DIR, read_doc


def _check_understanding_section(text: str) -> str:
    """Return the slice of ``text`` starting at the '检查理解' heading.

    Only a real Markdown heading (a line starting with ``#``) is accepted, so
    that a table-of-contents link such as ``7. [检查理解](#7-检查理解)`` is not
    mistaken for the section itself.
    """
    m = re.search(r"(?m)^#+\s.*检查理解", text)
    if m is None:
        return ""
    return text[m.start():]


def _count_exercises(text: str) -> int:
    """Count exercises inside the '检查理解' section using both styles."""
    section = _check_understanding_section(text)
    # Style A: **练习 1（…）**
    style_a = len(re.findall(r"练习\s*\d+", section))
    # Style B: numbered bold items like "1. **概念题**"
    style_b = len(re.findall(r"(?m)^\s*\d+\.\s+\*\*", section))
    return max(style_a, style_b)


def test_all_fourteen_docs_exist():
    """The foundations layer must contain exactly the 14 numbered docs."""
    for number in DOC_NUMBERS:
        matches = sorted(FOUNDATIONS_DIR.glob(f"{number}-*.md"))
        assert len(matches) == 1, f"Expected one doc for {number}, got {matches}"


def test_each_doc_has_check_understanding_section(doc_path_for_number):
    """Every foundations doc must contain a '检查理解' heading."""
    text = read_doc(doc_path_for_number)
    assert "检查理解" in text, f"{doc_path_for_number.name} is missing '检查理解'"
    section = _check_understanding_section(text)
    assert section, (
        f"{doc_path_for_number.name} has '检查理解' text but no dedicated heading"
    )


def test_each_doc_has_at_least_three_exercises(doc_path_for_number):
    """Each '检查理解' section must contain at least 3 exercises."""
    text = read_doc(doc_path_for_number)
    count = _count_exercises(text)
    assert count >= 3, (
        f"{doc_path_for_number.name}: expected >=3 exercises, found {count}"
    )


def test_exercise_counts_are_reasonable():
    """Sanity check: every doc has between 3 and 12 exercises (catches parser drift)."""
    for number in DOC_NUMBERS:
        matches = sorted(FOUNDATIONS_DIR.glob(f"{number}-*.md"))
        assert len(matches) == 1
        count = _count_exercises(read_doc(matches[0]))
        assert 3 <= count <= 12, f"{matches[0].name}: exercise count {count} out of range"
