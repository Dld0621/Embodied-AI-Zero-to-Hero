"""Verify the Foundations Layer roadmap (``docs/foundations/00-roadmap.md``).

Checks:
  * The roadmap table lists all 10 numbered docs.
  * Every doc listed in the roadmap exists on disk.
  * The prerequisite column matches the expected chain (parsed from the table
    itself, then asserted against a hard-coded expected mapping).
  * Prerequisites only reference valid doc numbers and form an acyclic chain.
"""
from __future__ import annotations

import re

import pytest

from ._helpers import DOC_NUMBERS, FOUNDATIONS_DIR, read_doc

#: Expected prerequisite chain, as documented in the roadmap's prose and table.
EXPECTED_PREREQS: dict[str, list[str]] = {
    "01": [],
    "02": ["01"],
    "03": ["02"],
    "04": ["03"],
    "05": ["02"],
    "06": ["05"],
    "07": ["06"],
    "08": ["07"],
    "09": ["08"],
    "10": ["03", "09"],
}

#: Regex matching a roadmap table data row that starts with a 2-digit doc number.
_ROW_RE = re.compile(r"^\|\s*(?P<num>\d{2})\s*\|")


def _parse_prereqs(prereq_cell: str) -> list[str]:
    """Turn a '前置要求' cell into a sorted list of doc-number strings."""
    cell = prereq_cell.strip()
    if cell == "无" or not cell:
        return []
    return [p.strip() for p in cell.split(",") if p.strip()]


def _parse_roadmap_table(text: str) -> dict[str, list[str]]:
    """Parse the roadmap's '课程列表' table into {doc_number: [prereqs]}."""
    rows: dict[str, list[str]] = {}
    for line in text.splitlines():
        m = _ROW_RE.match(line)
        if not m:
            continue
        cells = [c.strip() for c in line.split("|")]
        # cells[0] and cells[-1] are empty (outside the outer pipes)
        doc_no = cells[1]
        prereq_cell = cells[-2]
        # the doc-number cell should be a pure 2-digit number
        if not re.fullmatch(r"\d{2}", doc_no):
            continue
        rows[doc_no] = _parse_prereqs(prereq_cell)
    return rows


def _has_cycle(prereqs: dict[str, list[str]]) -> bool:
    """DFS cycle detection over the prerequisite graph."""
    WHITE, GRAY, BLACK = 0, 1, 2
    color = {n: WHITE for n in prereqs}

    def visit(node: str) -> bool:
        color[node] = GRAY
        for nxt in prereqs.get(node, []):
            if nxt not in color:
                continue
            if color[nxt] == GRAY:
                return True
            if color[nxt] == WHITE and visit(nxt):
                return True
        color[node] = BLACK
        return False

    return any(color[n] == WHITE and visit(n) for n in prereqs)


@pytest.fixture(scope="module")
def roadmap_text(roadmap_path):
    return read_doc(roadmap_path)


@pytest.fixture(scope="module")
def roadmap_table(roadmap_text):
    return _parse_roadmap_table(roadmap_text)


def test_roadmap_file_exists(roadmap_path):
    """The roadmap markdown must exist."""
    assert roadmap_path.exists(), f"Roadmap not found at {roadmap_path}"


def test_roadmap_lists_all_ten_docs(roadmap_table):
    """The roadmap table must contain exactly docs 01..10."""
    assert set(roadmap_table.keys()) == set(DOC_NUMBERS), (
        f"Roadmap docs {sorted(roadmap_table)} != expected {DOC_NUMBERS}"
    )


def test_roadmap_doc_files_exist(roadmap_text):
    """Each doc linked from the roadmap table must exist on disk."""
    for number in DOC_NUMBERS:
        matches = sorted(FOUNDATIONS_DIR.glob(f"{number}-*.md"))
        assert len(matches) == 1, f"Doc {number} file missing under {FOUNDATIONS_DIR}"
        # the filename must also appear (as a link) somewhere in the roadmap
        assert matches[0].name in roadmap_text, (
            f"{matches[0].name} is not linked from the roadmap"
        )


def test_prerequisite_chain_matches_expected(roadmap_table):
    """The parsed prerequisite column must equal the expected chain."""
    for number in DOC_NUMBERS:
        actual = sorted(roadmap_table.get(number, []))
        expected = sorted(EXPECTED_PREREQS[number])
        assert actual == expected, (
            f"doc {number}: parsed prereqs {actual} != expected {expected}"
        )


def test_prerequisites_reference_valid_docs(roadmap_table):
    """Every prerequisite must be a valid doc number 01..10."""
    valid = set(DOC_NUMBERS)
    for number, prereqs in roadmap_table.items():
        for p in prereqs:
            assert p in valid, (
                f"doc {number} lists invalid prerequisite '{p}'"
            )


def test_prerequisite_graph_is_acyclic(roadmap_table):
    """The prerequisite chain must not contain a cycle (it is a DAG)."""
    assert not _has_cycle(roadmap_table), "Prerequisite graph contains a cycle"


def test_learning_order_is_topologically_consistent(roadmap_table):
    """The roadmap's stated Stage order must respect prerequisites.

    The roadmap groups docs into stages 0..4 in increasing order; a doc must
    never list a prerequisite that appears at a later stage. We approximate
    'stage order' with the numeric doc order, which the roadmap follows.
    """
    for number, prereqs in roadmap_table.items():
        for p in prereqs:
            assert p < number, (
                f"doc {number} lists prerequisite {p} that comes after it "
                f"(violates the documented learning order)"
            )


def test_roadmap_describes_foundations_layer(roadmap_text):
    """The roadmap must describe the Foundations Layer and its purpose."""
    assert "Foundations" in roadmap_text
    assert "路线图" in roadmap_text or "课程列表" in roadmap_text
