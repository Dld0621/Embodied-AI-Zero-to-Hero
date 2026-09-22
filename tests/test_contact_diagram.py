"""Geometric contracts for the documented force-on-object diagram, not grasp validation."""

import re
import xml.etree.ElementTree as ET
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def diagram():
    text = (ROOT / "docs/pipelines/11-dexterous-manipulation.md").read_text(encoding="utf-8")
    svg = ET.fromstring(re.search(r"<svg\b.*?</svg>", text, flags=re.S).group())
    return {element.get("id"): element for element in svg.iter() if element.get("id")}


def test_force_components_are_orthogonal_and_normals_point_into_object():
    elements = diagram()
    for side, sign in (("left", 1), ("right", -1)):
        normal = elements[f"grasp-normal-{side}"].get("d")
        tangent = elements[f"grasp-tangent-{side}"].get("d")
        nx, ny, end_x = map(float, re.fullmatch(r"M(\d+) (\d+) H(\d+)", normal).groups())
        tx, ty, end_y = map(float, re.fullmatch(r"M(\d+) (\d+) V(\d+)", tangent).groups())
        assert (nx, ny) == (tx, ty)
        assert (end_x - nx) * sign > 0
        assert end_y != ty
        # The two cone-edge endpoints must be on the normal's inward side.
        coordinates = list(map(float, re.findall(r"\d+", elements[f"grasp-cone-{side}"].get("d"))))
        x1, y1, apex_x, apex_y, x2, y2 = coordinates
        assert (apex_x, apex_y) == (nx, ny)
        assert (x1 - nx) * sign > 0 and (x2 - nx) * sign > 0
        assert y1 < ny < y2
        # Drawn force resultant has tangent/normal ratio below the cone-edge slope.
        assert abs((end_y - ty) / (end_x - nx)) <= abs((y1 - ny) / (x1 - nx))


def test_every_force_arrow_references_an_existing_marker():
    elements = diagram()
    for element in elements.values():
        reference = element.get("marker-end")
        if reference:
            assert reference.startswith("url(#") and reference.endswith(")")
            assert elements[reference[5:-1]].tag == "marker"
