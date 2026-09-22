"""Offline content, diagram and generated-output contracts; not browser acceptance."""

from __future__ import annotations

import copy
import json
import math
import re
import xml.etree.ElementTree as ET
from pathlib import Path

import pytest

from scripts import build_knowledge_atlas as atlas

ROOT = Path(__file__).resolve().parents[1]
NS = {"s": "http://www.w3.org/2000/svg"}


@pytest.fixture(scope="module")
def curriculum():
    return atlas.load_atoms()


def test_every_declared_node_has_distinct_teachable_atoms(curriculum):
    graph, atoms, counts = curriculum
    assert len(graph["nodes"]) == 45
    assert len(graph["domains"]) == 9
    assert set(counts) == {n["id"] for n in graph["nodes"]}
    assert all(4 <= count <= 10 for count in counts.values())
    assert len({a["id"] for a in atoms}) == len(atoms)
    assert len({tuple(a["mechanism"]) for a in atoms}) == len(atoms)
    assert {a["visual"]["kind"] for a in atoms} == atlas.KINDS
    # Sources, examples, checks, misconceptions and reading guidance are mandatory.
    nodes = {n["id"]: n for n in graph["nodes"]}
    assert all(not atlas.atom_errors(a, nodes) for a in atoms)


def test_committed_outputs_are_current_and_deterministic(curriculum):
    graph, atoms, counts = curriculum
    expected = atlas.outputs(graph, atoms, counts)
    assert expected == atlas.outputs(graph, atoms, counts)
    for path, content in expected.items():
        assert path.read_text(encoding="utf-8") == content, path
    assert set(atlas.ASSETS.glob("*.svg")) == {p for p in expected if p.suffix == ".svg"}
    assert set(atlas.LESSONS.rglob("*.md")) == {p for p in expected if p.suffix == ".md"}
    coverage = json.loads((ROOT / "knowledge/atlas-coverage.json").read_text(encoding="utf-8"))
    assert coverage["complete"] is True
    assert coverage["atoms"] == len(atoms)
    assert coverage["responsive_svg_files"] == 2 * len(atoms)
    assert "not exhaustive" in coverage["boundary"]


def test_each_original_lesson_and_navigation_link_to_its_close_up(curriculum):
    graph, _, _ = curriculum
    nav = (ROOT / "mkdocs.yml").read_text(encoding="utf-8")
    for node in graph["nodes"]:
        source = (ROOT / node["document"]).read_text(encoding="utf-8")
        assert f"knowledge-atlas/{node['id']}/index.md" in source
        assert f"knowledge-atlas/{node['id']}/index.md" in nav


def test_figures_are_responsive_accessible_and_source_paths_resolve(curriculum):
    graph, atoms, _ = curriculum
    nodes = {n["id"]: n for n in graph["nodes"]}
    for node_id in nodes:
        group = [a for a in atoms if a["node_id"] == node_id]
        chapter = (atlas.LESSONS / node_id / "index.md").read_text(encoding="utf-8")
        assert '<figure class="atlas-visual"' not in chapter
        assert all(f'<a id="{a["id"]}"></a>' in chapter for a in group)
        assert all(f"{a['id']}/index.md" in chapter for a in group)
        page = atlas.LESSONS / node_id / "complete" / "index.md"
        content = page.read_text(encoding="utf-8")
        assert content.count('<figure class="atlas-visual"') == len(group)
        assert content.count('<details class="atlas-answer"') == len(group)
        assert "open=" not in content
        assert "本节点原有验收要求" in content
        assert all(f'<a id="{a["id"]}"></a>' in content for a in group)
        for path in re.findall(r'(?:src|srcset)="([^"]+)"', content):
            assert not path.startswith(("http:", "https:", "//"))
            assert (page.parent / path).resolve().is_file()
            # A nested index.md deploys at the same directory depth. Raw HTML
            # picture URLs therefore resolve in both GitHub and MkDocs output.
            deployed = ROOT / "site" / page.parent.relative_to(ROOT / "docs") / path
            assert (
                deployed.resolve()
                .relative_to(ROOT / "site")
                .as_posix()
                .startswith("assets/knowledge-atlas/")
            )


def test_individual_lessons_preserve_every_authored_explanation(curriculum):
    _, atoms, _ = curriculum
    for atom in atoms:
        page = atlas.atom_path(atom)
        content = page.read_text(encoding="utf-8")
        assert content.count('<figure class="atlas-visual"') == 1
        assert content.count('<details class="atlas-answer"') == 1
        assert "../index.md" in content and "../complete/index.md" in content
        assert 'class="study-pagination"' in content
        for text in [
            atom["intuition"],
            *atom["mechanism"],
            *atom["worked_example"],
            *atom["reading"],
            atom["check"]["answer"],
            *atom["misconception"].values(),
        ]:
            assert text in content, atom["id"]
        for target in re.findall(r'(?:src|srcset)="([^"]+)"', content):
            assert (page.parent / target).resolve().is_file()


def test_all_svg_variants_are_local_finite_and_have_text_descriptions(curriculum):
    _, atoms, _ = curriculum
    for atom in atoms:
        for width in (360, 720):
            source = atlas.diagram(atom["visual"], width)
            svg = ET.fromstring(source)
            assert svg.find("s:title", NS).text == atom["visual"]["title"]
            assert svg.find("s:desc", NS).text == atom["visual"]["caption"]
            assert int(svg.attrib["width"]) == width
            assert svg.attrib["role"] == "img"
            assert svg.findall(".//s:text", NS), atom["id"]
            assert "<script" not in source and "<foreignObject" not in source
            assert "href=" not in source and "url(" not in source
            assert not re.search(r"(?i)(?:=|,|\s)(?:nan|inf)(?:\W|$)", source)
            assert "\ufffd" not in source
            height = int(svg.attrib["height"])
            # Bounds check catches clipped labels even when the SVG is valid XML.
            for text in svg.findall(".//s:text", NS):
                y = float(text.attrib["y"])
                for span in text.findall("s:tspan", NS):
                    y += float(span.attrib["dy"])
                    assert 0 <= y <= height - 4, (atom["id"], width, span.text, y, height)


def test_chart_tables_preserve_authored_values(curriculum):
    _, atoms, _ = curriculum
    for atom in atoms:
        v = atom["visual"]
        table = atlas.data_table(v)
        if v["kind"] in {"flow", "compare"}:
            assert table == ""
        else:
            assert "查看作图数据" in table
        if v["kind"] == "plot":
            for series in v["series"]:
                for x, y in series["points"]:
                    assert f"| {x} | {y} |" in table
        if v["kind"] == "matrix":
            for row in v["values"]:
                assert " | ".join(str(x) for x in row) in table


def test_interactive_connections_target_existing_experiments():
    javascript = (ROOT / "docs/javascripts/learning-lab.js").read_text(encoding="utf-8")
    for lab, prompt in atlas.LAB_CONNECTIONS.values():
        assert f'"{lab}"' in javascript
        assert len(prompt) >= 15


@pytest.mark.parametrize("bad", [math.nan, math.inf, -math.inf, True, "1"])
def test_invalid_chart_coordinates_are_rejected(bad):
    v = {
        "kind": "plot",
        "title": "测试",
        "caption": "教学测试",
        "x_label": "时间 (s)",
        "y_label": "位置 (m)",
        "series": [{"label": "轨迹", "points": [[0, 0], [1, bad]]}],
    }
    assert atlas.visual_errors(v)


def test_invalid_shapes_and_edges_are_rejected():
    assert atlas.visual_errors(None)
    assert atlas.visual_errors({"kind": "imaginary"})
    assert atlas.visual_errors(
        {"kind": "matrix", "title": "x", "caption": "y", "values": [[1, 2], [3]]}
    )
    assert atlas.visual_errors(
        {
            "kind": "flow",
            "title": "x",
            "caption": "y",
            "nodes": [{"label": "a", "detail": "b"}] * 3,
            "edges": [{"from": 0, "to": 3}],
        }
    )


def test_missing_instruction_parts_and_encoding_damage_fail(curriculum):
    graph, atoms, _ = curriculum
    nodes = {n["id"]: n for n in graph["nodes"]}
    for key in atlas.FIELDS:
        modified = copy.deepcopy(atoms[0])
        del modified[key]
        assert atlas.atom_errors(modified, nodes)
    modified = copy.deepcopy(atoms[0])
    modified["intuition"] += "\ufffd"
    assert "encoding damage" in atlas.atom_errors(modified, nodes)


def test_partial_authoring_cannot_pass_release_check(tmp_path, monkeypatch):
    source = tmp_path / "atlas"
    source.mkdir()
    (source / "empty.json").write_text('{"schema_version": 1, "atoms": []}', encoding="utf-8")
    monkeypatch.setattr(atlas, "AUTHOR_DIR", source)
    with pytest.raises(ValueError, match="at least 4 distinct atoms"):
        atlas.load_atoms()
    _, atoms, counts = atlas.load_atoms(allow_partial=True)
    assert not atoms and not any(counts.values())


def test_cjk_label_wrapping_preserves_text():
    label = "旋转矩阵 R 的第 1 列表示哪个坐标轴？"
    wrapped = atlas.text_lines(label, 12)
    assert "".join(wrapped) == label
    assert all(sum(2 if ord(c) > 255 else 1 for c in line) <= 12 for line in wrapped)


def test_built_html_audit_rejects_missing_images_answers_and_anchors(tmp_path):
    from scripts.check_site_atlas import audit_page, audit_site

    atom = {"id": "sample", "visual": {"kind": "plot"}}
    errors = audit_page("<h1>Looks finished</h1>", tmp_path / "index.html", tmp_path, [atom])
    assert any("figure" in e for e in errors)
    assert any("answers" in e for e in errors)
    assert any("source data" in e for e in errors)
    errors, pages, _ = audit_site(tmp_path)
    assert errors and pages == 0


def test_built_html_audit_detects_remote_or_stale_diagrams(tmp_path):
    from scripts.check_site_atlas import audit_page

    html = """<a id="sample"></a><figure class="atlas-visual" tabindex="0" aria-label="Test">
    <picture><source media="(max-width: 600px)" srcset="sample-mobile.svg">
    <img src="https://example.com/sample.svg" alt="Example" loading="lazy"></picture>
    </figure><details class="atlas-answer" open></details>"""
    errors = audit_page(
        html, tmp_path / "index.html", tmp_path, [{"id": "sample", "visual": {"kind": "flow"}}]
    )
    assert any("local site" in e for e in errors)
    assert any("mismatched diagram" in e for e in errors)
    assert any("collapsed" in e for e in errors)


def test_selected_worked_numbers_match_independent_calculations(curriculum):
    _, atoms, _ = curriculum
    visuals = {a["id"]: a["visual"] for a in atoms}
    softmax = math.exp(1) / (math.exp(1) + 1)
    assert visuals["transformer-scaled-softmax"]["values"] == pytest.approx([softmax, 1 - softmax])
    assert visuals["transformer-position-information"]["values"][1] == pytest.approx(
        [1 + math.sin(1), math.cos(1)]
    )
    assert visuals["rl-bellman-bootstrap"]["values"] == pytest.approx(
        [3, 3 + 0.2 * (1 + 0.9 * 5 - 3), 1 + 0.9 * 5]
    )
    for ratio, value in visuals["rl-ppo-clipping"]["series"][0]["points"]:
        assert value == pytest.approx(min(ratio * 2, min(1.2, max(0.8, ratio)) * 2))
    assert visuals["navigation-stopping-distance"]["values"] == pytest.approx(
        [0.5 * 0.2, 0.5**2 / 2, 0.5 * 0.2 + 0.5**2 / 2]
    )
    assert visuals["manipulation-friction-grasp"]["values"] == pytest.approx(
        [0.5 * 9.81, 2 * 0.5 * 4, 2 * 0.5 * 4.905]
    )
    captured = visuals["locomotion-capture-point"]["vectors"][-1]["x"]
    assert captured == pytest.approx(0.05 + 0.3 / math.sqrt(9.81 / 1.09))
    values = visuals["statistics-factorial-ablation"]["values"]
    assert values[1][1] - values[1][0] - values[0][1] + values[0][0] == 15
    for n, series in zip((10, 100), visuals["statistics-wilson-interval"]["series"]):
        p, z = 0.8, 1.96
        denominator = 1 + z * z / n
        center = (p + z * z / (2 * n)) / denominator
        half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denominator
        assert [point[0] for point in series["points"]] == pytest.approx(
            [center - half, center + half], abs=0.00051
        )
