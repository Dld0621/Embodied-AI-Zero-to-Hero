"""Source-level learning-layout checks; no claim of browser or usability acceptance."""

import json
from pathlib import Path

import numpy as np
import yaml

from scripts import build_learning_home

ROOT = Path(__file__).resolve().parents[1]


def test_generated_homes_preserve_every_stage_and_chapter():
    graph = json.loads((ROOT / "knowledge/manifest.json").read_text(encoding="utf-8"))
    for zh, name in ((True, "index_cn.md"), (False, "index.md")):
        source = (ROOT / "docs" / name).read_text(encoding="utf-8")
        assert source == build_learning_home.home(graph, zh)
        assert source.count('class="study-stage"') == 6
        for node in graph["nodes"]:
            assert f"knowledge-atlas/{node['id']}/index.md" in source
        assert 'class="study-resume" hidden' in source
        assert "VALIDATION.md" in source


def test_navigation_is_six_stable_learning_areas():
    config = yaml.safe_load((ROOT / "mkdocs.yml").read_text(encoding="utf-8"))
    assert len(config["nav"]) == 6
    assert [next(iter(group)) for group in config["nav"]] == [
        "学习中心",
        "基础与图解",
        "交互实验",
        "工程实践",
        "专项与科研",
        "评估与资料",
    ]
    assert config["theme"]["custom_dir"] == "docs/overrides"
    assert config["extra_css"][-1] == "stylesheets/learning-shell.css"
    assert config["extra_javascript"][-1] == "javascripts/learning-shell.js"


def test_reading_controls_are_progressive_and_do_not_submit_progress():
    template = (ROOT / "docs/overrides/main.html").read_text(encoding="utf-8")
    js = (ROOT / "docs/javascripts/learning-shell.js").read_text(encoding="utf-8")
    css = (ROOT / "docs/stylesheets/learning-shell.css").read_text(encoding="utf-8")
    assert 'class="study-toolbar" hidden' in template
    assert 'aria-pressed="false"' in template and 'role="status"' in template
    assert "localStorage" in js and "document$.subscribe(init)" in js
    for forbidden in ("fetch(", "XMLHttpRequest", "sendBeacon", "eval(", "new Function("):
        assert forbidden not in js
    assert "prefers-reduced-motion" in css and "@media print" in css
    assert "study-focus" in css and "study-font" in css
    assert ".md-typeset :is(details, .atlas-answer, .atlas-data) { font-size: inherit; }" in css
    assert "[dir] .md-main .md-main__inner .md-content > .md-content__inner" in css
    assert 'querySelector(".study-domain-nav")' in js


def test_numpy_views_example_does_not_teach_python_list_slicing_as_shared_storage():
    atoms = json.loads((ROOT / "knowledge/atlas/foundations.json").read_text(encoding="utf-8"))[
        "atoms"
    ]
    atom = next(a for a in atoms if a["id"] == "numpy-copy-aliasing")
    assert "a=np.array([1,2,3])" in atom["worked_example"][0]
    a = np.array([1, 2, 3])
    v, c = a[:2], a[:2].copy()
    v[0] = 9
    c[1] = 8
    assert a.tolist() == [9, 2, 3] and c.tolist() == [1, 8]
    original = [1, 2, 3]
    sliced = original[:2]
    sliced[0] = 9
    assert original == [1, 2, 3]


def test_built_reading_contract_rejects_empty_or_incomplete_sites(tmp_path):
    from scripts.check_site_learning_shell import audit_site

    errors, count = audit_site(tmp_path)
    assert count == 2
    assert any("missing learning page" in error for error in errors)
    assert any("search index missing" in error for error in errors)


def test_learning_html_parser_limits_links_to_content():
    from scripts.check_site_learning_shell import ReadingHTML

    parsed = ReadingHTML()
    parsed.feed(
        '<nav><a href="outside">Sidebar</a></nav><article class="md-typeset">'
        '<div class="study-toolbar" hidden></div><h1 id="lesson">Title</h1>'
        '<a href="next/">Next</a></article><script src="learning-shell.js"></script>'
    )
    assert parsed.links == ["next/"]
    assert "hidden" in parsed.classes["study-toolbar"][0]
    assert "lesson" in parsed.ids
    assert parsed.scripts == ["learning-shell.js"]


def test_labs_keep_deep_links_and_worked_examples_without_javascript():
    for name in ("learning-lab.md", "learning-lab-cn.md"):
        text = (ROOT / "docs" / name).read_text(encoding="utf-8")
        for name in ("frames", "kinematics", "control", "timing", "evaluation"):
            assert f'id="{name}"' in text
            assert f'href="#{name}-guide"' in text
