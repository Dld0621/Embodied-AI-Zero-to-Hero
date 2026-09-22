"""Regression guards for documented working directories and data boundaries."""

import ast
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RFM = ROOT / "examples/robot_foundation_models/README.md"
DATA = ROOT / "examples/robot_foundation_models/smolvla/datasets/README.md"
DEX = ROOT / "examples/dexmv_style_retargeting/README.md"


def test_rfm_documented_script_entries_resolve_from_repository_root():
    for path in (RFM, DATA):
        text = path.read_text(encoding="utf-8")
        entries = re.findall(r"^python ([^\s]+\.py)(?:\s|$)", text, re.MULTILINE)
        assert entries
        for entry in entries:
            assert (ROOT / entry).is_file(), entry
        assert "cd ../../../.." not in text
        assert "cd ../../benchmarks" not in text


def test_mock_serialization_is_distinguished_from_real_dataset_compatibility():
    text = DATA.read_text(encoding="utf-8")
    assert "mock=True" in text
    assert "pushcube_mock_parquet" in text
    assert "does **not** include `action_type`" in text
    assert "not a full schema validation" in text
    # Check the actual writer, not a second invented schema in this test.
    tree = ast.parse((ROOT / "examples/robot_foundation_models/common/to_lerobot.py").read_text())
    metadata = next(
        n.value for n in ast.walk(tree)
        if isinstance(n, ast.Assign)
        and any(isinstance(t, ast.Name) and t.id == "meta" for t in n.targets)
    )
    keys = {k.value for k in metadata.keys if isinstance(k, ast.Constant)}
    assert {"action_dim", "state_dim", "n_episodes"} <= keys
    assert "action_type" not in keys


def test_dexmv_readme_does_not_label_formula_or_invalid_comprehension_as_python():
    text = DEX.read_text(encoding="utf-8")
    for block in re.findall(r"```python\n(.*?)```", text, re.DOTALL):
        ast.parse(block)
    assert "不能据此保证 100 Hz" in text
    assert "不等于机器人的全局坐标" in text
