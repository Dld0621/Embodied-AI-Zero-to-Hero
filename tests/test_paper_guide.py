"""Guard verified paper-guide corrections; not independent scientific proof."""

import re
from pathlib import Path

SOURCE = Path(__file__).resolve().parents[1] / "docs/02-key-papers.md"


def test_paper_identity_matches_in_entries_and_resource_table():
    text = SOURCE.read_text(encoding="utf-8")
    for obsolete in (
        "https://arxiv.org/abs/2204.02111",
        "https://arxiv.org/abs/2311.11603",
        "https://github.com/physical-intelligence/pi0",
    ):
        assert obsolete not in text
    for corrected in (
        "https://arxiv.org/abs/2212.06817",
        "https://arxiv.org/abs/2312.02976",
        "https://github.com/allenai/spoc-robot-training",
        "https://github.com/Physical-Intelligence/openpi",
    ):
        assert text.count(corrected) >= 2
    assert len(re.findall(r"^### \d+\. ", text, re.MULTILINE)) == 14


def test_paper_guide_distinguishes_model_variants_and_marks_pseudocode():
    text = SOURCE.read_text(encoding="utf-8")
    assert "vanilla OpenVLA" in text and "交叉熵" in text
    assert "81 个 token" in text and "6 帧合计 48 个" in text
    assert "10 步 flow 积分" in text
    assert "机器人控制端到端速率" in text
    for block in re.findall(r"```python\n(.*?)```", text, re.DOTALL):
        assert "denoiser.step" not in block
        assert "cross_entropy(logits" not in block
