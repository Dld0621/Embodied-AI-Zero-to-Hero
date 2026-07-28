"""
tests/test_imports.py
=====================
基础导入测试（Smoke Tests）。

确保核心示例模块可以无错误导入，捕获语法错误和
循环依赖等基础问题。

运行：python -m pytest tests/test_imports.py -v
"""

import sys
import importlib
import unittest
from pathlib import Path

# 将项目根目录加入 Python 路径
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


class TestCoreImports(unittest.TestCase):
    """测试核心模块可导入性。"""

    def test_import_numpy_scipy(self):
        """基础科学计算库。"""
        import numpy as np
        import scipy
        self.assertTrue(hasattr(np, 'array'))
        self.assertTrue(hasattr(scipy, '__version__'))

    def test_import_matplotlib(self):
        """绘图库。"""
        import matplotlib
        matplotlib.use('Agg')  # 无 GUI 后端
        import matplotlib.pyplot as plt
        self.assertTrue(callable(plt.figure))


class TestExampleImports(unittest.TestCase):
    """测试示例模块可导入性（不执行主逻辑）。"""

    def test_freshman_zero_to_one(self):
        """大一新生入门示例。"""
        spec = importlib.util.spec_from_file_location(
            "freshman_zero_to_one",
            PROJECT_ROOT / "examples" / "freshman_zero_to_one.py"
        )
        module = importlib.util.module_from_spec(spec)
        # 只加载不执行 __main__
        spec.loader.exec_module(module)
        self.assertTrue(hasattr(module, 'HumanHand21'))
        self.assertTrue(hasattr(module, 'DexMVRetargeter'))

    def test_evaluation_framework(self):
        """评估框架。"""
        spec = importlib.util.spec_from_file_location(
            "evaluation_framework",
            PROJECT_ROOT / "examples" / "evaluation_framework.py"
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        self.assertTrue(hasattr(module, 'EvaluationMetrics'))
        self.assertTrue(hasattr(module, 'benchmark_all_methods'))

    def test_fk_ik_demo(self):
        """FK/IK 演示。"""
        spec = importlib.util.spec_from_file_location(
            "fk_ik_demo",
            PROJECT_ROOT / "examples" / "fk_ik_demo.py"
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        self.assertTrue(hasattr(module, 'PlanarArm2D'))

    def test_minimal_vla(self):
        """最小 VLA（需 torch，无环境时跳过）。"""
        try:
            import torch
        except ImportError:
            self.skipTest("torch 未安装，跳过 VLA 导入测试")
        spec = importlib.util.spec_from_file_location(
            "minimal_vla",
            PROJECT_ROOT / "examples" / "minimal_vla.py"
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        self.assertTrue(hasattr(module, 'MinimalVLA'))

    def test_rl_demo(self):
        """RL 演示。"""
        spec = importlib.util.spec_from_file_location(
            "rl_demo",
            PROJECT_ROOT / "examples" / "rl_demo.py"
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        self.assertTrue(hasattr(module, 'run_demo'))
        self.assertTrue(hasattr(module, 'run_train'))

    def test_world_model_demo(self):
        """世界模型演示。"""
        spec = importlib.util.spec_from_file_location(
            "world_model_demo",
            PROJECT_ROOT / "examples" / "world_model_demo.py"
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        self.assertTrue(hasattr(module, 'run_concept_demo'))


class TestDocsExistence(unittest.TestCase):
    """测试关键文档存在性。"""

    REQUIRED_DOCS = [
        "README.md",
        "CONTRIBUTING.md",
        "CHANGELOG.md",
        "LICENSE",
        "requirements.txt",
        "setup/environment.yml",
        "docs/00-joint-concepts.md",
        "docs/01-what-is-vla.md",
        "docs/12-freshman-zero-to-one.md",
        "docs/13-vla-zero-to-one.md",
        "docs/14-rl-zero-to-one.md",
        "docs/15-world-model-zero-to-one.md",
        "docs/17-research-trends-and-positioning.md",
        "docs/18-frontier-papers-online.md",
    ]

    def test_required_docs_exist(self):
        """所有关键文档必须存在。"""
        for doc in self.REQUIRED_DOCS:
            path = PROJECT_ROOT / doc
            self.assertTrue(
                path.exists(),
                f"Required document missing: {doc}"
            )


class TestBilingualReadme(unittest.TestCase):
    """测试中英文双 README 一致性。"""

    def test_both_readmes_exist(self):
        """README.md 和 README_CN.md 必须同时存在。"""
        self.assertTrue(
            (PROJECT_ROOT / "README.md").exists(),
            "README.md (English) is missing"
        )
        self.assertTrue(
            (PROJECT_ROOT / "README_CN.md").exists(),
            "README_CN.md (Chinese) is missing"
        )

    def test_language_switcher_en(self):
        """README.md 顶部必须有语言切换入口。"""
        en_text = (PROJECT_ROOT / "README.md").read_text(encoding="utf-8")
        self.assertIn("README_CN.md", en_text,
                      "README.md must link to README_CN.md for language switching")

    def test_language_switcher_cn(self):
        """README_CN.md 顶部必须有语言切换入口。"""
        cn_text = (PROJECT_ROOT / "README_CN.md").read_text(encoding="utf-8")
        self.assertIn("README.md", cn_text,
                      "README_CN.md must link to README.md for language switching")

    def test_en_readme_no_mixed_chinese(self):
        """README.md 除语言切换行外不应混入中文段落。"""
        en_text = (PROJECT_ROOT / "README.md").read_text(encoding="utf-8")
        import re
        # 找出所有包含中文的行
        lines_with_cjk = []
        for i, line in enumerate(en_text.split("\n"), 1):
            if re.search(r'[\u4e00-\u9fff]', line):
                lines_with_cjk.append((i, line.strip()[:80]))
        # 允许的唯一中文行：语言切换链接（包含 README_CN.md）
        allowed = [
            (i, line) for i, line in lines_with_cjk
            if "README_CN.md" in line
        ]
        violations = [
            (i, line) for i, line in lines_with_cjk
            if "README_CN.md" not in line
        ]
        if violations:
            self.fail(
                f"README.md contains Chinese text outside language switcher:\n" +
                "\n".join(f"  L{i}: {line}" for i, line in violations)
            )

    def test_section_count_match(self):
        """两版 README 的 ## 章节数量必须一致。"""
        import re
        en_text = (PROJECT_ROOT / "README.md").read_text(encoding="utf-8")
        cn_text = (PROJECT_ROOT / "README_CN.md").read_text(encoding="utf-8")

        en_sections = [m.group(1) for m in re.finditer(r'^## (.+)$', en_text, re.MULTILINE)]
        cn_sections = [m.group(1) for m in re.finditer(r'^## (.+)$', cn_text, re.MULTILINE)]

        self.assertEqual(
            len(en_sections), len(cn_sections),
            f"Section count mismatch: EN={len(en_sections)}, CN={len(cn_sections)}"
        )

    def test_internal_links_valid(self):
        """两版 README 中的内部相对链接必须指向存在的文件。"""
        import re
        for readme_name in ["README.md", "README_CN.md"]:
            text = (PROJECT_ROOT / readme_name).read_text(encoding="utf-8")
            # 提取 markdown 链接 [text](path) 但不包含外部 http/https URL
            links = re.findall(r'\[([^\]]+)\]\(([^)]+)\)', text)
            for label, url in links:
                if url.startswith(("http://", "https://", "#", "mailto:")):
                    continue
                # 去除锚点
                url_clean = url.split("#")[0]
                resolved = (PROJECT_ROOT / url_clean).resolve()
                if not resolved.exists():
                    self.fail(
                        f"Broken link in {readme_name}: [{label}]({url}) → {url_clean} not found"
                    )

    def test_status_markers_match(self):
        """两版 README 的状态标记（✅/🟡/⏳/🔒）数量必须一致。"""
        import re
        en_text = (PROJECT_ROOT / "README.md").read_text(encoding="utf-8")
        cn_text = (PROJECT_ROOT / "README_CN.md").read_text(encoding="utf-8")

        en_count = len(re.findall(r'[✅🟡⏳🔒]', en_text))
        cn_count = len(re.findall(r'[✅🟡⏳🔒]', cn_text))

        self.assertEqual(
            en_count, cn_count,
            f"Status marker count mismatch: EN={en_count}, CN={cn_count}"
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
