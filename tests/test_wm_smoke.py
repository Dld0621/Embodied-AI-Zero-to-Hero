"""
tests/test_wm_smoke.py
======================
端到端 Smoke Test：验证 WorldModelPolicyPipeline 能在 CPU 上完整运行。

测试在 CI 环境中用最小参数快速执行（目标 < 60 秒），
确保 Pipeline 的 run() 方法返回正确结构且所有数值有限。

运行：python -m pytest tests/test_wm_smoke.py -v
"""

import sys
import unittest
import importlib.util
import numpy as np
from pathlib import Path

# 将项目根目录加入 Python 路径
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


class TestWMPipelineSmoke(unittest.TestCase):
    """WorldModelPolicyPipeline 端到端 smoke test（仅 CPU）。"""

    @classmethod
    def setUpClass(cls):
        """通过 importlib 加载 examples/ 目录下的 pipeline 模块。"""
        try:
            import torch  # noqa: F401
        except ImportError:
            raise unittest.SkipTest("torch 未安装，跳过 World Model smoke test")

        spec = importlib.util.spec_from_file_location(
            "world_model_vla_pipeline",
            PROJECT_ROOT / "examples" / "world_model_vla_pipeline.py",
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        cls.WMPipeline = module.WorldModelPolicyPipeline

    def test_pipeline_run_on_cpu(self):
        """验证 Pipeline 在 CPU 上完整运行并返回 5 个有限数值结果。"""
        pipeline = self.WMPipeline(device="cpu")
        results = pipeline.run(
            num_demos=5,
            wm_epochs=1,
            bc_epochs=1,
            eval_episodes=2,
            plot=False,
        )

        # 验证返回结果包含所有 5 个 key
        expected_keys = {
            "BC Baseline",
            "WM Data Gen",
            "WM Evaluator",
            "WM Planner",
            "Latent BC",
        }
        self.assertEqual(set(results.keys()), expected_keys)

        # 验证所有值都是有限数值
        for name, value in results.items():
            self.assertIsInstance(
                value, (int, float, np.floating, np.integer),
                f"{name} 的值类型应为数值，实际为 {type(value).__name__}",
            )
            self.assertTrue(
                np.isfinite(value),
                f"{name} 的值应为有限数值，实际为 {value}",
            )


if __name__ == "__main__":
    unittest.main(verbosity=2)
