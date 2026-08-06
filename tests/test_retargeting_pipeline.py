"""Regression tests for the complete synthetic retargeting pipeline."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest


pytest.importorskip("scipy")

ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "examples" / "complete_retargeting_pipeline.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("complete_retargeting_pipeline", MODULE_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_vector_optimizer_respects_joint_bounds():
    module = _load_module()
    retargeter = module.VectorOptimizationRetargeter(joint_limits=(0.0, 1.2))
    target_tip = np.array([0.04, 0.01, 0.02], dtype=float)

    joints = retargeter.retarget_finger(
        target_tip=target_tip,
        finger_name="index",
        initial_guess=[0.3, 0.3],
    )

    assert joints.shape == (2,)
    assert np.all(np.isfinite(joints))
    assert np.all(joints >= 0.0)
    assert np.all(joints <= 1.2)
