"""Numerical checks of contracts and causal controls, without GPU/model downloads."""
import importlib.util
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("vla_contract_lab", ROOT / "examples/vla_contract_lab.py")
lab = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(lab)


def test_action_units_are_controlled_by_training_bounds():
    assert float(lab.denormalize(0.5, -0.02, 0.02)) == pytest.approx(0.01)
    assert float(lab.denormalize(0.5, -0.1, 0.1)) == pytest.approx(0.05)
    np.testing.assert_allclose(lab.denormalize([-1, 0, 1], 2, 4), [2, 3, 4])


@pytest.mark.parametrize("value,low,high", [(1.1, -1, 1), (float("nan"), -1, 1), (0, 2, 2), (0, float("inf"), 3)])
def test_invalid_contracts_fail_explicitly(value, low, high):
    with pytest.raises(ValueError):
        lab.denormalize(value, low, high)


def test_teaching_quantizer_has_bounded_error_including_endpoints():
    values = np.linspace(-1, 1, 4097)
    ids, decoded = lab.quantize_centers(values)
    assert ids.min() == 0 and ids.max() == 255
    assert np.max(np.abs(decoded - values)) <= 1 / 256 + 1e-15
    assert np.all(np.diff(decoded) >= 0)


def test_future_padding_does_not_cross_episode_or_affect_masked_loss():
    actions = np.array([[1.0, 2.0], [3.0, 4.0]])
    before = actions.copy()
    chunk, mask = lab.action_window(actions, 1, 4)
    np.testing.assert_array_equal(actions, before)
    assert mask.tolist() == [True, False, False, False]
    prediction = chunk.copy()
    prediction[~mask] = 1e8
    assert np.mean((prediction[mask] - chunk[mask]) ** 2) == 0
    assert np.mean((prediction - chunk) ** 2) > 0


@pytest.mark.parametrize("start,horizon", [(-1, 4), (2, 4), (0, 0), (True, 4)])
def test_action_window_rejects_invalid_offsets(start, horizon):
    with pytest.raises(ValueError):
        lab.action_window([[0.0], [1.0]], start, horizon)


def test_language_controls_break_success_against_original_task():
    result = lab.language_experiment()
    assert result["correct"]["within_0_05_m"] == 4
    assert result["correct"]["mse_against_original_goal"] < 1e-20
    for key in ("swapped", "blank", "visual_only"):
        assert result[key]["within_0_05_m"] == 0
    assert result["swapped"]["mse_against_original_goal"] == pytest.approx(1.0)
    assert result["blank"]["mse_against_original_goal"] == pytest.approx(0.25)


def test_cli_produces_finite_teaching_receipt():
    process = subprocess.run([sys.executable, str(ROOT / "examples/vla_contract_lab.py"), "--json"], check=True, capture_output=True, text=True)
    result = json.loads(process.stdout)
    assert result["evidence_level"] == "teaching"
    assert result["serial_timing"]["commands_per_s"] == pytest.approx(40 / 3)
    assert result["serial_timing"]["last_command_start_age_s"] == pytest.approx(0.25)
