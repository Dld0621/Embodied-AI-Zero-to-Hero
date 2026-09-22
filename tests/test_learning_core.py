"""Numerical and content contracts for the supplemental learning course."""

import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import xml.etree.ElementTree as ET

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("learning_core_lab", ROOT / "examples/learning_core_lab.py")
lab = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(lab)


def test_standardizer_uses_only_training_values_and_handles_constants():
    mean, scale = lab.fit_standardizer(np.array([[0.0, 7.0], [2.0, 7.0]]))
    np.testing.assert_allclose(mean, [1, 7])
    np.testing.assert_allclose(scale, [1, 1])
    assert lab.data_experiment()["held_out_transformed"] == [[99.0]]
    groups = [set(ids) for ids in lab.data_experiment()["episode_ids"].values()]
    assert all(not a & b for i, a in enumerate(groups) for b in groups[i + 1:])


@pytest.mark.parametrize("bad", [np.array([]), np.empty((0, 2)), np.array([[np.nan]])])
def test_standardizer_rejects_invalid_data(bad):
    with pytest.raises(ValueError):
        lab.fit_standardizer(bad)


@pytest.mark.parametrize("seed", [0, 7, 19])
def test_manual_backprop_matches_every_parameter_and_preserves_inputs(seed):
    params = lab.make_parameters(seed)
    before = {key: value.copy() for key, value in params.items()}
    x = np.array([[-0.7], [0.2], [0.9]])
    y = np.sin(2 * x)
    assert lab.gradient_check(params, x, y) < 1e-7
    _, grads = lab.loss_and_gradients(params, x, y)
    for key in params:
        np.testing.assert_array_equal(params[key], before[key])
        assert params[key].shape == grads[key].shape


def test_gradient_check_detects_incorrect_derivative(monkeypatch):
    original = lab.loss_and_gradients

    def broken(*args):
        loss, grads = original(*args)
        grads["w1"] = grads["w1"] + 0.1
        return loss, grads

    monkeypatch.setattr(lab, "loss_and_gradients", broken)
    assert lab.gradient_check(lab.make_parameters(), np.array([[0.4]]), np.array([[0.8]])) > 0.09


def test_training_updates_reduce_loss_with_separate_interpolation_points():
    result = lab.gradient_experiment()
    assert result["max_gradient_absolute_error"] < 1e-7
    assert result["final_train_mse"] < result["initial_train_mse"] * 0.1
    assert result["held_out_interpolation_mse"] < 0.03
    assert result == lab.gradient_experiment()


def test_bc_missing_recovery_and_expert_aggregation():
    result = lab.imitation_experiment()
    assert result["bc_gain"] == 0
    assert result["aggregated_gain"] == pytest.approx(-1)
    assert result["bc"]["final_absolute_error"] == pytest.approx(0.6)
    assert result["after_expert_query"]["final_absolute_error"] < 1e-12
    for disturbance in [-1.3, 0.0, 0.4]:
        assert lab.rollout(result["aggregated_gain"], disturbance)["final_absolute_error"] < 1e-12


def test_rollout_observation_action_dynamics_are_aligned():
    result = lab.rollout(-0.5)
    for state, action, following in zip(result["observed_states"], result["actions"], result["next_states"]):
        assert following == pytest.approx(state + action)
    assert result["observed_states"][4] == pytest.approx(0.6)
    assert result["next_states"][4] == pytest.approx(0.3)


def test_multimodal_mean_is_not_a_valid_mode():
    result = lab.multimodal_experiment()
    assert result["mse_optimal_constant"] == 0
    assert result["mean_action_blocked"]
    assert not any(result["expert_actions_blocked"])


def test_cli_reports_finite_json():
    process = subprocess.run(
        [sys.executable, str(ROOT / "examples/learning_core_lab.py"), "--json"],
        check=True, capture_output=True, text=True,
    )
    assert set(json.loads(process.stdout)) == {"data", "gradients", "imitation", "multimodal"}


def test_seven_lessons_have_accessible_diagrams_and_self_checks():
    folder = ROOT / "docs/learning-core"
    lessons = sorted(folder.glob("[0-9][0-9]-*.md"))
    assert len(lessons) == 7
    for lesson in lessons:
        content = lesson.read_text()
        assert "先修：" in content
        assert content.count("<details>") == content.count("</details>") == 2
        assert "实践验收" in content or "结业实践" in content
        assert f"assets/{lesson.stem}.svg" in content
        assert "\ufffd" not in content
    assets = sorted((folder / "assets").glob("*.svg"))
    assert len(assets) == 9
    for asset in assets:
        svg = ET.parse(asset).getroot()
        ns = {"svg": "http://www.w3.org/2000/svg"}
        assert svg.attrib["role"] == "img"
        assert svg.find("svg:title", ns).text
        assert svg.find("svg:desc", ns).text
