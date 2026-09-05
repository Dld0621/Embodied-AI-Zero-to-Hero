"""Regressions for H02-H06 corrected legacy retargeting lessons.

The tests extract the actual Python fences from the five Markdown pages.  They
exercise only selected offline numerical snippets: no training, downloads,
viewer, network, controller, or hardware access.  PyTorch-dependent shape tests
skip explicitly when PyTorch is unavailable; a skip is not a reproduced result.
"""

from __future__ import annotations

import ast
import re
import textwrap
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest


DOCS = Path(__file__).resolve().parents[1] / "docs"


def _source(name: str) -> str:
    return (DOCS / name).read_text(encoding="utf-8")


def _python_blocks(name: str) -> list[str]:
    pattern = r"^[ \t]*```python[^\S\n]*\n(.*?)^[ \t]*```[^\S\n]*$"
    return [
        textwrap.dedent(block)
        for block in re.findall(pattern, _source(name), flags=re.MULTILINE | re.DOTALL)
    ]


def _block_containing(name: str, marker: str) -> str:
    matches = [block for block in _python_blocks(name) if marker in block]
    assert len(matches) == 1, f"Expected one {marker!r} example in {name}"
    return matches[0]


def _definitions(name: str, marker: str, *definition_names: str) -> ast.Module:
    tree = ast.parse(_block_containing(name, marker))
    selected = [
        node
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name in definition_names
    ]
    assert [node.name for node in selected] == list(definition_names)
    return ast.Module(body=selected, type_ignores=[])


def _execute(tree: ast.AST, name: str, namespace: dict | None = None) -> dict:
    namespace = {} if namespace is None else namespace
    exec(compile(ast.fix_missing_locations(tree), name, "exec"), namespace)
    return namespace


def test_h02_piecewise_mapping_is_continuous_and_odd_at_threshold():
    name = "02-retargeting-taxonomy.md"
    namespace = _execute(
        _definitions(name, "def piecewise_mapping", "piecewise_mapping"),
        name,
        {"np": np},
    )
    mapping = namespace["piecewise_mapping"]

    eps = 1e-9
    left, at, right = (mapping(0.5 - eps), mapping(0.5), mapping(0.5 + eps))
    assert at == pytest.approx(0.6)
    assert left < at < right
    assert max(at - left, right - at) < 2e-9
    for value in (0.2, 0.5, 0.9):
        assert mapping(-value) == pytest.approx(-mapping(value))


@pytest.mark.parametrize(("target", "expected"), [(2.0, 1.0), (-2.0, -1.0)])
def test_h02_trf_solution_respects_active_joint_bound(target: float, expected: float):
    least_squares = pytest.importorskip("scipy.optimize").least_squares
    name = "02-retargeting-taxonomy.md"
    namespace = _execute(
        _definitions(name, "def retarget_task_space", "retarget_task_space"),
        name,
        {
            "np": np,
            "least_squares": least_squares,
            "extract_fingertips": lambda landmarks: landmarks[:5],
        },
    )

    class OneJointHand:
        @staticmethod
        def forward_kinematics(joints):
            return np.full((5, 3), joints[0])

    landmarks = np.zeros((21, 3))
    landmarks[:5] = target
    solution = namespace["retarget_task_space"](
        landmarks,
        OneJointHand(),
        initial_guess=np.array([0.0]),
        joint_lower_bounds=np.array([-1.0]),
        joint_upper_bounds=np.array([1.0]),
    )
    assert -1.0 <= solution[0] <= 1.0
    assert solution[0] == pytest.approx(expected, abs=1e-6)


def _h03_geometry_namespace() -> dict:
    name = "03-human-hand-to-robot-hand.md"
    center = _definitions(name, "def center_at_wrist", "center_at_wrist")
    palm = _definitions(name, "def to_palm_coordinates", "to_palm_coordinates")
    return _execute(
        ast.Module(body=[*center.body, *palm.body], type_ignores=[]),
        name,
        {"np": np},
    )


def test_h03_palm_coordinates_remove_translation_and_rotation():
    namespace = _h03_geometry_namespace()
    rng = np.random.default_rng(7)
    landmarks = rng.normal(size=(21, 3))
    landmarks[0] = [0.0, 0.0, 0.0]
    landmarks[5] = [1.0, 1.0, 0.0]
    landmarks[9] = [0.0, 1.2, 0.1]
    landmarks[17] = [-1.0, 1.0, 0.0]

    angle = 0.73
    rotation = np.array(
        [
            [np.cos(angle), -np.sin(angle), 0.0],
            [np.sin(angle), np.cos(angle), 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    transformed = landmarks @ rotation.T + np.array([3.0, -2.0, 0.7])

    palm_a = namespace["to_palm_coordinates"](landmarks)
    palm_b = namespace["to_palm_coordinates"](transformed)
    np.testing.assert_allclose(palm_a, palm_b, atol=1e-12)
    assert not np.allclose(
        namespace["center_at_wrist"](landmarks),
        namespace["center_at_wrist"](transformed),
    )


def test_h03_palm_frame_rejects_degenerate_landmarks():
    namespace = _h03_geometry_namespace()
    landmarks = np.zeros((21, 3))
    landmarks[9] = [0.0, 1.0, 0.0]
    with pytest.raises(ValueError, match="lateral axis"):
        namespace["to_palm_coordinates"](landmarks)


def test_h03_flexion_is_zero_when_straight_and_rejects_zero_length_bone():
    name = "03-human-hand-to-robot-hand.md"
    namespace = _execute(
        _definitions(name, "def compute_flexion_angle", "compute_flexion_angle"),
        name,
        {"np": np},
    )
    flexion = namespace["compute_flexion_angle"]

    straight = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]])
    right_angle = straight.copy()
    right_angle[2] = [1.0, 1.0, 0.0]
    assert flexion(straight, [0, 1, 2]) == pytest.approx(0.0)
    assert flexion(right_angle, [0, 1, 2]) == pytest.approx(np.pi / 2)

    degenerate = straight.copy()
    degenerate[1] = degenerate[0]
    with pytest.raises(ValueError, match="zero-length"):
        flexion(degenerate, [0, 1, 2])


def test_h03_freejoint_example_uses_distinct_qpos_and_qvel_addresses():
    name = "03-human-hand-to-robot-hand.md"
    tree = ast.parse(_block_containing(name, "jnt_qposadr"))
    model = SimpleNamespace(jnt_qposadr=np.array([2]), jnt_dofadr=np.array([5]))
    data = SimpleNamespace(qpos=np.full(12, 9.0), qvel=np.full(14, 7.0))
    mujoco = SimpleNamespace(
        mjtObj=SimpleNamespace(mjOBJ_JOINT=object()),
        mj_name2id=lambda _model, _kind, name: 0 if name == "palm_free" else -1,
    )
    target = np.array([0.1, 0.2, 0.3])
    _execute(
        tree,
        name,
        {"np": np, "model": model, "data": data, "mujoco": mujoco, "target_position": target},
    )
    np.testing.assert_allclose(data.qpos[2:5], target)
    np.testing.assert_allclose(data.qvel[5:11], 0.0)
    assert data.qpos[5] == 9.0 and data.qvel[2] == 7.0


def test_h04_adaptive_damping_has_defined_threshold_behavior():
    name = "04-optimization-methods.md"
    namespace = _execute(
        _definitions(name, "def adaptive_damping", "adaptive_damping"),
        name,
        {"np": np},
    )
    damping = namespace["adaptive_damping"]
    assert damping(0.0, w0=0.1, lambda0=0.2) == pytest.approx(0.2)
    assert damping(0.05, w0=0.1, lambda0=0.2) == pytest.approx(0.05)
    assert damping(0.1, w0=0.1, lambda0=0.2) == 0.0
    assert damping(0.2, w0=0.1, lambda0=0.2) == 0.0


def test_h04_dls_regularizes_the_near_singular_direction():
    name = "04-optimization-methods.md"
    namespace = _execute(
        _definitions(name, "def dls_ik", "dls_ik"),
        name,
        {"np": np},
    )
    J = np.diag([1.0, 1e-8])
    error = np.array([0.0, 1.0])
    damped = namespace["dls_ik"](J, error, lambda_damp=0.1)
    undamped_pseudoinverse = np.linalg.pinv(J) @ error
    assert damped[1] == pytest.approx(1e-8 / (1e-16 + 0.1**2))
    assert abs(damped[1]) < abs(undamped_pseudoinverse[1])


def test_h05_all_python_fences_are_syntactically_valid():
    name = "05-learning-based-methods.md"
    blocks = _python_blocks(name)
    assert blocks
    for block in blocks:
        ast.parse(block)


def test_h05_time_encoding_is_applied_before_temporal_transformer():
    name = "05-learning-based-methods.md"
    tree = ast.parse(_block_containing(name, "class SinusoidalTimeEncoding"))
    policy = next(
        node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "ImageToHandPolicy"
    )
    forward = next(
        node for node in policy.body if isinstance(node, ast.FunctionDef) and node.name == "forward"
    )
    calls = [
        node
        for node in ast.walk(forward)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
    ]
    time_call = next(node for node in calls if node.func.attr == "time_encoding")
    transformer_call = next(node for node in calls if node.func.attr == "temporal_fusion")
    assert time_call.lineno < transformer_call.lineno


def test_h05_landmark_network_shape_when_torch_is_available():
    torch = pytest.importorskip("torch")
    name = "05-learning-based-methods.md"
    namespace = _execute(ast.parse(_block_containing(name, "class LandmarkToJointNet")), name)
    model = namespace["LandmarkToJointNet"]().eval()
    with torch.no_grad():
        output = model(torch.randn(4, 21, 3))
    assert output.shape == (4, 10)


def test_h05_time_encoding_shape_when_torch_is_available():
    torch = pytest.importorskip("torch")
    import torch.nn as nn

    name = "05-learning-based-methods.md"
    tree = _definitions(name, "class SinusoidalTimeEncoding", "SinusoidalTimeEncoding")
    namespace = _execute(tree, name, {"math": __import__("math"), "torch": torch, "nn": nn})
    encoding = namespace["SinusoidalTimeEncoding"](d_model=8, max_len=5)
    output = encoding(torch.zeros(2, 4, 8))
    assert output.shape == (2, 4, 8)
    assert not torch.allclose(output[:, 0], output[:, 1])


def test_h06_orientation_error_reports_rotation_angle_in_radians():
    name = "06-evaluation-metrics.md"
    namespace = _execute(
        _definitions(name, "def orientation_error", "orientation_error"),
        name,
        {"np": np},
    )
    angle = np.pi / 3
    rotation = np.array(
        [
            [np.cos(angle), -np.sin(angle), 0.0],
            [np.sin(angle), np.cos(angle), 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    assert namespace["orientation_error"](np.eye(3), rotation) == pytest.approx(angle)


def test_h06_jerk_matches_cubic_trajectory_and_rejects_bad_dt():
    name = "06-evaluation-metrics.md"
    namespace = _execute(
        _definitions(name, "def compute_jerk", "compute_jerk"),
        name,
        {"np": np},
    )
    dt = 0.02
    sample_times = np.arange(8) * dt
    cubic = np.column_stack((sample_times**3, 2.0 * sample_times**3))
    expected_norm = np.linalg.norm([6.0, 12.0])
    assert namespace["compute_jerk"](cubic, dt=dt) == pytest.approx(expected_norm)
    with pytest.raises(ValueError, match="dt must be positive"):
        namespace["compute_jerk"](cubic, dt=0.0)


def test_h06_signal_delay_uses_positive_sign_for_robot_lag():
    name = "06-evaluation-metrics.md"
    namespace = _execute(
        _definitions(name, "def estimate_signal_delay", "estimate_signal_delay"),
        name,
        {"np": np},
    )
    rng = np.random.default_rng(11)
    human = rng.normal(size=64)
    lag_samples = 4
    robot = np.concatenate((np.zeros(lag_samples), human[:-lag_samples]))
    assert namespace["estimate_signal_delay"](human, robot, dt=0.01) == pytest.approx(0.04)


def test_h06_empty_evaluation_returns_none_instead_of_nan():
    name = "06-evaluation-metrics.md"
    namespace = _execute(
        _definitions(name, "def comprehensive_evaluation", "comprehensive_evaluation"),
        name,
        {"np": np, "time": time},
    )
    summary = namespace["comprehensive_evaluation"](
        lambda _landmarks: pytest.fail("empty dataset must not call retargeting"),
        [],
        robot_model=None,
    )
    assert summary
    assert all(value is None for value in summary.values())
