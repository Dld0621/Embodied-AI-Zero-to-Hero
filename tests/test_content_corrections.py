"""Offline regressions for corrected teaching snippets, not whole-course proof.

These tests execute the actual Markdown examples (or their named expressions).
They do not launch viewers, render images, train policies, or operate hardware.
Optional SciPy, MuJoCo and PyTorch checks skip when the dependency is absent;
a skip must not be reported as a successfully reproduced numerical example.
"""

from __future__ import annotations

import ast
import re
import textwrap
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import pytest

FOUNDATIONS = Path(__file__).resolve().parents[1] / "docs" / "foundations"


def _source(name: str) -> str:
    return (FOUNDATIONS / name).read_text(encoding="utf-8")


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


def _execute_offline(tree: ast.AST, label: str, namespace: dict, allowed_imports: set[str]) -> dict:
    """Refuse rendering/I/O additions to these specifically scoped examples.

    This is a regression guard for trusted repository snippets, not a general
    sandbox for executing untrusted Markdown or arbitrary Python programs.
    """
    forbidden_calls = {
        "open",
        "exec",
        "eval",
        "compile",
        "__import__",
        "show",
        "savefig",
        "launch",
        "launch_passive",
        "Renderer",
        "render",
        "write",
        "write_text",
        "write_bytes",
        "save",
        "savez",
        "savetxt",
        "system",
        "Popen",
        "run",
    }
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            assert all(alias.name.split(".")[0] in allowed_imports for alias in node.names)
            assert all("viewer" not in alias.name.split(".") for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            assert (node.module or "").split(".")[0] in allowed_imports
            assert "viewer" not in (node.module or "").split(".")
            assert all(alias.name != "viewer" for alias in node.names)
        elif isinstance(node, ast.Call):
            target = node.func
            name = target.id if isinstance(target, ast.Name) else getattr(target, "attr", "")
            assert name not in forbidden_calls, f"Non-offline call {name!r} in {label}"
    exec(compile(ast.fix_missing_locations(tree), label, "exec"), namespace)
    return namespace


def test_dls_example_checks_the_same_arm_geometry_it_solved():
    name = "07-fk-jacobian-ik.md"
    helpers = ast.parse(_block_containing(name, "def jacobian_2dof"))
    jacobian = [
        node
        for node in helpers.body
        if isinstance(node, ast.FunctionDef) and node.name == "jacobian_2dof"
    ]
    assert len(jacobian) == 1
    namespace = {"np": np}
    _execute_offline(ast.Module(body=jacobian, type_ignores=[]), name, namespace, {"numpy"})
    _execute_offline(
        ast.parse(_block_containing(name, "def ik_dls_2dof")), name, namespace, {"numpy"}
    )

    # The old snippet solved with l2=.8 but verified with fk_2dof's l2=1 default.
    reached = namespace["fk_2dof"](namespace["sol"], l1=namespace["l1"], l2=namespace["l2"])
    residual = np.linalg.norm(reached - namespace["target"])
    np.testing.assert_allclose(namespace["reached"], reached, atol=1e-12)
    assert namespace["residual"] == pytest.approx(residual)
    assert residual < 1e-4


def test_euler_gimbal_lock_example_uses_degrees():
    rotation = pytest.importorskip("scipy.spatial.transform").Rotation
    name = "06-se3-and-rotation.md"
    namespace = {"np": np, "R": rotation}
    _execute_offline(
        ast.parse(_block_containing(name, "rot_lock =")), name, namespace, {"numpy", "scipy"}
    )
    np.testing.assert_allclose(
        namespace["rot_lock"].as_matrix(), namespace["rot_same"].as_matrix(), atol=1e-12
    )


def test_euler_self_check_expressions_do_not_silently_use_radians():
    rotation = pytest.importorskip("scipy.spatial.transform").Rotation
    question = next(
        line
        for line in _source("06-se3-and-rotation.md").splitlines()
        if line.startswith("3. **万向锁题**")
    )
    expressions = [
        text for text in re.findall(r"`([^`]+)`", question) if text.startswith("R.from_euler(")
    ]
    assert len(expressions) == 2
    matrices = []
    for expression in expressions:
        call = ast.parse(expression, mode="eval").body
        assert isinstance(call, ast.Call)
        assert any(
            keyword.arg == "degrees"
            and isinstance(keyword.value, ast.Constant)
            and keyword.value.value is True
            for keyword in call.keywords
        )
        assignment = ast.Assign(targets=[ast.Name(id="result", ctx=ast.Store())], value=call)
        namespace = _execute_offline(
            ast.Module(body=[assignment], type_ignores=[]),
            "Euler self check",
            {"R": rotation},
            set(),
        )
        matrices.append(namespace["result"].as_matrix())
    np.testing.assert_allclose(matrices[0], matrices[1], atol=1e-12)


def test_documented_mujoco_quaternion_conversion_is_a_public_callable():
    mujoco = pytest.importorskip("mujoco")
    name = "06-se3-and-rotation.md"
    expressions = [
        text
        for text in re.findall(r"`([^`\n]+)`", _source(name))
        if text.startswith("mujoco.") and "(result9, quat4)" in text
    ]
    assert len(expressions) == 1
    namespace = {
        "mujoco": mujoco,
        "result9": np.zeros(9),
        "quat4": np.array([np.sqrt(0.5), 0.0, 0.0, np.sqrt(0.5)]),
    }
    _execute_offline(ast.parse(expressions[0]), name, namespace, set())
    # Scalar-first quaternion for +90 degrees around z, not the identity case.
    np.testing.assert_allclose(
        namespace["result9"].reshape(3, 3),
        np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]]),
        atol=1e-12,
    )


def _mujoco_demo_tree() -> ast.Module:
    return ast.parse(_block_containing("09-mujoco-basics.md", 'MJCF = """'))


def _assignment(tree: ast.Module, name: str) -> ast.Assign:
    matches = [
        node
        for node in tree.body
        if isinstance(node, ast.Assign)
        and any(isinstance(target, ast.Name) and target.id == name for target in node.targets)
    ]
    assert len(matches) == 1, f"Expected one assignment for {name}"
    return matches[0]


def test_mujoco_actual_demo_has_radian_limits_and_a_reachable_target():
    mujoco = pytest.importorskip("mujoco")
    tree = _mujoco_demo_tree()
    xml = ET.fromstring(ast.literal_eval(_assignment(tree, "MJCF").value))
    assert xml.find("compiler").get("angle") == "radian"
    assert xml.find("worldbody/geom[@type='plane']") is not None
    namespace = _execute_offline(tree, "MuJoCo teaching demo", {}, {"numpy", "mujoco"})
    model, data = namespace["model"], namespace["data"]
    jnt_id, target = namespace["jnt_id"], namespace["q_target"]
    np.testing.assert_allclose(model.jnt_range[jnt_id], [-1.57, 1.57], atol=1e-12)
    assert model.jnt_range[jnt_id, 0] < target < model.jnt_range[jnt_id, 1]
    assert data.time == pytest.approx(namespace["n_steps"] * model.opt.timestep)
    assert abs(data.qpos[0] - target) < 0.03  # PD has no gravity compensation here.
    assert np.isfinite(data.qpos).all() and np.isfinite(data.qvel).all()
    assert set(model.sensor_type) == {
        mujoco.mjtSensor.mjSENS_JOINTPOS,
        mujoco.mjtSensor.mjSENS_JOINTVEL,
    }
    np.testing.assert_allclose(data.sensordata, [data.qpos[0], data.qvel[0]], atol=1e-12)


def test_mujoco_documented_sphere_extension_can_contact_the_documented_floor():
    pytest.importorskip("mujoco")
    source = _source("09-mujoco-basics.md")
    # Extract the experimental change from the prose, then reuse the actual loop.
    experiment = next(line for line in source.splitlines() if "基础模型的摆长" in line)
    sphere_tags = re.findall(r'`(<geom type="sphere"[^`]+/>)`', experiment)
    targets = re.findall(r"`q_target=([0-9.]+)`", experiment)
    assert len(sphere_tags) == len(targets) == 1
    tree = _mujoco_demo_tree()
    xml_assignment = _assignment(tree, "MJCF")
    xml = ET.fromstring(ast.literal_eval(xml_assignment.value))
    xml.find("worldbody/body[@name='arm']").append(ET.fromstring(sphere_tags[0]))
    xml_assignment.value = ast.Constant(value=ET.tostring(xml, encoding="unicode"))
    _assignment(tree, "q_target").value = ast.Constant(value=float(targets[0]))
    namespace = _execute_offline(tree, "MuJoCo contact extension", {}, {"numpy", "mujoco"})
    data = namespace["data"]
    assert data.ncon > 0
    assert np.isfinite(data.qpos).all() and np.isfinite(data.qvel).all()
    assert data.qpos[0] < namespace["q_target"]  # The floor obstructs this target.


def test_attention_example_checks_unrounded_normalization_when_torch_available():
    torch = pytest.importorskip("torch")
    name = "04-transformer-basics.md"
    tree = ast.parse(_block_containing(name, "class SelfAttention"))
    with torch.random.fork_rng(devices=[]):
        namespace = _execute_offline(tree, name, {}, {"torch"})
    for weights in (namespace["attn_single"], namespace["attn_multi"]):
        assert weights.shape == (1, 4, 4)
        torch.testing.assert_close(weights.sum(dim=-1), torch.ones(1, 4), atol=1e-6, rtol=1e-6)


def test_tokenizer_fence_is_executable_and_retains_color_identity():
    name = "04-transformer-basics.md"
    namespace = _execute_offline(
        ast.parse(_block_containing(name, "def tokenize")), name, {}, set()
    )
    red = namespace["tokenize"]("push the red cube")
    green = namespace["tokenize"]("push the green cube")
    assert len(red) == len(green) == namespace["MAX_LEN"]
    assert red != green
    assert namespace["tokenize"]("unknownword")[0] == namespace["VOCAB"]["<unk>"]
