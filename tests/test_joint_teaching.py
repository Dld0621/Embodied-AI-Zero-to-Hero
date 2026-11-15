"""Regressions for specific corrected joint explanations, not robot safety checks."""

import ast
import re
from pathlib import Path

import numpy as np
import pytest

SOURCE = Path(__file__).resolve().parents[1] / "docs/00-joint-concepts.md"


def blocks():
    return re.findall(r"```python\n(.*?)\n```", SOURCE.read_text(encoding="utf-8"), re.S)


def test_documented_flexion_is_zero_for_straight_landmarks_and_rejects_degeneracy():
    tree = ast.parse(next(block for block in blocks() if "def compute_flexion_angle(" in block))
    function = next(node for node in tree.body if isinstance(node, ast.FunctionDef))
    namespace = {"np": np}
    exec(compile(ast.Module(body=[function], type_ignores=[]), str(SOURCE), "exec"), namespace)
    flexion = namespace["compute_flexion_angle"]
    straight = np.array([[-1.0, 0, 0], [0, 0, 0], [1, 0, 0]])
    bent = np.array([[0.0, 1, 0], [0, 0, 0], [1, 0, 0]])
    assert flexion(straight, 0, 1, 2) == pytest.approx(0)
    assert flexion(bent, 0, 1, 2) == pytest.approx(np.pi / 2)
    for invalid in (np.zeros((3, 3)), np.full((3, 3), np.nan)):
        with pytest.raises(ValueError):
            flexion(invalid, 0, 1, 2)


def test_documented_velocity_uses_its_own_address_after_a_free_joint():
    mujoco = pytest.importorskip("mujoco")
    model = mujoco.MjModel.from_xml_string("""<mujoco><worldbody>
      <body pos="0 0 1"><freejoint/><geom type="sphere" size=".1"/>
        <body pos="0 0 .2"><joint name="hinge" type="hinge"/>
          <geom type="sphere" size=".1"/>
        </body></body></worldbody></mujoco>""")
    data = mujoco.MjData(model)
    joint_id = model.joint("hinge").id
    qpos_adr = model.jnt_qposadr[joint_id]
    dof_adr = model.jnt_dofadr[joint_id]
    assert qpos_adr != dof_adr and qpos_adr >= model.nv
    data.qvel[dof_adr] = 0.7
    source = next(block for block in blocks() if "velocity = data.qvel" in block)
    namespace = {"model": model, "data": data, "joint_id": joint_id, "qpos_adr": qpos_adr}
    exec(compile(source, str(SOURCE), "exec"), namespace)
    assert namespace["velocity"] == pytest.approx(0.7)
