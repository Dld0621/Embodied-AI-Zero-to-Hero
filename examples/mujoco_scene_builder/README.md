# MuJoCo Scene Builder / MuJoCo 场景搭建模板

**English** · [中文说明](#中文说明)

This example is a small, reusable MuJoCo workcell built only from public, generic primitives. The environment and robot are intentionally separated:

- `scene.xml`: simulation options, lights, camera, floor, table, task object, and target.
- `robot.xml`: a two-link arm, visual/collision geometry, joints, actuators, sites, and sensors.
- `run_scene.py`: loading, name-based inspection, stepping, viewer, rendering, JSON evidence, and model export.

```bash
pip install mujoco numpy matplotlib

# Compile, step, inspect, and write a JSON report.
python examples/mujoco_scene_builder/run_scene.py --check

# Open the interactive viewer.
python examples/mujoco_scene_builder/run_scene.py --viewer

# Render one final frame and export canonical/binary models.
python examples/mujoco_scene_builder/run_scene.py \
  --render results/tutorials/mujoco_scene_builder/frame.png \
  --save-canonical results/tutorials/mujoco_scene_builder/canonical.xml \
  --save-mjb results/tutorials/mujoco_scene_builder/compiled.mjb
```

The smoke report proves that this modular fixture compiles, named elements resolve, and the actuated scene steps with finite state. It does not establish task-policy quality, physical-parameter accuracy, or real-robot transfer.

## 中文说明

这个示例是一套可以直接复制的模块化工作站：环境写在 `scene.xml`，机器人写在 `robot.xml`，两者通过 `<include file="robot.xml"/>` 组合。修改时建议一次只改变一层：

1. 在 `scene.xml` 中调整桌面、物体、目标、相机与光照。
2. 在 `robot.xml` 中调整连杆、关节、执行器、传感器和末端工具。
3. 保持视觉 geom 与碰撞 geom 分离，避免复杂 mesh 直接承担高频碰撞计算。
4. 给所有会在 Python 中访问的 body、joint、geom、site、actuator、sensor 命名。
5. 每次修改先运行 `--check`，再打开 Viewer；不要把“能加载”当成“任务成功”。

完整教程：[MuJoCo 场景搭建与建模](../../docs/tutorials/mujoco-scene-building.md)。
