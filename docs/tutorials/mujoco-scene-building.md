# MuJoCo Scene Building and Modeling / MuJoCo 场景搭建与建模

> **Scope / 范围：** This is an independent teaching workcell built from public, generic MuJoCo concepts. It contains no current-project algorithms, data, model parameters, experimental protocol, or research conclusion. / 本教程是基于公开通用知识独立构建的教学工作站，不包含当前项目的算法、数据、模型参数、实验协议或研究结论。

## English guide

### What you will build

The committed template separates a robot from its environment and then compiles both into one runtime model:

```text
robot.xml ─┐
           ├─ scene.xml → parser / mjSpec → compiler → mjModel + mjData
workcell ──┘                                      ↓
                                     control → mj_step → state/contact
```

| File | Responsibility |
|:---|:---|
| [`robot.xml`](../../examples/mujoco_scene_builder/robot.xml) | bodies, joints, visual and collision geoms, actuators, sites, sensors |
| [`scene.xml`](../../examples/mujoco_scene_builder/scene.xml) | compiler/options, assets, lights, camera, floor, table, object, target, robot include |
| [`run_scene.py`](../../examples/mujoco_scene_builder/run_scene.py) | loading, inspection, stepping, viewer, rendering, JSON report, MJCF/MJB export |

### Run it

```bash
pip install mujoco numpy matplotlib

# The standalone viewer can also accept drag-and-drop.
python -m mujoco.viewer --mjcf=examples/mujoco_scene_builder/scene.xml

# Deterministic compile-and-step smoke.
python examples/mujoco_scene_builder/run_scene.py --check

# Script-controlled passive viewer.
python examples/mujoco_scene_builder/run_scene.py --viewer

# Offscreen frame plus model exports.
python examples/mujoco_scene_builder/run_scene.py \
  --render results/tutorials/mujoco_scene_builder/frame.png \
  --save-canonical results/tutorials/mujoco_scene_builder/canonical.xml \
  --save-mjb results/tutorials/mujoco_scene_builder/compiled.mjb
```

On macOS, run passive-viewer scripts with `mjpython` as required by the official Python viewer documentation.

### Modeling workflow

1. Write the task contract and success predicate before geometry.
2. Fix units and conventions: meters, kilograms, seconds, radians, right-handed frames.
3. Build the workcell from primitives first; add meshes only after dynamics work.
4. Keep the robot model reusable and include it from the scene.
5. Separate visual geoms (`contype="0" conaffinity="0"`) from simple collision geoms.
6. Check mass, inertia, joint axes, ranges, damping, and actuator limits.
7. Add named sites and sensors for every state the controller or evaluator needs.
8. Compile after each small edit and resolve parser/compiler errors first.
9. Step deterministically, inspect finite state, contacts, limits, and energy behavior.
10. Add perception noise, randomization, controllers, and task evaluation only after the base scene is stable.

Maintained source should remain MJCF/URDF or a packaged MJZ. A compiled MJB loads quickly but is version-specific and cannot replace the editable source model.

---

## 中文完整教程

## 1. MuJoCo 里的“场景”和“模型”分别是什么

MuJoCo 不是 CAD 软件。CAD 主要描述外形；MuJoCo 需要描述**几何、质量、惯量、自由度、驱动、传感、接触和求解参数**。一个完整仿真通常分为四层：

| 层 | 负责内容 | 常用 MJCF 元素 |
|:---|:---|:---|
| 物理世界 | 时间步长、重力、求解器、接触模型 | `<option>`、`<size>` |
| 场景 | 地面、桌子、物体、灯光、相机、目标区域 | `<worldbody>`、`<asset>`、`<visual>` |
| 机器人 | 刚体树、关节、碰撞体、驱动器、传感器 | `<body>`、`<joint>`、`<geom>`、`<actuator>`、`<sensor>` |
| 任务与控制 | 控制循环、状态读取、成功条件、记录 | Python 中的 `mjData`、`mj_step` 与评测代码 |

MJCF/URDF 首先被解析成高层 `mjSpec`，再编译为运行时 `mjModel`；每一次仿真状态保存在独立的 `mjData` 中。**结构修改应发生在 MJCF 或 `mjSpec`，状态和控制修改发生在 `mjData`。**

## 2. 推荐的工程目录

从一个机器人文件和一个场景文件开始：

```text
my_mujoco_project/
├─ scene.xml              # 环境、物体、相机、灯光、全局参数
├─ robot.xml              # 机器人刚体树、驱动器、传感器
├─ assets/
│  ├─ meshes/             # STL / OBJ 等视觉网格
│  └─ textures/           # 纹理
├─ run_scene.py           # 控制、步进、Viewer、记录
└─ tests/                 # 编译、名称、有限值、关节/接触回归
```

主场景组合机器人：

```xml
<mujoco model="my workcell">
  <compiler angle="radian" autolimits="true"/>
  <option timestep="0.002" gravity="0 0 -9.81"/>
  <include file="robot.xml"/>
  <!-- scene assets and worldbody follow -->
</mujoco>
```

`include` 路径相对于主 MJCF 文件。多个文件合并后，全局设置和同类型名称必须兼容；尤其不要让两个模块分别假设 degree/radian 或重复使用同一个 body/geom 名称。

## 3. 第一步：固定单位、坐标和求解设置

```xml
<compiler angle="radian" autolimits="true"/>
<option timestep="0.002"
        gravity="0 0 -9.81"
        integrator="implicitfast"
        cone="elliptic"/>
```

- 长度使用米，质量使用千克，时间使用秒，角度建议统一使用弧度。
- `body pos/quat` 相对父 body；`geom`、`site`、`joint axis` 使用所在 body 的局部坐标。
- `box size="x y z"` 填的是三个**半边长**；最终尺寸为 `2x × 2y × 2z`。
- 使用 `fromto` 定义 capsule 时，`size` 的第一个值是半径，两个端点定义中轴线。
- 先使用默认求解参数。如果出现抖动、穿透或打滑，先检查质量、碰撞几何、初始重叠、时间步和控制增益，再调整 `solref/solimp`。

## 4. 第二步：搭建环境

### 地面、桌面与动态物体

```xml
<worldbody>
  <geom name="floor" type="plane" size="1 1 0.05"/>
  <geom name="table" type="box" pos="0 0 0.40" size="0.48 0.36 0.04"/>

  <body name="object" pos="0.18 0.10 0.485">
    <freejoint name="object_freejoint"/>
    <geom name="object_collision" type="box" size="0.04 0.04 0.04"
          mass="0.12" friction="0.8 0.01 0.001"/>
  </body>
</worldbody>
```

- 没有 joint 的 body 固定在父坐标系中。
- `<freejoint/>` 提供 6 个速度自由度，但对应 `qpos` 为 7 个数：3D 平移 + 四元数。
- 不要通过固定数组下标猜关节位置，应使用 `model.jnt_qposadr[joint_id]`。
- 初始姿态不要穿进地面、桌面或机器人，否则第一步就会产生很大的接触冲量。

### 灯光、相机和任务标记

```xml
<light name="key_light" pos="0 -1 1.6" dir="0 0.5 -1" directional="true"/>
<camera name="overview" pos="1.05 -1.15 0.95"
        xyaxes="0.74 0.67 0 -0.28 0.31 0.91"/>
<site name="task_target" type="cylinder" pos="0.18 -0.16 0.445"
      size="0.075 0.002" rgba="0.20 0.75 0.52 0.35"/>
```

`site` 是轻量坐标标记，适合表达末端点、接触候选点、传感器安装位和任务目标；它本身不等于物理碰撞体。

## 5. 第三步：搭建机器人刚体树

```xml
<body name="arm_base" pos="-0.25 0 0.54">
  <body name="link1">
    <joint name="shoulder" type="hinge" axis="0 0 1" range="-1.4 1.4"/>
    <geom type="capsule" fromto="0 0 0 0.28 0 0" size="0.035"/>

    <body name="link2" pos="0.28 0 0">
      <joint name="elbow" type="hinge" axis="0 0 1" range="-2.2 2.2"/>
      <geom type="capsule" fromto="0 0 0 0.24 0 0" size="0.03"/>
    </body>
  </body>
</body>
```

建模顺序是：父 body 原点 → 关节轴 → 连杆几何 → 子 body 原点。先只做一个关节并在 Viewer 中检查旋转方向，再逐级添加连杆；这样最容易定位坐标错误。

### 视觉几何与碰撞几何分离

```xml
<!-- 好看的视觉模型，不参与碰撞。 -->
<geom name="link_visual" type="mesh" mesh="link_mesh"
      group="2" contype="0" conaffinity="0" mass="0"/>

<!-- 简单稳定的碰撞代理，承担质量/惯量与接触。 -->
<geom name="link_collision" type="capsule" fromto="0 0 0 0.28 0 0"
      size="0.032" group="3" contype="1" conaffinity="1" density="650"/>
```

高面数 mesh 适合显示，不一定适合实时碰撞。优先用 box、sphere、capsule、ellipsoid 或凸分解结果作为碰撞代理。否则可能出现接触不稳定、速度过慢和网格尺度错误。

### 质量与惯量

- 教学模型可以从 `mass` 或 `density` 让编译器推导惯量。
- 真实机器人应尽可能使用可靠 CAD/标定得到的质量、质心和惯量，并检查惯量矩阵物理可行性。
- 同一连杆若同时有视觉 geom 和碰撞 geom，不要让两者重复贡献质量；本模板给视觉 geom 设置 `mass="0"`。
- “能够编译”不代表惯量正确。观察自由落体、摆动频率、静态重力力矩和能量是否合理。

## 6. 第四步：加入执行器与传感器

### 执行器

```xml
<actuator>
  <position name="shoulder_position" joint="shoulder"
            kp="70" kv="8"
            ctrllimited="true" ctrlrange="-1.4 1.4"
            forcelimited="true" forcerange="-24 24"/>
</actuator>
```

| 执行器 | 适合用途 | 注意事项 |
|:---|:---|:---|
| `<motor>` | 直接力矩/广义力控制 | 控制器必须自己闭环；检查 gear 与力矩单位 |
| `<position>` | 位置伺服与入门场景 | `kp/kv` 太高会放大接触抖动 |
| `<velocity>` | 速度跟踪 | 仍需力限制和关节范围 |
| `<general>` | 自定义增益、偏置和传动 | 表达力强，但更容易配置错误 |

控制范围不是安全证明。模型内限幅、Python 安全层、Viewer 调试和硬件安全授权是不同层级。

### 传感器与 site

```xml
<sensor>
  <jointpos name="shoulder_position_sensor" joint="shoulder"/>
  <jointvel name="shoulder_velocity_sensor" joint="shoulder"/>
  <actuatorfrc name="shoulder_force_sensor" actuator="shoulder_position"/>
  <framepos name="end_effector_position_sensor"
            objtype="site" objname="end_effector"/>
</sensor>
```

传感器数据按编译后的 sensor 地址存放在 `data.sensordata`。复杂系统不要依赖声明顺序硬编码切片，应通过名称找到 sensor id，再读取 `model.sensor_adr` 与 `model.sensor_dim`。

## 7. 第五步：加载、控制和检查

```python
from pathlib import Path
import mujoco

scene = Path("examples/mujoco_scene_builder/scene.xml").resolve()
model = mujoco.MjModel.from_xml_path(str(scene))
data = mujoco.MjData(model)
mujoco.mj_forward(model, data)  # 填充 xpos/site_xpos 等派生量

shoulder = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR,
                             "shoulder_position")
for _ in range(1000):
    data.ctrl[shoulder] = 0.3
    mujoco.mj_step(model, data)
```

推荐每次编译后打印：

- `nq / nv / nu`：位置、速度和控制维度。
- `nbody / njnt / ngeom / nsensor`：元素数量是否符合预期。
- 所有运行时需要的名称是否可以通过 `mj_name2id` 找到。
- `qpos/qvel/sensordata` 是否始终有限。
- 关节是否超限、初始状态是否有异常接触、仿真时间是否按 timestep 增长。

本仓库一条命令完成这些检查：

```bash
python examples/mujoco_scene_builder/run_scene.py --steps 1200 --check
```

报告默认保存到 `results/tutorials/mujoco_scene_builder/report.json`，并明确记录它能支持和不能支持的结论。

## 8. Viewer、离屏渲染与模型导出

### 直接打开 Viewer

```bash
python -m mujoco.viewer --mjcf=examples/mujoco_scene_builder/scene.xml
```

也可以打开空 Viewer 后把 XML 拖进去。编辑 XML → 保存 → 重新加载，是搭场景最快的循环。

### 在控制循环中使用被动 Viewer

```bash
python examples/mujoco_scene_builder/run_scene.py --viewer
```

被动 Viewer 不负责推进物理；脚本必须调用 `mj_step` 和 `viewer.sync()`。程序修改共享的 `mjModel/mjData` 或 Viewer 状态时，需要遵守 Viewer 锁与同步约定。

### 离屏渲染

```bash
python examples/mujoco_scene_builder/run_scene.py \
  --render results/tutorials/mujoco_scene_builder/frame.png
```

场景在 `<visual><global offwidth="960" offheight="720"/></visual>` 中声明离屏 framebuffer 大小。服务器渲染还需要可用的 OpenGL/EGL/OSMesa 环境；物理 smoke 不应依赖渲染上下文。

### 导出规范 MJCF 与 MJB

```bash
python examples/mujoco_scene_builder/run_scene.py \
  --save-canonical results/tutorials/mujoco_scene_builder/canonical.xml \
  --save-mjb results/tutorials/mujoco_scene_builder/compiled.mjb
```

- `mj_saveLastXML` 把最近编译的模型写为规范化 MJCF，适合检查 include/default 展开后的最终结构。
- MJB 是编译后二进制模型，加载快但与 MuJoCo 版本绑定，不能反编译为维护源码。
- 维护与版本控制应保留 MJCF/URDF 和资产；复杂可分发模型可进一步打包为 MJZ。

## 9. 接入 CAD、mesh 与 URDF

### CAD / mesh

```xml
<compiler meshdir="assets/meshes"/>
<asset>
  <mesh name="link_mesh" file="link_visual.stl" scale="0.001 0.001 0.001"/>
</asset>
```

导入前检查：

1. CAD 导出单位是否为毫米；若是，转换到米并明确 `scale`。
2. mesh 原点和轴方向是否与关节坐标一致。
3. 法线、重复面、非流形边和破损三角形是否清理。
4. 视觉 mesh 是否降面；碰撞是否使用原语或凸代理。
5. 质量、质心和惯量是否来自可靠数据，而不是由错误尺度 mesh 推断。
6. 资产路径是否相对主 MJCF，可在另一台机器和 CI 中复现。

### URDF

MuJoCo 可以直接加载 URDF：

```python
model = mujoco.MjModel.from_xml_path("robot.urdf")
mujoco.mj_saveLastXML("robot_canonical.xml", model)
```

推荐流程：URDF 保留机器人交换描述 → 加载并修正 meshdir/惯量等扩展 → 保存一次规范 MJCF → 用 `include` 把机器人放进不同场景 → 在 MJCF 层补充执行器、传感器、接触和任务资产。URDF 表达范围比 MJCF 小，而且 URDF 及其 MuJoCo 扩展不会得到与 MJCF 完全相同的 schema 检查，因此必须做名称、维度和动力学回归。

## 10. 把模板改成自己的任务

### 添加新物体

复制 `object` body，改名、位姿、`freejoint` 名称、几何、质量和摩擦。先用 box/cylinder 验证，再替换 mesh。

### 更换机器人

保持 `scene.xml` 的地面、桌子、物体和相机，只替换 `<include file="robot.xml"/>`。新机器人需要重新核对：基座位姿、名称、控制维度、末端 site、碰撞层和默认初始姿态。

### 增加相机与数据采集

添加命名 `<camera>`，再使用 `mujoco.Renderer.update_scene(data, camera="camera_name")`。RGB、深度、关节状态和动作必须用同一仿真时刻记录，并明确 episode 起止和随机种子。

### 增加随机化

一次只随机化一个因素并保留 nominal 对照：物体位姿 → 尺寸/质量 → 摩擦 → 传感噪声 → 控制延迟。随机化范围是实验协议，不应隐藏在代码常数里。

### 定义任务成功

MJCF 负责物理模型，但“成功”通常由 Python 评测器定义。例如：物体中心进入目标区域、保持指定时长、未发生掉落且安全约束未触发。视觉上看起来正确不能代替状态判定。

## 11. 常见错误与排查顺序

| 现象 | 优先检查 | 不要先做什么 |
|:---|:---|:---|
| 编译失败 | XML 行列、重复名称、资产路径、joint/body 结构 | 随机改 solver |
| 模型飞走 | 初始穿透、尺度、质量/惯量、时间步、控制增益 | 关闭重力掩盖问题 |
| 关节方向错误 | body 层级、局部 axis、父子坐标 | 在控制器里反复取负 |
| 物体一直滑 | 接触法向、摩擦、夹持力、碰撞代理、步长 | 只把摩擦调到极大 |
| 接触很慢 | 高面数 mesh、过多碰撞对、复杂 condim | 降低控制频率代替优化 |
| Viewer 不动 | 脚本是否 `mj_step`、是否 `viewer.sync()` | 认为 Viewer 自动推进 |
| 渲染失败 | framebuffer、OpenGL/EGL/OSMesa、相机名 | 把渲染加入核心物理测试 |
| 真机不一致 | 尺度、惯量、摩擦、驱动、延迟、传感噪声 | 把仿真成功当作部署授权 |

推荐排查顺序：**能编译 → 静态姿态正确 → 无异常初始接触 → 无驱动自由运动合理 → 单关节低增益 → 多关节控制 → 物体接触 → 任务与随机化**。

## 12. 完成标准与证据边界

完成本教程后，你应该能够：

- 从空目录搭建一个模块化 MJCF 工作站。
- 正确使用 body、joint、geom、site、actuator、sensor、camera、asset 和 include。
- 分离视觉与碰撞几何，并检查质量、惯量、尺度和初始接触。
- 用名称而非固定下标连接 Python 控制与评测代码。
- 运行 Viewer、离屏渲染、规范 MJCF/MJB 导出和自动 smoke test。
- 清楚区分模型可编译、仿真数值稳定、任务成功和真机有效性。

本模板当前只证明：模块化 MJCF 可以编译、命名元素可解析、带执行器的场景能以有限状态完成确定性步进。它不证明控制策略有效、物理参数已经标定或能够迁移到真实机器人。

## Official references

- [MuJoCo overview and model instances](https://mujoco.readthedocs.io/en/stable/overview.html)
- [MJCF modeling guide](https://mujoco.readthedocs.io/en/stable/modeling.html)
- [MJCF XML reference](https://mujoco.readthedocs.io/en/stable/XMLreference.html)
- [Official Python bindings and Viewer](https://mujoco.readthedocs.io/en/latest/python.html)
- [Programmatic model editing and MJZ](https://mujoco.readthedocs.io/en/latest/programming/modeledit.html)
- [MuJoCo Menagerie model library](https://github.com/google-deepmind/mujoco_menagerie)
