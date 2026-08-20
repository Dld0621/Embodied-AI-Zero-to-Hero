# MuJoCo Basics / MuJoCo 仿真基础

> English contract: [Foundations overview](README_EN.md#route) · Primary references: [MuJoCo](../SOURCES.md#09-mujoco)

> **前置要求**: 完成 [`08-control-basics.md`](08-control-basics.md)（理解控制循环与离散时间）
> **预计学习时间**: 4–6 小时
> **完成后你能**: 安装并运行 MuJoCo；读懂 MJCF 模型文件；区分 URDF 与 MJCF 并能互转；写出完整的仿真循环；理解 timestep / gravity / contact / friction 的作用；使用 viewer 交互可视化；从零搭建模块化机器人工作站；检查并导出编译模型

---

## 目录

1. [MuJoCo 是什么](#1-mujoco-是什么)
2. [安装与第一次仿真](#2-安装与第一次仿真)
3. [MJCF 格式](#3-mjcf-格式)
4. [URDF vs MJCF](#4-urdf-vs-mjcf)
5. [仿真循环](#5-仿真循环)
6. [关键概念：timestep / gravity / contact / friction](#6-关键概念timestep--gravity--contact--friction)
7. [MuJoCo Viewer：交互可视化](#7-mujoco-viewer交互可视化)
8. [连接项目代码](#8-连接项目代码)
9. [可运行代码：加载模型并跑仿真循环](#9-可运行代码加载模型并跑仿真循环)
10. [从零搭建完整场景](#10-从零搭建完整场景)
11. [检查理解](#11-检查理解)

---

## 1. MuJoCo 是什么

**MuJoCo**（Multi-Joint dynamics with Contact）是面向机器人学和强化学习的高性能**物理仿真引擎**。给定机器人的模型和当前状态，它用数值积分算出下一时刻的状态——关节角、速度、接触力等。

对机械工程学生来说，它就是一个"虚拟样机台"：把 CAD 模型转成描述文件，加上关节和驱动器，它就替你解动力学方程（牛顿-欧拉 + 接触约束），让你在不碰真机的情况下测试控制算法。它输出的核心是 `data.qpos`（关节位置）、`data.qvel`（关节速度）、`data.sensordata`（传感器）、`data.contact`（接触信息）。

本项目使用 **MuJoCo 3.x**。Python 包可直接通过 `pip` 安装，并包含对应的 MuJoCo 库。

---

## 2. 安装与第一次仿真

```bash
pip install mujoco numpy
python -c "import mujoco; print(mujoco.__version__)"   # 验证安装
```

> **GPU 加速**：MuJoCo 的物理求解主要在 CPU 上，单机仿真一般不需要 GPU。但若要在容器里跑带渲染或大规模并行仿真的工作流，项目推荐用 Docker 挂载 GPU，例如 `docker run --gpus all ...`（参见 [`docs/20-vla-deployment-guide.md`](../20-vla-deployment-guide.md) 的部署示例）。

MuJoCo 自带测试模型，无需外部文件即可跑：

```python
import mujoco
model = mujoco.MjModel.from_xml_string("<mujoco/>")  # 最小空模型
print("nq =", model.nq, " nbody =", model.nbody)
```

---

## 3. MJCF 格式

**MJCF**（MuJoCo XML format）是 MuJoCo 原生的模型描述格式。核心元素：

| 元素 | 作用 | 类比 |
|:-----|:-----|:-----|
| `<body>` | 刚体，有位姿、质量、惯量 | CAD 中的一个零件 |
| `<joint>` | 关节，定义自由度（铰链 hinge / 滑动 slide / 自由 free） | 机械铰链 |
| `<geom>` | 几何体（box / sphere / mesh），用于碰撞和可视 | 零件的几何形状 |
| `<actuator>` | 驱动器，把控制信号映射成关节力矩 | 电机 |
| `<sensor>` | 传感器（关节角、力矩、接触力、相机图像等） | 编码器 / 力传感器 |

一个最小 MJCF（自由下落的盒子）：

```xml
<mujoco>
  <worldbody>
    <body name="box" pos="0 0 1">
      <freejoint/>
      <geom type="box" size="0.1 0.1 0.1" mass="1"/>
    </body>
  </worldbody>
</mujoco>
```

- `<worldbody>` 是世界坐标系根节点；`<body pos="0 0 1">` 定义刚体，初始在 1m 高。
- `<freejoint/>` 给它 6 自由度（可平移可旋转）。
- `<geom>` 给定碰撞与外观（边长 0.2m 方块，质量 1kg）。

> **直觉**：MJCF 把"有什么零件、零件之间怎么连、什么形状、谁来驱动、要测什么"全写在 XML 里。MuJoCo 读进去就建好了完整的动力学模型。

---

## 4. URDF vs MJCF

**URDF**（Unified Robot Description Format）是 ROS 生态通用的机器人描述格式，本项目 `pretrained/urdf/` 目录里就存放着多个机器人的 URDF 文件（见 [`pretrained/urdf/README.md`](../../pretrained/urdf/README.md)）。

| 特性 | URDF | MJCF |
|:-----|:-----|:-----|
| 起源 | ROS 生态 | MuJoCo 原生 |
| 动力学 | 有限（需额外配置） | 一等公民，原生支持接触、摩擦、执行器 |
| 接触/摩擦 | 弱（需 SDF 补充） | 完整建模 |
| MuJoCo 直接加载 | 可以（自动转换） | 可以（原生） |
| 互转 | `from_xml_path` 可直接读 URDF | 加载 URDF 后用 `mj_saveLastXML` 保存规范 MJCF |

**重要事实**：MuJoCo 3.x 可以**直接加载 URDF**，无需先转 MJCF。本项目就是这么做的：

```python
import mujoco
model = mujoco.MjModel.from_xml_path('leap_hand_sim/assets/leap_hand/robot.urdf')   # URDF
model = mujoco.MjModel.from_xml_path('mujoco_menagerie/shadow_hand/scene_right.xml') # MJCF
```

> **什么时候用 MJCF？** 需要精细控制接触、摩擦、执行器增益、传感器时，MJCF 表达力更强。URDF 适合跨工具交换（ROS / Pinocchio / PyBullet / MuJoCo 都能读）。

---

## 5. 仿真循环

MuJoCo 的核心是一个**步进循环（stepping loop）**：每次调用 `mujoco.mj_step(model, data)` 推进一个 `timestep` 的物理。一个完整循环就是"写指令 → 步进 → 读状态"，和 [`08-control-basics.md`](08-control-basics.md) 的离散控制循环结构一致：

```python
import mujoco
model = mujoco.MjModel.from_xml_path("your_model.xml")
data  = mujoco.MjData(model)
for i in range(1000):
    data.ctrl[:] = some_control_signal          # 1. 写控制指令
    mujoco.mj_step(model, data)                 # 2. 推进物理一步
    q, qdot, sensor = data.qpos.copy(), data.qvel.copy(), data.sensordata.copy()  # 3. 读状态
```

区别只是这里的"步进"由物理引擎完成，而不是我们自己手写动力学积分。读出的状态会回送给控制器算下一拍 `ctrl`。

---

## 6. 关键概念：timestep / gravity / contact / friction

<div class="dof-principle" role="group" aria-label="MuJoCo 中控制、物理步进、接触和状态反馈的闭环">
  <p class="dof-principle__caption"><strong>原理图 · One simulation step closes a physical loop</strong>：每一拍控制器写入 <code>ctrl</code>，物理引擎在一个 <code>timestep</code> 内积分动力学并求解接触约束，随后返回新的位置、速度与接触信息。下一拍不能跳过这份反馈。</p>
  <div class="dof-principle__canvas">
    <svg viewBox="0 0 860 244" role="img" aria-labelledby="mujoco-step-title">
      <title id="mujoco-step-title">MuJoCo 单步仿真控制闭环</title><rect class="dof-diagram-surface" x="18" y="61" width="160" height="83" rx="16"/><text class="dof-diagram-title" x="48" y="93">Controller</text><text class="dof-diagram-math" x="52" y="121">ctrlₜ</text><path class="dof-diagram-accent" d="M193 102 H262"/><path class="dof-diagram-arrow" d="M262 102 l-11 -6 v12z"/>
      <rect class="dof-diagram-fill-blue" x="280" y="42" width="274" height="123" rx="18"/><text class="dof-diagram-title" x="356" y="76">mujoco.mj_step</text><text class="dof-diagram-label" x="314" y="108">actuator → dynamics → integration</text><text class="dof-diagram-label" x="314" y="133">contact constraints + friction</text><text class="dof-diagram-note" x="314" y="151">advance exactly one timestep Δt</text><path class="dof-diagram-accent" d="M569 102 H638"/><path class="dof-diagram-arrow" d="M638 102 l-11 -6 v12z"/>
      <rect class="dof-diagram-surface" x="656" y="43" width="184" height="122" rx="18"/><text class="dof-diagram-title" x="686" y="75">State</text><text class="dof-diagram-math" x="684" y="106">qpos, qvel</text><text class="dof-diagram-math" x="684" y="131">contact, sensors</text><path class="dof-diagram-violet" d="M747 181 C747 220 98 220 98 158"/><path class="dof-diagram-arrow-violet" d="M98 158 l-7 11 h13z"/><text class="dof-diagram-note" x="344" y="220">read state → compute the next control action</text>
    </svg>
  </div>
</div>

### timestep（步长 `model.opt.timestep`）

每个 `mj_step` 推进的物理时间，默认 `0.002 s`（500 Hz）。步长越小越精确但越慢。接触丰富的任务（灵巧手抓握）建议 `1–2 ms`，刚体大范围运动可用 `2–5 ms`。**稳定性铁律**：timestep 必须小于系统最快动态周期，否则数值积分发散（和第 8 篇的频率约束同理）。

### gravity（重力 `model.opt.gravity`）

默认 `[0, 0, -9.81]`。仿真里"地面"和真实重力一致，控制算法才需要做重力补偿。

### contact（接触）

MuJoCo 用**软接触**模型：两个 geom 重叠时产生法向力（像弹簧推开），加上切向摩擦力。`data.ncon` 是当前接触对数：

```python
for i in range(data.ncon):
    force = np.zeros(6); mujoco.mj_contactForce(model, data, i, force)
    print(f"接触 {i}: 法向力 {force[0]:.2f} N")
```

`mj_contactForce` 返回接触坐标系中的 3D force + 3D torque；MuJoCo 约定接触坐标系的第一个轴（x）是法向，因此法向力是 `force[0]`，不是通常可视化直觉中的 z 分量。

### friction（摩擦）

每个 geom 有摩擦系数 `friction="1 0.005 0.0001"`，分别是滑动摩擦、扭转摩擦、滚动摩擦。摩擦调不对会导致物体打滑或卡死——这是仿真和真机对齐（sim-to-real）的关键参数。

> **工程直觉**：仿真接触参数（刚度、阻尼、摩擦）往往不等于真实物理值，而是"调出来让仿真行为接近真机"的等效值。详见 [`docs/19-sim-to-real-guide.md`](../19-sim-to-real-guide.md)。

---

## 7. MuJoCo Viewer：交互可视化

MuJoCo 3.x 提供原生交互窗口 `mujoco.viewer`，可实时旋转、缩放、查看接触力：

```python
import mujoco, mujoco.viewer

model = mujoco.MjModel.from_xml_path("your_model.xml")
data  = mujoco.MjData(model)

with mujoco.viewer.launch_passive(model, data) as viewer:
    while viewer.is_running():
        data.ctrl[:] = 0.0            # 在此写控制指令
        mujoco.mj_step(model, data)
        viewer.sync()                 # 刷新画面
```

`launch_passive` 是非阻塞模式：循环继续跑物理，窗口只负责显示。`launch`（阻塞模式）适合纯交互调试。无显示器的服务器（Docker 容器、CI）用**离屏渲染** `mujoco.renderer.Renderer` 出图，不开窗口——项目里 [`run_pipeline.py`](../../examples/dexmv_style_retargeting/run_pipeline.py) 就用了 `from mujoco import renderer`。

---

## 8. 连接项目代码

本项目 [`pretrained/urdf/`](../../pretrained/urdf/) 目录托管了多个机器人的 URDF / MJCF 模型，可直接被 MuJoCo 加载：

| 模型 | 路径 | 类型 | 来源 |
|:-----|:-----|:-----|:-----|
| LEAP Hand | `leap_hand_sim/` | URDF | LEAP_Hand_Sim |
| Shadow Hand | `mujoco_menagerie/shadow_hand/` | MJCF | MuJoCo Menagerie |
| Allegro Hand | `allegro_hand_right/` | URDF | Allegro ROS |
| Franka FR3 | `mujoco_menagerie/franka_fr3/` | MJCF | MuJoCo Menagerie |

[`examples/dexmv_style_retargeting/dexmv_retargeting.py`](../../examples/dexmv_style_retargeting/dexmv_retargeting.py) 是项目里直接使用 MuJoCo 的典型例子，演示了第 5 节的完整循环：

```python
import mujoco
self.model = mujoco.MjModel.from_xml_path(model_path)
self.data  = mujoco.MjData(self.model)
# 按名字查刚体 id
self.body_ids = [mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, n)
                 for n in fingertip_body_names]
# 读取可控关节 (排除 freejoint / world), 并读关节限位
for i in range(self.model.njnt):
    if self.model.jnt_type[i] in (mujoco.mjtJoint.mjJNT_HINGE, mujoco.mjtJoint.mjJNT_SLIDE):
        self.joint_ids.append(i)
        lo, hi = self.model.jnt_range[i]   # 关节限位
```

可以看到：`mj_name2id` 按名字查刚体、`jnt_type` 区分关节类型、`jnt_range` 读取限位——这正是第 3 节 MJCF 元素在 Python API 里的对应。这些关节限位随后会喂给 [`SafetyFilter`](../../examples/robot_foundation_models/common/safety_filter.py)（见第 8 篇），把"仿真读到的限位"变成"运行时安全约束"。

---

## 9. 可运行代码：加载模型并跑仿真循环

下面代码**不依赖任何外部模型文件**——用 `from_xml_string` 内联一个最小 MJCF，模拟一个带关节和力矩传感器的摆。即使没装 MuJoCo，代码也完整可读；装了 MuJoCo 可直接运行。

```python
"""MuJoCo 最小仿真循环: 内联 MJCF (单关节摆 + 力矩执行器 + 关节角传感器)。
运行: python mujoco_basics_demo.py  依赖: pip install mujoco numpy (未安装会打印提示并退出)"""
try:
    import mujoco
except ImportError:
    print("未检测到 mujoco, 请先安装: pip install mujoco")
    raise

import numpy as np

# --- 1. 内联 MJCF 模型 ---
MJCF = """
<mujoco model="single_pendulum">
  <option timestep="0.002" gravity="0 0 -9.81"/>
  <worldbody>
    <body name="arm" pos="0 0 0.5">
      <joint name="shoulder" type="hinge" axis="0 1 0" range="-1.57 1.57"/>
      <geom type="capsule" fromto="0 0 0 0.3 0 0" size="0.02" mass="0.5"/>
      <site name="tip" pos="0.3 0 0" size="0.01"/>
    </body>
  </worldbody>
  <actuator>
    <motor name="shoulder_torque" joint="shoulder" gear="1"/>
  </actuator>
  <sensor>
    <jointpos name="shoulder_pos" joint="shoulder"/>
    <jointvel name="shoulder_vel" joint="shoulder"/>
  </sensor>
</mujoco>
"""

# --- 2. 加载模型 ---
model = mujoco.MjModel.from_xml_string(MJCF)
data  = mujoco.MjData(model)
print(f"模型加载成功: nq={model.nq}, nu={model.nu}, nbody={model.nbody}, "
      f"timestep={model.opt.timestep*1000:.1f}ms, gravity={model.opt.gravity.tolist()}")
jnt_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "shoulder")
print(f"  关节 shoulder: type={model.jnt_type[jnt_id]}, range={model.jnt_range[jnt_id].tolist()}")
# --- 3. PD 控制器 (回顾第 8 篇) ---
Kp, Kd = 50.0, 5.0
q_target = 1.0           # 目标角度 1 rad
# --- 4. 仿真循环 ---
n_steps = 1000
q_hist = np.zeros(n_steps)
print("\n开始仿真循环 (1000 步)...")

for i in range(n_steps):
    q, qd = data.qpos[0], data.qvel[0]              # 读状态
    tau = Kp * (q_target - q) - Kd * qd              # PD -> 力矩
    data.ctrl[0] = tau                                # 写指令
    mujoco.mj_step(model, data)                       # 步进物理
    sensor_pos = data.sensordata[0]                    # 读传感器 (按 <sensor> 声明顺序)
    sensor_vel = data.sensordata[1]
    q_hist[i] = sensor_pos
    if i % 200 == 0:
        print(f"  step {i:4d}: q={sensor_pos:+.3f} rad, "
              f"qd={sensor_vel:+.3f} rad/s, tau={tau:+.2f} N·m, ncon={data.ncon}")

# --- 5. 打印结果 ---
print(f"\n仿真结束: 最终角度={q_hist[-1]:.4f} rad (目标 {q_target}), "
      f"稳态误差={q_target - q_hist[-1]:.5f}, 超调={(q_hist.max() - q_target)*100:.1f}%")
print("提示: 把循环放进 mujoco.viewer.launch_passive 即可看到动画。")
```

### 动手实验

`Kp=500` 看振荡；`timestep=0.02` 看仿真失稳；摆末端加 `<geom type="sphere" size="0.05" mass="2" pos="0.5 0 0"/>` 看接触地板时 `data.ncon` 变化；用 `viewer.launch_passive` 包住循环看实时动画。

---

## 10. 从零搭建完整场景

仓库提供一套可以直接复制的模块化工作站：

```text
examples/mujoco_scene_builder/
├─ scene.xml       # 地面、桌面、物体、目标、灯光、相机与全局参数
├─ robot.xml       # 刚体树、关节、视觉/碰撞几何、执行器与传感器
├─ run_scene.py    # 加载、检查、控制、Viewer、渲染与导出
└─ README.md       # 最短使用说明
```

最小使用流程：

```bash
# 1. 编译、运行、检查名称/状态/关节限位，并生成 JSON 报告
python examples/mujoco_scene_builder/run_scene.py --check

# 2. 打开脚本控制的交互 Viewer
python examples/mujoco_scene_builder/run_scene.py --viewer

# 3. 离屏渲染，并导出展开后的规范 MJCF 与编译 MJB
python examples/mujoco_scene_builder/run_scene.py \
  --render results/tutorials/mujoco_scene_builder/frame.png \
  --save-canonical results/tutorials/mujoco_scene_builder/canonical.xml \
  --save-mjb results/tutorials/mujoco_scene_builder/compiled.mjb
```

模板使用 `<include file="robot.xml"/>` 把机器人和场景分开；视觉 geom 不参与碰撞，简单碰撞 geom 承担接触与质量/惯量；所有需要在 Python 中读取的元素都使用稳定名称。回归测试会检查模型编译、命名元素、有限状态、关节范围、视觉/碰撞层和 MJCF/MJB 往返加载。

完整的逐步说明、CAD/mesh/URDF 接入、执行器/传感器配置和故障排查见：

- [MuJoCo 场景搭建与建模完整教程](../tutorials/mujoco-scene-building.md)
- [可复制示例目录](../../examples/mujoco_scene_builder/README.md)

## 11. 检查理解

1. **概念题**：URDF 和 MJCF 都能描述机器人，为什么接触丰富的任务（如灵巧手抓握）更推荐用 MJCF？

2. **MJCF 元素**：解释 `<joint>`、`<geom>`、`<actuator>`、`<sensor>` 四个元素分别对应真实机器人的什么部件。

3. **仿真循环**：为什么必须先 `data.ctrl[:] = ...` 再 `mj_step`，而不能反过来？

4. **timestep 选择**：一个灵巧手任务，最快振荡周期约 `0.01 s`。你会把 `timestep` 设成多少？为什么不能用 `0.01 s`？

5. **接触与摩擦**：代码里 `data.ncon` 在什么时候会从 0 变成非零？如果仿真里物体一直打滑，应调整 MJCF 里的哪个参数？

6. **代码题**：在示例基础上，加一个 `<camera>` 元素并用 `mujoco.renderer.Renderer` 每隔 100 步离屏渲染一张图保存为 PNG。

7. **连接项目**：阅读 [`dexmv_retargeting.py`](../../examples/dexmv_style_retargeting/dexmv_retargeting.py) 的 `__init__`。它如何区分"可控关节"和"自由关节"？读出的 `joint_limits` 随后被用来做什么（结合第 8 篇的 `SafetyFilter`）？

> 完成后建议进入 [`10-dataset-and-training.md`](10-dataset-and-training.md)，学习如何把仿真里采集的数据组织成训练数据集。
