# Control Basics / 控制基础

> **逐点图解 / Concept close-ups：**[刚体动力学与积分](../knowledge-atlas/robot-rigid-body-dynamics/index.md) · [反馈、轨迹与饱和](../knowledge-atlas/system-feedback-control/index.md) · [力、阻抗与柔顺交互](../knowledge-atlas/system-force-compliance/index.md)。每个小点配原理、算例、图、自测；这是中文细解，保留英文术语。

> English contract: [Foundations overview](README_EN.md#route) · Primary references: [Control](../SOURCES.md#08-control)

> **前置要求**: 完成 [`07-fk-jacobian-ik.md`](07-fk-jacobian-ik.md)（理解 Jacobian、阻尼最小二乘 IK）
> **预计学习时间**: 2–3 小时
> **完成后你能**: 区分开环与闭环控制；理解并手写一个 PID 控制器；理解阻抗控制的弹簧-阻尼类比；区分关节位置 / 速度 / 力矩三种控制模式；实现离散时间控制器；理解安全约束与 Safety Filter 的作用

---

## 目录

1. [开环 vs 闭环控制](#1-开环-vs-闭环控制)
2. [PID 控制](#2-pid-控制)
3. [阻抗控制](#3-阻抗控制)
4. [位置控制 vs 速度控制 vs 力矩控制](#4-位置控制-vs-速度控制-vs-力矩控制)
5. [控制频率与离散时间实现](#5-控制频率与离散时间实现)
6. [安全考虑](#6-安全考虑)
7. [连接项目代码](#7-连接项目代码)
8. [可运行代码：1-DOF PID 阶跃响应](#8-可运行代码1-dof-pid-阶跃响应)
9. [检查理解](#9-检查理解)

---

## 1. 开环 vs 闭环控制

**开环控制（Open-loop）**：控制器发出指令后，不读取实际结果，不修正误差。就像蒙着眼睛倒水——你按预定的时长倒，但不知道杯子满了没。

**闭环控制（Closed-loop）**：控制器持续读取实际状态，与目标比较，根据误差修正指令。就像看着水位线倒水——多了就停，少了就继续。

```
开环:  指令 ──> 执行器 ──> 机器人        (没有反馈)
闭环:  指令 ──> 执行器 ──> 机器人 ──┐
                 ^                  │ 传感器读数
                 └── 误差 = 目标 - 实际 ──┘
```

<div class="dof-principle" role="group" aria-label="PID 闭环控制原理图">
  <p class="dof-principle__caption"><strong>原理图 · Feedback, not a one-shot command.</strong> 误差由“目标减实际”得到；PID 依据当前误差、历史误差与变化速度生成控制量，传感器再把真实结果送回。</p>
  <div class="dof-principle__canvas">
    <svg viewBox="0 0 920 315" role="img" aria-labelledby="control-figure-title control-figure-desc">
      <title id="control-figure-title">PID feedback control loop</title>
      <desc id="control-figure-desc">Reference goes to a summing point, PID controller, actuator and robot. The sensor output feeds back with a negative sign to the summing point.</desc>
      <defs>
        <marker id="control-arrow" markerWidth="8" markerHeight="8" refX="6" refY="3" orient="auto"><path class="dof-diagram-arrow" d="M0,0 L7,3 L0,6 Z"/></marker>
        <marker id="control-arrow-good" markerWidth="8" markerHeight="8" refX="6" refY="3" orient="auto"><path class="dof-diagram-arrow-good" d="M0,0 L7,3 L0,6 Z"/></marker>
      </defs>
      <rect class="dof-diagram-surface" x="38" y="94" width="130" height="72" rx="16"/>
      <circle class="dof-diagram-fill-violet" cx="238" cy="130" r="28"/>
      <rect class="dof-diagram-surface" x="315" y="72" width="185" height="116" rx="16"/>
      <rect class="dof-diagram-surface" x="575" y="94" width="132" height="72" rx="16"/>
      <rect class="dof-diagram-surface" x="768" y="94" width="120" height="72" rx="16"/>
      <text class="dof-diagram-label" x="66" y="124">target r(t)</text><text class="dof-diagram-note" x="66" y="145">desired state</text>
      <text class="dof-diagram-title" x="229" y="136">Σ</text><text class="dof-diagram-note" x="228" y="171">e(t)</text>
      <text class="dof-diagram-title" x="384" y="107">PID controller</text>
      <text class="dof-diagram-math" x="342" y="138">u = P + I + D</text>
      <text class="dof-diagram-note" x="341" y="164">now · history · change</text>
      <text class="dof-diagram-label" x="599" y="124">actuator</text><text class="dof-diagram-note" x="599" y="145">torque / velocity</text>
      <text class="dof-diagram-label" x="792" y="124">robot</text><text class="dof-diagram-note" x="792" y="145">state y(t)</text>
      <path class="dof-diagram-accent" d="M170 130 H204" marker-end="url(#control-arrow)"/>
      <path class="dof-diagram-accent" d="M268 130 H306" marker-end="url(#control-arrow)"/>
      <path class="dof-diagram-accent" d="M502 130 H566" marker-end="url(#control-arrow)"/>
      <path class="dof-diagram-accent" d="M709 130 H760" marker-end="url(#control-arrow)"/>
      <path class="dof-diagram-good" d="M826 170 V240 H238 V167" marker-end="url(#control-arrow-good)"/>
      <text class="dof-diagram-note" x="508" y="258">sensor measurement y(t) · feedback closes the loop</text>
      <text class="dof-diagram-title" x="221" y="112">+</text><text class="dof-diagram-title" x="221" y="158">−</text>
    </svg>
  </div>
</div>

对机械工程学生来说，闭环就是反馈控制（feedback control），和 PID 调速、伺服阀位控是同一回事。机器人几乎全部使用闭环控制，因为存在摩擦、间隙、负载变化等不确定因素，开环无法保证精度。

> **直觉**：误差 `e(t) = 目标 - 实际`。控制器的任务就是把 `e(t)` 驱赶到 0。

---

## 2. PID 控制

> **交互推理 / Interactive reasoning：** 用[反馈控制实验](../learning-lab-cn.md#control)先固定延迟为零，只调 P 与 D；再固定增益、增加延迟。用“误差变化、饱和、测量滞后”解释曲线。实验是带力限幅的一维 PD 模型，不含积分项；无延迟、无饱和时的阻尼比不能直接保证延迟系统或真机稳定。[English lab](../learning-lab.md#control)。

PID 是最经典的闭环控制器，由三项叠加而成：

$$u(t) = K_p\, e(t) + K_i \int_0^t e(\tau)\, d\tau + K_d\, \frac{de(t)}{dt}$$

| 项 | 名称 | 作用 | 物理直觉 |
|:--|:-----|:-----|:---------|
| **P** 比例 | $K_p e$ | 误差越大，输出越大 | 弹簧：偏得越多，拉力越大 |
| **I** 积分 | $K_i \int e\,dt$ | 累积历史误差，消除稳态误差 | 慢慢加力，把残留偏差推到 0 |
| **D** 微分 | $K_d \dot e$ | 误差变化越快，阻尼越大 | 减振器：抑制超调和振荡 |

### 调参直觉

- **Kp 太小**：响应慢，跟不上目标。
- **Kp 太大**：振荡、超调，系统不稳定。
- **Ki 太大**：积分饱和（integral windup），超调严重，响应迟钝。
- **Kd 太大**：对噪声极其敏感（微分会放大高频噪声），执行器抖动。

典型调参顺序：先调 P 直到响应快但轻微振荡 → 加 D 抑制振荡 → 最后加小量 I 消除稳态误差。这就是经典的 **Ziegler-Nichols** 思路的简化版。

> **积分饱和（Windup）**：当执行器饱和（如电机力矩到顶）时，误差仍在积分，I 项会变得很大，导致松开后严重超调。工程上常用 **anti-windup** 机制：饱和时停止积分。

---

## 3. 阻抗控制

PID 关心的是"位置跟踪得准不准"。但机器人要抓取易碎物体或与人协作时，我们需要的是**柔顺性（compliance）**——遇到外力时能"让一让"，而不是硬刚。

阻抗控制把机器人关节表现得像一个**弹簧-阻尼系统**：

$$\tau = K (q_{des} - q) + D (\dot q_{des} - \dot q) + \tau_{ff}$$

- $q_{des}$：期望关节角
- $K$：刚度（弹簧），越大越"硬"
- $D$：阻尼，越大越"稳"，不抖
- $\tau_{ff}$：前馈力矩（如重力补偿）

**物理类比**：想象关节和目标之间连着一根弹簧（K）和一个减振器（D）。目标在哪，弹簧就把关节拉向哪；减振器保证它不震荡。

- **K 大、D 小**：硬而弹，跟踪准但容易抖。
- **K 小、D 大**：软而稳，遇到障碍会顺从让开——这正是协作机器人安全接触所需的"柔顺"。

> **与 PID 的区别**：PID 直接输出控制量去逼位置；阻抗控制显式建模"虚拟弹簧"，让你能独立调节"硬度"和"阻尼"，更适合接触式任务（擦玻璃、装配、人机协作）。

---

## 4. 位置控制 vs 速度控制 vs 力矩控制

机器人底层接口（low-level interface）通常提供三种控制模式：

| 模式 | 输入 | 反馈来源 | 适用场景 |
|:-----|:-----|:---------|:---------|
| **位置控制** | 目标关节角 $q_{des}$ | 编码器 | 轨迹跟踪、点到点运动 |
| **速度控制** | 目标关节速度 $\dot q_{des}$ | 编码器差分 | 连续运动、传送带跟随 |
| **力矩控制** | 目标关节力矩 $\tau$ | 力/电流传感器 | 接触力调节、阻抗控制、柔顺装配 |

**典型级联结构（Cascade Servo）**：在实际的伺服控制器内部，这三种模式并非并列关系，而是层层嵌套的级联控制环。正确的级联方向是——**位置控制是最外环**（读期望位置，输出期望速度）；**速度控制是中环**（读期望速度，输出期望力矩/电流）；**电流/力矩控制是最内环**（读期望电流/力矩，输出电机电压）。外环的输出即内环的设定值：

```
位置外环 (Position loop)
    ↓ 期望速度 dq_des
速度中环 (Velocity loop)
    ↓ 期望力矩/电流 τ_des
电流/力矩内环 (Current/Torque loop)
    ↓ 电机电压 (PWM)
```

也就是说：位置控制之所以"最省心"，正是因为它把下面两层（速度环 + 力矩环）都封装好了；而下到力矩层就等于绕过外面两环，自己接管最底层的物理量。注意不要把层级误解成"力矩环在外、包着位置 PID、再包着速度环"——内环最靠近硬件、跑得最快，外环面向任务目标、跑得较慢。

> **不同机器人 API 暴露的层级不同**：并非所有机器人接口都让你看到完整级联。很多机械臂 SDK 只开放**位置层**（如 `set_joint_positions`），给目标角即可；有些研究型平台开放到**力矩层**（如 `set_joint_torques`），适合做阻抗/柔顺控制；部分平台（如 Franka、UR 的实时接口）允许位置外环 + 力矩前馈同时使用。使用时务必查清接口暴露的是哪一层——直接用力矩指令时，位置/速度环不会自动保护你，需要自己在上层做限速和限位。

选哪种取决于任务与接口。本文的关节阻抗公式直接输出力矩，需要相应力矩接口；但**柔顺接触并非只能用力矩接口**。导纳控制（admittance control）可把测得的外力换算成位置/速度修正量，再交给已有位置/速度伺服执行。两条路径都需验证传感器、带宽、延迟、限幅与接触稳定性，不能仅凭模式名称保证安全。见 [ROS 2 导纳控制器的接口说明](https://control.ros.org/rolling/doc/ros2_controllers/admittance_controller/doc/userdoc.html)。

---

## 5. 控制频率与离散时间实现

真实控制器运行在**离散时间**里，以固定频率 `f_ctrl`（Hz）循环。每个周期 `dt = 1/f_ctrl`：

1. 读传感器 → 计算误差 `e`
2. 更新积分项 `I += e * dt`
3. 计算微分项 `D = (e - e_prev) / dt`
4. 输出 `u = Kp*e + Ki*I + Kd*D`
5. 下发指令，等待下一个 `dt`

**离散 PID 公式**：

```python
integral += error * dt
derivative = (error - prev_error) / dt
output = Kp * error + Ki * integral + Kd * derivative
prev_error = error
```

### 频率多高才够？

- **高频（500–1000 Hz）**：关节伺服、力矩控制。频率太低会导致力矩抖动、不稳定。
- **中频（50–200 Hz）**：笛卡尔空间运动学控制、阻抗控制外环。
- **低频（10–30 Hz）**：VLA 策略输出（视觉推理慢，受限于相机帧率和模型推理）。

> **关键约束**：控制周期 `dt` 必须**远小于**系统的最快动态时间常数，否则离散化会引入相位滞后甚至失稳。一般要求 `dt < T_fastest / 10`。

---

## 6. 安全考虑

机器人是物理设备，控制错误会造成损坏甚至伤人。任何指令下发前都要经过**安全检查**：

1. **关节限位（Joint Limits）**：每个关节有 `[q_min, q_max]`，超出会撞坏机械结构。
2. **速度与单步增量限制**：`|q_{t+1} - q_t|` 是位移增量，不是速度。对周期 `dt`，平均速度约束应写成 `|q_{t+1} - q_t| / dt ≤ v_max`；实际轨迹还需限制瞬时速度、加速度，并检查单位和消息时效。
3. **碰撞避免（Collision Avoidance）**：既检查目标配置，也检查运动路径是否穿过环境或自身；端点不碰撞不等于中途安全。
4. **NaN / Inf 检查**：策略网络可能输出非法值，必须拦截。
5. **停止与急停**：异常时进入机器人已定义并验证的停止流程，独立的硬件急停不能由这个 Python 检查函数代替。停止策略必须结合控制模式、制动与负载，不能一概发零。

处理策略必须明确动作语义：**clip** 后仍需对最终候选动作做完整检查；**hold** 指进入经过验证的保持状态，不是重复上一条增量；**abort** 指拒绝命令并触发约定的停止状态。绝对位置命令的零向量是“回到零位”，可能导致大幅运动；零力矩也可能让负载下落。`safe=True` 只能表达检查合同内的结果，不是整机安全认证。

---

## 7. 连接项目代码

本项目的系统架构（见 [`README.md`](../../README.md)）里，控制与安全是 pipeline 的最后两环：

```
VLA Policy → Robot Adapter → Low-level Controller (PID / Impedance / Joint Servo)
                            → Safety Filter (Joint Limits / Collision / Velocity)
                            → Simulation / Real Robot
```

对应到代码：

| 架构模块 | 项目实现 | 文件 |
|:---------|:---------|:-----|
| **Low-level Controller** | PID / 阻抗 / 关节伺服 | 架构层概念，对应本文的 PID 与阻抗控制 |
| **Safety Filter** | 教学检查接口，存在待修缺陷；不可作为硬件安全屏障 | [`examples/robot_foundation_models/common/safety_filter.py`](../../examples/robot_foundation_models/common/safety_filter.py) |

> **已知缺陷，禁止直接用于硬件安全**：本轮离线审查发现，`SafetyFilter.check()` 的关节/速度裁剪分支会提前返回，可能跳过后续速度或碰撞检查；`max_velocity` 实际比较单步增量，未结合 `dt`；`ABORT` 返回零向量，也没有区分绝对位置与其他动作语义。代码目前不能被描述为“实现了全部安全检查”。本节只纠正文档，未修复该控制源码、未连接真机。修复和复测状态见 [内容正确性审查](../reviews/content-correctness-audit.md)。

### 控制频率与阻尼

项目中的运行频率数据（见 [`tutorials/05-complete-pipeline/README.md`](../../tutorials/05-complete-pipeline/README.md) §5.7.2）：

| 模块 | 频率 | 说明 |
|:-----|:----:|:-----|
| 视觉检测 | 30 Hz | 摄像头帧率限制 |
| 机械臂 IK | **25 Hz** | `damping=0.06, iterations=5` |

这里的 `damping=0.06` 正是阻尼最小二乘 IK（DLS-IK）的阻尼系数（见 [`tutorials/03-vector-optimization/README.md`](../../tutorials/03-vector-optimization/README.md)）：

```python
def damped_least_squares_ik(target, robot_joints, damping=0.06):
    J = compute_jacobian(robot_joints)
    error = target - forward_kinematics(robot_joints)
    delta = J.T @ np.linalg.inv(J @ J.T + damping**2 * np.eye(3)) @ error
    return robot_joints + delta
```

DLS 的 `damping` 是数值正则化参数；PID 的 `Kd`、阻抗的 `D` 则参与速度相关的控制作用。可以用“抑制过激变化”帮助理解，但三者不是同一物理量，单位和稳定性条件也不同。`damping=0.06` 只是该 IK 示例的配置，不能直接搬作电机阻尼增益。

---

## 8. 可运行代码：1-DOF PID 阶跃响应

下面用一个 1 自由度系统（带惯性和摩擦的转轴）演示离散 PID。只需 NumPy + Matplotlib。

```python
"""
1-DOF PID 阶跃响应演示
=====================
模拟一个带惯性和摩擦的转轴，用离散 PID 把它转到目标角度。
运行: python control_basics_pid_demo.py
依赖: numpy, matplotlib
"""
import numpy as np
import matplotlib.pyplot as plt

# --- 被控对象: 单轴二阶系统 ---
# I * q_ddot + b * q_dot = tau   (转动惯量 I, 粘性摩擦 b)
I_inertia = 0.01      # kg·m²  转动惯量
b_friction = 0.1      # N·m·s  粘性摩擦系数
tau_max = 2.0         # N·m    力矩饱和(模拟电机上限)

# --- 离散化参数 ---
dt = 0.001            # s  控制周期 1ms -> 1000 Hz
T_total = 2.0         # s  仿真总时长
n_steps = int(T_total / dt)
t = np.arange(n_steps) * dt

# 目标: 0.5s 时从 0 阶跃到 1.0 rad
q_target = np.where(t >= 0.5, 1.0, 0.0)

# --- PID 参数 ---
Kp, Ki, Kd = 8.0, 5.0, 0.3   # 先调 P, 再加 D 抑振, 最后加少量 I 消稳态误差

# --- 状态 ---
q = 0.0          # 当前角度
q_dot = 0.0      # 当前角速度
integral = 0.0
prev_error = 0.0

q_hist = np.zeros(n_steps)
u_hist = np.zeros(n_steps)

# --- 离散控制循环 ---
for k in range(n_steps):
    error = q_target[k] - q                       # 1. 读传感器算误差
    integral += error * dt                        # 2. 积分项
    derivative = (error - prev_error) / dt        # 3. 微分项

    # anti-windup: 如果输出已饱和且误差同向, 停止积分
    u = Kp * error + Ki * integral + Kd * derivative
    u_sat = np.clip(u, -tau_max, tau_max)         # 执行器饱和
    if (u > tau_max and error > 0) or (u < -tau_max and error < 0):
        integral -= error * dt                    # 撤销本次积分

    # 动力学积分 (半隐式 Euler)
    q_ddot = (u_sat - b_friction * q_dot) / I_inertia
    q_dot += q_ddot * dt
    q += q_dot * dt

    q_hist[k] = q
    u_hist[k] = u_sat
    prev_error = error

# --- 可视化 ---
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 5), sharex=True)
ax1.plot(t, q_target, 'k--', label='目标 (target)')
ax1.plot(t, q_hist, 'b-', label='实际角度 q (Kp=8,Ki=5,Kd=0.3)')
ax1.set_ylabel('角度 (rad)')
ax1.set_title('1-DOF PID 阶跃响应')
ax1.legend(loc='lower right'); ax1.grid(True)

ax2.plot(t, u_hist, 'r-', label='控制力矩 u (饱和±2)')
ax2.axhline(tau_max, color='gray', ls=':', lw=0.8)
ax2.axhline(-tau_max, color='gray', ls=':', lw=0.8)
ax2.set_ylabel('力矩 (N·m)'); ax2.set_xlabel('时间 (s)')
ax2.legend(loc='lower right'); ax2.grid(True)
plt.tight_layout()
plt.savefig('pid_step_response.png', dpi=120)
plt.show()
print("阶跃响应图已保存为 pid_step_response.png")

# --- 打印性能指标 ---
sample_idx = np.argmin(np.abs(t - 1.8))  # 有限时刻的误差，不自动等于稳态误差
print(f"角度 (t=1.8s): {q_hist[sample_idx]:.4f} rad | 此时误差: {1.0 - q_hist[sample_idx]:.5f} rad | 超调量: {(q_hist.max() - 1.0) * 100:.1f}%")
```

### 动手实验

每次只改变一个条件，并记录误差、超调和饱和时长：提高 `Kp`、设 `Kd=0`、或去掉 anti-windup 并降低 `tau_max`，比较响应。结果依赖参数，不能预先断言一定发散或一定严重超调。

本模型只有**粘性摩擦**，静止时 `b*q_dot=0`。在稳定、可达且最终不饱和的条件下，`Ki=0` 的 PD 对恒定目标也可消除误差；增大 `b` 会影响收敛速度，不会凭空制造恒定偏差。若要研究积分对恒定负载的补偿，可明确新增 `tau_load=0.2` N·m，并将动力学改为 `q_ddot=(u_sat-b_friction*q_dot-tau_load)/I_inertia`。此时稳定 PD 的平衡关系是 `Kp*e=tau_load`；用 `Kp=8` 得到预计稳态偏差 0.025 rad。延长运行时间并检查末段速度/误差变化后再讨论稳态，不把 t=1.8 s 的单点读数当作稳态证明。

---

## 9. 检查理解

1. **概念题**：用一句话解释为什么机器人几乎都用闭环控制而不是开环控制。

2. **PID 调参**：给定一个系统响应曲线——上升快但严重振荡，最后稳态有恒定偏差。你应该分别调整 `Kp / Ki / Kd` 中的哪一项？怎么调？

3. **阻抗控制**：如果要做一个擦玻璃的机器人手臂，希望它碰到玻璃时"顺从"不硬顶，你应该把虚拟弹簧刚度 `K` 调大还是调小？为什么？

4. **控制模式选择**：在纯仿真接触任务中，力矩接口的阻抗控制与位置/速度接口的导纳控制各需要什么输入和反馈？为什么两者都不能仅靠模式名称保证接触安全？

5. **频率约束**：项目里机械臂 IK 运行在 25 Hz（`dt = 0.04 s`）。如果系统最快振荡周期是 `0.05 s`，这个控制频率是否足够？依据是什么？

6. **代码题**：在示例代码基础上，把控制器从 PID 改成纯阻抗控制（`tau = K*(q_des - q) + D*(qd_des - qd)`，无积分项），令 `K=20, D=2`，重新跑阶跃响应。比较两种控制器的稳态误差和超调，解释差异。

7. **连接项目**：只阅读 [`safety_filter.py`](../../examples/robot_foundation_models/common/safety_filter.py)，不要连接硬件。当 `CLIP` 分支提前返回时，哪些检查被跳过？为什么绝对位置零向量不能代表急停？结合审查报告列出修复应覆盖的离线回归情形。

> 完成后建议进入 [`09-mujoco-basics.md`](09-mujoco-basics.md)，把这里的控制循环放进物理仿真引擎里运行。
