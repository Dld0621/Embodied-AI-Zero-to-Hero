# Control Basics / 控制基础

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

对机械工程学生来说，闭环就是反馈控制（feedback control），和 PID 调速、伺服阀位控是同一回事。机器人几乎全部使用闭环控制，因为存在摩擦、间隙、负载变化等不确定因素，开环无法保证精度。

> **直觉**：误差 `e(t) = 目标 - 实际`。控制器的任务就是把 `e(t)` 驱赶到 0。

---

## 2. PID 控制

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

选哪种取决于任务：自由空间运动用位置控制最省心；要做柔顺接触、力调节，必须下到力矩层（阻抗控制就运行在力矩层）。

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
2. **速度限制（Velocity Limits）**：单步位移 `|q_{t+1} - q_t|` 不能超过 `dq_max`，否则会失速或伤人。
3. **碰撞避免（Collision Avoidance）**：检查目标位姿是否与环境（桌面、自身）碰撞。
4. **NaN / Inf 检查**：策略网络可能输出非法值，必须拦截。
5. **急停（Emergency Stop）**：异常时立即归零。

违反约束时的处理策略：**clip**（裁剪到合法范围）、**hold**（保持上一个安全动作）、**abort**（归零并停机）。

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
| **Safety Filter** | 关节限位、速度限制、碰撞、NaN、急停 | [`examples/robot_foundation_models/common/safety_filter.py`](../../examples/robot_foundation_models/common/safety_filter.py) |

`SafetyFilter` 类实现了第 6 节的全部检查：`check()` 方法依次做关节限位裁剪、速度裁剪、碰撞检查，违反时按 `CLIP / HOLD / ABORT` 三种策略处理。这正是本文"安全考虑"的工程落地。

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

DLS 里的 `damping` 和本文 PID 里的 `Kd`、阻抗控制里的 `D` 是同一个物理直觉：**加阻尼换稳定性，代价是精度/速度**。`damping=0.06` 是经验值——太大 IK 收敛慢，太小在奇异点附近会爆。

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
steady_idx = np.argmin(np.abs(t - 1.8))   # 取 1.8s 处看稳态
print(f"稳态角度 (t=1.8s): {q_hist[steady_idx]:.4f} rad | 稳态误差: {1.0 - q_hist[steady_idx]:.5f} rad | 超调量: {(q_hist.max() - 1.0) * 100:.1f}%")
```

### 动手实验

修改参数观察变化（调参直觉的来源）：`Kp=30` 看超调振荡；`Kd=0` 看振荡无法被抑制；`Ki=0`（并加大摩擦 `b`）看稳态误差不归零；去掉 anti-windup 且 `tau_max=0.5` 看严重超调。

---

## 9. 检查理解

1. **概念题**：用一句话解释为什么机器人几乎都用闭环控制而不是开环控制。

2. **PID 调参**：给定一个系统响应曲线——上升快但严重振荡，最后稳态有恒定偏差。你应该分别调整 `Kp / Ki / Kd` 中的哪一项？怎么调？

3. **阻抗控制**：如果要做一个擦玻璃的机器人手臂，希望它碰到玻璃时"顺从"不硬顶，你应该把虚拟弹簧刚度 `K` 调大还是调小？为什么？

4. **控制模式选择**：协作机械臂在自由空间做轨迹跟踪时用位置控制；当末端要和人类握手时，应该切换到哪种控制模式？为什么？

5. **频率约束**：项目里机械臂 IK 运行在 25 Hz（`dt = 0.04 s`）。如果系统最快振荡周期是 `0.05 s`，这个控制频率是否足够？依据是什么？

6. **代码题**：在示例代码基础上，把控制器从 PID 改成纯阻抗控制（`tau = K*(q_des - q) + D*(qd_des - qd)`，无积分项），令 `K=20, D=2`，重新跑阶跃响应。比较两种控制器的稳态误差和超调，解释差异。

7. **连接项目**：阅读 [`safety_filter.py`](../../examples/robot_foundation_models/common/safety_filter.py) 的 `check()` 方法。当 `violation_action=CLIP` 且关节超限时，它返回的 `SafetyStatus.safe` 是 `True` 还是 `False`？这种设计合理吗？

> 完成后建议进入 [`09-mujoco-basics.md`](09-mujoco-basics.md)，把这里的控制循环放进物理仿真引擎里运行。
