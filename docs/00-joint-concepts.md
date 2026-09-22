# 关节、关节角与 Ctrl 核心概念

> 详细阐述人手与机器人灵巧手在"关节（Joint）"、"关节角（Joint Angle）"和"控制量（Ctrl）"三个层面的定义、区别与联系。这是理解 IK Retargeting 的物理基础。

---

## 1. 关节（Joint）—— 运动的物理连接点

### 1.1 定义

**关节（Joint）** 是连接两个刚体（link）的运动副，它定义了相邻部件之间**允许的运动类型**和**运动轴方向**。

在机器人学中，关节是运动链的基本单元。一个机械臂或灵巧手由一系列关节和连杆组成：

```
连杆0 ——关节0—— 连杆1 ——关节1—— 连杆2 ——关节2—— ... —— 末端执行器
(base)          (link1)          (link2)                    (fingertip)
```

### 1.2 关节类型

| 类型 | 英文 | 符号 | 运动描述 | 示例 |
|------|------|------|---------|------|
| **旋转关节** | Revolute | `hinge` | 绕固定轴旋转（1 DOF） | 手指 MCP/PIP 屈曲 |
| **平移关节** | Prismatic | `slide` | 沿固定轴直线移动（1 DOF） | 线性推杆 |
| **球铰关节** | Ball / Spherical | `ball` | 绕三轴旋转（3 DOF） | 人肩、手腕 |
| **自由关节** | Free | `free` | 6 DOF（3 旋转 + 3 平移） | 手掌在空间的位姿 |
| **固定关节** | Fixed | — | 无相对运动（0 DOF） | 掌心基座 |

### 1.3 URDF / MJCF 中的关节定义

以下是说明字段的 URDF 片段，不是可独立加载的完整机器人文件；具体型号需使用其已核实的模型与限制：

```xml
<!-- 旋转关节示例：食指 MCP 屈曲 -->
<joint name="index_mcp" type="revolute">
  <parent link="palm"/>
  <child link="index_proximal"/>
  <axis xyz="0 0 1"/>          <!-- 绕 Z 轴旋转 -->
  <limit lower="0" upper="1.2"/>  <!-- 运动范围 [0, 1.2] rad -->
</joint>
```

关键字段说明：

| 字段 | 含义 |
|------|------|
| `type="revolute"` | 旋转关节，产生角位移 |
| `axis` | 关节坐标系中的运动轴方向，不是默认世界轴 |
| `limit lower/upper` | 关节角度硬限位（弧度） |
| `origin` | 关节坐标系相对父连杆的位姿（位置和方向） |

`origin` 含旋转时，关节坐标中的 Z 轴通常不等于父系或世界系的 Z 轴。参见 [URDF Joint 定义](https://docs.ros.org/en/rolling/p/urdfdom_headers/generated/classurdf_1_1Joint.html)。

### 1.4 人手关节 vs 机器人关节

| 维度 | 人手关节 | 机器人关节 |
|------|---------|-----------|
| **生物学基础** | 骨骼 + 韧带 + 滑膜 | 轴承 + 电机 + 减速器 |
| **自由度** | 维数取决于建模层级，运动范围受组织限制 | 维数由机械结构定义，角度取值通常仍连续 |
| **运动轴** | 由软组织约束，非严格固定 | 精确定义的机械轴 |
| **反向驱动** | 自然可反向（被动） | 需特殊设计（力控模式） |
| **传感器** | 本体感觉 + 触觉 | 编码器 + 力矩传感器 |

**人手关键关节**：

```
拇指: CMC（腕掌）→ MCP（掌指）→ IP（指间）
食指/中指/环指/小指: MCP → PIP → DIP
```

**机器人灵巧手关节（以 O10 为例）**：

```
每指仅 2 个 revolute 关节:
  - base_joint: 外展/内收（abduction/adduction）
  - curl_joint: 屈曲（flexion，PIP + DIP 耦合）
```

---

## 2. 关节角（Joint Angle）—— 关节的状态量

### 2.1 定义

**关节角（Joint Angle）** 是描述关节当前**配置状态**的标量值：
- 对于 **revolute** 关节：旋转角度（弧度 rad）
- 对于 **prismatic** 关节：线性位移（米 m）

### 2.2 人手的关节角

人手没有直接的"关节角传感器"，关节角需要通过视觉捕捉的 landmarks **间接计算**。

#### 弯曲角（Flexion Angle）

先求相邻三点的内角，再用 `π - 内角` 定义本例的屈曲代理：伸直为 0，直角弯曲为 π/2。这是无符号几何量，不是解剖关节轴上的完整转角。

```python
import numpy as np

def compute_flexion_angle(landmarks, i, j, k):
    """
    三点屈曲代理：π 减去向量 ji 与 jk 的内角；伸直为 0。
    """
    v1 = landmarks[i] - landmarks[j]
    v2 = landmarks[k] - landmarks[j]

    denominator = np.linalg.norm(v1) * np.linalg.norm(v2)
    if not np.isfinite(denominator) or denominator < 1e-12:
        raise ValueError("Landmark segment is missing, non-finite or too short")
    cos_angle = np.dot(v1, v2) / denominator
    cos_angle = np.clip(cos_angle, -1.0, 1.0)
    angle = np.arccos(cos_angle)

    return np.pi - angle  # 屈曲代理，弧度 [0, π]

# 示例：食指 PIP 弯曲角 = landmarks[5] (MCP) - [6] (PIP) - [7] (DIP)
pip_angle = compute_flexion_angle(landmarks, 5, 6, 7)
```

#### 外展角（Abduction Angle）

两根手指根部方向向量的无符号夹角可作为张开程度的代理；它不等于某根手指绕指定掌面轴的有符号外展角，不能直接当作机器人关节目标。

```python
def compute_abduction_angle(landmarks, mcp_a, mcp_b, wrist=0):
    """
    计算两根手指的外展角
    """
    v1 = landmarks[mcp_a] - landmarks[wrist]
    v2 = landmarks[mcp_b] - landmarks[wrist]

    cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-8)
    return np.arccos(np.clip(cos_angle, -1.0, 1.0))
```

#### 人手关节角的特点

| 特点 | 说明 |
|------|------|
| **间接计算** | 从 21 点坐标推导，存在误差 |
| **无硬限位** | 人手关节范围由软组织决定，有弹性 |
| **连续平滑** | 生物运动自然连续，无机械间隙 |
| **个体差异大** | 不同人手尺寸、比例差异显著 |

### 2.3 机器人的关节角

仿真状态可在重置时直接设置；实际机器人通常读取编码器，再向控制器发送目标。修改仿真 `qpos` 不等于物理执行了运动。

#### MuJoCo 中的关节角：`data.qpos`

```python
import mujoco

model = mujoco.MjModel.from_xml_path("o10_hand.xml")
data = mujoco.MjData(model)

# qpos: 存储所有关节的当前位置（广义坐标）
# 对于 revolute 关节，qpos 值 = 关节角（弧度）
print(data.qpos)  # [q0, q1, q2, ..., qn]

# 获取特定关节的索引
joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "index_mcp")
qpos_adr = model.jnt_qposadr[joint_id]  # 该关节在 qpos 中的起始地址
angle = data.qpos[qpos_adr]  # 当前关节角
```

#### MuJoCo 中的关节速度：`data.qvel`

```python
# 本段仅针对已验证存在的 hinge 关节；qpos 与 qvel 地址分别查询。
dof_adr = model.jnt_dofadr[joint_id]
velocity = data.qvel[dof_adr]  # hinge 的角速度，rad/s
```

ball 关节使用 4 个 qpos 元素、3 个 qvel 元素；free 关节分别为 7 和 6，所以 `jnt_qposadr` 不能当作 `jnt_dofadr`。仿真重置需明确关节类型、位置和四元数范围，再调用 `mj_forward` 更新派生量；这不是硬件停止方法。

#### 实际机器人中的关节角

O10 灵巧手的实际控制：

```python
from omnihand import OmniHand2025, Finger, ControlMode, HandType

def set_hand_position(hand: OmniHand2025, positions: list):
    """
    设置 O10 灵巧手关节角
    positions: [10] 列表，每个元素是 [0, 1.2] 范围内的弧度值
    """
    hand.set_all_active_joint_angles(positions)
    print("当前关节角:", hand.get_all_active_joint_angles())
```

#### 机器人关节角的特点

| 特点 | 说明 |
|------|------|
| **直接测量** | 编码器直接读取，精度高（0.01° 级） |
| **硬限位** | 机械结构限制，超出范围会报错或损坏 |
| **离散采样** | 控制频率通常为 100Hz-1kHz |
| **可重复** | 同样的关节角对应同样的姿态 |

### 2.4 关节限位（Joint Limits）

机器人关节有严格的运动范围，Retargeting 时必须考虑：

```python
# O10 灵巧手关节限位
O10_LIMITS = {
    "thumb":   {"base": [0.0, 1.2], "curl": [0.0, 1.2]},
    "index":   {"base": [0.0, 1.2], "curl": [0.0, 1.2]},
    "middle":  {"base": [0.0, 1.2], "curl": [0.0, 1.2]},
    "ring":    {"base": [0.0, 1.2], "curl": [0.0, 1.2]},
    "pinky":   {"base": [0.0, 1.2], "curl": [0.0, 1.2]},
}

# 裁剪到限位范围
def clip_joints(joints, limits):
    return np.clip(joints, limits[:, 0], limits[:, 1])
```

**人手关节角 vs 机器人关节角的核心区别**：
- 人手关节角是**计算出来的**（从 landmarks 推导）
- 机器人关节角是**直接控制/测量的**（从编码器读取）

---

## 3. Ctrl —— 控制量 / 驱动指令

### 3.1 定义

**Ctrl** 是发送给机器人**执行器（Actuator）**的控制指令。它不等于关节角，而是驱动关节到达目标状态的**输入信号**。

### 3.2 MuJoCo 中的 Ctrl：`data.ctrl`

在 MuJoCo 中，`data.ctrl` 是 actuator 的控制数组：

```python
# data.ctrl[i] = 第 i 个 actuator 的控制量
# 控制量的物理含义取决于 actuator 的类型
```

#### Actuator 类型与 Ctrl 含义

| Actuator 类型 | Ctrl 含义 | 公式 | 应用场景 |
|--------------|----------|------|---------|
| **Position** | 目标关节角（下述单位传动比条件） | `torque = kp*(ctrl-qpos)-kv*qvel` | 位置控制 |
| **Motor** | 目标力矩 | `torque = ctrl` | 力控 / 阻抗控制 |
| **Velocity** | 目标角速度（下述条件） | `torque = kv*(ctrl-qvel)` | 速度控制 |
| **General** | 自定义 | 用户定义 | 高级控制 |

表中简式仅适用于单位 `gear=1` 的标量 hinge 传动，且忽略饱和等限制。一般情况下 actuator 的长度/速度、增益、偏置和传动映射共同决定关节广义力；motor 的 `ctrl` 也不普遍等于 N·m。见 [MuJoCo 执行器机制](https://mujoco.readthedocs.io/en/stable/computation/index.html#actuation-model)。

#### Position Actuator（位置控制）

这是灵巧手控制中最常用的类型：

```xml
<!-- MJCF 中的 position actuator 定义 -->
<actuator>
  <position joint="index_mcp" kp="100" kv="10"/>
  <position joint="index_curl" kp="100" kv="10"/>
  <!-- ... -->
</actuator>
```

```python
# 设置目标关节角 → 通过 position actuator 驱动关节到达该角度
data.ctrl[0] = 0.5   # 设置 index_mcp 目标角度为 0.5 rad
data.ctrl[1] = 0.8   # 设置 index_curl 目标角度为 0.8 rad

# 推进仿真一步
mujoco.mj_step(model, data)

# 一步之后，data.qpos[0] 会趋近于 0.5（受 kp/kv 影响）
```

**物理过程**：
```
data.ctrl[i] → actuator → joint torque → joint acceleration →
joint velocity (qvel) → joint angle (qpos)
```

### 3.3 Ctrl 与 Joint Angle 的关系

| 场景 | 关系 | 说明 |
|------|------|------|
| **Position Actuator** | 无负载稳定平衡时 `ctrl ≈ qpos` | 本表沿用单位 gear 的 hinge 假设 |
| **Motor Actuator** | 单位 gear 标量 hinge 时 `ctrl = torque` | 一般仍需传动映射与限幅 |
| **有外力时** | `ctrl ≠ qpos` | 接触力导致 qpos 偏离 ctrl |
| **瞬态过程** | `ctrl ≠ qpos` | 关节正在运动中，尚未到达目标 |

```python
# 示例：展示 ctrl 和 qpos 的差异
data.ctrl[0] = 1.0  # 目标角度 1.0 rad

# 单步仿真后
mujoco.mj_step(model, data)
print(data.qpos[0])  # 可能 = 0.02 rad（还远未达到目标）

# 多步仿真后
for _ in range(100):
    mujoco.mj_step(model, data)
print(data.qpos[0])  # 可能 = 0.99 rad（接近目标）
```

### 3.4 O10 中的 Ctrl 应用

在 O10 MuJoCo 仿真中，通常使用 **position actuator**：

```python
def set_hand_position(joint_angles):
    """
    设置灵巧手关节角
    joint_angles: [10] 目标关节角（弧度）
    """
    for i, angle in enumerate(joint_angles):
        data.ctrl[i] = angle  # 设置 position actuator 的目标角度

    # 推进仿真
    mujoco.mj_step(model, data)
```

在真实 O10 硬件中，`set_all_active_joint_angles()` 内部完成电机位置闭环：

```
目标角度 → 电机驱动器 → 电流 → 力矩 → 关节运动 → 编码器反馈 → PID 闭环
```

### 3.5 Ctrl 的物理单位

下表仍限于上面的单位 gear 标量 hinge。slide 的位置/速度是 m、m/s；其它传动与缩放必须查模型合同，不能只按 actuator 名称猜单位。

| Actuator 类型 | Ctrl 单位 | 说明 |
|--------------|----------|------|
| Position | rad（弧度） | 目标关节角 |
| Motor | N·m（牛·米） | 目标力矩 |
| Velocity | rad/s | 目标角速度 |

---

## 4. 三者的关系：从人手到机器人的完整链条

### 4.1 概念层级对比

```
人手（生物系统）                        机器人（机电系统）
┌─────────────────┐                   ┌─────────────────┐
│  21 点 Landmarks │                   │   URDF/MJCF     │
│  (视觉捕捉)      │                   │   (关节定义)     │
└────────┬────────┘                   └────────┬────────┘
         ↓                                     ↓
┌─────────────────┐                   ┌─────────────────┐
│  弯曲角/外展角   │                   │   data.qpos     │
│  (计算出的角度)  │                   │   (当前关节角)   │
└────────┬────────┘                   └────────┬────────┘
         ↓                                     ↓
┌─────────────────┐                   ┌─────────────────┐
│  无直接对应     │                   │   data.ctrl     │
│  (神经控制)     │                   │   (控制指令)     │
└─────────────────┘                   └────────┬────────┘
                                               ↓
                                         ┌─────────────┐
                                         │  Actuator   │
                                         │  (执行器)    │
                                         └─────────────┘
```

### 4.2 Retargeting 的核心：连接两个世界

Retargeting 的本质是建立**人手关节角**与**机器人 ctrl** 之间的映射：

```
人手 Landmarks ──→ 人手关节角 ──→ [Retargeting] ──→ 机器人关节角 ──→ data.ctrl
    (21点)          (弯曲角)        (映射方法)        (目标qpos)       (控制指令)
```

三种映射方式：

| 方法 | 映射关系 | 说明 |
|------|---------|------|
| **Rule-based** | `ctrl = scale * human_angle + offset` | 直接角度映射 |
| **Vector Opt** | `ctrl = argmin ||FK(ctrl) - target_landmark||` | 任务空间优化 |
| **Learning** | `ctrl = Network(human_landmarks)` | 神经网络端到端 |

### 4.3 控制接口流程（概念伪代码）

以下是待实现接口，不是可直接运行的 O10 控制器。10 个输出与人手关键点没有天然一一对应关系；外展和屈曲不能用同一个三点夹角代替。模型文件、标定、关节名称、传动、索引与反馈地址都必须先核实。

```text
模型合同 = 读取并核对模型、关节名、执行器名、传动与单位
有效观测 = 检查关键点有限性、置信度、尺度、参考系与时间戳
候选关节目标 = 已标定的重定向器(有效观测, 模型合同)
断言目标数量和关节名称一一对应，并满足状态、路径与限幅检查
仿真动作 = 已验证的执行器转换(候选关节目标, 实测状态, 模型合同)
只在仿真中执行并读取按 joint id / qpos address 获取的反馈
保存目标、实测、失败事件和模型配置
```

同为 10 维不代表语义对应；这里不提供通用缩放常数，也不把数组前十项假定为所需关节。完整实践入口见[运动学课程](foundations/07-fk-jacobian-ik.md)和[重定向 Pipeline](pipelines/08-dexterous-retargeting.md)。

### 4.4 关键区别总结

| 概念 | 人手侧 | 机器人侧 | 单位 |
|------|--------|---------|------|
| **关节** | 骨骼 + 软组织 | 机械运动副 | — |
| **关节角** | 从 landmarks 计算 | qpos / 编码器读取 | rad |
| **Ctrl** | 神经肌肉信号（无形） | data.ctrl / 电机指令 | rad / N·m |

**核心要点**：
- **关节**定义了"能怎么动"
- **关节角**描述了"现在在哪"
- **Ctrl**命令了"要去哪"

在 Retargeting 中，我们从人手的"现在在哪"（计算的关节角）推导出机器人的"要去哪"（ctrl），最终通过 actuator 驱动机器人到达目标状态。

---

## 5. 常见问题

### Q1: 为什么 `data.ctrl` 和 `data.qpos` 可能不相等？

**A**: 在 position actuator 中，`ctrl` 是**目标角度**，`qpos` 是**当前角度**。由于：
- 关节需要时间运动（惯性）
- 存在外力干扰（接触力）
- Actuator 的 kp/kv 参数影响响应速度

所以两者通常不相等，只有在稳态无外力时才近似相等。

### Q2: O10 的 10 个 ctrl 值分别控制什么？

**A**: O10 的 ctrl 数组顺序通常为：

```python
ctrl = [
    thumb_abd, thumb_curl,      # 拇指
    index_abd, index_curl,      # 食指
    middle_abd, middle_curl,    # 中指
    ring_abd, ring_curl,        # 环指
    pinky_abd, pinky_curl,      # 小指
]
```

### Q3: 人手没有 ctrl 概念，怎么对应？

**A**: 人手的"ctrl"等价于**神经肌肉激活信号**，无法直接测量。Retargeting 跳过这一层，直接从人手几何状态（landmarks）映射到机器人控制量（ctrl）。

### Q4: 关节角和指尖位置有什么关系？

**A**: 通过**正运动学（FK）**计算：

```python
# 已知关节角，计算指尖位置
tip_position = forward_kinematics(joint_angles, dh_parameters)

# 已知指尖位置，求解关节角（逆运动学 IK）
joint_angles = inverse_kinematics(tip_position, dh_parameters)
```

Vector Optimization Retargeting 就是在任务空间（指尖位置）优化，而非关节空间。
