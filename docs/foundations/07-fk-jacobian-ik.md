# FK, Jacobian & IK 正运动学、雅可比与逆运动学

> English contract: [Foundations overview](README_EN.md#route) · Primary references: [Kinematics, Jacobians, and IK](../SOURCES.md#07-kinematics-jacobians-and-ik)

> **前置要求**: [`05-coordinate-transform.md`](05-coordinate-transform.md)、[`06-se3-and-rotation.md`](06-se3-and-rotation.md)（齐次变换、SO(3)/SE(3)）
> **预计学习时间**: 3–4 小时
> **完成后你能**: 用 DH 参数或几何法写正运动学，计算 Jacobian 并判断奇异点，用解析法和阻尼最小二乘数值法求解逆运动学；看懂项目中 `fk_ik_demo.py` 与 `finger_chain_3d.py` 的实现。

---

## 1. 正运动学（FK）

**正运动学 (Forward Kinematics, FK)**：在机器人拓扑、关节状态和坐标约定都已给定时，计算末端执行器位姿；确定的运动学模型会给出确定的几何位姿。四元数等表示本身可能不唯一（例如 $q$ 与 $-q$ 表示同一旋转），闭链机构还需满足其约束，因此不宜笼统写成“任何情况下表示都唯一”。

```
关节角 q = [θ₁, θ₂, ...]  ──FK──►  末端位姿 T (位置 + 朝向)
```

<div class="dof-principle" role="group" aria-label="正逆运动学与雅可比原理图">
  <p class="dof-principle__caption"><strong>原理图 · FK, IK, and local linearization.</strong> FK 沿连杆把关节角映射到末端；IK 从目标反求关节角；Jacobian 描述当前位置附近“小关节变化如何变成小末端变化”。</p>
  <div class="dof-principle__canvas">
    <svg viewBox="0 0 920 330" role="img" aria-labelledby="kinematics-figure-title kinematics-figure-desc">
      <title id="kinematics-figure-title">Two link arm showing FK, IK, and Jacobian</title>
      <desc id="kinematics-figure-desc">A two-link arm maps joint angles theta one and theta two to an end-effector point. A local Jacobian maps small joint updates to small Cartesian updates.</desc>
      <defs>
        <marker id="kinematics-arrow" markerWidth="8" markerHeight="8" refX="6" refY="3" orient="auto"><path class="dof-diagram-arrow" d="M0,0 L7,3 L0,6 Z"/></marker>
        <marker id="kinematics-arrow-violet" markerWidth="8" markerHeight="8" refX="6" refY="3" orient="auto"><path class="dof-diagram-arrow-violet" d="M0,0 L7,3 L0,6 Z"/></marker>
      </defs>
      <path class="dof-diagram-dash" d="M78 252 H580"/>
      <circle class="dof-diagram-fill-blue" cx="120" cy="252" r="14"/><circle class="dof-diagram-fill-violet" cx="324" cy="118" r="12"/><circle class="dof-diagram-fill-good" cx="520" cy="204" r="13"/>
      <path class="dof-diagram-accent" d="M120 252 L324 118 L520 204"/>
      <path class="dof-diagram-violet" d="M170 250 A52 52 0 0 0 154 207" marker-end="url(#kinematics-arrow-violet)"/>
      <path class="dof-diagram-violet" d="M365 140 A48 48 0 0 1 403 148" marker-end="url(#kinematics-arrow-violet)"/>
      <text class="dof-diagram-label" x="94" y="282">base</text><text class="dof-diagram-label" x="296" y="96">elbow</text><text class="dof-diagram-label" x="530" y="207">end effector x(q)</text>
      <text class="dof-diagram-math" x="145" y="214">θ₁</text><text class="dof-diagram-math" x="368" y="122">θ₂</text>
      <text class="dof-diagram-note" x="198" y="168">link l₁</text><text class="dof-diagram-note" x="420" y="174">link l₂</text>
      <path class="dof-diagram-good" d="M526 196 C556 164, 588 151, 624 147" marker-end="url(#kinematics-arrow)"/>
      <text class="dof-diagram-math" x="575" y="135">Δx</text>
      <rect class="dof-diagram-surface" x="642" y="54" width="236" height="214" rx="18"/>
      <text class="dof-diagram-title" x="668" y="87">Same mechanism, three views</text>
      <text class="dof-diagram-label" x="668" y="122">FK</text><text class="dof-diagram-note" x="716" y="122">q → x(q)</text>
      <text class="dof-diagram-label" x="668" y="158">IK</text><text class="dof-diagram-note" x="716" y="158">x* → q  (may be multiple / none)</text>
      <text class="dof-diagram-label" x="668" y="194">Jacobian</text><text class="dof-diagram-math" x="668" y="219">Δx ≈ J(q) Δq</text>
      <text class="dof-diagram-note" x="668" y="244">local mapping; ill-conditioned near a singularity</text>
      <path class="dof-diagram-dash" d="M638 198 H554" marker-end="url(#kinematics-arrow)"/>
    </svg>
  </div>
</div>

### 1.1 几何法（2-DOF 平面臂）

最直观的例子：两段连杆长度 l₁、l₂，关节角 θ₁、θ₂。末端位置由三角函数直接给出：

```
x = l₁·cos(θ₁) + l₂·cos(θ₁+θ₂)
y = l₁·sin(θ₁) + l₂·sin(θ₁+θ₂)
```

> **直觉**：第一段先转到 θ₁，第二段在第一段末端基础上再转 θ₂。这正是 [`05-coordinate-transform.md`](05-coordinate-transform.md) 里"链式复合"的简化版。

### 1.2 DH 参数（标准方法）

对于任意串联机械臂，**Denavit-Hartenberg (DH) 参数**用 4 个数 `(a, α, d, θ)` 描述相邻连杆坐标系的关系，每个关节一个 4×4 齐次变换矩阵：

```
T_i = Rot_z(θ_i) · Trans_z(d_i) · Trans_x(a_i) · Rot_x(α_i)
```

末端位姿 = 所有连杆变换相乘：

```
T_0n = T_01 · T_12 · ... · T_(n-1)n
```

> DH 参数是机械臂建模的"通用语言"。工业机械臂（UR、Franka）的说明书都给 DH 表。

```python
import numpy as np

def dh_transform(theta, d, a, alpha):
    """单个 DH 参数对应的 4x4 齐次变换矩阵"""
    ct, st = np.cos(theta), np.sin(theta)
    ca, sa = np.cos(alpha), np.sin(alpha)
    return np.array([
        [ct,      -st*ca,   st*sa,  a*ct],
        [st,       ct*ca,  -ct*sa,  a*st],
        [0,         sa,      ca,     d  ],
        [0,         0,       0,      1  ],
    ])

def fk_dh(dh_params, joint_angles):
    """给定 DH 表和关节角，求末端 4x4 位姿"""
    T = np.eye(4)
    for (a, alpha, d), th in zip(dh_params, joint_angles):
        T = T @ dh_transform(th, d, a, alpha)
    return T

# 示例：2-DOF 平面臂（a=l1,l2; d=0; alpha=0）
dh = [(1.0, 0.0, 0.0), (1.0, 0.0, 0.0)]
print("末端位姿:\n", fk_dh(dh, [np.pi/4, np.pi/4]).round(3))
```

---

## 2. Jacobian 矩阵

### 2.1 定义

**Jacobian 矩阵 J** 描述"关节角微小变化 → 末端位姿微小变化"的线性映射：

```
ẋ = J(q) · q̇

 末端速度 = Jacobian × 关节速度
```

对于 2-DOF 平面臂，末端位置对关节角求偏导，得到 2×2 的 Jacobian：

```
J = [ ∂x/∂θ₁   ∂x/∂θ₂ ]   [ -l₁sinθ₁ - l₂sin(θ₁+θ₂)   -l₂sin(θ₁+θ₂) ]
    [ ∂y/∂θ₁   ∂y/∂θ₂ ] = [  l₁cosθ₁ + l₂cos(θ₁+θ₂)    l₂cos(θ₁+θ₂) ]
```

> **直觉**：Jacobian 的第 i 列 = "只有第 i 个关节动"时末端的速度方向。它告诉你"此刻哪个关节对末端最有效"。

### 2.2 几何 Jacobian vs 解析 Jacobian

| 类型 | 含义 | 适用 |
|:-----|:-----|:-----|
| **几何 Jacobian** | 直接对应物理线速度/角速度 | 动力学、奇异分析 |
| **解析 Jacobian** | 对位姿参数（如欧拉角）求偏导 | 数值 IK、优化 |

两者在纯位置 IK 时一致；涉及朝向时，几何 Jacobian 用角速度，解析 Jacobian 用欧拉角导数（需转换）。

```python
def jacobian_2dof(theta1, theta2, l1=1.0, l2=1.0):
    """2-DOF 平面臂的解析 Jacobian (2x2)"""
    s1, c1 = np.sin(theta1), np.cos(theta1)
    s12, c12 = np.sin(theta1+theta2), np.cos(theta1+theta2)
    return np.array([
        [-l1*s1 - l2*s12, -l2*s12],
        [ l1*c1 + l2*c12,  l2*c12],
    ])

J = jacobian_2dof(np.pi/4, np.pi/4)
print("Jacobian:\n", J.round(3))
```

---

## 3. Jacobian 的性质：奇异与可操作度

### 3.1 奇异 (Singularity)

当 Jacobian 行列式 `det(J) = 0` 时，机械臂处于**奇异位形**：某些方向上"无论怎么动关节，末端都动不了"。2-DOF 平面臂的奇异发生在两连杆完全伸直（θ₂=0）或完全折回（θ₂=π）时。

```python
# 奇异点：两连杆伸直 theta2=0
J_sing = jacobian_2dof(0.3, 0.0)
print("det(J) at 奇异点:", np.linalg.det(J_sing).round(6))   # ≈ 0

# 正常位形
J_norm = jacobian_2dof(0.3, 1.0)
print("det(J) at 正常点:", np.linalg.det(J_norm).round(6))   # ≠ 0
```

### 3.2 可操作度 (Manipulability)

可操作度 `w = √det(J·Jᵀ)` 衡量末端在当前位置"各方向都能灵活运动"的程度。w 越大越灵活，w=0 即奇异。它是机械臂轨迹规划避开奇异的重要指标。

```python
def manipulability(J):
    return np.sqrt(max(np.linalg.det(J @ J.T), 0.0))

print("奇异点可操作度:", manipulability(J_sing).round(4))   # 0
print("正常点可操作度:", manipulability(J_norm).round(4))   # >0
```

---

## 4. 逆运动学（IK）

**逆运动学 (Inverse Kinematics, IK)**：已知目标末端位姿，反求关节角。这是 FK 的反问题，**可能无解、多解或难解**。

```
目标末端位姿  ──IK──►  关节角 q = [θ₁, θ₂, ...]
```

| 方法 | 适用 | 特点 |
|:-----|:-----|:-----|
| **解析法 (Analytical)** | 自由度低、结构简单（如 2-DOF 臂） | 闭式解、极快、能得到所有解 |
| **数值法 (Numerical)** | 任意结构、高自由度 | 迭代逼近、只能得一个解、可能陷入局部最优 |

### 4.1 解析法（2-DOF 平面臂）

用余弦定理求 θ₂，再用几何关系求 θ₁，可得到"肘上""肘下"两个解（详见 `examples/fk_ik_demo.py` 的 `inverse_kinematics_analytical`）。

---

## 5. 数值 IK：伪逆与阻尼最小二乘

### 5.1 Jacobian 伪逆法

由 `ẋ = J·q̇`，反解 `q̇ = J⁺·ẋ`，其中 `J⁺ = Jᵀ(JJᵀ)⁻¹` 是伪逆 (Moore-Penrose pseudoinverse)。每次迭代：

```
Δq = J⁺ · e        (e = 目标位置 - 当前位置)
q ← q + Δq
```

> 问题：在奇异点附近 `JJᵀ` 接近奇异，伪逆会爆出极大步长，关节剧烈抖动。

### 5.2 阻尼最小二乘（DLS）

**Damped Least Squares (DLS)** 在 `JJᵀ` 上加一个阻尼项 `λ²I`，避免奇异点数值爆炸：

```
Δq = Jᵀ · (J·Jᵀ + λ²·I)⁻¹ · e
```

- λ 大 → 稳定但收敛慢
- λ 小 → 快但奇异点附近易抖

为防止步长过大，通常再裁剪 `Δq`（step limit）。这正是项目里 `finger_chain_3d.py` 的做法。

```python
def fk_2dof(theta, l1=1.0, l2=1.0):
    """2-DOF 平面臂正运动学，返回末端 (x, y)"""
    t1, t2 = theta
    x = l1*np.cos(t1) + l2*np.cos(t1+t2)
    y = l1*np.sin(t1) + l2*np.sin(t1+t2)
    return np.array([x, y])

def ik_dls_2dof(target, theta0=np.array([0.5, 0.5]),
                l1=1.0, l2=0.8, lam=0.1, step_limit=0.05,
                tol=1e-4, max_iter=50):
    """阻尼最小二乘数值 IK"""
    theta = theta0.copy().astype(float)
    for _ in range(max_iter):
        e = target - fk_2dof(theta, l1, l2)
        if np.linalg.norm(e) < tol:
            break
        J = jacobian_2dof(theta[0], theta[1], l1, l2)
        # DLS: Δθ = Jᵀ (J Jᵀ + λ²I)⁻¹ e
        JJt = J @ J.T
        dtheta = J.T @ np.linalg.solve(JJt + lam**2*np.eye(2), e)
        # 步长限制
        if np.linalg.norm(dtheta) > step_limit:
            dtheta *= step_limit / np.linalg.norm(dtheta)
        theta += dtheta
    return theta

sol = ik_dls_2dof(np.array([1.2, 0.5]))
print("IK 解:", sol.round(3), " 验证 FK:", fk_2dof(sol).round(3))
```

---

## 6. 连接项目

- **`examples/fk_ik_demo.py`**：实现了 2-DOF 平面臂的完整 FK + IK。其中 `forward_kinematics` 是几何法 FK，`inverse_kinematics_analytical` 用余弦定理给闭式解，`inverse_kinematics_numerical` 用阻尼最小二乘（damping=0.1）数值求解。运行 `python examples/fk_ik_demo.py --mode ik` 可对比两种解。
- **`examples/finger_chain_3d.py`**：3D 手指链的 DLS 逆运动学，参数为 `lambda_damp=0.06`（阻尼）、`step_limit`（步长上限）。这与手部重定向（retargeting）管线 `complete_retargeting_pipeline.py` 直接对接。
- **项目实时控制约定**：retargeting 在 **25 Hz** 下运行，每帧只迭代约 **5 次**，并施加 **step_limit≈0.025** 的小步长限制。这样在保证不抖动的同时满足实时性——这正是 DLS"小步快走"思想的工程体现。

> **工程启示**：实时系统里 IK 不追求一次收敛到完美解，而是"每帧走一小步、靠下一帧继续逼近"。阻尼 λ 和步长上限共同把数值 IK 驯化成稳定可控的伺服器。

---

## 检查理解

1. **概念题**：FK 和 IK 哪个一定有唯一解？为什么？IK 在什么情况下会无解、多解？
2. **Jacobian 题**：写出 2-DOF 平面臂 Jacobian 的表达式。当 θ₂=0 时 `det(J)` 等于多少？对应的物理姿态是什么？
3. **奇异题**：用代码画出 `det(J)` 随 θ₂（从 -π 到 π）变化的曲线，找出两个奇异点，并解释为什么伸直和折回都是奇异。
4. **DLS 题**：把上面 `ik_dls_2dof` 的阻尼 `lam` 分别设成 0.001、0.1、1.0，对同一目标 `(1.2, 0.5)` 求解，比较收敛步数与末端误差。哪个最稳？
5. **项目题**：阅读 `fk_ik_demo.py` 的 `inverse_kinematics_numerical`，指出它对应本文第 5.2 节哪几行公式。再把 `finger_chain_3d.py` 的 `inverse_kinematics_dls` 参数（`lambda_damp=0.06`）改成 0.6 和 0.006，观察收敛行为差异，并解释实时 25 Hz 场景下为什么要加 `step_limit`。

---

> **下一篇**: [`08-control-basics.md`](08-control-basics.md) —— 把 IK 算出的关节角变成让电机真正跟踪的指令：PID 与阻抗控制。
