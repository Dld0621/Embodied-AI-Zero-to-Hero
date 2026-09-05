# Python for Robotics

> **逐点图解 / Concept close-ups：**[Python、NumPy 与张量形状](../knowledge-atlas/computing-python-numpy/index.md)。每个小点配原理、算例、图、自测；这是中文细解，保留英文术语。

> English contract: [Foundations overview](README_EN.md#route) · Primary references: [Python and numerical computing](../SOURCES.md#01-python-and-numerical-computing)

> **前置要求**: 无（本课是整个 Foundations Layer 的起点）
> **预计学习时间**: 2–3 小时
> **完成后你能**: 用 NumPy 表示机器人状态向量与关节角，用 Matplotlib 画出 2D/3D 机械臂位姿，看懂项目里 `examples/fk_ik_demo.py` 与 `examples/unified_pushcube_env.py` 的核心代码结构

---

## 目录

1. [为什么机器人开发用 Python？](#1-为什么机器人开发用-python)
2. [Python 基础速览：变量、函数、类](#2-python-基础速览变量函数类)
3. [NumPy：机器人计算的基石](#3-numpy机器人计算的基石)
4. [Matplotlib：机器人可视化](#4-matplotlib机器人可视化)
5. [实战：用 NumPy + Matplotlib 画出 2D 机械臂](#5-实战用-numpy--matplotlib-画出-2d-机械臂)
6. [连接项目代码](#6-连接项目代码)
7. [检查理解](#7-检查理解)

---

## 1. 为什么机器人开发用 Python？

机器人学涉及大量线性代数运算（矩阵乘法、求逆）和数值优化。C/C++ 常用于延迟敏感、资源受限或需要确定性控制的组件；Python 借助 NumPy/SciPy 可把大量数值运算交给经过优化的底层实现，同时拥有成熟的可视化与机器学习生态。具体性能取决于算法、实现、编译器、硬件和数据规模，不能只按语言名称排序。本仓库的示例代码（FK/IK、PushCube 环境、VLA 训练）以 Python + NumPy 为基础。

```
机器人代码栈
├─ 算法层（FK/IK、策略训练）   ← Python + NumPy / PyTorch   ← 本课重点
├─ 仿真层（MuJoCo、Isaac）     ← Python 调用底层 C/C++ 引擎
└─ 控制层（实时伺服、运动控制） ← C++ / ROS（追求实时性）
```

> **直觉**：训练策略、调试算法时用 Python 快速迭代；真正部署到机器人上追求毫秒级响应时再换 C++。本课程聚焦算法层。

<div class="dof-principle" role="group" aria-label="机器人状态在 Python 和 NumPy 中的表示与处理">
  <p class="dof-principle__caption"><strong>原理图 · State as an array</strong>：机器人程序不是“处理一堆变量”，而是在保持 <code>shape</code>、单位和顺序的前提下，把状态向量送入数值运算与可视化。切片决定了每一段数值代表哪个物理量。</p>
  <div class="dof-principle__canvas">
    <svg viewBox="0 0 860 248" role="img" aria-labelledby="python-state-title">
      <title id="python-state-title">机器人状态向量经过切片、矩阵计算和可视化的过程</title>
      <rect class="dof-diagram-surface" x="10" y="18" width="250" height="204" rx="18"/><text class="dof-diagram-title" x="32" y="51">Robot state · q / state</text><text class="dof-diagram-note" x="32" y="72">one ordered NumPy array</text>
      <g transform="translate(32 94)"><rect class="dof-diagram-fill-blue" x="0" y="0" width="30" height="42" rx="6"/><rect class="dof-diagram-fill-blue" x="34" y="0" width="30" height="42" rx="6"/><rect class="dof-diagram-fill-violet" x="68" y="0" width="30" height="42" rx="6"/><rect class="dof-diagram-fill-violet" x="102" y="0" width="30" height="42" rx="6"/><rect class="dof-diagram-fill-good" x="136" y="0" width="30" height="42" rx="6"/><rect class="dof-diagram-fill-good" x="170" y="0" width="30" height="42" rx="6"/><text class="dof-diagram-math" x="4" y="27">q₁</text><text class="dof-diagram-math" x="38" y="27">q₂</text><text class="dof-diagram-math" x="72" y="27">x</text><text class="dof-diagram-math" x="106" y="27">y</text><text class="dof-diagram-math" x="140" y="27">vₓ</text><text class="dof-diagram-math" x="174" y="27">vᵧ</text></g>
      <text class="dof-diagram-label" x="32" y="171">state[0:2] → joints</text><text class="dof-diagram-label" x="32" y="197">state[2:4] → object pose</text><path class="dof-diagram-accent" d="M279 121 H340"/><path class="dof-diagram-arrow" d="M340 121 l-10 -6 v12z"/>
      <rect class="dof-diagram-surface" x="358" y="18" width="205" height="204" rx="18"/><text class="dof-diagram-title" x="382" y="51">Vectorized math</text><text class="dof-diagram-note" x="382" y="72">the code follows the equation</text><rect class="dof-diagram-fill-violet" x="385" y="99" width="148" height="40" rx="10"/><text class="dof-diagram-math" x="410" y="125">x_next = A @ x</text><rect class="dof-diagram-fill-blue" x="385" y="153" width="148" height="40" rx="10"/><text class="dof-diagram-math" x="407" y="179">τ = r × F</text><path class="dof-diagram-accent" d="M582 121 H642"/><path class="dof-diagram-arrow" d="M642 121 l-10 -6 v12z"/>
      <rect class="dof-diagram-surface" x="660" y="18" width="190" height="204" rx="18"/><text class="dof-diagram-title" x="685" y="51">Inspect visually</text><text class="dof-diagram-note" x="685" y="72">plot the state or pose</text><path class="dof-diagram-line" d="M691 187 V99 H821"/><path class="dof-diagram-accent" d="M700 164 C727 117 752 183 780 133 S818 115 830 91"/><circle class="dof-diagram-fill-good" cx="780" cy="133" r="5"/><text class="dof-diagram-label" x="693" y="208">trajectory / workspace</text>
    </svg>
  </div>
</div>

---

## 2. Python 基础速览：变量、函数、类

如果你已经会 Python，可跳到第 3 节。这里只讲机器人代码里最常用的部分。

**变量与类型**：机器人状态用浮点数，关节角常用弧度（radian）。

```python
joint_angle = 1.57          # float，弧度（约 90°）
num_joints = 7              # int，自由度
is_grasping = False         # bool，是否抓取
```

**函数**：把可复用的计算封装起来。机器人代码里大量用 `numpy` 数组作为输入输出。

```python
import numpy as np

def degree_to_radian(deg: float) -> float:
    """角度转弧度"""
    return deg * np.pi / 180.0

print(degree_to_radian(90))   # 1.5707963...
```

**类**：机器人天然适合面向对象——一个机械臂是一个对象，有连杆长度等属性、有正运动学等方法。项目里的 `PlanarArm2D` 就是典型例子。

```python
class Joint:
    def __init__(self, name: str, angle: float = 0.0):
        self.name, self.angle = name, angle   # 关节名、当前角（弧度）

    def move(self, delta: float):
        self.angle += delta

j = Joint("shoulder", 0.5); j.move(0.1)
print(j.angle)   # 0.6
```

> **要点**：后续你会看到 `class PlanarArm2D`、`class PushCubeEnv`——属性存状态、方法存行为，正是机器人代码的标准组织方式。

---

## 3. NumPy：机器人计算的基石

机器人状态几乎都是数组：关节角向量、末端位置、图像像素。NumPy 用 C 实现底层运算，比纯 Python 循环快几十倍。

### 3.1 创建数组

```python
import numpy as np

q = np.array([0.0, 0.5, -0.3, 1.2, 0.0, 0.8, 0.0])  # 7 自由度关节角
print(q.shape)          # (7,)

pos = np.zeros(3)       # 末端位置 [0. 0. 0.]
I = np.eye(3)           # 3x3 单位阵（初始化变换矩阵常用）
```

### 3.2 索引与切片

```python
state = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6])  # 某机器人状态

arm_xy   = state[0:2]     # [0.1, 0.2]  机械臂位置
cube_pos = state[2:4]     # [0.3, 0.4]  方块位置
last     = state[-1]      # 0.6         最后一维

print(state[state > 0.3]) # [0.4 0.5 0.6]  布尔索引
```

> **直觉**：切片 `[a:b]` 取 a 到 b-1 的元素。机器人状态向量里不同区段代表不同物体，靠切片来"拆解"状态。

### 3.3 矩阵运算：dot、cross、transpose、inverse

这是机器人学里出现频率最高的一组运算。

```python
A = np.array([[1, 2], [3, 4]])
b = np.array([1.0, 0.0])

# 矩阵乘向量  y = A @ b
y = A @ b                      # 或 np.dot(A, b)
print(y)                       # [1. 3.]

# 矩阵乘矩阵
C = A @ np.eye(2)              # 乘单位阵不变

# 转置
print(A.T)                     # [[1 3], [2 4]]

# 求逆
A_inv = np.linalg.inv(A)
print(A @ A_inv)               # ≈ 单位阵（有浮点误差）

# 叉乘（3D 向量专属，常算力矩 / 法向量）
v1 = np.array([1, 0, 0])
v2 = np.array([0, 1, 0])
print(np.cross(v1, v2))        # [0 0 1]  即 z 方向

# 范数（算距离）
dist = np.linalg.norm(np.array([3.0, 4.0]))   # 5.0
```

> **机器人含义**：
> - `A @ b` = 把向量 b 用矩阵 A 变换（如旋转、缩放）
> - `np.linalg.inv` = 求逆变换，逆运动学里频繁出现
> - `np.cross` = 算力矩 τ = r × F，或求平面法向量
> - `np.linalg.norm` = 算末端到目标的距离，决定奖励

---

## 4. Matplotlib：机器人可视化

"画出来说明你真的理解了。"可视化在调试运动学、展示策略效果时不可或缺。

### 4.1 2D 绘图：画一个点轨迹

```python
import matplotlib.pyplot as plt

t = np.linspace(0, 2 * np.pi, 100)
x = np.cos(t)
y = np.sin(t)

plt.figure(figsize=(4, 4))
plt.plot(x, y, 'b-')
plt.axis('equal')
plt.grid(True)
plt.title('End-effector circle')
plt.show()
```

### 4.2 3D 绘图：画末端工作空间

```python
fig = plt.figure(figsize=(5, 5))
ax = fig.add_subplot(111, projection='3d')

# 在球面上采样点（np.outer 做外积，生成网格）
u, v = np.linspace(0, 2*np.pi, 50), np.linspace(0, np.pi, 50)
X = np.outer(np.cos(u), np.sin(v))
Y = np.outer(np.sin(u), np.sin(v))
Z = np.outer(np.ones_like(u), np.cos(v))

ax.plot_surface(X, Y, Z, alpha=0.3)
ax.set_xlabel('x'); ax.set_ylabel('y'); ax.set_zlabel('z')
plt.title('Reachable workspace (sphere)')
plt.show()
```

> **要点**：`plot` 画连杆/轨迹（2D），`plot_surface`/`scatter` 画工作空间/点云（3D）。

---

## 5. 实战：用 NumPy + Matplotlib 画出 2D 机械臂

下面这个例子浓缩了项目 `examples/fk_ik_demo.py` 的核心思想：给定关节角，用三角函数算出每个关节的位置（正运动学），再用 matplotlib 把连杆画出来。

```python
import numpy as np
import matplotlib.pyplot as plt


def forward_kinematics_2d(thetas, lengths):
    """
    N-连杆平面机械臂正运动学。

    参数:
        thetas:  各关节角（弧度），shape (n,)
        lengths: 各连杆长度，       shape (n,)
    返回:
        joints:  各关节点坐标，shape (n+1, 2)，第 0 个是基座原点
    """
    joints = [np.zeros(2)]           # 基座在原点
    cum_angle = 0.0
    for th, L in zip(thetas, lengths):
        cum_angle += th              # 累积角度（关节角是相对的）
        prev = joints[-1]
        nxt = prev + L * np.array([np.cos(cum_angle), np.sin(cum_angle)])
        joints.append(nxt)
    return np.array(joints)


def draw_arm(thetas, lengths):
    joints = forward_kinematics_2d(thetas, lengths)
    xs, ys = joints[:, 0], joints[:, 1]

    plt.figure(figsize=(5, 5))
    plt.plot(xs, ys, 'o-', linewidth=3, markersize=8)   # 连杆 + 关节
    plt.plot(xs[-1], ys[-1], 'r*', markersize=15)        # 末端执行器
    plt.axis('equal'); plt.grid(True)
    plt.xlim(-3, 3); plt.ylim(-3, 3)
    plt.title(f'2D arm  joints={np.round(thetas,2)}')
    plt.show()


# 3-连杆机械臂
draw_arm(thetas=[np.pi/4, -np.pi/3, np.pi/6],
         lengths=[1.0, 0.8, 0.6])
```

运行后会看到一条三段折线，红色星号是末端位置。试着改 `thetas` 观察末端如何移动——这正是"正运动学：关节角 → 末端位置"。

> **与项目的对应**：`forward_kinematics_2d` 对应 `examples/fk_ik_demo.py` 中 `PlanarArm2D.forward_kinematics`；累积角度 `cum_angle += th` 对应原代码里 `theta1 + theta2` 的累加。

---

## 6. 连接项目代码

学完本课，你应该能读懂项目里这两个文件的骨架。

### 6.1 `examples/fk_ik_demo.py`：NumPy 做正运动学，Matplotlib 做动画

`PlanarArm2D.forward_kinematics` 用 `np.cos / np.sin` 算关节坐标（与第 5 节的 `forward_kinematics_2d` 同理）；数值逆运动学 `inverse_kinematics_numerical` 用 `np.linalg.inv` 求逆、`J.T @ ... @ error` 做 Jacobian 迭代、`np.linalg.norm` 判断收敛（第 7 课详讲）；`matplotlib.animation.FuncAnimation` 把 IK 求解做成动画。

### 6.2 `examples/unified_pushcube_env.py`：状态是 14 维 NumPy 数组

环境把所有信息压成一个 14 维向量，供 VLA / RL / World Model 统一使用：

```python
# 节选自 unified_pushcube_env.py
def get_state_vector(self) -> np.ndarray:
    return np.concatenate([
        self.arm_pos,                  # 2  机械臂 xy
        self.cube_positions[0],        # 2  方块1 xy
        self.cube_positions[1],        # 2  方块2 xy
        self.target_pos,               # 2  目标 xy
        self.cube_colors[0],           # 2  方块1 (R,G)
        self.cube_colors[1],           # 2  方块2 (R,G)
        self.get_goal_color_onehot(),  # 2  要推哪个颜色
    ]).astype(np.float32)              # 共 14 维
```

| 区段 | 维度 | 含义 |
|------|------|------|
| `arm_pos` | 0–1 | 机械臂位置 |
| `cube_positions` | 2–5 | 两个方块位置 |
| `target_pos` | 6–7 | 目标区位置 |
| `cube_colors` | 8–11 | 两个方块颜色 (R,G) |
| `goal_onehot` | 12–13 | 语言指代的目标颜色 |

> **思考**：`np.concatenate` 把 7 段小向量拼成一段——这正是第 2 课要讲的"向量"在工程中的直接体现。颜色和位置混在同一个向量里喂给神经网络，网络要学会"哪几维是位置、哪几维是颜色"。

---

## 7. 检查理解

完成下面的练习来检验学习效果，答案可在运行代码后自行核对。

**练习 1（数组与切片）**：给定 PushCube 的 14 维状态向量 `s`，写出表达式分别取出：机械臂位置、两个方块的位置、目标颜色 one-hot。

**练习 2（矩阵运算）**：构造一个 2×2 矩阵 `M = [[2,0],[0,3]]`，验证 `M @ M_inv` 约等于单位阵，并解释 `M` 对向量的作用（提示：分别缩放 x、y）。

**练习 3（叉乘直觉）**：计算 `np.cross([1,0,0], [0,0,1])`，画图说明为什么结果指向 -y 方向。

**练习 4（动手可视化）**：把第 5 节的 `draw_arm` 改成 2 连杆，让末端画出一个圆轨迹（提示：在循环里让 `theta1` 从 0 扫到 2π，记录末端坐标后用 `plt.plot` 画出来）。

**练习 5（连接项目）**：打开 `examples/fk_ik_demo.py`，找到 `inverse_kinematics_numerical` 方法，指出其中哪一行用到了本课讲的：(a) 矩阵求逆 `np.linalg.inv`；(b) 矩阵乘法 `@`；(c) 范数 `np.linalg.norm`。并解释阻尼最小二乘公式 `J.T @ inv(J@J.T + λ²I) @ error` 中每一项的含义。

> 完成 5 道题中的 4 道，即可进入 [02-linear-algebra.md](02-linear-algebra.md)。
