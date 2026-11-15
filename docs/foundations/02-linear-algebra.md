# Linear Algebra

> **逐点图解 / Concept close-ups：**[线性代数与最小二乘](../knowledge-atlas/math-linear-algebra/index.md)。每个小点配原理、算例、图、自测；这是中文细解，保留英文术语。

> English contract: [Foundations overview](README_EN.md#route) · Primary references: [Linear algebra](../SOURCES.md#02-linear-algebra)

> **前置要求**: [01-python-for-robotics.md](01-python-for-robotics.md)（会 NumPy 基本操作）
> **预计学习时间**: 3–4 小时
> **完成后你能**: 用向量/矩阵描述机器人状态与变换，计算特征值并理解其几何含义，掌握概率分布/期望/方差，看懂项目中 14 维状态向量、神经网络权重矩阵与 Jacobian 的数学结构

---

## 目录

1. [为什么机器人离不开线性代数？](#1-为什么机器人离不开线性代数)
2. [向量：机器人状态的基本单位](#2-向量机器人状态的基本单位)
3. [矩阵：变换与权重](#3-矩阵变换与权重)
4. [特征值与特征向量](#4-特征值与特征向量)
5. [向量空间与线性变换](#5-向量空间与线性变换)
6. [概率基础：分布、期望、方差](#6-概率基础分布期望方差)
7. [连接项目代码](#7-连接项目代码)
8. [检查理解](#8-检查理解)

---

## 1. 为什么机器人离不开线性代数？

机器人学的几乎每一个公式都是线性代数：

```
状态表示    → 向量     (关节角 q ∈ ℝⁿ, 末端位姿 ∈ ℝ⁶)
坐标变换    → 矩阵乘法 (旋转 R、平移 t、齐次变换 T)
微分关系    → Jacobian  (∂x/∂q，一个矩阵)
神经网络    → 矩阵乘法 (y = Wx + b)
概率/统计   → 向量空间 (期望、协方差)
```

> **直觉**：把机器人想成一台"搬运数字的机器"——输入一个状态向量，经过若干矩阵相乘（变换、网络层），输出一个动作向量。线性代数就是描述这台机器的语言。

---

## 2. 向量：机器人状态的基本单位

**定义**：向量是一组有序的数 $\mathbf{v} = [v_1, v_2, \dots, v_n]^\top \in \mathbb{R}^n$。在机器人里，它常表示"一组同时存在的量"，比如 7 个关节角、3D 位置。

### 2.1 加法与减法

对应分量相加减，要求维度相同。几何上就是"首尾相接"的平行四边形法则。

```python
import numpy as np

pos    = np.array([1.0, 2.0])    # 当前位置
delta  = np.array([0.5, -0.3])   # 位移
new_pos = pos + delta            # [1.5, 1.7]
```

> **机器人含义**：`pos + delta` 正是 PushCube 环境里机械臂移动的写法——`new_arm = self.arm_pos + movement`。

### 2.2 点乘（内积）

$\mathbf{a} \cdot \mathbf{b} = \sum_i a_i b_i = \|\mathbf{a}\|\|\mathbf{b}\|\cos\theta$。

点乘衡量"两个向量有多同向"：结果为正表示夹角 < 90°，为 0 表示垂直。

```python
a = np.array([1.0, 0.0])
b = np.array([0.0, 1.0])
print(np.dot(a, b))           # 0  → 互相垂直

# 计算 a 在 b 方向上的投影长度
proj = np.dot(a, b) / np.linalg.norm(b)
```

> **机器人含义**：专家策略 `expert_action` 里用 `np.dot(arm_rel, behind_dir)` 判断机械臂是否在方块"后方"——点乘为正说明二者同向。

### 2.3 叉乘（外积，仅 3D）

$\mathbf{a} \times \mathbf{b}$ 得到一个同时垂直于 a、b 的向量，方向由右手定则决定，模长等于二者张成的平行四边形面积。

```python
a = np.array([1, 0, 0])
b = np.array([0, 1, 0])
print(np.cross(a, b))         # [0 0 1]  z 方向
```

> **机器人含义**：力矩 $\boldsymbol{\tau} = \mathbf{r} \times \mathbf{F}$；用叉乘求平面的法向量（碰撞检测、接触建模常用）。

### 2.4 范数（长度）

$\|\mathbf{v}\|_2 = \sqrt{\sum_i v_i^2}$，即欧氏距离。

```python
dist = np.linalg.norm(np.array([3.0, 4.0]))   # 5.0
```

> **机器人含义**：PushCube 的奖励 `reward = -dist`、成功判定 `dist < goal_threshold` 都靠范数算距离。

<div class="dof-principle" role="group" aria-label="点乘和向量投影的几何原理">
  <p class="dof-principle__caption"><strong>原理图 · Dot product is a projection</strong>：点乘不是逐项相乘的记忆题；它测量向量 <em>a</em> 在方向 <em>b</em> 上投影的大小。投影为正表示同向，为零表示垂直，为负表示反向。</p>
  <div class="dof-principle__canvas">
    <svg viewBox="0 0 860 260" role="img" aria-labelledby="linear-dot-title">
      <title id="linear-dot-title">点乘和向量投影的几何解释</title>
      <rect class="dof-diagram-surface" x="10" y="16" width="500" height="226" rx="18"/><text class="dof-diagram-title" x="34" y="49">Project a onto direction b</text><path class="dof-diagram-line" d="M66 204 H456"/><path class="dof-diagram-accent" d="M82 204 L390 116"/><path class="dof-diagram-arrow" d="M390 116 l-14 1 l7 10z"/><path class="dof-diagram-violet" d="M82 204 L438 204"/><path class="dof-diagram-arrow-violet" d="M438 204 l-12 -6 v12z"/><path class="dof-diagram-dash" d="M390 116 V204"/><circle class="dof-diagram-fill-good" cx="390" cy="204" r="5"/><text class="dof-diagram-label" x="318" y="104">a</text><text class="dof-diagram-label" x="432" y="224">b</text><text class="dof-diagram-note" x="293" y="226">projection of a on b</text><path class="dof-diagram-line" d="M144 204 A62 62 0 0 0 139 190"/><text class="dof-diagram-math" x="148" y="185">θ</text>
      <rect class="dof-diagram-fill-blue" x="552" y="43" width="270" height="63" rx="12"/><text class="dof-diagram-math" x="579" y="72">a · b = ||a|| ||b|| cos θ</text><text class="dof-diagram-note" x="579" y="91">signed alignment / projection</text><rect class="dof-diagram-fill-violet" x="552" y="124" width="270" height="82" rx="12"/><text class="dof-diagram-label" x="576" y="153">a · b &gt; 0   angle &lt; 90°</text><text class="dof-diagram-label" x="576" y="177">a · b = 0   orthogonal</text><text class="dof-diagram-label" x="576" y="201">a · b &lt; 0   angle &gt; 90°</text>
    </svg>
  </div>
</div>

---

## 3. 矩阵：变换与权重

**定义**：矩阵 $A \in \mathbb{R}^{m \times n}$ 是 m 行 n 列的数表。它有两种角色：① 一个线性变换（把向量从 ℝⁿ 映射到 ℝᵐ）；② 一组向量的集合（每列是一个向量）。

### 3.1 矩阵乘法

$C = AB$ 要求 A 的列数 = B 的行数。 $C_{ij} = \sum_k A_{ik} B_{kj}$。

```python
A = np.array([[1, 2], [3, 4]])
B = np.array([[5, 6], [7, 8]])
print(A @ B)                  # [[19 22], [43 50]]
```

> **机器人含义**：神经网络的一层就是 `y = W @ x + b`——权重矩阵 W 把输入向量 x 变换成特征向量。整章深度学习都建立在这个运算上。

### 3.2 转置

$A^\top$ 把行列互换。 $(AB)^\top = B^\top A^\top$。

```python
print(A.T)                    # [[1 3], [2 4]]
```

> **机器人含义**：Jacobian 伪逆用到 $J^\top$；`A.T @ A` 在最小二乘里反复出现。

### 3.3 逆矩阵

$A^{-1}$ 满足 $A A^{-1} = I$。只有方阵且行列式非零时才可逆。

```python
A_inv = np.linalg.inv(A)
print(A @ A_inv)              # ≈ 单位阵
```

> **机器人含义**：只有方阵 Jacobian 非奇异时，局部线性化关系才能写成 $\Delta q = J^{-1} \Delta x$；实际计算优先解线性方程，不显式构造逆。一般非方阵、奇异或接近奇异时，考虑最小二乘、伪逆或阻尼最小二乘，并检查残差、步长和关节约束（见第 7 课）。

### 3.4 行列式

$\det(A)$ 的绝对值是实方阵线性变换对体积的缩放因子，负号表示朝向翻转，不是“负体积”。 $\det = 0$ 表示矩阵"压扁"了空间（不可逆，存在信息丢失方向）。

```python
print(np.linalg.det(A))       # -2.0 (非零 → 可逆)
```

> **机器人含义**：Jacobian 在某位形的秩低于该机构能达到的最大秩，才称为运动学奇异。仅对最大可达秩等于维数的方阵 Jacobian，才可用行列式为零判别；非方阵没有普通行列式。丢失的是某些局部瞬时运动方向，不等于整个机械臂都不能移动。参见 [Modern Robotics：奇异性](https://modernrobotics.northwestern.edu/nu-gm-book-resource/5-3-singularities/)。

---

## 4. 特征值与特征向量

**定义**：对方阵 $A$，若存在标量 $\lambda$ 和非零向量 $\mathbf{v}$ 使 $A\mathbf{v} = \lambda\mathbf{v}$，则称 $\lambda$ 为特征值， $\mathbf{v}$ 为特征向量。

**直觉**：对实特征值及其非零实特征向量，矩阵作用后的结果仍在同一条过原点的直线上：正特征值保持朝向，负特征值翻转朝向，零特征值把该向量压成零。复特征值不能直接画成一个实平面中的“不变方向”。

```python
A = np.array([[4, -2], [1, 1]])
eigvals, eigvecs = np.linalg.eig(A)
print(eigvals)                # [3. 2.]
print(eigvecs)                # 每列是一个特征向量
# 验证: A @ v ≈ λ * v
print(A @ eigvecs[:, 0], eigvals[0] * eigvecs[:, 0])
```

> **机器人含义**：
> - **协方差矩阵的特征值** = 数据在各主方向上的方差，主成分分析（PCA）据此降维。
> - **惯性张量的特征值** = 刚体绕主轴的转动惯量（机械设计中决定稳定性）。
> - **动力系统矩阵的特征值**：对无输入的连续时间线性时不变系统，特征值实部全部严格为负对应原点渐近稳定；离散时间对应所有特征值模长严格小于 1。不能把连续时间判据直接套给采样更新矩阵，边界情形还需分析。参见 [MIT：连续与离散自然频率](https://introcontrol.mit.edu/fall24/prelabs/prelab3/est)。

---

## 5. 向量空间与线性变换

**向量空间**：一组向量的集合，对加法和数乘封闭。机器人里最常见的是 $\mathbb{R}^n$（所有 n 维实向量）。

**线性变换**：满足 $T(\mathbf{u}+\mathbf{v})=T(\mathbf{u})+T(\mathbf{v})$ 且 $T(c\mathbf{u})=c\,T(\mathbf{u})$ 的映射。任何有限维线性变换都可用矩阵表示。

```python
# 旋转 30° 的线性变换（2D 旋转矩阵）
theta = np.deg2rad(30)
R = np.array([[np.cos(theta), -np.sin(theta)],
              [np.sin(theta),  np.cos(theta)]])
v = np.array([1.0, 0.0])
v_rot = R @ v                  # 把 v 旋转 30°
print(v_rot)
```

> **机器人含义**：旋转矩阵 $R$、齐次变换矩阵 $T$ 都是线性变换（第 5、6 课详讲）。"基变换"在坐标系之间转换坐标时出现。神经网络每一层 `W @ x` 也是一个线性变换——把输入"扭"到更适合分类/回归的空间。

**秩（rank）**：矩阵列向量张成空间的维数，即"变换后保留的维度数"。秩亏意味着信息丢失，RL 中状态维度过高但实际有效维度低时，可用秩分析冗余。

---

## 6. 概率基础：分布、期望、方差

机器学习本质是在不确定性下做决策，概率是描述不确定性的语言。

### 6.1 随机变量与分布

**期望**： $E[X] = \sum_i x_i p_i$，即"长期平均"。**方差**： $\mathrm{Var}(X) = E[(X-E[X])^2]$，衡量波动大小。

```python
rng = np.random.RandomState(42)

# 均匀分布：PushCube 复位时方块位置用它采样
samples = rng.uniform(-0.4, 0.4, size=10000)
print("期望(理论0):", samples.mean())           # ≈ 0
print("方差(理论 (0.8²)/12≈0.0533):", samples.var())

# 正态分布：神经网络权重初始化、传感器噪声建模常用
noise = rng.normal(loc=0.0, scale=0.01, size=1000)
print("噪声期望:", noise.mean(), "标准差:", noise.std())
```

### 6.2 协方差矩阵

多变量情况下，协方差矩阵 $\Sigma$ 的对角元是各维方差，非对角元是两两协方差。

```python
data = rng.multivariate_normal(
    mean=[0, 0],
    cov=[[1.0, 0.8], [0.8, 1.0]],   # 两维强正相关
    size=2000,
)
print(np.cov(data.T))              # 接近 [[1, 0.8],[0.8, 1]]
```

> **机器人含义**：
> - **环境随机性**：PushCube 每次复位用 `rng.uniform` 随机化方块位置，正是均匀分布的工程实例。
> - **策略随机性**：RL 策略 $\pi(a|s)$ 是动作的概率分布，训练时按分布采样动作。
> - **不确定性估计**：World Model 预测下一状态时输出均值+方差，方差大表示预测不可靠。
> - 协方差矩阵的特征向量就是数据主方向（联系第 4 节 PCA）。

---

## 7. 连接项目代码

把本课概念映射到项目实际代码：

| 线性代数概念 | 在项目中的体现 |
|------------|--------------|
| **14 维向量** | `unified_pushcube_env.py` 的 `get_state_vector()` 拼出 $\mathbb{R}^{14}$ 状态 |
| **向量加法** | `step()` 里 `new_arm = self.arm_pos + movement` |
| **点乘** | `expert_action()` 用 `np.dot(arm_rel, behind_dir)` 判断机械臂相对方块的位置 |
| **范数（距离）** | `np.linalg.norm(active_cube - self.target_pos)` 算奖励与成功 |
| **矩阵乘法** | 神经网络层 `y = W @ x + b`（VLA/RL/WM 全用到） |
| **矩阵求逆** | `fk_ik_demo.py` 数值 IK 用 `np.linalg.inv(J @ J.T + λ²I)` |
| **Jacobian 矩阵** | `_jacobian()` 返回 $\partial \text{末端}/\partial \text{关节}$，把关节速度映射到末端速度 |

**Jacobian 是什么？** 它是一个矩阵 $J$，满足 $\dot{\mathbf{x}} = J\,\dot{\mathbf{q}}$——末端速度 = Jacobian × 关节速度。它把"关节空间"线性映射到"任务空间"，是连接第 3 节矩阵与第 5 节线性变换的完美例子。

```python
# 节选自 fk_ik_demo.py 的 _jacobian（2-DOF 臂）
def _jacobian(self, theta1, theta2):
    J = np.array([
        [-self.l1*np.sin(theta1) - self.l2*np.sin(theta1+theta2),
         -self.l2*np.sin(theta1+theta2)],
        [ self.l1*np.cos(theta1) + self.l2*np.cos(theta1+theta2),
          self.l2*np.cos(theta1+theta2)]
    ])
    return J       # shape (2,2)：2 个末端维 × 2 个关节
```

> **思考**：阻尼最小二乘 $\Delta q = J^\top (J J^\top + \lambda^2 I)^{-1} \Delta x$ 里同时出现了转置 $J^\top$、矩阵乘法 $JJ^\top$、求逆 $(\cdot)^{-1}$、单位阵 $I$——本课的每一个矩阵运算都在这一个公式里登场。

---

## 8. 检查理解

**练习 1（向量运算）**：给定机械臂位置 $\mathbf{p}=[0.2, -0.1]$、目标 $\mathbf{g}=[0.4, 0.3]$，用 NumPy 求：(a) 指向目标的单位向量；(b) 当前距离；(c) 若以单位向量×0.08 为步长移动一步后的新位置。

**练习 2（点乘直觉）**：解释为什么 `np.dot(arm_rel, behind_dir) > 0` 能判断机械臂在方块"后方"。如果改成 `< 0` 代表什么几何关系？

**练习 3（矩阵与逆）**：构造 $A = [[2,1],[1,3]]$，用代码验证 $A A^{-1} = I$，并计算 $\det(A)$，说明为什么它一定可逆。

**练习 4（特征值）**：对协方差矩阵 $\Sigma = [[1, 0.8],[0.8, 1]]$ 求特征值，解释两个特征值分别代表数据在哪个方向上方差最大/最小（提示：画散点图观察）。

**练习 5（Jacobian）**：在 `fk_ik_demo.py` 的 `_jacobian` 中， $J$ 是 2×2 矩阵。若 $\dot{q}=[0.1, 0.2]$，用 `J @ dq` 算出末端速度，并说明它的两个分量分别对应 x、y 方向。

**练习 6（概率）**：PushCube 用 `rng.uniform(-0.4, 0.4)` 采样方块位置。写出该均匀分布的期望和方差的理论值，并用 10000 次采样验证。

> 完成 6 道题中的 5 道，即可进入 [03-deep-learning-basics.md](03-deep-learning-basics.md)。
