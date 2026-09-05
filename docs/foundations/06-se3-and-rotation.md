# SO(3) & SE(3) 旋转与刚体变换

> **逐点图解 / Concept close-ups：**[SO(3)、SE(3) 与旋转表示](../knowledge-atlas/robot-so3-se3/index.md)。每个小点配原理、算例、图、自测；这是中文细解，保留英文术语。

> English contract: [Foundations overview](README_EN.md#route) · Primary references: [SO(3) and SE(3)](../SOURCES.md#06-so3-and-se3)

> **前置要求**: [`05-coordinate-transform.md`](05-coordinate-transform.md)（齐次坐标、3D 变换矩阵）
> **预计学习时间**: 2–3 小时
> **完成后你能**: 在欧拉角、旋转矩阵、轴角、四元数之间互相转换；解释 SO(3) 与 SE(3) 的群性质；理解万向锁成因，以及 MuJoCo 在哪些状态变量中使用四元数。

---

## 1. 旋转表示总览

同一个三维旋转，有四种常见"语言"描述它。它们描述的是同一件事，但各有优劣：

| 表示法 | 自由度 | 参数数 | 主要优点 | 主要缺点 |
|:-------|:------:|:------:|:---------|:---------|
| **欧拉角 (RPY)** | 3 | 3 | 直观、人好读 | 万向锁、顺序依赖 |
| **旋转矩阵** | 3 | 9 | 直接做坐标变换、无歧义 | 9 个数冗余、需正交化 |
| **轴角 (Axis-Angle)** | 3 | 4 | 直观（绕某轴转某角） | 0°/180° 附近不唯一 |
| **四元数 (Quaternion)** | 3 | 4 | 无万向锁、插值平滑、紧凑 | 不直观、需归一化 |

> **自由度 vs 参数数**：3D 旋转只有 3 个自由度，但旋转矩阵有 9 个数（多出 6 个约束），四元数有 4 个数（多出 1 个约束 ‖q‖=1）。参数数 > 自由度 意味着有冗余约束。

---

## 2. SO(3)：特殊正交群

**SO(3)**（Special Orthogonal group）就是所有 3D 旋转矩阵的集合：

```
SO(3) = { R ∈ ℝ³ˣ³ | RᵀR = I,  det(R) = +1 }
```

它有两个约束条件，对应物理含义：

1. **RᵀR = I（正交）**：旋转不改变长度也不改变角度。每一列都是单位向量、两两垂直。
2. **det(R) = +1（特殊）**：排除掉镜像反射（det = -1）。镜像不是"旋转能到"的。

<div class="dof-principle" role="group" aria-label="SO3 旋转和 SE3 刚体位姿的几何区别">
  <p class="dof-principle__caption"><strong>原理图 · Rotation versus pose</strong>：<code>SO(3)</code> 只改变方向，原点不动；<code>SE(3)</code> 在旋转 <code>R</code> 之外再加入平移 <code>t</code>。机器人末端“位姿”必须同时包含二者。</p>
  <div class="dof-principle__canvas">
    <svg viewBox="0 0 860 260" role="img" aria-labelledby="se3-title">
      <title id="se3-title">SO3 旋转和 SE3 刚体变换的区别</title><rect class="dof-diagram-surface" x="12" y="16" width="364" height="225" rx="18"/><text class="dof-diagram-title" x="37" y="49">SO(3) · same origin</text><path class="dof-diagram-line" d="M95 191 V92 M57 161 H189"/><path class="dof-diagram-accent" d="M95 161 L163 112"/><path class="dof-diagram-arrow" d="M163 112 l-12 2 l7 9z"/><path class="dof-diagram-violet" d="M95 161 L135 86"/><path class="dof-diagram-arrow-violet" d="M135 86 l-12 5 l10 7z"/><path class="dof-diagram-dash" d="M95 161 A72 72 0 0 1 164 112"/><text class="dof-diagram-math" x="183" y="132">R p</text><text class="dof-diagram-math" x="140" y="83">p</text><text class="dof-diagram-note" x="37" y="216">RᵀR = I, det(R) = 1</text><path class="dof-diagram-accent" d="M396 129 H459"/><path class="dof-diagram-arrow" d="M459 129 l-11 -6 v12z"/>
      <rect class="dof-diagram-surface" x="480" y="16" width="368" height="225" rx="18"/><text class="dof-diagram-title" x="505" y="49">SE(3) · rotate + translate</text><path class="dof-diagram-line" d="M545 191 V92 M507 161 H642"/><path class="dof-diagram-accent" d="M545 161 L614 112"/><path class="dof-diagram-arrow" d="M614 112 l-12 2 l7 9z"/><path class="dof-diagram-line" d="M684 191 V92 M646 161 H781"/><path class="dof-diagram-violet" d="M684 161 L753 112"/><path class="dof-diagram-arrow-violet" d="M753 112 l-12 2 l7 9z"/><path class="dof-diagram-dash" d="M616 113 C654 83 678 83 724 109"/><text class="dof-diagram-math" x="631" y="81">+ t</text><rect class="dof-diagram-fill-blue" x="573" y="201" width="202" height="25" rx="7"/><text class="dof-diagram-math" x="600" y="219">p′ = R p + t</text>
    </svg>
  </div>
</div>

### 旋转矩阵的性质

| 性质 | 公式 | 物理含义 |
|:-----|:-----|:---------|
| 逆 = 转置 | R⁻¹ = Rᵀ | 反向旋转 |
| 保长度 | ‖Rp‖ = ‖p‖ | 刚体转动不变形 |
| 保角度 | (Rp)·(Rq) = p·q | 刚体转动不改变夹角 |
| 行列式为 1 | det(R)=1 | 纯旋转，无翻转 |
| 闭合 | R₁R₂ ∈ SO(3) | 两个旋转复合仍是旋转 |

> **直觉**：SO(3) 是一个"群"——你可以把任意两个旋转相乘得到另一个合法旋转，也能求逆。这正是机器人连杆旋转能链式复合的数学基础。

---

## 3. SE(3)：特殊欧氏群

**SE(3)**（Special Euclidean group）= 旋转 + 平移，即上一节学的 4×4 齐次变换矩阵的集合：

```
SE(3) = { T = [R t; 0 1] | R ∈ SO(3), t ∈ ℝ³ }
```

SE(3) 描述**刚体变换 (rigid body transformation)**：既旋转又平移，但不发生形变。机械臂末端位姿、相机位姿、物体 6D 位姿都属于 SE(3)。

```
SE(3) 的维数：6 (= 3 旋转 + 3 平移)
```

> **记忆**：SO(3) 管转（3 自由度），SE(3) 管转 + 移（6 自由度）。机器人里常说的"6D 位姿 (6-DoF pose)"指的就是 SE(3) 中的一个元素。

---

## 4. 四元数基础

### 4.1 为什么用四元数？

欧拉角在 pitch = ±90° 时会丢失一个自由度（万向锁，见第 5 节），而旋转矩阵有 9 个冗余数。四元数用 4 个数 `q = [w, x, y, z]` 描述旋转，**无万向锁、数值稳定、适合插值**，是仿真器和游戏引擎的标配。

### 4.2 四元数的定义

单位四元数 `q = [w, x, y, z]`，满足 `w² + x² + y² + z² = 1`，对应"绕单位轴 `û = (ux, uy, uz)` 转 θ 角"：

```
q = [cos(θ/2),  ux·sin(θ/2),  uy·sin(θ/2),  uz·sin(θ/2)]
     ↑w           ↑x             ↑y             ↑z
```

> **直觉**：四元数把"轴 + 角"打包成 4 个数。θ/2 是因为四元数在"转两圈才回起点"——这是它的数学结构决定的，记住即可。

### 4.3 Hamilton 乘积

两个四元数相乘（Hamilton product）等价于两次旋转的复合。顺序与矩阵一样不可交换：

```
q = q₁ ⊗ q₂   (先 q₂ 后 q₁)
```

### 4.4 四元数 → 旋转矩阵

```python
import numpy as np
from scipy.spatial.transform import Rotation as R

# ---- 四元数 <-> 旋转矩阵 (用 scipy 标准化) ----
# scipy 用 [x, y, z, w] 顺序，注意与 [w,x,y,z] 区分
quat_xyzw = [0, 0, np.sin(np.pi/4), np.cos(np.pi/4)]   # 绕 z 轴转 90°
rot = R.from_quat(quat_xyzw)

R_mat = rot.as_matrix()           # 四元数 → 旋转矩阵
print("旋转矩阵:\n", R_mat)

euler = rot.as_euler('xyz', degrees=True)   # → 欧拉角
print("欧拉角 (xyz):", euler)               # [0, 0, 90]
```

```python
def quat_to_rotmat(w, x, y, z):
    """四元数 [w,x,y,z] → 3x3 旋转矩阵（手写版，便于理解结构）"""
    return np.array([
        [1-2*(y*y+z*z), 2*(x*y-w*z),   2*(x*z+w*y)],
        [2*(x*y+w*z),   1-2*(x*x+z*z), 2*(y*z-w*x)],
        [2*(x*z-w*y),   2*(y*z+w*x),   1-2*(x*x+y*y)],
    ])

q = [np.cos(np.pi/4), 0, 0, np.sin(np.pi/4)]   # [w,x,y,z], 绕 z 转 90°
print(quat_to_rotmat(*q))
```

---

## 5. 万向锁（Gimbal Lock）

### 5.1 成因

欧拉角按固定顺序分解旋转（如 ZYX = yaw→pitch→roll）。当中间角 pitch = ±90° 时，第一和第三个旋转轴重合，丢失一个自由度——这就是万向锁。

```
正常情况：三个轴各管一个方向      万向锁 (pitch=90°)：首尾两轴重合
   z                                z
   ↑                                ↑
   │  y                              \  y=x (重合!)
   │ /                                \ /
   ●───→ x                            ●───→
```

> **直觉**：飞机机头上仰 90° 后，再"偏航"和再"滚转"效果一样——你分不清是动哪个轴了。

### 5.2 四元数如何解决

单位四元数不需要用三个依次旋转的角来描述朝向，因此避免了欧拉角在特定姿态下的参数化奇异性。**这不表示旋转复合与顺序无关**：四元数乘法和旋转矩阵乘法一样，通常不可交换。MuJoCo 用四元数表示 ball / free joint 的三维朝向，同时也使用旋转矩阵和标量关节坐标，不是所有状态都统一存成四元数。

```python
# 演示万向锁：pitch=90° 时，yaw 与 roll 耦合，不同组合得到同一旋转
rot_lock = R.from_euler('ZYX', [30, 90, 10], degrees=True)     # yaw=30,pitch=90,roll=10
rot_same = R.from_euler('ZYX', [40, 90, 20], degrees=True)     # yaw=40,pitch=90,roll=20
print("两个矩阵几乎相同:", np.allclose(rot_lock.as_matrix(),
                                     rot_same.as_matrix()))    # True → 自由度丢失
```

---

## 6. 指数映射与对数映射（简介）

旋转还可以用"旋转矢量"ω（3 维，方向=转轴、模长=转角）表示。它与旋转矩阵之间通过指数/对数映射互转：

```
旋转矩阵:   R = exp(ω̂)        （指数映射，ω̂ 是 ω 的反对称矩阵）
矩阵对数:   ω̂ = log(R)        （结果是 3×3 反对称矩阵）
旋转矢量:   ω = vee(log(R))   （vee 将该矩阵还原为 3 维向量）
```

so(3) 是 3×3 反对称矩阵的空间；hat / vee 负责它与 3 维向量之间的转换。选择旋转对数分支后，上述关系才是一致的；旋转角为 π 时要特别处理轴的符号歧义。**指数/对数映射加上 hat / vee 是“旋转矢量 ↔ 旋转矩阵”的桥梁**。见 [Modern Robotics：指数坐标](https://modernrobotics.northwestern.edu/nu-gm-book-resource/3-2-3-exponential-coordinates-of-rotation-part-2-of-2/)。`scipy` 提供 `as_rotvec()` 直接返回向量：

```python
rot = R.from_euler('z', 90, degrees=True)
omega = rot.as_rotvec()              # 旋转矢量，对应 vee(log(R))，不是 3×3 矩阵
print("旋转向量:", omega)            # [0, 0, 1.5708] → 绕 z 轴转 90°
print("还原:", R.from_rotvec(omega).as_euler('xyz', degrees=True))  # [0,0,90]
```

---

## 7. 旋转表示互相转换一览

```
        ┌──────────┐  as_euler   ┌──────────┐
        │ 旋转矩阵  │ ──────────► │  欧拉角   │
        │  (R)     │ ◄────────── │ (RPY)    │
        └────┬─────┘  from_euler └──────────┘
             │ as_quat / from_quat
             ▼
        ┌──────────┐  as_rotvec  ┌──────────┐
        │  四元数   │ ──────────► │  轴角/    │
        │  (q)     │ ◄────────── │ 旋转向量  │
        └──────────┘  from_rotvec└──────────┘
```

```python
# 完整转换演示：同一旋转走遍四种表示
rot = R.from_euler('XYZ', [30, 45, 60], degrees=True)

print("旋转矩阵:\n", rot.as_matrix().round(3))
print("欧拉角 :", rot.as_euler('XYZ', degrees=True).round(1))
print("四元数 :", rot.as_quat().round(3))      # [x,y,z,w]
print("旋转向量:", rot.as_rotvec().round(3))    # 方向=轴, 模长=角(弧度)

# 关键：无论从哪种表示出发，as_matrix() 得到的旋转矩阵一致
assert np.allclose(R.from_quat(rot.as_quat()).as_matrix(),
                   R.from_rotvec(rot.as_rotvec()).as_matrix())
```

---

## 8. 连接项目

- **关节角 → 末端位姿**：`examples/fk_ik_demo.py` 输入关节角 `(theta1, theta2)`，输出末端 `(x, y)`。每段连杆的旋转拼起来就是 SO(3)/SE(3) 的链式复合（2D 下退化为 SO(2)）。
- **MuJoCo 状态表示**：`qpos` 中 ball joint 使用 4 个四元数分量，free joint 使用 3 个位置分量 + 4 个四元数分量；hinge / slide joint 各用 1 个标量。四元数顺序为 `[w, x, y, z]`，与 SciPy 默认的 `[x, y, z, w]` 不同。Python 的转换 API 是 `mujoco.mju_quat2Mat(result9, quat4)`，结果写入长度为 9 的数组，可再 `reshape(3, 3)`；不要使用内部辅助函数名 `mjuu_quat2mat`。见 [MuJoCo 状态约定](https://mujoco.readthedocs.io/en/stable/overview.html)与 [四元数转换 API](https://mujoco.readthedocs.io/en/stable/APIreference/APIfunctions.html#mju-quat2mat)。
- **3D 手指链**：`examples/finger_chain_3d.py` 在三维空间里做正运动学，每段连杆的旋转就是 SO(3) 元素，末端位姿属于 SE(3)。

---

## 检查理解

1. **概念题**：SO(3) 的两个约束 `RᵀR=I` 和 `det(R)=+1` 分别排除了哪类"不合法"的变换？如果允许 `det(R)=-1` 会发生什么？
2. **辨析题**：3D 旋转只有 3 个自由度，为什么旋转矩阵有 9 个数、四元数有 4 个数？多出来的参数受什么约束？
3. **万向锁题**：用代码验证 `R.from_euler('ZYX', [30,90,10], degrees=True)` 与 `R.from_euler('ZYX', [40,90,20], degrees=True)` 的旋转矩阵相同，并解释为什么。这里大写 `ZYX` 是内禀旋转；`degrees=True` 不可省略，否则数值按弧度解释，见 [SciPy 约定](https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.transform.Rotation.from_euler.html)。
4. **转换题**：给定四元数 `q=[0.5, 0.5, 0.5, 0.5]`（先判断它是否单位四元数），分别求出对应的旋转矩阵、欧拉角和旋转向量。
5. **项目题**：`finger_chain_3d.py` 中每段连杆的旋转属于 SO(3) 还是 SE(3)？整个手指末端的 6D 位姿属于哪个群？若要把它存入 MuJoCo，应该用哪种旋转表示？

---

> **下一篇**: [`07-fk-jacobian-ik.md`](07-fk-jacobian-ik.md) —— 把这些变换串成正运动学，并用 Jacobian 反解关节角。
