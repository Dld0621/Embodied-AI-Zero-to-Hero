# Coordinate Transform 坐标变换

> English contract: [Foundations overview](README_EN.md#route) · Primary references: [Coordinate transforms](../SOURCES.md#05-coordinate-transforms)

> **前置要求**: [`02-linear-algebra.md`](02-linear-algebra.md)（向量、矩阵乘法、逆矩阵）
> **预计学习时间**: 2–3 小时
> **完成后你能**: 用齐次坐标描述 2D/3D 刚体位姿，串联多个坐标系变换，并区分主动变换与被动变换；理解项目中 PushCube 的 2D 坐标与机械臂各连杆坐标系的来源。

---

## 1. 为什么需要坐标变换？

机械工程里我们早就接触过坐标系：画零件图用工程坐标系，加工时用机床坐标系，测量时用测头坐标系。机器人也一样——同一个点，在不同坐标系下数值不同，但物理位置不变。

```
        世界坐标系 W
        ┌────────────────────────┐
        │                        │
        │     ●  cube            │   ← 推方块任务：方块在 W 下的 (x, y)
        │                        │
        │        [基座]──────[末端]   ← 机械臂：末端在基座 B 下的 (x, y)
        │                        │
        └────────────────────────┘
```

**核心问题**：已知末端在基座坐标系 B 下的坐标，如何得到它在世界坐标系 W 下的坐标？这就是坐标变换要解决的事。

---

## 2. 三种常见坐标系

| 坐标系 | 符号 | 作用 | 项目中的例子 |
|:-------|:-----|:-----|:------------|
| **世界坐标系 (World frame)** | W | 固定不动的全局参考 | PushCube 中方块、桌面位置 |
| **物体坐标系 (Body frame)** | B | 固连在刚体上，随刚体运动 | 机械臂基座、各连杆、末端执行器 |
| **传感器坐标系 (Sensor frame)** | S | 固连在相机/IMU 上 | 手眼相机的成像原点 |

> **直觉**：坐标系 = 一把"尺子"。换坐标系 = 换一把尺子量同一个物体，物体没动，读数变了。

---

## 3. 2D 变换

### 3.1 旋转矩阵

在平面内，把坐标系（或向量）绕原点逆时针转角 θ，新坐标与旧坐标的关系为：

```
x'   [ cosθ  -sinθ ]   x
y' = [ sinθ   cosθ ] · y
```

记旋转矩阵 `R(θ)`。物理直觉：矩阵的两列就是**新坐标系的 x 轴、y 轴在旧坐标系下的方向**。

### 3.2 平移

纯平移只加一个偏移向量 `t = [tx, ty]ᵀ`：

```
p' = p + t
```

### 3.3 齐次坐标

旋转是乘法、平移是加法，两者混在一起写起来很乱。**齐次坐标**的妙处是把平移也变成矩阵乘法：在二维向量末尾补一个 `1`，写成 3×1：

```
x'   [ cosθ  -sinθ  tx ]   x
y' = [ sinθ   cosθ  ty ] · y
1    [  0      0     1 ]   1
```

这个 3×3 矩阵就是 **2D 齐次变换矩阵**，记作 `T`。它把"旋转 + 平移"打包成一个矩阵，后续就能用矩阵乘法串联。

```python
import numpy as np

def transform_2d(theta, tx, ty):
    """构造 2D 齐次变换矩阵 (旋转 theta + 平移 (tx, ty))"""
    c, s = np.cos(theta), np.sin(theta)
    return np.array([
        [c, -s, tx],
        [s,  c, ty],
        [0,  0,  1 ],
    ])

def apply_2d(T, p):
    """用齐次变换矩阵作用到 2D 点上"""
    p_h = np.array([p[0], p[1], 1.0])      # 转齐次坐标
    return (T @ p_h)[:2]                    # 取前两维

# 把点 (1, 0) 先旋转 90°、再平移 (2, 3)  （矩阵 [R, t] 作用: p' = R·p + t）
T = transform_2d(np.pi/2, 2.0, 3.0)
print(apply_2d(T, (1.0, 0.0)))   # ≈ [2.0, 4.0]
```

---

## 4. 3D 变换

### 4.1 3D 旋转

三维旋转可以绕三个轴进行。绕 z 轴转 ψ、绕 y 轴转 θ、绕 x 轴转 φ 的基本旋转矩阵为：

```
         [ cψ -sψ  0]            [ cθ  0  sθ]            [ 1   0    0]
Rz(ψ) = [ sψ  cψ  0]   Ry(θ) = [  0   1   0]   Rx(φ) = [ 0  cφ -sφ]
         [  0   0   1]            [-sθ  0  cθ]            [ 0  sφ  cφ]
```

任意 3D 旋转可由这三个矩阵相乘得到（顺序很重要！详见 [06-se3-and-rotation.md](06-se3-and-rotation.md)）。

### 4.2 3D 齐次变换矩阵

把 3D 旋转 `R`（3×3）和平移 `t`（3×1）打包成 4×4 矩阵：

```
T = [ R   t ]
    [ 0   1 ]

    [ r11 r12 r13 | tx ]
  = [ r21 r22 r23 | ty ]
    [ r31 r32 r33 | tz ]
    [  0   0   0  |  1 ]
```

这个矩阵完整描述了一个刚体在三维空间的**位姿 (pose)** = 位置 (position) + 朝向 (orientation)。

```python
def transform_3d_from_euler(roll, pitch, yaw, tx, ty, tz):
    """由 RPY 欧拉角 + 平移构造 3D 齐次变换矩阵 (ZYX 顺序)"""
    cr, sr = np.cos(roll),  np.sin(roll)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cy, sy = np.cos(yaw),   np.sin(yaw)
    Rx = np.array([[1,0,0],[0,cr,-sr],[0,sr,cr]])
    Ry = np.array([[cp,0,sp],[0,1,0],[-sp,0,cp]])
    Rz = np.array([[cy,-sy,0],[sy,cy,0],[0,0,1]])
    R = Rz @ Ry @ Rx
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3]  = [tx, ty, tz]
    return T

def apply_3d(T, p):
    """用 4x4 变换矩阵作用到 3D 点上"""
    p_h = np.array([p[0], p[1], p[2], 1.0])
    return (T @ p_h)[:3]

T = transform_3d_from_euler(0, 0, np.pi/2, 1.0, 0.0, 0.0)  # 绕 z 转 90° + x 方向平移 1
print(apply_3d(T, (1.0, 0.0, 0.0)))   # ≈ [1.0, 1.0, 0.0]
```

---

## 5. 变换的复合（链式法则）

这是最重要的一节。设坐标系 A、B、C，已知：

- `T_AB`：B 相对 A 的位姿（"用 A 的尺子量 B"）
- `T_BC`：C 相对 B 的位姿

那么 **C 相对 A 的位姿**就是把两个矩阵相乘：

```
T_AC = T_AB · T_BC
```

> **记忆口诀**：相邻字母消掉（B 消掉，剩 A→C）。注意顺序——矩阵乘法不满足交换律！

物理直觉：你先站在 A 看 B，再"跳"到 B 看 C，两次跳的累积效果就是从 A 直接看 C。

```
   A ──T_AB──> B ──T_BC──> C
   └─────────T_AC──────────┘   (T_AC = T_AB · T_BC)
```

<div class="dof-principle" role="group" aria-label="坐标变换链原理图">
  <p class="dof-principle__caption"><strong>原理图 · Transform composition.</strong> 每条箭头都说明“后一个坐标系如何用前一个坐标系表达”；沿路径相乘，得到跨越整条链的变换。</p>
  <div class="dof-principle__canvas">
    <svg viewBox="0 0 920 300" role="img" aria-labelledby="transform-figure-title transform-figure-desc">
      <title id="transform-figure-title">Three coordinate frames and their transform composition</title>
      <desc id="transform-figure-desc">Frame A connects to B and B connects to C. The composed transform from A to C equals T AB times T BC.</desc>
      <defs>
        <marker id="transform-arrow" markerWidth="8" markerHeight="8" refX="6" refY="3" orient="auto"><path class="dof-diagram-arrow" d="M0,0 L7,3 L0,6 Z"/></marker>
      </defs>
      <rect class="dof-diagram-surface" x="34" y="66" width="200" height="140" rx="18"/>
      <rect class="dof-diagram-surface" x="360" y="66" width="200" height="140" rx="18"/>
      <rect class="dof-diagram-surface" x="686" y="66" width="200" height="140" rx="18"/>
      <text class="dof-diagram-title" x="58" y="98">A · World / Base</text>
      <text class="dof-diagram-title" x="384" y="98">B · Wrist / Link</text>
      <text class="dof-diagram-title" x="710" y="98">C · Tool / Camera</text>
      <path class="dof-diagram-accent" d="M92 165 h55" marker-end="url(#transform-arrow)"/>
      <path class="dof-diagram-violet" d="M92 165 v-50" marker-end="url(#transform-arrow)"/>
      <text class="dof-diagram-note" x="154" y="169">x</text><text class="dof-diagram-note" x="88" y="110">y</text>
      <path class="dof-diagram-accent" d="M418 165 h55" marker-end="url(#transform-arrow)"/>
      <path class="dof-diagram-violet" d="M418 165 v-50" marker-end="url(#transform-arrow)"/>
      <text class="dof-diagram-note" x="480" y="169">x</text><text class="dof-diagram-note" x="414" y="110">y</text>
      <path class="dof-diagram-accent" d="M744 165 h55" marker-end="url(#transform-arrow)"/>
      <path class="dof-diagram-violet" d="M744 165 v-50" marker-end="url(#transform-arrow)"/>
      <text class="dof-diagram-note" x="806" y="169">x</text><text class="dof-diagram-note" x="740" y="110">y</text>
      <path class="dof-diagram-accent" d="M240 120 H352" marker-end="url(#transform-arrow)"/>
      <path class="dof-diagram-accent" d="M566 120 H678" marker-end="url(#transform-arrow)"/>
      <text class="dof-diagram-math" x="266" y="106">T_AB</text>
      <text class="dof-diagram-math" x="592" y="106">T_BC</text>
      <path class="dof-diagram-dash" d="M232 232 C372 282, 548 282, 688 232" marker-end="url(#transform-arrow)"/>
      <text class="dof-diagram-math" x="406" y="266">T_AC = T_AB · T_BC</text>
      <text class="dof-diagram-note" x="52" y="230">same physical point, different coordinates</text>
    </svg>
  </div>
</div>

```python
# 机械臂：基座 A → 肘关节 B → 末端 C
# 一段连杆的变换 = 先转关节角，再沿(局部)x 轴平移连杆长度
def link_transform_2d(theta, length):
    """连杆变换：转 theta 后沿自身 x 轴平移 length"""
    c, s = np.cos(theta), np.sin(theta)
    return np.array([
        [c, -s, length*c],
        [s,  c, length*s],
        [0,  0,  1       ],
    ])

T_AB = link_transform_2d(np.pi/4, 1.0)   # 第一段：转 45°，连杆长 1
T_BC = link_transform_2d(np.pi/4, 1.0)   # 第二段：相对第一段再转 45°，连杆长 1
T_AC = T_AB @ T_BC                        # 末端相对基座

print("末端位置:", T_AC[:2, 2])           # ≈ [0.707, 1.707]
```

### 5.1 逆变换

`T_AB` 的逆 `T_AB⁻¹` 表示 A 相对 B 的位姿。齐次变换矩阵的逆有简洁公式：

```
T⁻¹ = [ Rᵀ   -Rᵀ·t ]
      [  0      1   ]
```

```python
def invert_transform(T):
    """求齐次变换矩阵的逆（兼容 2D 的 3x3 与 3D 的 4x4）"""
    n = T.shape[0] - 1          # 空间维度: 2 或 3
    R = T[:n, :n]
    t = T[:n, n]
    T_inv = np.eye(n + 1)
    T_inv[:n, :n] = R.T
    T_inv[:n, n]  = -R.T @ t
    return T_inv

# 验证 T · T⁻¹ = I
T_AC_inv = invert_transform(T_AC)
print(np.allclose(T_AC @ T_AC_inv, np.eye(T_AC.shape[0])))   # True
```

---

## 6. 主动变换 vs 被动变换

这两个概念极易混淆，但对理解代码至关重要。

| | 主动变换 (Active) | 被动变换 (Passive) |
|:--|:------------------|:-------------------|
| **谁在动** | 物体（点）在动 | 坐标系在动 |
| **物理含义** | 把点 p 旋转/平移到新位置 p' | 换一把尺子量同一个静止的点 |
| **矩阵关系** | p' = R · p | p_new = R⁻¹ · p_old = Rᵀ · p_old |
| **项目场景** | 控制末端移动到目标 | 把相机观测转换到基座坐标系 |

> **直觉**：主动变换 = "把书转 90°"；被动变换 = "你绕着书走 90°再读数"。书没变，但读数变了。
>
> 在机器人代码里，`apply_3d(T, p)`（让点动）是主动；`T⁻¹`（换坐标系）是被动。两者用同一个矩阵 T，但含义相反。

---

## 7. 连接项目

- **PushCube 任务**：`examples/unified_pushcube_env.py` 使用 2D 坐标 `(x, y)` 描述方块与末端位置。方块从世界坐标系算出"在末端后面 0.06"的接近点（approach point），本质就是一次 2D 平移变换。
- **机械臂连杆**：从基座 → 各关节 → 末端的位姿传递，正是第 5 节的链式复合 `T_AC = T_AB · T_BC`。`examples/fk_ik_demo.py` 中的 2-DOF 平面臂就是把两段连杆变换相乘得到末端位置。
- **手眼标定**：相机观测在传感器坐标系 S 下，需要 `T_BS` 把它转到机械臂基座 B，再 `T_WB` 转到世界 W——三段链式复合。

---

## 检查理解

1. **概念题**：齐次坐标为什么要给 2D 点补一个 `1`、给 3D 点补一个 `1`？如果补成别的数会怎样？
2. **计算题**：已知 `T_AB` 表示绕 z 轴转 90° 且平移 `(1, 0, 0)`，`T_BC` 表示绕 z 轴转 90° 且平移 `(0, 1, 0)`。手算 `T_AC = T_AB · T_BC`，并验证 `T_AC · T_AC⁻¹ = I`。
3. **辨析题**：机械臂控制器让末端从 `(0.3, 0.2)` 移动到 `(0.5, 0.4)`，这是主动变换还是被动变换？把相机拍到的物体坐标换算到基座坐标系下呢？
4. **编程题**：写一个函数，输入按顺序排列的变换列表 `[T_01, T_12, ..., T_(n-1)n]`，返回总变换 `T_0n`。测试 n=3 的情况。
5. **项目题**：在 `fk_ik_demo.py` 的 `PlanarArm2D.forward_kinematics` 中，找出与"链式复合"对应的两行代码。若改成 3 段连杆，公式该如何扩展？

---

> **下一篇**: [`06-se3-and-rotation.md`](06-se3-and-rotation.md) —— 旋转的更深一层：SO(3)、SE(3) 与四元数。
