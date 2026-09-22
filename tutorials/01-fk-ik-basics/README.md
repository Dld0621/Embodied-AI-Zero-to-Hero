# Stage 1: 正运动学与逆运动学基础

> 理解机器人关节链的基本数学原理，为 retargeting 打下运动学基础。

---

## 正运动学（Forward Kinematics）

### 定义

已知各关节角度，计算末端执行器在任务空间中的位置。

### 2-DOF 平面臂

```python
import numpy as np

def forward_kinematics_2d(theta1, theta2, l1=1.0, l2=0.8):
    """
    2-DOF 平面臂的正运动学
    """
    x1 = l1 * np.cos(theta1)
    y1 = l1 * np.sin(theta1)

    x2 = x1 + l2 * np.cos(theta1 + theta2)
    y2 = y1 + l2 * np.sin(theta1 + theta2)

    return (x2, y2)
```

### 齐次变换矩阵

对于 3D 空间中的机械臂，使用 4×4 齐次变换矩阵：

```python
def rotation_matrix_x(angle):
    c, s = np.cos(angle), np.sin(angle)
    return np.array([
        [1, 0, 0, 0],
        [0, c, -s, 0],
        [0, s, c, 0],
        [0, 0, 0, 1]
    ])

def translation_matrix(x, y, z):
    return np.array([
        [1, 0, 0, x],
        [0, 1, 0, y],
        [0, 0, 1, z],
        [0, 0, 0, 1]
    ])
```

---

## 逆运动学（Inverse Kinematics）

### 定义

已知末端执行器的目标位置，求解各关节角度。

### 解析法

某些简单结构（如 2-DOF 平面臂）有闭式解。

以下取 `theta2 >= 0` 的一个肘部支路，不是列出所有解；另一支路取负的 `arccos` 后重新计算 theta1。默认无关节限位，最终仍需用同一组连杆长度做 FK 验算。

```python
def inverse_kinematics_2d(x, y, l1=1.0, l2=0.8):
    """
    2-DOF 平面臂的解析法 IK
    """
    d = np.sqrt(x**2 + y**2)

    if d > l1 + l2 or d < abs(l1 - l2):
        return None  # 不可达

    cos_theta2 = (x**2 + y**2 - l1**2 - l2**2) / (2 * l1 * l2)
    cos_theta2 = np.clip(cos_theta2, -1.0, 1.0)

    theta2 = np.arccos(cos_theta2)

    alpha = np.arctan2(y, x)
    beta = np.arctan2(l2 * np.sin(theta2), l1 + l2 * np.cos(theta2))
    theta1 = alpha - beta

    return theta1, theta2
```

### 数值法：Jacobian 迭代

对于复杂机械臂，使用数值优化方法。下列是接口骨架：`forward_kinematics` 与 `compute_jacobian` 需要你按具体模型实现，并保证任务空间维度和坐标系一致；它们不是上一段的同名函数。可先运行[基础篇完整 2-DOF DLS 示例](../../docs/foundations/07-fk-jacobian-ik.md)。

```python
def jacobian_ik(target, initial_guess, max_iter=100, damping=0.1):
    """
    阻尼最小二乘 IK
    """
    theta = np.array(initial_guess, dtype=float)
    target = np.asarray(target, dtype=float)

    for _ in range(max_iter):
        current = forward_kinematics(theta)
        error = target - current

        if np.linalg.norm(error) < 1e-4:
            break

        J = compute_jacobian(theta)
        # 阻尼最小二乘: delta = J^T (J J^T + lambda^2 I)^{-1} error
        delta = J.T @ np.linalg.solve(J @ J.T + damping**2 * np.eye(error.size), error)
        theta += delta

    return theta
```

---

## 运行示例

从仓库根目录执行；动画只用于桌面可视化，不涉及机器人硬件。

```bash
cd examples
python fk_ik_demo.py --mode fk
python fk_ik_demo.py --mode ik
python fk_ik_demo.py --mode animate --target 1.2,0.5
```

---

## 延伸阅读

- [DH 参数法](https://en.wikipedia.org/wiki/Denavit%E2%80%93Hartenberg_parameters) — 描述关节链的标准方法
- [Jacobian 矩阵](https://en.wikipedia.org/wiki/Jacobian_matrix_and_determinant) — 关节空间与任务空间的微分映射
