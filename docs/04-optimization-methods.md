# 优化方法深入：从 Jacobian 到约束 IK

> 逆运动学（IK）的数学基础：Jacobian 矩阵推导、阻尼最小二乘（DLS）理论、多指耦合约束优化。这是 retargeting 从"能跑"到"跑得准"的核心。

---

## 1. Jacobian 矩阵：关节空间与任务空间的微分映射

### 1.1 定义

对于 $n$ 自由度机械臂，末端执行器位置 $\mathbf{p} \in \mathbb{R}^3$ 是关节角 $\boldsymbol{\theta} \in \mathbb{R}^n$ 的函数：

$$\mathbf{p} = f(\boldsymbol{\theta})$$

Jacobian 矩阵 $J \in \mathbb{R}^{3 \times n}$ 描述关节速度到末端线速度的映射：

$$\dot{\mathbf{p}} = J \dot{\boldsymbol{\theta}}$$

其中：

$$J = \begin{bmatrix} \frac{\partial p_x}{\partial \theta_1} & \cdots & \frac{\partial p_x}{\partial \theta_n} \\ \frac{\partial p_y}{\partial \theta_1} & \cdots & \frac{\partial p_y}{\partial \theta_n} \\ \frac{\partial p_z}{\partial \theta_1} & \cdots & \frac{\partial p_z}{\partial \theta_n} \end{bmatrix}$$

### 1.2 几何法计算 Jacobian（旋转关节）

对于旋转关节 $i$，Jacobian 的第 $i$ 列为：

$$J_i = \mathbf{z}_i \times (\mathbf{p}_{\text{end}} - \mathbf{p}_i)$$

其中：
- $\mathbf{z}_i$：关节 $i$ 的旋转轴方向（世界坐标系）
- $\mathbf{p}_i$：关节 $i$ 的位置
- $\mathbf{p}_{\text{end}}$：末端执行器位置

```python
import numpy as np

def compute_jacobian_column(rotation_axis, joint_pos, end_effector_pos):
    """
    计算旋转关节的 Jacobian 列

    Args:
        rotation_axis: [3] 旋转轴方向 (世界坐标系, 单位向量)
        joint_pos: [3] 关节位置
        end_effector_pos: [3] 末端执行器位置

    Returns:
        j_col: [3] Jacobian 列向量
    """
    j_col = np.cross(rotation_axis, end_effector_pos - joint_pos)
    return j_col
```

### 1.3 多指尖 Jacobian（灵巧手）

对于多指灵巧手，需要同时控制多个 fingertip。将每个 fingertip 的 Jacobian 堆叠：

$$J_{\text{hand}} = \begin{bmatrix} J_{\text{thumb}} \\ J_{\text{index}} \\ J_{\text{middle}} \\ J_{\text{ring}} \\ J_{\text{pinky}} \end{bmatrix} \in \mathbb{R}^{15 \times n}$$

**关键问题**：手指之间共享手掌关节（如腕关节），导致 Jacobian 存在耦合。

```python
def compute_hand_jacobian(finger_jacobians, shared_joint_indices):
    """
    组合多指 Jacobian，处理共享关节耦合

    Args:
        finger_jacobians: list of [3, n_dof_per_finger]
        shared_joint_indices: 共享关节的索引
    """
    n_fingers = len(finger_jacobians)
    total_task_dim = 3 * n_fingers

    # 构建完整 Jacobian
    J_hand = np.zeros((total_task_dim, total_dof))

    for i, J_finger in enumerate(finger_jacobians):
        row_start = i * 3
        # 将手指 Jacobian 填入对应位置
        # 注意共享关节列的叠加
        ...

    return J_hand
```

---

## 2. 阻尼最小二乘（Damped Least Squares, DLS）

### 2.1 问题背景

标准 Jacobian 转置法：

$$\Delta \boldsymbol{\theta} = J^T \Delta \mathbf{p}$$

**问题**：当机械臂接近奇异位形时， $J$ 接近秩亏， $J^T$ 会放大误差，导致关节速度爆炸。

### 2.2 DLS 推导

在标准最小二乘目标中加入阻尼项：

$$\min_{\Delta \boldsymbol{\theta}} \|J \Delta \boldsymbol{\theta} - \Delta \mathbf{p}\|^2 + \lambda^2 \|\Delta \boldsymbol{\theta}\|^2$$

解析解：

$$\Delta \boldsymbol{\theta} = J^T (J J^T + \lambda^2 I)^{-1} \Delta \mathbf{p}$$

**为什么有效**：
- 当 $J$ 满秩时， $\lambda$ 很小，接近标准伪逆解
- 当 $J$ 接近奇异时， $\lambda^2 I$ 保证矩阵可逆，防止数值爆炸

### 2.3 阻尼系数 $\lambda$ 的选择

| 策略 | 公式 | 说明 |
|------|------|------|
| **常数** | $\lambda = 0.1$ | 简单，但远离奇异时过阻尼 |
| **自适应** | $\lambda = \lambda_0 \cdot (1 - w/w_0)^2$ | $w$ 为可操作度，接近奇异时增大 |
| **SVD-based** | $\lambda_i = \lambda_0$ if $\sigma_i < \epsilon$ else $0$ | 仅对接近零的奇异值加阻尼 |

```python
def dls_ik(J, error, lambda_damp=0.06):
    """
    阻尼最小二乘 IK 求解

    Args:
        J: [m, n] Jacobian
        error: [m] 任务空间误差
        lambda_damp: 阻尼系数

    Returns:
        delta_theta: [n] 关节增量
    """
    m, n = J.shape

    # DLS 解
    JJt = J @ J.T
    damping_matrix = lambda_damp**2 * np.eye(m)

    delta_theta = J.T @ np.linalg.solve(JJt + damping_matrix, error)

    return delta_theta
```

### 2.4 与 Scipy least_squares 的关系

`scipy.optimize.least_squares` 的 `'lm'` 和 `'trf'` 方法本质上也是 DLS 的变体：
- `'lm'`：Levenberg-Marquardt，自适应调整阻尼
- `'trf'`：Trust Region Reflective，处理边界约束

```python
from scipy.optimize import least_squares

def ik_least_squares(target, initial_guess, forward_kin, jac_fn, bounds):
    """
    使用 scipy least_squares 求解 IK
    """
    def residual(theta):
        return forward_kin(theta) - target

    result = least_squares(
        residual,
        x0=initial_guess,
        jac=jac_fn,
        bounds=bounds,
        method='trf',
        ftol=1e-8,
        xtol=1e-8,
        max_nfev=100,
    )
    return result.x
```

---

## 3. 多指耦合约束优化

### 3.1 问题定义

灵巧手 retargeting 不仅是单指 IK，还需要满足全局约束：

1. **手掌姿态约束**：手腕位置 + 朝向必须合理
2. **手指间距离约束**：防止手指交叉穿透
3. **关节限位**：每个关节必须在范围内
4. **自碰撞约束**：手指不能穿透手掌或其他手指

### 3.2 带约束的优化形式

$$\min_{\boldsymbol{\theta}} \sum_{i=1}^{5} \|\mathbf{p}_i^{\text{robot}}(\boldsymbol{\theta}) - \mathbf{p}_i^{\text{human}}\|^2 + \lambda_{\text{reg}} \|\boldsymbol{\theta} - \boldsymbol{\theta}_{\text{nominal}}\|^2$$

约束：
$$\text{s.t.} \quad \boldsymbol{\theta}_{\min} \leq \boldsymbol{\theta} \leq \boldsymbol{\theta}_{\max}$$
$$\|\mathbf{p}_i - \mathbf{p}_j\| \geq d_{\min}, \quad \forall i \neq j \quad \text{(自碰撞)}$$

### 3.3 SLSQP 约束优化

```python
from scipy.optimize import minimize

def constrained_retargeting(human_tips, robot_model, theta_nominal):
    """
    带约束的 retargeting（SLSQP）
    """
    def objective(theta):
        robot_tips = robot_model.forward_kinematics(theta)
        task_error = np.sum((robot_tips - human_tips) ** 2)
        reg = 0.01 * np.sum((theta - theta_nominal) ** 2)
        return task_error + reg

    def collision_constraint(theta):
        """自碰撞约束：返回必须 >= 0 的值"""
        finger_positions = robot_model.get_finger_positions(theta)
        min_dist = float('inf')
        for i in range(len(finger_positions)):
            for j in range(i+1, len(finger_positions)):
                dist = np.linalg.norm(finger_positions[i] - finger_positions[j])
                min_dist = min(min_dist, dist - 0.01)  # 0.01m 安全距离
        return min_dist

    constraints = [
        {'type': 'ineq', 'fun': collision_constraint}
    ]

    bounds = [(low, high) for low, high in zip(robot_model.joint_limits[:, 0],
                                                robot_model.joint_limits[:, 1])]

    result = minimize(
        objective,
        x0=theta_nominal,
        method='SLSQP',
        bounds=bounds,
        constraints=constraints,
        options={'ftol': 1e-8, 'maxiter': 200}
    )

    return result.x
```

### 3.4 CMA-ES：无梯度全局优化

当目标函数非凸（如存在自碰撞惩罚）时，进化策略可以找到更好的解。

```python
import cma

def cma_es_retargeting(human_tips, robot_model, sigma0=0.3):
    """
    使用 CMA-ES 进行全局优化 retargeting
    """
    def objective(theta):
        robot_tips = robot_model.forward_kinematics(theta)
        error = np.sum((robot_tips - human_tips) ** 2)

        # 关节限位惩罚
        penalty = 0
        for i, (t, low, high) in enumerate(zip(theta, robot_model.joint_limits[:, 0],
                                                    robot_model.joint_limits[:, 1])):
            if t < low:
                penalty += (low - t) ** 2
            elif t > high:
                penalty += (t - high) ** 2

        return error + 100 * penalty

    es = cma.CMAEvolutionStrategy(
        x0=robot_model.nominal_pose,
        sigma0=sigma0,
        inopts={'bounds': [robot_model.joint_limits[:, 0], robot_model.joint_limits[:, 1]]}
    )

    while not es.stop():
        solutions = es.ask()
        es.tell(solutions, [objective(x) for x in solutions])

    return es.result.xbest
```

---

## 4. 手掌姿态估计

### 4.1 从 Landmarks 估计手掌坐标系

手掌姿态（位置和朝向）可以通过手腕 + 3 个 MCP 关键点构建正交基来估计：

```python
def estimate_palm_pose(landmarks):
    """
    从 21 点 landmarks 估计手掌姿态

    Returns:
        position: [3] 手掌位置（手腕位置）
        R: [3, 3] 手掌旋转矩阵
    """
    wrist = landmarks[0]
    index_mcp = landmarks[5]
    middle_mcp = landmarks[9]
    pinky_mcp = landmarks[17]

    # 构建正交基
    v1 = index_mcp - wrist  # X 方向（食指方向）
    v2 = pinky_mcp - wrist  # 辅助向量

    x_axis = v1 / np.linalg.norm(v1)
    z_axis = np.cross(v1, v2)
    z_axis = z_axis / np.linalg.norm(z_axis)
    y_axis = np.cross(z_axis, x_axis)

    R = np.stack([x_axis, y_axis, z_axis], axis=1)

    return wrist, R
```

### 4.2 手掌姿态在 Retargeting 中的作用

1. **确定手指根部的世界坐标**：手掌姿态决定各手指 MCP 的基准位置
2. **处理手掌旋转**：手腕翻转时，手指相对位置发生变化
3. **与机械臂 IK 联动**：手掌位置由机械臂 IK 控制，手指由灵巧手 retargeting 控制

---

## 5. 数值稳定性技巧

### 5.1 步长限制

防止单次迭代步长过大导致震荡：

```python
def step_limit(delta_theta, max_step=0.1):
    """限制关节增量步长"""
    norm = np.linalg.norm(delta_theta)
    if norm > max_step:
        delta_theta = delta_theta * (max_step / norm)
    return delta_theta
```

### 5.2 多初始点策略

避免陷入局部最优：

```python
def multi_start_ik(target, robot_model, n_starts=5):
    """多初始点 IK"""
    best_error = float('inf')
    best_theta = None

    for _ in range(n_starts):
        theta0 = np.random.uniform(robot_model.joint_limits[:, 0],
                                   robot_model.joint_limits[:, 1])
        theta = solve_ik(target, theta0, robot_model)
        error = np.linalg.norm(robot_model.forward_kinematics(theta) - target)

        if error < best_error:
            best_error = error
            best_theta = theta

    return best_theta
```

### 5.3 零空间探索（Null Space Exploration）

在满足主任务（fingertip 对齐）的同时，利用零空间优化次要目标：

$$\Delta \boldsymbol{\theta} = J^\dagger \Delta \mathbf{p} + (I - J^\dagger J) \Delta \boldsymbol{\theta}_{\text{null}}$$

其中 $J^\dagger = J^T (J J^T)^{-1}$ 是伪逆， $(I - J^\dagger J)$ 投影到零空间。

```python
def null_space_ik(J, error, theta, theta_nominal, w_null=0.1):
    """
    零空间 IK：主任务 + 关节正则化
    """
    J_pinv = np.linalg.pinv(J)

    # 主任务
    delta_main = J_pinv @ error

    # 零空间投影矩阵
    P_null = np.eye(len(theta)) - J_pinv @ J

    # 次要任务：靠近 nominal pose
    delta_null = w_null * (theta_nominal - theta)

    delta_theta = delta_main + P_null @ delta_null
    return delta_theta
```

---

## 6. 方法对比总结

| 方法 | 收敛速度 | 全局最优 | 约束处理 | 适用场景 |
|------|---------|---------|---------|---------|
| Jacobian 转置 | 快 | 否 | 需投影 | 简单结构、远离奇异 |
| DLS (解析) | 快 | 否 | 需投影 | 实时控制、一般场景 |
| Scipy least_squares | 中等 | 可能 | 边界约束 | 精度要求高 |
| SLSQP | 中等 | 否 | 等式/不等式 | 有复杂约束 |
| CMA-ES | 慢 | 是 | 边界 | 非凸、多模态 |
