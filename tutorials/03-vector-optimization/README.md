# Stage 3: Vector Optimization Retargeting

> 使用数值优化方法，在任务空间（指尖位置）上对齐人手和机器人手，自动补偿尺寸差异。

---

## 核心思想

不直接映射关节角，而是让机器人的 fingertip 位置尽可能接近人手的 fingertip 位置。

```
目标: min || FK_robot(theta) - fingertips_human ||^2
约束: theta_min <= theta <= theta_max
```

---

## Scipy least_squares 实现

```python
from scipy.optimize import least_squares

def retarget_vector_optimization(human_landmarks, robot_model):
    """
    向量优化 retargeting
    """
    # 提取人手 fingertip
    human_tips = extract_fingertips(human_landmarks)

    def objective(robot_joints):
        robot_tips = robot_model.forward_kinematics(robot_joints)
        return (robot_tips - human_tips).flatten()

    result = least_squares(
        objective,
        x0=initial_guess,
        bounds=(joint_lower, joint_upper),
        method='trf',
        ftol=1e-6,
    )

    return result.x
```

---

## 阻尼最小二乘 IK

在机器人控制中常用的阻尼最小二乘（Damped Least Squares）方法：

```python
def damped_least_squares_ik(target, robot_joints, damping=0.06):
    """
    阻尼最小二乘 IK

    参数:
        damping: 阻尼系数（越大越稳定但越慢）
    """
    J = compute_jacobian(robot_joints)
    error = target - forward_kinematics(robot_joints)

    # delta = J^T (J J^T + lambda^2 I)^{-1} error
    delta = J.T @ np.linalg.inv(J @ J.T + damping**2 * np.eye(3)) @ error

    return robot_joints + delta
```

---

## 运行示例

```bash
cd examples
python minimal_retargeting.py --method compare
```

---

## 工程注意事项

1. **拇指校准问题**：Vector Optimization 对拇指 MCP 的校准容易不准，建议拇指使用 Rule-based
2. **多初始点**：对每只手指分别优化，避免全局耦合
3. **平滑处理**：优化结果需要时域滤波，避免抖动
