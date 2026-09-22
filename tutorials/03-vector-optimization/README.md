# Stage 3: Vector Optimization Retargeting

> 使用数值优化方法，在统一坐标和尺度后对齐任务空间目标；优化本身不会自动完成标定或保证可达。

---

## 核心思想

不直接映射关节角，而是让机器人的 fingertip 位置尽可能接近人手的 fingertip 位置。

下面实际展示的是**位置目标优化**。严格的相对向量目标会比较两关键点之差，例如 `robot_tip - robot_wrist` 与已经过旋转/尺度映射的 `human_tip - human_wrist`。两种目标不能只因都用了数组就视为完全相同。

```
目标: min || FK_robot(theta) - fingertips_human ||^2
约束: theta_min <= theta <= theta_max
```

---

## Scipy least_squares 实现

接口骨架：需要已有 FK、指尖提取函数与模型关节边界。`human_landmarks` 必须先映射到 FK 使用的坐标系和米制尺度，不能直接传入 MediaPipe 图像归一化结果。未在本页提供完整数据/模型，也不声称端到端复现。

```python
import numpy as np
from scipy.optimize import least_squares

def retarget_vector_optimization(human_landmarks, robot_model, initial_guess,
                                joint_lower, joint_upper):
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

    return result  # 检查 success、cost 与实际残差；数值终止不等于精度达标
```

---

## 阻尼最小二乘 IK

在机器人控制中常用的阻尼最小二乘（Damped Least Squares）方法：

下列同样是单次更新骨架，依赖具体模型的 Jacobian 与 FK；阻尼抑制近奇异方向的大更新，但不保证所有初值都收敛。

```python
def damped_least_squares_ik(target, robot_joints, damping=0.06):
    """
    阻尼最小二乘 IK

    参数:
        damping: 正阻尼系数，控制近奇异方向的更新幅度
    """
    J = compute_jacobian(robot_joints)
    error = target - forward_kinematics(robot_joints)

    # delta = J^T (J J^T + lambda^2 I)^{-1} error
    delta = J.T @ np.linalg.solve(J @ J.T + damping**2 * np.eye(error.size), error)

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

1. **拇指校准**：先核对拇指轴、关节耦合与目标定义；规则或优化方案孰优需在同一数据和误差定义下比较。
2. **多初始点与耦合**：多初值可减轻局部极值；逐指优化更简单，但忽略指间碰撞/接触时可能需要全手联合约束。
3. **平滑处理**：滤波可以减噪也会增加延迟，之后仍需检查轨迹限位、速度与任务空间误差。
