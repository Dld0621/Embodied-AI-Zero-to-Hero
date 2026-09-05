# 评估指标与基准

> **H06 已修订：**旋转误差已区分矩阵对数与旋转向量范数；jerk 的差分阶数、`dt`、汇总和单位已与实现统一；互相关延迟已明确输入信号、采样间隔与正负号；空指标返回 `None`，retargeting 函数调用耗时不再冒充端到端延迟。合成轨迹与延迟已有离线回归，但尚未采集真实传感器—执行器端到端时间戳或真机任务结果。复核范围见 [修订记录](reviews/retargeting-revision-review.md#复核结论)。

> **逐点图解 / Concept close-ups：**[泛化、鲁棒性与分布偏移](knowledge-atlas/eval-generalization-robustness/index.md)。每个小点配原理、算例、图、自测；这是中文细解，保留英文术语。

> 如何定量评估 retargeting 的质量？从关节空间到任务空间，从静态姿态到动态轨迹，建立完整的评估体系。

---

## 1. 为什么需要评估体系？

Retargeting 的质量不能仅凭"看起来对"来判断。一个完整的评估体系需要回答：

1. **静态精度**：单个手势的关节角/指尖位置误差有多大？
2. **动态一致性**：连续动作是否平滑？是否有抖动？
3. **语义保持**：手势的语义（如"抓取"、"张开"）是否被正确保留？
4. **物理可行性**：机器人关节是否在限位内？是否有自碰撞？
5. **任务成功率**：retargeting 后的动作能否完成目标任务？

---

## 2. 关节空间指标

### 2.1 关节角度误差（Joint Angle Error, JAE）

$$\text{JAE} = \frac{1}{N} \sum_{i=1}^{N} |\theta_i^{\text{pred}} - \theta_i^{\text{gt}}|$$

```python
def joint_angle_error(pred_joints, gt_joints):
    """
    平均关节角度误差（弧度）

    Args:
        pred_joints: [n_dof] 预测关节角
        gt_joints: [n_dof] 真实关节角（来自动捕或优化求解）
    """
    return np.mean(np.abs(pred_joints - gt_joints))
```

**适用场景**：有 ground truth 关节角时（如动捕数据）。

### 2.2 关节角度 RMSE

$$\text{RMSE} = \sqrt{\frac{1}{N} \sum_{i=1}^{N} (\theta_i^{\text{pred}} - \theta_i^{\text{gt}})^2}$$

对大误差更敏感，适合惩罚异常值。

### 2.3 关节限位违反率

$$\text{Limit Violation Rate} = \frac{1}{N} \sum_{i=1}^{N} \mathbb{1}[\theta_i \notin [\theta_i^{\min}, \theta_i^{\max}]]$$

```python
def limit_violation_rate(joints, joint_limits):
    """
    关节限位违反率

    Returns:
        rate: [0, 1] 违反比例
        violations: list of (joint_idx, value, limit)
    """
    violations = []
    for i, (j, low, high) in enumerate(zip(joints, joint_limits[:, 0], joint_limits[:, 1])):
        if j < low or j > high:
            violations.append((i, j, (low, high)))

    rate = len(violations) / len(joints)
    return rate, violations
```

---

## 3. 任务空间指标

### 3.1 指尖位置误差（Fingertip Position Error, FPE）

$$\text{FPE} = \frac{1}{5} \sum_{f=1}^{5} \|\mathbf{p}_f^{\text{robot}} - \mathbf{p}_f^{\text{human}}\|$$

```python
def fingertip_position_error(pred_joints, gt_landmarks, robot_model):
    """
    指尖位置误差

    Args:
        pred_joints: [n_dof] 预测机器人关节角
        gt_landmarks: [21, 3] 人手 landmarks
        robot_model: 机器人模型（含 FK）

    Returns:
        fpe: float 平均指尖误差（米）
        per_finger: dict {finger_name: error}
    """
    # 机器人 fingertip 位置
    robot_tips = robot_model.get_fingertip_positions(pred_joints)

    # 人手 fingertip 位置（从 landmarks 提取）
    human_tips = extract_fingertips(gt_landmarks)

    # 对齐尺度（如果必要）
    # ...

    errors = {}
    for finger, (r_tip, h_tip) in enumerate(zip(robot_tips, human_tips)):
        errors[finger] = np.linalg.norm(r_tip - h_tip)

    fpe = np.mean(list(errors.values()))
    return fpe, errors
```

**这是最核心的指标**，因为 retargeting 的最终目标是让机器人手的姿态"看起来像"人手。

### 3.2 归一化 fingertip 误差

消除人手尺寸差异的影响：

$$\text{Normalized FPE} = \frac{\text{FPE}}{L_{\text{hand}}} \times 100\%$$

其中 $L_{\text{hand}}$ 是人手中指长度（手腕到中指 TIP）。

### 3.3 手掌姿态误差

令 $R_\Delta=R_{\text{pred}}^T R_{\text{gt}}\in SO(3)$，矩阵对数 $\operatorname{Log}(R_\Delta)$ 是 $3\times3$ 反对称矩阵，不是三维向量。若 $\boldsymbol\phi=\operatorname{Log}(R_\Delta)^\vee$ 是对应旋转向量，则

$$\text{Orientation Error}=\|\boldsymbol\phi\|_2
=\frac{1}{\sqrt{2}}\|\operatorname{Log}(R_\Delta)\|_F
=\arccos\!\left(\frac{\operatorname{tr}(R_\Delta)-1}{2}\right)$$

结果单位为弧度，取主值区间 $[0,\pi]$。输入必须是同一坐标约定下的有效旋转矩阵。

```python
def orientation_error(R_pred, R_gt):
    """
    旋转矩阵之间的测地线距离
    """
    R_pred = np.asarray(R_pred, dtype=float)
    R_gt = np.asarray(R_gt, dtype=float)
    if R_pred.shape != (3, 3) or R_gt.shape != (3, 3):
        raise ValueError("R_pred and R_gt must have shape [3, 3]")
    for R in (R_pred, R_gt):
        if not np.allclose(R.T @ R, np.eye(3), atol=1e-6) or not np.isclose(
            np.linalg.det(R), 1.0, atol=1e-6
        ):
            raise ValueError("inputs must be valid SO(3) rotation matrices")

    R_diff = R_pred.T @ R_gt
    # 从旋转矩阵提取角度
    trace = np.trace(R_diff)
    angle = np.arccos(np.clip((trace - 1) / 2, -1, 1))
    return angle
```

---

## 4. 动态指标

### 4.1 时域抖动（Jerk / 加速度变化率）

对等间隔采样 $\boldsymbol\theta_0,\ldots,\boldsymbol\theta_{T-1}$，三阶前向差分为

$$\mathbf{j}_t \approx
\frac{\boldsymbol\theta_{t+3}-3\boldsymbol\theta_{t+2}+3\boldsymbol\theta_{t+1}-\boldsymbol\theta_t}{\Delta t^3},
\quad t=0,\ldots,T-4$$

本页汇总标量定义为 $\frac{1}{T-3}\sum_{t=0}^{T-4}\|\mathbf{j}_t\|_2$，不再混用平方范数。若关节角单位为 rad、时间为 s，则结果单位为 rad/s³；不同采样率、滤波器或关节数的结果不能直接混比。

```python
def compute_jerk(joint_trajectory, dt=0.033):
    """
    计算轨迹的 jerk（加速度变化率）

    Args:
        joint_trajectory: [T, n_dof] 关节角轨迹
        dt: 时间步长（秒）

    Returns:
        jerk: float 平均 jerk
    """
    trajectory = np.asarray(joint_trajectory, dtype=float)
    if trajectory.ndim != 2 or trajectory.shape[0] < 4:
        raise ValueError("joint_trajectory must have shape [T, n_dof] with T >= 4")
    if not np.isfinite(trajectory).all() or not np.isfinite(dt) or dt <= 0:
        raise ValueError("trajectory must be finite and dt must be positive")

    # 速度
    velocity = np.diff(trajectory, axis=0) / dt  # [T-1, n_dof]

    # 加速度
    acceleration = np.diff(velocity, axis=0) / dt  # [T-2, n_dof]

    # jerk
    jerk = np.diff(acceleration, axis=0) / dt  # [T-3, n_dof]

    mean_jerk = np.mean(np.linalg.norm(jerk, axis=1))
    return mean_jerk
```

jerk 没有脱离协议的通用“优秀阈值”。至少要固定采样率、滤波、关节集合、单位和任务，再与同协议的人类输入、基线方法或任务结果一起报告。

### 4.2 延迟（Latency）

从人手运动到机器人响应的时间：

$$\text{Latency} = t_{\text{robot}} - t_{\text{human}}$$

```python
def estimate_signal_delay(human_signal, robot_signal, dt):
    """
    通过互相关估计同频采样的一维信号延迟。

    返回值 > 0 表示 robot_signal 相对 human_signal 滞后。
    """
    human = np.asarray(human_signal, dtype=float)
    robot = np.asarray(robot_signal, dtype=float)
    if human.ndim != 1 or robot.ndim != 1 or human.size != robot.size:
        raise ValueError("signals must be one-dimensional arrays with equal length")
    if human.size < 2 or not np.isfinite(dt) or dt <= 0:
        raise ValueError("signals need at least two samples and dt must be positive")
    if not np.isfinite(human).all() or not np.isfinite(robot).all():
        raise ValueError("signals must be finite")

    human = human - human.mean()
    robot = robot - robot.mean()
    if np.linalg.norm(human) == 0 or np.linalg.norm(robot) == 0:
        raise ValueError("constant signals do not define a correlation delay")

    # np.correlate(a, v): c[k] = sum_n a[n+k] * v[n]。
    # 因而把 robot 放在第一个参数时，正 lag 表示 robot 更晚出现。
    correlation = np.correlate(robot, human, mode="full")
    lag_samples = int(np.argmax(correlation) - (human.size - 1))
    return lag_samples * dt
```

互相关估计的是两个已对齐采样流之间的**相对波形位移**，不是自动得到完整系统 latency。周期信号可能有多个峰；相机曝光、时间戳同步、网络、缓冲、求解器、控制周期、执行器响应和测量点都要单独定义。端到端延迟应优先用同步时钟下的事件时间戳验证。

---

## 5. 语义指标

### 5.1 手势分类准确率

如果 retargeting 后机器人手势的语义（如"张开"、"握拳"、"捏取"）与人手一致：

```python
def gesture_classification_accuracy(pred_joints, gt_gesture_labels, classifier):
    """
    手势分类准确率

    Args:
        pred_joints: [N, n_dof] 预测的机器人关节序列
        gt_gesture_labels: [N] 人手手势标签
        classifier: 预训练的手势分类器
    """
    pred_labels = classifier.predict(pred_joints)
    accuracy = np.mean(pred_labels == gt_gesture_labels)
    return accuracy
```

### 5.2 抓取成功率（Grasp Success Rate）

在仿真环境中测试 retargeting 后的抓取能力：

```python
def evaluate_grasp_success(robot_env, retargeting_fn, test_objects, n_trials=50):
    """
    评估抓取成功率

    Returns:
        success_rate: [0, 1]
    """
    successes = 0

    for obj in test_objects:
        for _ in range(n_trials):
            # 随机人手抓取姿态
            human_grasp = sample_human_grasp(obj)

            # Retargeting
            robot_joints = retargeting_fn(human_grasp)

            # 仿真测试
            success = robot_env.test_grasp(robot_joints, obj)
            if success:
                successes += 1

    success_rate = successes / (len(test_objects) * n_trials)
    return success_rate
```

---

## 6. 综合评估框架

### 6.1 评估 Pipeline

```python
def comprehensive_evaluation(retargeting_fn, test_dataset, robot_model):
    """
    综合评估框架

    Args:
        retargeting_fn: 待评估的 retargeting 函数
        test_dataset: 测试数据集（人手 landmarks + ground truth）
        robot_model: 机器人模型

    Returns:
        metrics: dict 包含所有指标
    """
    metrics = {
        'jae': [],           # 关节角度误差
        'fpe': [],           # 指尖位置误差
        'limit_violation': [],# 限位违反率
        'retargeting_call_time_s': [], # 仅函数调用耗时，不是端到端延迟
    }

    for sample in test_dataset:
        landmarks = sample['landmarks']
        gt_joints = sample.get('gt_joints')

        # 运行 retargeting
        start_time = time.perf_counter()
        pred_joints = retargeting_fn(landmarks)
        call_time = time.perf_counter() - start_time

        # 关节空间指标
        if gt_joints is not None:
            metrics['jae'].append(joint_angle_error(pred_joints, gt_joints))

        # 任务空间指标
        fpe, _ = fingertip_position_error(pred_joints, landmarks, robot_model)
        metrics['fpe'].append(fpe)

        # 限位检查
        v_rate, _ = limit_violation_rate(pred_joints, robot_model.joint_limits)
        metrics['limit_violation'].append(v_rate)

        metrics['retargeting_call_time_s'].append(call_time)

    # 空指标返回 None，避免 np.mean([]) 产生 NaN 后被误当成有效分数。
    summary = {
        name: (float(np.mean(values)) if values else None)
        for name, values in metrics.items()
    }
    # 这些指标需要轨迹或外部时间戳，本函数没有采集，所以显式标为未测。
    summary['jerk_rad_s3'] = None
    summary['signal_delay_s'] = None
    summary['end_to_end_latency_s'] = None
    return summary
```

### 6.2 评估报告格式

```text
========================================
Retargeting Evaluation Report
========================================

Method: <method and version>
Test Samples / Sequences: <count>
Coordinate / scale alignment: <protocol>

Joint Space:
  Mean JAE: <value rad, or N/A with reason>
  Limit Violation Rate: <value %, limits source>

Task Space:
  Mean FPE: <value mm, coordinate alignment>
  Normalized FPE: <value %, normalization length>

Dynamic:
  Mean Jerk: <value rad/s^3, dt and filtering; or N/A>
  Retargeting Call Time: <distribution in ms, measurement boundary>
  Signal Delay: <value ms and sign convention; or N/A>
  End-to-End Latency: <value ms and endpoints; or N/A>

Task:
  Grasp Success Rate: <successes / trials and success criterion; or N/A>

Overall Score: <only if a preregistered aggregation rule exists; otherwise N/A>
========================================
```

---

## 7. 基准对比

### 7.1 在相同测试集上对比不同方法

```python
def benchmark_comparison(test_dataset, robot_model):
    """
    多种方法的基准对比
    """
    methods = {
        'Rule-based (scale=1.0)': rule_based_retargeting,
        'Rule-based (scale=1.6)': rule_based_retargeting_v2,
        'Vector Optimization': vector_opt_retargeting,
        'MLP (trained)': mlp_retargeting,
        'CVAE (trained)': cvae_retargeting,
    }

    results = {}
    for name, fn in methods.items():
        print(f"Evaluating {name}...")
        metrics = comprehensive_evaluation(fn, test_dataset, robot_model)
        results[name] = metrics

    # 打印对比表格
    print("\n" + "="*80)
    print(f"{'Method':<30} {'JAE(rad)':<12} {'FPE(mm)':<12} {'Call(ms)':<12}")
    print("="*80)
    for name, m in results.items():
        def show(value, scale=1.0, digits=4):
            return "N/A" if value is None else f"{value * scale:.{digits}f}"

        print(
            f"{name:<30} "
            f"{show(m['jae']):<12} "
            f"{show(m['fpe'], 1000.0, 2):<12} "
            f"{show(m['retargeting_call_time_s'], 1000.0, 2):<12}"
        )

    return results
```

### 7.2 如何制定评分标准

不能脱离机器人、任务和采样协议给出通用的“优秀/合格”阈值。建议先冻结以下合同，再在同一测试集上比较：

1. 关节集合、角度单位、ground truth 来源及其不确定度；
2. 人手与机器人任务空间的坐标/尺度对齐方式；
3. 轨迹采样率、滤波器、差分实现和缺帧处理；
4. latency 的起止事件、时钟同步、预热和统计量（至少中位数及高分位）；
5. 自碰撞所覆盖的 link/geom 与路径采样密度；
6. 任务成功判据、物体集合、试验次数和随机种子。

只有在这些条件已固定、各指标均有有效样本且聚合权重事先定义时，才计算 Overall Score；否则保留 `N/A`，并报告各项原始统计。
