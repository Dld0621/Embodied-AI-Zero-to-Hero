# 评估指标与基准

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

$$\text{Orientation Error} = \|\text{Log}(R_{\text{robot}}^T R_{\text{human}})\|$$

```python
def orientation_error(R_pred, R_gt):
    """
    旋转矩阵之间的测地线距离
    """
    R_diff = R_pred.T @ R_gt
    # 从旋转矩阵提取角度
    trace = np.trace(R_diff)
    angle = np.arccos(np.clip((trace - 1) / 2, -1, 1))
    return angle
```

---

## 4. 动态指标

### 4.1 时域抖动（Jerk / 加速度变化率）

衡量动作平滑度：

$$\text{Jerk} = \frac{1}{T-2} \sum_{t=2}^{T-1} \|\ddot{\theta}_t - \ddot{\theta}_{t-1}\|^2$$

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
    # 速度
    velocity = np.diff(joint_trajectory, axis=0) / dt  # [T-1, n_dof]

    # 加速度
    acceleration = np.diff(velocity, axis=0) / dt  # [T-2, n_dof]

    # jerk
    jerk = np.diff(acceleration, axis=0) / dt  # [T-3, n_dof]

    mean_jerk = np.mean(np.linalg.norm(jerk, axis=1))
    return mean_jerk
```

**好的 retargeting**：jerk 应该接近人类自然运动的 jerk（约 $10^3 \text{ rad/s}^3$）。

### 4.2 延迟（Latency）

从人手运动到机器人响应的时间：

$$\text{Latency} = t_{\text{robot}} - t_{\text{human}}$$

```python
def measure_latency(human_times, robot_times):
    """
    通过互相关计算延迟
    """
    correlation = np.correlate(human_times, robot_times, mode='full')
    lag = np.argmax(correlation) - len(human_times) + 1
    latency = lag * dt
    return latency
```

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
        'fpe_normalized': [],# 归一化 fingertip 误差
        'limit_violation': [],# 限位违反率
        'jerk': [],          # 轨迹平滑度
        'latency': [],       # 延迟
    }

    for sample in test_dataset:
        landmarks = sample['landmarks']
        gt_joints = sample.get('gt_joints')

        # 运行 retargeting
        start_time = time.time()
        pred_joints = retargeting_fn(landmarks)
        latency = time.time() - start_time

        # 关节空间指标
        if gt_joints is not None:
            metrics['jae'].append(joint_angle_error(pred_joints, gt_joints))

        # 任务空间指标
        fpe, _ = fingertip_position_error(pred_joints, landmarks, robot_model)
        metrics['fpe'].append(fpe)

        # 限位检查
        v_rate, _ = limit_violation_rate(pred_joints, robot_model.joint_limits)
        metrics['limit_violation'].append(v_rate)

        # 延迟
        metrics['latency'].append(latency)

    # 汇总
    summary = {k: np.mean(v) for k, v in metrics.items()}
    return summary
```

### 6.2 评估报告格式

```
========================================
Retargeting Evaluation Report
========================================

Method: Rule-based (scale=1.60)
Test Samples: 1000

Joint Space:
  Mean JAE: 0.085 rad (4.87 deg)
  Max JAE:  0.312 rad (17.88 deg)
  Limit Violation Rate: 0.0%

Task Space:
  Mean FPE: 12.3 mm
  Normalized FPE: 8.2%
  Per-finger FPE:
    Thumb:  18.5 mm
    Index:   9.2 mm
    Middle:  8.1 mm
    Ring:   10.3 mm
    Pinky:  15.7 mm

Dynamic:
  Mean Jerk: 2.3e3 rad/s^3
  Latency: 0.8 ms

Grasp Success Rate: 78.5%

Overall Score: 82/100
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
    print(f"{'Method':<30} {'JAE(rad)':<12} {'FPE(mm)':<12} {'Jerk':<12} {'Latency(ms)':<12}")
    print("="*80)
    for name, m in results.items():
        print(f"{name:<30} {m['jae']:<12.4f} {m['fpe']*1000:<12.2f} {m['jerk']:<12.1f} {m['latency']*1000:<12.1f}")

    return results
```

### 7.2 建议的评分标准

| 指标 | 优秀 (>90) | 良好 (70-90) | 合格 (50-70) | 差 (<50) |
|------|-----------|-------------|-------------|---------|
| JAE (deg) | < 3 | 3-8 | 8-15 | > 15 |
| FPE (%) | < 5% | 5-10% | 10-20% | > 20% |
| Limit Violation | 0% | < 1% | < 5% | > 5% |
| Jerk (rad/s^3) | < 1e3 | 1e3-3e3 | 3e3-1e4 | > 1e4 |
| Latency (ms) | < 1 | 1-5 | 5-20 | > 20 |
