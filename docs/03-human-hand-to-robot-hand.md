# 人手到机器人手映射详解

> **H03 已修订：**本页已把“手腕中心化”与“旋转到手掌坐标系”分开，屈曲角改为伸直时为零，并显式拒绝退化骨段；MuJoCo 的 `qpos` / `qvel` 地址也已区分。相关几何片段已有离线回归，但具体 O10 关节名、顺序、方向、传动、限位和硬件控制接口仍须对目标模型逐项核实，未作真机验证。复核范围见 [修订记录](reviews/retargeting-revision-review.md#复核结论)。

> 从视觉捕捉的 21 点坐标到机器人特征的教学 pipeline，包括坐标系转换、左右手合同与关节限位核对；接入具体机器人仍需模型级验证。

---

## MediaPipe 21 点模型

MediaPipe Hands 输出人手的 21 个 3D landmarks，编号如下：

```
手腕 (0)
│
├─ 拇指: CMC(1) → MCP(2) → IP(3) → TIP(4)
├─ 食指: MCP(5) → PIP(6) → DIP(7) → TIP(8)
├─ 中指: MCP(9) → PIP(10) → DIP(11) → TIP(12)
├─ 无名指: MCP(13) → PIP(14) → DIP(15) → TIP(16)
└─ 小指: MCP(17) → PIP(18) → DIP(19) → TIP(20)
```

### 关键点分组

```python
FINGER_INDICES = {
    "thumb":  [1, 2, 3, 4],
    "index":  [5, 6, 7, 8],
    "middle": [9, 10, 11, 12],
    "ring":   [13, 14, 15, 16],
    "pinky":  [17, 18, 19, 20],
}

WRIST_IDX = 0
MCP_INDICES = [2, 5, 9, 13, 17]  # 拇指 MCP 是 2，其他手指 MCP 是 5,9,13,17
```

---

## 坐标系转换

### 步骤 1：手腕中心化（只去平移）

视觉捕捉的 landmarks 通常在相机坐标系下。减去手腕位置只能把原点移到手腕；所有坐标轴仍与相机坐标轴平行，所以这一步**没有消除手掌旋转**。

```python
def center_at_wrist(landmarks):
    """
    将手腕移到原点；输出仍用相机坐标轴表达。

    Args:
        landmarks: [21, 3] 3D 坐标

    Returns:
        centered_landmarks: [21, 3] 手腕中心化的相机系坐标
    """
    points = np.asarray(landmarks, dtype=float)
    if points.shape != (21, 3) or not np.isfinite(points).all():
        raise ValueError("landmarks must be a finite [21, 3] array")
    return points - points[0]
```

### 步骤 2：旋转到手掌坐标系

下面用“小指 MCP → 食指 MCP”定义手掌横向轴，用“手腕 → 中指 MCP”提供远端方向，再用 Gram–Schmidt 正交化构造右手正交基。若关键点重合或两条方向近乎平行，姿态不可辨识，应该拒绝这一帧，而不是悄悄除以极小数。

```python
def to_palm_coordinates(landmarks, eps=1e-8):
    """相机系 [21, 3] → 以手腕为原点、以手掌轴为基的坐标。"""
    centered = center_at_wrist(landmarks)

    x_seed = centered[5] - centered[17]  # pinky MCP -> index MCP
    y_seed = centered[9]                 # wrist -> middle MCP

    x_norm = np.linalg.norm(x_seed)
    if x_norm < eps:
        raise ValueError("index and pinky MCPs do not define a lateral axis")
    x_axis = x_seed / x_norm

    y_orthogonal = y_seed - np.dot(y_seed, x_axis) * x_axis
    y_norm = np.linalg.norm(y_orthogonal)
    if y_norm < eps:
        raise ValueError("palm directions are degenerate or nearly parallel")
    y_axis = y_orthogonal / y_norm
    z_axis = np.cross(x_axis, y_axis)

    # 列向量是手掌基在相机坐标系中的表达；行向量右乘 R 得到手掌坐标。
    R_camera_from_palm = np.column_stack((x_axis, y_axis, z_axis))
    return centered @ R_camera_from_palm
```

### 步骤 3：尺度归一化

以手腕到中指 MCP 的距离作为单位长度，消除不同人手尺寸的影响。

```python
def normalize_scale(local_landmarks):
    """
    尺度归一化
    """
    wrist = local_landmarks[0]
    middle_mcp = local_landmarks[9]
    scale = np.linalg.norm(middle_mcp - wrist)

    if scale < 1e-6:
        return local_landmarks  # 避免除零

    normalized = local_landmarks / scale
    return normalized
```

### 步骤 4：按目标合同处理左右手

“左手一定把 Y 轴取反”不是通用规律。反射哪个轴、是否需要反射，取决于上游手掌坐标系和目标机器人是左手、右手还是统一的右手规范。反射会改变坐标系手性，不是普通旋转；必须用已知姿态校准。

```python
def reflect_for_target_hand(landmarks, source_hand, target_hand, lateral_axis=0):
    """
    仅在源手与目标手不同侧时，按已校准的手掌横向轴做反射。

    source_hand / target_hand: "left" 或 "right"
    lateral_axis: 由 palm-frame 合同定义；本页坐标系中横向轴为 0 (x)
    """
    if source_hand not in {"left", "right"} or target_hand not in {"left", "right"}:
        raise ValueError("source_hand and target_hand must be 'left' or 'right'")
    result = np.asarray(landmarks, dtype=float).copy()
    if source_hand != target_hand:
        result[:, lateral_axis] *= -1
    return result
```

### 完整坐标转换 Pipeline

```python
def preprocess_landmarks(landmarks, source_hand, target_hand):
    """
    完整的 landmarks 预处理
    """
    # 1. 平移 + 旋转到手掌坐标系
    palm_local = to_palm_coordinates(landmarks)

    # 2. 尺度归一化（此时手腕已经是零向量）
    normalized = normalize_scale(palm_local)

    # 3. 只有源手和目标手不同侧时才按接口合同反射
    mirrored = reflect_for_target_hand(normalized, source_hand, target_hand)

    return mirrored
```

---

## 从 Landmarks 计算角度特征

### 弯曲角（Flexion Angle）

这里把屈曲量定义为“连续两根骨段的方向变化”：完全伸直约为 $0$，弯曲时增大。若先计算以关节点为顶点的几何内角 $\alpha$（伸直时约为 $\pi$），则屈曲量为 $\pi-\alpha$；不能直接把 $\alpha$ 当屈曲量。

```python
def compute_flexion_angle(landmarks, joint_indices, eps=1e-8):
    """
    计算手指关节的弯曲角

    Args:
        landmarks: [21, 3]
        joint_indices: [i, j, k] 三个连续关键点的索引

    Returns:
        angle: 弯曲角（弧度）
    """
    p1 = landmarks[joint_indices[0]]
    p2 = landmarks[joint_indices[1]]
    p3 = landmarks[joint_indices[2]]

    proximal_direction = p2 - p1
    distal_direction = p3 - p2

    proximal_norm = np.linalg.norm(proximal_direction)
    distal_norm = np.linalg.norm(distal_direction)
    if proximal_norm < eps or distal_norm < eps:
        raise ValueError("flexion angle is undefined for a zero-length bone segment")

    cos_angle = np.dot(proximal_direction, distal_direction) / (
        proximal_norm * distal_norm
    )
    cos_angle = np.clip(cos_angle, -1.0, 1.0)
    angle = np.arccos(cos_angle)

    return angle

# 示例：计算食指 PIP 弯曲角
index_pip_angle = compute_flexion_angle(landmarks, [5, 6, 7])
```

### 外展角（Abduction Angle）

相邻手指根部方向向量之间的夹角，反映手指张开程度。

```python
def compute_abduction_angle(landmarks, finger1_base, finger2_base, wrist_idx=0):
    """
    计算两根手指的外展角

    Args:
        finger1_base: 第一根手指的 MCP 索引
        finger2_base: 第二根手指的 MCP 索引
    """
    wrist = landmarks[wrist_idx]
    mcp1 = landmarks[finger1_base]
    mcp2 = landmarks[finger2_base]

    v1 = mcp1 - wrist
    v2 = mcp2 - wrist

    cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-8)
    cos_angle = np.clip(cos_angle, -1.0, 1.0)
    angle = np.arccos(cos_angle)

    return angle
```

---

## 教学示例：构造 10 维屈曲特征

下面只演示“每指两个屈曲特征”的数组构造，**不声称这就是某个 O10 模型的控制向量**。机器人自由度、主动关节数、关节顺序、正方向、耦合/传动和命令接口必须以目标 MJCF/URDF 与控制器合同为准。

### Rule-based 映射

```python
def map_to_ten_flexion_features(
    human_landmarks,
    joint_limits,
    source_hand="right",
    target_hand="right",
    gains=None,
):
    """
    将人手 21 点转换为 10 个教学用屈曲特征。

    Returns:
        features: [10] 经显式增益和调用方限位处理的特征（弧度）
    """
    landmarks = preprocess_landmarks(human_landmarks, source_hand, target_hand)

    joints = []

    # 拇指: MCP(2), IP(3)
    thumb_mcp = compute_flexion_angle(landmarks, [1, 2, 3])
    thumb_ip = compute_flexion_angle(landmarks, [2, 3, 4])
    joints.extend([thumb_mcp, thumb_ip])

    # 食指: MCP(5), PIP(6)
    index_mcp = compute_flexion_angle(landmarks, [0, 5, 6])
    index_pip = compute_flexion_angle(landmarks, [5, 6, 7])
    joints.extend([index_mcp, index_pip])

    # 中指: MCP(9), PIP(10)
    middle_mcp = compute_flexion_angle(landmarks, [0, 9, 10])
    middle_pip = compute_flexion_angle(landmarks, [9, 10, 11])
    joints.extend([middle_mcp, middle_pip])

    # 无名指: MCP(13), PIP(14)
    ring_mcp = compute_flexion_angle(landmarks, [0, 13, 14])
    ring_pip = compute_flexion_angle(landmarks, [13, 14, 15])
    joints.extend([ring_mcp, ring_pip])

    # 小指: MCP(17), PIP(18)
    pinky_mcp = compute_flexion_angle(landmarks, [0, 17, 18])
    pinky_pip = compute_flexion_angle(landmarks, [17, 18, 19])
    joints.extend([pinky_mcp, pinky_pip])

    # 转换为 numpy 并应用缩放系数
    joints = np.array(joints)

    limits = np.asarray(joint_limits, dtype=float)
    if limits.shape != (10, 2) or np.any(limits[:, 0] >= limits[:, 1]):
        raise ValueError("joint_limits must have shape [10, 2] with lower < upper")
    gains = np.ones(10) if gains is None else np.asarray(gains, dtype=float)
    if gains.shape != (10,):
        raise ValueError("gains must have shape [10]")

    joints = np.clip(joints * gains, limits[:, 0], limits[:, 1])

    return joints
```

### 接入具体机器人前的核对表

1. **名称与顺序**：从实际 MJCF/URDF 或驱动接口读取，不把 landmark 编号当关节或 actuator 地址。
2. **位置、速度与控制地址**：在 MuJoCo 中分别核对 `jnt_qposadr`、`jnt_dofadr` 与 actuator / `ctrl` 索引。
3. **单位、零位与正方向**：用单关节小步扫描验证弧度/角度、offset 和 sign。
4. **传动与耦合**：一个 actuator 可能驱动多个关节；一个关节也不一定对应一个独立 actuator。
5. **限位来源**：从已加载模型或硬件手册读取 `jnt_range` / `ctrlrange`，不要沿用本页占位数值。
6. **增益与左右手**：用标定集估计；未验证的 `1.60` 或固定轴镜像都不能冒充通用参数。

---

## 常见工程问题与解决方案

### 问题 1：关节振荡

**症状**：机器人手指快速来回抖动

**原因**：
- 视觉捕捉噪声导致 landmarks 抖动
- 控制频率过高，微小变化被放大

**解决**：
```python
# 时域滤波（指数移动平均）
alpha = 0.3  # 平滑系数
smoothed_joints = alpha * current_joints + (1 - alpha) * prev_joints
```

### 问题 2：关节空间插值失真

**症状**：在关节空间做三次样条插值后，任务空间（指尖位置）轨迹不自然

**原因**：关节空间非线性，插值后 FK 结果与预期不符

**解决**：
```python
# 正确做法：先对 landmarks 插值，再通过 IK 生成关节角
interpolated_landmarks = spline_interpolate(landmarks_sequence)
for lm in interpolated_landmarks:
    joints = ik_solver.solve(lm)
    robot.set_joints(joints)
```

### 问题 3：拇指校准不准

**症状**：拇指 MCP 角度与其他手指不协调

**原因**：Vector Optimization 方法中拇指自由度复杂，容易陷入局部最优

**解决**：
- 对拇指使用 Rule-based 映射，其他手指使用 Vector Optimization
- 增加拇指的优化权重
- 使用多初始点优化

### 问题 4：手掌漂移

**症状**：停止运动后，机器人手掌位置缓慢漂移

**原因**：freejoint 的速度未清零

**解决**：
```python
# freejoint 的 qpos 有 7 项（位置 3 + 四元数 4），qvel 有 6 项。
joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "palm_free")
if joint_id < 0:
    raise ValueError("joint 'palm_free' not found")
qpos_adr = model.jnt_qposadr[joint_id]  # qpos 地址
dof_adr = model.jnt_dofadr[joint_id]    # qvel 地址
data.qpos[qpos_adr:qpos_adr + 3] = target_position
data.qvel[dof_adr:dof_adr + 6] = 0.0
```

---

## 双手系统注意事项

### UDP 数据格式

左右手 landmarks 在同一 UDP 包中发送：

```python
packet = {
    "left_landmarks": [[x, y, z], ...],   # 21 点
    "right_landmarks": [[x, y, z], ...],  # 21 点
}
```

### 端口管理

- 网页控制器：端口 8782
- UDP 数据：端口 9000
- 避免冲突

### 双手分别遵循目标合同

```python
# 左手
left_joints = map_to_ten_flexion_features(
    packet["left_landmarks"], left_joint_limits, source_hand="left", target_hand="left"
)

# 右手
right_joints = map_to_ten_flexion_features(
    packet["right_landmarks"], right_joint_limits, source_hand="right", target_hand="right"
)
```
