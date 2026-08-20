# 人手到机器人手映射详解

> 从视觉捕捉的 21 点坐标到机器人关节角的完整 pipeline，包括坐标系转换、左右手镜像、关节限位等工程细节。

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

### 步骤 1：转换到手腕局部坐标系

视觉捕捉的 landmarks 通常在相机坐标系下，需要转换到以手腕为原点的局部坐标系。

```python
def to_local_coordinates(landmarks):
    """
    将 landmarks 从相机坐标系转换到手腕局部坐标系

    Args:
        landmarks: [21, 3] 3D 坐标

    Returns:
        local_landmarks: [21, 3] 以手腕为原点的坐标
    """
    wrist = landmarks[0]
    local = landmarks - wrist  # 平移：手腕 → 原点
    return local
```

### 步骤 2：尺度归一化

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

### 步骤 3：左右手镜像

双手系统中，左手需要进行 Y 轴镜像，确保左右手在局部坐标系中对称。

```
左手: index @ -Y, pinky @ +Y  →  Y 取反后: index @ +Y, pinky @ -Y
右手: index @ +Y, pinky @ -Y  →  无需处理
```

```python
def mirror_left_hand(landmarks, is_left):
    """
    左手镜像处理

    Args:
        landmarks: [21, 3] 局部坐标系下的坐标
        is_left: 是否为左手
    """
    if is_left:
        landmarks = landmarks.copy()
        landmarks[:, 1] *= -1  # Y 轴取反
    return landmarks
```

### 完整坐标转换 Pipeline

```python
def preprocess_landmarks(landmarks, is_left):
    """
    完整的 landmarks 预处理
    """
    # 1. 局部坐标系
    local = to_local_coordinates(landmarks)

    # 2. 尺度归一化
    normalized = normalize_scale(local)

    # 3. 左右手镜像
    mirrored = mirror_left_hand(normalized, is_left)

    return mirrored
```

---

## 从 Landmarks 计算角度特征

### 弯曲角（Flexion Angle）

相邻关键点向量之间的夹角，反映手指卷曲程度。

```python
def compute_flexion_angle(landmarks, joint_indices):
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

    v1 = p1 - p2
    v2 = p3 - p2

    cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-8)
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

## 人手到 O10 灵巧手的映射

O10 灵巧手有 10 个主动关节（每根手指 2 个：MCP 弯曲 + PIP 弯曲）。

### Rule-based 映射

```python
def map_to_o10(human_landmarks, is_left=False):
    """
    将人手 21 点坐标映射到 O10 关节角

    Returns:
        joint_angles: [10] 关节角（弧度）
    """
    landmarks = preprocess_landmarks(human_landmarks, is_left)

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

    # 经验：调整归一化分母和缩放系数以确保手势到位
    # 21点 → finger_curl 往返转换存在衰减，需要补偿
    scale_factor = 1.60  # 经验值
    joints = joints * scale_factor

    # 裁剪到 O10 关节范围
    o10_limits = np.array([
        [0.0, 1.2], [0.0, 1.2],   # 拇指
        [0.0, 1.2], [0.0, 1.2],   # 食指
        [0.0, 1.2], [0.0, 1.2],   # 中指
        [0.0, 1.2], [0.0, 1.2],   # 无名指
        [0.0, 1.2], [0.0, 1.2],   # 小指
    ])
    joints = np.clip(joints, o10_limits[:, 0], o10_limits[:, 1])

    return joints
```

### 关键工程细节

1. **归一化分母调整**：从 1.45 调到 0.95，补偿 landmark 到 curl 的衰减
2. **Actuator 缩放系数**：从 1.25 调到 1.60，确保手势到位
3. **关节限位**：O10 每个关节都有严格的角度范围，必须裁剪
4. **左右手镜像**：左手 Y 轴取反后再计算角度

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
# 重置位置和速度
data.qpos[dof_adr:dof_adr+3] = target_position
data.qvel[dof_adr:dof_adr+6] = 0.0  # 关键：速度清零
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

### 双手镜像对称

```python
# 左手
left_joints = map_to_o10(packet["left_landmarks"], is_left=True)

# 右手
right_joints = map_to_o10(packet["right_landmarks"], is_left=False)
```
