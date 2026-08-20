# Stage 5: 从 0 到 1 的完整重定向流水线

> **目标**：理解从人手视觉捕捉到机器人灵巧手控制的完整链路，掌握每一步的前因后果和优化过程。

---

## 为什么需要完整流水线？

重定向不是一步到位的。从摄像头看到人手，到机器人手指动起来，中间需要经过 **7 个关键步骤**：

```
人手图像
  ↓  ① 视觉检测（MediaPipe / HaMeR）
21点 3D 坐标（局部坐标系）
  ↓  ② 左右手分离与镜像
左手（Y镜像）+ 右手（原始）
  ↓  ③ 坐标归一化与对齐
手腕为原点的局部坐标
  ↓  ④ Retargeting 映射
人手关键点 → 机器人关节角度
  ↓  ⑤ 关节限幅与平滑
[0, 1.2] rad 限幅 + 时序滤波
  ↓  ⑥ 物理仿真验证
MuJoCo 仿真检查碰撞/穿透
  ↓  ⑦ 真实机器人部署
GeoRT / ROS2 控制真机
```

每一步都有**为什么要做**和**怎么做得更好**的问题。本教程带你逐层深入。

---

## 前置要求

- 完成 Stage 1-4（FK/IK 基础、Rule-based、Vector Optimization、Landmark Pipeline）
- 了解 MediaPipe 21 点手部模型
- 了解 O10 灵巧手（10 DOF）的基本结构

---

## 5.1 视觉检测：从图像到 21 点

### 5.1.1 MediaPipe 方案（入门首选）

MediaPipe Hands 提供 21 个 3D 关键点：

```
0: 手腕 (WRIST)
1-4: 拇指 (THUMB_CMC, THUMB_MCP, THUMB_IP, THUMB_TIP)
5-8: 食指 (INDEX_FINGER_MCP, PIP, DIP, TIP)
9-12: 中指
13-16: 无名指
17-20: 小指
```

**核心问题**：MediaPipe 输出的是**相机坐标系**下的 3D 坐标，如何转换到以手腕为原点的**局部坐标系**？

```python
# 关键转换：相机坐标系 → 局部坐标系
import numpy as np

def convert_to_local_frame(landmarks_21x3):
    """
    将 MediaPipe 输出的 21 个 3D 点从相机坐标系转换到以手腕为原点的局部坐标系。

    landmarks_21x3: [21, 3] 的 numpy 数组

    为什么要这样做？
    - 相机坐标系下，手的位置随摄像头距离变化
    - 局部坐标系下，只有手指相对手腕的姿态，与距离无关
    - 这样 retargeting 只关心"手指怎么弯"，不关心"手在哪里"
    """
    wrist = landmarks_21x3[0]  # 第 0 点是手腕
    local = landmarks_21x3 - wrist  # 平移到手腕为原点
    return local
```

### 5.1.2 为什么需要局部坐标系？

| 坐标系 | 特点 | 问题 |
|--------|------|------|
| **相机坐标系** | 绝对位置，包含深度 | 手离摄像头远，坐标值大；retargeting 会受距离影响 |
| **局部坐标系** | 相对手腕的位置 | 只保留姿态信息，与距离无关，retargeting 更稳定 |

**优化**：在局部坐标系中进一步做**尺度归一化**（除以手掌长度），让不同人手尺寸都能映射到同一个机器人手。

```python
def normalize_scale(local_landmarks):
    """
    用手掌长度做尺度归一化。

    为什么要归一化？
    - 大人手 vs 小孩手，绝对尺寸差 2-3 倍
    - 但"手指弯曲比例"是一样的
    - 归一化后，retargeting 对所有人手一视同仁
    """
    # 手掌长度 = 手腕到中指 MCP 的距离
    palm_length = np.linalg.norm(local_landmarks[9])  # 中指 MCP 是第 9 点
    normalized = local_landmarks / (palm_length + 1e-8)
    return normalized, palm_length
```

---

## 5.2 左右手分离与镜像处理

### 5.2.1 为什么要镜像？

双手系统中，左右手的坐标系定义不同：

```
右手: index @ +Y, pinky @ -Y
左手: index @ -Y, pinky @ +Y（需要 Y 轴镜像）
```

**如果不镜像**：左右手的食指和小指方向会颠倒，导致左手 retargeting 完全错误。

```python
def mirror_left_hand(local_landmarks):
    """
    左手 Y 轴镜像，使左右手在局部坐标系中具有相同的语义。

    镜像前：左手 index @ -Y, pinky @ +Y
    镜像后：左手 index @ +Y, pinky @ -Y（与右手一致）

    为什么要这样做？
    - 让左右手共享同一个 retargeting 映射函数
    - 不需要为左右手分别训练/调参
    """
    mirrored = local_landmarks.copy()
    mirrored[:, 1] *= -1  # Y 轴取反
    return mirrored
```

### 5.2.2 双手数据打包

```python
def pack_dual_hand_data(left_landmarks, right_landmarks):
    """
    将左右手 landmarks 打包到一个 UDP 包中发送。

    格式：{"left_landmarks": [[21, 3]], "right_landmarks": [[21, 3]]}

    为什么要打包在一起？
    - 保证左右手数据的时间同步
    - 避免左右手分别传输导致的时序错位
    """
    import json

    left_local = convert_to_local_frame(left_landmarks)
    left_local = mirror_left_hand(left_local)
    left_norm, _ = normalize_scale(left_local)

    right_local = convert_to_local_frame(right_landmarks)
    # 右手不需要镜像
    right_norm, _ = normalize_scale(right_local)

    packet = {
        "left_landmarks": left_norm.tolist(),
        "right_landmarks": right_norm.tolist(),
    }
    return json.dumps(packet)
```

---

## 5.3 Retargeting 映射：从 21 点到关节角度

### 5.3.1 方法演进路线

这是重定向的**核心问题**，有三种方法，效果递增：

```
┌─────────────────────────────────────────────────────────────────┐
│  方法 1: Rule-based（直接角度映射）                                │
│  ─────────────────────────────────                               │
│  思路：根据相邻关键点夹角直接计算关节角度                           │
│  优点：简单、实时、无需优化                                        │
│  缺点：无法补偿人手与机器人手的尺寸差异                              │
│  适用：快速原型验证                                               │
│                                                                  │
│  方法 2: Vector Optimization（向量优化）                           │
│  ─────────────────────────────────                               │
│  思路：最小化指尖位置误差，用 IK 求解关节角度                        │
│  优点：任务空间精确，可处理尺寸差异                                │
│  缺点：计算成本高，需要好的初值                                    │
│  适用：高精度场景                                                 │
│                                                                  │
│  方法 3: Learning-based（学习映射）                                │
│  ─────────────────────────────────                               │
│  思路：用神经网络学习 landmarks → joint angles 的映射               │
│  优点：端到端，可学习复杂非线性映射                                │
│  缺点：需要大量标注数据，泛化性依赖数据分布                          │
│  适用：大规模部署                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 5.3.2 Rule-based 方法详解

```python
import numpy as np

def compute_finger_curl(landmarks, finger_indices):
    """
    计算手指弯曲角（相邻关键点向量夹角）。

    参数：
        landmarks: [21, 3] 局部坐标系下的关键点
        finger_indices: 该手指的关键点索引列表，如 [5,6,7,8] 表示食指

    返回：
        curl: 弯曲程度 [0, 1]，0=伸直，1=完全弯曲

    为什么要用夹角？
    - 手指弯曲时，相邻骨节之间的夹角会变化
    - 夹角可以直接反映"弯曲程度"
    """
    # 取该手指的 4 个关键点（MCP, PIP, DIP, TIP）
    pts = landmarks[finger_indices]

    # 计算相邻向量
    v1 = pts[1] - pts[0]  # MCP → PIP
    v2 = pts[2] - pts[1]  # PIP → DIP
    v3 = pts[3] - pts[2]  # DIP → TIP

    # 计算夹角（弯曲角）
    def angle_between(v_a, v_b):
        cos = np.dot(v_a, v_b) / (np.linalg.norm(v_a) * np.linalg.norm(v_b) + 1e-8)
        return np.arccos(np.clip(cos, -1, 1))

    angle1 = angle_between(v1, v2)
    angle2 = angle_between(v2, v3)

    # 归一化为 [0, 1]
    max_angle = np.pi * 0.6  # 手指最大弯曲约 108 度
    curl = ((angle1 + angle2) / 2) / max_angle
    return np.clip(curl, 0, 1)


def rule_based_retarget(landmarks_21x3):
    """
    Rule-based retargeting：从 21 点 landmarks 映射到 O10 灵巧手 10 个关节角度。

    O10 灵巧手有 10 个主动关节（每指 2 个：MCP + PIP 耦合）。

    为什么要这样映射？
    - 人手有关节（MCP, PIP, DIP, TIP），机器人手也有关节
    - 但机器人关节数通常少于人手（O10 是 10 DOF，人手是 20+ DOF）
    - 所以需要"合并"一些人手关节信息到同一个机器人关节
    """
    # 各手指的关键点索引（MediaPipe 格式）
    FINGER_INDICES = {
        "thumb": [1, 2, 3, 4],
        "index": [5, 6, 7, 8],
        "middle": [9, 10, 11, 12],
        "ring": [13, 14, 15, 16],
        "pinky": [17, 18, 19, 20],
    }

    joint_angles = {}
    for finger_name, indices in FINGER_INDICES.items():
        curl = compute_finger_curl(landmarks_21x3, indices)

        # O10 每指 2 个关节：MCP 和 PIP（耦合）
        # 将 curl 映射到关节角度 [0, 1.2] rad
        joint_angles[f"{finger_name}_mcp"] = curl * 1.2
        joint_angles[f"{finger_name}_pip"] = curl * 1.2 * 0.8  # PIP 略小

    return joint_angles
```

### 5.3.3 Rule-based 的问题与优化

**问题 1：尺寸差异**

人手和机器人手尺寸不同。直接角度映射会导致"手指够不到"或"手指过度弯曲"。

**优化**：调整归一化分母和 actuator 缩放。

```python
# 从项目 memory 中总结的经验教训：
# 21点landmark到finger_curl的往返转换存在衰减
# 需要调整归一化分母（1.45→0.95）和actuator缩放系数（1.25→1.60）

# 优化后的映射
def optimized_rule_based_retarget(landmarks_21x3):
    """
    优化后的 Rule-based retargeting。

    关键调整：
    1. 归一化分母从 1.45 改为 0.95（补偿衰减）
    2. actuator 缩放系数从 1.25 改为 1.60（确保手势到位）
    """
    # ... 基础计算 ...

    # 调整 1：更激进的归一化
    max_angle = 0.95  # 原来是 1.45
    curl = ((angle1 + angle2) / 2) / max_angle
    curl = np.clip(curl, 0, 1)

    # 调整 2：更大的 actuator 缩放
    scale = 1.60  # 原来是 1.25
    joint_angle = curl * 1.2 * scale
    joint_angle = np.clip(joint_angle, 0, 1.2)  # 最终限幅

    return joint_angle
```

**问题 2：拇指的特殊性**

拇指有 3 个自由度（内收/外展 + 屈伸 + 旋转），与其他 4 指不同。简单角度映射无法捕捉拇指的复杂运动。

**优化**：对拇指使用独立的映射逻辑，或改用 Vector Optimization。

---

## 5.4 Vector Optimization 方法详解

### 5.4.1 核心思想

不直接映射角度，而是**最小化指尖在任务空间的位置误差**：

```
min ||f_robot(joints) - target_landmark||^2
```

其中 `f_robot` 是机器人手的正运动学（FK），`target_landmark` 是人手关键点位置。

### 5.4.2 为什么比 Rule-based 更好？

| 对比维度 | Rule-based | Vector Optimization |
|---------|-----------|---------------------|
| 补偿尺寸差异 | 不能 | 能（通过位置优化自动适应） |
| 处理缺失关节 | 困难 | 自然（优化器自动分配） |
| 计算成本 | 极低（O(1)） | 中等（每次迭代 O(n^2)） |
| 实时性 | 高（>100Hz） | 中（~25Hz） |
| 稳定性 | 高 | 依赖初值和阻尼参数 |

### 5.4.3 Damped Least Squares（DLS）实现

```python
import numpy as np
from scipy.optimize import least_squares

def vector_retarget(target_landmarks, initial_joints, finger_chain):
    """
    向量优化重定向。

    参数：
        target_landmarks: [N, 3] 目标指尖位置（来自人手）
        initial_joints: [n] 初始关节角度
        finger_chain: FingerChain3D 对象（包含 DH 参数和 FK）

    返回：
        optimal_joints: [n] 优化后的关节角度

    优化方法：Damped Least Squares（Levenberg-Marquardt）
    """
    def residuals(joints):
        # 正运动学：计算当前关节角度下的指尖位置
        current_tips = finger_chain.forward_kinematics(joints)
        # 残差 = 当前位置 - 目标位置
        return (current_tips - target_landmarks).flatten()

    # DLS 求解
    result = least_squares(
        residuals,
        initial_joints,
        method='lm',  # Levenberg-Marquardt
        ftol=1e-6,
        max_nfev=100,
        bounds=(0, 1.2),  # O10 关节限幅
    )

    return result.x
```

### 5.4.4 从 Rule-based 到 Vector Optimization 的演进

```
Step 1: Rule-based 快速验证（能跑起来）
   ↓ 发现问题：尺寸不匹配，手势不到位
Step 2: 调整归一化参数和缩放系数（经验调参）
   ↓ 发现问题：拇指总是不准，复杂手势无法复现
Step 3: 引入 Vector Optimization（任务空间精确匹配）
   ↓ 发现问题：计算慢，实时性不够
Step 4: 用 Rule-based 提供初值 + Vector 做精修（混合方案）
   ↓ 最终：25Hz 实时运行，精度满足要求
```

---

## 5.5 关节限幅与时序平滑

### 5.5.1 为什么要限幅？

机器人关节有物理限制。O10 灵巧手的关节范围是 `[0, 1.2]` rad（约 0° 到 68°）。

**如果不限幅**：
- 仿真中会出现关节穿透、非法姿态
- 真机上可能损坏硬件
- 优化器可能发散到不可行区域

```python
def clamp_joints(joints, min_val=0.0, max_val=1.2):
    """关节限幅。"""
    return np.clip(joints, min_val, max_val)
```

### 5.5.2 为什么要时序平滑？

视觉检测有噪声（每帧关键点位置会抖动）。直接输出会导致机器人手指抖动。

**解决方案**：

```python
class TemporalSmoother:
    """
    时序平滑器：用指数移动平均（EMA）消除抖动。

    为什么要平滑？
    - MediaPipe 每帧输出有 ±2mm 的抖动
    - 直接输出会导致机器人手指高频振动
    - EMA 在响应速度和平滑度之间取得平衡
    """

    def __init__(self, alpha=0.3):
        self.alpha = alpha  # 平滑系数，越小越平滑
        self.prev = None

    def smooth(self, current):
        if self.prev is None:
            self.prev = current
            return current

        smoothed = self.alpha * current + (1 - self.alpha) * self.prev
        self.prev = smoothed
        return smoothed
```

### 5.5.3 插值优化

**关键教训（来自项目 memory）**：

> 在关节空间（ctrl值）进行三次样条插值会导致任务空间（指尖位置）失真。应先对 landmarks 进行插值，再通过 IK 生成关节角度。

```python
# 错误做法：在关节空间插值
def wrong_interpolation(joint_seq):
    from scipy.interpolate import CubicSpline
    t = np.arange(len(joint_seq))
    cs = CubicSpline(t, joint_seq)
    return cs(np.linspace(0, len(joint_seq)-1, 100))
    # 问题：关节空间的平滑 ≠ 任务空间的平滑

# 正确做法：在 landmark 空间插值，再用 IK
def correct_interpolation(landmark_seq):
    from scipy.interpolate import CubicSpline
    t = np.arange(len(landmark_seq))
    cs = CubicSpline(t, landmark_seq, axis=0)
    smooth_landmarks = cs(np.linspace(0, len(landmark_seq)-1, 100))

    # 对每帧 smooth_landmarks 做 IK
    joint_seq = [vector_retarget(lm) for lm in smooth_landmarks]
    return joint_seq
```

---

## 5.6 物理仿真验证

### 5.6.1 为什么需要仿真？

在部署到真实机器人之前，先在 MuJoCo 中验证：
- 关节角度是否会导致手指穿透物体
- 手掌位姿估计是否准确
- 整体运动是否稳定

### 5.6.2 手掌位姿估计

```python
def estimate_palm_pose(landmarks_21x3):
    """
    从 21 点估计手掌位姿（位置 + 旋转）。

    方法：用手腕 + 3 个 MCP 关键点构建正交基。

    为什么要估计手掌位姿？
    - 机器人手臂需要知道手掌在哪里、朝向哪个方向
    - 只有手指角度不够，还需要手掌的 6D 位姿
    """
    wrist = landmarks_21x3[0]
    index_mcp = landmarks_21x3[5]
    middle_mcp = landmarks_21x3[9]
    pinky_mcp = landmarks_21x3[17]

    # 构建正交基
    x_axis = index_mcp - wrist
    x_axis = x_axis / np.linalg.norm(x_axis)

    y_temp = pinky_mcp - wrist
    z_axis = np.cross(x_axis, y_temp)
    z_axis = z_axis / np.linalg.norm(z_axis)

    y_axis = np.cross(z_axis, x_axis)

    rotation_matrix = np.stack([x_axis, y_axis, z_axis], axis=1)
    return wrist, rotation_matrix
```

### 5.6.3 Stop 后的漂移问题

**关键教训（来自项目 memory）**：

> 手掌位置控制需同时重置 freejoint 位置和速度（`data.qvel[dof_adr:dof_adr+6] = 0.0`），以防止 Stop 后漂移。

```python
def reset_hand_position(data, body_id, target_pos, target_quat, dof_adr):
    """
    重置手掌位置，防止 Stop 后漂移。

    为什么需要同时重置位置和速度？
    - MuJoCo 中 freejoint 有 6 个自由度（3 位置 + 3 旋转）
    - 如果只重置位置，残余速度会导致漂移
    - 同时重置位置和速度才能完全停止
    """
    # 重置位置
    data.qpos[dof_adr:dof_adr+3] = target_pos
    # 重置旋转（四元数）
    data.qpos[dof_adr+3:dof_adr+7] = target_quat
    # 重置速度（关键！防止漂移）
    data.qvel[dof_adr:dof_adr+6] = 0.0
```

---

## 5.7 真实机器人部署

### 5.7.1 接入 GeoRT

GeoRT 是一个基于 PyBullet 的遥操作框架。接入需要：

1. **URDF 文件**：将 MJCF 模型转换为 URDF
2. **配置文件**：在 `geort/config/` 下创建 JSON，定义 `joint_order` 等参数
3. **控制接口**：复用 `init_hand()` 和 `set_hand_position()` 方法

```python
# 参考项目 memory 中的经验
def deploy_to_geort(joint_angles, urdf_path, config_path):
    """
    将重定向结果部署到 GeoRT。

    步骤：
    1. 加载 URDF（从 MJCF 转换而来）
    2. 读取配置文件（定义 joint_order）
    3. 按 joint_order 排列关节角度
    4. 发送控制指令
    """
    # ... 实现细节参考 Omnihand_o10_yudie.py ...
    pass
```

### 5.7.2 运行频率

| 模块 | 频率 | 说明 |
|------|------|------|
| 视觉检测（MediaPipe） | 30 Hz | 摄像头帧率限制 |
| Retargeting（Rule-based） | >100 Hz | 纯计算，无瓶颈 |
| Retargeting（Vector Opt） | ~25 Hz | DLS 迭代求解 |
| 机械臂 IK | 25 Hz | damping=0.06, iterations=5 |
| 灵巧手控制 | 50 Hz | 平滑后发送 |

---

## 5.8 完整 Pipeline 代码

参考 [`examples/complete_retargeting_pipeline.py`](../../examples/complete_retargeting_pipeline.py)。

---

## 5.9 常见问题与解决方案

### Q1: 手指弯曲不到位？

**原因**：Rule-based 映射的归一化分母太大，导致输出角度偏小。

**解决**：将归一化分母从 1.45 改为 0.95，actuator 缩放从 1.25 改为 1.60。

### Q2: 拇指总是不准？

**原因**：拇指有 3 个自由度，Rule-based 的 2D 角度映射无法捕捉。

**解决**：拇指改用 Vector Optimization，或对拇指独立调参。

### Q3: 左右手同时运行时抖动？

**原因**：左右手 landmarks 未做时间同步，或平滑系数不够。

**解决**：确保左右手在同一个 UDP 包中发送，加大 EMA 平滑系数。

### Q4: 仿真中手指穿透物体？

**原因**：关节角度超出合理范围，或碰撞检测参数设置不当。

**解决**：加强关节限幅，检查 MuJoCo 碰撞参数（margin, gap）。

### Q5: Stop 后手掌漂移？

**原因**：只重置了位置，没重置速度。

**解决**：同时设置 `data.qvel[dof_adr:dof_adr+6] = 0.0`。

---

## 学习路线总结

```
Stage 1: FK/IK 基础 → 理解关节空间与任务空间
  ↓
Stage 2: Rule-based → 快速验证，掌握角度映射
  ↓
Stage 3: Vector Optimization → 解决精度问题，掌握 DLS
  ↓
Stage 4: Landmark Pipeline → 理解从视觉到控制的完整链路
  ↓
Stage 5: 完整 Pipeline → 掌握前因后果、优化过程、部署细节
  ↓
进阶: 复现 AnyTeleop / HaMeR / LEAP Hand 等开源项目
```

## 推荐阅读

- [`docs/01-what-is-ik-retargeting.md`](../../docs/01-what-is-ik-retargeting.md) — 核心概念
- [`docs/03-human-hand-to-robot-hand.md`](../../docs/03-human-hand-to-robot-hand.md) — 人手→机器人手映射详解
- [`docs/04-optimization-methods.md`](../../docs/04-optimization-methods.md) — Jacobian 与 DLS 理论
- [`docs/08-open-source-projects.md`](../../docs/08-open-source-projects.md) — 优质开源项目推荐
- [`examples/complete_retargeting_pipeline.py`](../../examples/complete_retargeting_pipeline.py) — 完整可运行代码