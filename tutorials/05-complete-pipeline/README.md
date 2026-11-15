# Stage 5: 从 0 到 1 的完整重定向流水线

> **目标**：理解从人手视觉捕捉到机器人灵巧手控制的完整链路，掌握每一步的前因后果和优化过程。

> 范围：本文是分模块教学，不是端到端或真机部署认证。相机标定、型号参数与硬件驱动未在本页完整提供；示意片段中的占位函数不能视为已实现。本次内容检查未开启相机、训练或运行硬件。先完成仿真验证，再单独审核硬件安全。

---

## 为什么需要完整流水线？

重定向不是一步到位的。从摄像头看到人手，到机器人手指动起来，中间需要经过 **7 个关键步骤**：

```
人手图像
  ↓  ① 视觉检测（MediaPipe / HaMeR）
21点检测结果（先识别坐标定义）
  ↓  ② 坐标合同、尺度与轴对齐
统一单位与掌面轴，手腕为原点
  ↓  ③ 左右手分离与模型映射
仅在明确约定的掌面轴中做需要的镜像
  ↓  ④ Retargeting 映射
人手关键点 → 机器人关节角度
  ↓  ⑤ 关节限幅与平滑
按实际模型逐关节限幅 + 时序滤波
  ↓  ⑥ 物理仿真验证
MuJoCo 仿真检查碰撞/穿透
  ↓  ⑦ 真实机器人部署
另行接入经审核的控制器与硬件安全机制
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

**核心问题**：先确认坐标是什么。MediaPipe 图像 landmarks 是归一化坐标，不是米制相机 3D；world landmarks 是以手部几何中心为原点的米制估计，也不直接给出机器人世界位姿。先选择并标定合适输入，再讨论平移和轴对齐。[官方输出定义](https://developers.google.com/edge/mediapipe/solutions/vision/hand_landmarker/python#handle_and_display_results)

```python
# 仅做去平移：输入须是统一尺度的 3D 点；不负责估计相机深度或旋转对齐
import numpy as np

def convert_to_local_frame(landmarks_21x3):
    """
    将 21 个 3D 点的原点移到手腕，保留输入的坐标轴与长度单位。

    landmarks_21x3: [21, 3] 的 numpy 数组

    为什么要这样做？
    - 减去手腕只去掉整体平移
    - 不会消除图像透视缩放、输入噪声或整体旋转
    - 要获得统一掌面坐标，还需构建并应用正交基
    """
    wrist = landmarks_21x3[0]  # 第 0 点是手腕
    local = landmarks_21x3 - wrist  # 平移到手腕为原点
    return local
```

### 5.1.2 为什么需要局部坐标系？

| 坐标系 | 特点 | 问题 |
|--------|------|------|
| **图像归一化点** | x/y 依赖图像尺寸与透视 | 不能直接当米；去平移不消除远近缩放 |
| **米制点去手腕平移** | 保留相对位置、输入轴与单位 | 仍含整体旋转、估计误差和人手尺寸差异 |

**尺度处理**：除以掌长可表达无量纲的相对形状，但不消除手指比例差异。送入米制机器人 FK 目标前还需乘目标机器人尺度并对齐坐标轴；不能把无量纲输出直接当米。

```python
def normalize_scale(local_landmarks):
    """
    用手掌长度做尺度归一化。

    为什么要归一化？
    - 不同人的整体掌长不同
    - 掌长归一化能去除一个整体尺度因子
    - 手指比例、活动范围与检测偏差仍需校准
    """
    # 手掌长度 = 手腕到中指 MCP 的距离
    palm_length = np.linalg.norm(local_landmarks[9])  # 中指 MCP 是第 9 点
    normalized = local_landmarks / (palm_length + 1e-8)
    return normalized, palm_length
```

---

## 5.2 左右手分离与镜像处理

### 5.2.1 为什么要镜像？

以下只是一种**约定好的掌面轴**示例，并非 MediaPipe 原始相机轴的通用规则：

```
右手: index @ +Y, pinky @ -Y
左手: index @ -Y, pinky @ +Y（需要 Y 轴镜像）
```

只有先对齐到上述掌面轴、且机器人映射采用相应左右对称约定，才适用下面的 Y 反射。镜像不是旋转；带拇指不对称或不同关节轴的模型仍需分别校准。

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
def pack_dual_hand_data(left_landmarks, right_landmarks, frame_id, timestamp_ns):
    """
    将左右手 landmarks 打包到一个 UDP 包中发送。

    格式：{"left_landmarks": [[21, 3]], "right_landmarks": [[21, 3]]}

    为什么要打包在一起？
    - 携带同一采集帧的左右手检测；调用者须保证输入已在约定掌面轴中
    - 单包不保证采集同步，接收端还需验证帧号、时间戳与新鲜度
    """
    import json

    left_local = convert_to_local_frame(left_landmarks)
    left_local = mirror_left_hand(left_local)
    left_norm, _ = normalize_scale(left_local)

    right_local = convert_to_local_frame(right_landmarks)
    # 右手不需要镜像
    right_norm, _ = normalize_scale(right_local)

    packet = {
        "frame_id": frame_id,
        "timestamp_ns": timestamp_ns,
        "units": "palm_length_normalized",
        "left_landmarks": left_norm.tolist(),
        "right_landmarks": right_norm.tolist(),
    }
    return json.dumps(packet)
```

---

## 5.3 Retargeting 映射：从 21 点到关节角度

### 5.3.1 方法演进路线

这是重定向的**核心问题**，有三种常见方法；效果没有固定递增顺序，须在同一数据、机器人和误差协议下比较：

```
┌─────────────────────────────────────────────────────────────────┐
│  方法 1: Rule-based（直接角度映射）                                │
│  ─────────────────────────────────                               │
│  思路：根据相邻关键点夹角直接计算关节角度                           │
│  优点：简单、实时、无需优化                                        │
│  缺点：角度相似不保证不同尺寸手的指尖位置相同                        │
│  适用：快速原型验证                                               │
│                                                                  │
│  方法 2: Vector Optimization（向量优化）                           │
│  ─────────────────────────────────                               │
│  思路：优化已标定的关键点位置或相对向量目标                          │
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
    教学用 10 关节映射，不是厂商 O10 关节合同。

    本例假设每指 2 个目标变量；具体型号的主动关节、耦合与轴向需查模型。

    为什么要这样映射？
    - 人手有关节（MCP, PIP, DIP）；TIP 是指尖关键点，不是关节
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

        # 教学假设：每指 2 个变量，名称不代表真实执行器顺序
        # 将 curl 映射到关节角度 [0, 1.2] rad
        joint_angles[f"{finger_name}_mcp"] = curl * 1.2
        joint_angles[f"{finger_name}_pip"] = curl * 1.2 * 0.8  # PIP 略小

    return joint_angles
```

### 5.3.3 Rule-based 的问题与优化

**问题 1：尺寸差异**

人手和机器人手尺寸不同。直接角度映射会导致"手指够不到"或"手指过度弯曲"。

**待验证的校准方向**：检查检测与模型后，再测量映射增益。下面是部分代码骨架，`angle1/angle2` 来自前面的角度计算；数值不是安全推荐值，也没有附带实验日志。

```python
# 教学调参例子，不能由这些数值推断任何型号已校准成功

# 优化后的映射
def optimized_rule_based_retarget(landmarks_21x3):
    """
    优化后的 Rule-based retargeting。

    关键调整：
    1. 归一化分母从 1.45 改为 0.95（补偿衰减）
    2. actuator 缩放系数从 1.25 改为 1.60（可能更早饱和，需重新验证）
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

拇指的对掌运动涉及多个关节与耦合；不能用固定“3 个自由度”概括所有解剖或机器人模型。简单弯曲角映射可能遗漏对掌与外展信息。

**优化**：对拇指使用独立的映射逻辑，或改用 Vector Optimization。

---

## 5.4 Vector Optimization 方法详解

### 5.4.1 核心思想

不直接映射角度，而是**最小化指尖在任务空间的位置误差**：

```
min ||f_robot(joints) - target_landmark||^2
```

其中 `f_robot` 是机器人 FK，目标必须已映射到同一坐标系与单位。此式是位置残差；相对向量优化则比较成对关键点之差。优化器不会自动推断人手到机器人之间的标定。

### 5.4.2 如何比较两种方法？

| 对比维度 | Rule-based | Vector Optimization |
|---------|-----------|---------------------|
| 尺寸差异 | 可标定映射，但不保证任务空间位置 | 需先映射尺度与目标，且目标未必可达 |
| 缺失关节 | 需要规则与耦合模型 | 需要目标权重、约束与模型选择 |
| 计算成本 | 依赖关键点数和规则 | 依赖 FK、Jacobian、维数和迭代次数 |
| 实时性 | 测量计算及端到端延迟 | 使用相同硬件/输入协议测量 |
| 稳定性 | 受检测噪声和增益影响 | 受初值、目标、约束与求解器影响 |

### 5.4.3 带边界的非线性最小二乘

下面用 SciPy TRF；`method='lm'` 不支持 bounds，会直接报错，不能把它与带限位的 TRF 混写。`finger_chain` 仍是需要具体实现的 FK 接口，示例不含机器人模型。[SciPy 文档](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.least_squares.html)

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

    优化方法：Trust Region Reflective（TRF），不是 LM。
    """
    def residuals(joints):
        # 正运动学：计算当前关节角度下的指尖位置
        current_tips = finger_chain.forward_kinematics(joints)
        # 残差 = 当前位置 - 目标位置
        return (current_tips - target_landmarks).flatten()

    # 本例边界只是教学假设，实际需传入逐关节模型边界
    result = least_squares(
        residuals,
        initial_joints,
        method='trf',  # 支持 bounds 的求解器
        ftol=1e-6,
        max_nfev=100,
        bounds=(0, 1.2),  # 教学边界；initial_joints 必须在边界内
    )

    if not result.success:
        raise RuntimeError(result.message)
    return result.x  # 仍须检查实际任务残差；数值终止不等于目标达到
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
   ↓ 记录：实际残差、失败率、求解耗时与端到端延迟
```

---

## 5.5 关节限幅与时序平滑

### 5.5.1 为什么要限幅？

机器人关节有不同的物理限制。下面统一 `[0, 1.2]` rad 只是教学例子，不是 O10 或其他型号的真实逐关节范围。限位裁剪也不能防止所有碰撞，须另查执行器力、速度与接触。

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
    - 检测噪声需在指定距离、遮挡与标定协议下测量，不能固定假设为 ±2mm
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

关节空间与任务空间插值各有适用目标，不能简单称一种错误、另一种正确。FK 非线性使平滑关节轨迹未必对应期望指尖路径；任务空间插值也可能不可达、越界或引起 IK 跳支。三次样条还可能过冲，两者都需重新检查约束；下面是离线处理，不是因果实时滤波。

```python
# 方案 A：关节空间插值，随后检查任务空间轨迹
def joint_space_interpolation(joint_seq):
    from scipy.interpolate import CubicSpline
    t = np.arange(len(joint_seq))
    cs = CubicSpline(t, joint_seq)
    return cs(np.linspace(0, len(joint_seq)-1, 100))
    # 问题：关节空间的平滑 ≠ 任务空间的平滑

# 方案 B：已标定的 landmark 空间插值，再用模型 IK
def task_space_interpolation(landmark_seq, initial_joints, finger_chain):
    from scipy.interpolate import CubicSpline
    t = np.arange(len(landmark_seq))
    cs = CubicSpline(t, landmark_seq, axis=0)
    smooth_landmarks = cs(np.linspace(0, len(landmark_seq)-1, 100))

    # 对每帧 smooth_landmarks 做 IK
    joint_seq = []
    q = np.asarray(initial_joints, dtype=float)
    for lm in smooth_landmarks:
        q = vector_retarget(lm, q, finger_chain)
        joint_seq.append(q.copy())
    return joint_seq
```

---

## 5.6 物理仿真验证

几何与数值仿真验证只支持当前模型/配置，不能推出真机安全或检测器绝对精度。

### 5.6.1 为什么需要仿真？

在部署到真实机器人之前，先在 MuJoCo 中验证：
- 关节角度是否会导致手指穿透物体
- 手掌位姿估计是否准确
- 整体运动是否稳定

### 5.6.2 手掌位姿估计

```python
def estimate_palm_pose(landmarks_21x3):
    """
    从已标定的米制点构建手掌坐标；原始图像点不能给出绝对世界位置。

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

仿真初始化可以同时重置位姿与速度，但这不是硬件急停，也不能阻止下一步受重力或外力继续运动。freejoint 用 7 个 qpos 数（位置 3 + 四元数 4）、6 个 qvel 数；两个起始地址必须分别取 `jnt_qposadr` 与 `jnt_dofadr`。[MuJoCo 数据结构](https://mujoco.readthedocs.io/en/stable/APIreference/APItypes.html)

```python
import mujoco

def reset_hand_position(model, data, joint_id, target_pos, target_quat):
    """
    仅重置指定 freejoint 的仿真状态；不会保证后续动力学停止。
    """
    if model.jnt_type[joint_id] != mujoco.mjtJoint.mjJNT_FREE:
        raise ValueError("joint_id 必须指向 freejoint")
    qpos_adr = model.jnt_qposadr[joint_id]
    dof_adr = model.jnt_dofadr[joint_id]
    quat = np.asarray(target_quat, dtype=float)  # MuJoCo 顺序 w, x, y, z
    if quat.shape != (4,) or not np.isfinite(quat).all() or np.linalg.norm(quat) == 0:
        raise ValueError("四元数必须是有限非零的 4 维向量")
    # 重置位置
    data.qpos[qpos_adr:qpos_adr+3] = target_pos
    # 重置旋转（四元数）
    data.qpos[qpos_adr+3:qpos_adr+7] = quat / np.linalg.norm(quat)
    # 重置速度（关键！防止漂移）
    data.qvel[dof_adr:dof_adr+6] = 0.0
    mujoco.mj_forward(model, data)
```

---

## 5.7 真实机器人部署

本节只是接口需求清单；下面函数为空桩，未实现 GeoRT/ROS2 驱动，也没有真机执行证据。不能直接据此部署。具体第三方框架需另查其官方 API、许可证与型号支持。

### 5.7.1 接入 GeoRT

接入外部遥操作框架通常需要核对：

1. **URDF 文件**：将 MJCF 模型转换为 URDF
2. **配置文件**：在 `geort/config/` 下创建 JSON，定义 `joint_order` 等参数
3. **控制接口**：复用 `init_hand()` 和 `set_hand_position()` 方法

```python
# 未实现的接口占位，不是可执行部署入口
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

以下是待测量的调度预算示例，不是本教程已复现的性能。算法内部耗时不能当作相机到机器人端到端延迟。

| 模块 | 频率 | 说明 |
|------|------|------|
| 视觉检测（MediaPipe） | 30 Hz | 摄像头帧率限制 |
| Retargeting（Rule-based） | 100 Hz 目标 | 需测量输入与计算耗时 |
| Retargeting（Vector Opt） | 25 Hz 目标 | 需记录收敛率与迭代耗时 |
| 机械臂 IK | 25 Hz | damping=0.06, iterations=5 |
| 灵巧手控制 | 50 Hz | 平滑后发送 |

---

## 5.8 完整 Pipeline 代码

参考 [`examples/complete_retargeting_pipeline.py`](../../examples/complete_retargeting_pipeline.py)。

---

## 5.9 常见问题与解决方案

### Q1: 手指弯曲不到位？

**可能原因**：检测坐标/比例、零位、执行器增益、关节顺序或限位不一致；归一化分母只是候选之一。

**排查**：先记录输入角度与实际关节反馈，再在仿真逐项标定，不直接套用放大系数。

### Q2: 拇指总是不准？

**可能原因**：拇指轴、耦合或对掌目标未建模；需要结合具体手模型检查。

**解决**：拇指改用 Vector Optimization，或对拇指独立调参。

### Q3: 左右手同时运行时抖动？

**原因**：左右手 landmarks 未做时间同步，或平滑系数不够。

**排查**：先检查同一采集时间、帧号与陈旧数据。对本文 `alpha*current+(1-alpha)*prev` 定义，**减小 alpha** 才更平滑，但会增加滞后；打包本身不保证同步。

### Q4: 仿真中手指穿透物体？

**原因**：关节角度超出合理范围，或碰撞检测参数设置不当。

**解决**：加强关节限幅，检查 MuJoCo 碰撞参数（margin, gap）。

### Q5: Stop 后手掌漂移？

**可能原因**：残余速度、重力、外力、接触或继续更新的控制命令。

**排查**：初始化时按正确地址清零速度并刷新状态；若下一步仍受力，需设计保持/约束策略。不要把仿真状态清零等同急停。

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
- [`examples/complete_retargeting_pipeline.py`](../../examples/complete_retargeting_pipeline.py) — 教学实现入口，外部接入与安全能力需另验
