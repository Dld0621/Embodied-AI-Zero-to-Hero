# Retargeting 方法分类体系

> **示例待修：**本页分段映射不连续，带关节限位的 LM 示例不适用。请勿直接照抄运行；先学习 [新版 FK / Jacobian / IK](foundations/07-fk-jacobian-ik.md)。具体修订任务见 [审查交接 H02](reviews/content-audit-handoff.md)。

> 三种主流 retargeting 方法：Rule-based、Vector Optimization、Learning-based，以及它们的适用场景与 trade-off。

---

## 方法分类总览

```
Retargeting Methods
├── Rule-based（基于规则）
│   ├── 直接角度映射
│   ├── 分段线性映射
│   └── 关节限位裁剪
├── Vector Optimization（向量优化）
│   ├── 任务空间 IK
│   ├── Scipy least_squares
│   └── 关节空间 IK
└── Learning-based（基于学习）
    ├── Neural Network 映射
    ├── VAE / CVAE
    └── Diffusion Policy
```

---

## 1. Rule-based 方法

### 1.1 直接角度映射

**思想**：将人手的关节角度直接或按比例映射到机器人关节。

```python
def direct_angle_mapping(human_joint, scale=1.0, offset=0.0):
    """
    人手关节角 → 机器人关节角
    """
    robot_joint = human_joint * scale + offset
    # 限位裁剪
    robot_joint = np.clip(robot_joint, joint_min, joint_max)
    return robot_joint
```

**优点**：
- 简单、实时性好
- 无需训练数据
- 易于调试

**缺点**：
- 泛化性差：换一个操作者或机器人，参数失效
- 无法补偿尺寸差异：人手手指长，机器人手指短，同样角度导致指尖位置差异大
- 忽略手指间耦合

**适用场景**：快速原型验证、固定操作者 + 固定机器人

### 1.2 分段线性映射

**思想**：不同关节范围使用不同的映射比例。

```python
def piecewise_mapping(human_joint):
    """
    分段线性：小角度精细，大角度粗略
    """
    if abs(human_joint) < 0.5:
        return human_joint * 1.2  # 小角度放大
    else:
        return human_joint * 0.8  # 大角度缩小
```

### 1.3 关节限位裁剪

**思想**：映射后的角度如果超出机器人关节范围，直接截断到边界。

```python
robot_joint = np.clip(mapped_joint, robot_joint_limits[:, 0], robot_joint_limits[:, 1])
```

---

## 2. Vector Optimization 方法

### 2.1 任务空间 IK（核心方法）

**思想**：不直接映射角度，而是让人手关键点（如指尖）和机器人指尖在任务空间（3D 坐标）上对齐，通过 IK 求解机器人关节角。

```python
from scipy.optimize import least_squares

def retarget_task_space(human_landmarks, robot_model):
    """
    任务空间 IK retargeting

    Args:
        human_landmarks: 人手 21 点坐标 [21, 3]
        robot_model: 机器人手模型（含 FK、关节限位）

    Returns:
        robot_joint_angles: 机器人关节角 [n_dof]
    """
    # 提取人手 fingertip 位置
    human_tips = extract_fingertips(human_landmarks)  # [5, 3]

    # 优化目标：机器人 fingertip 与人手 fingertip 对齐
    def objective(joint_angles):
        robot_tips = robot_model.forward_kinematics(joint_angles)  # [5, 3]
        return (robot_tips - human_tips).flatten()

    # 阻尼最小二乘优化
    result = least_squares(
        objective,
        x0=initial_guess,
        bounds=(joint_lower_bounds, joint_upper_bounds),
        method='lm',
        ftol=1e-6,
    )
    return result.x
```

**优点**：
- 自动补偿尺寸差异（人手和机器人手手指长度不同）
- 任务空间直观（ fingertip 对齐 = 手势一致）
- 可加入关节限位约束

**缺点**：
- 优化求解需要时间（通常 1-10ms）
- 可能存在局部最优（拇指 MCP 校准不准）
- 需要准确的机器人运动学模型

**适用场景**：对精度要求高、机器人模型已知的场景

### 2.2 关节空间优化

**思想**：在关节空间直接优化，让机器人关节角尽可能接近归一化后的人手关节角。

```python
def objective_joint_space(robot_joints):
    normalized_human = normalize_human_joints(human_joints)
    return robot_joints - normalized_human
```

**与任务空间的区别**：
- 关节空间：优化速度快，但忽略尺寸差异
- 任务空间：精度高，但计算量大

### 2.3 混合优化

**思想**：同时优化任务空间误差和关节空间正则化。

```python
def objective_hybrid(robot_joints):
    # 任务空间误差
    robot_tips = fk(robot_joints)
    task_error = np.sum((robot_tips - target_tips) ** 2)

    # 关节空间正则化（避免奇怪姿态）
    joint_reg = np.sum((robot_joints - nominal_pose) ** 2)

    return task_error + 0.1 * joint_reg
```

---

## 3. Learning-based 方法

### 3.1 神经网络映射

**思想**：用神经网络学习人手特征到机器人关节角的映射。

```python
import torch.nn as nn

class RetargetingNet(nn.Module):
    def __init__(self, input_dim=63, output_dim=10):  # 21*3=63, O10=10
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, output_dim),
            nn.Tanh(),  # 输出 [-1, 1]，再映射到关节范围
        )

    def forward(self, landmarks):
        return self.net(landmarks)
```

**训练数据**：
- 人手 21 点坐标 + 对应机器人关节角
- 可以通过 Vector Optimization 方法生成伪标签
- 或使用真实遥操作数据

**优点**：
- 推理速度快（单次前向传播）
- 可以隐式学习人手到机器人手的复杂映射
- 可以处理噪声输入

**缺点**：
- 需要训练数据
- 泛化到新机器人需要重新训练
- 黑盒，难以调试

### 3.2 VAE / CVAE

**思想**：学习人手姿态的隐空间表示，再解码到机器人关节空间。

```python
class CVAE_Retargeting(nn.Module):
    """
    条件 VAE：输入人手 landmarks，输出机器人关节分布
    """
    def encode(self, landmarks, robot_joints):
        # 编码到隐空间
        pass

    def decode(self, z, landmarks):
        # 从隐空间解码到机器人关节
        pass
```

**优点**：生成多模态输出（一个手势对应多种机器人姿态）
**缺点**：训练更复杂

### 3.3 Diffusion Policy

**思想**：将 retargeting 建模为去噪过程，从噪声中逐步生成合理的机器人动作。

**适用场景**：需要生成平滑、多峰的动作分布时

---

## 方法对比

| 维度 | Rule-based | Vector Optimization | Learning-based |
|------|-----------|---------------------|----------------|
| **速度** | 极快 (<1ms) | 中等 (1-10ms) | 快 (1-5ms) |
| **精度** | 低 | 高 | 中-高（取决于数据） |
| **泛化性** | 差 | 好（换机器人需改模型） | 中（需重新训练） |
| **数据需求** | 无 | 无（需机器人模型） | 大量配对数据 |
| **调试难度** | 简单 | 中等 | 困难 |
| **拇指校准** | 一般 | 可能不准 | 取决于数据覆盖 |
| **Sim-to-Real** | 直接可用 | 直接可用 | 可能需要域适应 |

## 选择建议

| 场景 | 推荐方法 |
|------|---------|
| 快速原型验证 | Rule-based |
| 高精度遥操作 | Vector Optimization |
| 实时性要求极高 | Rule-based 或 Learning-based |
| 多个操作者/机器人 | Learning-based（训练一次） |
| 无机器人模型 | Rule-based 或 Learning-based |
| 需要平滑动作 | Vector Optimization + 插值 |

---

## 工程经验

来自实际项目的经验（O10 灵巧手 + MediaPipe）：

1. **Rule-based 最稳定**：Vector Optimization 存在拇指 MCP 校准不准的问题，Rule-based 在工程上更可靠
2. **21点 → finger_curl 往返转换存在衰减**：需要调整归一化分母和 actuator 缩放系数
3. **先插值 landmarks 再 IK**：在关节空间插值会导致任务空间失真，应先对 landmarks 做样条插值
4. **左右手镜像不可忘**：左手 index@-Y，右手 index@+Y
5. **关节限位必须加**：无论哪种方法，最终都要裁剪到机器人关节范围内
