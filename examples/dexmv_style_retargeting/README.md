# DexMV-Style 高精度 IK Retargeting 实践

> **证据范围提示：**这是位置目标优化的本地教学实现，不构成 DexMV 论文复现、跨平台验收或真机证据。下文修正了坐标与性能措辞；真实输入仍需要标定和适配，示例不能直接用于机器人控制。详见 [审查交接 H07](../../docs/reviews/content-audit-handoff.md)。

> 基于 **DexMV (ECCV 2022)** 核心算法，使用 MuJoCo 3.x + scipy 重新实现的高精度 IK Retargeting Pipeline。

## 概述

本项目以 **位置优化（Position Optimization）** 为思路，用 **MuJoCo 3.x + scipy.optimize.minimize** 实现指尖目标求解。核心依赖是 NumPy、SciPy、MuJoCo；没有逐平台执行记录时，不能保证 Windows / Linux / macOS 的模型资源和 viewer 均可直接运行，也不声称它是原论文中“精度最高”的方法。

### 核心算法

| 组件 | 本地实现 | 阅读时核对 |
|------|----------|------------|
| **IK 求解器** | SciPy SLSQP | 终止状态、残差与边界 |
| **FK / Jacobian** | MuJoCo 位置与 body Jacobian | 目标点是否就是所用 body 原点 |
| **损失函数** | NumPy Huber + 帧间正则 | 单位、delta、权重比例 |
| **约束** | 关节 bounds | 不等于全连杆碰撞、力矩或速度安全 |
| **环境** | numpy、scipy、mujoco | 模型资源路径和目标系统的实际运行记录 |

### 算法流程

```
经标定并映射到机器人坐标系的米制 Landmarks (21点)
    ↓
提取 Fingertip 位置 (5×3)
    ↓
[DexMVRetargeter]
    ├─ 设置目标 fingertip 位置
    ├─ SLSQP 优化: min HuberLoss(FK(q) - target) + smoothing(q - q_prev)
    ├─ 解析梯度: dLoss/dq = huber_grad @ Jacobian
    └─ 关节限位约束
    ↓
机器人关节角序列 (n_frames × n_dofs)
    ↓
MuJoCo 检查；真机控制需另建并验证控制与安全接口
```

## 文件结构

```
dexmv_style_retargeting/
├── dexmv_retargeting.py      # 核心 retargeting 算法
│   ├── DexMVRetargeter       # 主类: 加载模型 + 优化
│   ├── HuberLoss             # Huber 损失函数
│   └── SyntheticHandDataGenerator  # 合成数据生成器
├── run_pipeline.py           # 完整 pipeline 运行脚本
│   ├── 模型选择 (shadow/allegro/leap)
│   ├── 工作空间校准
│   ├── 合成数据生成
│   ├── Retargeting (单帧/序列)
│   ├── 精度评估 (FPE, Jerk, Loss)
│   └── MuJoCo 可视化
└── README.md                 # 本文档
```

## 快速开始

### 环境要求

```bash
pip install numpy scipy mujoco
```

### 运行 Pipeline

以下在 `examples/dexmv_style_retargeting/` 目录运行，先核对脚本所需模型文件已在预期位置。

```bash
# Shadow Hand (24 DOF, 5 指)
python run_pipeline.py --model shadow --n_frames 60

# Allegro Hand (16 DOF, 4 指)
python run_pipeline.py --model allegro --n_frames 30

# LEAP Hand (16 DOF, 4 指)
python run_pipeline.py --model leap --n_frames 30

# 带可视化 (需要 MuJoCo renderer)
python run_pipeline.py --model shadow --n_frames 30 --visualize

# 调整优化参数
python run_pipeline.py --model shadow --n_frames 60 --huber_delta 0.002 --smoothing 0.001
```

### 参数说明

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--model` | `shadow` | 机器人手模型 (shadow/allegro/leap) |
| `--n_frames` | 60 | 序列帧数 |
| `--gestures` | open fist pinch | 手势序列 |
| `--huber_delta` | 0.005 | Huber 二次/线性区间阈值；须为正，越小不保证越精确 |
| `--smoothing` | 0.002 | 时序平滑权重 |
| `--visualize` | False | MuJoCo 可视化 |
| `--record` | False | 录制视频 (需 imageio) |

## 历史输出示例与比较协议

### Shadow Hand (24 DOF)

```bash
python run_pipeline.py --model shadow --n_frames 30
```

**历史输出示例（本轮未重跑；没有附同协议对比日志）**:

```
Model:        SHADOW
Frames:       30
Mean FPE:     76.778 mm
Max FPE:      146.969 mm
Time/frame:   0.7 ms

Per-finger mean FPE:
    Thumb   : 122.954 mm
    Index   : 84.107 mm
    Middle  : 73.249 mm
    Ring    : 51.015 mm
    Pinky   : 52.567 mm
```

**说明**：FPE 是所给目标点与模型点之间的误差，须写明坐标、尺度、指尖定义和可达性。不能仅因数据是合成的就称几十毫米误差“合理”，也不能据此推出真实数据会达到 <10 mm。

### 如何做可比较的评估

| 比较项 | 必须保持一致或披露 |
|--------|--------------------|
| 几何 | 同一手模型、坐标变换、米制尺度、指尖点定义 |
| 数据 | 相同输入帧、可达目标比例、缺失与遮挡处理 |
| 优化 | 相同初值约定、时间预算、约束、失败处理 |
| 指标 | FPE 分布、失败率、抖动/滞后、求解与端到端时间分别报告 |

撤下原表无同协议依据的 Rule-based / Vector Opt / AnyTeleop / DexPilot 数值排名与仓库可见性判断。真实输入也可能不可达或含噪声，性能必须重新测量。

## 核心代码解析

### 1. Huber Loss

下式是 Huber 定义；某些库的 Smooth L1 定义还除以 delta，不能在相同正则权重下直接当作数值相同。输入 diff 的单位决定 delta 的单位。

```python
import numpy as np

class HuberLoss:
    def __init__(self, delta: float = 0.01):
        self.delta = delta

    def __call__(self, diff: np.ndarray) -> float:
        abs_diff = np.abs(diff)
        quadratic = np.minimum(abs_diff, self.delta)
        linear = abs_diff - quadratic
        return np.sum(0.5 * quadratic ** 2 + self.delta * linear)
```

**为什么用 Huber Loss？**
- 对 **小误差** 使用 L2（二次），平滑可导
- 对 **大误差** 使用 L1（线性），抵抗离群点
- 比纯 L2 更鲁棒，比纯 L1 更平滑

### 2. SLSQP 优化 (带解析梯度)

下面是接口骨架，`obj_fn`、边界和模型变量需由调用方定义：

```python
result = minimize(
    obj_fn,      # Huber loss + smoothing
    init_qpos,   # 初始猜测 (上一帧结果)
    method="SLSQP",
    jac=grad_fn, # 解析梯度 (Jacobian @ huber_grad)
    bounds=list(zip(lower_limits, upper_limits)),  # 每个标量关节的上下界
    options={"ftol": 1e-5, "maxiter": 200},
)
```

**梯度计算**:

```
dLoss/dq = dLoss/dpos * dpos/dq
         = huber_grad^T * Jacobian
```

### 3. 时序平滑 (Temporal Smoothing)

```text
loss = huber_loss + smoothing_weight * ||q - q_prev||^2
```

- 减少帧间抖动
- 利用上一帧结果作为 warm-start，加速收敛
- 默认权重 `2e-3`

### 4. Jacobian 计算 (MuJoCo 3.x)

以下列选择仅适用于逐个标量 hinge/slide 关节；ball/free 关节有多个速度自由度，不能各取一列就代表全部。还需核对 body 原点是否等于目标指尖点。

```python
# MuJoCo 自动计算 body 的位置 Jacobian
jac_body = np.zeros((3, model.nv))
mujoco.mj_jacBody(model, data, jac_body, None, body_id)

# 提取可控关节对应的列
for j, jnt_id in enumerate(joint_ids):
    dof_adr = model.jnt_dofadr[jnt_id]
    J[i*3:(i+1)*3, j] = jac_body[:, dof_adr]
```

## 扩展到真实数据

### 从 MediaPipe 到米制目标：必须先有标定

`multi_hand_landmarks` 的 x/y 是图像归一化坐标，z 也不能直接当世界坐标米。world landmarks 的米制相对手坐标仍不等于机器人的全局坐标。以下只列未实现的接入步骤；缺少深度/标定时不要假设归一化图像点能直接唯一恢复绝对 3D 位置。

```text
检测成功？否则丢弃该帧并记录缺测
选择并注明 image landmarks 或 world landmarks
建立深度/尺度及相机到机器人坐标合同
映射人手几何到机器人目标，统一为米并核对左右手
按目标机器人指尖顺序取点 → 检查可达性 → 仿真求解
```

### 从 InterHand2.6M 数据集提取

不能照搬 MediaPipe 的 `[4,8,12,16,20]` 作为其他数据集的关节顺序。先查该版标注的骨架、左右手、单位与相机定义，再显式映射。

```text
读取目标版本的标注与相机参数
按该数据集关节名称找到指尖，而不是复用另一个骨架的数组下标
将单位与坐标变换到机器人模型约定
构成每帧已标定目标序列，再检查求解状态和误差
```

## 已知限制

1. **模型资源**：核对 Allegro / Shadow / LEAP 具体模型的惯性、关节顺序、指尖 body 和初始姿态；加载成功不证明与真实型号一一对应或所有关节参数已校准。

2. **LEAP Hand**: 合成数据的 fingertip 位置与 LEAP Hand 工作空间匹配度较低，建议使用真实人手数据。

3. **计算速度**：历史求解耗时不包括采集、检测、传输、调度和执行器响应，不能据此保证 100 Hz 闭环。减少迭代预算也可能增加求解失败；请分别量测 p50/p95/p99 和截止时间违约率。

4. **拇指精度**: 拇指的 IK 求解通常精度较低（误差较大），因为拇指的运动学链更复杂，且目标 fingertip 位置可能超出可达空间。

## 参考

- **DexMV**: Qin et al., "DexMV: Imitation Learning for Dexterous Manipulation from Human Videos", ECCV 2022. [GitHub](https://github.com/yzqin/dexmv-sim)
- **MuJoCo 3.x**: [Documentation](https://mujoco.readthedocs.io/)
- **MediaPipe 坐标输出约定**：[官方 Hand Landmarker 说明](https://ai.google.dev/edge/mediapipe/solutions/vision/hand_landmarker/python)
- **Huber Loss**: Huber, P. J. (1964). "Robust estimation of a location parameter".
