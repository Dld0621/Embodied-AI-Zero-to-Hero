# Foundations Layer: Zero to Hero 基础课程

> **目标**：为机械工程背景的学生补齐从"零基础"到"能跑通 VLA / RL / World Model 代码"所需的全部基础知识。不假设读者有 CS 或 ML 背景。

---

## 为什么需要这一层？

本仓库的主线文档（VLA、World Model、RL、Robot Foundation Models）默认读者已具备一定的 Python、线性代数、深度学习和机器人学基础。但对于机械工程背景的学生，以下跳跃经常出现：

```
Python → PyTorch → Transformer → VLA
```

这中间缺少了太多环节。Foundations Layer 就是为了填平这些沟壑。

---

## 课程路线图

```
Stage 0: Programming
  └─ 01-python-for-robotics     ← NumPy 数组、矩阵运算、可视化
       ↓
Stage 1: Mathematics
  └─ 02-linear-algebra          ← 向量、矩阵、特征值、概率
       ↓
Stage 2: Deep Learning
  └─ 03-deep-learning-basics    ← 神经网络、反向传播、损失函数
  └─ 04-transformer-basics      ← Attention、Self-Attention、Transformer
       ↓
Stage 3: Robotics
  └─ 05-coordinate-transform    ← 坐标系、齐次变换
  └─ 06-se3-and-rotation        ← SO(3)、SE(3)、旋转表示
  └─ 07-fk-jacobian-ik          ← 正运动学、Jacobian、逆运动学
  └─ 08-control-basics          ← PID、阻抗控制、关节伺服
       ↓
Stage 4: Simulation & Data
  └─ 09-mujoco-basics           ← MuJoCo 引擎、URDF/MJCF、仿真循环
  └─ 10-dataset-and-training    ← 数据集格式、训练循环、评估
       ↓
Stage 5: Reliable Robot Learning
  └─ 11-probability-and-optimization ← 概率、统计、优化与不确定性
  └─ 12-perception-and-sensors       ← 相机、深度、状态、力觉与同步
  └─ 13-robot-systems-and-safety     ← 控制栈、接口、实时性与安全门禁
  └─ 14-evaluation-and-reproducibility ← 指标、基线、复现与证据等级
       ↓
Stage 6: End-to-End Pipelines (方向主线)
  └─ docs/pipelines/README_CN.md → 选择方向 → smoke test → benchmark
```

---

## 课程列表

| # | 文档 | 主题 | 预计学习时间 | 前置要求 |
|:--|:-----|:-----|:----------:|:---------|
| 01 | [`01-python-for-robotics.md`](01-python-for-robotics.md) | Python for Robotics | 3–5h | 无 |
| 02 | [`02-linear-algebra.md`](02-linear-algebra.md) | Linear Algebra | 4–6h | 01 |
| 03 | [`03-deep-learning-basics.md`](03-deep-learning-basics.md) | Deep Learning Basics | 5–8h | 02 |
| 04 | [`04-transformer-basics.md`](04-transformer-basics.md) | Transformer Basics | 3–5h | 03 |
| 05 | [`05-coordinate-transform.md`](05-coordinate-transform.md) | Coordinate Transform | 3–4h | 02 |
| 06 | [`06-se3-and-rotation.md`](06-se3-and-rotation.md) | SO(3) & SE(3) | 3–5h | 05 |
| 07 | [`07-fk-jacobian-ik.md`](07-fk-jacobian-ik.md) | FK, Jacobian & IK | 4–6h | 06 |
| 08 | [`08-control-basics.md`](08-control-basics.md) | Control Basics | 2–4h | 07 |
| 09 | [`09-mujoco-basics.md`](09-mujoco-basics.md) | MuJoCo Basics | 3–5h | 08 |
| 10 | [`10-dataset-and-training.md`](10-dataset-and-training.md) | Dataset & Training | 3–5h | 03, 09 |
| 11 | [`11-probability-and-optimization.md`](11-probability-and-optimization.md) | Probability, Statistics & Optimization | 3–4h | 02, 03 |
| 12 | [`12-perception-and-sensors.md`](12-perception-and-sensors.md) | Perception & Sensors | 3–4h | 05, 09 |
| 13 | [`13-robot-systems-and-safety.md`](13-robot-systems-and-safety.md) | Robot Systems & Safety | 3–4h | 08, 12 |
| 14 | [`14-evaluation-and-reproducibility.md`](14-evaluation-and-reproducibility.md) | Evaluation & Reproducibility | 2–3h | 10, 11 |

**总学习时间**：约 44–68 小时。无需一次学完全部内容；先完成公共基础，再按目标 Pipeline 补对应章节。

---

## 学习路径建议

### 路径 A：机械工程背景（推荐）

```
01 Python → 02 线性代数 → 05 坐标变换 → 06 SE(3) → 07 FK/IK → 08 控制
→ 03 深度学习 → 04 Transformer → 09 MuJoCo → 10 数据集
→ 11 概率优化 → 12 感知 → 13 系统安全 → 14 评估复现
→ 选择一条端到端 Pipeline
```

### 路径 B：CS / ML 背景

```
01 Python → 05 坐标变换 → 06 SE(3) → 07 FK/IK → 08 控制
→ 09 MuJoCo → 12 感知 → 13 系统安全 → 14 评估复现
→ 选择一条端到端 Pipeline
```

### 路径 C：快速入门（已有部分基础）

```
先阅读 14 评估复现 → 运行 `python scripts/run_pipeline.py --list`
→ 按所选方向补前置章节 → 运行 smoke test → 再进入完整训练
```

---

## 与主线文档的衔接

完成 Foundations Layer 后，你将能顺利进入以下主线：

| 主线文档 | 需要的 Foundations 知识 |
|:---------|:----------------------|
| [`docs/01-what-is-vla.md`](../01-what-is-vla.md) | 03 深度学习、04 Transformer |
| [`docs/06-rl-fundamentals-for-vla.md`](../06-rl-fundamentals-for-vla.md) | 02 线性代数、03 深度学习 |
| [`docs/07-world-models-for-vla.md`](../07-world-models-for-vla.md) | 03 深度学习、04 Transformer |
| [`docs/13-vla-zero-to-one.md`](../13-vla-zero-to-one.md) | 01–04 全部 |
| [`docs/23-robot-foundation-models.md`](../23-robot-foundation-models.md) | 03、04、10、12、13 |
| [`docs/19-sim-to-real-guide.md`](../19-sim-to-real-guide.md) | 05–09、12–14 |
| [`docs/pipelines/README_CN.md`](../pipelines/README_CN.md) | 依据十条主线选择对应前置章节 |

---

## 学习方法建议

1. **不要跳过代码**：每个文档都包含可运行的 Python 代码片段，请实际运行
2. **做练习**：每个文档末尾有"检查理解"练习题
3. **建立直觉**：数学公式旁边都有物理直觉解释
4. **连接项目**：每个概念都会标注在项目代码中的对应位置

---

> **提示**：如果你是第一次接触这些内容，建议按照路径 A 的顺序学习。每完成一个文档，回到本路线图确认进度。
