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
Stage 5: Main Tracks (主线)
  └─ docs/01-what-is-vla.md → docs/13-vla-zero-to-one.md → ...
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

**总学习时间**：约 33–53 小时

---

## 学习路径建议

### 路径 A：机械工程背景（推荐）

```
01 Python → 02 线性代数 → 05 坐标变换 → 06 SE(3) → 07 FK/IK → 08 控制
→ 03 深度学习 → 04 Transformer → 09 MuJoCo → 10 数据集
→ 主线：VLA / RL / World Model
```

### 路径 B：CS / ML 背景

```
01 Python → 05 坐标变换 → 06 SE(3) → 07 FK/IK → 08 控制
→ 09 MuJoCo → 10 数据集
→ 主线：VLA / RL / World Model
```

### 路径 C：快速入门（已有部分基础）

```
直接跳到 07 FK/IK（检查理解）→ 09 MuJoCo → 10 数据集
→ 主线：unified_pushcube_vla.py
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
| [`docs/23-robot-foundation-models.md`](../23-robot-foundation-models.md) | 01–10 全部 |
| [`docs/19-sim-to-real-guide.md`](../19-sim-to-real-guide.md) | 05–09 全部 |

---

## 学习方法建议

1. **不要跳过代码**：每个文档都包含可运行的 Python 代码片段，请实际运行
2. **做练习**：每个文档末尾有"检查理解"练习题
3. **建立直觉**：数学公式旁边都有物理直觉解释
4. **连接项目**：每个概念都会标注在项目代码中的对应位置

---

> **提示**：如果你是第一次接触这些内容，建议按照路径 A 的顺序学习。每完成一个文档，回到本路线图确认进度。
