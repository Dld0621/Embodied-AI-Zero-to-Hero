# 大一新生也能跑通：人手仿真与 DexMV 重定向 0→1 实战

> **目标**: 不需要任何先验知识，跟着本文一步步操作，完成"人手 21 点生成 → 可视化 → 坐标输入 → IK 重定向 → 机器人可视化"的完整 pipeline。**无需下载任何额外开源代码**。

---

## 目录

1. [你能学到什么](#1-你能学到什么)
2. [环境准备（10 分钟）](#2-环境准备10-分钟)
3. [第一次运行（1 分钟）](#3-第一次运行1-分钟)
4. [理解 pipeline 的 6 个步骤](#4-理解-pipeline-的-6-个步骤)
5. [5 种手势体验](#5-种手势体验)
6. [用自己输入的坐标运行](#6-用自己输入的坐标运行)
7. [可视化：看到人手和机器人](#7-可视化看到人手和机器人)
8. [理解背后的原理](#8-理解背后的原理)
9. [常见问题排查](#9-常见问题排查)
10. [下一步学什么](#10-下一步学什么)

---

## 1. 你能学到什么

完成本文后，你将能够：

- **生成** 人手 21 点坐标（MediaPipe 格式）
- **可视化** 3D 人手骨架（matplotlib）
- **提取** fingertip 位置（拇指、食指、中指、无名指、小指）
- **运行** DexMV 高精度 IK 重定向（SLSQP + Huber Loss）
- **查看** 机器人手的重定向结果与精度指标
- **输入自己的坐标** 并立即获得机器人关节角

> 全程只需要 1 个 Python 文件：`examples/freshman_zero_to_one.py`

---

## 2. 环境准备（10 分钟）

### 2.1 安装 Python（如果还没有）

- 访问 [python.org](https://python.org)，下载 Python 3.10 或更高版本
- 安装时勾选 **"Add Python to PATH"**

### 2.2 安装依赖

打开终端（Windows: PowerShell / CMD；Mac/Linux: Terminal），执行：

```bash
pip install numpy scipy mujoco matplotlib
```

**这些包的作用**：

| 包名 | 作用 |
|------|------|
| `numpy` | 数值计算（数组、矩阵运算） |
| `scipy` | 科学计算（优化算法 SLSQP） |
| `mujoco` | 机器人仿真引擎（加载 URDF、计算运动学） |
| `matplotlib` | 绘图（人手 3D 可视化） |

### 2.3 下载本项目

```bash
git clone https://github.com/Dld0621/Dexterous-Retargeting-Guide.git
cd Dexterous-Retargeting-Guide
```

> 注意：本项目已包含所有需要的 URDF 模型文件，**无需额外下载**其他开源仓库。

### 2.4 验证模型文件存在

```bash
ls pretrained/urdf/mujoco_menagerie/shadow_hand/
```

如果看到 `scene_right.xml` 等文件，说明模型已就绪。

---

## 3. 第一次运行（1 分钟）

### 3.1 最简单的命令

```bash
cd examples
python freshman_zero_to_one.py --gesture open --model shadow
```

**你应该看到类似输出**：

```
======================================================================
 Freshman Zero-to-One: Human Hand → Robot Hand Retargeting
======================================================================

[Step 1/6] 获取人手 21 点坐标
  内置手势: open
  形状: (1, 21, 3)

[Step 2/6] 人手可视化已跳过...

[Step 3/6] 提取 Fingertip 坐标
  Fingertip 坐标形状: (1, 5, 3)

[Step 4/6] Workspace 校准...
  机器人模型: SHADOW
[DexMVRetargeter] 加载成功: 24 DOFs, 5 fingertips
  缩放因子: 1.518

[Step 5/6] DexMV Retargeting (SLSQP + Huber Loss)
  重定向耗时: 0.003s (2.5 ms/帧)

[Step 6/6] 评估重定向精度
  平均 FPE: 61.02 mm
  最大 FPE: 114.62 mm
  每指平均 FPE:
    Thumb   : 114.62 mm
    Index   : 73.06 mm
    Middle  : 70.62 mm
    Ring    : 27.84 mm
    Pinky   : 18.96 mm

======================================================================
 Pipeline 完成！
======================================================================
```

**恭喜！** 你已经成功运行了从人手到机器人手的 IK 重定向。

### 3.2 关键输出解读

| 输出 | 含义 |
|------|------|
| `24 DOFs` | Shadow Hand 有 24 个可控制关节 |
| `5 fingertips` | 重定向目标是 5 个指尖位置 |
| `缩放因子: 1.518` | 人手坐标被缩放到机器人手的实际尺寸 |
| `平均 FPE: 61.02 mm` | Fingertip Position Error，指尖位置误差 |
| `2.5 ms/帧` | 每帧重定向耗时，远小于 10 ms（满足 100 Hz 实时） |

> **FPE 是什么？** 机器人 fingertip 实际位置与目标位置的欧氏距离。61 mm 对于合成数据是正常范围，真实人手数据通常 < 10 mm。

---

## 4. 理解 Pipeline 的 6 个步骤

```
人手坐标 (21点)
    ↓
Step 1: 获取坐标 —— 内置生成 / 文件读取
    ↓
Step 2: 人手可视化 —— matplotlib 3D 画图
    ↓
Step 3: 提取 Fingertip —— 取 5 个指尖（索引 4,8,12,16,20）
    ↓
Step 4: Workspace 校准 —— 缩放到机器人手尺寸
    ↓
Step 5: DexMV Retargeting —— SLSQP 优化求解关节角
    ↓
Step 6: 评估 —— 计算 FPE 精度指标
    ↓
机器人关节角 (qpos)
```

### Step 1: 获取人手 21 点坐标

程序内置了 5 种手势的坐标：

```python
# 张开手
landmarks = HumanHand21.get_open_hand()   # shape: (21, 3)

# 握拳
landmarks = HumanHand21.get_fist()

# 捏合
landmarks = HumanHand21.get_pinch()
```

每个 landmark 是 `[x, y, z]` 坐标（单位：米），以手腕为原点。

### Step 2: 人手可视化

加 `--visualize-human` 即可看到 3D 人手：

```bash
python freshman_zero_to_one.py --gesture open --visualize-human
```

你会看到：
- 灰色连线 = 手指骨骼
- 彩色点 = 不同手指（拇指红、食指青、中指蓝、无名指绿、小指黄）
- 红色文字标注 = 5 个 fingertip 名称

### Step 3: 提取 Fingertip

MediaPipe 21 点中，fingertip 的索引是固定的：

| 手指 | 索引 | 名称 |
|------|------|------|
| 拇指 | 4 | Thumb Tip |
| 食指 | 8 | Index Tip |
| 中指 | 12 | Middle Tip |
| 无名指 | 16 | Ring Tip |
| 小指 | 20 | Pinky Tip |

```python
tip_indices = [4, 8, 12, 16, 20]
fingertip_positions = landmarks[tip_indices]  # shape: (5, 3)
```

### Step 4: Workspace 校准

人手和机器人手的尺寸不同，需要缩放：

```python
# 机器人手 reference pose 的 fingertip 散布
ref_spread = 0.08  # 约 8 cm

# 人手坐标 fingertip 散布
syn_spread = 0.12  # 约 12 cm

# 缩放因子
scale = ref_spread / syn_spread  # ≈ 0.67

# 应用缩放 + 偏移到机器人掌心位置
fingertip_positions = fingertip_positions * scale * 0.8 + palm_center
```

### Step 5: DexMV Retargeting（核心）

这是整个 pipeline 最复杂的部分，但你**不需要理解所有细节**就能使用：

```python
# 初始化重定向器
retargeter = DexMVRetargeter(
    model_path="shadow_hand.xml",           # 机器人模型
    fingertip_body_names=["rh_ffdistal",   # 指尖 body 名称
                          "rh_mfdistal",
                          "rh_rfdistal",
                          "rh_lfdistal",
                          "rh_thdistal"],
    huber_delta=0.005,
    smoothing_weight=0.002,
)

# 运行重定向
qpos = retargeter.retarget_single(fingertip_positions)
```

内部发生了什么？
1. **FK**（正向运动学）：给定关节角，计算 fingertip 位置
2. **计算误差**：当前位置 vs 目标位置
3. **Huber Loss**：衡量匹配程度（越小越好）
4. **Jacobian**：误差对关节角的梯度
5. **SLSQP 优化**：迭代调整关节角，直到误差最小
6. **关节限位**：确保角度在物理可行范围内

### Step 6: 评估

计算 **FPE（Fingertip Position Error）**：

```python
error = ||robot_fingertip - target_fingertip||  # 单位：米
fpe_mm = error * 1000  # 转换为毫米
```

---

## 5. 5 种手势体验

### 5.1 逐个体验

```bash
# 1. 张开手（五指伸直）
python freshman_zero_to_one.py --gesture open --model shadow

# 2. 握拳（所有手指卷曲）
python freshman_zero_to_one.py --gesture fist --model shadow

# 3. 捏合（拇指+食指接触）
python freshman_zero_to_one.py --gesture pinch --model shadow

# 4. OK 手势
python freshman_zero_to_one.py --gesture ok --model shadow

# 5. 食指指向前方
python freshman_zero_to_one.py --gesture pointing --model shadow
```

### 5.2 手势序列

```bash
python freshman_zero_to_one.py --gesture sequence --gesture-sequence open,fist,pinch --n-frames 30
```

这会生成 30 帧的过渡动画：张开 → 握拳 → 捏合。

### 5.3 对比不同机器人手

```bash
# Shadow Hand（24 DOF，5 指，最复杂）
python freshman_zero_to_one.py --model shadow --gesture open

# Allegro Hand（16 DOF，4 指）
python freshman_zero_to_one.py --model allegro --gesture open

# LEAP Hand（16 DOF，4 指，低成本）
python freshman_zero_to_one.py --model leap --gesture open
```

---

## 6. 用自己输入的坐标运行

这是本文最重要的部分：**如何输入你自己的坐标**。

### 6.1 方法 1: JSON 文件

创建文件 `my_hand.json`：

```json
{
  "landmarks": [
    [0.0, 0.0, 0.0],
    [0.025, 0.01, 0.005],
    [0.035, 0.025, 0.008],
    [0.042, 0.04, 0.01],
    [0.048, 0.055, 0.012],
    [0.02, 0.055, 0.002],
    [0.022, 0.075, 0.003],
    [0.023, 0.095, 0.004],
    [0.024, 0.115, 0.005],
    [0.005, 0.06, 0.0],
    [0.006, 0.085, 0.001],
    [0.007, 0.11, 0.002],
    [0.008, 0.135, 0.003],
    [-0.012, 0.055, 0.0],
    [-0.014, 0.075, 0.001],
    [-0.015, 0.095, 0.002],
    [-0.016, 0.115, 0.003],
    [-0.025, 0.045, 0.002],
    [-0.028, 0.06, 0.003],
    [-0.03, 0.075, 0.004],
    [-0.032, 0.09, 0.005]
  ]
}
```

运行：

```bash
python freshman_zero_to_one.py --mode file --input my_hand.json --model shadow
```

### 6.2 方法 2: NumPy 文件

```python
import numpy as np

# 创建你自己的 21 点坐标（单位：米）
landmarks = np.random.randn(21, 3) * 0.02  # 随机生成示例
np.save("my_hand.npy", landmarks)
```

运行：

```bash
python freshman_zero_to_one.py --mode file --input my_hand.npy --model shadow
```

### 6.3 方法 3: 从 MediaPipe 获取（摄像头）

如果你想用真实摄像头输入，先安装 MediaPipe：

```bash
pip install mediapipe opencv-python
```

然后运行摄像头采集脚本（参考 `examples/landmark_to_joint.py`），将结果保存为 JSON：

```python
import json
import numpy as np

# 假设你从 MediaPipe 获取了 landmarks
landmarks = np.array([...])  # shape: (21, 3)

with open("camera_hand.json", "w") as f:
    json.dump({"landmarks": landmarks.tolist()}, f)
```

然后：

```bash
python freshman_zero_to_one.py --mode file --input camera_hand.json --model shadow
```

### 6.4 坐标格式说明

无论你从哪获取坐标，必须满足：

- **形状**: `(21, 3)` — 21 个点，每个点 `[x, y, z]`
- **单位**: 米（meters）
- **原点**: 手腕（第 0 点）应该在原点或靠近原点
- **坐标系**: 右手坐标系，Y 轴指向食指方向，Z 轴垂直手掌向外

**21 点顺序（MediaPipe 标准）**：

```
0  : Wrist
1  : Thumb CMC    (拇指腕掌关节)
2  : Thumb MCP    (拇指掌指关节)
3  : Thumb IP     (拇指指间关节)
4  : Thumb Tip    (拇指指尖)  ★
5  : Index MCP    (食指掌指关节)
6  : Index PIP    (食指近端指间关节)
7  : Index DIP    (食指远端指间关节)
8  : Index Tip    (食指指尖)  ★
9  : Middle MCP   (中指掌指关节)
10 : Middle PIP   (中指近端指间关节)
11 : Middle DIP   (中指远端指间关节)
12 : Middle Tip   (中指指尖)  ★
13 : Ring MCP     (无名指掌指关节)
14 : Ring PIP     (无名指近端指间关节)
15 : Ring DIP     (无名指远端指间关节)
16 : Ring Tip     (无名指指尖)  ★
17 : Pinky MCP    (小指掌指关节)
18 : Pinky PIP    (小指近端指间关节)
19 : Pinky DIP    (小指远端指间关节)
20 : Pinky Tip    (小指指尖)  ★
```

> ★ 标记的是 5 个 fingertip，重定向只用到这 5 个点。

---

## 7. 可视化：看到人手和机器人

### 7.1 人手可视化

```bash
python freshman_zero_to_one.py --gesture open --visualize-human
```

弹出 matplotlib 3D 窗口，显示：
- 灰色连线 = 骨骼
- 彩色散点 = 关节点
- 红色标签 = fingertip 名称

保存为图片：

```bash
python freshman_zero_to_one.py --gesture open --visualize-human --save-human hand_open.png
```

### 7.2 人手动画（序列）

```bash
python freshman_zero_to_one.py --gesture sequence --gesture-sequence open,fist,pinch --n-frames 30 --visualize-human --save-human hand_animation.gif
```

### 7.3 机器人可视化

```bash
python freshman_zero_to_one.py --gesture open --visualize-robot
```

弹出 MuJoCo 渲染窗口，显示机器人手的重定向结果。

**注意**: MuJoCo 渲染窗口需要图形界面，远程服务器可能无法显示。

### 7.4 同时显示两者

```bash
python freshman_zero_to_one.py --gesture open --visualize-human --visualize-robot
```

---

## 8. 理解背后的原理

### 8.1 什么是 IK Retargeting？

```
人手运动                机器人手运动
   ↓                        ↓
21 点坐标  ──Retargeting──→  关节角度
(fingertip 位置)         (qpos)
```

**核心问题**: 人手有 21 个关节点，机器人手有 16-24 个关节，如何把人手"翻译"成机器人的语言？

### 8.2 为什么用 fingertip 位置而不是关节角？

| 方法 | 优点 | 缺点 |
|------|------|------|
| **关节角映射** | 直观 | 人手和机器人手指长度不同，直接映射会导致指尖位置偏差大 |
| **fingertip 位置优化** | 直接控制最终效果（指尖位置） | 需要求解 IK，计算量稍大 |

DexMV 选择 **fingertip 位置优化**，因为：
- 最终抓取物体的是指尖，不是关节
- 补偿了人手和机器人手的尺寸差异

### 8.3 Huber Loss 是什么？

想象你在瞄准靶心：
- **L2 Loss（均方误差）**: 偏离 10 cm 的惩罚是偏离 1 cm 的 **100 倍** —— 对离群点太敏感
- **L1 Loss（绝对误差）**: 偏离 10 cm 的惩罚是偏离 1 cm 的 **10 倍** —— 在零点不可导
- **Huber Loss**: 偏离 1 cm 内用 L2（平滑），偏离超过 1 cm 用 L1（鲁棒） —— **两者兼顾**

### 8.4 为什么叫 "DexMV"？

DexMV 是论文 "Dexterous Manipulation from Human Videos"（ECCV 2022）的缩写。作者来自 UC San Diego，核心贡献是：
1. 从人手视频中提取 3D 运动
2. 用位置优化方法重定向到 Shadow Hand
3. 实现了接近 **< 10 mm** 的 fingertip 精度

---

## 9. 常见问题排查

### Q1: 报错 `找不到模型文件`

```
FileNotFoundError: 找不到模型文件: .../shadow_hand/scene_right.xml
```

**解决**: 模型文件在本项目的 `pretrained/urdf/` 目录中。确保你运行命令的位置正确：

```bash
cd Dexterous-Retargeting-Guide/examples
python freshman_zero_to_one.py --gesture open
```

如果从其他目录运行，用 `--model-dir` 指定模型路径：

```bash
python freshman_zero_to_one.py --gesture open --model-dir /path/to/pretrained/urdf
```

### Q2: 报错 `找不到 body 'rh_thdistal'`

**解决**: Shadow Hand 的 body 名称有 `rh_` 前缀。程序已内置正确名称，如果你修改了代码，请检查：

```python
shadow_tips = ["rh_thdistal", "rh_ffdistal", "rh_mfdistal", "rh_rfdistal", "rh_lfdistal"]
```

### Q3: FPE 很大（> 200 mm）

**原因**: 输入的坐标尺度不对（比如用了毫米而不是米）。

**解决**: 确保坐标单位是**米**。如果原始数据是毫米，除以 1000：

```python
landmarks = landmarks / 1000.0  # mm → m
```

### Q4: MuJoCo 窗口打不开

**原因**: 远程服务器没有图形界面，或 MuJoCo renderer 不可用。

**解决**: 
- 本地运行（有显示器的电脑）
- 或不加 `--visualize-robot`，只看终端输出

### Q5:  Allegro Hand 报错 `inertia must satisfy A + B >= C`

**解决**: 项目已修复 Allegro Hand 的 URDF 惯性矩阵。如果仍然报错，尝试：

```bash
python freshman_zero_to_one.py --model shadow  # 先用 Shadow Hand
```

### Q6: 如何保存结果？

```bash
python freshman_zero_to_one.py --gesture open --output joint_angles.npy
```

然后用 numpy 读取：

```python
import numpy as np
qpos = np.load("joint_angles.npy")
print(qpos.shape)  # (n_frames, n_dofs)
```

---

## 10. 下一步学什么

完成本文后，你已经掌握了 DexMV 重定向的**使用方法**。下一步可以深入学习：

| 方向 | 资源 | 难度 |
|------|------|------|
| **理解算法细节** | `docs/11-dexmv-research-guide.md` | ⭐⭐⭐ |
| **用自己的摄像头输入** | `examples/landmark_to_joint.py` | ⭐⭐ |
| **对比不同重定向方法** | `examples/minimal_retargeting.py` | ⭐⭐ |
| **评估框架** | `examples/evaluation_framework.py` | ⭐⭐⭐ |
| **完整 0→1 Pipeline** | `examples/complete_retargeting_pipeline.py` | ⭐⭐⭐ |
| **学习 FK/IK 原理** | `tutorials/01-fk-ik-basics/README.md` | ⭐⭐ |
| **Rule-based 方法** | `tutorials/02-rule-based-retargeting/README.md` | ⭐ |
| **向量优化方法** | `tutorials/03-vector-optimization/README.md` | ⭐⭐ |
| **真实工程代码（OmniHand）** | `../../02_OUR_METHOD/OmniHand_v19/` | ⭐⭐⭐ |
| **MuJoCo 仿真平台** | `../../omnihand_mujoco/` | ⭐⭐⭐ |

> **从教程到工程**：本文档的 `freshman_zero_to_one.py` 是一个"零外部依赖"的教学实现。当你理解了核心概念后，可以直接阅读 `../../02_OUR_METHOD/OmniHand_v19/` 中的实际工程代码，它处理了我们刻意简化的部分：mimic joint 补偿、双手镜像对齐、InterHand 数据集坐标系转换、实时 UDP 驱动等。

---

## 附录：命令速查表

```bash
# === 基础运行 ===
python freshman_zero_to_one.py --gesture open
python freshman_zero_to_one.py --gesture fist --model allegro

# === 可视化 ===
python freshman_zero_to_one.py --gesture open --visualize-human
python freshman_zero_to_one.py --gesture open --visualize-robot
python freshman_zero_to_one.py --gesture open --visualize-human --visualize-robot

# === 手势序列 ===
python freshman_zero_to_one.py --gesture sequence --gesture-sequence open,fist,pinch --n-frames 30

# === 文件输入 ===
python freshman_zero_to_one.py --mode file --input my_hand.json --model shadow
python freshman_zero_to_one.py --mode file --input my_hand.npy --model allegro

# === 保存输出 ===
python freshman_zero_to_one.py --gesture open --output joint_angles.npy
python freshman_zero_to_one.py --gesture open --visualize-human --save-human hand.png

# === 调参 ===
python freshman_zero_to_one.py --gesture open --huber-delta 0.001 --smoothing 0.001
```

---

> **本文目标达成**: 你现在可以在**不需要任何外部开源代码**的情况下，输入人手 21 点坐标，立即获得机器人灵巧手的关节角度。这就是 IK Retargeting 的 0→1。
