# Dataset & Training / 数据集与训练

> **逐点图解 / Concept close-ups：**[Episode 协议与多模态对齐](../knowledge-atlas/data-episode-schema/index.md) · [数据质量、覆盖与无泄漏切分](../knowledge-atlas/data-quality-splits/index.md)。每个小点配原理、算例、图、自测；这是中文细解，保留英文术语。

> English contract: [Foundations overview](README_EN.md#route) · Primary references: [Datasets and training](../SOURCES.md#10-datasets-and-training)

> **前置要求**: 完成 [`03-deep-learning-basics.md`](03-deep-learning-basics.md) 与 [`09-mujoco-basics.md`](09-mujoco-basics.md)
> **预计学习时间**: 2–3 小时
> **完成后你能**: 理解机器人数据的采集方式；读懂 episode / frame 数据结构；按 episode 正确划分 train / val / test 避免泄漏；实现状态与动作归一化；用 PyTorch DataLoader 写出训练循环；区分离线评估与闭环评估；识别行为克隆的过拟合信号

---

## 目录

1. [数据采集方式](#1-数据采集方式)
2. [数据集格式](#2-数据集格式)
3. [Episode vs Frame](#3-episode-vs-frame)
4. [数据归一化](#4-数据归一化)
5. [训练 Pipeline](#5-训练-pipeline)
6. [评估：离线 vs 闭环](#6-评估离线-vs-闭环)
7. [过拟合信号](#7-过拟合信号)
8. [连接项目代码](#8-连接项目代码)
9. [可运行代码：数据集类与训练循环](#9-可运行代码数据集类与训练循环)
10. [检查理解](#10-检查理解)

---

## 1. 数据采集方式

机器人学习策略需要"示范数据"——告诉模型"在这种状态下该做什么动作"。常见采集方式：

| 方式 | 说明 | 优点 | 缺点 |
|:-----|:-----|:-----|:-----|
| **遥操作（Teleoperation）** | 人用主控臂/VR 手套/键盘远程操控真机，记录关节与图像 | 质量高、接近真实分布 | 慢、需硬件、人力成本高 |
| **脚本策略（Scripted Policy）** | 手写启发式规则自动生成数据 | 快、可大规模、可重复 | 行为单一、缺乏多样性 |
| **人类演示（Human Demonstration）** | 人直接动真机或拖动示教，记录轨迹 | 自然流畅 | 难精确对齐到机器人本体（需 retargeting） |

本项目 PushCube 数据用**脚本策略**：一个手写三阶段启发式专家（flank→approach→push）在仿真里自动跑出 50 条成功轨迹（见 [`collect_pushcube_dataset.py`](../../examples/robot_foundation_models/smolvla/collect_pushcube_dataset.py)）。专家成功率约 100%，省去人工采集。

> **直觉**：数据决定策略上限。脚本策略数据"干净但单一"，所以学到的策略容易过拟合到专家的固定行为模式——这正是第 7 节的问题。

---

## 2. 数据集格式

本项目用一套统一的 **Canonical Episode** 格式（见 [`canonical_dataset.py`](../../examples/robot_foundation_models/common/canonical_dataset.py)）。每条数据帧（frame）包含：

| 字段 | 含义 | 形状 / 类型 |
|:-----|:-----|:-----------|
| `observation.images.<cam>` | 相机图像 | `(H, W, 3)` uint8 |
| `observation.state` | 机器人状态（关节角、物体位姿等） | `(state_dim,)` float32 |
| `action` | 执行的动作 | `(action_dim,)` float32 |
| `language` | 语言指令（每帧重复） | str |
| `timestamps` / `reward` / `success` | 时间戳 / 奖励 / 是否成功 | float / float / bool |
| `task` | 任务描述（episode 级） | str |

PushCube 示例（"图像 + 状态 + 语言 → 动作"正是 VLA 模型的输入输出）：

```python
{
    "task": "push the red cube to the target",
    "control_frequency": 20.0,
    "timestamps": [0.0, 0.05, 0.10, ...],          # (T,)
    "observation": {
        "images": {"front": [(128,128,3), ...]},    # 图像
        "state":  [(14,), ...],                     # 14-D 状态
    },
    "action":   [(2,), ...],                        # 2-D 动作 [dx, dy]
    "language": ["push the red cube...", ...],      # 每帧重复
    "reward":   [0.0, ...], "success": [False, ..., True],
}
```

`canonical_dataset.py` 还提供 `compute_action_statistics()` 计算 mean / std / min / max，供第 4 节归一化使用。

---

## 3. Episode vs Frame

这是机器人数据最关键、也最容易出错的概念。

- **Episode（回合）**：机器人从开始到结束（成功或超时）的一条完整轨迹，由连续 frame 组成。
- **Frame（帧）**：一个时间步的样本 `(obs_t, action_t)`。

### 为什么不能随机打乱 frame 来划分 train/val？

一条 episode 内相邻 frame 高度相关——`obs_t` 和 `obs_{t+1}` 几乎一样。若按 frame 随机划分，同一条轨迹的前半段可能在训练集、后半段在验证集，模型只需"记住同一条轨迹"就能在验证集上作弊，导致**严重的数据泄漏（data leakage）**，验证指标虚高。

### 正确做法：按 episode 划分

整条 episode 要么全在训练集，要么全在验证集：

```
错误: 把所有 frame 打乱, 随机抽 20% 做验证   ← 同一轨迹跨集合
正确: 把 50 条 episode 打乱, 抽 10 条做验证  ← 轨迹不跨集合
```

本项目 PushCube 默认划分：50 条 episode → **40 训练 / 5 验证 / 5 测试**（见 [`train_lightweight_vla.py`](../../examples/robot_foundation_models/smolvla/train_lightweight_vla.py) 的 `split_episodes`）。

> **铁律**：评估策略泛化能力时，验证/测试的 episode 必须是训练时**完全没见过的初始条件**。

<div class="dof-concept" role="group" aria-label="Episode-first dataset split">
  <span class="dof-concept__eyebrow">Episode first · frame second</span>
  <p class="dof-concept__title">先按完整轨迹划分，再在各自集合内采样 frame；同一条轨迹永远不跨集合。</p>
  <div class="dof-concept-grid">
    <div class="dof-concept-panel dof-concept-panel--warn">
      <span>Wrong · 数据泄漏</span>
      <strong>随机切 frame</strong>
      <small>相邻帧会同时进入训练与验证，离线指标可能虚高。</small>
    </div>
    <div class="dof-concept-panel dof-concept-panel--good">
      <span>Right · 泛化检查</span>
      <strong>完整切 episode</strong>
      <small>每组拥有不同初始条件与完整轨迹，训练时再打乱本组 frame。</small>
    </div>
  </div>
  <div class="dof-episode-stack" aria-label="Three independent episode groups">
    <div class="dof-episode"><span>TRAIN episodes</span><div class="dof-episode__frames" aria-hidden="true"><i></i><i></i><i></i><i></i><i></i><i></i></div></div>
    <div class="dof-episode dof-episode--val"><span>VAL episodes</span><div class="dof-episode__frames" aria-hidden="true"><i></i><i></i><i></i><i></i></div></div>
    <div class="dof-episode dof-episode--test"><span>TEST episodes</span><div class="dof-episode__frames" aria-hidden="true"><i></i><i></i><i></i><i></i></div></div>
  </div>
</div>

---

## 4. 数据归一化

神经网络对输入尺度敏感。状态和动作的物理量纲差异很大（关节角 vs 像素值 vs 末端速度），必须归一化到相近范围。

**Mean-Std 归一化（最常用）**： $\hat{x} = (x - \mu) / (\sigma + \epsilon)$，使每维均值 0、方差 1。对动作尤其重要——预测的 MSE loss 才不会某个维度因数值大而主导。

```python
action_mean = all_actions.mean(axis=0)
action_std  = all_actions.std(axis=0) + 1e-8
action_norm  = (action - action_mean) / action_std
```

**Min-Max 归一化**： $\hat{x} = (x - x_{\min}) / (x_{\max} - x_{\min}) \cdot 2 - 1$，压到 `[-1, 1]`，适合已知明确上下界（如关节限位）的量。

**图像归一化**：只需除以 255 压到 `[0, 1]`（本项目的 `PushCubeFrameDataset` 就是 `img.astype(float32) / 255.0`）。

> **重要**：`mean / std` 必须**只用训练集计算**，再套用到验证/测试集。否则又泄漏了。

---

## 5. 训练 Pipeline

标准 PyTorch 训练流程：(1) Dataset 把 episode 列表展开成逐帧样本 `(image, state, lang, action)`；(2) DataLoader 按 `batch_size` 打包、`shuffle` 打乱（训练集打乱、验证集不打乱）；(3) 前向 `model(image, state, lang)` → 预测动作；(4) Loss：行为克隆用 MSE `||pred - action||²`；(5) 反向 + 更新；(6) 每 epoch 在验证集算 loss，保存最优 checkpoint。

```python
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
val_loader   = DataLoader(val_dataset,   batch_size=64, shuffle=False)

for epoch in range(epochs):
    model.train()
    for imgs, states, lang, actions in train_loader:
        pred = model(imgs, states, lang)
        loss = F.mse_loss(pred, actions)
        optimizer.zero_grad(); loss.backward(); optimizer.step()
```

> **shuffle 的对象**：DataLoader 的 `shuffle=True` 打乱的是**训练集内的 frame 顺序**（打破时序相关、让 batch 更多样），这和第 3 节"按 episode 划分"不矛盾——划分保证 episode 不跨集合，shuffle 只影响训练时 frame 的组合顺序。

---

## 6. 评估：离线 vs 闭环

这是机器人学习区别于普通深度学习的核心。

| 评估类型 | 做法 | 指标 | 能说明什么 |
|:---------|:-----|:-----|:-----------|
| **离线（Offline / Open-loop）** | 在留出的测试帧上算 `‖pred - action‖²` | loss / MAE / accuracy | 模型能否模仿专家的单步动作 |
| **闭环（Closed-loop）** | 把模型放进仿真器，用预测动作驱动机器人，看是否完成任务 | 成功率 | 模型能否真正完成整个任务 |

离线 loss 低 ≠ 闭环能成功。闭环里每一步的小误差会累积、状态会偏离训练分布（distribution shift），模型可能从未见过这种状态，于是彻底失败。这叫**复合误差（compounding error）**——行为克隆（BC）的经典痛点。

<div class="dof-principle" role="group" aria-label="离线模仿误差与闭环复合误差原理图">
  <p class="dof-principle__caption"><strong>原理图 · Why low offline loss can still fail closed-loop.</strong> 离线评估只比较一个已给定状态下的动作；闭环中，预测动作会改变下一时刻的状态，微小偏差因此可能不断反馈、累积。</p>
  <div class="dof-principle__canvas">
    <svg viewBox="0 0 920 330" role="img" aria-labelledby="shift-figure-title shift-figure-desc">
      <title id="shift-figure-title">Offline imitation compared with closed-loop compounding error</title>
      <desc id="shift-figure-desc">Offline evaluation compares a prediction with an expert action at one fixed observation. Closed-loop evaluation feeds the predicted action to the environment, so later observations can leave the training distribution.</desc>
      <defs>
        <marker id="shift-arrow" markerWidth="8" markerHeight="8" refX="6" refY="3" orient="auto"><path class="dof-diagram-arrow" d="M0,0 L7,3 L0,6 Z"/></marker>
        <marker id="shift-arrow-violet" markerWidth="8" markerHeight="8" refX="6" refY="3" orient="auto"><path class="dof-diagram-arrow-violet" d="M0,0 L7,3 L0,6 Z"/></marker>
      </defs>
      <rect class="dof-diagram-surface" x="32" y="34" width="380" height="262" rx="20"/>
      <text class="dof-diagram-title" x="58" y="68">Offline / one-step imitation</text>
      <rect class="dof-diagram-fill-blue" x="62" y="112" width="94" height="56" rx="12"/>
      <rect class="dof-diagram-surface" x="206" y="112" width="94" height="56" rx="12"/>
      <rect class="dof-diagram-fill-good" x="206" y="214" width="94" height="56" rx="12"/>
      <text class="dof-diagram-label" x="80" y="137">obsₜ</text><text class="dof-diagram-note" x="76" y="155">held fixed</text>
      <text class="dof-diagram-label" x="222" y="137">policy</text><text class="dof-diagram-note" x="216" y="155">predict âₜ</text>
      <text class="dof-diagram-label" x="218" y="238">expert a*ₜ</text><text class="dof-diagram-note" x="220" y="256">label</text>
      <path class="dof-diagram-accent" d="M158 140 H198" marker-end="url(#shift-arrow)"/>
      <path class="dof-diagram-dash" d="M254 171 V205"/>
      <text class="dof-diagram-math" x="315" y="146">loss(âₜ, a*ₜ)</text>
      <text class="dof-diagram-note" x="58" y="286">Does not ask what action changes next state.</text>
      <rect class="dof-diagram-surface" x="452" y="34" width="436" height="262" rx="20"/>
      <text class="dof-diagram-title" x="478" y="68">Closed loop / task execution</text>
      <rect class="dof-diagram-fill-blue" x="484" y="116" width="84" height="54" rx="12"/>
      <rect class="dof-diagram-surface" x="618" y="116" width="84" height="54" rx="12"/>
      <rect class="dof-diagram-fill-violet" x="752" y="116" width="94" height="54" rx="12"/>
      <text class="dof-diagram-label" x="500" y="139">obsₜ</text><text class="dof-diagram-note" x="494" y="157">state</text>
      <text class="dof-diagram-label" x="633" y="139">policy</text><text class="dof-diagram-note" x="631" y="157">âₜ</text>
      <text class="dof-diagram-label" x="773" y="139">environment</text><text class="dof-diagram-note" x="771" y="157">next state</text>
      <path class="dof-diagram-accent" d="M570 143 H610" marker-end="url(#shift-arrow)"/>
      <path class="dof-diagram-accent" d="M704 143 H744" marker-end="url(#shift-arrow)"/>
      <path class="dof-diagram-violet" d="M799 174 V226 H526 V177" marker-end="url(#shift-arrow-violet)"/>
      <text class="dof-diagram-note" x="594" y="246">obsₜ₊₁ may differ from the training distribution</text>
      <path class="dof-diagram-violet" d="M542 270 C600 247, 670 292, 818 260"/>
      <text class="dof-diagram-math" x="684" y="280">small errors can compound</text>
    </svg>
  </div>
</div>

---

## 7. 过拟合信号

本项目记录了一个值得进一步诊断的**训练—闭环性能差距**。SmolVLA（450M）在 PushCube 上的报告汇总如下（见 [`docs/28-smolvla-gpu-finetuning-runbook.md`](../28-smolvla-gpu-finetuning-runbook.md)）：

| 阶段 | 训练 loss | 闭环成功率 |
|:-----|:---------:|:----------:|
| 初始 | 0.47 | 0% |
| 500 步 | 0.10（best 0.028） | 0% |
| 10K 步 | **0.03**（best 0.004） | **0%** |

**能得出的结论**：报告的训练 loss 从 0.47 降到 0.03，而报告的闭环成功率仍为 0%，所以训练损失不能替代闭环任务指标。仅凭这两个数字还不能证明模型“记住了训练轨迹”；需要轨迹级留出集、变化因素留出集、开环动作误差和逐回合日志来区分过拟合、预处理错误、动作接口错误与闭环分布偏移。

> **诊断规则**：训练 loss 下降但闭环成功率不动 → 标记为“训练—闭环差距”，然后依次核查数据划分、开环泛化、输入输出归一化、控制接口和闭环状态分布。扩充多样数据、正则化、DAgger 或 RL 都是待验证方案，不应在消融前宣布唯一原因或通用数据阈值。

---

## 8. 连接项目代码

| 概念 | 项目实现 | 文件 |
|:-----|:---------|:-----|
| Canonical 数据格式 | `CanonicalEpisode` / `EpisodeBuilder` | [`common/canonical_dataset.py`](../../examples/robot_foundation_models/common/canonical_dataset.py) |
| 脚本策略采集 | 三阶段专家启发式，50 episodes | [`smolvla/collect_pushcube_dataset.py`](../../examples/robot_foundation_models/smolvla/collect_pushcube_dataset.py) |
| 按 episode 划分 | `split_episodes()` → 40/5/5 | [`smolvla/train_lightweight_vla.py`](../../examples/robot_foundation_models/smolvla/train_lightweight_vla.py) |
| 动作统计 | `compute_action_statistics()` → mean/std/min/max | [`common/canonical_dataset.py`](../../examples/robot_foundation_models/common/canonical_dataset.py) |
| LeRobot 格式转换 | `convert_to_lerobot()` | [`common/to_lerobot.py`](../../examples/robot_foundation_models/common/to_lerobot.py) |
| 闭环评估 | closed-loop success rate | [`smolvla/closed_loop_eval.py`](../../examples/robot_foundation_models/smolvla/closed_loop_eval.py) |

**数据规模**：PushCube 数据集 **50 episodes / ~1788 frames**（见 `train_lightweight_vla.py` 文件头），平均每条约 36 帧，控制频率 20 Hz。这是"教学规模"——足够跑通 pipeline，但不足以支撑任务级成功（第 7 节的过拟合即源于此）。

**LeRobot 格式**：Canonical 是项目内部标准；要喂给 HuggingFace LeRobot 训练 SmolVLA，需用 `to_lerobot.py` 转换：

```python
from canonical_dataset import load_episodes_from_dir
from to_lerobot import convert_to_lerobot
episodes = load_episodes_from_dir("datasets/pushcube_canonical/")
convert_to_lerobot(episodes, "datasets/pushcube_lerobot/")
```

转换后字段名变为 LeRobot 规范：`observation.images.front`、`observation.state`、`action`、`task`（语言指令）。

---

## 9. 可运行代码：数据集类与训练循环

用合成数据演示完整的"按 episode 划分 + Dataset + DataLoader + 训练循环"。不依赖项目数据文件，可直接运行。

```python
"""Episode 级划分 + PyTorch 训练循环演示
用合成数据模拟 10 条 episode, 演示按 episode 划分 / 动作归一化 / 训练循环。
运行: python dataset_training_demo.py   依赖: numpy, torch"""
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

torch.manual_seed(0); np.random.seed(0)

# --- 1. 合成 episode 数据 (模拟 10 条轨迹) ---
N_EPISODES, EP_LEN, STATE_DIM, ACTION_DIM = 10, 40, 4, 2
episodes = []
for ep_id in range(N_EPISODES):
    bias = np.random.randn(1).item() * 0.5   # 每条 episode 有自己的偏置 (模拟不同初始条件)
    states  = np.random.randn(EP_LEN, STATE_DIM).astype(np.float32)
    actions = (states[:, :2] + bias + 0.1 * np.random.randn(EP_LEN, 2)).astype(np.float32)
    episodes.append({"states": states, "actions": actions, "id": ep_id})

# --- 2. 按 EPISODE 划分 (绝不按 frame!) ---
rng = np.random.RandomState(42)
perm = rng.permutation(N_EPISODES)
train_eps = [episodes[i] for i in perm[:8]]   # 8 条训练
val_eps   = [episodes[i] for i in perm[8:]]    # 2 条验证
print(f"划分: train={len(train_eps)} eps, val={len(val_eps)} eps")
print(f"  train ids: {[e['id'] for e in train_eps]}  val ids: {[e['id'] for e in val_eps]}")

# --- 3. 动作归一化 (仅用训练集统计!) ---
train_actions = np.concatenate([e["actions"] for e in train_eps], axis=0)
act_mean = train_actions.mean(axis=0)
act_std  = train_actions.std(axis=0) + 1e-8
print(f"动作归一化: mean={act_mean.round(3)}, std={act_std.round(3)}")

# --- 4. Dataset 类 ---
class FrameDataset(Dataset):
    """把 episode 列表展开成逐帧样本, 并对动作做归一化."""
    def __init__(self, eps, mean, std):
        self.samples = []
        for ep in eps:
            for t in range(len(ep["actions"])):
                self.samples.append((ep["states"][t], (ep["actions"][t] - mean) / std))
    def __len__(self): return len(self.samples)
    def __getitem__(self, i):
        s, a = self.samples[i]
        return torch.from_numpy(s), torch.from_numpy(a)

train_ds = FrameDataset(train_eps, act_mean, act_std)
val_ds   = FrameDataset(val_eps,   act_mean, act_std)   # 复用训练集统计!
print(f"帧数: train={len(train_ds)}, val={len(val_ds)}")
train_loader = DataLoader(train_ds, batch_size=16, shuffle=True)
val_loader   = DataLoader(val_ds,   batch_size=16, shuffle=False)

# --- 5. 模型 + 训练循环 ---
model = nn.Sequential(nn.Linear(STATE_DIM, 32), nn.ReLU(),
                      nn.Linear(32, 32), nn.ReLU(), nn.Linear(32, ACTION_DIM))
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
best_val = float("inf")
for epoch in range(1, 51):
    model.train(); tr_loss = 0.0
    for states, actions in train_loader:
        loss = F.mse_loss(model(states), actions)
        optimizer.zero_grad(); loss.backward(); optimizer.step()
        tr_loss += loss.item()
    tr_loss /= len(train_loader)
    model.eval(); val_loss = 0.0
    with torch.no_grad():
        for states, actions in val_loader:
            val_loss += F.mse_loss(model(states), actions).item()
    val_loss /= len(val_loader)
    best_val = min(best_val, val_loss)
    if epoch % 10 == 0 or epoch == 1:
        print(f"epoch {epoch:3d} | train_loss={tr_loss:.4f} | val_loss={val_loss:.4f}")

print(f"\n训练完成. best_val_loss={best_val:.4f}")
print("注意: val_loss 是离线指标; 要判断策略能否完成任务须做闭环评估 (仿真器跑完整 episode 看成功率).")
```

### 动手实验

1. **泄漏对比**：把划分改成"所有 frame 打乱后随机分"，对比 val_loss——错误划分的 val_loss 会低很多（虚高）。
2. **归一化泄漏**：故意用全部 episode（含验证集）算 `act_mean/act_std`，观察 val_loss 被人为压低。
3. **过拟合复现**：换成大网络（如 256-256-256）训练 500 轮，观察 train_loss 持续下降而 val_loss 触底反弹。

---

## 10. 检查理解

1. **概念题**：解释为什么 episode 内相邻 frame 高度相关，以及这种相关性会让"按 frame 随机划分"产生数据泄漏。

2. **划分原则**：50 条 episode 做 80/10/10 划分，写出 train / val / test 各多少条，并说明为什么三者的 episode id 必须**互不重叠**。

3. **归一化**：mean/std 归一化和 min-max 归一化分别适合什么场景？为什么归一化统计量必须只用训练集计算？

4. **离线 vs 闭环**：SmolVLA 训练 loss 从 0.47 降到 0.03，但闭环成功率始终 0%。请用"复合误差"和"分布偏移"解释为什么 loss 低不代表能成功。

5. **过拟合诊断**：训练 loss 持续下降但验证 loss 开始上升，说明什么？此时继续加 epoch 会让闭环成功率变好还是变差？

6. **代码题**：在示例里加一个 `test` 划分（再抽 1 条 episode），训练后报告 test_loss。若 test_loss 远高于 val_loss，可能的原因有哪些？

7. **连接项目**：阅读 [`canonical_dataset.py`](../../examples/robot_foundation_models/common/canonical_dataset.py) 的 `compute_action_statistics()`，它返回哪四个统计量？再读 [`to_lerobot.py`](../../examples/robot_foundation_models/common/to_lerobot.py)，说明 canonical 转 LeRobot 时 `task` 字段和语言指令如何处理。

> 完成本 Foundations Layer（01–10）后，建议从 [`docs/01-what-is-vla.md`](../01-what-is-vla.md) 或 [`examples/unified_pushcube_vla.py`](../../examples/unified_pushcube_vla.py) 开始主线实践。
