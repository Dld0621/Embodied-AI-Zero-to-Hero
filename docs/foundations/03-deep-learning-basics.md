# 深度学习基础

> **前置要求**: [`02-linear-algebra.md`](02-linear-algebra.md)（向量、矩阵乘法）、[`01-python-for-robotics.md`](01-python-for-robotics.md)（NumPy 基础）
> **预计学习时间**: 3–4 小时
> **完成后你能**: 用 PyTorch 从零写一个 MLP 并训练它；看懂反向传播在做什么；读懂本项目里 State-BC 的训练循环与 loss 曲线

---

## 1. 神经元的直觉：一个神经元在做什么？

把一个神经元想象成一个"加权投票器"。它接收若干输入，每个输入乘以一个权重（权重表示该输入有多重要），全部相加后加一个偏置，最后通过一个**激活函数**决定输出：

```
输入 x1 ──(w1)──┐
输入 x2 ──(w2)──┼──► z = w1·x1 + w2·x2 + b ──► f(z) ──► 输出 a
输入 x3 ──(w3)──┘
```

用矩阵写就是两步：`z = W·x + b`，`a = f(z)`。`W` 是权重矩阵，`b` 是偏置向量。

**机械类比**：这像做受力分析——多个力（输入）按各自方向系数（权重）叠加得到合力 `z`，再判断是否超过阈值（激活函数）才"动作"。

### 为什么要分多层？

单层神经元只能画一条直线（超平面）来切分数据。但真实关系是非线性的：推立方体时"该绕到它后面再推"这种决策，一条直线表达不了。多层网络逐层组合：

- 第 1 层学到简单几何量（"手臂在立方体哪一侧"）
- 第 2 层把几何量组合成决策（"该靠近还是该推"）
- 更深的非线性叠加，理论上可逼近任意连续函数（万能逼近定理）

**关键前提**：层与层之间必须有非线性激活函数，否则多层线性变换叠起来仍等价于一层线性变换。

---

## 2. 前向传播：线性变换 + 激活函数

数据从输入逐层流向输出，叫**前向传播**（forward pass）。每一层做两件事：

1. **线性变换** `z = W·x + b`：`W` 旋转/缩放输入空间，`b` 平移它。
2. **激活函数** `a = f(z)`：引入非线性。

### 常见激活函数

| 激活函数 | 公式 | 特点 | 直觉 |
|---------|------|------|------|
| **ReLU** | `max(0, z)` | 计算快、不易饱和，最常用 | 像单向阀：负值堵住为 0，正值全通过 |
| **GELU** | `z·Φ(z)` | 平滑的 ReLU，Transformer 标配 | 像"软"的单向阀，过渡更平滑 |
| **sigmoid** | `1/(1+e⁻ᶻ)` | 输出压到 (0,1)，可当概率 | 两端饱和、梯度易消失，深层少用 |
| **tanh** | `(eᶻ−e⁻ᶻ)/(eᶻ+e⁻ᶻ)` | 输出在 (−1,1)，零中心 | 适合做动作输出的最后一步 |

> **为什么 ReLU 这么常用？** 它在正区间梯度恒为 1，不会像 sigmoid 那样在两端"梯度消失"，使得深层网络也能训练。本项目 State-BC 的隐藏层全用 ReLU，最后一层用 tanh 把动作压到 `[-1, 1]`。

---

## 3. 损失函数：衡量"错得有多离谱"

训练的目标是让预测接近真实值。**损失函数**（loss）量化这个差距，是个标量——越小越好。

| 损失函数 | 公式 | 何时用 | 项目中的例子 |
|---------|------|--------|------------|
| **MSE**（均方误差） | `mean((ŷ−y)²)` | 连续值回归，对大误差敏感 | State-BC 预测 2 维动作增量用 `F.mse_loss` |
| **L1**（平均绝对误差） | `mean(|ŷ−y|)` | 回归，但对离群点更鲁棒 | OpenVLA-OFT 动作头用 L1 回归 |
| **Cross-Entropy**（交叉熵） | `−Σ y·log(ŷ)` | 分类（离散类别） | 把动作离散成 token 的 VLA（如 RT-2） |

**直觉**：MSE 像"惩罚按误差平方放大"，所以一个差 2 的样本比两个差 1 的样本贡献更多 loss；L1 是线性惩罚，更宽容离群点；交叉熵则专为"选哪一个类别"设计，配 softmax 使用。

> 注意：机器人连续动作（关节角、末端增量）几乎都用 MSE 或 L1；只有把动作离散成 token 时才用交叉熵。

---

## 4. 反向传播：链式法则与计算图

前向传播算出 loss 后，我们要知道"每个参数该往哪个方向调一点点，才能让 loss 下降"。这就是**反向传播**（backpropagation）。

### 计算图与链式法则

前向传播天然形成一张有向无环图（计算图）。例如两层网络：

```
x ──► z1=W1·x+b1 ──► a1=ReLU(z1) ──► z2=W2·a1+b2 ──► ŷ ──► loss
```

要算 `loss` 对 `W1` 的梯度，就沿图**从后往前**逐层用链式法则相乘：

```
∂loss/∂W1 = (∂loss/∂ŷ) · (∂ŷ/∂z2) · (∂z2/∂a1) · (∂a1/∂z1) · (∂z1/∂W1)
```

**机械类比**：梯度就是"loss 对参数的灵敏度"——告诉我"这个螺丝拧松 0.01 圈，误差会变大还是变小、变多少"。PyTorch 的 `loss.backward()` 自动把整条链乘好，把梯度存进每个参数的 `.grad` 里。

> 梯度指向 loss **上升最快**的方向，所以更新参数时要**减去**梯度——这就是梯度下降。

---

## 5. 优化器：怎么走下坡路

知道梯度后，"走多大一步、怎么走"由**优化器**决定。

| 优化器 | 直觉 | 特点 |
|--------|------|------|
| **SGD** | 小球沿斜坡滚下，步长=学习率×梯度 | 简单，但各方向步长一样，易震荡、卡在鞍点 |
| **Momentum** | 给球加惯性，积累历史方向 | 平滑震荡、冲过局部小坑 |
| **Adam** | 每个参数有自己的"自适应步长"（基于梯度一阶/二阶矩） | 几乎不用调参，默认首选 |
| **AdamW** | Adam + 解耦的权重衰减 | 大模型微调标配（VLA/LLM 几乎都用它） |

**为什么 Adam 好用？** 梯度一直很大的参数，它会把步长自动调小（防止发散）；梯度一直很小的参数，它会把步长相对放大（防止停滞）。本项目 State-BC 用 `torch.optim.Adam(lr=3e-3)`，SmolVLA 微调用 `AdamW(lr=1e-4)`。

---

## 6. 训练循环：五步心法

所有监督学习训练，本质都是这个循环（项目里 `train_state_policy` 一字不差地体现）：

```python
for epoch in range(epochs):
    for x, y in dataloader:
        pred  = model(x)                 # 1. 前向
        loss  = loss_fn(pred, y)         # 2. 算损失
        optimizer.zero_grad()            # 3. 清旧梯度
        loss.backward()                  # 4. 反向传播算梯度
        optimizer.step()                 # 5. 更新参数
```

五个动作缺一不可：忘了 `zero_grad` 会把梯度累加（除非你故意要梯度累积）；忘了 `backward` 参数根本不动；忘了 `step` 白算梯度。一个 epoch = 把训练数据完整过一遍。

---

## 7. 过拟合与正则化

**过拟合**：模型把训练集"背"下来了，但没学到规律，遇到新数据就崩。典型表现：训练 loss 一直降，验证 loss 先降后升。

本项目里有两个**真实的过拟合案例**：

1. **LightweightVLA**（`examples/robot_foundation_models/smolvla/models/lightweight_vla/training_history.json`）：100 epoch 里 `train_loss` 从 0.369 一路降到 0.210，而 `val_loss` 在第 8 epoch 达到最佳 0.316 后**反弹**到 0.525。这是教科书式的过拟合曲线。
2. **SmolVLA 10K 步微调**（见 `docs/28-smolvla-gpu-finetuning-runbook.md`）：训练 loss 从 0.10 降到 0.03（降了 3 倍），但闭环成功率仍是 0%——模型背会了训练轨迹，却无法泛化到新初始条件。

### 常用对抗手段

| 方法 | 做法 | 直觉 |
|------|------|------|
| **Dropout** | 训练时随机把一部分神经元输出置 0 | 不让模型过度依赖任何单个神经元，强迫它学冗余表示 |
| **Weight Decay**（权重衰减） | 在 loss 里加 `λ·‖W‖²` | 惩罚过大的权重，让模型更"温和"，不易记住噪声 |
| **早停**（Early Stopping） | 监控验证 loss，一旦上升就停 | 在"开始背书"前停下 |
| **数据增强 / 更多数据** | 扩充训练集分布 | 根本解法：见过的越多越难死记 |

> AdamW 的 "W" 就是把 weight decay 做对了——与 Adam 的自适应更新解耦，避免衰减被梯度缩放扭曲，所以大模型微调都用 AdamW 而非 Adam+L2。

---

## 8. 连接到项目

你现在能读懂本仓库里的训练代码了：

- **State-BC 达到 90% 成功**（`examples/unified_pushcube_vla.py` 的 `build_state_policy` / `train_state_policy`，结果见 `results/benchmarks/pushcube_summary.json`）：
  - 架构就是 MLP：`state(14) + 几何特征(13) + 语言嵌入(16) → 128 → 128 → 2 → tanh`
  - 隐藏层 ReLU、输出 tanh、BatchNorm 稳定输入、MSE 损失、Adam 优化器、Cosine 学习率衰减、梯度裁剪（`max_norm=1.0`）
  - 50 epoch 后在 PushCube 上 **90.0% 成功率**，证明"仅靠结构化状态 + MLP"就能学会推方块
- **SmolVLA 微调 loss 0.47→0.03**（`docs/28-smolvla-gpu-finetuning-runbook.md`）：
  - 450M 参数、LoRA 只训 100M、bf16、RTX 3060
  - 500 步：loss 0.47→0.10（最佳 0.028）；续训到 10K 步：0.10→0.03（最佳 0.004）
  - 但闭环成功率仍 0%——这就是 **BC 过拟合**：open-loop loss 越来越低，closed-loop 却不灵
- **Tiny-VLA 反例**：同一个脚本里只用 CNN+MLP 的视觉策略，50 epoch 后 train_loss 降到 0.013，但成功率 0%——模型太小、数据太少（100 episodes），学不动

> 启示：loss 低 ≠ 策略好。机械工程里"仿真跑得很顺，上实物就崩"和这里的"训练 loss 很低，闭环就废"是同一个道理——**泛化**才是关键。

---

## 9. 动手代码：用 PyTorch 训练一个 MLP

下面这段代码可直接运行（`pip install torch`）。它用一个两层 MLP 去拟合非线性函数 `y = sin(x) + 噪声`，完整演示前向→损失→反向→更新，并打印 loss 下降曲线。

```python
# deep_learning_basics.py — 可直接运行
import torch
import torch.nn as nn
import math

torch.manual_seed(0)

# 1. 合成数据：y = sin(x) + 小噪声
x = torch.linspace(-math.pi, math.pi, 200).unsqueeze(1)        # (200, 1)
y = torch.sin(x) + 0.05 * torch.randn_like(x)                 # (200, 1)

# 2. 定义 MLP：1 -> 32 (ReLU) -> 32 (ReLU) -> 1
class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1, 32), nn.ReLU(),
            nn.Linear(32, 32), nn.ReLU(),
            nn.Linear(32, 1),          # 回归，输出不加激活
        )
    def forward(self, x):
        return self.net(x)

model = MLP()
loss_fn = nn.MSELoss()                              # 回归用 MSE
optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)   # Adam，默认首选

# 3. 训练循环（五步心法）
losses = []
for epoch in range(300):
    pred = model(x)                  # 前向
    loss = loss_fn(pred, y)          # 损失
    optimizer.zero_grad()            # 清梯度
    loss.backward()                  # 反向传播
    optimizer.step()                 # 更新参数
    losses.append(loss.item())

# 4. 打印 loss 曲线（纯文本，无需 matplotlib）
print("epoch   loss    (ASCII 曲线, 每个 # ≈ 0.02)")
for epoch in range(0, 300, 30):
    bar = "#" * int(losses[epoch] / 0.02)
    print(f"{epoch:4d}  {losses[epoch]:.4f}  {bar}")
print(f"final loss = {losses[-1]:.4f}   (初始 {losses[0]:.4f})")
```

预期输出（`torch.manual_seed(0)` 固定，结果可复现）：

```
epoch   loss    (ASCII 曲线, 每个 # ≈ 0.02)
   0  0.6426  ################################
  30  0.0455  ##
  60  0.0045
  90  0.0025
 120  0.0023
 150  0.0022
 270  0.0021
final loss = 0.0021   (初始 0.6426)
```

loss 从 ~0.64 降到 ~0.002，说明 MLP 学会了 `sin` 的非线性形状。把 `nn.ReLU()` 换成 `nn.Identity()`（即去掉非线性）再跑一次，你会发现 loss 卡住降不下去——这就验证了第 1 节"必须有非线性"的结论。

> 对照阅读：把上面 `MLP` 的结构和 `train` 循环，与 `examples/unified_pushcube_vla.py` 里的 `StateSCPolicy` 和 `train_state_policy` 逐行对比——你会发现项目代码就是这段教学代码的"加强版"（多了 BatchNorm、几何特征、语言嵌入、梯度裁剪、Cosine 学习率）。

---

## 10. 检查理解

试着回答下面问题（答案可在文中找到，建议先合上文档自己想）：

1. **概念题**：如果把两层 `Linear` 直接堆叠、中间不加任何激活函数，它和一层 `Linear` 等价吗？为什么？用矩阵乘法的结合律解释。
2. **选型题**：你要训练一个策略输出 7 维连续关节增量，该选 MSE、L1 还是 Cross-Entropy？如果改成把每个关节角度离散成 256 个 bin 的 token 呢？
3. **梯度题**：反向传播为什么要"从后往前"？如果忘了调用 `optimizer.zero_grad()`，会发生什么？
4. **优化器题**：用一句话解释 Adam 相比朴素 SGD 的优势。为什么 SmolVLA 微调用 AdamW 而不是 Adam？
5. **过拟合题**：看 LightweightVLA 的曲线——`train_loss` 在降、`val_loss` 在升，这说明什么？给出三种缓解方法，并指出哪种是"根本解法"。
6. **项目题**：State-BC 的输出层为什么用 `tanh` 而不是 `ReLU`？（提示：动作取值范围）SmolVLA 的 loss 从 0.47 降到 0.03，但闭环成功率仍是 0%，这属于什么现象？
7. **动手题**：运行第 9 节代码，记录 final loss；然后把隐藏层宽度从 32 改成 4，loss 还能降下来吗？再把训练数据从 200 点减到 20 点，观察是否出现过拟合迹象。

> 完成本节后，进入 [`04-transformer-basics.md`](04-transformer-basics.md)：把"一个神经元"升级成"一组互相注意的神经元"，理解 VLA 模型的骨干为什么是 Transformer。
