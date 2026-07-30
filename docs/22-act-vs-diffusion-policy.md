# ACT vs Diffusion Policy：模仿学习 Baseline 对比

> **目标**：理解两种主流模仿学习方法的差异——ACT（Action Chunking with Transformers）通过自回归生成动作块，Diffusion Policy 通过去噪扩散建模动作分布——并在统一 PushCube 任务上运行最小实现。

---

## 目录

1. [为什么需要对比这两个 Baseline？](#1-为什么需要对比这两个-baseline)
2. [核心思想一句话概括](#2-核心思想一句话概括)
3. [ACT 详解](#3-act-详解)
4. [Diffusion Policy 详解](#4-diffusion-policy-详解)
5. [架构对比表](#5-架构对比表)
6. [在 PushCube 上的最小实现](#6-在-pushcube-上的最小实现)
7. [实验结果与选择建议](#7-实验结果与选择建议)
8. [常见问题](#8-常见问题)

---

## 1. 为什么需要对比这两个 Baseline？

在机器人模仿学习中，行为克隆（Behavior Cloning, BC）是最基础的起点。但标准 BC 存在两个核心问题：

- **复合误差（Compounding Errors）**：单步预测的微小误差会在长序列中累积，导致轨迹偏离。
- **多模态动作（Multimodal Actions）**：同一观测可能对应多种合理动作（如绕开障碍物的左右两条路径），标准 MSE 损失会迫使网络预测平均值，导致模糊、无效的动作。

ACT 和 Diffusion Policy 分别用不同思路解决这两个问题：

| 方法 | 解决的核心问题 | 关键洞察 |
|------|--------------|---------|
| **ACT** | 复合误差 | 一次预测未来 T 步动作块，减少推理频率 |
| **Diffusion Policy** | 多模态动作 | 用扩散模型建模动作分布，而非单点估计 |

---

## 2. 核心思想一句话概括

- **ACT**：Transformer 编码器处理图像+语言，CVAE 解码器生成未来 T 步的动作块。
- **Diffusion Policy**：将动作生成视为去噪过程——从随机噪声出发，通过条件扩散模型逐步去噪得到合理动作序列。

---

## 3. ACT 详解

### 3.1 架构

```
Input: 图像 (B, T_obs, 3, H, W) + 语言 token (B, L)
    ├── Vision Encoder (ResNet / ViT) -> 图像特征 (B, T_obs, D_v)
    ├── Language Encoder (BERT / 简单 Embedding) -> 语言特征 (B, D_l)
    ├── Concat -> 联合特征 (B, T_obs, D_v + D_l)
    └── Transformer Encoder -> 上下文特征 (B, T_obs, D)

CVAE Decoder:
    ├── 动作块作为 "target token"
    ├── 风格变量 z ~ N(0, I) 注入随机性
    └── Transformer Decoder -> 输出动作块 (B, T_action, action_dim)
```

### 3.2 关键设计

**动作分块（Action Chunking）**

不是每步都预测一个动作，而是每 K 步预测一次未来 T 步的动作块：

```python
chunk = policy.predict(observation)  # (T_action, action_dim)
for t in range(T_action):
    env.step(chunk[t])
    if done:
        break
```

**时间集成（Temporal Ensembling）**

重叠的 action chunk 可以通过加权平均平滑：

```python
# chunk[0] 被当前步执行，chunk[1] 被下一步执行...
# 如果有重叠，对新预测和旧预测的对应步做加权平均
action = 0.3 * new_chunk[0] + 0.7 * old_chunk[1]
```

**CVAE 训练目标**

```
L = L_reconstruction + beta * L_KL

L_reconstruction = MSE(pred_action_chunk, true_action_chunk)
L_KL = KL(q(z|obs, action) || p(z))
```

### 3.3 推理时的动作队列

```python
class ACTPolicy:
    def __init__(self):
        self.action_queue = deque()
        self.chunk_size = T_action

    def predict(self, obs):
        if not self.action_queue:
            # 队列为空，重新生成一个 chunk
            chunk = self.model(obs)
            self.action_queue.extend(chunk)
        return self.action_queue.popleft()
```

---

## 4. Diffusion Policy 详解

### 4.1 核心洞察

标准 BC 用 MSE 损失：`L = ||a_pred - a_true||^2`

当同一观测对应多个合理动作时（多模态），网络被迫预测平均值：

```
真实动作分布:  a1 ~ 向左绕, a2 ~ 向右绕
MSE 最优解:    a_pred = (a1 + a2) / 2  ->  直撞障碍物（不合理）
```

Diffusion Policy 直接建模条件动作分布 `p(a | obs)`，通过扩散过程采样得到合理动作。

### 4.2 扩散过程

**前向过程（训练时）**：逐步给动作加噪声

```python
a_t = sqrt(alpha_t) * a_0 + sqrt(1 - alpha_t) * epsilon
# a_0: 原始动作
# epsilon: 标准高斯噪声
# alpha_t: 预定义的噪声 schedule
```

**反向过程（推理时）**：逐步去噪，恢复动作

```python
a_T ~ N(0, I)  # 从纯噪声开始
for t in reversed(range(T)):
    epsilon_pred = noise_prediction_net(a_t, t, observation)
    a_{t-1} = denoise_step(a_t, epsilon_pred, t)
return a_0
```

### 4.3 网络架构

Diffusion Policy 可以使用多种 backbone：

| Backbone | 特点 | 适用场景 |
|----------|------|---------|
| **MLP** | 简单、快速 | 低维状态输入 |
| **1D CNN** | 捕捉动作序列的局部时序 | 动作块生成 |
| **Transformer** | 长程依赖、与 ACT 共享编码器 | 图像+语言条件 |

### 4.4 训练目标

```python
def diffusion_loss(model, observation, action_true):
    t = random.randint(0, T-1)
    epsilon = torch.randn_like(action_true)
    a_noisy = sqrt(alpha[t]) * action_true + sqrt(1 - alpha[t]) * epsilon
    epsilon_pred = model(a_noisy, t, observation)
    return MSE(epsilon_pred, epsilon)
```

---

## 5. 架构对比表

| 维度 | ACT | Diffusion Policy |
|------|-----|------------------|
| **核心机制** | Transformer + CVAE 编码-解码 | 条件扩散去噪 |
| **输出形式** | 确定性动作块（+CVAE 随机性） | 从分布中采样 |
| **处理多模态** | CVAE 的隐变量 z 提供有限多样性 | 扩散过程天然支持多模态 |
| **推理速度** | 快（单次前向） | 慢（需多步去噪，通常 10-100 步）|
| **动作平滑性** | 好（块内自回归，块间时间集成） | 取决于去噪步数和 schedule |
| **训练稳定性** | 高 | 中（需仔细调 noise schedule）|
| **超参数敏感度** | 中（chunk size, beta） | 高（diffusion steps, beta schedule）|
| **硬件要求** | 低 | 中（推理时需多步迭代）|
| **典型应用场景** | 需要高速推理的实时控制 | 需要多模态动作多样性的复杂操作 |

---

## 6. 在 PushCube 上的最小实现

以下代码基于统一 PushCube 环境，实现 ACT 和 Diffusion Policy 的最小版本。

完整代码见：
- [`examples/unified_pushcube_act.py`](../examples/unified_pushcube_act.py)
- [`examples/unified_pushcube_diffusion.py`](../examples/unified_pushcube_diffusion.py)

### 6.1 ACT 核心片段

```python
class MinimalACT(nn.Module):
    """最简 ACT：CNN 编码图像，Transformer 生成动作块。"""

    def __init__(self, action_dim=2, chunk_size=10, hidden_dim=64):
        super().__init__()
        self.chunk_size = chunk_size

        # 图像编码器
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 16, 5, stride=2, padding=2),
            nn.ReLU(),
            nn.Conv2d(16, 32, 5, stride=2, padding=2),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(32 * 32 * 32, hidden_dim),
        )

        # Transformer 编码器
        encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=4, batch_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=2)

        # 动作解码头
        self.action_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim * chunk_size),
        )

    def forward(self, image):
        # image: (B, 3, 128, 128)
        feat = self.cnn(image)  # (B, hidden_dim)
        feat = feat.unsqueeze(1)  # (B, 1, hidden_dim)
        encoded = self.encoder(feat)  # (B, 1, hidden_dim)
        actions = self.action_head(encoded[:, 0])  # (B, action_dim * chunk_size)
        return actions.view(-1, self.chunk_size, action_dim)
```

### 6.2 Diffusion Policy 核心片段

```python
class MinimalDiffusionPolicy(nn.Module):
    """最简 Diffusion Policy：1D CNN 去噪网络。"""

    def __init__(self, action_dim=2, obs_dim=32, hidden_dim=64, n_diffusion_steps=20):
        super().__init__()
        self.n_steps = n_diffusion_steps

        # 时间步嵌入
        self.time_embed = nn.Embedding(n_diffusion_steps, hidden_dim)

        # 观测编码器
        self.obs_encoder = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
        )

        # 1D CNN 去噪网络
        self.noise_pred = nn.Sequential(
            nn.Linear(action_dim + hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
        )

        # 预计算扩散 schedule
        self.register_buffer("betas", torch.linspace(1e-4, 0.02, n_diffusion_steps))
        alphas = 1.0 - self.betas
        self.register_buffer("alphas_cumprod", torch.cumprod(alphas, dim=0))

    def forward(self, noisy_action, t, obs_feat):
        # noisy_action: (B, action_dim)
        # t: (B,) int
        # obs_feat: (B, obs_dim)
        t_emb = self.time_embed(t)  # (B, hidden_dim)
        obs_emb = self.obs_encoder(obs_feat)  # (B, hidden_dim)
        inp = torch.cat([noisy_action, t_emb, obs_emb], dim=-1)
        return self.noise_pred(inp)  # (B, action_dim)

    def sample(self, obs_feat):
        # 从噪声开始去噪
        action = torch.randn(1, action_dim)
        for t in reversed(range(self.n_steps)):
            t_batch = torch.tensor([t])
            epsilon_pred = self.forward(action, t_batch, obs_feat)
            # 单步去噪（简化版 DDPM）
            alpha_t = self.alphas_cumprod[t]
            action = (action - torch.sqrt(1 - alpha_t) * epsilon_pred) / torch.sqrt(alpha_t)
            if t > 0:
                action += torch.sqrt(self.betas[t]) * torch.randn_like(action)
        return action
```

### 6.3 运行命令

```bash
cd examples

# ACT
python unified_pushcube_act.py --n-episodes 100 --epochs 30 --chunk-size 10

# Diffusion Policy
python unified_pushcube_diffusion.py --n-episodes 200 --epochs 50 --diffusion-steps 20
```

---

## 7. 实验结果与选择建议

### 7.1 预期结果（基于 PushCube 合成数据）

| 方法 | 成功率 | 推理延迟 | 训练时间 | 备注 |
|------|--------|---------|---------|------|
| 标准 BC | ~5% | 极快 | 短 | 单步预测，复合误差严重 |
| ACT | ~15-25% | 快 | 中 | 动作块减少复合误差 |
| Diffusion Policy | ~10-20% | 慢（多步去噪）| 长 | 多模态支持，但轻量任务优势不明显 |

> 注：PushCube 是 lightweight 任务，多模态动作需求不强，因此 ACT 的动作块设计比 Diffusion 的多模态建模更有优势。在真实复杂操作任务中，Diffusion Policy 通常表现更好。

### 7.2 选择建议

**选择 ACT，如果：**
- 需要**实时推理**（控制频率 > 10 Hz）
- 任务动作分布**相对单峰**（没有多种等价执行路径）
- 计算资源有限
- 需要**动作平滑性**（时间集成天然提供）

**选择 Diffusion Policy，如果：**
- 任务存在**明显的多模态动作**（如左右都可以绕开障碍物）
- 可以承受**更高的推理延迟**
- 有**充足的训练数据和计算资源**
- 需要生成**多样化的候选动作**（用于规划或安全评估）

---

## 8. 常见问题

**Q: ACT 的 chunk size 怎么选？**

A: 典型值 10-50。越大则推理频率越低、复合误差越小，但预测难度增加。建议从 10 开始，根据任务时序长度调整。

**Q: Diffusion Policy 的推理延迟太高怎么办？**

A: 三种优化方向：(1) 减少去噪步数（如从 100 减到 10，用 DDIM 加速）；(2) 使用更快的 backbone（MLP 替代 Transformer）；(3) 蒸馏为单步生成器。

**Q: 能否将两者结合？**

A: 可以。一种思路是用 Diffusion Policy 生成动作块的初始分布，再用 ACT 的 Transformer 做精细调整；或者将 Diffusion 作为数据增强器，为 ACT 提供更多样化的训练样本。

**Q: 在 PushCube 上为什么成功率都不高？**

A: PushCube 的启发式策略本身成功率就有限（约 30-40%），而 BC 只能从演示数据中学习，无法超越演示者。要提高成功率，需要：(1) 更多演示数据；(2) 更优的演示策略；（3）引入 RL 微调。

---

## 参考文献

1. Zhan et al., "Learning Fine-Grained Bimanual Manipulation with Low-Cost Hardware", RSS 2023 (ACT)
2. Chi et al., "Diffusion Policy: Visuomotor Policy Learning via Action Diffusion", RSS 2023
3. Brohan et al., "RT-1: Robotics Transformer for Real-World Control at Scale", RSS 2023
4. Shafiullah et al., "Behavior Transformers: Cloning k modes with one stone", NeurIPS 2022
