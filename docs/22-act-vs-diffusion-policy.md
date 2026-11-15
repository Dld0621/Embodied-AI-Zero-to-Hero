# Minimal Action-Chunking Policy vs Diffusion Policy：模仿学习 Baseline 对比

> **内容状态：已读，仍有已确认待修项（2026-09-05）。** 正文 DDPM sampler 存在未定义 n_steps、方差与确定性表述问题（补审 F14）；目前未在 PyTorch 中验证这段实现。保留作待修教学材料，不把它当标准算法复现。 具体位置与原始来源见 [补充独立审查](reviews/remaining-source-review.md)。

> **逐点图解 / Concept close-ups：**[行为克隆与协变量偏移](knowledge-atlas/learning-behavior-cloning/index.md)。每个小点配原理、算例、图、自测；这是中文细解，保留英文术语。

> **目标**：理解两种主流模仿学习方法的差异——Minimal Action-Chunking Policy 通过 Transformer 编码器并行生成动作块，Diffusion Policy 通过去噪扩散建模动作分布——并在统一 PushCube 双方块任务上运行最小实现。

---

## 目录

1. [为什么需要对比这两个 Baseline？](#1-为什么需要对比这两个-baseline)
2. [核心思想一句话概括](#2-核心思想一句话概括)
3. [Minimal Action-Chunking Policy 详解](#3-minimal-action-chunking-policy-详解)
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

Action Chunking 和 Diffusion Policy 分别用不同思路解决这两个问题：

| 方法 | 解决的核心问题 | 关键洞察 |
|------|--------------|---------|
| **Action-Chunking Policy** | 复合误差 | 一次预测未来 T 步动作块，减少推理频率 |
| **Diffusion Policy** | 多模态动作 | 用扩散模型建模动作分布，而非单点估计 |

> **重要声明**：本仓库实现的 Action-Chunking Policy **不是完整 ACT**（Zhao et al., 2023）。完整 ACT 还需要 CVAE 隐变量、Transformer Decoder 和更完整的时间集成。本实现是教学版，只保留了多帧观测 token、语言条件、时间位置编码和简化时间集成。详见 [第 3 节](#3-minimal-action-chunking-policy-详解)。

---

## 2. 核心思想一句话概括

- **Minimal Action-Chunking Policy**：Transformer 编码器处理 K 帧图像 token + 1 个语言 token，**并行**输出未来 T 步动作块；重叠块通过指数加权进行时间集成。
- **Diffusion Policy**：将动作生成视为去噪过程——从随机噪声出发，通过条件扩散模型逐步去噪得到合理动作序列（整个 action horizon）。

两者都接收 **语言指令** 作为额外条件输入，以区分双方块场景中需要推动哪个方块。

---

## 3. Minimal Action-Chunking Policy 详解

### 3.1 与完整 ACT 的区别

完整 ACT（Zhao et al., 2023, "Learning Fine-Grained Bimanual Manipulation with Low-Cost Hardware"）包含三个核心组件：

| 组件 | 完整 ACT | 本仓库实现 |
|------|---------|-----------|
| **CVAE 隐变量** | 有，从后验 q(z\|o,a) 采样，KL 正则 | **无** |
| **多帧观测 token** | 编码器和解码器两侧都有 | **有**（K 帧视觉 token + 1 语言 token） |
| **时间集成** | 指数加权聚合重叠块 | **有**（简化版指数加权） |
| **输出方式** | 并行输出动作块 | **并行输出**（非自回归） |
| **时间位置编码** | Transformer 内置 | **有**（learned embedding） |

因此本实现诚实地命名为 **Minimal Action-Chunking Policy**，而非 ACT。

### 3.2 架构

```
Input: K 帧图像 (B, K, 3, 128, 128) + 语言 token (B, L)
    ├── Vision Encoder (CNN) -> K 个视觉 token (B, K, hidden_dim)
    ├── Language Encoder (Embedding + FC) -> 1 个语言 token (B, 1, hidden_dim)
    ├── Concat -> (B, K+1, hidden_dim)
    ├── + Temporal Positional Encoding (learned, K+1 positions)
    └── Transformer Encoder (2 layers, 4 heads) -> (B, K+1, hidden_dim)

Action Head:
    └── 取语言 token 位置 -> MLP -> (B, T * action_dim) -> reshape (B, T, action_dim)
```

### 3.3 关键设计

**多帧观测 token + 时间位置编码**

每一帧图像独立编码为一个 token，再加上一个语言 token，形成长度为 K+1 的序列：

```python
# 视觉编码：每帧独立 -> K 个 token
imgs_flat = images.reshape(B * K, C, H, W)
vis_feats = self.encode_frame(imgs_flat)     # (B*K, hidden_dim)
vis_feats = vis_feats.reshape(B, K, -1)      # (B, K, hidden_dim)

# 语言编码 -> 1 个 token
w = self.word_embed(text_tokens).mean(dim=1)
lang_feat = self.lang_fc(w).unsqueeze(1)     # (B, 1, hidden_dim)

# 拼接
all_tokens = torch.cat([vis_feats, lang_feat], dim=1)  # (B, K+1, hidden_dim)

# 时间位置编码
positions = torch.arange(K + 1, device=all_tokens.device)
pos_emb = self.temporal_pos(positions)       # (K+1, hidden_dim)
all_tokens = all_tokens + pos_emb.unsqueeze(0)
```

**时间位置编码是必要的**：没有它，Transformer 无法区分 K 帧视觉 token 的时间顺序，因为 self-attention 是排列不变的。加入 learned embedding 后，模型可以区分"最近的帧"和"最早的帧"。

**并行输出（非自回归）**

动作块通过 `action_head` 一次性并行生成，而非逐动作自回归解码：

```python
# 取语言 token 位置（最后一个）的编码输出
actions = self.action_head(encoded[:, -1])    # (B, T*action_dim)
return actions.reshape(-1, self.chunk_size, self.action_dim)
```

**指数加权时间集成**

重叠的动作块通过指数加权聚合，新预测权重更高：

```python
# 对当前步有贡献的所有 chunk 按距离加权
weight = exp(-decay * (current_step - chunk_start))
action = sum(weight * chunk[current_step - chunk_start]) / sum(weights)
```

### 3.4 语言条件

语言指令通过 word embedding + FC 编码为 1 个 token，拼接到视觉 token 序列末尾。Transformer 的 self-attention 使语言 token 可以关注所有视觉帧，从而将颜色词与正确的方块位置关联。

---

## 4. Diffusion Policy 详解

### 4.1 核心洞察

标准 BC 用 MSE 损失：`L = ||a_pred - a_true||^2`

当同一观测对应多个合理动作时（多模态），网络被迫预测平均值：

```
真实动作分布:  a1 ~ 向左绕, a2 ~ 向右绕
MSE 最优解:    a_pred = (a1 + a2) / 2  ->  直撞障碍物（不合理）
```

Diffusion Policy 直接建模条件动作分布 `p(a | obs, lang)`，通过扩散过程采样得到合理动作。

### 4.2 扩散过程

**前向过程（训练时）**：逐步给动作序列加噪声

```python
# x_0: 原始动作 horizon (horizon * action_dim)
# epsilon: 标准高斯噪声
# alpha_bar_t = alphas_cumprod[t]  (累计乘积)
x_t = sqrt(alpha_bar_t) * x_0 + sqrt(1 - alpha_bar_t) * epsilon
```

**反向过程（推理时）**：逐步去噪，恢复动作序列

```python
x_T ~ N(0, I)  # 从纯噪声开始
for t in reversed(range(T)):
    epsilon_pred = noise_prediction_net(x_t, t, obs_feat, lang_feat)
    # 标准 DDPM 反向步骤
    mean = (1 / sqrt(alpha_t)) * (x_t - (1 - alpha_t) / sqrt(1 - alpha_bar_t) * epsilon_pred)
    if t > 0 and not deterministic:
        x_{t-1} = mean + sqrt(beta_t) * z     # 随机采样
    else:
        x_{t-1} = mean                          # 确定性评估
return x_0  # (horizon, action_dim)
```

**关键区分**：`alpha_t` 是单步系数（= 1 - beta_t），`alpha_bar_t` 是累计乘积。混淆两者会导致采样尺度失真。

### 4.3 网络架构

```
Input: 噪声动作 (B, horizon*action_dim) + 时间步 t + 图像 (B,3,128,128) + 语言 token (B,L)
    ├── Time Embedding (B, hidden_dim)
    ├── Vision Encoder (CNN + FC) -> obs_feat (B, obs_dim)
    ├── Language Encoder (Embedding + FC) -> lang_feat (B, lang_dim)
    └── Concat [noise, time, obs, lang] -> MLP -> noise_pred (B, horizon*action_dim)
```

### 4.4 训练目标

```python
def diffusion_loss(model, image, text_tokens, action_true):
    # action_true: (B, horizon, action_dim) -> flatten
    B = action_true.size(0)
    action_flat = action_true.reshape(B, model.flat_dim)

    t = torch.randint(0, n_steps, (B,))
    epsilon = torch.randn_like(action_flat)

    # 前向加噪
    a_noisy = sqrt_alphas_cumprod[t] * action_flat + sqrt_one_minus_alphas_cumprod[t] * epsilon

    # 预测噪声
    obs_feat, lang_feat = model.encode_obs(image, text_tokens)
    epsilon_pred = model(a_noisy, t, obs_feat, lang_feat)

    return MSE(epsilon_pred, epsilon)
```

### 4.5 确定性评估

评估时设置 `deterministic=True`：
- 固定 RNG 种子（`torch.manual_seed(eval_seed)`）
- 反向过程中 **不添加任何噪声**（`sigma_t = 0`）
- 因此相同输入必定产生相同输出

---

## 5. 架构对比表

| 维度 | Minimal Action-Chunking Policy | Diffusion Policy |
|------|------|------------------|
| **核心机制** | Transformer Encoder + 动作块头 | 条件扩散去噪 |
| **CVAE** | 无 | 无（扩散本身提供多模态） |
| **输出形式** | 确定性动作块（并行输出） | 从分布中采样 |
| **处理多模态** | 无（确定性输出） | 扩散过程天然支持多模态 |
| **推理速度** | 快（单次前向） | 慢（需多步去噪，通常 10-100 步）|
| **动作平滑性** | 好（块内并行，块间时间集成） | 取决于去噪步数和 schedule |
| **训练稳定性** | 高 | 中（需仔细调 noise schedule）|
| **超参数敏感度** | 中（chunk size, hist_len） | 高（diffusion steps, beta schedule） |
| **语言条件** | 有（语言 token 参与 self-attention） | 有（语言特征拼接进噪声预测网络） |
| **时间位置编码** | 有（learned embedding） | 不需要（扩散步数隐含时间信息） |
| **典型应用场景** | 需要高速推理的实时控制 | 需要多模态动作多样性的复杂操作 |

---

## 6. 在 PushCube 上的最小实现

以下代码基于统一 PushCube 双方块环境，实现两种方法的最小版本。两者都接收 **语言指令** 作为条件输入。

完整代码见：
- [`examples/unified_pushcube_act.py`](../examples/unified_pushcube_act.py)
- [`examples/unified_pushcube_diffusion.py`](../examples/unified_pushcube_diffusion.py)

### 6.1 Minimal Action-Chunking Policy 核心片段

```python
class MinimalActionChunkingPolicy(nn.Module):
    """K 帧视觉 token + 1 语言 token -> Transformer Encoder -> 动作块"""

    def __init__(self, action_dim=2, chunk_size=10, hist_len=3,
                 hidden_dim=64, vocab_size=13, embed_dim=16):
        super().__init__()
        # 视觉编码器：每帧 128x128 -> hidden_dim
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 8, 5, stride=2, padding=2),    # 64x64
            nn.ReLU(),
            nn.Conv2d(8, 16, 5, stride=2, padding=2),   # 32x32
            nn.ReLU(),
            nn.Conv2d(16, 16, 5, stride=2, padding=2),  # 16x16
            nn.ReLU(),
            nn.Conv2d(16, 8, 5, stride=2, padding=2),   # 8x8
            nn.ReLU(),
        )
        self.vision_fc = nn.Linear(8 * 8 * 8, hidden_dim)

        # 语言编码器
        self.word_embed = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lang_fc = nn.Linear(embed_dim, hidden_dim)

        # 时间位置编码（learned）：K 帧 + 1 语言 = K+1 位置
        self.temporal_pos = nn.Embedding(hist_len + 1, hidden_dim)

        # Transformer Encoder
        enc_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, nhead=4, batch_first=True
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=2)

        # 动作解码头：并行输出 T * action_dim
        self.action_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim * chunk_size),
            nn.Tanh(),
        )

    def forward(self, images, text_tokens):
        # images: (B, K, 3, 128, 128), text_tokens: (B, MAX_LEN)
        B, K, C, H, W = images.shape

        # 每帧独立编码 -> K 个 token
        imgs_flat = images.reshape(B * K, C, H, W)
        vis_feats = self.encode_frame(imgs_flat).reshape(B, K, -1)

        # 语言 -> 1 个 token
        w = self.word_embed(text_tokens).mean(dim=1)
        lang_feat = self.lang_fc(w).unsqueeze(1)  # (B, 1, hidden_dim)

        # 拼接 + 时间位置编码
        all_tokens = torch.cat([vis_feats, lang_feat], dim=1)  # (B, K+1, hidden_dim)
        positions = torch.arange(K + 1, device=all_tokens.device)
        all_tokens = all_tokens + self.temporal_pos(positions).unsqueeze(0)

        # Self-attention + 并行输出
        encoded = self.encoder(all_tokens)
        actions = self.action_head(encoded[:, -1])  # 取语言 token 位置
        return actions.reshape(-1, self.chunk_size, self.action_dim)
```

### 6.2 Diffusion Policy 核心片段

```python
class MinimalDiffusionPolicy(nn.Module):
    """语言条件 DDPM，预测整个 action horizon 的噪声"""

    def __init__(self, action_dim=2, horizon=10, obs_dim=32, lang_dim=16,
                 hidden_dim=64, n_steps=20, vocab_size=13, embed_dim=16):
        super().__init__()
        self.horizon = horizon
        self.action_dim = action_dim
        self.flat_dim = horizon * action_dim

        self.time_embed = nn.Embedding(n_steps, hidden_dim)

        # 视觉 + 语言编码器
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 8, 5, stride=2, padding=2),
            nn.ReLU(),
            nn.Conv2d(8, 16, 5, stride=2, padding=2),
            nn.ReLU(),
            nn.Conv2d(16, 16, 5, stride=2, padding=2),
            nn.ReLU(),
            nn.Conv2d(16, 8, 5, stride=2, padding=2),
            nn.ReLU(),
        )
        self.vision_fc = nn.Linear(8 * 8 * 8, obs_dim)
        self.word_embed = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lang_fc = nn.Linear(embed_dim, lang_dim)

        # 噪声预测网络：输入 = 噪声动作 + 时间 + 视觉 + 语言
        self.noise_pred = nn.Sequential(
            nn.Linear(self.flat_dim + hidden_dim + obs_dim + lang_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, self.flat_dim),
        )

        # DDPM noise schedule
        self.register_buffer("betas", torch.linspace(1e-4, 0.02, n_steps))
        alphas = 1.0 - self.betas                       # 单步 alpha_t
        self.register_buffer("alphas", alphas)
        self.register_buffer("alphas_cumprod", torch.cumprod(alphas, dim=0))
        self.register_buffer("sqrt_alphas_cumprod", torch.sqrt(self.alphas_cumprod))
        self.register_buffer("sqrt_one_minus_alphas_cumprod",
                             torch.sqrt(1.0 - self.alphas_cumprod))
        self.register_buffer("sqrt_betas", torch.sqrt(self.betas))

    def forward(self, noisy_action, t, obs_feat, lang_feat):
        t_emb = self.time_embed(t)
        inp = torch.cat([noisy_action, t_emb, obs_feat, lang_feat], dim=-1)
        return self.noise_pred(inp)

    @torch.no_grad()
    def sample(self, image, text_tokens, deterministic=False):
        obs_feat, lang_feat = self.encode_obs(image, text_tokens)
        B = image.size(0)
        x = torch.randn(B, self.flat_dim, device=image.device)

        for t in reversed(range(n_steps)):
            t_batch = torch.full((B,), t, device=image.device, dtype=torch.long)
            eps_pred = self.forward(x, t_batch, obs_feat, lang_feat)

            alpha_t = self.alphas[t]              # 单步
            alpha_bar_t = self.alphas_cumprod[t]  # 累计

            # 标准 DDPM 反向公式
            mean = (1.0 / torch.sqrt(alpha_t)) * (
                x - (1.0 - alpha_t) / torch.sqrt(1.0 - alpha_bar_t) * eps_pred
            )
            if t > 0 and not deterministic:
                x = mean + self.sqrt_betas[t] * torch.randn_like(x)
            else:
                x = mean  # 确定性评估：不加噪声

        return x.reshape(B, self.horizon, self.action_dim)
```

### 6.3 运行命令

```bash
cd examples

# Minimal Action-Chunking Policy
python unified_pushcube_act.py --n-episodes 100 --epochs 30 --chunk-size 10

# Diffusion Policy
python unified_pushcube_diffusion.py --n-episodes 200 --epochs 50 --diffusion-steps 20
```

---

## 7. 实验结果与选择建议

### 7.1 统一输入条件

为保证公平比较，所有五种方法接收等价的目标身份信息：

| 方法 | 输入模态 | 目标身份来源 |
|------|---------|------------|
| VLA | RGB + 语言 | 语言中的颜色词 |
| Action-Chunking | RGB 历史 + 语言 | 语言中的颜色词 |
| Diffusion Policy | RGB + 语言 | 语言中的颜色词 |
| RL | 14-D 状态 | goal-color one-hot |
| World Model | 14-D 状态 | goal-color one-hot |

语言和 one-hot 携带相同信息（哪个颜色的方块是目标），但通过不同模态传递。

### 7.2 语言消融设计

VLA 轨道的语言消融采用 **单模型、同 episode、多条件** 评估：

1. 训练一个 Full-VLA 模型（正确语言 + 专家动作）
2. 在相同的评估 episode 上，分别输入：
   - (a) 正确语言 → 记录正确方块成功率
   - (b) 交换语言 → 记录错误方块成功率（应上升）
   - (c) 零语言 → 记录选择准确率（应降至 ~50%）
3. 额外训练一个 Vision-Only 基线（零语言 token 训练），作为独立对照

这消除了"训练不同模型"带来的混杂因素。

### 7.3 选择建议

**选择 Action-Chunking Policy，如果：**
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

**Q: 为什么不叫 ACT？**

A: 完整 ACT（Zhao et al., 2023）包含 CVAE 隐变量、Transformer Decoder 和完整时间集成。本实现缺少 CVAE，因此诚实地改名为 Minimal Action-Chunking Policy，并在代码和文档中明确声明限制。

**Q: 动作块是自回归生成还是并行输出？**

A: 并行输出。当前代码和标准 ACT 都通过 `action_head` 一次性生成整个动作块，而非逐动作自回归解码。这更高效，也符合原始论文的设计。

**Q: 时间位置编码为什么重要？**

A: Transformer 的 self-attention 是排列不变的——如果不加位置编码，模型无法区分"最近的帧"和"最早的帧"。learned embedding 让每个位置获得独特的偏置，使模型能利用时序信息。

**Q: Diffusion Policy 的推理延迟太高怎么办？**

A: 三种优化方向：(1) 减少去噪步数（如从 100 减到 10，用 DDIM 加速）；(2) 使用更快的 backbone（MLP 替代 Transformer）；(3) 蒸馏为单步生成器。

**Q: DDPM 反向公式中 alpha_t 和 alpha_bar_t 有什么区别？**

A: `alpha_t = 1 - beta_t` 是单步系数，`alpha_bar_t = prod(alpha_1..alpha_t)` 是累计乘积。标准 DDPM 反向步骤中，均值修正项的系数是 `(1 - alpha_t) / sqrt(1 - alpha_bar_t)`，缩放系数是 `1 / sqrt(alpha_t)`。混淆两者会导致采样尺度失真。

**Q: 在 PushCube 上为什么成功率都不高？**

A: 不能只根据当前成功率判定单一原因。BC 会受到示教覆盖范围和闭环分布偏移影响，但学习策略可能通过去噪与跨示例泛化超过平均演示表现，也可能因视觉标定、动作解码或评估实现而失败。应先分别检查：(1) 演示覆盖与验证集；(2) 开环动作误差；(3) 闭环状态分布；(4) 相机和动作归一化；(5) 评估种子与成功判定。扩大数据、改进示教、DAgger 或 RL 微调是候选方案，不是未经消融即可确认的唯一解释。

---

## 参考文献

1. Zhao et al., "Learning Fine-Grained Bimanual Manipulation with Low-Cost Hardware", RSS 2023 (ACT)
2. Chi et al., "Diffusion Policy: Visuomotor Policy Learning via Action Diffusion", RSS 2023
3. Ho et al., "Denoising Diffusion Probabilistic Models", NeurIPS 2020 (DDPM)
4. Brohan et al., "RT-1: Robotics Transformer for Real-World Control at Scale", RSS 2023
5. Shafiullah et al., "Behavior Transformers: Cloning k modes with one stone", NeurIPS 2022
