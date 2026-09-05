# Transformer 基础

> **逐点图解 / Concept close-ups：**[Transformer 与多模态表示](../knowledge-atlas/learning-transformers-multimodal/index.md)。每个小点配原理、算例、图、自测；这是中文细解，保留英文术语。

> English contract: [Foundations overview](README_EN.md#route) · Primary references: [Transformers](../SOURCES.md#04-transformers)

> **前置要求**: [`03-deep-learning-basics.md`](03-deep-learning-basics.md)（MLP、反向传播、训练循环）、[`02-linear-algebra.md`](02-linear-algebra.md)（矩阵乘法、向量内积）
> **预计学习时间**: 2–3 小时
> **完成后你能**: 解释 self-attention 的 Q/K/V 在做什么；看懂 ViT 如何把一张图变成"一串词"；理解 VLA 模型的骨干为什么是 Transformer

---

## 1. 序列建模：为什么需要 attention？

MLP 一次只吃一个固定长度的向量。但很多任务是**序列**：一句话有先后顺序、一段动作有时间先后。处理序列的经典思路是 RNN（循环神经网络）。

### RNN 的局限

RNN 像接力跑：从左到右逐个读 token，每读一个就把信息塞进一个"隐藏状态"再传给下一步。

```
x1 → h1 → x2 → h2 → x3 → h3 → ...   (串行，必须等上一步算完)
```

三个致命问题：
1. **远距离遗忘**：信息要一路传到末尾，长句子开头的信息到结尾早被冲淡了。
2. **无法并行**：第 3 步必须等第 2 步，GPU 的大量并行算力用不上，训练慢。
3. **难对齐**：句子里"红方块"的"红"要和远处的"方块"绑定，RNN 只能靠隐藏状态"碰运气"记住这种关系。

**Attention 的核心想法**：别接力跑了，开个会——让序列里每个位置都能**直接看到**所有其他位置，自己决定该关注谁。这样距离不再是问题，而且整段序列可以一次性并行算完。

---

## 2. Self-Attention 机制：Q、K、V

Self-attention 把每个位置变换成三个角色：

| 角色 | 含义 | 直觉（图书馆类比） |
|------|------|-------------------|
| **Q**（Query 查询） | "我想找什么" | 你提交的检索词 |
| **K**（Key 键） | "我能被怎样匹配" | 每本书的标签/索引 |
| **V**（Value 值） | "我真正的内容" | 书的正文内容 |

每个位置用自己的 Q 去和所有位置的 K 做点积，得到"相似度分数"；分数越高说明越该关注那个位置。再用 softmax 把分数变成权重（和为 1），最后用这些权重对所有 V 加权求和——这就是该位置的新表示。

<div class="dof-principle" role="group" aria-label="Transformer self-attention 中 query key value 的关系">
  <p class="dof-principle__caption"><strong>原理图 · Attention is content-dependent routing</strong>：每个 token 用 Query 问“我该看谁”，与全部 Key 比较后得到权重，再对 Value 加权求和。因此同一个词在不同上下文会读取不同信息。</p>
  <div class="dof-principle__canvas">
    <svg viewBox="0 0 860 276" role="img" aria-labelledby="attention-title">
      <title id="attention-title">Query Key Value 计算注意力权重并聚合信息</title><rect class="dof-diagram-surface" x="12" y="17" width="220" height="240" rx="18"/><text class="dof-diagram-title" x="36" y="49">Tokens</text><rect class="dof-diagram-fill-blue" x="39" y="70" width="154" height="34" rx="9"/><text class="dof-diagram-label" x="63" y="93">robot</text><rect class="dof-diagram-fill-violet" x="39" y="115" width="154" height="34" rx="9"/><text class="dof-diagram-label" x="63" y="138">pick</text><rect class="dof-diagram-fill-good" x="39" y="160" width="154" height="34" rx="9"/><text class="dof-diagram-label" x="63" y="183">cube</text><text class="dof-diagram-note" x="39" y="225">each token → Q, K, V</text><path class="dof-diagram-accent" d="M244 128 H322"/><path class="dof-diagram-arrow" d="M322 128 l-11 -6 v12z"/>
      <rect class="dof-diagram-surface" x="340" y="17" width="248" height="240" rx="18"/><text class="dof-diagram-title" x="365" y="49">Match query to keys</text><text class="dof-diagram-math" x="375" y="89">scores = QKᵀ / √d</text><g transform="translate(378 113)"><rect class="dof-diagram-fill-blue" x="0" y="0" width="40" height="40" rx="6"/><rect class="dof-diagram-fill-violet" x="46" y="0" width="40" height="40" rx="6"/><rect class="dof-diagram-fill-good" x="92" y="0" width="40" height="40" rx="6"/><text class="dof-diagram-math" x="9" y="26">0.1</text><text class="dof-diagram-math" x="55" y="26">0.7</text><text class="dof-diagram-math" x="101" y="26">0.2</text></g><text class="dof-diagram-note" x="378" y="185">softmax → weights sum to 1</text><path class="dof-diagram-accent" d="M602 128 H667"/><path class="dof-diagram-arrow" d="M667 128 l-11 -6 v12z"/>
      <rect class="dof-diagram-surface" x="685" y="17" width="163" height="240" rx="18"/><text class="dof-diagram-title" x="708" y="49">Read values</text><text class="dof-diagram-math" x="706" y="89">Σ αᵢ Vᵢ</text><path class="dof-diagram-violet" d="M717 128 H816 M717 148 H786 M717 168 H748"/><text class="dof-diagram-label" x="708" y="211">context-aware</text><text class="dof-diagram-label" x="708" y="231">representation</text>
    </svg>
  </div>
</div>

### 缩放点积注意力

```
Attention(Q, K, V) = softmax( Q · Kᵀ / √d_k ) · V
```

逐步拆解（设序列长度 `n`，特征维度 `d`）：

1. `Q·Kᵀ` → `(n, n)` 的分数矩阵，第 `i` 行第 `j` 列 = "位置 i 对位置 j 的关注程度"。
2. `/ √d_k` 缩放：在 Q、K 各分量近似独立、零均值、单位方差的分析假设下，点积方差随 `d_k` 增长，除以 `√d_k` 后方差为 1。它缓解大分数差把 softmax 推入饱和区的问题，不保证训练中实际分数方差始终恰为 1。见 [原始 Transformer 论文 §3.2.1](https://arxiv.org/abs/1706.03762)。
3. `softmax(·, dim=-1)`：每行归一化成权重。
4. `· V` → `(n, d)`：用权重把所有 V 混合，得到每个位置的新向量。

**机械类比**：像做加权平均的力合成——每个位置对其他位置施加"注意力权重"，合力是所有 V 按权重叠加的结果。

> 关键性质：self-attention 本身**不关心顺序**（打乱输入序列、输出也跟着打乱）。所以必须额外注入位置信息（见第 4 节）。

---

## 3. 多头注意力：为什么要多个头？

单个注意力头在一个投影子空间内计算注意力，表示容量有限；多个头允许模型并行使用不同的投影子空间。不同头可能学习语法、语义、指代或其他关系，但这不是人为保证的一一对应。各头输出随后被拼接并再次投影：

```
head_i = Attention(Q·W_i^Q, K·W_i^K, V·W_i^V)
MultiHead = Concat(head_1, ..., head_h) · W^O
```

每个头用自己的一组投影矩阵 `W_i`，把输入映到子空间各自做注意力，最后 concat 后再线性投影回原维度。

**直觉**：像一组工程师从不同角度审同一张图纸——一个看结构、一个看公差、一个看装配顺序，各出一份意见再综合。本项目里 SmolVLA 的 Transformer 主干就是多头注意力堆叠而成。

---

## 4. 位置编码：给"无序"的 attention 加上顺序

因为 attention 对顺序无感，我们要手动把"这是第几个 token"的信息加进去。两种主流做法：

| 方法 | 做法 | 特点 |
|------|------|------|
| **正弦位置编码** | 用不同频率的 sin/cos 固定生成，直接加到 embedding 上 | 无需学习、可外推到更长序列，原始 Transformer 用它 |
| **可学习位置编码** | 给每个位置一个可训练向量 | 灵活、效果常略好，但不能超出训练见过的长度 |

注意是**加**（不是 concat）：把位置向量逐元素加到 token embedding 上，让"内容"和"位置"混在同一个向量里被 attention 处理。

---

## 5. Transformer 编码器/解码器

把上面的零件组装起来。一个 **Transformer Encoder 层** = 多头注意力 + 前馈网络（MLP），各带残差连接和 LayerNorm：

```
x ─► [Multi-Head Attention] ─► +x ─► LayerNorm   (子层1: 自己看自己)
   ─► [Feed-Forward MLP]     ─► +  ─► LayerNorm   (子层2: 逐位置非线性)
```

- **残差连接**（`+x`）：把输入直接加到输出上，缓解深层梯度消失，让训练深层成为可能。
- **LayerNorm**：对每个样本的特征维度归一化，稳定训练（对比 BatchNorm 是对 batch 维归一化）。
- **前馈网络**：对每个位置独立做一次 `Linear→GELU→Linear`，给注意力混合后的特征再加非线性。

**Decoder** 多一个"带掩码的注意力"：掩码让每个位置只能看到它**之前**的位置（不能偷看未来），这是自回归生成所必需的。Encoder 用于理解（双向看全部），Decoder 用于生成（单向看过去）。

---

## 6. Vision Transformer (ViT)：把图像当成一串词

Transformer 原本为文本设计。ViT 的妙招是：**把图像切成小块，每一块当成一个 token**。

```
224×224 图像 ──切 16×16 小块──► 14×14 = 196 个 patch
每个 patch 16×16×3=768 ──线性投影──► 维度 d 的向量  (patch embedding)
额外加一个 [CLS] token  +  位置编码
──► 送入 Transformer Encoder ──► 每个位置一个特征向量
```

为什么这样行得通？因为"patch 序列"和"词序列"在结构上一样——都是一串向量，attention 让每个 patch 都能看到其他 patch，自然学到了物体部件间的关系。

> 这正是 VLA 视觉编码器的做法。OpenVLA 用 **DINOv2 + SigLIP** 双 ViT 编码图像（一个擅长空间结构、一个擅长语言对齐），SmolVLA 用 **SmolVLM2 的 ViT** 编码 3 路摄像头图像。详见 [`docs/01-what-is-vla.md`](../01-what-is-vla.md)。

---

## 7. 语言模型：tokenization、embedding、自回归生成

VLA 的另一半是理解语言指令。一个语言模型的三步：

1. **Tokenization（分词）**：把文本切成整数 ID。本项目 `examples/unified_pushcube_vla.py` 里就有个迷你版：

   ```python
   VOCAB = {"<pad>":0, "push":1, "the":2, "red":3, "green":4, "cube":5, "<unk>":6}
   MAX_LEN = 8
   def tokenize(text):
       words = text.lower().replace(".", "").split()
       toks = [VOCAB.get(w, VOCAB["<unk>"]) for w in words]
       return toks[:MAX_LEN] + [0] * (MAX_LEN - len(toks))   # 补齐长度
   ```
   真实模型用 BPE/WordPiece（词表几万），但原理一样：文本 → 整数序列。

2. **Embedding（嵌入）**：`nn.Embedding(num_tokens, d)` 把每个整数 ID 查表成一个 `d` 维稠密向量。项目里 `self.word_embed = nn.Embedding(vocab_size, embed_dim)` 正是这一步。

3. **自回归生成**：给定前 `t` 个 token，预测第 `t+1` 个；再把预测接回去继续预测下一个，像接龙一样逐个生成。

> 对比点：项目里的 Tiny-VLA 把词嵌入**平均**（`word_embed(tokens).mean(dim=1)`）当语言特征。它对词序不敏感，但**不会必然丢掉词的身份**：若其他词相同，`red=[1,0]` 与 `green=[0,1]` 仍会产生不同的平均向量。它难以区分的是词集合相同、关系不同的指令，例如“red pushes green”和“green pushes red”。词向量平均本身可用于分类，见 [Deep Averaging Networks 原论文](https://aclanthology.org/P15-1162/)。仓库中约 45% 的选择准确率只能描述该次实验，不能证明“没有 attention 就分不清红绿”；需要词序、视觉特征、数据覆盖与训练条件的受控消融。

---

## 8. 连接到项目：VLA = ViT + 语言模型 + 动作头

把第 6、7 节拼起来，就是现代 VLA 的标准架构：

```
图像 ──► ViT (patch embedding + Transformer) ──┐
                                                ├─► Transformer 融合 ──► 动作头 ──► 机器人动作
语言指令 ──► Tokenize + Embedding + Transformer ┘
```

| 模型 | 视觉编码器 | 语言主干 | 动作生成 |
|------|-----------|---------|---------|
| **OpenVLA** | DINOv2 + SigLIP（双 ViT） | Llama 2（7B） | 自回归离散 token |
| **SmolVLA** | SmolVLM2 ViT | SmolLM 主干 | Flow Matching 连续动作 |
| **本项目 Tiny-VLA** | 4 层 CNN | 词嵌入平均（无 attention） | MLP 回归 |

- **OpenVLA**（`examples/robot_foundation_models/openvla/`）：原始论文采用 7B Llama 2 骨干与 DINOv2、SigLIP 视觉特征，并把动作离散成 Token。见 [`VLA 从零到一`](../specializations/vla-zero-to-one-cn.md)的算法族与证据边界。
- **SmolVLA**（`examples/robot_foundation_models/smolvla/`）：原始论文报告约 450M 参数，并使用 Flow Matching 生成连续动作块；具体动作维度、频率和归一化由目标数据集与机器人合同决定，不能写成模型固有常数。见 [`VLA 从零到一`](../specializations/vla-zero-to-one-cn.md)的 Flow Matching 章节。
- **Action-Chunking 策略**（`examples/unified_pushcube_act.py`）：用 K 帧 Transformer 一次性预测未来 T 步动作——这就是 encoder 堆叠 + 多头注意力的直接应用。

> 现在你能回答："为什么 VLA 不用 MLP？" 因为它要同时处理图像（成百上千 patch）和语言（变长 token 序列），这种"变长、需对齐、需远距离建模"的活，正是 Transformer 的主场；MLP 只能处理定长向量。

---

## 9. 动手代码：实现一个最小的 Self-Attention

下面代码可直接运行（`pip install torch`）。它实现单头和多头 self-attention，喂一段 4 词的"指令"，打印注意力权重矩阵，让你直观看到"谁在关注谁"。

```python
# transformer_basics.py — 可直接运行
import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(0)

# 1. 单头 self-attention
class SelfAttention(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        self.q = nn.Linear(embed_dim, embed_dim)
        self.k = nn.Linear(embed_dim, embed_dim)
        self.v = nn.Linear(embed_dim, embed_dim)
        self.scale = embed_dim ** 0.5

    def forward(self, x):
        # x: (batch, seq_len, embed_dim)
        Q, K, V = self.q(x), self.k(x), self.v(x)
        scores = Q @ K.transpose(-2, -1) / self.scale   # (batch, n, n)
        attn   = F.softmax(scores, dim=-1)              # 每行归一化
        out    = attn @ V                               # (batch, n, embed_dim)
        return out, attn

# 2. 多头 attention（用 nn.MultiheadAttention 封装）
class MultiHeadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)

    def forward(self, x):
        out, attn = self.attn(x, x, x, need_weights=True)  # 自注意力: Q=K=V=x
        return out, attn

# 3. 演示：4 个 token 的小序列
embed_dim, seq_len = 8, 4
x = torch.randn(1, seq_len, embed_dim)          # 假装是 4 个词的嵌入

sa = SelfAttention(embed_dim)
_, attn_single = sa(x)
print("单头注意力权重 (每行=某token对各token的关注度，行和=1):")
print(attn_single[0].round(decimals=3))

mha = MultiHeadAttention(embed_dim, num_heads=2)
_, attn_multi = mha(x)
print("\n多头注意力权重 (2 头取平均):")
print(attn_multi[0].round(decimals=3))

# 检查未舍入张量；浮点计算使用容差，不比较精确相等
for weights in (attn_single, attn_multi):
    assert weights.shape == (1, seq_len, seq_len)
    assert torch.allclose(weights.sum(dim=-1), torch.ones(1, seq_len), atol=1e-6)
print("单头/多头权重形状正确，未舍入的每行权重和均接近 1")
```

应观察到两个 4×4 权重矩阵，以及最后的检查通过提示。随机初始化的具体数值不作为验收答案：固定种子可帮助在同一环境复现，但不能保证跨 PyTorch 版本和设备逐位一致，见 [PyTorch 可复现性说明](https://docs.pytorch.org/docs/stable/notes/randomness.html)。

```
单头/多头权重形状正确，未舍入的每行权重和均接近 1
```

数学上，softmax 沿指定维度归一化后的和为 1；浮点计算只能在容差内成立，打印到三位小数后甚至可能看到 0.999 或 1.001。这里未启用 attention dropout；若训练时在权重上应用 dropout，单次采样的行和也不必为 1。把 attention 与 `LayerNorm + 残差 + 前馈 MLP` 组合可构建 Transformer 层，但 OpenVLA 的语言骨干是带因果掩码的 decoder，不能把这里无掩码的 encoder 演示当作其完整实现。见 [PyTorch Softmax 定义](https://docs.pytorch.org/docs/stable/generated/torch.nn.modules.activation.Softmax.html)。

> 对照阅读：比较 Q/K/V 投影与 `word_embed(...).mean(dim=1)`。后者把整句话压成一个词序不敏感的向量；前者可进行依赖上下文的信息交互，但仍需位置编码才能利用词序。架构提供能力，不等于训练后一定获得了视觉—语言绑定；最终要用闭环实验和受控消融检验。

---

## 10. 检查理解

试着回答下面问题（答案可在文中找到，建议先合上文档自己想）：

1. **RNN 题目**：RNN 处理序列有三个致命问题，是哪三个？attention 分别解决了其中哪几个？
2. **Q/K/V 题目**：用自己的话解释 Q、K、V 各自的角色。为什么点积 `Q·Kᵀ` 能衡量"该关注谁"？
3. **缩放题**：公式里为什么要除以 `√d_k`？不除会发生什么？（提示：softmax 在大数值下的行为）
4. **位置题**：self-attention 本身是"无序"的，这句话什么意思？位置编码是加还是拼接到 embedding 上？正弦编码和可学习编码各自的优缺点？
5. **多头题**：用一个工程师团队的类比解释多头注意力。如果只有 1 个头，可能漏掉什么？
6. **ViT 题目**：ViT 如何把一张 224×224 的图变成"一串词"？这一串有多少个 token（patch 大小 16）？为什么 attention 适合处理它？
7. **项目题**：OpenVLA 和 SmolVLA 各用哪个 ViT 编码图像、哪个语言主干？构造一个“平均词嵌入能区分红绿”的反例，再构造一对词集合相同但关系相反的指令。为什么约 45% 的选择准确率不足以定位失败原因？
8. **动手题**：运行第 9 节代码，观察注意力权重矩阵。把 `seq_len` 从 4 改成 16，权重矩阵会变成几乘几？再把 `num_heads` 从 2 改成 8，观察权重分布是否变化。

> 完成本节后，你具备进入 [`VLA 从零到一`](../specializations/vla-zero-to-one-cn.md)的模型基础；仍需补齐机器人动作、时序、控制与评估合同。ViT、语言模型和 Transformer 是重要组件，但不是完整 VLA 系统的全部前置知识。
