# Stage 1: VLM 基础

> 理解视觉-语言模型如何将图像和文本映射到同一嵌入空间。

> 运行边界：模型推理片段需要自行准备图像、模型权重与兼容依赖；本次内容审查未下载权重或执行 GPU 推理。能得到相似度不等于已经具备机器人控制能力。

---

## 学习目标

完成本阶段后，你应该能够：

1. 解释 CLIP 的对比学习原理
2. 使用 HuggingFace Transformers 加载预训练 VLM
3. 计算图像-文本相似度
4. 理解 ViT 如何将图像 token 化

---

## 1.1 CLIP 原理

CLIP（Contrastive Language-Image Pre-training）是 VLA 的视觉-语言对齐基石。

### 核心思想

在大量（图像，文本）对上训练两个编码器，使得：
- 匹配的图像-文本对在嵌入空间距离近
- 不匹配的距离远

### 对比损失

```python
import torch
import torch.nn.functional as F

def contrastive_loss(image_features, text_features, temperature=0.07):
    """
    image_features: [N, D]
    text_features: [N, D]
    """
    if image_features.shape != text_features.shape or image_features.ndim != 2:
        raise ValueError("需要同形状 [N, D] 的一一配对图文特征")
    if temperature <= 0 or image_features.shape[0] == 0:
        raise ValueError("temperature 和批大小必须为正")
    N = image_features.shape[0]
    # 归一化
    image_features = F.normalize(image_features, dim=-1)
    text_features = F.normalize(text_features, dim=-1)

    # 计算相似度矩阵
    logits = image_features @ text_features.T / temperature  # [N, N]

    # 对角线是正样本
    labels = torch.arange(N, device=logits.device)

    # 双向交叉熵
    loss_i2t = F.cross_entropy(logits, labels)
    loss_t2i = F.cross_entropy(logits.T, labels)

    return (loss_i2t + loss_t2i) / 2
```

### 关键洞察

- **对比学习** 不需要人工标注的类别标签，只需要图文配对
- **Zero-shot**：训练后可以直接做图像分类（用类别名称作为文本查询）
- OpenVLA 的视觉组件包含 SigLIP 与 DINOv2；RT-2 基于 PaLI-X / PaLM-E，而不是直接采用 CLIP。模型谱系与“共享图文对齐思想”不能混为一谈。[OpenVLA](https://github.com/openvla/openvla)、[RT-2](https://robotics-transformer2.github.io/)

---

## 1.2 运行 CLIP 推理

```python
from transformers import CLIPProcessor, CLIPModel
from PIL import Image

# 加载模型
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32", attn_implementation="eager")
model.eval()
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

# 准备输入
image = Image.open("robot_scene.jpg")
texts = [
    "a robot arm reaching for a cup",
    "a person sitting at a desk",
    "a kitchen with appliances",
]

# 预处理
inputs = processor(text=texts, images=image, return_tensors="pt", padding=True)

# 推理
outputs = model(**inputs)
logits = outputs.logits_per_image  # [1, 3] 图像与每个文本的相似度
probs = logits.softmax(dim=1)

# 结果
for text, prob in zip(texts, probs[0]):
    print(f"{text}: {prob:.3f}")
```

**练习**：换不同的图像和文本，观察相似度变化。思考为什么 CLIP 对空间关系（"left of", "above"）理解较弱？

---

## 1.3 ViT 视觉 Token 化

Vision Transformer（ViT）将图像切分为 patch，每个 patch 变成一个 token：

下面以 **patch 16** 为例；上面加载的 CLIP **patch 32** 在 224×224 输入下有 7×7=49 个 patch，加 CLS 后共 50 个 token，不能把两者的网格混用。[模型配置](https://huggingface.co/openai/clip-vit-base-patch32/blob/main/config.json)

```
输入图像: 224x224x3
    ↓
切分为 16x16 的 patch → 14x14 = 196 个 patch
    ↓
每个 patch 线性映射到 D 维 → 196 个 visual token
    ↓
加上 [CLS] token → 197 个 token
    ↓
输入 Transformer Encoder
```

### 可视化 Attention

```python
import matplotlib.pyplot as plt

# 获取最后一层的 attention
outputs = model.vision_model(inputs.pixel_values, output_attentions=True)
attentions = outputs.attentions  # tuple of [B, heads, N, N]

# 可视化 [CLS] token 对所有 patch 的注意力
patch_size = model.config.vision_config.patch_size
height, width = inputs.pixel_values.shape[-2:]
rows, cols = height // patch_size, width // patch_size
patch_attention = attentions[-1][0, 0, 0, 1:]
assert patch_attention.numel() == rows * cols
attn = patch_attention.reshape(rows, cols).detach().cpu().numpy()
plt.imshow(attn, cmap='viridis')
plt.title("CLS attention over image patches")
plt.show()
```

**练习**：对比 CLIP ViT 和 DINOv2 的 attention map，观察两者关注区域的不同。

---

## 1.4 VLM 推理示例

使用 LLaVA（或类似模型）进行视觉问答：

```bash
pip install transformers accelerate
```

```python
from transformers import AutoProcessor, AutoModelForPreTraining
from PIL import Image

# 加载 LLaVA（示例，具体模型名可能变化）
processor = AutoProcessor.from_pretrained("llava-hf/llava-1.5-7b-hf")
model = AutoModelForPreTraining.from_pretrained("llava-hf/llava-1.5-7b-hf")

image = Image.open("scene.jpg")
prompt = "USER: <image>\nWhat objects can the robot interact with?\nASSISTANT:"

inputs = processor(text=prompt, images=image, return_tensors="pt")
output = model.generate(**inputs, max_new_tokens=100)
response = processor.decode(output[0], skip_special_tokens=True)
print(response)
```

---

## 验证检查点

完成以下任务即通过 Stage 1：

- [ ] 能解释 CLIP contrastive loss 的公式
- [ ] 能运行 CLIP 推理并解释相似度结果
- [ ] 能计算给定图像-文本对的相似度分数
- [ ] 能根据输入分辨率、patch 大小和 CLS 配置计算 token 数，而不是固定记成 197

---

## 延伸阅读

- [CLIP 论文](https://arxiv.org/abs/2103.00020)
- [LLaVA 论文](https://arxiv.org/abs/2304.08485)
- [SigLIP 论文](https://arxiv.org/abs/2303.15343)
