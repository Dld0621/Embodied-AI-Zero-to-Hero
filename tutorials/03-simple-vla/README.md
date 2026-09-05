# Stage 3: 简单 VLA

> 搭建一个最小的 VLA 推理 pipeline，从预训练组件组装到真实模型推理。

> 运行边界：下面区分“架构教学”与“官方模型 API 示例”。外部权重、图像、CUDA/JAX 环境不随教程提供，本次审查未执行模型下载或端到端推理；任何输出都不能直接接真机。

---

## 学习目标

完成本阶段后，你应该能够：

1. 用预训练组件搭建最小 VLA 架构
2. 成功运行 OpenVLA 推理
3. 理解模型输出的含义和反归一化
4. 理解 VLA 推理的完整 pipeline

---

## 3.1 最小 VLA 架构

组合预训练组件，构建端到端 VLA：

```python
import torch
import torch.nn as nn
from transformers import CLIPModel, CLIPProcessor, AutoModel, AutoTokenizer

class SimpleVLA(nn.Module):
    """
    使用预训练 CLIP 双编码器 + 随机初始化 MLP 动作头的最小架构。
    仅用于教学，不用于实际控制。
    """
    def __init__(self, action_dim=7):
        super().__init__()
        # 加载预训练 CLIP
        self.clip = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        self.clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

        # 冻结 CLIP（可选）
        for param in self.clip.parameters():
            param.requires_grad = False

        clip_dim = 512  # CLIP base 的输出维度

        # 融合层
        self.fusion = nn.Sequential(
            nn.Linear(clip_dim * 2, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
        )

        # 策略头
        self.policy_head = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim),
            nn.Tanh(),
        )

    def forward(self, images, texts):
        """
        Args:
            images: list of PIL Images
            texts: list of strings
        Returns:
            actions: [B, action_dim]
        """
        # CLIP 编码
        inputs = self.clip_processor(text=texts, images=images, return_tensors="pt", padding=True)
        outputs = self.clip(**inputs)

        image_features = outputs.image_embeds  # [B, 512]
        text_features = outputs.text_embeds     # [B, 512]

        # 融合
        fused = torch.cat([image_features, text_features], dim=-1)
        fused = self.fusion(fused)

        # 动作
        actions = self.policy_head(fused)
        return actions
```

**关键理解**：
- 视觉编码器和文本编码器都是**预训练**的，提供强大的表征
- 只有融合层和策略头是**随机初始化**的，需要训练
- 这是冻结编码器、学习动作映射的一种迁移学习方案；其他 VLA 也可能联合微调视觉或语言组件。

---

## 3.2 使用 OpenVLA 推理

### 环境准备

先按 [OpenVLA 官方安装说明](https://github.com/openvla/openvla#installation) 配置独立环境，记录依赖与模型 revision。不要把任意最新版 Transformers 当成兼容保证；官方注明过版本兼容限制。以下用支持 bfloat16 的 CUDA 设备，`trust_remote_code=True` 会执行模型仓库代码，应先审查来源并固定 revision。

### 单图、单动作 API 示例

```python
import torch
from PIL import Image
from transformers import AutoModelForVision2Seq, AutoProcessor

# 加载模型
model = AutoModelForVision2Seq.from_pretrained(
    "openvla/openvla-7b",
    torch_dtype=torch.bfloat16,
    trust_remote_code=True,
)
processor = AutoProcessor.from_pretrained(
    "openvla/openvla-7b",
    trust_remote_code=True,
)

model = model.to("cuda")
model.eval()

# 加载图像
image = Image.open("scene.jpg").convert("RGB")

# 构建 prompt
# OpenVLA 使用特定格式："In: ...\nOut:"
prompt = "In: What action should the robot take to pick up the red cup?\nOut:"

# 预处理
inputs = processor(prompt, image).to("cuda", dtype=torch.bfloat16)

# 推理
with torch.no_grad():
    action = model.predict_action(**inputs, unnorm_key="bridge_orig", do_sample=False)

print(f"Predicted action: {action}")
```

### OpenVLA 输出解析

此基础检查点的 Bridge 动作接口返回一个 7 维动作向量，而不是 7 个时间步：

```
action = [dx, dy, dz, droll, dpitch, dyaw, gripper]
```

| 维度 | 含义 | 单位 |
|------|------|------|
| 0-2 | 末端位置增量 (x, y, z) | 按所选训练数据的动作合同解释 |
| 3-5 | 末端旋转增量 (roll, pitch, yaw) | 核对参考系、单位和组合约定 |
| 6 | 夹爪开合 | 核对取值范围、开闭方向和控制器语义 |

向量长度相同不代表可跨机器人直接执行。还需匹配控制周期、坐标系、归一化和夹爪约定，并先验证仿真适配器。

### 反归一化（Unnormalization）

基础 OpenVLA 的 `predict_action` **已经进行一次反归一化**：读取检查点内所选数据集的 `q01` / `q99` 与 `mask`，不是再乘自造的标准差、加均值。[官方实现](https://huggingface.co/openvla/openvla-7b/blob/main/modeling_prismatic.py)

```python
print("可用数据集统计键：", list(model.norm_stats))
stats = model.get_action_stats("bridge_orig")
print("本接口的分位数统计：", stats["q01"], stats["q99"])
# action 已完成 API 内置的反归一化，不要在这里再缩放一次。
```

`unnorm_key` 必须存在于该检查点，而且与训练/目标接口相符；不能把任意数据集名字填进去。自定义微调还需保存并正确装载对应统计量。

---

## 3.3 VLA 推理 Pipeline 完整流程

```
1. 图像采集
   └── 从相机获取 RGB 图像 (H, W, 3)

2. 语言指令
   └── 用户输入: "pick up the red cup"

3. 预处理
   └── 图像: resize → normalize → tensor
   └── 文本: tokenize → input_ids

4. 模型推理
   └── VLA model.forward(image, text) → action

5. 后处理
   └── 核对输出是否已反归一化，避免重复处理
   └── 校验接口、单位、限位与异常值；裁剪不等于安全保证
   └── 可选: 平滑滤波

6. 执行
   └── 先接已验证的仿真适配器
   └── 真机需要单独的安全审核、急停与现场授权

7. 循环
   └── 获取新图像 → 重复步骤 1-6
```

---

## 3.4 使用 Octo（轻量替代）

Octo 使用不同的 JAX 模型和输入合同，不能只替换模型名字。以下按[官方推理 notebook](https://github.com/octo-models/octo/blob/main/examples/01_inference_pretrained.ipynb) 展示小模型接口；是否适合你的内存与运行平台需要实际测量。

```python
import jax
from octo.model.octo_model import OctoModel

# 加载模型
model = OctoModel.load_pretrained("hf://rail-berkeley/octo-small-1.5")

# 准备输入
from PIL import Image
import numpy as np

image = np.array(Image.open("scene.jpg").convert("RGB").resize((256, 256)))
observation = {
    "image_primary": image[None, None],  # [batch, history, H, W, C]
    "timestep_pad_mask": np.array([[True]]),
}
task = model.create_tasks(texts=["pick up the red cup"])

# 推理
actions = model.sample_actions(
    observation,
    task,
    unnormalization_statistics=model.dataset_statistics["bridge_dataset"]["action"],
    rng=jax.random.PRNGKey(0),
)
print(actions.shape)  # [batch, action_chunk, action_dim]，不是带 "actions" 键的字典
```

Octo 的优势：
- 27M-93M 参数，可在单卡甚至 CPU 运行
- 支持 Goal Image Conditioning
- 更灵活的输入格式

---

## 3.5 推理优化技巧

各小节是独立示例，不要顺序复用上一节的 Octo `model` 变量；OpenVLA 小节需重新建立对应的模型、processor 与匹配 dtype。优化效果仍需实测。

### 1. 量化（Quantization）

减少显存占用：

```python
from transformers import BitsAndBytesConfig

bnb_config = BitsAndBytesConfig(load_in_8bit=True)
model = AutoModelForVision2Seq.from_pretrained(
    "openvla/openvla-7b",
    quantization_config=bnb_config,
    trust_remote_code=True,
)
```

### 2. 批量推理

基础 OpenVLA 的 `predict_action` 实现取第一个生成序列，不能假设传入一批图像就返回一批动作。多个样本先逐个调用；这不是吞吐优化：

```python
actions = []
for i in range(4):
    image = Image.open(f"scene_{i}.jpg").convert("RGB")
    inputs = processor(prompt, image).to("cuda", dtype=torch.bfloat16)
    with torch.no_grad():
        actions.append(model.predict_action(**inputs, unnorm_key="bridge_orig", do_sample=False))
```

### 3. 推理频率控制

基础 OpenVLA 的单动作接口不能通过索引变成 chunking。要预测未来多步，需另一个支持且经过相应训练的动作头/检查点；更长开环执行也会降低纠错频率。

```python
import numpy as np
action = np.asarray(actions[0])  # 前面逐样本推理的第一个完整动作
assert action.shape == (7,)
# action[0] 是 dx 这个分量，不是第 0 个未来动作。此处不执行机器人控制。
```

---

## 验证检查点

- [ ] 能成功加载并运行 OpenVLA 推理
- [ ] 理解 OpenVLA 输出的 7 个维度的含义
- [ ] 理解 `unnorm_key` 的作用
- [ ] 能解释从图像到动作的完整 pipeline
- [ ] （可选）能在 Octo 上运行推理

---

## 常见问题

**Q: 运行 OpenVLA 时出现 OOM？**
A: 尝试：1) 使用 bfloat16；2) 8-bit 量化；3) 换用 Octo；4) 使用更小 batch_size。

**Q: 模型输出的动作看起来随机？**
A: 检查图像预处理、prompt、`unnorm_key` 和动作合同。随机或分布外图像没有有效任务语义，但不意味着模型输出必然是随机数。

**Q: 推理速度太慢？**
A: 先测量预处理、推理、传输的端到端耗时；再评估量化或更小模型。Chunking 需要对应训练与接口，不能给基础单动作输出直接加时间索引。

---

## 延伸阅读

- [OpenVLA 官方文档](https://openvla.github.io/)
- [Octo 官方文档](https://octo-models.github.io/)
- [Transformers 多模态教程](https://huggingface.co/docs/transformers/model_doc/vision-encoder-decoder)
