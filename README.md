<p align="center">
  <img src="https://raw.githubusercontent.com/Dld0621/VLA-Zero-to-Hero/master/assets/banner.jpg" alt="VLA Zero to Hero Banner" width="100%">
</p>

<h1 align="center">VLA Zero to Hero · 从零入门视觉-语言-动作模型</h1>

<p align="center">
  <b>从 VLM 基础到 LIBERO 仿真微调 + 强化学习 + 世界模型，系统掌握 VLA 全栈能力。全部代码可运行。</b>
</p>

<p align="center">
  <a href="https://github.com/Dld0621/VLA-Zero-to-Hero"><img src="https://img.shields.io/github/stars/Dld0621/VLA-Zero-to-Hero?style=flat-square" alt="Stars"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/License-MIT-blue?style=flat-square" alt="License"></a>
  <a href="https://python.org"><img src="https://img.shields.io/badge/Python-3.10+-3776AB?style=flat-square&logo=python" alt="Python"></a>
  <a href="https://pytorch.org"><img src="https://img.shields.io/badge/PyTorch-2.0+-EE4C2C?style=flat-square&logo=pytorch" alt="PyTorch"></a>
  <a href="https://github.com/openvla/openvla"><img src="https://img.shields.io/badge/OpenVLA-7B-blueviolet?style=flat-square" alt="OpenVLA"></a>
  <a href="https://github.com/Lifelong-Robot-Learning/LIBERO"><img src="https://img.shields.io/badge/LIBERO-Simulation-ff9900?style=flat-square" alt="LIBERO"></a>
  <a href="examples/quick_start.ipynb"><img src="https://img.shields.io/badge/Google_Colab-Compatible-F9AB00?style=flat-square&logo=googlecolab" alt="Colab"></a>
</p>

---

## 什么是 VLA？

**VLA（Vision-Language-Action）** 是具身智能领域的核心范式：模型同时接收**视觉图像**和**自然语言指令**，直接输出**机器人动作**，实现"看图 + 听话 → 执行"的端到端闭环。

```
视觉输入 (RGB图像) ──┐
                    ├──→ VLA 模型 ──→ 机器人动作 (关节角度/末端位姿/增量)
语言指令 ("拿起红杯子") ──┘
```

VLA 的本质是将大语言模型的语义理解能力与视觉感知融合，再映射到低维动作空间，是通往通用机器人智能的关键路径。

---

## 学完能达到什么水平？

完成全部 4 个阶段后，你将能够：

- **理解** VLA 的完整架构（视觉编码器 → 语言主干 → 融合 → 策略头）
- **运行** OpenVLA-7B 推理，输出可执行的机器人动作
- **搭建** PyBullet 仿真闭环 demo（图像 → VLA → 动作 → 仿真 → 循环）
- **微调** OpenVLA 在 LIBERO 仿真基准上训练，达到可评估的成功率
- **分析** 评估结果，调试失败案例，迭代优化策略

**一句话：从"知道 VLA 是什么"到"能在仿真上微调并评估自己的 VLA 模型"。**

---

## 学习路径

| 阶段 | 主题 | 目标 | 可运行代码 | 预计时间 |
|------|------|------|-----------|---------|
| **Stage 1** | VLM 基础 | CLIP 对比学习、ViT token 化、attention 可视化 | `README.md` 代码片段 | 1-2 天 |
| **Stage 2** | 动作表示 | FK/IK 动画、4 种动作表示对比 | `fk_ik_demo.py`, `action_representation_comparison.py` | 1 天 |
| **Stage 3** | 简单 VLA | 从零搭建 VLA、OpenVLA 推理、多指令对比 | `build_vla_from_scratch.py`, `openvla_inference_tutorial.py` | 2-3 天 |
| **Stage 4** | 微调实践 | LoRA 微调、LIBERO 评估、自定义数据训练 | `finetune_libero.py`, `evaluate_libero.py`, `train_custom_data.py` | 3-5 天 |

---

## 仓库结构

```
VLA-Zero-to-Hero/
├── README.md                              # 本文件
├── docs/                                  # 概念文档
│   ├── 01-what-is-vla.md                 # VLA 核心概念详解
│   ├── 02-key-papers.md                  # 10 篇关键论文导读
│   ├── 03-learning-path.md               # 完整学习路线（含周计划）
│   ├── 04-glossary.md                    # 术语表（A-Z + 缩写速查）
│   ├── 05-interview-prep.md              # VLA 工程师面试题 88 道
│   ├── 06-rl-fundamentals-for-vla.md     # 强化学习基础（RL + VLA 交叉点）
│   └── 07-world-models-for-vla.md        # 世界模型指南（Dreamer/JEPA/DIAMOND）
├── tutorials/                             # 分阶段教程（含可运行脚本）
│   ├── 01-vlm-basics/                    # Stage 1: CLIP、ViT、VLM
│   ├── 02-action-representation/        # Stage 2: FK/IK、动作表示、Action Chunking
│   │   ├── fk_ik_demo.py                 #   2-DOF 机械臂动画
│   │   └── action_representation_comparison.py  #   4 种表示对比
│   ├── 03-simple-vla/                    # Stage 3: 搭建 VLA、OpenVLA 推理
│   │   ├── build_vla_from_scratch.py     #   CLIP + MLP 教学版 VLA
│   │   └── openvla_inference_tutorial.py  #   OpenVLA 多指令对比推理
│   └── 04-fine-tuning/                   # Stage 4: 微调与评估
│       ├── finetune_libero.py            #   LIBERO LoRA 微调（完整训练循环）
│       ├── evaluate_libero.py            #   LIBERO 闭环评估 + 成功率统计
│       ├── train_custom_data.py          #   自定义 JSONL 数据微调
│       ├── dataset_utils.py              #   数据工具函数库
│       └── README.md                      #   详细文档 + 常见问题
├── examples/                              # 可运行示例
│   ├── minimal_vla.py                    # 最小 VLA 架构 demo
│   ├── inference_demo.py                  # OpenVLA 推理示例
│   ├── sim_closed_loop_demo.py            # PyBullet 仿真闭环 demo
│   ├── visualize_vla.py                  # 动作轨迹 / 注意力 / 评估可视化
│   └── quick_start.ipynb                 # Colab 快速入门 Notebook
├── setup/
│   └── environment.yml                    # Conda 环境
└── resources/
    └── README.md                          # 数据集/模型/工具资源索引
```

---

## 快速开始

### 方式 A：Google Colab（零配置，推荐入门）

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](examples/quick_start.ipynb)

直接打开 `examples/quick_start.ipynb`，在浏览器中从 CLIP 基础走到 OpenVLA 推理。

### 方式 B：本地环境

```bash
# 克隆仓库
git clone https://github.com/Dld0621/VLA-Zero-to-Hero.git
cd VLA-Zero-to-Hero

# 创建 conda 环境（Stage 1-3 基础版）
conda env create -f setup/environment.yml
conda activate vla-zero

# Stage 1: 运行最小 VLA（无需 GPU）
python examples/minimal_vla.py --instruction "pick up the object"

# Stage 2: FK/IK 动画
python tutorials/02-action-representation/fk_ik_demo.py --target 1.2,0.8

# Stage 3: 从零搭建 VLA（需要 GPU，或 Colab）
python tutorials/03-simple-vla/build_vla_from_scratch.py --num_epochs 50

# Stage 4: LIBERO 微调（需要 A100 / 48GB+ GPU）
python tutorials/04-fine-tuning/finetune_libero.py \
    --vla_path openvla/openvla-7b \
    --task_suite libero_spatial \
    --batch_size 4 \
    --lora_rank 32

# Stage 4: LIBERO 评估
python tutorials/04-fine-tuning/evaluate_libero.py \
    --pretrained_checkpoint checkpoints/my_vla \
    --task_suite_name libero_spatial

# 仿真闭环 demo
pip install pybullet
python examples/sim_closed_loop_demo.py --mode scripted

# 可视化评估结果
python examples/visualize_vla.py --mode eval_results --eval_path results/eval_results.json
```

---

## 硬件要求

| 阶段 | 最低 GPU | 推荐 GPU | 可否 CPU |
|------|---------|---------|---------|
| Stage 1 | 无 | 无 | ✓ |
| Stage 2 | 无 | 无 | ✓ |
| Stage 3 (搭建) | T4 (16GB) | A100 (40GB) | 量化模式 |
| Stage 3 (OpenVLA) | T4 (16GB) | A100 (40GB) | ✗ |
| Stage 4 (微调) | RTX 4090 (24GB) | A100 (80GB) | ✗ |
| 仿真 demo | 无 GPU 需要 | 无 GPU 需要 | ✓ |

省显存技巧：Stage 4 支持 `--batch_size 1 --grad_accumulation_steps 8` 和 `--load_in_8bit` 降低到 27GB。

---

## 前置知识

- **Python** 编程（NumPy、PyTorch 基础）
- **深度学习** 基础（CNN、Transformer、注意力机制）
- **机器人学** 基础（关节空间 vs 任务空间）

如不具备，`tutorials/01-vlm-basics/` 提供补充材料。

---

## 关键论文速览

| 论文 | 年份 | 贡献 |
|------|------|------|
| RT-1 / RT-2 | 2022/2023 | Google DeepMind 机器人 Transformer 系列，VLA 的先驱 |
| OpenVLA | 2024 | 7B 参数开源 VLA，DINOv2 + SigLIP + Llama 2 |
| π0 (pi-zero) | 2024 | Physical Intelligence 流匹配 VLA，精细操作 |
| Octo | 2024 | 开源通用策略，27M 参数，多机器人支持 |

完整论文导读见 [`docs/02-key-papers.md`](docs/02-key-papers.md)。

---

## 核心概念一句话总结

- **VLM**：看图 + 读文字 → 输出文字描述
- **VLA**：看图 + 读文字 → 输出机器人动作
- **Action Chunking**：一次预测多步动作，减少推理延迟
- **LoRA**：只训练 0.1% 参数即可微调 7B 模型
- **LIBERO**：标准仿真基准，评估 VLA 操作成功率
- **RL (强化学习)**：agent 通过试错与环境交互，最大化累积奖励（VLA 的 BC 天花板需 RL 突破）
- **World Model**：预测"做了动作后世界怎么变"，用于规划/评估/生成数据（Dreamer/JEPA/DIAMOND）
- **PPO / SAC**：VLA 微调和对齐的主流 RL 算法

完整术语表见 [`docs/04-glossary.md`](docs/04-glossary.md)。强化学习基础见 [`docs/06-rl-fundamentals-for-vla.md`](docs/06-rl-fundamentals-for-vla.md)。世界模型见 [`docs/07-world-models-for-vla.md`](docs/07-world-models-for-vla.md)。

---

## 推荐数据集与模型

| 名称 | 类型 | 说明 |
|------|------|------|
| Open X-Embodiment | 数据集 | 1M+ 轨迹，22+ 机器人平台 |
| LIBERO | 数据集 | 130 语言条件任务，4 个 benchmark |
| OpenVLA-7B | 模型 | 开源 VLA，可直接推理或微调 |
| Octo-Base | 模型 | 27M 参数，CPU 可运行 |

更多资源见 [`resources/README.md`](resources/README.md)。

---

## 贡献指南

欢迎提交 Issue 和 PR！你可以：

- 补充更多论文解读
- 添加新的代码示例
- 修正文档错误
- 分享学习笔记

---

## 许可证

[MIT License](LICENSE)

---

## Acknowledgments

- [OpenVLA](https://github.com/openvla/openvla) — 开源 VLA 模型与微调脚本
- [Octo](https://github.com/octo-models/octo) — 开源通用机器人策略
- [Diffusion Policy](https://github.com/real-stanford/diffusion_policy) — 扩散策略
- [LIBERO](https://github.com/Lifelong-Robot-Learning/LIBERO) — 仿真评估基准
