<p align="center">
  <img src="https://raw.githubusercontent.com/Dld0621/VLA-Zero-to-Hero/master/assets/banner.jpg" alt="Embodied AI Zero to Hero Banner" width="100%">
</p>

<h1 align="center">Embodied AI Zero to Hero</h1>

<p align="center">
  <b>VLA · 强化学习 · 世界模型 — 从零到具身智能工程师的完整学习路径</b>
</p>

<p align="center">
  <a href="https://github.com/Dld0621/VLA-Zero-to-Hero"><img src="https://img.shields.io/github/stars/Dld0621/VLA-Zero-to-Hero?style=flat-square" alt="Stars"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/License-MIT-blue?style=flat-square" alt="License"></a>
  <a href="https://python.org"><img src="https://img.shields.io/badge/Python-3.10+-3776AB?style=flat-square&logo=python" alt="Python"></a>
  <a href="https://pytorch.org"><img src="https://img.shields.io/badge/PyTorch-2.0+-EE4C2C?style=flat-square&logo=pytorch" alt="PyTorch"></a>
  <a href="https://github.com/openvla/openvla"><img src="https://img.shields.io/badge/OpenVLA-7B-blueviolet?style=flat-square" alt="OpenVLA"></a>
  <a href="https://github.com/Lifelong-Robot-Learning/LIBERO"><img src="https://img.shields.io/badge/LIBERO-Simulation-ff9900?style=flat-square" alt="LIBERO"></a>
  <a href="docs/07-world-models-for-vla.md"><img src="https://img.shields.io/badge/World_Models-Dreamer%20%7C%20JEPA%20%7C%20DIAMOND-9cf" alt="World Models"></a>
  <a href="docs/06-rl-fundamentals-for-vla.md"><img src="https://img.shields.io/badge/RL-PPO%20%7C%20SAC%20%7C%20DQN-orange" alt="RL"></a>
  <a href="docs/05-interview-prep.md"><img src="https://img.shields.io/badge/Interview_Prep-88_Qs-green" alt="Interview"></a>
  <a href="examples/quick_start.ipynb"><img src="https://img.shields.io/badge/Google_Colab-Compatible-F9AB00?style=flat-square&logo=googlecolab" alt="Colab"></a>
</p>

---

## 这是什么？

**具身智能（Embodied AI）** 是让智能体通过"感知 → 思考 → 行动"的闭环在物理世界中完成任务。本项目覆盖三大核心技术支柱：

```
┌─────────────────────────────────────────────────────────┐
│                    具身智能技术栈                         │
│                                                         │
│  ┌──────────┐    ┌──────────────┐    ┌───────────────┐  │
│  │   VLA    │    │  强化学习 RL   │    │   世界模型     │  │
│  │ 看+听→做  │    │ 试错→最优策略  │    │ 预测世界变化   │  │
│  │ RT-2     │    │  PPO / SAC    │    │  Dreamer V3   │  │
│  │ OpenVLA  │    │  DQN / TD3    │    │  V-JEPA 2     │  │
│  │ π0       │    │  BC + RL      │    │  DIAMOND      │  │
│  └──────────┘    └──────────────┘    └───────────────┘  │
│         ↕               ↕                   ↕         │
│         └──────── 三者深度融合 ────────┘                  │
│                   通用机器人智能                        │
└─────────────────────────────────────────────────────────┘
```

- **VLA（Vision-Language-Action）**：看图 + 听话 → 直接输出机器人动作（RT-2、OpenVLA、π0）
- **强化学习（RL）**：通过试错与环境交互，学习最优策略（PPO、SAC、DQN），突破 VLA 行为克隆的天花板
- **世界模型（World Model）**：预测"做了动作后世界怎么变"（Dreamer、JEPA、DIAMOND），用于规划/评估/数据生成

---

## 学完能达到什么水平？

- **理解** VLA / RL / World Model 三大支柱的核心原理与交叉关系
- **运行** OpenVLA-7B 推理，输出可执行的机器人动作
- **搭建** PyBullet 仿真闭环 demo（图像 → VLA → 动作 → 仿真 → 循环）
- **微调** OpenVLA 在 LIBERO 仿真基准上训练并评估
- **掌握** PPO/SAC 等 RL 算法，理解 BC + RL 两阶段训练
- **理解** Dreamer/JEPA/DIAMOND 等世界模型路线，知道如何与 VLA 融合
- **准备** VLA / 具身智能算法工程师面试（88 道题）

**一句话：从"知道具身智能是什么"到"能在仿真上微调 VLA、理解 RL 和世界模型、通过算法工程师面试"。**

---

## 知识地图

| 支柱 | 子方向 | 关键技术 | 学习资源 |
|------|--------|---------|---------|
| **VLA** | 视觉编码器 | CLIP, DINOv2, SigLIP | `tutorials/01-vlm-basics/` |
| | 语言主干 | LLaMA 2, T5, PaLI-X | `docs/01-what-is-vla.md` |
| | 策略头 | MLP 回归, 扩散/流匹配, 离散 token | `tutorials/03-simple-vla/` |
| | 动作表示 | 关节角, 末端位姿, delta, chunking | `tutorials/02-action-representation/` |
| | 微调与部署 | LoRA, OFT, 量化, REST API | `tutorials/04-fine-tuning/` |
| **RL** | Value-Based | DQN, Double DQN, Dueling DQN | `docs/06-rl-fundamentals-for-vla.md` |
| | Policy Gradient | REINFORCE, PPO, A2C/A3C | `docs/06-rl-fundamentals-for-vla.md` |
| | Actor-Critic | SAC, TD3, DDPG | `docs/06-rl-fundamentals-for-vla.md` |
| | VLA + RL | RLVF, BC+RL 两阶段, Hierarchical RL | `docs/06-rl-fundamentals-for-vla.md` |
| **World Model** | Latent Dynamics | Dreamer V1/V2/V3, RSSM | `docs/07-world-models-for-vla.md` |
| | 隐式动力学 | MuZero, EfficientZero | `docs/07-world-models-for-vla.md` |
| | Transformer/Diffusion | IRIS, DIAMOND, STORM | `docs/07-world-models-for-vla.md` |
| | 非生成式 | V-JEPA, V-JEPA 2 | `docs/07-world-models-for-vla.md` |
| | Foundation WM | Genie, UniSim, Cosmos | `docs/07-world-models-for-vla.md` |
| | WM + VLA 融合 | 数据生成, 评估, 规划, WAM | `docs/07-world-models-for-vla.md` |
| **面试** | DL 基础 | Transformer, LoRA, Flash Attention | `docs/05-interview-prep.md` |
| | 算法深挖 | RT-2, OpenVLA, π0, Diffusion Policy | `docs/05-interview-prep.md` |
| | 系统设计 | 端到端抓取, 跨平台 pipeline | `docs/05-interview-prep.md` |

---

## 学习路径

| 阶段 | 主题 | 目标 | 可运行代码 | 预计时间 |
|------|------|------|-----------|---------|
| **Stage 1** | VLM 基础 | CLIP 对比学习、ViT token 化、attention 可视化 | `README.md` 代码片段 | 1-2 天 |
| **Stage 2** | 动作表示 | FK/IK 动画、4 种动作表示对比 | `fk_ik_demo.py`, `action_representation_comparison.py` | 1 天 |
| **Stage 3** | VLA 搭建与推理 | 从零搭建 VLA、OpenVLA 推理、多指令对比 | `build_vla_from_scratch.py`, `openvla_inference_tutorial.py` | 2-3 天 |
| **Stage 4** | 微调实践 | LoRA 微调、LIBERO 评估、自定义数据 | `finetune_libero.py`, `evaluate_libero.py` | 3-5 天 |
| **进阶 A** | 强化学习 | MDP → PPO/SAC → BC+RL 两阶段训练 | `docs/06-rl-fundamentals-for-vla.md` | 1-2 天 |
| **进阶 B** | 世界模型 | Dreamer/JEPA/DIAMOND → WM + VLA 融合 | `docs/07-world-models-for-vla.md` | 1-2 天 |
| **进阶 C** | 面试准备 | 88 道面试题（DL/RL/VLA/论文/系统设计） | `docs/05-interview-prep.md` | 随时复习 |

---

## 仓库结构

```
Embodied AI Zero to Hero/
├── docs/                                  # 7 本核心文档
│   ├── 01-what-is-vla.md                 # VLA 核心概念：架构、动作表示、训练范式
│   ├── 02-key-papers.md                  # 10 篇必读论文（RT-1→V-JEPA 2）
│   ├── 03-learning-path.md               # 完整学习路线 + 周计划
│   ├── 04-glossary.md                    # 术语表（A-Z，50+ 术语）
│   ├── 05-interview-prep.md              # 面试题 88 道（DL/RL/VLA/论文/系统设计）
│   ├── 06-rl-fundamentals-for-vla.md     # 强化学习基础 + VLA 交叉应用
│   └── 07-world-models-for-vla.md        # 世界模型指南（5 路线 + VLA 融合）
├── tutorials/                             # 4 阶段实战教程
│   ├── 01-vlm-basics/                    # CLIP、ViT、VLM 推理
│   ├── 02-action-representation/        # FK/IK 动画、4 种动作表示
│   ├── 03-simple-vla/                    # 从零搭建 VLA、OpenVLA 推理
│   └── 04-fine-tuning/                   # LoRA 微调、LIBERO 评估、自定义数据
├── examples/                              # 5 个可运行 demo
│   ├── minimal_vla.py                    # 最小 VLA 架构（教学用）
│   ├── inference_demo.py                  # OpenVLA 真实推理
│   ├── sim_closed_loop_demo.py            # PyBullet 仿真闭环
│   ├── visualize_vla.py                  # 动作轨迹/注意力/评估可视化
│   └── quick_start.ipynb                 # Colab 快速入门
├── setup/environment.yml                  # Conda 环境
└── resources/README.md                    # 数据集/模型/工具索引
```

---

## 快速开始

### Google Colab（零配置）

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](examples/quick_start.ipynb)

### 本地环境

```bash
git clone https://github.com/Dld0621/VLA-Zero-to-Hero.git && cd VLA-Zero-to-Hero
conda env create -f setup/environment.yml && conda activate vla-zero

# Stage 1-2: 无需 GPU
python examples/minimal_vla.py --instruction "pick up the object"
python tutorials/02-action-representation/fk_ik_demo.py --target 1.2,0.8

# Stage 3: 需要 GPU
python tutorials/03-simple-vla/build_vla_from_scratch.py --num_epochs 50

# Stage 4: 需要 A100
python tutorials/04-fine-tuning/finetune_libero.py \
    --vla_path openvla/openvla-7b --task_suite libero_spatial --batch_size 4 --lora_rank 32

# 仿真闭环
python examples/sim_closed_loop_demo.py --mode scripted
```

---

## 硬件要求

| 阶段 | 最低 GPU | 推荐 GPU | CPU |
|------|---------|---------|-----|
| Stage 1-2 | 无 | 无 | ✓ |
| Stage 3 (搭建) | T4 16GB | A100 40GB | 量化 |
| Stage 3 (OpenVLA) | T4 16GB | A100 40GB | ✗ |
| Stage 4 (微调) | 4090 24GB | A100 80GB | ✗ |
| 仿真 demo | — | — | ✓ |

省显存：`--batch_size 1 --grad_accumulation_steps 8 --load_in_8bit`（降至 27GB）。

---

## 核心概念速查

| 概念 | 一句话 |
|------|--------|
| **VLA** | 看图 + 听话 → 输出机器人动作 |
| **VLM** | 看图 + 听话 → 输出文字描述 |
| **Action Chunking** | 一次预测多步动作，减少推理延迟 |
| **LoRA** | 只训练 0.1% 参数即可微调 7B 模型 |
| **PPO** | 策略梯度 + clip，VLA 微调主流 RL 算法 |
| **SAC** | 最大熵 RL，连续控制样本效率高 |
| **World Model** | 预测"做了动作后世界怎么变"（Dreamer/JEPA/DIAMOND） |
| **WAM** | 同时预测未来状态和动作，长程任务 |
| **LIBERO** | 标准仿真基准，评估 VLA 操作成功率 |
| **Sim-to-Real** | 仿真训练 → 真实机器人部署的域迁移 |

完整术语表见 [`docs/04-glossary.md`](docs/04-glossary.md)。

---

## 推荐数据集与模型

| 名称 | 类型 | 说明 |
|------|------|------|
| Open X-Embodiment | 数据集 | 1M+ 轨迹，22+ 机器人平台 |
| LIBERO | 数据集 | 130 语言条件任务，4 个 benchmark |
| OpenVLA-7B | 模型 | 开源 VLA，可直接推理或微调 |
| Octo-Base | 模型 | 27M 参数，CPU 可运行 |
| Dreamer V3 | 模型 | 基于世界模型的 RL |
| DIAMOND | 模型 | Diffusion 做世界模型 |

更多资源见 [`resources/README.md`](resources/README.md)。

---

## 贡献指南

欢迎提交 Issue 和 PR！支持方向：

- 补充论文解读和代码示例
- 添加新算法实现（SAC、Dreamer 等）
- 修正文档错误、分享学习笔记
- 完善面试题库

---

## 许可证

[MIT License](LICENSE)

---

## 相关项目

- [Embodied-AI-Paper-Analysis](https://github.com/Dld0621/Embodied-AI-Paper-Analysis) — 具身智能论文体系化梳理（VLA / RL / World Model 等 7 大方向，按顶会分类，带 venue tier 标注）

## Acknowledgments

- [OpenVLA](https://github.com/openvla/openvla) — 开源 VLA 模型与微调脚本
- [Octo](https://github.com/octo-models/octo) — 开源通用机器人策略
- [Diffusion Policy](https://github.com/real-stanford/diffusion_policy) — 扩散策略
- [LIBERO](https://github.com/Lifelong-Robot-Learning/LIBERO) — 仿真评估基准
- [Dreamer V3](https://github.com/danijar/dreamerv3) — 基于世界模型的 RL
- [DIAMOND](https://github.com/ethz-rl/diamond) — Diffusion 世界模型
- [V-JEPA 2](https://github.com/facebookresearch/v-jepa2) — 非生成式世界模型
