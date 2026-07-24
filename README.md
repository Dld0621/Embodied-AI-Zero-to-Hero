<h1 align="center">Embodied AI Zero to Zero</h1>

<p align="center">
  <b>具身智能完整知识体系 — 从大一新生零基础到 2026 前沿研究</b>
</p>

<p align="center">
  <a href="https://github.com/Dld0621/Embodied-AI-Zero-to-Zero"><img src="https://img.shields.io/github/stars/Dld0621/Embodied-AI-Zero-to-Zero?style=flat-square" alt="Stars"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/License-MIT-blue?style=flat-square" alt="License"></a>
  <a href="https://python.org"><img src="https://img.shields.io/badge/Python-3.10+-3776AB?style=flat-square&logo=python" alt="Python"></a>
  <a href="https://pytorch.org"><img src="https://img.shields.io/badge/PyTorch-2.0+-EE4C2C?style=flat-square&logo=pytorch" alt="PyTorch"></a>
  <a href="docs/07-key-papers.md"><img src="https://img.shields.io/badge/Retargeting_Papers-37-blueviolet?style=flat-square" alt="Papers"></a>
  <a href="docs/02-key-papers.md"><img src="https://img.shields.io/badge/VLA_Papers-15-success?style=flat-square" alt="VLA Papers"></a>
  <a href="#项目结构"><img src="https://img.shields.io/badge/Docs-27-important?style=flat-square" alt="Docs"></a>
  <a href="examples/evaluation_framework.py"><img src="https://img.shields.io/badge/Benchmark-5_methods-orange?style=flat-square" alt="Benchmark"></a>
</p>

<p align="center">
  <b>27 本文档 · 18+ 个可运行示例 · 52+ 篇关键论文导读 · 80+ 篇 Arxiv 扫描 · 20+ 篇前沿论文在线链接</b>
</p>

---

## 整体框架

本项目覆盖具身智能四大支柱，每个支柱都有完整的 Pipeline 文档和可运行的代码。

```
                    ┌─────────────────────────────────────┐
                    │      Embodied AI Zero to Zero       │
                    │   具身智能完整知识体系 (4 大支柱)    │
                    └──────────────┬──────────────────────┘
                                   │
        ┌──────────────┬───────────┼───────────┬──────────────┐
        │              │           │           │              │
        ▼              ▼           ▼           ▼              ▼
   ┌─────────┐  ┌─────────┐  ┌─────────┐  ┌─────────┐  ┌─────────────┐
   │重定向    │  │ VLA     │  │  RL     │  │ 世界模型 │  │  前沿研究    │
   │Retarget │  │视觉-语言│  │强化学习  │  │World    │  │ 2026 趋势   │
   │ing      │  │-动作    │  │         │  │Models   │  │ 面试题      │
   └────┬────┘  └────┬────┘  └────┬────┘  └────┬────┘  └──────┬──────┘
        │            │            │            │               │
   人手运动->     图像+语言     试错学习      预测未来       研究转向
   机器人手       -> 动作      -> 最优策略    -> 规划       + 开源项目
```

| 支柱 | 一句话定位 | 完整 Pipeline | 可运行代码 | 外部开源项目 |
|------|-----------|-------------|-----------|------------|
| **重定向** | 人手运动映射到机器人灵巧手 | [Pipeline 详解](docs/12-freshman-zero-to-one.md) | [ freshman_zero_to_one.py](examples/freshman_zero_to_one.py) | [DexMV](https://github.com/yzqin/dexmv-sim) · [SPIDER](https://github.com/facebookresearch/spider) · [TopoRetarget](https://toporetarget2026.github.io/TopoRetarget/) |
| **VLA** | 让机器人"看懂+听懂"再行动 | [Pipeline 详解](docs/13-vla-zero-to-one.md) | [vla_demo.py](examples/vla_demo.py) · [minimal_vla.py](examples/minimal_vla.py) | [OpenVLA](https://github.com/openvla/openvla) · [Octo](https://github.com/octo-models/octo) · [LeRobot](https://github.com/huggingface/lerobot) |
| **RL** | 通过环境交互学习最优策略 | [Pipeline 详解](docs/14-rl-zero-to-one.md) | [rl_demo.py](examples/rl_demo.py) | [Stable Baselines3](https://stable-baselines3.readthedocs.io/) · [DexGraspNet](https://github.com/PKU-EPIC/DexGraspNet) |
| **世界模型** | 预测世界变化，辅助规划与训练 | [Pipeline 详解](docs/15-world-model-zero-to-one.md) | [world_model_demo.py](examples/world_model_demo.py) · [dreamer_rssm.py](examples/dreamer_rssm.py) | [DIAMOND](https://github.com/eloialonso/diamond) · [DreamDojo](https://github.com/NVIDIA/DreamDojo) · [PointWorld](https://github.com/NVlabs/PointWorld) |

---

## 四大支柱详解

### 1. 重定向 (Retargeting) — 人手运动 -> 机器人灵巧手

> 将人手 21 点坐标映射到 Shadow/Allegro/LEAP 等灵巧手的关节角度，是遥操作和模仿学习的核心环节。

**完整 Pipeline（点击深入）**：

| 阶段 | 内容 | 文档 | 示例代码 |
|------|------|------|---------|
| **Step 0** | 关节/关节角/Ctrl 核心概念 | [`docs/00-joint-concepts.md`](docs/00-joint-concepts.md) | — |
| **Step 1** | 人手 21 点仿真 + 坐标获取 | [`docs/12-freshman-zero-to-one.md`](docs/12-freshman-zero-to-one.md) | [`freshman_zero_to_one.py`](examples/freshman_zero_to_one.py) |
| **Step 2** | FK/IK 原理与 Jacobian | [`tutorials/01-fk-ik-basics/`](tutorials/01-fk-ik-basics/) | [`fk_ik_demo.py`](examples/fk_ik_demo.py) |
| **Step 3** | Rule-based 角度映射 | [`tutorials/02-rule-based-retargeting/`](tutorials/02-rule-based-retargeting/) | [`landmark_to_joint.py`](examples/landmark_to_joint.py) |
| **Step 4** | Vector Optimization (scipy) | [`tutorials/03-vector-optimization/`](tutorials/03-vector-optimization/) | [`minimal_retargeting.py`](examples/minimal_retargeting.py) |
| **Step 5** | 完整 Pipeline：视觉->坐标->重定向 | [`tutorials/05-complete-pipeline/`](tutorials/05-complete-pipeline/) | [`complete_retargeting_pipeline.py`](examples/complete_retargeting_pipeline.py) |
| **Step 6** | DexMV 高精度 IK (SLSQP + Huber Loss) | [`docs/11-dexmv-research-guide.md`](docs/11-dexmv-research-guide.md) | [`dexmv_style_retargeting/`](examples/dexmv_style_retargeting/) |
| **Step 7** | 评估框架：5 方法基准对比 | [`docs/06-evaluation-metrics.md`](docs/06-evaluation-metrics.md) | [`evaluation_framework.py`](examples/evaluation_framework.py) |

**推荐开源项目实践**：

| 项目 | 说明 | 复刻价值 |
|------|------|---------|
| [DexMV](https://github.com/yzqin/dexmv-sim) | ECCV 2022，SLSQP + Huber Loss | **精度最高**的 IK Retargeting baseline |
| [SPIDER](https://github.com/facebookresearch/spider) | Meta FAIR，物理感知重定向 | 运动学可行 -> 动力学可行轨迹 |
| [DexMachina](https://github.com/MandiZhao/dexmachina) | Stanford+NVIDIA，功能性重定向 | 跨 embodiment 课程学习 |
| [TopoRetarget](https://toporetarget2026.github.io/TopoRetarget/) | 清华 IIIS，交互保持 | 距离加权拉普拉斯优化 |
| [AnyTeleop](https://github.com/dexsuite/dex-teleop) | 视觉遥操作框架 | MediaPipe -> IK -> Shadow Hand 完整链路 |
| [HaMeR](https://github.com/geopavlakos/hamer) | CVPR 2024，手部 mesh 恢复 | 单 RGB -> MANO 参数化模型 |

**→ 进阶工程实践（本团队实际项目）**：

| 项目 | 路径 | 说明 |
|------|------|------|
| **OmniHand MuJoCo** | `../../omnihand_mujoco/` | OmniHand O10 双手 MuJoCo 仿真 + 实时 UDP 遥操作，含 InterHand 数据集复刻与评估 |
| **OmniHand v19** | `../../02_OUR_METHOD/OmniHand_v19/` | 自研重定向主线：21点 → 局部坐标系 → 加权 IK → mimic 补偿 |
| **GeoRT** | `../../GeoRT/` | 基于学习的几何重定向（IROS 2025），支持任意机器人手快速适配 |

> 这些工程项目的**核心实现**与本文档中的理论概念（FK/IK、Vector Optimization、双手镜像）直接对应，可作为从"教程"到"真实代码"的桥梁。

---

### 2. VLA (Vision-Language-Action) — 图像 + 语言指令 -> 机器人动作

> 让机器人不仅能"看懂"图像，还能"听懂"自然语言指令，输出物理动作。是具身智能的主流范式。

**完整 Pipeline（点击深入）**：

| 阶段 | 内容 | 文档 | 示例代码 |
|------|------|------|---------|
| **Step 0** | VLA 核心概念：从 VLM 到 VLA | [`docs/01-what-is-vla.md`](docs/01-what-is-vla.md) | — |
| **Step 1** | VLM 基础：CLIP、视觉 token 化 | [`tutorials/01-vlm-basics/`](tutorials/01-vlm-basics/) | — |
| **Step 2** | 动作表示：FK/IK、Action Chunking | [`tutorials/02-action-representation/`](tutorials/02-action-representation/) | [`fk_ik_demo.py`](examples/fk_ik_demo.py) |
| **Step 3** | 搭建最小 VLA + OpenVLA 推理 | [`tutorials/03-simple-vla/`](tutorials/03-simple-vla/) | [`minimal_vla.py`](examples/minimal_vla.py) |
| **Step 4** | LIBERO 微调实践 | [`tutorials/04-fine-tuning/`](tutorials/04-fine-tuning/) | [`finetune_libero.py`](tutorials/04-fine-tuning/finetune_libero.py) |
| **Step 5** | VLA 论文导读与面试题 | [`docs/02-key-papers.md`](docs/02-key-papers.md) · [`docs/05-interview-prep.md`](docs/05-interview-prep.md) | — |

**推荐开源项目实践**：

| 项目 | 说明 | 复刻价值 |
|------|------|---------|
| [OpenVLA](https://github.com/openvla/openvla) | Stanford/Berkeley 7B VLA | 推理 + 微调 + 部署完整链路 |
| [Octo](https://github.com/octo-models/octo) | 27M-93M 通用机器人策略 | 任意观察/任务/机器人 |
| [LeRobot](https://github.com/huggingface/lerobot) | HuggingFace 社区框架 | SmolVLA + Diffusion Policy + 数据集 |
| [Diffusion Policy](https://github.com/real-stanford/diffusion_policy) | 扩散策略 | 多峰动作分布，平滑动作生成 |
| [RDT-1B](https://github.com/thu-ml/RDT-1B) | 双臂扩散策略 | Diffusion Transformer |

---

### 3. RL (Reinforcement Learning) — 试错学习 -> 最优策略

> 通过环境与奖励信号的交互，让机器人自主学会完成任务的策略。VLA 的 BC 预训练后常用 RL 做 fine-tuning。

**完整 Pipeline（点击深入）**：

| 阶段 | 内容 | 文档 | 示例代码 |
|------|------|------|---------|
| **Step 0** | RL 基础：MDP、Value Function、Policy Gradient | [`docs/06-rl-fundamentals-for-vla.md`](docs/06-rl-fundamentals-for-vla.md) | — |
| **Step 1** | Q-Learning 原理演示（纯 numpy） | [`docs/14-rl-zero-to-one.md`](docs/14-rl-zero-to-one.md) | [`rl_demo.py --mode demo`](examples/rl_demo.py) |
| **Step 2** | SAC + HER + Shadow Hand 灵巧操作 | [`docs/14-rl-zero-to-one.md`](docs/14-rl-zero-to-one.md) | [`rl_demo.py --mode train`](examples/rl_demo.py) |
| **Step 3** | PPO Fine-tuning VLA 策略 | [`docs/06-rl-fundamentals-for-vla.md`](docs/06-rl-fundamentals-for-vla.md) | — |
| **Step 4** | Reward Shaping 设计原则 | [`docs/06-rl-fundamentals-for-vla.md`](docs/06-rl-fundamentals-for-vla.md) | — |

**推荐开源项目实践**：

| 项目 | 说明 | 复刻价值 |
|------|------|---------|
| [Stable Baselines3](https://stable-baselines3.readthedocs.io/) | PyTorch RL 算法库 | PPO/SAC/TRPO 完整实现 |
| [DexGraspNet](https://github.com/PKU-EPIC/DexGraspNet) | 灵巧抓取数据集 | 百万级抓取 + 物理筛选 |
| [OpenAI Spinning Up](https://spinningup.openai.com/en/latest/) | RL 入门教程 | 理论 + 代码双管齐下 |

---

### 4. 世界模型 (World Models) — 预测未来 -> 辅助规划与训练

> 学习"世界的运行规律"：给定当前观测和动作，预测下一个观测。用于数据生成、安全验证、长程规划。

**完整 Pipeline（点击深入）**：

| 阶段 | 内容 | 文档 | 示例代码 |
|------|------|------|---------|
| **Step 0** | 世界模型核心概念与分类 | [`docs/07-world-models-for-vla.md`](docs/07-world-models-for-vla.md) | — |
| **Step 1** | 最小世界模型（线性 + MPC） | [`docs/15-world-model-zero-to-one.md`](docs/15-world-model-zero-to-one.md) | [`world_model_demo.py`](examples/world_model_demo.py) |
| **Step 2** | DreamerV3 RSSM 深度实现 | [`docs/15-world-model-zero-to-one.md`](docs/15-world-model-zero-to-one.md) | [`dreamer_rssm.py`](examples/dreamer_rssm.py) |
| **Step 3** | 世界模型 + VLA 四种融合方式 | [`docs/07-world-models-for-vla.md`](docs/07-world-models-for-vla.md) | [`world_model_vla_pipeline.py`](examples/world_model_vla_pipeline.py) |
| **Step 4** | 2026 前沿：PointWorld / DreamDojo / RISE | [`docs/07-world-models-for-vla.md`](docs/07-world-models-for-vla.md) | — |

**推荐开源项目实践**：

| 项目 | 说明 | 复刻价值 |
|------|------|---------|
| [DIAMOND](https://github.com/ethz-rl/diamond) | Diffusion 世界模型 | 像素级预测 + Atari 验证 |
| [DreamDojo](https://github.com/NVIDIA/DreamDojo) | 通用世界模型 | 4.4 万小时人类视频预训练 |
| [RISE](https://github.com/OpenDriveLab/RISE) | 组合式世界模型 | 想象 RL + 真实机器人部署 |
| [PointWorld](https://github.com/NVlabs/PointWorld) | 3D 跨本体世界模型 | 3D Point Flow + MPC 实时规划 |
| [IRIS](https://github.com/eloialonso/iris) | Transformer 世界模型 | 代码结构清晰，适合学习 |

---

## 30 秒快速开始

```bash
git clone https://github.com/Dld0621/Embodied-AI-Zero-to-Zero.git
cd Embodied-AI-Zero-to-Zero
pip install numpy scipy mujoco matplotlib

# 大一新生 0->1：人手仿真 -> 坐标输入 -> 机器人重定向
cd examples
python freshman_zero_to_one.py --gesture open --model shadow
```

**输出**：
```
[Step 1/6] 获取人手 21 点坐标
[Step 5/6] DexMV Retargeting (SLSQP + Huber Loss)
  重定向耗时: 0.003s (2.5 ms/帧)
[Step 6/6] 评估重定向精度
  平均 FPE: 61.02 mm
```

> **完成！** 你已经成功将人手坐标重定向到了 Shadow Hand 的 24 个关节。详见 [`docs/12-freshman-zero-to-one.md`](docs/12-freshman-zero-to-one.md)。

---

## 谁适合本项目？

| 人群 | 起点 | 推荐入口 |
|------|------|---------|
| **大一新生 / 零基础** | 只学过 Python 基础 | [重定向 Step 1](docs/12-freshman-zero-to-one.md) |
| **研究生 / 转方向** | 有机器学习/控制基础 | [DexMV 研究指南](docs/11-dexmv-research-guide.md) · [VLA 入门](docs/01-what-is-vla.md) |
| **VLA 学习者** | 想了解视觉-语言-动作模型 | [VLA 核心概念](docs/01-what-is-vla.md) · [最小 VLA](examples/minimal_vla.py) |
| **工程师 / 快速落地** | 需要直接可用的代码 | [DexMV 代码](examples/dexmv_style_retargeting/) · [OpenVLA 推理](examples/vla_demo.py) |
| **面试准备** | 找实习/全职工作 | [面试题汇总](docs/05-interview-prep.md) · [术语表](docs/04-glossary.md) |
| **前沿追踪者** | 想了解 2026 最新进展 | [研究趋势](docs/17-research-trends-and-positioning.md) · [前沿论文](docs/18-frontier-papers-online.md) |

---

## 2026 前沿研究亮点

> 2026 年具身智能已进入**"后几何时代"**——从 `human pose -> robot pose` 转向 `human intent + morphology + object interaction + physics -> robot-feasible action`。

### 六大研究转向

| 转向 | 传统方法 | 2026 新方向 | 代表论文 |
|------|---------|------------|---------|
| 几何 -> 物理 | 纯运动学优化 | 物理可行性约束 | [SPIDER](https://arxiv.org/abs/2511.09484) |
| 标定 -> 免标定 | 手工规则校准 | 少量示范自适应 | [AnyDexRT](https://arxiv.org/abs/2607.08341) |
| 姿态 -> 交互 | 指尖位置匹配 | 手-物接触拓扑保持 | [TopoRetarget](https://arxiv.org/abs/2606.16272) |
| 统一 -> 条件化 | 单一映射 | 手势/阶段条件化 | [VTAP Gripper](https://arxiv.org/abs/2607.15448) |
| 在线 -> 推理 | 每帧迭代求解 | 学习后前向推理 | [GeoRT](https://arxiv.org/abs/2503.07541) |
| 关节 -> 功能 | 复制关节形状 | 功能等效 | [DexMachina](https://arxiv.org/abs/2505.24853) |

### 有开源代码的前沿项目

| 项目 | 方向 | 代码 | 核心方法 |
|------|------|------|---------|
| [SPIDER](https://github.com/facebookresearch/spider) | 物理感知重定向 | | Meta FAIR，运动学->动力学可行轨迹 |
| [DexMachina](https://github.com/MandiZhao/dexmachina) | 功能性重定向 | | Stanford+NVIDIA，课程学习跨 embodiment |
| [TopoRetarget](https://toporetarget2026.github.io/TopoRetarget/) | 交互保持重定向 | 项目页 | 清华 IIIS，距离加权拉普拉斯优化 |
| [EquiDexFlow](https://equidexflow.github.io/) | 接触感知生成流 | 项目页 | SE(3) 等变，接触+法向+力学联合预测 |
| [DreamDojo](https://github.com/NVIDIA/DreamDojo) | 通用世界模型 | | NVIDIA，从 4.4 万小时人类视频预训练 |
| [RISE](https://github.com/OpenDriveLab/RISE) | 组合式世界模型 | | OpenDriveLab，想象 RL + 真实部署 |
| [PointWorld](https://github.com/NVlabs/PointWorld) | 3D 跨本体世界模型 | | NVIDIA，3D Point Flow + MPC 规划 |

> 完整前沿论文在线链接见 [`docs/18-frontier-papers-online.md`](docs/18-frontier-papers-online.md)

---

## 项目结构

```
Embodied-AI-Zero-to-Zero/
|-- docs/                              # 27 本核心文档
|   |-- 00-concepts-encyclopedia.md    # 全概念百科
|   |-- 00-joint-concepts.md           # 关节/关节角/Ctrl 核心概念
|   |-- 01-what-is-ik-retargeting.md   # Retargeting 核心概念
|   |-- 01-what-is-vla.md              # VLA 核心概念详解
|   |-- 02-key-papers.md               # VLA 关键论文导读
|   |-- 02-retargeting-taxonomy.md     # 方法分类体系
|   |-- 03-human-hand-to-robot-hand.md # 人手->机器人手映射
|   |-- 03-learning-path.md            # VLA 完整学习路线
|   |-- 04-glossary.md                 # VLA 术语表（A-Z）
|   |-- 04-optimization-methods.md     # 优化方法深入
|   |-- 05-interview-prep.md           # 面试题汇总
|   |-- 05-learning-based-methods.md   # 基于学习的方法
|   |-- 06-evaluation-metrics.md       # 评估指标与基准
|   |-- 06-rl-fundamentals-for-vla.md  # RL 基础（VLA 视角）
|   |-- 07-key-papers.md               # 37 篇重定向关键论文
|   |-- 07-world-models-for-vla.md     # 世界模型详解
|   |-- 08-open-source-projects.md     # 开源项目复现指南
|   |-- 09-dexterous-hands-analysis.md # 灵巧手对比
|   |-- 10-manipulation-datasets.md    # 灵巧操作数据集
|   |-- 11-dexmv-research-guide.md     # DexMV 高精度 IK
|   |-- 12-freshman-zero-to-one.md     # 大一新生 0->1 实战
|   |-- 13-vla-zero-to-one.md          # VLA 实战
|   |-- 14-rl-zero-to-one.md           # RL 实战
|   |-- 15-world-model-zero-to-one.md  # 世界模型实战
|   |-- 16-arxiv-retargeting-scan.md   # Arxiv 论文全景扫描
|   |-- 17-research-trends-and-positioning.md  # 研究趋势
|   |-- 18-frontier-papers-online.md   # 前沿论文在线链接
|
|-- examples/                          # 18+ 个可运行示例
|   |-- freshman_zero_to_one.py        # 人手仿真 + DexMV 重定向
|   |-- vla_demo.py                    # VLA 推理演示
|   |-- minimal_vla.py                 # 最小 VLA 架构
|   |-- rl_demo.py                     # RL 演示
|   |-- world_model_demo.py            # 世界模型演示
|   |-- dreamer_rssm.py                # DreamerV3 RSSM
|   |-- world_model_vla_pipeline.py    # WM + VLA 融合
|   |-- fk_ik_demo.py                  # 2D 正逆运动学
|   |-- finger_chain_3d.py             # 3D 手指链 FK/IK
|   |-- landmark_to_joint.py           # 21 点 -> 关节角
|   |-- minimal_retargeting.py         # 三种方法对比
|   |-- evaluation_framework.py        # 综合评估框架
|   |-- complete_retargeting_pipeline.py   # 完整 Pipeline
|   |-- dexmv_style_retargeting/       # DexMV 高精度 IK
|
|-- tutorials/                         # 10 阶段教程
|   |-- 01-fk-ik-basics/               # 正逆运动学（重定向）
|   |-- 01-vlm-basics/                 # VLM 基础（VLA）
|   |-- 02-action-representation/      # 动作表示（VLA）
|   |-- 02-rule-based-retargeting/     # Rule-based（重定向）
|   |-- 03-simple-vla/                 # 简单 VLA（VLA）
|   |-- 03-vector-optimization/        # 向量优化（重定向）
|   |-- 04-fine-tuning/                # VLA 微调（VLA）
|   |-- 04-landmark-pipeline/          # 21 点 landmark（重定向）
|   |-- 05-complete-pipeline/          # 完整 Pipeline（重定向）
|   |-- 05-world-models/               # 世界模型（VLA）
|
|-- pretrained/                        # 预训练模型 + URDF 模型
|-- datasets/                          # 数据集下载脚本
|-- setup/
|   |-- environment.yml                # Conda 环境配置
|-- resources/
|   |-- README.md                      # 数据集/工具/模型/VLA 资源索引
|-- tests/
|   |-- test_imports.py                # 基础导入测试（Smoke Tests）
|-- CONTRIBUTING.md                    # 贡献指南（PR 规范、审查清单）
|-- CHANGELOG.md                       # 版本变更记录
|-- requirements.txt                   # Pip 依赖文件
|-- LICENSE                            # MIT 许可证
|-- .gitignore                         # Git 忽略规则
```

---

## 核心概念速查

### 基础概念

| 概念 | 一句话解释 |
|------|-----------|
| **Joint（关节）** | 连接刚体的运动副，定义"能怎么动" |
| **Joint Angle** | 关节当前状态量，从编码器读取或从 landmarks 计算 |
| **Ctrl（控制量）** | 发给执行器的指令，position actuator 中 ctrl = 目标关节角 |
| **Retargeting** | 将源运动（人手）映射到目标运动（机器人手） |
| **FK** | 已知关节角 -> 计算末端位置（正向） |
| **IK** | 已知末端位置 -> 求解关节角（逆向） |
| **21 点模型** | 人手 21 个关键点：手腕 + 5 指 x 4 关节 |
| **Huber Loss** | Smooth L1：小误差二次惩罚，大误差线性惩罚 |
| **SLSQP** | 序列最小二乘规划，带约束的数值优化方法 |
| **Jacobian** | 末端位置对关节角的偏导数矩阵 |
| **Warm-start** | 用上一帧优化结果作为下一帧初始值，加速收敛 |
| **FPE** | Fingertip Position Error，重定向精度核心指标 |

### VLA 核心概念

| 概念 | 一句话解释 |
|------|-----------|
| **VLA** | 视觉-语言-动作模型：图像 + 语言指令 -> 机器人动作 |
| **VLM** | 视觉-语言模型：图像 + 文本 -> 文本 |
| **Action Chunking** | 一次预测未来多步动作序列，减少推理频率 |
| **Policy Head** | 将融合特征映射为动作输出的模型尾部 |
| **BC (Behavior Cloning)** | 监督学习：模仿专家演示数据 |
| **OXE (Open X-Embodiment)** | 最大开源机器人数据集 |
| **Sim-to-Real** | 仿真训练策略迁移到真实机器人 |

### 2026 前沿概念

| 概念 | 一句话解释 | 代表论文 |
|------|-----------|---------|
| **Interaction-Preserving** | 不只匹配指尖位置，更保持手-物接触拓扑关系 | TopoRetarget |
| **Physics-Informed** | 运动学可行 != 物理可行，需纳入碰撞/力/接触约束 | SPIDER |
| **Calibration-Free** | 免标定，少量人工示范即可适配新操作者/新机器人 | AnyDexRT |
| **Functional Retargeting** | 不追求关节形状复制，追求功能等效 | DexMachina |
| **Gesture-Conditioned** | 根据手势/操作阶段切换不同重定向映射 | VTAP Gripper |
| **Morphology Gap** | 人手与机器人手的形态差异，从误差因素升级为对齐变量 | DexGrasp-Zero |
| **Cross-Embodiment** | 跨机器人形态泛化，训练于 A 手迁移到 B 手 | One-Policy-Fits-All |
| **RSSM** | DreamerV3 核心：确定性 GRU + 随机潜状态 | DreamerV3 |
| **World Action Model** | 世界模型同时预测状态和动作 | DreamZero |
| **3D Point Flow** | 跨本体的统一 3D 世界表示 | PointWorld |

---

## 代码速查

```bash
# === 大一新生 0->1（重定向，推荐首次运行）===
cd examples
python freshman_zero_to_one.py --gesture open --model shadow
python freshman_zero_to_one.py --gesture open --visualize-human --visualize-robot

# === FK/IK 基础 ===
python examples/fk_ik_demo.py --mode fk
python examples/fk_ik_demo.py --mode ik
python examples/finger_chain_3d.py --mode ik

# === Rule-based Retargeting ===
python examples/landmark_to_joint.py --hand right --gesture open

# === 三种方法对比 ===
python examples/minimal_retargeting.py --method compare

# === DexMV 高精度 Retargeting ===
cd examples/dexmv_style_retargeting
python run_pipeline.py --model shadow --n_frames 60 --visualize

# === 完整 Pipeline ===
python examples/complete_retargeting_pipeline.py --method all --visualize

# === 评估框架 ===
python examples/evaluation_framework.py --method all --n_samples 100

# === VLA 推理演示 ===
python examples/vla_demo.py --mode synthetic --task "pick up the apple"
python examples/minimal_vla.py

# === RL 强化学习 ===
python examples/rl_demo.py --mode demo     # numpy Q-learning（无需安装）
python examples/rl_demo.py --mode train    # SAC+HER Shadow Hand（需要GPU）

# === 世界模型 ===
python examples/world_model_demo.py --mode concept  # numpy 线性模型 + MPC
python examples/dreamer_rssm.py --epochs 25         # DreamerV3 RSSM
python examples/world_model_vla_pipeline.py         # WM + VLA 融合
```

---

## 支持的手模型

| 模型 | DOF | 指数量 | 文件 | 状态 |
|------|-----|--------|------|------|
| **Shadow Hand** | 24 | 5 | `pretrained/urdf/mujoco_menagerie/shadow_hand/` | 完整支持 |
| **Allegro Hand** | 16 | 4 | `pretrained/urdf/allegro_hand_right/` | 完整支持 |
| **LEAP Hand** | 16 | 4 | `pretrained/urdf/leap_hand_sim/` | 完整支持 |
| **O10 / OmniHand** | 10 | 5 | 外部硬件 | 需真机 |

---

## 外部学习资源

### 教材与课程

| 资源 | 类型 | 说明 |
|------|------|------|
| [Modern Robotics (Lynch & Park)](http://hades.mech.northwestern.edu/index.php/Modern_Robotics) | 教材 | 刚体运动学、Jacobian、开链/闭链系统 |
| [MuJoCo Menagerie](https://github.com/google-deepmind/mujoco_menagerie) | 代码 | 预构建机器人模型库 |
| [Diffusion Policy 官方教程](https://diffusion-policy.cs.columbia.edu/) | 教程 | 扩散策略从原理到代码 |
| [Stanford CS224R](https://cs224r.stanford.edu/) | 课程 | Stanford 机器人学习课程 |
| [OpenAI Spinning Up](https://spinningup.openai.com/en/latest/) | 教程 | RL 最经典入门教程 |
| [UCB CS285 -- Deep RL](https://rail.eecs.berkeley.edu/deeprlcourse/) | 课程 | Berkeley 深度强化学习 |

### 相关项目

- [Embodied-AI-Paper-Analysis](https://github.com/Dld0621/Embodied-AI-Paper-Analysis) -- 具身智能论文体系化梳理

---

## 贡献指南

欢迎提交 Issue 和 PR！支持方向：

- 补充新的 retargeting 方法（Diffusion Policy、强化学习等）
- 添加更多机器人手模型（Inspire Hand、SVH 等）
- 完善 VLA 微调教程和评估基准
- 补充世界模型与 VLA 融合的最新进展
- 补充前沿论文在线链接和代码复现指南
- 修正文档错误、分享学习笔记

---

## 许可证

[MIT License](LICENSE)

---

## Acknowledgments

- [MediaPipe](https://mediapipe-studio.webapps.google.com/demo/hand_landmarker) -- 实时手部 landmarks 检测
- [InterHand2.6M](https://mks0601.github.io/InterHand2.6M/) -- 双手 3D 姿态数据集
- [MuJoCo Menagerie](https://github.com/google-deepmind/mujoco_menagerie) -- 机器人模型库
- [DexMV](https://github.com/yzqin/dexmv-sim) -- ECCV 2022 高精度 IK Retargeting
- [OpenVLA](https://github.com/openvla/openvla) -- Stanford / Berkeley 开源 VLA
- [Octo](https://github.com/octo-models/octo) -- 通用机器人策略
- [SPIDER](https://github.com/facebookresearch/spider) -- Meta FAIR 物理感知重定向
- [DexMachina](https://github.com/MandiZhao/dexmachina) -- Stanford+NVIDIA 功能性重定向
- [TopoRetarget](https://toporetarget2026.github.io/TopoRetarget/) -- 清华 IIIS 交互保持重定向
- [DreamDojo](https://github.com/NVIDIA/DreamDojo) -- NVIDIA 通用世界模型
- [PointWorld](https://github.com/NVlabs/PointWorld) -- NVIDIA 3D 跨本体世界模型
