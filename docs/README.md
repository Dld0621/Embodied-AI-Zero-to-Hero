# Documentation Index

> 本文档是 `docs/` 目录的完整索引。README 中未展开的细节、命令速查、概念百科和外部资源均存放于此。

---

## 文档分类索引

### 基础概念 (Foundations)

| 文档 | 内容 | 标签 |
|:-----|:-----|:-----|
| [`01-what-is-vla.md`](01-what-is-vla.md) | VLA 核心概念详解 | Concept |
| [`04-glossary.md`](04-glossary.md) | VLA 术语表（A-Z） | Concept |

### Robot Foundation Models

| 文档 | 内容 | 标签 |
|:-----|:-----|:-----|
| [`23-robot-foundation-models.md`](23-robot-foundation-models.md) | RFM 总览：统一接口与架构 | Concept |
| [`24-action-representation-and-tokenization.md`](24-action-representation-and-tokenization.md) | 动作表示与 Tokenization | Tutorial |
| [`25-cross-embodiment-adaptation.md`](25-cross-embodiment-adaptation.md) | Cross-Embodiment 适配 | Tutorial |
| [`26-rfm-finetuning-and-evaluation.md`](26-rfm-finetuning-and-evaluation.md) | RFM 微调与评测 | Tutorial |
| [`27-embodied-reasoning-and-planning.md`](27-embodied-reasoning-and-planning.md) | 具身推理与规划 | Concept |
| [`28-smolvla-gpu-finetuning-runbook.md`](28-smolvla-gpu-finetuning-runbook.md) | SmolVLA GPU 微调完整指南 | Engineering |
| [`29-learning-tracks-detail.md`](29-learning-tracks-detail.md) | 四大研究方向详细分解（流程、学习层级、实现状态） | Reference |

### VLA (Vision-Language-Action)

| 文档 | 内容 | 标签 |
|:-----|:-----|:-----|
| [`02-key-papers.md`](02-key-papers.md) | VLA 关键论文导读 | Paper |
| [`03-learning-path.md`](03-learning-path.md) | VLA 完整学习路线 | Tutorial |
| [`05-interview-prep.md`](05-interview-prep.md) | 面试题汇总（100+ 题） | Resource |
| [`13-vla-zero-to-one.md`](13-vla-zero-to-one.md) | VLA 实战（SmolVLA） | Tutorial |
| [`20-vla-deployment-guide.md`](20-vla-deployment-guide.md) | VLA 部署优化与边缘计算 | Engineering |
| [`21-vla-dataset-organization.md`](21-vla-dataset-organization.md) | VLA 数据组织与同步 | Tutorial |
| [`22-act-vs-diffusion-policy.md`](22-act-vs-diffusion-policy.md) | ACT vs Diffusion Policy 对比 | Tutorial |

### 世界模型 (World Models)

| 文档 | 内容 | 标签 |
|:-----|:-----|:-----|
| [`07-world-models-for-vla.md`](07-world-models-for-vla.md) | 世界模型详解与 VLA 融合 | Concept |
| [`15-world-model-zero-to-one.md`](15-world-model-zero-to-one.md) | 世界模型实战 | Tutorial |

### 强化学习 (RL)

| 文档 | 内容 | 标签 |
|:-----|:-----|:-----|
| [`06-rl-fundamentals-for-vla.md`](06-rl-fundamentals-for-vla.md) | RL 基础（VLA 视角） | Concept |
| [`14-rl-zero-to-one.md`](14-rl-zero-to-one.md) | RL 实战（SAC+HER） | Tutorial |

### Sim-to-Real

| 文档 | 内容 | 标签 |
|:-----|:-----|:-----|
| [`19-sim-to-real-guide.md`](19-sim-to-real-guide.md) | Sim-to-Real 完整实战指南 | Engineering |

### 研究前沿 (Research)

| 文档 | 内容 | 标签 |
|:-----|:-----|:-----|
| [`17-research-trends-and-positioning.md`](17-research-trends-and-positioning.md) | 研究趋势与定位 | Research |
| [`18-frontier-papers-online.md`](18-frontier-papers-online.md) | 前沿论文在线链接 | Research |

---

## 完整项目结构

```
Embodied-AI-Zero-to-Hero/
|-- docs/                              # 核心文档
|   |-- 01-what-is-vla.md              # VLA 核心概念详解
|   |-- 02-key-papers.md               # VLA 关键论文导读
|   |-- 03-learning-path.md            # VLA 完整学习路线
|   |-- 04-glossary.md                 # VLA 术语表（A-Z）
|   |-- 05-interview-prep.md           # 面试题汇总（100+ 题）
|   |-- 06-rl-fundamentals-for-vla.md  # RL 基础（VLA 视角）
|   |-- 07-world-models-for-vla.md     # 世界模型详解
|   |-- 13-vla-zero-to-one.md          # VLA 实战
|   |-- 14-rl-zero-to-one.md           # RL 实战
|   |-- 15-world-model-zero-to-one.md  # 世界模型实战
|   |-- 17-research-trends-and-positioning.md  # 研究趋势
|   |-- 18-frontier-papers-online.md   # 前沿论文在线链接
|   |-- 19-sim-to-real-guide.md        # Sim-to-Real 实战指南
|   |-- 20-vla-deployment-guide.md     # VLA 部署优化指南
|   |-- 21-vla-dataset-organization.md # VLA 数据组织
|   |-- 22-act-vs-diffusion-policy.md  # ACT vs Diffusion Policy
|   |-- 23-robot-foundation-models.md  # RFM 总览
|   |-- 24-action-representation-and-tokenization.md  # 动作表示
|   |-- 25-cross-embodiment-adaptation.md  # Cross-Embodiment
|   |-- 26-rfm-finetuning-and-evaluation.md  # RFM 微调与评测
|   |-- 27-embodied-reasoning-and-planning.md  # 具身推理与规划
|   |-- 28-smolvla-gpu-finetuning-runbook.md  # SmolVLA GPU 微调指南
|   |-- 29-learning-tracks-detail.md  # 四大研究方向详细分解
|   |-- README.md                      # 本文档：文档索引
|
|-- examples/                          # 可运行示例
|   |-- unified_pushcube_env.py        # PushCube 环境（双方块）
|   |-- unified_pushcube_vla.py        # VLA + 语言消融
|   |-- unified_pushcube_wm.py         # 世界模型
|   |-- unified_pushcube_rl.py         # REINFORCE
|   |-- unified_pushcube_act.py        # 动作分块策略
|   |-- unified_pushcube_diffusion.py  # Diffusion Policy
|   |-- vla_demo.py                    # VLA 推理演示
|   |-- minimal_vla.py                 # 最小 VLA 架构
|   |-- rl_demo.py                     # RL 演示
|   |-- world_model_demo.py            # 世界模型演示
|   |-- dreamer_rssm.py                # DreamerV3 RSSM
|   |-- world_model_vla_pipeline.py    # WM + Policy 融合
|   |-- train_diffusion_policy.py      # 可训练 Diffusion Policy
|   |-- robot_foundation_models/       # RFM 模块
|       |-- common/                    # 统一接口（RobotObservation, ActionChunk, Protocol）
|       |-- smolvla/                   # SmolVLA 适配器
|       |-- openvla/                   # OpenVLA 适配器
|       |-- octo/                      # Octo 适配器
|       |-- groot/                     # GR00T 适配器
|       |-- planners/                  # 规则 + VLM 任务分解
|
|-- benchmarks/                        # 基准测试
|   |-- robot_foundation_models/       # RFM 评测脚本
|
|-- tutorials/                         # 教程
|-- tests/                             # 测试
|   |-- test_imports.py                # 基础导入测试
|   |-- test_pushcube_regression.py    # PushCube 回归测试
|-- CONTRIBUTING.md                    # 贡献指南
|-- CHANGELOG.md                       # 版本变更记录
|-- requirements.txt                   # Pip 依赖文件
|-- LICENSE                            # MIT 许可证
|-- .gitignore                         # Git 忽略规则
```

---

## 核心概念速查

### VLA 核心概念

| 概念 | 一句话解释 |
|:-----|:-----------|
| **VLA** | 视觉-语言-动作模型：图像 + 语言指令 -> 机器人动作 |
| **VLM** | 视觉-语言模型：图像 + 文本 -> 文本 |
| **Action Chunking** | 一次预测未来多步动作序列，减少推理频率 |
| **Policy Head** | 将融合特征映射为动作输出的模型尾部 |
| **BC (Behavior Cloning)** | 监督学习：模仿专家演示数据 |
| **OXE (Open X-Embodiment)** | 最大开源机器人数据集 |
| **Sim-to-Real** | 仿真训练策略迁移到真实机器人 |
| **FK** | 已知关节角 -> 计算末端位置（正向） |
| **IK** | 已知末端位置 -> 求解关节角（逆向） |

### 世界模型核心概念

| 概念 | 一句话解释 |
|:-----|:-----------|
| **RSSM** | DreamerV3 核心：确定性 GRU + 随机潜状态 |
| **World Action Model** | 世界模型同时预测状态和动作 |
| **ECoT** | Embodied Chain-of-Thought，VLA 显式推理与自我纠错 |

### 2026 前沿概念

| 概念 | 一句话解释 | 代表论文 |
|:-----|:-----------|:---------|
| **Cross-Embodiment** | 跨机器人形态泛化，训练于 A 机器人迁移到 B 机器人 | Octo |
| **Pose Token** | 离散姿态 token 作为通用 3D 空间表示 | Pose-VLA |
| **3D Point Flow** | 跨本体的统一 3D 世界表示 | PointWorld |

---

## 代码速查

```bash
# === PushCube VLA（推荐首次运行）===
cd examples
python unified_pushcube_vla.py --smoke-test --no-ablation

# === PushCube 全部五条路线 ===
python unified_pushcube_env.py             # 环境自测 + 专家基线
python unified_pushcube_vla.py             # VLA + 三条件消融
python unified_pushcube_wm.py              # 世界模型，多步预测
python unified_pushcube_rl.py --algo ppo    # PPO（主 RL 基线），1000 回合训练
python unified_pushcube_act.py             # 动作分块策略 + 时间集成
python unified_pushcube_diffusion.py       # 扩散策略，action horizon

# === VLA 推理演示 ===
python examples/vla_demo.py --mode synthetic --task "pick up the apple"
python examples/minimal_vla.py

# === RL 强化学习 ===
python examples/rl_demo.py --mode demo     # numpy Q-learning（无需安装）

# === 世界模型 ===
python examples/world_model_demo.py --mode concept  # numpy 线性模型 + MPC
python examples/dreamer_rssm.py --epochs 25         # DreamerV3 RSSM
python examples/world_model_vla_pipeline.py         # WM + VLA 融合

# === Diffusion Policy 训练 ===
python examples/train_diffusion_policy.py --mode train --data synthetic --epochs 50

# === Robot Foundation Models ===
cd examples/robot_foundation_models/smolvla
python inference.py                                    # SmolVLA 适配器（mock 模式）
python train_lightweight_vla.py --epochs 100 --batch_size 64  # 训练轻量 VLA（CPU）
python evaluate.py --mode closed_loop \
    --checkpoint models/lightweight_vla/lightweight_vla_pushcube.pt \
    --n_episodes 20                                    # 真实 checkpoint 闭环评估
```

---

## 外部学习资源

### 教材与课程

| 资源 | 类型 | 说明 |
|:-----|:-----|:-----|
| [Modern Robotics (Lynch & Park)](http://hades.mech.northwestern.edu/index.php/Modern_Robotics) | 教材 | 刚体运动学、Jacobian、开链/闭链系统 |
| [MuJoCo Menagerie](https://github.com/google-deepmind/mujoco_menagerie) | 代码 | 预构建机器人模型库 |
| [Diffusion Policy 官方教程](https://diffusion-policy.cs.columbia.edu/) | 教程 | 扩散策略从原理到代码 |
| [Stanford CS224R](https://cs224r.stanford.edu/) | 课程 | Stanford 机器人学习课程 |
| [OpenAI Spinning Up](https://spinningup.openai.com/en/latest/) | 教程 | RL 最经典入门教程 |
| [UCB CS285 -- Deep RL](https://rail.eecs.berkeley.edu/deeprlcourse/) | 课程 | Berkeley 深度强化学习 |

### 相关项目

- [Embodied-AI-Paper-Analysis](https://github.com/Dld0621/Embodied-AI-Paper-Analysis) — 具身智能论文体系化梳理

---

## 完整学习路线 (Stage 0–10)

```
Stage 0: Foundations
  └─ Robot learning basics, FK/IK, coordinate frames, MuJoCo basics

Stage 1: VLA Basics
  └─ Minimal VLA structure → PushCube VLA → Action representation

Stage 2: VLA Research
  └─ SmolVLA / OpenVLA / Octo / Diffusion Policy → Fine-tuning → Deployment

Stage 3: World Models
  └─ Linear dynamics → RSSM → Integration with VLA/RL

Stage 4: RL Basics
  └─ Q-Learning → SAC → HER → PushCube RL training

Stage 5: RL Research
  └─ RL fine-tuning of VLA → Sim-to-Real → Real robot safety

Stage 6: Sim-to-Real
  └─ Domain randomization → System ID → Visual adaptation → Hardware validation

Stage 7: Robot Foundation Models
  └─ Unified interface → SmolVLA adapter → Cross-embodiment → Embodied reasoning

Stage 8: Integration
  └─ End-to-end pipeline: Perception → VLA → Robot Adapter → Controller → Safety

Stage 9: Evaluation
  └─ Offline metrics → Closed-loop success → Generalization → Language ablation

Stage 10: Frontier Research
  └─ 2026 trends: Cross-embodiment, world action models, embodied reasoning
```
