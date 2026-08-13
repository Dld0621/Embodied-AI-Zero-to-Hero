# Documentation Index

> 本文档是 `docs/` 目录的完整索引。README 中未展开的细节、命令速查、概念百科和外部资源均存放于此。

> English readers: start from the [documentation home](index.md) and [14-lesson English contract](foundations/README_EN.md). All readers should use the [validation policy](VALIDATION.md) and [primary-source registry](SOURCES.md) when interpreting claims.

Quality and release references: [repository validation](VALIDATION.md) · [primary sources](SOURCES.md) · [release checklist](RELEASE_CHECKLIST.md) · [third-party notices](../THIRD_PARTY_NOTICES.md)

---

## 文档分类索引

### Research Routes（科研路线）

> 七条双语路线把研究问题映射到前置课程、Pipeline、交付物、指标、晋级门槛与证据边界。中文见 [`learning-paths/README_CN.md`](learning-paths/README_CN.md)，English 见 [`learning-paths/README.md`](learning-paths/README.md)。

### Foundations Layer（基础课程）

> 面向零基础读者的预备课程，覆盖 Python、线性代数、深度学习、机器人学和仿真基础。详见 [`foundations/00-roadmap.md`](foundations/00-roadmap.md)。

| 文档 | 内容 | 标签 |
|:-----|:-----|:-----|
| [`foundations/00-roadmap.md`](foundations/00-roadmap.md) | 基础课程路线图与学习路径 | Guide |
| [`foundations/01-python-for-robotics.md`](foundations/01-python-for-robotics.md) | Python for Robotics（NumPy / Matplotlib） | Tutorial |
| [`foundations/02-linear-algebra.md`](foundations/02-linear-algebra.md) | 线性代数（向量 / 矩阵 / 特征值 / 概率） | Tutorial |
| [`foundations/03-deep-learning-basics.md`](foundations/03-deep-learning-basics.md) | 深度学习基础（神经网络 / 反向传播 / 优化器） | Tutorial |
| [`foundations/04-transformer-basics.md`](foundations/04-transformer-basics.md) | Transformer 基础（Attention / Self-Attention / ViT） | Tutorial |
| [`foundations/05-coordinate-transform.md`](foundations/05-coordinate-transform.md) | 坐标变换（齐次坐标 / 变换复合） | Tutorial |
| [`foundations/06-se3-and-rotation.md`](foundations/06-se3-and-rotation.md) | SO(3) & SE(3)（旋转表示 / 四元数 / 万向锁） | Tutorial |
| [`foundations/07-fk-jacobian-ik.md`](foundations/07-fk-jacobian-ik.md) | FK / Jacobian / IK（正运动学 / 逆运动学） | Tutorial |
| [`foundations/08-control-basics.md`](foundations/08-control-basics.md) | 控制基础（PID / 阻抗控制 / 安全滤波） | Tutorial |
| [`foundations/09-mujoco-basics.md`](foundations/09-mujoco-basics.md) | MuJoCo 基础（MJCF / 仿真循环 / 渲染） | Tutorial |
| [`foundations/10-dataset-and-training.md`](foundations/10-dataset-and-training.md) | 数据集与训练（采集 / 格式 / 训练循环 / 评估） | Tutorial |
| [`foundations/11-probability-and-optimization.md`](foundations/11-probability-and-optimization.md) | 概率、统计与优化（不确定性 / MLE / 梯度 / 数值稳定） | Tutorial |
| [`foundations/12-perception-and-sensors.md`](foundations/12-perception-and-sensors.md) | 感知与传感器（相机 / 深度 / 状态 / 力觉 / 同步） | Tutorial |
| [`foundations/13-robot-systems-and-safety.md`](foundations/13-robot-systems-and-safety.md) | 机器人系统与安全（控制栈 / 接口 / 实时性 / 门禁） | Tutorial |
| [`foundations/14-evaluation-and-reproducibility.md`](foundations/14-evaluation-and-reproducibility.md) | 评估与复现（指标 / 基线 / 统计 / 证据等级） | Tutorial |

### End-to-End Pipelines（端到端工程闭环）

> 十一条路线统一描述前置知识、输入、阶段、产物、指标、验收门槛与失败模式。中文总览见 [`pipelines/README_CN.md`](pipelines/README_CN.md)，英文总览见 [`pipelines/README.md`](pipelines/README.md)。

| 文档 | 方向 | 当前证据 |
|:-----|:-----|:--------|
| [`pipelines/01-simulation-data.md`](pipelines/01-simulation-data.md) | 仿真与数据生成 | Smoke-tested |
| [`pipelines/02-vla-policy.md`](pipelines/02-vla-policy.md) | VLA 策略 | Smoke-tested teaching baseline |
| [`pipelines/03-world-model-planning.md`](pipelines/03-world-model-planning.md) | 世界模型与规划 | Model smoke-tested |
| [`pipelines/04-rl-post-training.md`](pipelines/04-rl-post-training.md) | RL 与后训练 | Smoke-tested teaching baseline |
| [`pipelines/05-rfm-cross-embodiment.md`](pipelines/05-rfm-cross-embodiment.md) | RFM 与跨本体适配 | Interface-tested |
| [`pipelines/06-embodied-reasoning.md`](pipelines/06-embodied-reasoning.md) | 具身推理与任务规划 | Interface-tested |
| [`pipelines/07-sim-to-real.md`](pipelines/07-sim-to-real.md) | Sim-to-Real 部署 | Documented, hardware-dependent |
| [`pipelines/08-dexterous-retargeting.md`](pipelines/08-dexterous-retargeting.md) | 灵巧手重定向 | Synthetic smoke-tested |
| [`pipelines/09-perception-state-estimation.md`](pipelines/09-perception-state-estimation.md) | 感知与状态估计 | Synthetic smoke-tested |
| [`pipelines/10-navigation-locomotion.md`](pipelines/10-navigation-locomotion.md) | 导航与运动控制 | Grid-navigation smoke-tested |
| [`pipelines/11-dexterous-manipulation.md`](pipelines/11-dexterous-manipulation.md) | 灵巧抓取与精细操作 | Abstract contact-dynamics smoke-tested |

### 基础概念 (Core Concepts)

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
| [`benchmark_report.md`](benchmark_report.md) | 论文式实验报告（Setup / Results / Failure Analysis / Discussion） | Research |

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
|   |-- foundations/                   # 基础课程（Zero → Hero 预备层）
|   |   |-- 00-roadmap.md              # 基础课程路线图
|   |   |-- 01-python-for-robotics.md  # Python for Robotics
|   |   |-- 02-linear-algebra.md       # 线性代数
|   |   |-- 03-deep-learning-basics.md # 深度学习基础
|   |   |-- 04-transformer-basics.md   # Transformer 基础
|   |   |-- 05-coordinate-transform.md # 坐标变换
|   |   |-- 06-se3-and-rotation.md     # SO(3) & SE(3)
|   |   |-- 07-fk-jacobian-ik.md       # FK / Jacobian / IK
|   |   |-- 08-control-basics.md       # 控制基础
|   |   |-- 09-mujoco-basics.md        # MuJoCo 基础
|   |   |-- 10-dataset-and-training.md # 数据集与训练
|   |   |-- 11-probability-and-optimization.md # 概率、统计与优化
|   |   |-- 12-perception-and-sensors.md # 感知与传感器
|   |   |-- 13-robot-systems-and-safety.md # 机器人系统与安全
|   |   |-- 14-evaluation-and-reproducibility.md # 评估与复现
|   |-- pipelines/                     # 十一条带证据标签的工程 Pipeline
|   |   |-- README_CN.md               # 中文总览与统一命令
|   |   |-- README.md                  # English catalog
|   |   |-- 01...11-*.md               # 分方向输入、阶段、产物与门禁
|   |-- learning-paths/                # 七方向双语科研路线
|   |   |-- README.md                  # English research routes
|   |   |-- README_CN.md               # 中文科研路线
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
Stage 0: Foundations Layer（基础课程）
  └─ docs/foundations/00-roadmap.md → 01 Python → 02 线性代数 → 03 深度学习
     → 04 Transformer → 05 坐标变换 → 06 SE(3) → 07 FK/IK → 08 控制
     → 09 MuJoCo → 10 数据集与训练 → 11 概率与优化 → 12 感知与传感器
     → 13 机器人系统与安全 → 14 评估与复现
     （完整路线约 44–68 小时，也可按 Pipeline 前置要求选学）

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
  └─ docs/pipelines/README_CN.md → 选择十一条闭环之一 → smoke test / documented gate → benchmark

Stage 9: Evaluation
  └─ Offline metrics → Closed-loop success → Generalization → Language ablation

Stage 10: Frontier Research
  └─ 2026 trends: Cross-embodiment, world action models, embodied reasoning
```
