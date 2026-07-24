# Documentation Index

> 本文档是 `docs/` 目录的完整索引。README 中未展开的细节、命令速查、概念百科和外部资源均存放于此。

---

## 文档分类索引

### 基础概念 (Foundations)

| 文档 | 内容 | 标签 |
|:-----|:-----|:-----|
| [`00-joint-concepts.md`](00-joint-concepts.md) | Joint / Joint Angle / Ctrl 核心概念 | Concept |
| [`00-concepts-encyclopedia.md`](00-concepts-encyclopedia.md) | 全概念百科（A-Z） | Concept |
| [`01-what-is-ik-retargeting.md`](01-what-is-ik-retargeting.md) | Retargeting 核心概念 | Concept |
| [`01-what-is-vla.md`](01-what-is-vla.md) | VLA 核心概念详解 | Concept |
| [`04-glossary.md`](04-glossary.md) | VLA 术语表（A-Z） | Concept |

### 重定向 (Retargeting)

| 文档 | 内容 | 标签 |
|:-----|:-----|:-----|
| [`02-retargeting-taxonomy.md`](02-retargeting-taxonomy.md) | 方法分类体系 | Concept |
| [`03-human-hand-to-robot-hand.md`](03-human-hand-to-robot-hand.md) | 人手→机器人手映射 | Concept |
| [`04-optimization-methods.md`](04-optimization-methods.md) | 优化方法深入 | Tutorial |
| [`05-learning-based-methods.md`](05-learning-based-methods.md) | 基于学习的方法 | Paper-inspired |
| [`06-evaluation-metrics.md`](06-evaluation-metrics.md) | 评估指标与基准 | Benchmark |
| [`07-key-papers.md`](07-key-papers.md) | 37 篇重定向关键论文导读 | Paper |
| [`08-open-source-projects.md`](08-open-source-projects.md) | 开源项目复现指南 | Engineering |
| [`09-dexterous-hands-analysis.md`](09-dexterous-hands-analysis.md) | 灵巧手对比分析 | Concept |
| [`10-manipulation-datasets.md`](10-manipulation-datasets.md) | 灵巧操作数据集 | Resource |
| [`11-dexmv-research-guide.md`](11-dexmv-research-guide.md) | DexMV 高精度 IK 研究指南 | Reproduction |
| [`12-freshman-zero-to-one.md`](12-freshman-zero-to-one.md) | 大一新生 0→1 实战 | Tutorial |
| [`16-arxiv-retargeting-scan.md`](16-arxiv-retargeting-scan.md) | 80+ 篇 ArXiv 重定向论文扫描 | Research |

### VLA (Vision-Language-Action)

| 文档 | 内容 | 标签 |
|:-----|:-----|:-----|
| [`02-key-papers.md`](02-key-papers.md) | VLA 关键论文导读 | Paper |
| [`03-learning-path.md`](03-learning-path.md) | VLA 完整学习路线 | Tutorial |
| [`05-interview-prep.md`](05-interview-prep.md) | 面试题汇总（100+ 题） | Resource |
| [`13-vla-zero-to-one.md`](13-vla-zero-to-one.md) | VLA 实战（SmolVLA） | Tutorial |
| [`20-vla-deployment-guide.md`](20-vla-deployment-guide.md) | VLA 部署优化与边缘计算 | Engineering |

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
|-- docs/                              # 29 本核心文档
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
|   |-- 05-interview-prep.md           # 面试题汇总（100+ 题）
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
|   |-- 19-sim-to-real-guide.md        # Sim-to-Real 实战指南
|   |-- 20-vla-deployment-guide.md     # VLA 部署优化指南
|   |-- README.md                      # 本文档：文档索引
|
|-- examples/                          # 19+ 个可运行示例
|   |-- freshman_zero_to_one.py        # 人手仿真 + DexMV 重定向
|   |-- train_diffusion_policy.py      # 可训练 Diffusion Policy
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
|-- CONTRIBUTING.md                    # 贡献指南
|-- CHANGELOG.md                       # 版本变更记录
|-- requirements.txt                   # Pip 依赖文件
|-- LICENSE                            # MIT 许可证
|-- .gitignore                         # Git 忽略规则
```

---

## 核心概念速查

### 基础概念

| 概念 | 一句话解释 |
|:-----|:-----------|
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
|:-----|:-----------|
| **VLA** | 视觉-语言-动作模型：图像 + 语言指令 -> 机器人动作 |
| **VLM** | 视觉-语言模型：图像 + 文本 -> 文本 |
| **Action Chunking** | 一次预测未来多步动作序列，减少推理频率 |
| **Policy Head** | 将融合特征映射为动作输出的模型尾部 |
| **BC (Behavior Cloning)** | 监督学习：模仿专家演示数据 |
| **OXE (Open X-Embodiment)** | 最大开源机器人数据集 |
| **Sim-to-Real** | 仿真训练策略迁移到真实机器人 |

### 2026 前沿概念

| 概念 | 一句话解释 | 代表论文 |
|:-----|:-----------|:---------|
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
| **ECoT** | Embodied Chain-of-Thought，VLA 显式推理与自我纠错 | ZR-0 |
| **Pose Token** | 离散姿态 token 作为通用 3D 空间表示 | Pose-VLA |

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

# === Diffusion Policy 训练 ===
python examples/train_diffusion_policy.py --mode train --data synthetic --epochs 50
python examples/train_diffusion_policy.py --mode visualize --checkpoint ./checkpoints/dp_best.pt

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
