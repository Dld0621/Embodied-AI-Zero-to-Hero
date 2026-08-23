<h1 align="center">具身智能 · 从入门到实践</h1>

<p align="center">
  <a href="README.md">English</a> · <b>简体中文</b>
</p>

<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="assets/dof-hero-cn-dark.svg">
    <img src="assets/dof-hero-cn.svg" alt="理解具身智能闭环，构建可信证据" width="100%">
  </picture>
</p>

<p align="center">
  <a href="#start"><b>开始</b></a>&nbsp;&nbsp;·&nbsp;&nbsp;
  <a href="#knowledge"><b>知识</b></a>&nbsp;&nbsp;·&nbsp;&nbsp;
  <a href="#system"><b>系统</b></a>&nbsp;&nbsp;·&nbsp;&nbsp;
  <a href="#pipelines"><b>管线</b></a>&nbsp;&nbsp;·&nbsp;&nbsp;
  <a href="#routes"><b>科研</b></a>&nbsp;&nbsp;·&nbsp;&nbsp;
  <a href="#evidence"><b>证据</b></a>&nbsp;&nbsp;·&nbsp;&nbsp;
  <a href="#docs"><b>文档</b></a>
</p>

<p align="center">
  <a href="https://github.com/Dld0621/Embodied-AI-Zero-to-Hero/actions/workflows/tests.yml"><img src="https://img.shields.io/github/actions/workflow/status/Dld0621/Embodied-AI-Zero-to-Hero/tests.yml?branch=master&style=flat&label=build" alt="构建状态"></a>
  <a href="https://github.com/Dld0621/Embodied-AI-Zero-to-Hero"><img src="https://img.shields.io/github/stars/Dld0621/Embodied-AI-Zero-to-Hero?style=flat&label=stars" alt="GitHub Stars"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/original%20content-MIT-4F7CFF?style=flat" alt="项目原创内容采用 MIT 许可证"></a>
  <a href="THIRD_PARTY_NOTICES.md"><img src="https://img.shields.io/badge/third--party%20assets-mixed%20licenses-6B7280?style=flat" alt="第三方资产采用不同许可证"></a>
</p>

<p align="center">
  <b>面向具身智能学习与科研的双语、证据优先系统。</b><br>
  <sub>理解前置知识，运行系统闭环，度量实验结果，遵守部署边界。</sub>
</p>

| **45** 个知识节点 | **9** 大领域 | **14** 章基础课程 | **11** 条工程管线 | **7** 条科研路线 |
|:---:|:---:|:---:|:---:|:---:|
| 前置依赖图 | 能力全景图 | 从概念到练习 | 从数据到部署 | 从问题到证据 |

> [!IMPORTANT]
> 脚本运行结束，只能证明执行链路接通，不能证明任务性能达到可用水平。本仓库明确区分**接口检查**、**合成 Smoke Test**、**教学基准**与**真机验证**。

<a id="start"></a>
## 选择你的入口

| 学习 | 配环境 | 构建 | 科研 |
|:---|:---|:---|:---|
| 解析前置依赖与验收目标。 | 搭建经过审阅的机器人开发工作站。 | 运行一条闭环工程路线。 | 把研究问题转化为实验契约。 |
| [知识体系 →](docs/knowledge-system/README_CN.md) | [环境指南 →](docs/setup/README_CN.md) | [Pipeline 目录 →](docs/pipelines/README_CN.md) | [科研路线 →](docs/learning-paths/README_CN.md) |

### 一分钟首次运行

```bash
git clone https://github.com/Dld0621/Embodied-AI-Zero-to-Hero.git
cd Embodied-AI-Zero-to-Hero
pip install numpy
python scripts/run_pipeline.py --run simulation-data
```

### 按目标导航

```bash
# 已经知道想学习的能力。
python scripts/run_knowledge_map.py --path-to task-dexterity-teleoperation --lang zh

# 已经知道想构建的系统。
python scripts/run_pipeline.py --list
python scripts/run_pipeline.py --show dexterous-manipulation

# 已经知道想研究的问题。
python scripts/run_learning_path.py --list --lang zh
python scripts/run_learning_path.py --show dexterity-teleoperation --lang zh
```

<a id="knowledge"></a>
## 连接知识点

<p align="center">
  <img src="docs/assets/knowledge-system-cn.svg" alt="包含 45 个节点、9 个领域与 6 个阶段的具身智能知识体系" width="100%">
</p>

[双语知识体系](docs/knowledge-system/README_CN.md)是仓库在前置依赖粒度上的单一事实源。每个节点都声明学习结果、验收方式、课程、Pipeline 映射与学习证据类型。

| L0 · 工具 | L1 · 数学 | L2 · 机器人闭环 | L3 · 数据与学习 | L4 · 任务 | L5 · 证据 |
|:---|:---|:---|:---|:---|:---|
| 运行并记录 | 推导并验证 | 感知、估计、控制 | 数据、策略、预测 | 组合并恢复 | 比较并控制风险 |

知识图谱连接 14 章课程、11 条 Pipeline 和 7 条科研路线，帮助学习者从缺失前置走向可度量产物，无需猜测中间步骤。

<a id="system"></a>
## 一个闭环系统

<p align="center">
  <img src="assets/system_architecture-cn.svg" alt="从观测到可评估行动的闭环具身智能系统" width="100%">
</p>

| 层级 | 核心问题 | 输出 |
|:---|:---|:---|
| **感知与状态** | 世界和机器人当前发生了什么？ | 带不确定性的同步观测 |
| **推理、策略与预测** | 下一目标、动作与后果是什么？ | 计划、动作块与预测风险 |
| **控制与安全** | 如何在约束内执行动作？ | 发往仿真器或机器人的有界命令 |
| **评测与学习** | 是否成功、泛化并保持安全？ | 证据、诊断与更新后的策略 |

<a id="pipelines"></a>
## 十一条工程管线

每份契约都定义前置知识、输入、阶段、产物、指标、晋级门槛和失败模式；状态标签只描述本仓库已有证据。

| 系统方向 | 闭环 | 仓库证据 |
|:---|:---|:---|
| [仿真与数据](docs/pipelines/01-simulation-data.md) | 任务 → 仿真器 → 专家 → 轨迹 → 质检 | 已有 Smoke Test |
| [VLA 策略](docs/pipelines/02-vla-policy.md) | 图像 + 语言 + 状态 → 策略 → 评测 | 教学基线可 Smoke Test |
| [世界模型](docs/pipelines/03-world-model-planning.md) | 转移数据 → 动力学 → Rollout → 规划 | 模型可 Smoke Test |
| [RL 后训练](docs/pipelines/04-rl-post-training.md) | MDP → 奖励 → PPO → 回归检查 | 教学基线可 Smoke Test |
| [机器人基础模型](docs/pipelines/05-rfm-cross-embodiment.md) | 标准观测 → 适配器 → 动作 → 安全 | 接口已验证 |
| [具身推理](docs/pipelines/06-embodied-reasoning.md) | 指令 → 计划 → 技能 → 反馈 → 重规划 | 接口已验证 |
| [Sim-to-Real](docs/pipelines/07-sim-to-real.md) | 鲁棒性 → HIL → 影子模式 → 受控上线 | 已文档化；依赖硬件 |
| [灵巧手重定向](docs/pipelines/08-dexterous-retargeting.md) | 关键点 → 几何 → 优化 → 平滑 | 合成输入可 Smoke Test |
| [感知与状态估计](docs/pipelines/09-perception-state-estimation.md) | 标定 → 同步 → 融合 → 不确定性 | 合成数据可 Smoke Test |
| [导航与运动控制](docs/pipelines/10-navigation-locomotion.md) | 状态 → 地图/地形 → 规划 → 控制 → 恢复 | 栅格导航可 Smoke Test |
| [灵巧抓取与精细操作](docs/pipelines/11-dexterous-manipulation.md) | 状态 → 预抓取 → 接触 → 抬升 → 保持/恢复 | 抽象接触动力学可 Smoke Test |

```bash
python scripts/run_pipeline.py --validate
python scripts/run_pipeline.py --run vla-policy --dry-run
python scripts/run_pipeline.py --run world-model-planning
```

机器可读契约位于 [`pipelines/manifest.json`](pipelines/manifest.json)。合成路径只验证连通性与限定行为，不代表复现了大规模或真实世界结果。

<a id="routes"></a>
## 七条科研路线

[路线地图](docs/learning-paths/README_CN.md)把每个方向拆成研究问题、前置集合、Pipeline 顺序、交付物、指标、晋级门槛与证据边界。

| 科研方向 | Pipeline 顺序 | 必须交付 |
|:---|:---|:---|
| [基础模型与 VLA](docs/learning-paths/README_CN.md#foundation-models-vla) | 数据 → VLA → RFM | 策略 + 适配器 + 消融 |
| [操作与模仿学习](docs/learning-paths/README_CN.md#manipulation-imitation) | 数据 → VLA → RL | 闭环基线 + 失败分类 |
| [灵巧操作与遥操作](docs/learning-paths/README_CN.md#dexterity-teleoperation) | 重定向 → 状态 → 抓取 → Sim-to-Real | 运动 + 接触/任务证据 |
| [导航与具身智能体](docs/learning-paths/README_CN.md#navigation-embodied-agents) | 状态 → 导航 → 推理 | 智能体闭环 + 恢复报告 |
| [人形与运动控制](docs/learning-paths/README_CN.md#humanoids-locomotion) | 运动 → RL → Sim-to-Real | 运动协议 + 安全门槛 |
| [感知与世界模型](docs/learning-paths/README_CN.md#perception-world-models) | 状态 → 世界模型 | 不确定状态 + 预测 Rollout |
| [仿真、数据与评测](docs/learning-paths/README_CN.md#simulation-data-evaluation) | 数据 → 世界模型 → Sim-to-Real | 数据说明 + 基准 + 晋级决策 |

| 01 · 前置知识 | 02 · 基础课程 | 03 · 工程 Pipeline | 04 · 科研 |
|:---:|:---:|:---:|:---:|
| 定位缺失节点 | 学习并验证 | 生成产物与指标 | 复现、消融、比较 |
| [45 节点图谱](docs/knowledge-system/README_CN.md) | [14 章课程路线](docs/foundations/00-roadmap.md) | [11 份契约](docs/pipelines/README_CN.md) | [7 条路线](docs/learning-paths/README_CN.md) |

<a id="evidence"></a>
## 先看证据，再下结论

| 标签 | 能证明什么 | 不能证明什么 |
|:---|:---|:---|
| **Smoke-tested** | 最小路径可以运行结束。 | 方法达到可用任务指标。 |
| **Interface-tested** | 协议、形状和适配器已经接通。 | 真实权重或硬件已经验证。 |
| **Benchmark** | 固定协议产生了可记录结果。 | 结果可以迁移到其他设置。 |
| **Hardware-dependent** | 门禁依赖指定机器人和安全流程。 | 仿真通过即可授权真机。 |

<details>
<summary><b>展开教学规模 PushCube 基准快照</b></summary>

所有方法共享一个双方块语言条件任务，但预算和评估次数不同。这是教学基准，不是严格受控的排行榜。

| 方法 | 输入 | 数据 / 计算 | 评估回合 | 成功率 |
|:---|:---|:---|---:|---:|
| 专家策略 | 状态 | 启发式 / CPU | 50 | **~100%** |
| State-BC | 14 维状态 | 100 回合 / CPU | 100 | **90%** |
| RL，BC 初始化 PPO | 14 维状态 | 500 回合 / CPU | 20 | **15%** |
| VLA | RGB + 语言 | 100 回合 / CPU | 100 | **0%** |
| WM-MPC | 14 维状态 | 100 回合 / CPU | 20 | **0%** |
| SmolVLA 450M | RGB + 语言 + 状态 | 50 回合、10K 步 / GPU | 20 | **0%** |
| Action Chunking / Diffusion | RGB + 语言 | 100 回合 / CPU | — | **N/A** |

结构化状态基线说明这个任务可学习。视觉策略的差距是负结果，它提示需要诊断数据、表示与闭环分布偏移。原始产物和范围说明见 [`BENCHMARK.md`](BENCHMARK.md) 与[基准报告](docs/benchmark_report.md)。

</details>

<details>
<summary><b>展开统一任务契约与平台边界</b></summary>

PushCube 固定 128×128 RGB + 语言观测、可选 14 维状态、二维末端增量动作与正确方块任务指标。教学任务保持不变，仅替换策略。

仓库包含已验证的本地 PushCube 环境、实验性的 Franka 与 AgiBot 适配器，以及规划中的 UR5e 与 Unitree 路径。本仓库**不声称**已在这些外部平台上本地复现真机性能。

核心入口：[`环境`](examples/unified_pushcube_env.py) · [`VLA`](examples/unified_pushcube_vla.py) · [`世界模型`](examples/unified_pushcube_wm.py) · [`RL`](examples/unified_pushcube_rl.py)

</details>

<a id="docs"></a>
## 文档导航

| 从这里开始 | 继续深入 | 用这些内容验证 |
|:---|:---|:---|
| [在线文档](https://dld0621.github.io/Embodied-AI-Zero-to-Hero/) | [领域地图](docs/field-map-cn.md) | [证据规范](docs/VALIDATION.md) |
| [知识体系](docs/knowledge-system/README_CN.md) | [基础课程](docs/foundations/00-roadmap.md) | [真实性审查](docs/CLAIM_REVIEW.md) |
| [环境配置](docs/setup/README_CN.md) | [MuJoCo 场景搭建](docs/tutorials/mujoco-scene-building.md) | [权威来源](docs/SOURCES.md) |
| [Pipeline 目录](docs/pipelines/README_CN.md) | [机器人基础模型](docs/23-robot-foundation-models.md) | [基准协议](BENCHMARK.md) |
| [科研路线](docs/learning-paths/README_CN.md) | [前沿论文指南](docs/18-frontier-papers-online.md) | [安全政策](SECURITY.md) |

<details>
<summary><b>展开仓库结构</b></summary>

```text
Embodied-AI-Zero-to-Hero/
├─ docs/                  课程、指南、科研路线与验证规范
├─ knowledge/             机器可读前置知识图谱
├─ pipelines/             机器可读工程契约
├─ learning_paths/        机器可读科研路线契约
├─ examples/              可运行教学与研究基线
├─ benchmarks/ + results/ 评估入口与已记录产物
├─ tools/robotdev/        只读工作站检查与技术栈选择器
├─ scripts/ + tests/      发现、校验、持续集成与回归测试
└─ assets/                双语界面与原理图
```

</details>

### 复现仓库质量检查

```bash
python scripts/check_markdown_links.py
python scripts/check_markdown_format.py
python scripts/check_claims.py
python scripts/run_knowledge_map.py --validate
python scripts/run_pipeline.py --validate
python scripts/run_learning_path.py --validate
python scripts/audit_repository.py
python -m pytest tests/ -q
```

证据链为：导入 → Smoke → 确定性测试 → 基准 → 真机验证。每项结果都应保留命令、随机种子、commit、数据版本、checkpoint、硬件、回合数和机器可读指标。

## 参与贡献

提交教程、Pipeline、基准结论或机器人适配器前，请阅读 [`CONTRIBUTING.md`](CONTRIBUTING.md)。可复现基线、失败案例、双语改进、新本体与有证据支持的纠错尤其有价值。

项目原创内容采用 [MIT License](LICENSE)。仓库中的上游代码、模型和资产保留各自许可条款，复用前请阅读[第三方声明](THIRD_PARTY_NOTICES.md)。引用信息位于 [`CITATION.cff`](CITATION.cff)。

<p align="center">
  <b>理解闭环，构建证据。</b><br>
  维护者：<a href="https://github.com/Dld0621">Gangwei Li</a>
</p>
