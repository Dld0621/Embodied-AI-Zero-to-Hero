# 具身智能：从入门到实践

**[English](README.md) · 简体中文**

一套面向具身智能学习、构建与评估的双语证据化课程体系：从数学前置知识，一直到闭环机器人系统。

[![测试](https://img.shields.io/github/actions/workflow/status/Dld0621/Embodied-AI-Zero-to-Hero/tests.yml?branch=master&label=tests)](https://github.com/Dld0621/Embodied-AI-Zero-to-Hero/actions/workflows/tests.yml)
[![文档](https://img.shields.io/badge/docs-online-2563eb)](https://dld0621.github.io/Embodied-AI-Zero-to-Hero/)
[![知识体系](https://img.shields.io/badge/knowledge_nodes-45-111827)](docs/knowledge-system/README_CN.md)
[![工程管线](https://img.shields.io/badge/engineering_pipelines-11-111827)](docs/pipelines/README_CN.md)
[![许可](https://img.shields.io/badge/original%20content-MIT-64748b)](LICENSE)
[![第三方资产](https://img.shields.io/badge/third--party%20assets-mixed%20licenses-94a3b8)](THIRD_PARTY_NOTICES.md)

> [!IMPORTANT]
> 代码能够运行，只能证明执行链路接通，不能证明任务性能达到可用水平。本仓库明确区分接口检查、合成 Smoke Test、教学基准与真机验证。

## 你可以从这个仓库获得什么

**[打开交互实验室](https://dld0621.github.io/Embodied-AI-Zero-to-Hero/learning-lab-cn/)**：不用安装机器人环境，就能调节坐标、关节、控制增益、动作延迟与评估样本数，观察图像变化，并导出自己的预测和解释。配套[手算推导与迁移练习](docs/learning-lab-cn.md)在 GitHub 内也可阅读；可操作界面运行在文档站。

| 学习 | 构建 | 评估 |
|---|---|---|
| 9 个领域、45 个带前置依赖的知识节点 | 11 条包含输入、阶段、产物与失败模式的工程 Pipeline | 明确的指标、晋级门禁、来源和证据边界 |
| 从 Python、SE(3) 到安全与可复现性的 14 章基础课程 | 覆盖仿真、VLA、世界模型、RL、感知、导航与灵巧操作的教学路径 | 统一 PushCube 协议和仓库级真实性、回归检查 |

这是一个结构化学习与工程系统，不代表其中每个基线都达到 SOTA、生产可用或已经过真机验证。

<a id="start"></a>
## 从你的目标开始

| 你的目标 | 第一站 | 需要产出的证据 |
|---|---|---|
| 完全小白，想知道第一小时做什么 | [从这里开始](docs/start-here-cn.md) | 第一张实验卡、故障记录和个人路线 |
| 从第一性原理学习具身智能 | [细粒度课程合同](docs/curriculum_cn.md) | 完成练习、推导和前置依赖路径 |
| 配置机器人开发工作站 | [环境配置](docs/setup/README_CN.md) | 带版本的环境回执和分层 Smoke Check |
| 构建一个完整机器人学习系统 | [Pipeline 目录](docs/pipelines/README_CN.md) | 输入、产物、指标和分阶段失败报告 |
| 深入学习 VLA 或 WAM | [VLA 与 WAM 专项](docs/specializations/README_CN.md) | 算法族选择、匹配基线、消融矩阵与闭环证据 |
| 进入一个科研方向 | [七条科研路线](docs/learning-paths/README_CN.md) | 问题、基线、消融计划、晋级门禁和证据边界 |
| 定位一个缺失的前置知识 | [知识体系](docs/knowledge-system/README_CN.md) | 按依赖排序的学习路径和验收目标 |

### 运行一条教学 Pipeline

```bash
git clone https://github.com/Dld0621/Embodied-AI-Zero-to-Hero.git
cd Embodied-AI-Zero-to-Hero
python -m pip install numpy
python scripts/run_pipeline.py --list
python scripts/run_pipeline.py --run simulation-data
```

已经知道目标时，可以直接查询机器可读地图：

```bash
python scripts/run_knowledge_map.py --path-to task-dexterity-teleoperation --lang zh
python scripts/run_pipeline.py --show dexterous-manipulation
python scripts/run_learning_path.py --show dexterity-teleoperation --lang zh
```

生成从小白到专家的证据化个人路线：

```bash
python scripts/run_curriculum.py --diagnose --lang zh
python scripts/run_curriculum.py --plan full-stack-expert --hours-per-week 8 --lang zh
```

课程进度由产物与评审门禁决定，而不是投入时长。完整评分、三级 Capstone 和本轮 85→100 质量合同见[统一评估](docs/assessment-cn.md)、[毕业项目](docs/capstone-cn.md)与[课程审查](docs/CURRICULUM_AUDIT_CN.md)。这里的 100 分只表示仓库课程质量合同全部实现并通过结构检查，不是通用专家认证或真机性能结论。

<a id="knowledge"></a>
## 先建立知识，再执行配方

[知识体系](docs/knowledge-system/README_CN.md)是前置依赖粒度的单一事实源。每个节点都声明学习结果、前置依赖、验收方式、主文档、Pipeline 映射和学习证据类型。

| 阶段 | 核心能力 | 退出证据 |
|---|---|---|
| **L0 · 计算与工具** | 运行、配置、检查和记录实验 | 可复现命令、配置与环境回执 |
| **L1 · 数学语言** | 处理坐标系、不确定性、目标函数和数值边界 | 推导和数值验证 |
| **L2 · 机器人闭环** | 建模、感知、估计、控制和仿真 | 带单位、频率、限位和故障诊断的闭环轨迹 |
| **L3 · 学习与预测** | 构建数据集、策略、世界模型与规划器 | 带数据泄漏和不确定性检查的训练、评估产物 |
| **L4 · 任务系统** | 组合操作、灵巧手、导航和运动控制 | 任务协议、分阶段失败与恢复行为 |
| **L5 · 证据与部署** | 比较、泛化并决定是否可以提高风险 | 可复现实验报告，以及明确的晋级或停止决定 |

![六阶段知识依赖图](docs/assets/knowledge-system-cn.svg)

[细粒度课程合同](docs/curriculum_cn.md)把六个阶段进一步拆分为学习者、工程者和科研者三条路线及明确检查点。机器可读事实源位于 [`knowledge/manifest.json`](knowledge/manifest.json)。

## 从零进入 VLA 与 WAM 科研

[VLA 与 WAM 专项](docs/specializations/README_CN.md)把两个方向拆成独立、带前置依赖的学习路线，覆盖数据与动作合同、多模态融合、离散与连续动作生成、扩散与流式目标、世界模型规划基线、视频—动作联合模型、算法选型、消融和闭环评估。

| 专项 | 起点 | 进入下一阶段的条件 |
|---|---|---|
| [VLA 从零到一](docs/specializations/vla-zero-to-one-cn.md) | 动作块行为克隆，再加入语言条件 | 匹配预算的策略基线可复现，语言/视觉消融通过 |
| [WAM 从零到一](docs/specializations/wam-zero-to-one-cn.md) | 动作条件动力学 + MPC | 先通过 Rollout 与规划基线，再扩展视频—动作联合模型 |

可以用可解释选型器，根据实际目标、数据、算力和延迟约束比较算法族：

```bash
python scripts/select_vla_wam_algorithm.py --goal language-generalization --compute single-gpu --data task-specific --latency hard
```

选型器服务于学习和实验设计，不是模型排行榜，也不构成部署保证。

<a id="system"></a>
## 系统必须形成闭环

![从观测到可评估行动的具身智能闭环](assets/system_architecture-cn.svg)

| 层级 | 核心问题 | 必须产出的结果 |
|---|---|---|
| 感知与状态 | 世界和机器人正在发生什么？ | 时间对齐的观测、标定坐标系和不确定性 |
| 策略、推理与预测 | 下一步应该做什么，之后可能发生什么？ | 目标、计划、动作表示和风险预测 |
| 控制与安全 | 如何在物理和运行边界内执行动作？ | 有界命令、Watchdog、停止路径与日志 |
| 评估与学习 | 任务是否成功、能够泛化并保持安全？ | 指标、失败分类、方法比较和更新后的策略 |

<a id="pipelines"></a>
## 十一条工程 Pipeline

每条 Pipeline 都声明前置依赖、输入、阶段、产物、指标、晋级门禁和失败模式。状态只描述仓库内已有证据。

| Pipeline | 闭环 | 当前仓库证据 |
|---|---|---|
| [仿真与数据](docs/pipelines/01-simulation-data.md) | 任务 → 仿真器 → 专家 → 轨迹 → QA | Smoke-tested |
| [VLA 策略](docs/pipelines/02-vla-policy.md) | 图像 + 语言 + 状态 → 策略 → 评估 | 教学基线 Smoke-tested |
| [世界模型与规划](docs/pipelines/03-world-model-planning.md) | 转移 → 动力学 → Rollout → 规划 | 模型 Smoke-tested |
| [RL 后训练](docs/pipelines/04-rl-post-training.md) | MDP → 奖励 → PPO → 回归 | 教学基线 Smoke-tested |
| [机器人基础模型](docs/pipelines/05-rfm-cross-embodiment.md) | 统一观测 → 适配器 → 动作 → 安全层 | Interface-tested |
| [具身推理](docs/pipelines/06-embodied-reasoning.md) | 指令 → 计划 → 技能 → 反馈 → 重规划 | Interface-tested |
| [Sim-to-Real](docs/pipelines/07-sim-to-real.md) | 鲁棒性 → HIL → 影子模式 → 受控部署 | 已文档化；依赖硬件 |
| [灵巧手重定向](docs/pipelines/08-dexterous-retargeting.md) | 关键点 → 几何 → 优化 → 平滑 | 合成 Smoke-tested |
| [感知与状态估计](docs/pipelines/09-perception-state-estimation.md) | 标定 → 同步 → 融合 → 不确定性 | 合成 Smoke-tested |
| [导航与运动控制](docs/pipelines/10-navigation-locomotion.md) | 状态 → 地图/地形 → 规划 → 控制 → 恢复 | 网格导航 Smoke-tested |
| [灵巧抓取与精细操作](docs/pipelines/11-dexterous-manipulation.md) | 状态 → 预抓取 → 接触 → 抬升 → 保持/恢复 | 抽象接触动力学 Smoke-tested |

```bash
python scripts/run_pipeline.py --validate
python scripts/run_pipeline.py --run vla-policy --dry-run
python scripts/run_pipeline.py --run world-model-planning
```

可执行契约位于 [`pipelines/manifest.json`](pipelines/manifest.json)。合成路径验证限定范围内的行为和接口连通性，不等于复现大规模结果或真机性能。

## 七条科研路线

| 方向 | Pipeline 序列 | 必须形成的研究产物 |
|---|---|---|
| [基础模型与 VLA](docs/learning-paths/README_CN.md#foundation-models-vla) | 数据 → VLA → RFM | 策略、适配器、基线和消融 |
| [操作与模仿学习](docs/learning-paths/README_CN.md#manipulation-imitation) | 数据 → VLA → RL | 闭环基线和失败分类 |
| [灵巧操作与遥操作](docs/learning-paths/README_CN.md#dexterity-teleoperation) | 重定向 → 状态 → 抓取 → Sim-to-Real | 运动、接触、保持和任务证据 |
| [导航与具身智能体](docs/learning-paths/README_CN.md#navigation-embodied-agents) | 状态 → 导航 → 推理 | 智能体闭环、恢复协议和报告 |
| [人形与运动控制](docs/learning-paths/README_CN.md#humanoids-locomotion) | 运动 → RL → Sim-to-Real | 运动协议、鲁棒性测试和安全门禁 |
| [感知与世界模型](docs/learning-paths/README_CN.md#perception-world-models) | 状态 → 世界模型 | 带不确定性的状态估计和预测 Rollout |
| [仿真、数据与评估](docs/learning-paths/README_CN.md#simulation-data-evaluation) | 数据 → 世界模型 → Sim-to-Real | 数据说明、基准和晋级决定 |

<a id="evidence"></a>
## 证据先于结论

| 证据等级 | 能够支持什么 | 不能支持什么 |
|---|---|---|
| **导入/接口检查** | 模块和 Schema 能够连接 | 有效行为或正确物理 |
| **合成 Smoke Test** | 限定路径能够确定性执行 | 泛化、基准质量或真机迁移 |
| **教学基准** | 固定协议产生了记录结果 | SOTA 结论或迁移到其他配置 |
| **真机验证** | 指定系统通过了有边界的物理协议 | 协议之外的安全性或性能 |

仓库内 PushCube 结果使用了不同的训练和评估预算，只能作为教学快照，不能作为受控排行榜。结构化状态 BC 基线取得了可用分数，而多个视觉策略基线仍是负结果。请在 [`BENCHMARK.md`](BENCHMARK.md) 和[基准报告](docs/benchmark_report.md)中同时阅读精确协议、每种方法的预算和原始产物边界。

<a id="docs"></a>
## 文档地图

| 学习 | 构建 | 验证 |
|---|---|---|
| [细粒度课程合同](docs/curriculum_cn.md) | [环境配置](docs/setup/README_CN.md) | [验证规范](docs/VALIDATION.md) |
| [从这里开始](docs/start-here-cn.md) | [学习者模板](learner/README.md) | [课程审查](docs/CURRICULUM_AUDIT_CN.md) |
| [知识体系](docs/knowledge-system/README_CN.md) | [MuJoCo 场景搭建](docs/tutorials/mujoco-scene-building.md) | [真实性审查](docs/CLAIM_REVIEW.md) |
| [基础课程](docs/foundations/00-roadmap.md) | [Pipeline 目录](docs/pipelines/README_CN.md) | [主要来源](docs/SOURCES.md) |
| [VLA 与 WAM 专项](docs/specializations/README_CN.md) | [算法族目录](learning_tracks/vla_wam_algorithms.json) | [VLA/WAM 证据边界](docs/specializations/README_CN.md#证据边界) |
| [领域地图](docs/field-map-cn.md) | [科研路线](docs/learning-paths/README_CN.md) | [基准协议](BENCHMARK.md) |

### 验证仓库

```bash
python scripts/check_markdown_links.py
python scripts/check_markdown_format.py
python scripts/check_claims.py
python scripts/run_knowledge_map.py --validate
python scripts/run_pipeline.py --validate
python scripts/run_learning_path.py --validate
python scripts/run_curriculum.py --validate
python scripts/audit_repository.py
python -m pytest tests/ -q
```

证据阶梯是：导入 → Smoke → 确定性测试 → Benchmark → 真机验证。每个结果都应保留命令、随机种子、提交、数据版本、检查点、硬件、Episode 数量和机器可读指标。

## 贡献与许可

提交课程、Pipeline、基准结论或机器人适配器之前，请阅读 [`CONTRIBUTING.md`](CONTRIBUTING.md)。项目原创内容采用 [MIT License](LICENSE)；随仓库提供的上游资产继续保留各自许可，详见[第三方声明](THIRD_PARTY_NOTICES.md)。

维护者：[Gangwei Li](https://github.com/Dld0621)。
