# 具身智能领域地图

[English](field-map.md){ .md-button }

具身智能不是某一种模型，而是从传感、建模、决策到受控执行的闭环系统。本页把**方向覆盖**与**仓库证据**分开，避免“写到了某个方向”被误读为“已经复现了结果”。

## 能力栈

| 能力 | 工程问题 | 基础章节 | 管线契约 |
|---|---|---|---|
| 传感与标定 | 观测是否同步、标定正确且健康？ | [感知与传感器](foundations/12-perception-and-sensors.md) | [感知与状态估计](pipelines/09-perception-state-estimation.md) |
| 状态估计 | 哪些状态可以安全使用，不确定性多大？ | [概率与优化](foundations/11-probability-and-optimization.md) | [感知与状态估计](pipelines/09-perception-state-estimation.md) |
| 数据与仿真 | 如何生成任务、演示、切分和扰动？ | [MuJoCo](foundations/09-mujoco-basics.md) · [数据与训练](foundations/10-dataset-and-training.md) | [仿真与数据](pipelines/01-simulation-data.md) |
| 策略学习 | 如何把观测和语言变成动作？ | [深度学习](foundations/03-deep-learning-basics.md) · [Transformer](foundations/04-transformer-basics.md) | [VLA 策略](pipelines/02-vla-policy.md) |
| 预测模型 | 执行动作后可能发生什么？ | [概率与优化](foundations/11-probability-and-optimization.md) | [世界模型规划](pipelines/03-world-model-planning.md) |
| 交互改进 | 如何利用奖励和失败改进策略？ | [控制基础](foundations/08-control-basics.md) | [RL 后训练](pipelines/04-rl-post-training.md) |
| 通用策略 | 模型如何跨数据集与本体适配？ | [数据与训练](foundations/10-dataset-and-training.md) | [机器人基础模型与跨本体](pipelines/05-rfm-cross-embodiment.md) |
| 任务推理 | 长指令如何拆分、执行和重规划？ | [机器人系统与安全](foundations/13-robot-systems-and-safety.md) | [具身推理](pipelines/06-embodied-reasoning.md) |
| 操作与灵巧性 | 几何或学习命令如何映射为受约束运动？ | [FK、Jacobian 与 IK](foundations/07-fk-jacobian-ik.md) | [灵巧手重定向](pipelines/08-dexterous-retargeting.md) |
| 导航与运动 | 本体如何在定位和稳定约束下移动？ | [控制基础](foundations/08-control-basics.md) · [系统与安全](foundations/13-robot-systems-and-safety.md) | [导航与运动控制](pipelines/10-navigation-locomotion.md) |
| 迁移与部署 | 提高风险前必须通过哪些门禁？ | [评估与复现](foundations/14-evaluation-and-reproducibility.md) | [Sim-to-Real](pipelines/07-sim-to-real.md) |

## 当前证据

| 等级 | 方向 | 含义 |
|---|---|---|
| **Smoke-tested** | 仿真/数据、VLA、世界模型、RL、灵巧手重定向 | 仓库内轻量路径可完成；性能结论需要独立证据。 |
| **Interface-tested** | 机器人基础模型/跨本体、具身推理 | 本地协议、适配器或规划器已接通；不代表真实权重或硬件通过。 |
| **Documented** | Sim-to-Real、感知/状态估计、导航/运动控制 | 已定义工程契约与门禁；不存在能代表完整系统的通用本地命令。 |

## 按研究目标选择

| 目标 | 起点 | 下一步要证明 |
|---|---|---|
| 从零学习机器人智能 | [中文基础路线](foundations/00-roadmap.md) | 完成一条 smoke 管线并保留产物。 |
| 构建多模态策略 | [VLA 管线](pipelines/02-vla-policy.md) | 闭环成功、语言消融、延迟与失败案例。 |
| 研究预测与规划 | [世界模型管线](pipelines/03-world-model-planning.md) | 分别报告多步预测误差与规划任务成功。 |
| 跨机器人本体研究 | [RFM 管线](pipelines/05-rfm-cross-embodiment.md) | 动作语义、适配覆盖率和分本体结果。 |
| 研究灵巧手 | [重定向管线](pipelines/08-dexterous-retargeting.md) | 分开验证几何、时序、接触/任务和硬件证据。 |
| 构建移动或足式系统 | [导航/运动契约](pipelines/10-navigation-locomotion.md) | 定位、跟踪、碰撞/跌倒、恢复与迁移证据。 |

## 明确不声称的内容

仓库目前**不声称**已复现 SLAM benchmark、导航成功率、足式运动策略、通用真机部署或具备竞争力的大规模基础模型结果。这些是后续扩展目标，不是隐藏的已完成项。
