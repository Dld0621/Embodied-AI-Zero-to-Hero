# 具身智能细粒度课程合同

**[English](curriculum.md) · 简体中文**

本课程把仓库的 45 节点前置依赖图转化为可执行学习顺序，明确每一阶段需要理解什么、构建什么、保留什么，以及达到什么证据后才能继续。

> 建议节奏只是规划模板，不是完成时间承诺。是否进入下一阶段取决于检查点，而不是已经投入多少小时。

第一次使用请先阅读[从这里开始](start-here-cn.md)。机器可读的 L0–L5、M00–M11、个人目标与三级 Capstone 位于 [`curriculum/manifest.json`](../curriculum/manifest.json)，可运行：

```bash
python scripts/run_curriculum.py --diagnose --lang zh
python scripts/run_curriculum.py --plan full-stack-expert --hours-per-week 8 --lang zh
```

模块如何评分见[统一评估](assessment-cn.md)，专家毕业要求见[三级 Capstone](capstone-cn.md)。

## 选择工作模式

| 模式 | 首要目标 | 最低保留产物 | 推荐入口 |
|---|---|---|---|
| **学习者** | 建立正确知识模型并连接概念 | 推导笔记、完成的练习和前置依赖路径 | 从 L0 开始，沿图谱学习 |
| **工程者** | 让一套闭环系统可复现、可诊断 | 环境回执、配置、日志、指标和失败报告 | L0 → L2 → 一条 Pipeline |
| **科研者** | 用受控基线检验研究假设 | 冻结协议、基线、消融、不确定性和结论 | 补齐缺失节点，再进入 L3 → L5 |

## 六阶段进阶

### L0 · 计算工具与实验纪律

**学习内容**

- Python、NumPy、张量形状、数据类型、单位和数组语义。
- 配置、接口、确定性随机种子、测试和版本控制。
- 实验身份：代码提交、数据版本、参数、环境与输出路径。

**实践产物**

- 在干净环境中运行一个仓库示例。
- 生成机器可读配置和带时间戳的结果目录。
- 跟踪示例中每个输入输出的形状。

**检查点**

- 其他人不需要猜测依赖或参数就能复现命令。
- 运行记录包含随机种子、提交、环境和产物位置。

主课程：[机器人 Python](foundations/01-python-for-robotics.md) · [评估与可复现性](foundations/14-evaluation-and-reproducibility.md)

### L1 · 物理系统的数学语言

**学习内容**

- 线性代数、最小二乘、概率、估计与优化。
- 坐标系、变换复合、SO(3)、SE(3) 和旋转表示。
- 条件数、离散化、有限差分和数值稳定性。

**实践产物**

- 推导并数值验证一条坐标变换链。
- 在已知退化场景附近比较至少两种旋转表示。
- 求解一个带约束最小二乘问题，并报告残差和条件数。

**检查点**

- 每个向量都标注坐标系、单位、形状和时间戳语义。
- 数值结果与独立检查在声明容差内一致。

主课程：[线性代数](foundations/02-linear-algebra.md) · [坐标变换](foundations/05-coordinate-transform.md) · [SO(3) 与 SE(3)](foundations/06-se3-and-rotation.md) · [概率与优化](foundations/11-probability-and-optimization.md)

### L2 · 机器人建模、感知、控制与仿真

**学习内容**

- 正运动学、Jacobian、逆运动学、刚体动力学与执行器边界。
- 接触、摩擦、抓取稳定性、反馈、阻抗、饱和与 Watchdog。
- 传感器模型、标定、同步、融合和不确定性。
- URDF/MJCF 结构、场景组合、接触参数、可观测性和 Reset 设计。

**实践产物**

- 创建或检查机器人模型，解释关节、坐标系、执行器和碰撞几何。
- 运行一套带明确频率、限位和停止条件的闭环控制器。
- 搭建 MuJoCo 场景，加入物体与传感器，验证接触和状态日志。
- 注入时序、标定或传感器故障并诊断下游影响。

**检查点**

- 一条轨迹能够连接观测 → 状态估计 → 命令 → 仿真响应 → 指标。
- 报告能够区分模型误差、控制误差、传感器误差和任务定义误差。

主课程：[FK、Jacobian 与 IK](foundations/07-fk-jacobian-ik.md) · [控制基础](foundations/08-control-basics.md) · [MuJoCo](foundations/09-mujoco-basics.md) · [感知与传感器](foundations/12-perception-and-sensors.md) · [机器人系统与安全](foundations/13-robot-systems-and-safety.md)

### L3 · 数据、策略、预测与规划

**学习内容**

- Episode Schema、多模态对齐、覆盖度、无泄漏划分和数据质量。
- 神经网络训练、Transformer、行为克隆、协变量偏移与动作表示。
- VLA 策略、RL、MDP/POMDP、世界模型与轨迹优化。

**实践产物**

- 采集或验证包含同步观测、动作、任务标签和元数据的数据集。
- 训练一个基线，并保留配置、检查点、学习曲线和评估输出。
- 比较开环预测和闭环任务行为。
- 增加一个面向失败的划分或扰动，不只报告平均分。

**检查点**

- 训练、验证和测试身份明确，数据泄漏检查通过。
- 评估使用固定协议，并报告样本数、种子策略、不确定性和失败案例。

主课程：[数据集与训练](foundations/10-dataset-and-training.md) · [Transformer](foundations/04-transformer-basics.md) · [VLA Pipeline](pipelines/02-vla-policy.md) · [世界模型 Pipeline](pipelines/03-world-model-planning.md) · [RL Pipeline](pipelines/04-rl-post-training.md)

**VLA/WAM 专项：**通过 L3 的数据与基线检查点后，使用[专项总览](specializations/README_CN.md)选择路线。[VLA 从零到一](specializations/vla-zero-to-one-cn.md)讲解多模态策略和动作生成算法族；[WAM 从零到一](specializations/wam-zero-to-one-cn.md)先建立世界模型规划基线，再进入视频—动作联合学习。匹配预算的策略与模型基线尚未完成时，不应直接从大规模联合模型开始。

### L4 · 任务级具身系统

选择一类任务并完成完整闭环。

| 任务族 | 必须包含的阶段 | 最低任务证据 |
|---|---|---|
| 操作与模仿学习 | 观测 → 策略/规划 → 控制 → 任务指标 → 恢复 | 成功定义、物体/状态轨迹、失败分类 |
| 灵巧操作与遥操作 | 人类输入 → 重定向 → 命令准入 → 接触 → 保持/任务 | 几何误差、限位、接触/保持、任务结果 |
| 导航与智能体 | 状态/地图 → 规划 → 动作 → 定位更新 → 恢复 | 目标完成、碰撞/路径指标、恢复案例 |
| 人形与运动控制 | 状态 → 运动命令 → 全身控制 → 平衡 → 扰动测试 | 稳定性、跟踪、跌倒/失败、安全边界 |

**检查点**

- 系统具有任务级成功定义，而不只是看起来合理的轨迹。
- 失败能够定位到具体阶段，且恢复或停止行为可观察。

主入口：[科研路线地图](learning-paths/README_CN.md) · [Pipeline 目录](pipelines/README_CN.md)

### L5 · 证据、泛化与部署决策

**学习内容**

- 基线、消融、不确定性、负结果、鲁棒性和分布偏移。
- 系统辨识、域随机化、Hardware-in-the-Loop、影子模式和受控部署。
- 运行授权、监督、有界命令、停止路径和审计日志。

**实践产物**

- 在比较方法之前冻结一套评估协议。
- 在匹配预算下运行一个基线和一个假设驱动的修改。
- 测试一种声明过的偏移：物体、场景、视角、本体、延迟或扰动。
- 写出晋级决定，同时列出支持证据和仍然存在的阻塞。

**检查点**

- 较低证据等级不会被描述成更高等级。
- 真机执行必须获得明确授权并遵循有边界安全协议；仿真不能替代授权。

主参考：[验证规范](VALIDATION.md) · [真实性审查](CLAIM_REVIEW.md) · [基准合同](../BENCHMARK.md) · [Sim-to-Real Pipeline](pipelines/07-sim-to-real.md)

## 十二个学习单元的建议顺序

这是一份排期模板。任何学习单元都可以重复或拆分，直到检查点通过。这些排期单元与正式模块 M00–M11 并非一一对应；验收和进度记录仍以[课程合同](../curriculum/manifest.json)中的正式模块 ID 为准。

| 学习单元 | 重点 | 交付物 |
|---:|---|---|
| 1 | 环境、Python、形状与单位 | 可复现环境回执和带数据流标注的示例 |
| 2 | 线性代数与坐标变换 | 经过验证的坐标系复合 Notebook |
| 3 | SO(3)、SE(3)、运动学与 IK | 带残差和失败案例的 FK/IK 对比 |
| 4 | 动力学、接触与控制 | 有界闭环仿真和故障诊断 |
| 5 | 传感器、标定与状态估计 | 带不确定性的同步状态估计轨迹 |
| 6 | MuJoCo 建模与任务设计 | 自定义场景、接触检查、Reset 分布和日志 |
| 7 | 数据 Schema 与质量 | Datasheet、划分清单、泄漏与覆盖报告 |
| 8 | 行为克隆与动作表示 | 训练基线及开环/闭环评估 |
| 9 | VLA、世界模型或 RL | 一种进阶策略/预测基线和失败报告 |
| 10 | 任务级 Pipeline | 完整的操作、灵巧手、导航或运动控制闭环 |
| 11 | 鲁棒性与泛化 | 声明过的偏移测试，以及置信区间或重复试验 |
| 12 | 科研与部署决定 | 可复现实验、消融、证据等级和晋级/停止决定 |

## 每个知识点的统一操作流程

1. **诊断**：使用 `python scripts/run_knowledge_map.py --path-to <node> --lang zh` 解析依赖。
2. **学习**：阅读主文档并复现最小练习。
3. **构建**：执行映射的 Pipeline，保留中间产物。
4. **测量**：在固定协议下统计任务指标和分阶段失败。
5. **决策**：判断证据支持继续、重复还是停止。

机器可读图谱位于 [`knowledge/manifest.json`](../knowledge/manifest.json)。验证命令：

```bash
python scripts/run_knowledge_map.py --validate
python scripts/run_knowledge_map.py --summary
python scripts/run_pipeline.py --validate
python scripts/run_learning_path.py --validate
python scripts/run_curriculum.py --validate
```
