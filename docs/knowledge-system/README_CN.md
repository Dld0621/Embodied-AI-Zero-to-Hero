# 具身智能知识体系

**[逐点图解：从一个概念开始拆开学](../knowledge-atlas/index.md)**。下面的 45 个节点是依赖地图；细解把每个节点继续拆成原理、手算例子、图和自测，且链接回原课程的验收任务。

[English](README.md){ .md-button } [查看机器可读知识图谱](https://github.com/Dld0621/Embodied-AI-Zero-to-Hero/blob/master/knowledge/manifest.json){ .md-button }

这是仓库在**知识点粒度**上的单一事实源：45 个知识节点、9 个领域、6 个阶段。每个节点都绑定前置依赖、主文档、工程 Pipeline、学习证据与验收方式。

![具身智能知识体系](../assets/knowledge-system-cn.svg)

> [!IMPORTANT]
> 知识节点是一份学习契约，不是“已经掌握”或“已经取得科研性能”的声明。能够解释、推导或跑通 smoke，不能推出 benchmark 成功，更不能推出真机安全。

## 用四层结构理解仓库

| 层级 | 回答的问题 | 单一事实源 |
|---|---|---|
| **知识节点** | 具体要懂什么，如何证明自己真的懂？ | [`knowledge/manifest.json`](https://github.com/Dld0621/Embodied-AI-Zero-to-Hero/blob/master/knowledge/manifest.json) |
| **基础文档** | 原理、推导、示例和权威来源在哪里？ | [`docs/foundations/`](../foundations/00-roadmap.md) 与对应专题文档 |
| **工程 Pipeline** | 多个知识点如何组成闭环产物？ | [Pipeline 目录](../pipelines/README_CN.md) |
| **科研路线** | 哪个序列回答研究问题，何时允许晋级？ | [七条科研路线](../learning-paths/README_CN.md) |

过去的课程、Pipeline 与科研路线分别可读，但依赖关系主要靠人工理解。新的知识图谱把三层连接变成可自动检查的契约，避免用一个“方向名称”代替实现与证据。

## 六阶段递进

| 阶段 | 目标 | 退出门槛 |
|---:|---|---|
| **L0 · 入门与工具** | 运行代码、检查接口、保留实验溯源。 | 仅凭回执重新运行一个确定性任务。 |
| **L1 · 数学模型** | 明确假设，理解几何、不确定性与优化。 | 完成推导并用数值实验验证。 |
| **L2 · 机器人闭环** | 连接本体、动力学、感知、状态、控制与安全。 | 注入扰动后仍能展示受限行为。 |
| **L3 · 数据与学习** | 构建数据、表示、策略、预测模型与 RL 闭环。 | 保留数据/模型溯源，并区分离线与闭环结果。 |
| **L4 · 任务智能** | 组合成操作、灵巧手、导航、运动与恢复系统。 | 按任务协议完成闭环并给出阶段级失败报告。 |
| **L5 · 证据与部署** | 比较方法、检查泛化、决定是否提高风险。 | 不越过现有证据边界地做出晋级决策。 |

## 从真实缺口开始

| 如果你已经会…… | 建议起点 | 不能跳过 |
|---|---|---|
| 写 Python，但不熟机器人坐标系 | `robot-coordinate-frames` | 单位、变换方向、往返检查 |
| 训练视觉模型，但不会执行机器人命令 | `learning-action-representations` | 控制频率、单位、边界与驱动语义 |
| 设计机械结构，但不会训练策略 | `learning-neural-networks` | 数据切分、验证与闭环分布偏移 |
| 能跑仿真，但不知道结果能否相信 | `sim-scene-dynamics` | 接触模型、传感语义、复位分布与证据边界 |
| 能读 VLA 论文，但没有可复现基线 | `learning-vla` | 数据溯源、消融、任务成功与延迟 |
| 能做人手重定向，但需要任务级灵巧操作 | `task-dexterity-teleoperation` | 可实现性、命令准入、接触、保持与任务证据 |
| 已有策略准备上机器人 | `deploy-sim-to-real` | HIL、影子模式、停止路径、监督与回滚 |

不要凭经验猜前置依赖，直接解析目标节点：

```bash
python scripts/run_knowledge_map.py --validate
python scripts/run_knowledge_map.py --stats
python scripts/run_knowledge_map.py --show learning-vla --lang zh
python scripts/run_knowledge_map.py --path-to task-dexterity-teleoperation --lang zh
```

## 九大知识领域

<a id="computing"></a>
### 1. 科学计算

| 节点 | 核心能力 | 主文档 | 学习证据 |
|---|---|---|---|
| `computing-python-numpy` | Python、NumPy、张量形状、类型、单位与数据流 | [机器人 Python](../foundations/01-python-for-robotics.md) | 执行 |
| `computing-software-contracts` | 显式观测/动作协议、配置、测试与非法输入拒绝 | [机器人系统与安全](../foundations/13-robot-systems-and-safety.md) | 执行 |
| `computing-experiment-workflow` | 提交、环境、种子、数据、检查点、指标与产物溯源 | [评测与复现](../foundations/14-evaluation-and-reproducibility.md) | 评估 |

**退出问题：**另一个人能否在不依赖本机隐藏状态的情况下复现实验产物？

<a id="mathematics"></a>
### 2. 数学与不确定性

| 节点 | 核心能力 | 主文档 | 学习证据 |
|---|---|---|---|
| `math-linear-algebra` | 向量、矩阵、投影、分解、最小二乘与条件数 | [线性代数](../foundations/02-linear-algebra.md) | 推导 |
| `math-probability-statistics` | 随机变量、估计、不确定性、置信与校准 | [概率与优化](../foundations/11-probability-and-optimization.md) | 推导 |
| `math-optimization` | 目标、约束、正则、不可行性与终止条件 | [概率与优化](../foundations/11-probability-and-optimization.md) | 推导 |
| `math-numerical-stability` | 离散化、奇异、阻尼、尺度与求解器敏感性 | [FK、Jacobian 与 IK](../foundations/07-fk-jacobian-ik.md) | 执行 |

**退出问题：**能否明确优化对象、成立假设，以及如何发现数值误差？

<a id="robot-modeling"></a>
### 3. 机器人建模与力学

| 节点 | 核心能力 | 主文档 | 学习证据 |
|---|---|---|---|
| `robot-coordinate-frames` | 坐标系、单位、变换方向与复合顺序 | [坐标变换](../foundations/05-coordinate-transform.md) | 推导 |
| `robot-so3-se3` | SO(3)、SE(3)、旋转表示、转换与插值 | [SO(3) 与 SE(3)](../foundations/06-se3-and-rotation.md) | 推导 |
| `robot-kinematics` | FK、Jacobian、数值 IK、关节限制与奇异处理 | [FK、Jacobian 与 IK](../foundations/07-fk-jacobian-ik.md) | 执行 |
| `robot-rigid-body-dynamics` | 力、加速度、状态积分、时间步与能量/跟踪行为 | [控制基础](../foundations/08-control-basics.md) | 推导 |
| `robot-contact-friction` | 碰撞、接触约束、摩擦、滑移、保持与抓取稳定 | [灵巧操作](../pipelines/11-dexterous-manipulation.md) | 评估 |
| `robot-actuation-transmission` | 驱动、传动、腱绳、可执行坐标与限制 | [概念百科](../00-concepts-encyclopedia.md) | 解释 |

**退出问题：**能否把任务空间意图一直追踪到机构上的受限运动或力？

<a id="sensing-control"></a>
### 4. 感知、估计与控制

| 节点 | 核心能力 | 主文档 | 学习证据 |
|---|---|---|---|
| `system-sensor-models` | 模态、坐标系、单位、频率、延迟、噪声与缺失语义 | [感知与传感器](../foundations/12-perception-and-sensors.md) | 解释 |
| `system-calibration-synchronization` | 空间标定、时钟、时间对齐与实测残差 | [感知与状态估计](../pipelines/09-perception-state-estimation.md) | 评估 |
| `system-state-estimation` | 滤波、融合、不确定性、陈旧数据处理与状态有效性 | [感知与状态估计](../pipelines/09-perception-state-estimation.md) | 执行 |
| `system-feedback-control` | 反馈、轨迹、频率、饱和、抗积分饱和与跟踪 | [控制基础](../foundations/08-control-basics.md) | 执行 |
| `system-force-compliance` | 力、阻抗、柔顺与受限接触交互 | [控制基础](../foundations/08-control-basics.md) | 评估 |
| `system-realtime-safety` | 看门狗、限制、状态机、停止路径、日志与人工权限 | [机器人系统与安全](../foundations/13-robot-systems-and-safety.md) | 部署门禁 |

**退出问题：**观测或命令迟到、非法或不安全时，系统是否进入已知的受限状态？

<a id="simulation-data"></a>
### 5. 仿真与数据

| 节点 | 核心能力 | 主文档 | 学习证据 |
|---|---|---|---|
| `sim-model-formats` | URDF/MJCF、资源、单位、惯量、驱动、传感与溯源 | [MuJoCo 场景搭建](../tutorials/mujoco-scene-building.md) | 执行 |
| `sim-scene-dynamics` | 场景动力学、接触、传感、控制输入、日志与确定性步进 | [MuJoCo 基础](../foundations/09-mujoco-basics.md) | 执行 |
| `sim-task-randomization` | 任务/复位定义、扰动、域随机化与覆盖 | [仿真与数据 Pipeline](../pipelines/01-simulation-data.md) | 评估 |
| `data-episode-schema` | 同步的观测、动作、语言、时间戳、任务与终止字段 | [数据与训练](../foundations/10-dataset-and-training.md) | 执行 |
| `data-quality-splits` | 完整性、覆盖率、分布偏移、无泄漏切分与数据说明 | [数据与训练](../foundations/10-dataset-and-training.md) | 评估 |

**退出问题：**能否重建机器人看到了什么、执行了什么、为何结束，以及该 episode 属于哪个分布？

<a id="robot-learning"></a>
### 6. 机器人学习

| 节点 | 核心能力 | 主文档 | 学习证据 |
|---|---|---|---|
| `learning-neural-networks` | 前向、损失、梯度、优化、验证与过拟合 | [深度学习](../foundations/03-deep-learning-basics.md) | 执行 |
| `learning-transformers-multimodal` | Token、图像、状态、掩码、时间与注意力形状追踪 | [Transformer](../foundations/04-transformer-basics.md) | 推导 |
| `learning-behavior-cloning` | 监督模仿、协变量偏移、闭环漂移与恢复边界 | [ACT 与 Diffusion Policy](../22-act-vs-diffusion-policy.md) | 评估 |
| `learning-action-representations` | 关节/任务动作、增量、分块、Token、扩散、频率与边界 | [动作表示](../24-action-representation-and-tokenization.md) | 推导 |
| `learning-vla` | 视觉语言对齐、动作预测、时序有效性与消融 | [VLA 从零到一](../specializations/vla-zero-to-one-cn.md) | 评估 |
| `learning-reinforcement-learning` | 状态、动作、奖励、终止、探索、后训练与安全 | [RL 零到一](../14-rl-zero-to-one.md) | 评估 |
| `learning-cross-embodiment` | 统一协议、适配器、本体语义与分机器人结果 | [跨本体适配](../25-cross-embodiment-adaptation.md) | 评估 |

**退出问题：**策略输出是否与下游控制器的语义、频率、单位、时域与安全包络一致？

<a id="prediction-planning"></a>
### 7. 预测、规划与推理

| 节点 | 核心能力 | 主文档 | 学习证据 |
|---|---|---|---|
| `planning-mdp-pomdp` | 可观测/隐藏状态、动作、奖励、转移、信念与终止 | [RL 基础](../06-rl-fundamentals-for-vla.md) | 推导 |
| `planning-motion-trajectory` | 路径、轨迹、碰撞、动力学、可行性、优化与反馈 | [优化方法](../04-optimization-methods.md) | 执行 |
| `planning-world-models` | 单步拟合、多步 rollout、不确定性与规划效用 | [世界模型零到一](../15-world-model-zero-to-one.md) | 评估 |
| `planning-task-and-motion` | 类型化目标、前置条件、技能、几何可行性与反馈 | [具身推理与规划](../27-embodied-reasoning-and-planning.md) | 评估 |
| `planning-reasoning-recovery` | 基于状态的监控、失败检测、受限恢复与重规划 | [具身推理 Pipeline](../pipelines/06-embodied-reasoning.md) | 评估 |

**退出问题：**系统能否发现计划已不再匹配世界，并选择受限恢复动作？

<a id="embodied-tasks"></a>
### 8. 具身任务系统

| 节点 | 核心能力 | 主文档 | 学习证据 |
|---|---|---|---|
| `task-manipulation` | 感知、接近、接触、运动、任务验证与恢复 | [灵巧操作 Pipeline](../pipelines/11-dexterous-manipulation.md) | 评估 |
| `task-dexterity-teleoperation` | 人手姿态、机器人可实现性、命令准入、接触、保持与任务证据 | [灵巧手重定向 Pipeline](../pipelines/08-dexterous-retargeting.md) | 评估 |
| `task-navigation` | 定位假设、规划、控制、碰撞、恢复与重规划 | [导航与运动 Pipeline](../pipelines/10-navigation-locomotion.md) | 评估 |
| `task-locomotion-humanoids` | 跟踪、平衡、接触、跌倒、恢复与本体限制 | [导航与运动 Pipeline](../pipelines/10-navigation-locomotion.md) | 部署门禁 |

**退出问题：**任务成功是否有阶段级证据，还是只有一条“看起来合理”的命令或轨迹？

<a id="evaluation-deployment"></a>
### 9. 评测与部署

| 节点 | 核心能力 | 主文档 | 学习证据 |
|---|---|---|---|
| `eval-task-metrics` | 任务定义、分子、分母、episode 协议与失败分类 | [评测与复现](../foundations/14-evaluation-and-reproducibility.md) | 评估 |
| `eval-generalization-robustness` | 留出因素、扰动、鲁棒性与分布偏移 | [评估指标](../06-evaluation-metrics.md) | 评估 |
| `eval-statistics-ablations` | 种子、波动、消融、计算量、负结果与公平比较 | [评测与复现](../foundations/14-evaluation-and-reproducibility.md) | 评估 |
| `deploy-sim-to-real` | 系统辨识、回放、随机化、HIL、影子模式与回滚 | [Sim-to-Real 指南](../19-sim-to-real-guide.md) | 部署门禁 |
| `deploy-hardware-gates` | 授权、监督、受限命令、停止路径、日志与硬件证据 | [验证政策](../VALIDATION.md) | 部署门禁 |

**退出问题：**哪些具体证据允许提高自主性或物理风险，还有哪些门禁未通过？

## 学习证据不等于仓库证据

| 学习证据 | 必须提供 | 不能证明 |
|---|---|---|
| **解释** | 定义概念、假设、单位与失败模式。 | 实现正确 |
| **推导** | 给出关系，并验证极限或数值案例。 | 闭环任务性能 |
| **执行** | 跑通确定性示例并检查产物。 | 泛化或 benchmark 质量 |
| **评估** | 使用固定协议、指标、种子与失败分析。 | 真机就绪 |
| **部署门禁** | 通过声明的安全与运行晋级标准。 | 作用域外的普遍安全 |

仓库的 smoke-tested、interface-tested、benchmark 与 hardware-dependent 标签仍由[验证政策](../VALIDATION.md)管理。两套词汇不能混用。

## 维护契约

新增知识点时：

1. 在 [`knowledge/manifest.json`](https://github.com/Dld0621/Embodied-AI-Zero-to-Hero/blob/master/knowledge/manifest.json) 中加入一个双语节点。
2. 链接真实存在的主文档和至少一条已注册 Pipeline。
3. 声明前置依赖、学习证据、学习结果与验收方式。
4. 在[权威来源表](../SOURCES.md)中新增或更新一手来源。
5. 运行 `python scripts/run_knowledge_map.py --validate` 与仓库测试。

验证器会拒绝缺失文档、未知 Pipeline、依赖未来阶段、重复 ID 与依赖环。它验证仓库结构，但不能自动认证文档语义正确，也不能授权真机操作。
