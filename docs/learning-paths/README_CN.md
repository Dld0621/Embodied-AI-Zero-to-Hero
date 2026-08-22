# 七条科研路线

选择路线前，如需解析准确前置，请先进入 [45 节点知识体系](../knowledge-system/README_CN.md)。科研路线定义目标与交付，知识图谱定义知识点顺序与验收方式。

[English](README.md) · **简体中文**

这一层把宽泛兴趣转化为可执行的研究任务书。每条路线都把研究问题连接到基础课程、已登记 Pipeline、明确交付物、评测指标、晋级门槛与证据边界。

> 路线用于组织工作，不会自动提高任何 Pipeline 的证据等级。提出结论前，必须检查所链接 Pipeline 的状态。

## 按科研产出选择

| 方向 | 最终交付 | 核心 Pipeline |
|:---|:---|:---|
| [基础模型与 VLA](#foundation-models-vla) | 语言条件策略 + 适配器 + 消融 | 数据 → VLA → RFM |
| [操作与模仿学习](#manipulation-imitation) | 闭环操作基线 + 失败分类 | 数据 → VLA → RL |
| [灵巧操作与遥操作](#dexterity-teleoperation) | 重定向运动 + 接触感知抓取实验 | 重定向 → 状态 → 抓取 → Sim-to-Real |
| [导航与具身智能体](#navigation-embodied-agents) | 状态感知智能体闭环 + 恢复报告 | 状态 → 导航 → 推理 |
| [人形与运动控制](#humanoids-locomotion) | 运动协议 + 安全与迁移门槛 | 运动 → RL → Sim-to-Real |
| [感知与世界模型](#perception-world-models) | 不确定状态流 + 预测 rollout | 状态 → 世界模型 |
| [仿真、数据与评测](#simulation-data-evaluation) | 数据说明 + 基准 + 晋级决策 | 数据 → 世界模型 → Sim-to-Real |

检查机器可读路线契约：

```bash
python scripts/run_learning_path.py --list --lang zh
python scripts/run_learning_path.py --show foundation-models-vla --lang zh
python scripts/run_learning_path.py --validate
```

<a id="foundation-models-vla"></a>
## 1. 基础模型与 VLA

**研究问题：** 如何让语言、视觉与机器人状态产生有效的闭环动作？

- **基础课程：** [深度学习](../foundations/03-deep-learning-basics.md)、[Transformer](../foundations/04-transformer-basics.md)、[数据与训练](../foundations/10-dataset-and-training.md)、[评测](../foundations/14-evaluation-and-reproducibility.md)
- **Pipeline 顺序：** [仿真与数据](../pipelines/01-simulation-data.md) → [VLA 策略](../pipelines/02-vla-policy.md) → [RFM 与跨本体](../pipelines/05-rfm-cross-embodiment.md)
- **交付物：** 语言条件闭环策略、适配器契约与消融报告
- **核心指标：** `task_success_rate`、`language_condition_gap`、`inference_latency_ms`、`adapter_coverage`
- **晋级门槛：** 在相同数据与协议下超过已声明基线，并通过动作模式与安全检查
- **证据边界：** RFM 路径仅完成接口测试，不宣称复现了有竞争力的大规模基础模型

<a id="manipulation-imitation"></a>
## 2. 操作与模仿学习

**研究问题：** 如何把示范转化为稳健的操作策略？

- **基础课程：** [运动学与 IK](../foundations/07-fk-jacobian-ik.md)、[控制](../foundations/08-control-basics.md)、[数据与训练](../foundations/10-dataset-and-training.md)、[评测](../foundations/14-evaluation-and-reproducibility.md)
- **Pipeline 顺序：** [仿真与数据](../pipelines/01-simulation-data.md) → [VLA 策略](../pipelines/02-vla-policy.md) → [RL 后训练](../pipelines/04-rl-post-training.md)
- **交付物：** 包含数据诊断、闭环评测与失败分类的操作基线
- **核心指标：** `task_success_rate`、`selection_accuracy`、`collision_rate`、`inference_latency_ms`
- **晋级门槛：** 在固定随机种子下获得可重复的任务增益，且安全违规不增加
- **证据边界：** PushCube 是教学规模任务，不能证明通用真实世界操作能力

<a id="dexterity-teleoperation"></a>
## 3. 灵巧操作、重定向与遥操作

**研究问题：** 如何把人手运动迁移到机器人手，并区分几何、接触与任务证据？

- **基础课程：** [SE(3)](../foundations/06-se3-and-rotation.md)、[运动学与 IK](../foundations/07-fk-jacobian-ik.md)、[优化](../foundations/11-probability-and-optimization.md)、[感知与传感器](../foundations/12-perception-and-sensors.md)
- **Pipeline 顺序：** [灵巧手重定向](../pipelines/08-dexterous-retargeting.md) → [感知与状态](../pipelines/09-perception-state-estimation.md) → [灵巧抓取与精细操作](../pipelines/11-dexterous-manipulation.md) → [Sim-to-Real](../pipelines/07-sim-to-real.md)
- **交付物：** 一段重定向关节序列与一项接触感知抓取实验，并分别报告几何、时序、碰撞、保持、任务与硬件证据
- **核心指标：** `retargeting_error`、`joint_limit_violation_rate`、`latency_ms`、`grasp_success_rate`、`mean_lift_height_m`、`max_lateral_slip_m`
- **晋级门槛：** 先通过几何与时序检查，再通过接触建立、抬升、保持、滑移与鲁棒性门槛；硬件执行前必须另设门槛
- **证据边界：** 已提交的接触动力学冒烟测试仅验证一个抽象 MuJoCo 抓取任务；不能证明在手重定位、学习策略质量、真实手型迁移或真机能力

<a id="navigation-embodied-agents"></a>
## 4. 导航与具身智能体

**研究问题：** 智能体如何在长时程中估计状态、规划、行动、恢复并重规划？

- **基础课程：** [坐标系](../foundations/05-coordinate-transform.md)、[控制](../foundations/08-control-basics.md)、[感知](../foundations/12-perception-and-sensors.md)、[系统与安全](../foundations/13-robot-systems-and-safety.md)、[评测](../foundations/14-evaluation-and-reproducibility.md)
- **Pipeline 顺序：** [感知与状态](../pipelines/09-perception-state-estimation.md) → [导航与运动](../pipelines/10-navigation-locomotion.md) → [具身推理](../pipelines/06-embodied-reasoning.md)
- **交付物：** 包含类型化计划、安全事件、恢复行为与场景报告的状态感知导航闭环
- **核心指标：** `localization_or_state_error`、`goal_success_rate`、`collision_or_fall_rate`、`recovery_success_rate`、`replan_count`
- **晋级门槛：** 在固定地图与扰动下达到场景级成功、碰撞与恢复阈值
- **证据边界：** 可运行路径是合成网格导航，不是 SLAM、移动操作或足式基准复现

<a id="humanoids-locomotion"></a>
## 5. 人形机器人与运动控制

**研究问题：** 运动策略如何跟踪指令、恢复并保持在安全包络内？

- **基础课程：** [SE(3)](../foundations/06-se3-and-rotation.md)、[运动学](../foundations/07-fk-jacobian-ik.md)、[控制](../foundations/08-control-basics.md)、[MuJoCo](../foundations/09-mujoco-basics.md)、[系统与安全](../foundations/13-robot-systems-and-safety.md)、[评测](../foundations/14-evaluation-and-reproducibility.md)
- **Pipeline 顺序：** [导航与运动](../pipelines/10-navigation-locomotion.md) → [RL 后训练](../pipelines/04-rl-post-training.md) → [Sim-to-Real](../pipelines/07-sim-to-real.md)
- **交付物：** 覆盖跟踪、扰动、恢复、安全与迁移门槛的运动控制实验协议
- **核心指标：** `path_or_velocity_tracking_error`、`collision_or_fall_rate`、`recovery_success_rate`、`sim_real_gap`
- **晋级门槛：** 任何半实物或真机试验前，先通过纯运动仿真与安全回归
- **证据边界：** 不宣称已复现人形运动策略或获得本地人形硬件验证结果

<a id="perception-world-models"></a>
## 6. 感知与世界模型

**研究问题：** 如何把不确定观测转化为状态估计与有用的预测 rollout？

- **基础课程：** [坐标系](../foundations/05-coordinate-transform.md)、[概率](../foundations/11-probability-and-optimization.md)、[感知与传感器](../foundations/12-perception-and-sensors.md)、[评测](../foundations/14-evaluation-and-reproducibility.md)
- **Pipeline 顺序：** [感知与状态](../pipelines/09-perception-state-estimation.md) → [世界模型与规划](../pipelines/03-world-model-planning.md)
- **交付物：** 带不确定性的状态流、预测 rollout 模型与从标定到规划的误差分析
- **核心指标：** `calibration_reprojection_error_px`、`sensor_sync_skew_ms`、`uncertainty_calibration_error`、`multi_step_rollout_error`、`planned_task_success_rate`
- **晋级门槛：** 在同一观测协议下证明不确定性得到校准，并改善下游规划
- **证据边界：** 合成状态与教学规模动力学不能证明开放世界视觉预测或真实传感器鲁棒性

<a id="simulation-data-evaluation"></a>
## 7. 仿真、数据与评测

**研究问题：** 实验如何产生可追溯的数据与经得住比较的证据？

- **基础课程：** [MuJoCo](../foundations/09-mujoco-basics.md)、[数据与训练](../foundations/10-dataset-and-training.md)、[系统与安全](../foundations/13-robot-systems-and-safety.md)、[评测](../foundations/14-evaluation-and-reproducibility.md)
- **Pipeline 顺序：** [仿真与数据](../pipelines/01-simulation-data.md) → [世界模型与规划](../pipelines/03-world-model-planning.md) → [Sim-to-Real](../pipelines/07-sim-to-real.md)
- **交付物：** 版本化数据说明、基准报告、原始指标与明确的部署晋级决策
- **核心指标：** `dataset_coverage`、`data_integrity_rate`、`task_success_rate`、`robustness_gap`、`sim_real_gap`
- **晋级门槛：** 晋级前保留协议、随机种子、数据版本、产物、负结果与安全证据
- **证据边界：** 仓库审计验证已提交的契约与产物，不认证外部数据集或硬件部署

## 科研闭环

每条路线都遵循：冻结问题 → 选择一个基线 → 运行最小路径 → 保留原始产物 → 分析失败 → 每次只改变一个变量 → 固定评测重跑 → 仅在声明门槛通过后晋级。

路线的唯一数据源是 [`learning_paths/manifest.json`](../../learning_paths/manifest.json)；Pipeline 的证据状态独立记录在 [`pipelines/manifest.json`](../../pipelines/manifest.json)。
