# 仿真与数据生成 / Simulation and Data Generation

## English contract

- **Objective:** turn a fixed task contract into deterministic simulation episodes with synchronized observations, actions, rewards, termination flags, and provenance.
- **Inputs:** robot/environment model, task reset distribution, observation/action schemas, controller or expert, seeds, and success definition.
- **Stages:** contract → simulator → expert validation → episode collection → dataset QA → versioned artifact.
- **Acceptance:** the expert reaches its declared threshold across fixed seeds; shapes, timestamps, ranges, resets, and split boundaries pass checks; rendering produces the documented observation.
- **Evidence:** the local entry point is smoke-tested. It proves the data path runs, not that a learned policy succeeds. Apply the [validation policy](../VALIDATION.md).

## 目标与边界

把一个机器人任务变成可重复的数据生成系统：任务定义明确、状态与动作可追踪、专家策略可复跑、失败轨迹不被静默丢弃。本仓库以二维 PushCube 教学环境验证接口；它不等同于高保真动力学或真实机器人数据。

## 前置知识与输入

- [坐标系与变换](../foundations/05-coordinate-transform.md)
- [MuJoCo 基础](../foundations/09-mujoco-basics.md)
- [数据集与训练闭环](../foundations/10-dataset-and-training.md)
- 输入：任务描述、环境参数、初始状态分布、控制频率、成功/失败条件和随机种子。

## Pipeline

| 阶段 | 关键动作 | 输出/检查 |
|---|---|---|
| 1. Task spec | 定义 observation、action、reward、termination | 维度、单位、坐标系、时间戳契约 |
| 2. Simulator | 建立 reset/step/render 和确定性种子 | 相同 seed 可重放 |
| 3. Expert/teleop | 生成成功与失败示范 | 行为分布和覆盖范围 |
| 4. Episode writer | 保存 transition、metadata、版本 | 无越界值、无错位帧 |
| 5. Dataset QA | 统计成功率、长度、缺失率、分布 | 数据卡与异常清单 |
| 6. Consumer test | 用训练 dataloader 读取一个 batch | shape、dtype、归一化一致 |

## 运行与产物

```bash
python scripts/run_pipeline.py --run simulation-data
```

入口：[unified_pushcube_env.py](../../examples/unified_pushcube_env.py)。当前 smoke test 输出专家成功率、观测维度和 RGB 渲染形状。生产数据还应保存 `episode_id`、seed、环境/代码版本、校准参数、成功标签和终止原因。

## 验收门槛

- 固定 seed 重跑得到相同初态和终止逻辑。
- observation/action 的维度、单位和频率被记录并由断言保护。
- 同时抽查成功、任务失败、超时与传感器异常轨迹。
- 训练端能够无手工修补地读取数据；划分按场景或 episode，避免帧级泄漏。

常见失败：把渲染帧率当控制频率、相机和关节时间戳错位、只保存成功示范、重置后残留上一回合状态。
