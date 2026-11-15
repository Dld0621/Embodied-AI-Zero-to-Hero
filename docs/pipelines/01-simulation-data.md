# 仿真与数据生成 / Simulation and Data Generation

> **逐点图解 / Concept close-ups：**[任务定义、复位、扰动与随机化](../knowledge-atlas/sim-task-randomization/index.md)。每个小点配原理、算例、图、自测；这是中文细解，保留英文术语。

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

<div class="dof-principle" role="group" aria-label="仿真 episode 中 transition 的数据契约">
  <p class="dof-principle__caption"><strong>原理图 · A dataset is a sequence of aligned transitions</strong>：一条轨迹不是孤立图片的集合，而是同一控制时钟下的 <code>(observation, action, reward, next observation, termination)</code> 链。episode 边界必须显式保存，不能让终点接到下一回合的起点。</p>
  <div class="dof-principle__canvas">
    <svg viewBox="0 0 860 238" role="img" aria-labelledby="episode-title">
      <title id="episode-title">仿真 episode 的时间序列 transition 契约</title><text class="dof-diagram-title" x="32" y="39">One transition at control time t</text><rect class="dof-diagram-fill-blue" x="32" y="83" width="145" height="64" rx="14"/><text class="dof-diagram-label" x="66" y="111">observation</text><text class="dof-diagram-math" x="87" y="133">oₜ</text><path class="dof-diagram-accent" d="M190 115 H245"/><path class="dof-diagram-arrow" d="M245 115 l-10 -6 v12z"/>
      <rect class="dof-diagram-fill-violet" x="260" y="83" width="145" height="64" rx="14"/><text class="dof-diagram-label" x="301" y="111">action</text><text class="dof-diagram-math" x="316" y="133">aₜ</text><path class="dof-diagram-accent" d="M418 115 H473"/><path class="dof-diagram-arrow" d="M473 115 l-10 -6 v12z"/>
      <rect class="dof-diagram-surface" x="488" y="83" width="145" height="64" rx="14"/><text class="dof-diagram-label" x="525" y="111">env.step</text><text class="dof-diagram-note" x="508" y="133">physics + task logic</text><path class="dof-diagram-accent" d="M646 115 H701"/><path class="dof-diagram-arrow" d="M701 115 l-10 -6 v12z"/>
      <rect class="dof-diagram-fill-good" x="716" y="83" width="112" height="64" rx="14"/><text class="dof-diagram-math" x="740" y="111">oₜ₊₁</text><text class="dof-diagram-note" x="733" y="133">rₜ, done</text>
      <path class="dof-diagram-line" d="M65 190 H794"/><circle class="dof-diagram-fill-blue" cx="123" cy="190" r="6"/><circle class="dof-diagram-fill-blue" cx="270" cy="190" r="6"/><circle class="dof-diagram-fill-blue" cx="417" cy="190" r="6"/><circle class="dof-diagram-fill-warn" cx="562" cy="190" r="6"/><circle class="dof-diagram-fill-good" cx="708" cy="190" r="6"/><text class="dof-diagram-note" x="32" y="219">same timestamp convention · fixed units and frames · explicit terminal boundary</text>
    </svg>
  </div>
</div>

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
