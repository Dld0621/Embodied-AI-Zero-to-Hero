# 具身推理与任务规划 / Embodied Reasoning and Task Planning

## English contract

- **Objective:** convert an instruction and world state into a typed, inspectable plan that can execute skills, observe outcomes, and replan after failure.
- **Inputs:** task instruction, grounded entities, available skills, preconditions/effects, safety constraints, and feedback events.
- **Stages:** parse → ground → plan → validate → execute skill → observe → replan or terminate.
- **Acceptance:** measure parse validity, grounding accuracy, subgoal completion, replan count, constraint violations, and final task success on held-out scenarios.
- **Evidence:** the included rule-based path is interface-tested; it does not validate general language-model reasoning. Apply the [validation policy](../VALIDATION.md).

## 目标与边界

把自然语言目标转成可执行、可监控、可重规划的技能序列。规划器只负责决定“做什么”；技能控制器、世界状态估计和安全监督决定“是否能安全做到”。

## 前置知识与输入

- [具身推理与规划](../27-embodied-reasoning-and-planning.md)
- [机器人系统与安全](../foundations/13-robot-systems-and-safety.md)
- 输入：用户目标、当前场景状态、对象与技能目录、前置条件、禁区、资源和完成判据。

## Pipeline

| 阶段 | 关键动作 | 输出/检查 |
|---|---|---|
| 1. Grounding | 解析实体、关系和约束 | grounded goal |
| 2. Task planning | 分解 locate/approach/grasp/move/place | typed `TaskPlan` |
| 3. Validation | 检查技能存在、参数合法、前置条件 | executable plan |
| 4. Skill execution | 调用感知与控制技能 | skill status/telemetry |
| 5. Monitoring | 判断完成、失败、超时和安全事件 | structured feedback |
| 6. Replanning | 更新世界状态并恢复或重规划 | revised plan |
| 7. Audit | 保存输入、计划、工具调用与结果 | trace 与错误分类 |

## 运行与产物

```bash
python scripts/run_pipeline.py --run embodied-reasoning
```

入口：[rule_based_planner.py](../../examples/robot_foundation_models/planners/rule_based_planner.py)。当前示例验证语言到 `TaskPlan/SubGoal` 的结构化接口；它不执行机器人技能。

## 验收门槛

- 计划只能调用白名单技能，参数必须通过 schema 验证。
- 每个 subgoal 有前置条件、成功条件、超时与失败恢复策略。
- 在含糊指令、对象不存在、技能失败和场景变化时测试。
- 报告计划解析成功率、子目标完成率、重规划次数和最终任务成功率。

常见失败：LLM 生成不存在的技能、只生成文本不生成类型化计划、执行失败后盲目重试、计划日志无法复现。
