# Failure Report / 失败报告

## Failure identity / 失败身份

- Experiment ID / 实验 ID:
- First failing timestamp / 首次失败时间:
- Code, data, checkpoint, environment / 代码、数据、检查点、环境:
- Exact command and exit code / 精确命令与退出码:

## Expected versus observed / 预期与实际

- Expected behavior / 预期行为:
- Observed behavior / 实际行为:
- Earliest divergent stage / 最早偏离阶段:
- Raw log, trace, image, or video / 原始日志、轨迹、图片或视频:

## Stage isolation / 阶段定位

| Stage / 阶段 | Evidence checked / 已检查证据 | Status / 状态 |
|---|---|---|
| Task and reset / 任务与复位 |  | unknown |
| Observation and timing / 观测与时序 |  | unknown |
| State and frames / 状态与坐标系 |  | unknown |
| Policy or planner / 策略或规划器 |  | unknown |
| Action interface / 动作接口 |  | unknown |
| Controller and plant / 控制器与被控对象 |  | unknown |
| Metric and termination / 指标与终止 |  | unknown |

Use `confirmed`, `rejected`, or `unknown`; do not turn a plausible cause into a fact.

使用 `confirmed`、`rejected` 或 `unknown`，不要把可能原因写成已证实事实。

## Hypotheses / 假设

| Candidate cause / 候选原因 | Supporting evidence / 支持证据 | Contradicting evidence / 反证 | Next test / 下一测试 |
|---|---|---|---|
|  |  |  |  |

## Decision / 决定

- Fixed, retained negative result, or blocked / 已修复、保留负结果或阻塞:
- What changed / 改变内容:
- Regression test / 回归测试:
- Remaining uncertainty / 剩余不确定性:
