# 世界模型与规划 / World Model and Planning

## English contract

- **Objective:** learn transition dynamics, measure compounding rollout error, and evaluate whether planning improves closed-loop task success.
- **Inputs:** state-action-next-state transitions, rewards or costs, terminal flags, horizons, constraints, and held-out episodes.
- **Stages:** transition dataset → one-step model → multi-step rollout → planner → closed-loop evaluation → error analysis.
- **Acceptance:** report validation loss and horizon-conditioned error, then evaluate the planner on unseen initial states. Low one-step loss never substitutes for task success.
- **Evidence:** the local model and planning path are smoke-tested at teaching scale. Apply the [validation policy](../VALIDATION.md).

## 目标与边界

学习环境转移 `p(s_{t+1}, r_t | s_t, a_t)`，再把短期预测用于候选动作评估或 MPC。预测误差低不自动等于控制成功；模型学习与规划器必须分别验证。

## 前置知识与输入

- [概率与优化](../foundations/11-probability-and-optimization.md)
- [VLA 中的世界模型](../07-world-models-for-vla.md)、[World Model Zero-to-One](../15-world-model-zero-to-one.md)
- 输入：连续 transition、动作、奖励/成本、终止信号、时间间隔与环境上下文。

## Pipeline

| 阶段 | 关键动作 | 输出/检查 |
|---|---|---|
| 1. Transition data | 保留 episode 边界并覆盖策略会访问的状态 | train/val/test transition |
| 2. Dynamics model | 预测下一状态、奖励或潜变量 | 单步误差与校准 |
| 3. Rollout audit | 自回归多步 rollout | 误差随 horizon 曲线 |
| 4. Planner | random shooting/CEM/MPC 搜索动作序列 | 候选成本和首动作 |
| 5. Closed loop | 执行首动作、重新观测、再规划 | 任务成功率与规划时延 |
| 6. Shift audit | 在未见初态和动作分布测试 | OOD 退化与不确定性 |

## 运行与产物

```bash
python scripts/run_pipeline.py --run world-model-planning
python scripts/run_pipeline.py --run world-model-planning --full
```

模型入口：[unified_pushcube_wm.py](../../examples/unified_pushcube_wm.py)；MPC 入口：[unified_pushcube_wm_mpc.py](../../examples/unified_pushcube_wm_mpc.py)。先训练得到 `pushcube_wm.pt` 和 `wm_results.json`，再显式把 checkpoint 交给规划器。

## 验收门槛

- 同时报告单步与多步误差，不能只用训练 loss。
- 与无模型策略、随机规划和专家策略比较任务成功率。
- horizon、候选数、规划频率和时延完整记录。
- 当不确定性或预测偏差超阈值时缩短 horizon、切换保守策略或停止。

常见失败：transition 穿越 episode 边界、只收专家数据导致规划分布外、长 rollout 漂移、planner 利用模型漏洞。
