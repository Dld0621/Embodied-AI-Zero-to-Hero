# 强化学习与后训练 / RL and Post-training

## 目标与边界

把任务写成可审计 MDP，通过 PPO 等算法优化闭环回报，并把 RL 用作已有策略的受控后训练手段。Smoke test 仅检查更新链路，不说明策略已经收敛。

## 前置知识与输入

- [控制基础](../foundations/08-control-basics.md)、[概率与优化](../foundations/11-probability-and-optimization.md)
- [RL 基础](../06-rl-fundamentals-for-vla.md)、[RL Zero-to-One](../14-rl-zero-to-one.md)
- 输入：状态/观测、动作空间、reward、termination、约束、初态分布和 baseline policy。

## Pipeline

| 阶段 | 关键动作 | 输出/检查 |
|---|---|---|
| 1. MDP audit | 区分终止与截断，检查可观测性 | MDP 契约和最小单元测试 |
| 2. Reward design | 主任务、进度、能耗和约束分项记录 | reward decomposition |
| 3. Baselines | random、scripted、BC policy | 可解释下限/上限 |
| 4. Rollout | 收集固定 horizon 轨迹 | return、advantage、mask |
| 5. PPO update | clipped objective、value、entropy | KL、clip fraction、梯度 |
| 6. Evaluation | 无探索评估多 seed | 成功率、回报、违规率 |
| 7. Regression | 保存配置与 checkpoint，重跑旧场景 | 退化报告 |

## 运行与产物

```bash
python scripts/run_pipeline.py --run rl-post-training
python scripts/run_pipeline.py --run rl-post-training --full
```

入口：[unified_pushcube_rl.py](../../examples/unified_pushcube_rl.py)。默认完整路线使用 PPO，输出 `pushcube_ppo_policy.pt` 与 `rl_results.json` 到 `results/pipelines/rl/`。

## 验收门槛

- scripted expert 能拿到高回报，random policy 明显更差，证明 reward 基本可用。
- 至少报告多 seed 均值、离散程度、评估 episode 数和环境版本。
- 监控 KL、clip fraction、value loss 和 entropy，异常更新必须停止或回滚。
- 后训练策略必须与原策略比较成功率、泛化和约束违规率。

常见失败：reward hacking、把超时当成功终止、只看回报不看任务成功、训练和评估使用同一随机轨迹。
