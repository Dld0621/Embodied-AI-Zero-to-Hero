# 强化学习与后训练 / RL and Post-training

## English contract

- **Objective:** improve or adapt a policy through interaction while preserving task competence and safety constraints.
- **Inputs:** MDP definition, bounded observation/action schemas, reward terms, reset logic, baseline policy, rollout budget, and evaluation seeds.
- **Stages:** contract → reward audit → baseline/BC initialization → rollout → update → evaluation → regression gate.
- **Acceptance:** report return, success, stability, intervention, and regression against the pre-training baseline across fixed seeds. Reward increase without task improvement is not success.
- **Evidence:** the included PPO path is a teaching-scale smoke test and currently underperforms the BC initialization. Apply the [validation policy](../VALIDATION.md).

## 目标与边界

把任务写成可审计 MDP，通过 PPO 等算法优化闭环回报，并把 RL 用作已有策略的受控后训练手段。Smoke test 仅检查更新链路，不说明策略已经收敛。

## 前置知识与输入

- [控制基础](../foundations/08-control-basics.md)、[概率与优化](../foundations/11-probability-and-optimization.md)
- [RL 基础](../06-rl-fundamentals-for-vla.md)、[RL Zero-to-One](../14-rl-zero-to-one.md)
- 输入：状态/观测、动作空间、reward、termination、约束、初态分布和 baseline policy。

<div class="dof-principle" role="group" aria-label="强化学习中MDP交互和策略更新的闭环">
  <p class="dof-principle__caption"><strong>原理图 · RL improves from consequences</strong>：策略在状态中选动作，环境返回下一状态、奖励与终止信号；整段 rollout 被用来估计 advantage，再约束性地更新策略。奖励增加仍需要和任务成功、安全约束分开核验。</p>
  <div class="dof-principle__canvas">
    <svg viewBox="0 0 860 246" role="img" aria-labelledby="rl-loop-title">
      <title id="rl-loop-title">强化学习交互循环和PPO更新</title><rect class="dof-diagram-surface" x="34" y="78" width="155" height="84" rx="16"/><text class="dof-diagram-label" x="74" y="111">environment</text><text class="dof-diagram-math" x="82" y="138">sₜ</text><path class="dof-diagram-accent" d="M202 120 H270"/><path class="dof-diagram-arrow" d="M270 120 l-10 -6 v12z"/>
      <rect class="dof-diagram-fill-violet" x="285" y="78" width="155" height="84" rx="16"/><text class="dof-diagram-label" x="328" y="111">policy πθ</text><text class="dof-diagram-math" x="340" y="138">aₜ</text><path class="dof-diagram-accent" d="M453 120 H521"/><path class="dof-diagram-arrow" d="M521 120 l-10 -6 v12z"/>
      <rect class="dof-diagram-fill-blue" x="536" y="78" width="155" height="84" rx="16"/><text class="dof-diagram-label" x="569" y="111">step / reward</text><text class="dof-diagram-math" x="562" y="138">sₜ₊₁, rₜ, done</text><path class="dof-diagram-violet" d="M613 177 C613 224 111 224 111 177"/><path class="dof-diagram-arrow-violet" d="M111 177 l-7 11 h13z"/><text class="dof-diagram-note" x="186" y="216">collect rollout → advantage → clipped update of θ</text>
      <rect class="dof-diagram-fill-warn" x="720" y="92" width="108" height="56" rx="14"/><text class="dof-diagram-label" x="743" y="117">evaluate</text><text class="dof-diagram-note" x="735" y="137">success + safety</text>
    </svg>
  </div>
</div>

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
