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

<div class="dof-principle" role="group" aria-label="世界模型的模型预测控制重规划机制">
  <p class="dof-principle__caption"><strong>原理图 · Plan many futures, execute only the first action</strong>：MPC 用世界模型比较多组未来动作序列，但只执行当前最优序列的第一步；获取新观测后再次规划。这能限制模型滚动误差，区别于一次规划后盲目开环执行。</p>
  <div class="dof-principle__canvas">
    <svg viewBox="0 0 860 246" role="img" aria-labelledby="mpc-title">
      <title id="mpc-title">世界模型预测候选动作序列并以MPC方式重新规划</title><rect class="dof-diagram-surface" x="27" y="75" width="151" height="93" rx="16"/><text class="dof-diagram-label" x="57" y="108">current state</text><text class="dof-diagram-math" x="84" y="135">sₜ</text><path class="dof-diagram-accent" d="M192 121 H259"/><path class="dof-diagram-arrow" d="M259 121 l-10 -6 v12z"/>
      <rect class="dof-diagram-fill-blue" x="275" y="56" width="213" height="130" rx="18"/><text class="dof-diagram-title" x="312" y="90">world model</text><text class="dof-diagram-note" x="307" y="117">roll out candidate futures</text><text class="dof-diagram-math" x="319" y="145">ŝₜ₊₁, ŝₜ₊₂, …</text><path class="dof-diagram-accent" d="M502 121 H564"/><path class="dof-diagram-arrow" d="M564 121 l-10 -6 v12z"/>
      <rect class="dof-diagram-surface" x="579" y="56" width="127" height="130" rx="18"/><text class="dof-diagram-label" x="607" y="88">score cost</text><path class="dof-diagram-violet" d="M604 111 H676 M604 130 H655 M604 149 H631"/><text class="dof-diagram-note" x="604" y="171">choose best</text><path class="dof-diagram-accent" d="M720 121 H775"/><path class="dof-diagram-arrow" d="M775 121 l-10 -6 v12z"/><text class="dof-diagram-math" x="742" y="101">aₜ*</text>
      <path class="dof-diagram-violet" d="M790 193 C790 232 101 232 101 188"/><path class="dof-diagram-arrow-violet" d="M101 188 l-7 11 h13z"/><text class="dof-diagram-note" x="278" y="225">observe sₜ₊₁, discard the old plan, and replan</text>
    </svg>
  </div>
</div>

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
