# Three Capstones: From Engineering Skill to Independent Research

**English · [简体中文](capstone-cn.md)**

A capstone integrates knowledge, engineering, evaluation, and communication into reviewable work. It is not a demo video or a collection of connected repositories. Any “expert” conclusion remains bounded to the reviewed task and evidence.

## C0 · Robot-loop diagnosis

- **Prerequisites:** M00–M04.
- **Task:** build or modify a small simulated loop connecting observation, state, control, and metrics.
- **Required fault:** inject a frame, timing, sensor, or actuator failure.
- **Deliver:** nominal and failed traces, logs, fault tree, safe stop, and reproduction receipt.
- **Pass:** 80/100 and at least one independent reviewer.

C0 demonstrates robot-loop engineering, not learned-policy quality.

## C1 · Closed-loop learning system

- **Prerequisites:** goal-required modules from M00–M07, plus M10 and M11.
- **Task:** implement a simple baseline and a learning method under the same task, data, and evaluation contract.
- **Required comparison:** a no-learning/rule or state-only baseline and one learned policy.
- **Required tests:** held-out initial states, perturbations, failure stages, and at least one counterfactual ablation.
- **Deliver:** dataset card, training receipt, checkpoint, closed-loop result, negative results, and model card.
- **Pass:** 85/100 and at least one independent reviewer.

C1 demonstrates a controlled robot-learning system, not general embodied intelligence.

## C2 · Independent reproduction and research extension

- **Prerequisites:** M00–M11 all passed.
- **A — Independent reproduction:** freeze the method, data, budget, and evaluation before reproducing a credible baseline; retain negative outcomes and differences.
- **B — Original hypothesis:** state a single-variable hypothesis that evidence can falsify.
- **C — Controlled extension:** compare baseline, change, and required ablations under matched budgets with multiple seeds and uncertainty.
- **D — Distribution shift:** test at least two axes among object/state, viewpoint, language, scene, latency, and embodiment.
- **E — System loop:** show how prediction/planning changes executed action and report safety, recovery, and stopping.
- **Deliver:** report, code, environment receipt, datasheet, raw metrics, checkpoints, video/traces, failure taxonomy, risk register, and defense material.
- **Pass:** 90/100 and at least two independent reviewers.

A successful reproduction alone cannot pass C2; a new method without a credible baseline cannot pass either.

## Common capstone score (100 points)

| Dimension | Points | Full-credit condition |
|---|---:|---|
| Question and success contract | 10 | Task, I/O, metrics, boundary, and stop conditions are testable |
| Prerequisites and system design | 10 | Architecture choices and failure propagation are explained |
| Data and interfaces | 10 | Schema, synchronization, split, units, and embodiment identity are traceable |
| Baseline quality | 10 | Simple, budget-matched, and independently reproducible |
| Method implementation | 10 | Learner owns the key contribution and passes unit/interface checks |
| Closed-loop evaluation | 10 | Multi-episode task outcomes, stage metrics, and failures are complete |
| Ablation, OOD, and statistics | 10 | Counterfactuals, two shift axes, seeds, and uncertainty are adequate |
| Systems, safety, and recovery | 10 | Latency, limits, watchdog, recovery, and authorization boundaries are clear |
| Reproducible evidence | 10 | Commit, environment, data, config, commands, and raw artifacts are complete |
| Explanation and defense | 10 | Handles counterexamples and distinguishes fact, inference, hypothesis, and unknown |

Any [critical failure](assessment.md#critical-failures-any-one-means-no-pass) overrides the total score.

## Example C2 questions

| Direction | Falsifiable question | Required baseline and ablation |
|---|---|---|
| VLA | Does language change the correct action under the same visual state? | State/vision BC; correct, shuffled, and missing language |
| WAM | Does action-conditioned future prediction improve planning utility? | Model-free policy and WM+MPC; action swap, no action, no lookahead |
| Dexterity | Does a retargeting/contact representation improve task retention rather than only geometry? | IK/optimization baseline; geometry, contact, lift, and retention stages |
| Navigation/agents | Does recovery/replanning improve perturbed-task success without more collisions? | Planner without recovery; perturbation and recovery-component ablations |
| Sim-to-Real | Which identification or randomization factor reduces a declared transfer gap? | No-adaptation policy; single-factor randomization, HIL, shadow mode |

## Review and revision

Use the [capstone review form](../learner/templates/capstone-review.md). A reviewer must provide:

1. one counterexample that could overturn the conclusion;
2. one raw artifact personally checked;
3. the most serious unknown risk;
4. a `pass`, `revision_required`, or `blocked` decision.

Revision must preserve earlier results. A new experiment card records the reason and the single changed variable.

## Hardware boundary

A capstone score never authorizes robot motion. Hardware requires site-, device-, operator-, and task-specific approval plus manufacturer limits and stop procedures. Learners without hardware may complete a simulation capstone, but must label the evidence level as simulation.
