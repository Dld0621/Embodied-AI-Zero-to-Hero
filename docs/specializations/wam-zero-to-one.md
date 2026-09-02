# WAM Zero to One: From World Models to Joint Video-Action Policies

**English · [简体中文](wam-zero-to-one-cn.md) · [Specialization home](README.md)**

World Action Models are an emerging research family, not a settled recipe. This track builds the prerequisites in the correct order: dynamics, rollout evaluation, planning, inverse dynamics, joint future/action learning, and only then large video-model backbones.

## 1. Terminology and evidence boundary

Use these terms precisely:

| System | Learned object | How actions are selected | WAM? |
|---|---|---|---|
| Behavior-cloned policy/VLA | $p(a_t\mid o_{\le t},\ell)$ | Direct policy output | No |
| Action-conditioned world model | $p(o_{t+1}\mid o_{\le t},a_t)$ or latent equivalent | Separate planner/policy | Not by itself |
| Latent model-based RL/MPC | Dynamics plus reward/value/policy | Search, optimization, or imagined policy learning | Important baseline, not narrow WAM |
| Joint image-action model | Future image/latent and action tokens in one framework | Joint model emits actions | WAM family |
| Joint video-action generative model | Aligned future video and action chunks | Generated future and inverse/action component | WAM family |

[WorldVLA](https://arxiv.org/abs/2506.21539) describes an autoregressive action-world model that unifies image and action understanding/generation. [DreamZero](https://arxiv.org/abs/2602.15922), released in 2026, uses “World Action Model” for a video-model-based system that jointly predicts future video and actions. The label is recent and different papers factorize the model differently; do not retroactively rename every world model as a WAM.

## 2. Exit contract

You complete this track only when you can:

1. implement and audit a one-step dynamics model;
2. measure compounding rollout error by prediction horizon;
3. use MPC/search and show whether it improves closed-loop task outcomes;
4. distinguish forward dynamics, inverse dynamics, policy learning, and joint modeling;
5. implement a small future/action joint model on a controlled task;
6. measure video/action alignment and control utility separately;
7. compare WAM against matched policy and world-model baselines;
8. explain the data, compute, latency, and safety reasons for or against WAM.

## 3. Mathematical progression

### 3.1 Markov and partially observed setup

In state space, a learned dynamics model may approximate

$$
p_\theta(s_{t+1},r_t,d_t\mid s_t,a_t),
$$

where $d_t$ is termination. In pixel control, the true state is hidden, so history is encoded into a latent state:

$$
z_t=e_\theta(o_{\le t}), \qquad
p_\theta(z_{t+1}\mid z_t,a_t).
$$

One-step fit is not sufficient because the model is recursively evaluated on its own predictions during planning.

### 3.2 Action-conditioned world model + MPC

Given candidate action sequences $A^{(i)}=[a_t,\ldots,a_{t+H-1}]$, roll out the model, score cost or value, execute only the first action of the best candidate, re-observe, and replan. CEM, random shooting, gradient optimization, or a learned proposal policy can produce candidates.

This modular baseline is essential because it separates:

- representation error;
- dynamics error;
- reward/value error;
- planner/search error;
- controller and system error.

### 3.3 Joint world-action model

A generic joint objective models

$$
p_\theta(O^+_t,A_t\mid h_t,\ell),
$$

where $h_t$ is past observation/state context, $O^+_t$ is a future visual sequence, and $A_t$ is the aligned action chunk. A useful conceptual factorization is

$$
p_\theta(O^+_t,A_t\mid h_t,\ell)
=p_\theta(O^+_t\mid h_t,\ell)
\,p_\theta(A_t\mid h_t,O^+_t,\ell).
$$

The first term is future/world prediction. The second acts like inverse dynamics: infer robot actions aligned with the current and predicted visual evolution. This is a teaching factorization, not a claim that all WAM implementations use identical modules or losses.

<div class="dof-principle" role="group" aria-label="World model and world action model comparison">
  <p class="dof-principle__caption"><strong>Principle · Prediction becomes a policy only through aligned actions.</strong> A conventional world model predicts consequences under proposed actions. A WAM family model learns future world evolution and aligned actions together. Plausible video without executable action alignment is not control.</p>
  <div class="dof-principle__canvas">
    <svg viewBox="0 0 940 300" role="img" aria-labelledby="wam-track-diagram-title">
      <title id="wam-track-diagram-title">Comparison of modular world model planning and joint world-action modeling</title>
      <text class="dof-diagram-title" x="28" y="34">modular baseline</text>
      <rect class="dof-diagram-surface" x="28" y="56" width="133" height="62" rx="12"/><text class="dof-diagram-label" x="57" y="83">current context</text><text class="dof-diagram-note" x="59" y="103">history + goal</text>
      <path class="dof-diagram-accent" d="M175 87 H222"/><path class="dof-diagram-arrow" d="M222 87 l-10 -6 v12z"/>
      <rect class="dof-diagram-fill-blue" x="238" y="56" width="146" height="62" rx="12"/><text class="dof-diagram-label" x="269" y="83">candidate actions</text><text class="dof-diagram-note" x="273" y="103">planner / search</text>
      <path class="dof-diagram-accent" d="M398 87 H445"/><path class="dof-diagram-arrow" d="M445 87 l-10 -6 v12z"/>
      <rect class="dof-diagram-fill-violet" x="461" y="56" width="151" height="62" rx="12"/><text class="dof-diagram-label" x="492" y="83">world rollout</text><text class="dof-diagram-note" x="489" y="103">predict + score</text>
      <path class="dof-diagram-accent" d="M626 87 H673"/><path class="dof-diagram-arrow" d="M673 87 l-10 -6 v12z"/>
      <rect class="dof-diagram-surface" x="689" y="56" width="144" height="62" rx="12"/><text class="dof-diagram-label" x="719" y="83">first action</text><text class="dof-diagram-note" x="715" y="103">execute + replan</text>
      <path class="dof-diagram-dash" d="M28 148 H910"/>
      <text class="dof-diagram-title" x="28" y="181">joint world-action family</text>
      <rect class="dof-diagram-surface" x="28" y="203" width="166" height="66" rx="12"/><text class="dof-diagram-label" x="62" y="231">past context</text><text class="dof-diagram-note" x="55" y="251">video · state · goal</text>
      <path class="dof-diagram-violet" d="M209 236 H272"/><path class="dof-diagram-arrow-violet" d="M272 236 l-10 -6 v12z"/>
      <rect class="dof-diagram-fill-violet" x="288" y="196" width="265" height="80" rx="14"/><text class="dof-diagram-title" x="337" y="227">shared world-action model</text><text class="dof-diagram-note" x="337" y="250">joint tokens or coupled generative objective</text>
      <path class="dof-diagram-violet" d="M568 218 H631"/><path class="dof-diagram-arrow-violet" d="M631 218 l-10 -6 v12z"/>
      <path class="dof-diagram-violet" d="M568 255 H631"/><path class="dof-diagram-arrow-violet" d="M631 255 l-10 -6 v12z"/>
      <rect class="dof-diagram-fill-blue" x="647" y="190" width="174" height="57" rx="12"/><text class="dof-diagram-label" x="684" y="218">future world</text><text class="dof-diagram-note" x="680" y="237">video / latent</text>
      <rect class="dof-diagram-fill-good" x="647" y="254" width="174" height="36" rx="10"/><text class="dof-diagram-label" x="682" y="278">aligned actions</text>
    </svg>
  </div>
</div>

## 4. Data contract

A WAM needs everything required by a VLA plus dense temporal alignment:

```text
past video/state context
future video or future latent targets
aligned action chunks and action semantics
instruction/goal and embodiment metadata
camera calibration, frame rate, exposure and dropped-frame markers
episode, reset, intervention, success and terminal boundaries
```

### 4.1 Action-video alignment

Let image timestamps be $t^I_k$, command timestamps $t^a_j$, and measured state timestamps $t^q_m$. Define which command caused which visual transition after camera, network, controller, and actuator delay. A one-frame offset may train an inverse model to predict the previous or next action instead of the causal action.

Audit alignment by applying a known action pulse in simulation or a safe recorded setup, then measuring the first observable state and image response. Retain the estimated delay and uncertainty.

### 4.2 Video-only data

Video-only data can pretrain visual dynamics or motion priors. It does not provide target-robot action labels. Using it for control requires an inverse-dynamics bridge, action annotation, embodiment adaptation, or another explicitly validated assumption. Do not describe video volume as robot-action supervision.

### 4.3 Dataset splits

Separate at least:

- task/verb shift;
- object shift;
- scene/environment shift;
- camera/viewpoint shift;
- motion/trajectory shift;
- embodiment shift.

“Zero-shot” must name which of these axes is held out. Unseen text with a seen motion is not the same as unseen physical motion.

## 5. Algorithms by component

### 5.1 Future representation

| Representation | Predicts | Strength | Limitation |
|---|---|---|---|
| Physical state | Joint/object state | Interpretable and compact | Requires state estimation and task-specific schema |
| Reconstructed pixels | Future RGB | Dense visual target | Expensive; wastes capacity on irrelevant detail |
| Compressed video latent | VAE/token latent | Scales generative modeling | Decoder artifacts and latent semantics |
| Predictive feature/JEPA latent | Future representation | Avoids pixel reconstruction | Harder to inspect; planning cost must be learned |
| Object/keypoint latent | Structured scene elements | Data-efficient geometry | Detector/tracker failure becomes model failure |

Choose the representation that preserves controllable task variables. A prettier future frame is not automatically a better planning state.

### 5.2 Dynamics model

| Family | Core update | Use when | Main risk |
|---|---|---|---|
| Deterministic MLP/RNN | $\hat z_{t+1}=f(z_t,a_t)$ | First baseline, low-dimensional state | Averages stochastic futures |
| Stochastic state-space model | Prior/posterior latent dynamics | Partial observability and uncertainty | Posterior collapse, calibration |
| Autoregressive token model | Predict future tokens sequentially | Discrete image/action tokenization | Long-sequence error and latency |
| Diffusion/flow future model | Iterative conditional generation | Multimodal visual futures | Compute, sampling, controllability |
| Decoder-free latent model | Predict reward/value/latent features | Planning rather than visualization | Can hide physically wrong features |

[TD-MPC2](https://arxiv.org/abs/2310.16828) is a useful decoder-free latent planning baseline. [V-JEPA 2-AC](https://arxiv.org/abs/2506.09985) is a useful example of action-conditioned latent predictive planning. Neither should be called WAM solely because it supports actions and prediction.

### 5.3 Planner for the modular baseline

- **Random shooting:** easiest correctness baseline; poor scaling.
- **CEM:** iteratively refits a distribution to low-cost action sequences; strong teaching/default MPC baseline.
- **Gradient-based planning:** efficient when model and cost are smooth; can exploit gradients/model errors.
- **MCTS/tree search:** useful for discrete or structured decisions; expensive in continuous high-dimensional action.
- **Learned proposal/value:** reduces search but couples planner quality to learned priors.

Always compare planning against random actions, an expert/behavior policy when available, and the same policy without world-model lookahead.

### 5.4 Inverse dynamics

An inverse model estimates actions from consecutive states or visual futures:

$$
q_\phi(a_t\mid o_{\le t},o_{t+1:t+H},\ell).
$$

This provides the bridge from “what the future may look like” to “which command may cause it.” Ambiguous inverse dynamics are common: multiple actions can yield similar visual change, and hidden force/contact can be invisible. A generative action distribution may be necessary, but force/state sensing may be the actual missing information.

### 5.5 Autoregressive joint image-action modeling

[WorldVLA](https://arxiv.org/abs/2506.21539) unifies image understanding/generation and action generation in an autoregressive framework. This family tokenizes or embeds image and action sequences, uses attention masks to control dependencies, and optimizes token prediction. Key design choices are:

- interleaved versus separate image/action streams;
- action-token codebook and round-trip error;
- causal attention mask;
- teacher forcing versus free-running training;
- chunk length and error propagation.

### 5.6 Joint video-action diffusion or flow

For video latent $Y$ and action chunk $A$, a simplified coupled flow objective is

$$
\begin{aligned}
\mathcal L ={}& \lambda_v\|v^Y_\theta(Y_\tau,A_\tau,c)-u^Y_\tau\|^2 \\
&+\lambda_a\|v^A_\theta(Y_\tau,A_\tau,c)-u^A_\tau\|^2.
\end{aligned}
$$

The two modalities may share a backbone while using different embeddings, heads, noise schedules, or loss weights. [DreamZero](https://arxiv.org/abs/2602.15922) is a frontier example built on a pretrained video diffusion backbone and trained to predict future video and actions. Its reported scale, optimization, transfer, and robot results are properties of that system, not guarantees of the WAM family.

### 5.7 Closed-loop inference

1. Buffer a bounded history.
2. Generate a future/action chunk.
3. Validate action shape, units, limits, freshness, and embodiment ID.
4. Execute only a safe prefix.
5. Re-observe and compare the real transition with the predicted future.
6. Replan, fall back, or stop when disagreement/uncertainty exceeds a validated boundary.

Open-loop generation of a visually coherent video is not robot task success.

## 6. How to choose between VLA, world model, and WAM

| Need | First family | Why |
|---|---|---|
| Semantic task choice, no need to render futures | VLA | Direct language-to-action is simpler |
| One task with precise feedback | Task policy/ACT/Diffusion + controller | Large semantic/video models may not solve precision |
| Counterfactual action search | World model + MPC | Modular errors and planning utility are measurable |
| Learn from video motion priors and aligned actions | Small joint WAM experiment | Tests whether dense future supervision helps |
| Heterogeneous tasks/environments/embodiments at scale | WAM research after baselines | Potentially uses broad video dynamics; very high burden |

Choose a joint WAM only if all are true:

- future visual evolution is part of the hypothesis, not decorative output;
- video and actions are aligned well enough to train inverse/joint dynamics;
- data covers the intended motion and shift axes;
- compute supports video training and closed-loop inference experiments;
- a policy baseline and modular world-model baseline already exist;
- success can be measured on tasks, not only video metrics.

Otherwise, prefer the smallest policy or world-model baseline that answers the question.

```bash
python scripts/select_vla_wam_algorithm.py \
  --goal future-video-and-action \
  --compute cluster \
  --data heterogeneous \
  --latency soft
```

## 7. Zero-to-one build sequence

| Stage | Build | Promotion gate |
|---:|---|---|
| 0 | State transition dataset | Episode boundaries, time alignment, units, and splits pass |
| 1 | One-step deterministic dynamics | Tiny-set overfit and held-out one-step error pass |
| 2 | Multi-step rollout | Error-by-horizon and stability are reported |
| 3 | CEM/random-shooting MPC | Planner improves task outcome over matched baselines |
| 4 | Visual/latent world model | Latent preserves task variables and predicts under actions |
| 5 | Inverse-dynamics action model | Action/video alignment and ambiguity are measured |
| 6 | Small joint future/action model | Joint model beats or explains failure against modular baseline |
| 7 | Shift evaluation | Task, scene, motion, camera, and embodiment shifts separated |
| 8 | Systems optimization | End-to-end rate, memory, stale context, and fallback pass |

Repository starting path:

```bash
python scripts/run_knowledge_map.py --path-to planning-world-models
python scripts/run_pipeline.py --run world-model-planning --dry-run
python scripts/run_pipeline.py --run world-model-planning
```

The local pipeline is a low-dimensional teaching world model + planner. It is a prerequisite baseline, not a DreamZero or WorldVLA reproduction.

## 8. Training objective and ablation checklist

At minimum compare:

1. policy-only action loss;
2. world-only future loss;
3. joint loss with matched backbone/data;
4. joint model without language;
5. joint model with future target masked or shuffled;
6. true versus shifted action-video alignment;
7. teacher-forced versus free-running rollout;
8. equal inference-time budget.

If the joint model improves only video quality but not action alignment or task behavior, the WAM hypothesis has not passed.

## 9. Evaluation matrix

| Axis | Required measurement |
|---|---|
| World prediction | One-step and horizon-conditioned latent/video error |
| Action | Per-dimension/chunk error, bounds, multimodal samples |
| Alignment | Real action vs inferred action for the same visual transition; temporal offset sweep |
| Planning utility | Task success/progress with and without model lookahead |
| Closed loop | Success, interventions, recovery, disagreement, constraint violations |
| Generalization | Separate task, object, environment, motion, camera, embodiment axes |
| Video | Visual metric plus task-relevant state/contact consistency |
| Systems | Generation rate, end-to-end latency, memory, cache/history age |
| Statistics | Trial count, seed allocation, confidence interval, negative results |

Video metrics such as reconstruction or perceptual similarity cannot establish action correctness. Action MSE cannot establish that the predicted future is physically consistent. Task success cannot by itself reveal whether the future-model component helped. Report all three layers.

## 10. Failure localization

| Symptom | Likely checks |
|---|---|
| Sharp one-step prediction, drifting rollout | Exposure bias, missing stochasticity, action coverage, recursive rollout |
| Plausible future, wrong action | Temporal alignment, inverse-dynamics ambiguity, action frame/normalization |
| Planner exploits impossible future | Model uncertainty, OOD candidate actions, constraints, shorter horizon |
| Joint loss improves, task does not | Loss weighting, shortcut, future target relevance, execution interface |
| Slow closed loop | Video resolution/tokens, sampling steps, cache, network, action prefix length |
| Cross-embodiment failure | Camera/morphology conditioning, adapter, action feasibility, per-robot metrics |
| Precision contact failure | Missing force/tactile/state feedback, visual resolution, controller bandwidth |

## 11. Research questions worth testing

- Does joint future prediction improve action generalization under a frozen motion shift?
- Which future representation preserves contact-relevant state with the least compute?
- Does inverse dynamics benefit from multiple future samples or explicit uncertainty?
- When does a modular planner outperform direct WAM action generation at equal latency?
- How should video-only data be weighted without overwhelming action-grounded data?
- Can future/real disagreement reliably trigger fallback before task failure?
- Which transfer gains come from motion priors versus extra model/data scale?

Every comparison must control data, backbone scale where possible, action semantics, training budget, inference budget, and trial allocation.

## 12. Primary sources

- [Dreamer V3](https://arxiv.org/abs/2301.04104)
- [TD-MPC2](https://arxiv.org/abs/2310.16828)
- [V-JEPA 2](https://arxiv.org/abs/2506.09985)
- [WorldVLA](https://arxiv.org/abs/2506.21539)
- [DreamZero](https://arxiv.org/abs/2602.15922)
- [Action Images](https://arxiv.org/abs/2604.06168) — an emerging 2026 preprint using pixel-grounded action images; treat it as a research direction, not an established default.

This track explains the algorithm families and defines reproducible comparisons. The repository does not claim to reproduce the large-scale or real-robot results of these sources.
