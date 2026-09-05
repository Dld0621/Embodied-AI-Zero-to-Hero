# VLA Zero to One: Algorithms, Selection, and Evidence

> **逐点图解 / Concept close-ups：**[视觉-语言-动作策略](../knowledge-atlas/learning-vla/index.md)。每个小点配原理、算例、图、自测；这是中文细解，保留英文术语。

**English · [简体中文](vla-zero-to-one-cn.md) · [Specialization home](README.md)**

This is the canonical VLA track. Its goal is not “run a large checkpoint once,” but understand every contract between synchronized robot data and closed-loop action, implement the principal algorithm families, and know when a VLA is the wrong choice.

## 1. Exit contract

You complete this track only when you can:

1. define observation, language, state, action, timing, and episode semantics without ambiguity;
2. implement a matched behavior-cloning baseline before using a pretrained VLA;
3. derive discrete-token, regression, diffusion, and flow-matching action objectives;
4. choose an action family from task multimodality, latency, data, and compute constraints;
5. run receding-horizon closed-loop evaluation with safety filters;
6. prove whether language affects behavior through controlled ablations;
7. report task success, failures, latency, uncertainty, and distribution shifts separately.

## 2. Problem formulation

At time $t$, define the context

$$
c_t = (I_{t-k:t}^{1:V},\; q_{t-k:t},\; \ell,\; m_t),
$$

where $I^{1:V}$ are synchronized camera observations, $q$ is robot state, $\ell$ is a task instruction or goal, and $m_t$ contains masks and embodiment metadata. A VLA policy predicts an action or action chunk:

$$
\pi_\theta(A_t \mid c_t), \qquad A_t=[a_t,\ldots,a_{t+H-1}].
$$

This equation hides the most common failures. Every symbol requires a contract:

| Contract | Questions that must be answered |
|---|---|
| Images | Which cameras, exposure, crop, color order, timestamps, and missing-frame policy? |
| State | Joint or task space? Position, velocity, force? Which frame, units, ordering, and rate? |
| Language | Episode-level or step-level? Template, paraphrase, goal image, or absent? |
| Action | Absolute or delta? Joint, end-effector, gripper, velocity, or torque? Which frame and horizon? |
| Timing | Observation-to-action delay, control rate, chunk prediction rate, and execution horizon? |
| Boundary | Reset, success, failure, timeout, intervention, and padding masks? |

If these are not explicit, training loss cannot be interpreted.

> **Timing experiment:** In the [action-chunk timeline](../learning-lab.md#timing), vary prediction horizon H, executed prefix E, and inference latency independently. Predicting 16 actions does not require executing all 16. In the serial model, 100 ms inference plus four 50 ms actions gives a 300 ms cycle; the last action starts with a 250 ms old observation. Explain that calculation before considering asynchronous inference. The lab is not a measured VLA speed claim.

<div class="dof-principle" role="group" aria-label="VLA component and control-loop diagram">
  <p class="dof-principle__caption"><strong>Principle · Representation is only half the system.</strong> The policy fuses visual, language, and state context, but a robot adapter, rate contract, command bounds, and re-observation determine whether its output becomes a valid closed loop.</p>
  <div class="dof-principle__canvas">
    <svg viewBox="0 0 920 270" role="img" aria-labelledby="vla-track-diagram-title">
      <title id="vla-track-diagram-title">VLA encoders, fusion, action generator, adapter, and feedback loop</title>
      <text class="dof-diagram-title" x="26" y="35">context encoding → multimodal fusion → action distribution → bounded execution</text>
      <rect class="dof-diagram-fill-blue" x="26" y="66" width="150" height="39" rx="8"/><text class="dof-diagram-label" x="55" y="91">image history</text>
      <rect class="dof-diagram-fill-violet" x="26" y="115" width="150" height="39" rx="8"/><text class="dof-diagram-label" x="53" y="140">instruction / goal</text>
      <rect class="dof-diagram-fill-good" x="26" y="164" width="150" height="39" rx="8"/><text class="dof-diagram-label" x="56" y="189">robot state</text>
      <path class="dof-diagram-accent" d="M190 135 H250"/><path class="dof-diagram-arrow" d="M250 135 l-10 -6 v12z"/>
      <rect class="dof-diagram-surface" x="266" y="79" width="157" height="112" rx="14"/><text class="dof-diagram-title" x="306" y="112">fusion</text><text class="dof-diagram-note" x="289" y="139">attention / FiLM</text><text class="dof-diagram-note" x="291" y="160">masks + metadata</text>
      <path class="dof-diagram-accent" d="M438 135 H496"/><path class="dof-diagram-arrow" d="M496 135 l-10 -6 v12z"/>
      <rect class="dof-diagram-fill-violet" x="512" y="79" width="170" height="112" rx="14"/><text class="dof-diagram-title" x="540" y="111">action model</text><text class="dof-diagram-note" x="535" y="138">token / regression</text><text class="dof-diagram-note" x="541" y="159">diffusion / flow</text>
      <path class="dof-diagram-accent" d="M697 135 H747"/><path class="dof-diagram-arrow" d="M747 135 l-10 -6 v12z"/>
      <rect class="dof-diagram-surface" x="763" y="79" width="132" height="112" rx="14"/><text class="dof-diagram-label" x="790" y="110">adapter</text><text class="dof-diagram-note" x="784" y="138">units · bounds</text><text class="dof-diagram-note" x="782" y="159">rate · watchdog</text>
      <path class="dof-diagram-violet" d="M829 205 C829 252 104 252 104 215"/><path class="dof-diagram-arrow-violet" d="M104 215 l-7 11 h13z"/><text class="dof-diagram-note" x="361" y="252">execute a bounded prefix, re-observe, and predict again</text>
    </svg>
  </div>
</div>

## 3. Data pipeline

### 3.1 Minimum episode schema

Each step should retain:

```text
episode_id, step_id, timestamp
images[camera_name]
robot_state, state_names, state_units
action, action_names, action_units, action_frame
instruction_or_goal, task_id
success, terminal, timeout, intervention
embodiment_id, calibration_id, dataset_version
```

Store the observation that was available when the action was chosen, not a later camera frame. Split by episode—and by object/scene/task/embodiment when testing that shift—rather than randomly splitting individual frames.

### 3.2 Normalization

Fit statistics on the training split only. Record the transform and its inverse. For an action dimension $j$, a robust bounded mapping may use dataset quantiles or known physical limits, but the choice must preserve units and saturation behavior. Verify a round trip:

$$
a \xrightarrow{N} \tilde a \xrightarrow{N^{-1}} \hat a,
\qquad \|a-\hat a\| < \varepsilon.
$$

Never assume two robot datasets share action semantics because their vectors have the same length.

### 3.3 Coverage audit

Report task, object, pose, background, camera, trajectory length, failure/intervention, and action-range coverage. Language diversity is not demonstrated by many paraphrases of identical motions; physical diversity is not demonstrated by many frames from one episode.

## 4. Algorithms by component

### 4.1 Visual representation

| Choice | Use when | Main risk | Required ablation |
|---|---|---|---|
| Small CNN trained from scratch | One controlled task and enough matched images | Overfits appearance | Background/viewpoint shift |
| Frozen pretrained vision encoder | Robot data is limited | Features may ignore contact geometry | Frozen vs partially tuned |
| VLM visual tower | Language grounding is central | Semantic features may lose fine geometry | Language and spatial perturbations |
| Video/history encoder | Occlusion, velocity, or contact phase matters | Temporal leakage and latency | Single-frame vs history |

Start with the smallest representation that exposes the task signal. More semantic pretraining does not replace calibration, depth, proprioception, or force sensing.

### 4.2 Language conditioning

Common mechanisms are token concatenation with attention, cross-attention, FiLM-style modulation, or a goal embedding. The important test is causal, not architectural:

- correct instruction;
- paraphrased instruction;
- swapped instruction from another task;
- empty or masked instruction;
- visually identical scene with a different goal.

If correct and swapped instructions behave the same, do not claim language grounding.

### 4.3 Multimodal fusion

| Fusion | Strength | Limitation |
|---|---|---|
| Concatenate pooled embeddings + MLP | Simple and fast | Weak fine-grained token interaction |
| Encoder Transformer | Flexible cross-modal attention | Quadratic token cost |
| Decoder-only token stream | Reuses autoregressive language modeling | Ordering, masks, and decoding latency matter |
| Separate VLM + action expert | Preserves semantic backbone while specializing control | Interface and optimization become more complex |

### 4.4 Action representation

Choose semantics before choosing a neural head.

| Representation | Appropriate when | Failure to watch |
|---|---|---|
| Joint position/delta | Fixed embodiment, reliable joint servo | Cross-robot incompatibility |
| End-effector delta + gripper | Manipulation with an IK/controller layer | Frame and IK singularity errors |
| Joint velocity | Smooth rate control | Integration drift and bounds |
| Torque | Dynamics-rich low-level control | High safety and model requirements |
| Canonical cross-embodiment action | Multi-robot data | Adapter hides infeasible commands |

Rotation requires an explicit representation and loss. Do not apply Euclidean MSE to wrapped Euler angles without handling discontinuities.

### 4.5 Direct regression and chunked behavior cloning

For a deterministic continuous chunk:

$$
\mathcal L_{\text{reg}} = \frac{1}{H}\sum_{h=0}^{H-1}
\|a_{t+h}-\hat a_{t+h}\|_1
$$

or MSE. This is a useful first baseline because it is cheap, diagnosable, and latency-friendly. The loss matters: minimizing expected squared error targets the conditional mean, whereas componentwise L1 targets conditional medians, which need not be unique. Neither single deterministic prediction generally represents a multimodal action distribution; it may select an intermediate action that does not correspond to a valid mode. See [prediction loss and decision theory](https://www.stat.cmu.edu/~cshalizi/sml/21/lectures/02/lecture-02.html).

[ACT](https://arxiv.org/abs/2304.13705) is an action-chunking generative policy developed for fine-grained bimanual imitation. Treat “chunking” and “ACT” separately: many models predict chunks without using ACT’s full CVAE formulation.

### 4.6 Discrete action tokens

Quantize each normalized action dimension or encode action vectors into discrete codes, then minimize token cross-entropy:

$$
\mathcal L_{\text{token}} = -\sum_n \log p_\theta(z_n\mid z_{<n},c_t).
$$

This integrates naturally with autoregressive VLM training, as in [RT-2](https://arxiv.org/abs/2307.15818) and the original [OpenVLA](https://arxiv.org/abs/2406.09246). The costs are quantization error, token ordering choices, and sequential decoding latency. Always report decode-to-action round-trip error.

### 4.7 Continuous parallel action chunks

A continuous head predicts the whole chunk in parallel. [OpenVLA-OFT](https://arxiv.org/abs/2502.19645) is evidence that a continuous action representation, parallel decoding, and action chunking can materially change the fine-tuning and inference trade-off of an autoregressive VLA. Do not transfer its reported throughput or benchmark gain to another implementation without reproducing its setup.

### 4.8 Diffusion action model

For a simplified noise-prediction formulation:

$$
x_\tau=\sqrt{\bar\alpha_\tau}A+\sqrt{1-\bar\alpha_\tau}\epsilon,
\qquad
\mathcal L_{\text{diff}}=\|\epsilon-\epsilon_\theta(x_\tau,\tau,c_t)\|^2.
$$

[Diffusion Policy](https://arxiv.org/abs/2303.04137) motivates this family for multimodal, high-dimensional action distributions with receding-horizon control. Its disadvantages are iterative inference and additional schedule/sampler choices. Compare under equal observation, data, chunk, and execution budgets.

### 4.9 Flow matching action expert

In a simple conditional flow-matching path, sample noise $x_0$, a data action chunk $x_1=A$, and time $\tau$:

$$
x_\tau=(1-\tau)x_0+\tau x_1,\quad
u_\tau=x_1-x_0,\quad
\mathcal L_{\text{FM}}=\|v_\theta(x_\tau,\tau,c_t)-u_\tau\|^2.
$$

Inference numerically integrates the learned vector field. [π0](https://arxiv.org/abs/2410.24164) and [SmolVLA](https://arxiv.org/abs/2506.01844) use flow-based continuous action generation, but their architecture and training details are not interchangeable. Sampling speed must be measured end to end; “flow” is not a universal latency guarantee.

## 5. How to choose an algorithm

### Step 1 · Does language change the required action?

- **No:** start with a simple BC, ACT, or Diffusion Policy baseline. A pretrained VLA may still offer transferable visual or action priors, but that benefit must be measured separately from language disambiguation. Compare against matched from-scratch training; do not claim language grounding from a fixed-instruction task. The [OpenVLA study](https://openvla.github.io/) illustrates that the relative benefit depends on the fine-tuning task, not the model label alone.
- **Yes:** add language to the matched baseline, then test swapped/absent instructions.

### Step 2 · Is the conditional action distribution multimodal?

- **Low or unknown:** continuous chunk regression first.
- **Clearly multimodal:** compare diffusion or flow matching against regression; retain multiple samples and task outcomes.

### Step 3 · What is the latency contract?

- **Hard real-time budget:** prefer parallel regression/chunking initially; measure preprocessing, model, postprocessing, network, and queueing together.
- **Soft budget:** generative heads are candidates, but still use receding-horizon execution.

### Step 4 · What data and compute exist?

| Situation | Recommended first experiment |
|---|---|
| Task-specific data, limited compute | Small chunked BC/ACT; language only if tasks vary |
| Multi-task data, one GPU | Fine-tune a compact/continuous-chunk VLA and keep a from-scratch baseline |
| Heterogeneous multi-robot data | Explicit embodiment adapters, per-robot normalization, and per-robot metrics |
| Large compute and strong language diversity | Compare discrete, continuous, and flow heads under matched data |

### Step 5 · Do not select by parameter count alone

Select using task semantics, data compatibility, control rate, action multimodality, reproducibility, license, and deployment constraints. The repository selector makes these assumptions visible:

```bash
python scripts/select_vla_wam_algorithm.py \
  --goal multimodal-action \
  --compute single-gpu \
  --data task-specific \
  --latency soft
```

## 6. Zero-to-one build sequence

| Stage | Build | Promotion gate |
|---:|---|---|
| 0 | Task and interface contract | Frames, units, rate, success, stop conditions reviewed |
| 1 | State-only BC | Tiny-set overfit and closed-loop state baseline pass |
| 2 | Vision BC | Image/action synchronization and unseen-initial-state evaluation pass |
| 3 | Language-conditioned baseline | Correct language beats swapped/absent language |
| 4 | Action chunks | Receding-horizon execution beats or matches one-step policy |
| 5 | Generative head | Multimodal benefit appears in task outcomes, not only likelihood |
| 6 | Pretrained VLA fine-tuning | Matched from-scratch baseline and data receipt retained |
| 7 | Shift suite | Object, pose, scene, language, and embodiment shifts reported separately |
| 8 | Deployment rehearsal | Latency, bounds, watchdog, shadow mode, and rollback pass |

Suggested repository path:

```bash
python scripts/run_knowledge_map.py --path-to learning-vla
python scripts/run_pipeline.py --run vla-policy --dry-run
python scripts/run_pipeline.py --run vla-policy
```

The included PushCube VLA is a teaching baseline. It does not reproduce OpenVLA, π0, or SmolVLA pretraining.

## 7. Training and fine-tuning protocol

1. Validate schema and normalization on a few samples.
2. Overfit one tiny subset; failure indicates a data/objective/implementation bug.
3. Freeze train/validation/test episode identities before tuning.
4. Record backbone initialization, frozen layers, optimizer groups, precision, accumulation, and augmentation.
5. Compare full tuning, partial tuning, and parameter-efficient tuning only under matched data and steps.
6. Select checkpoints by a declared validation rule, not by test performance.
7. Evaluate closed loop with fixed resets, seeds or trial allocation, and intervention rules.

The [official LeRobot SmolVLA guide](https://huggingface.co/docs/lerobot/smolvla) recommends task-specific fine-tuning and gives one dataset/training starting point. Treat its episode and step counts as framework guidance for that setup, not a universal sample-complexity law.

## 8. Evaluation matrix

| Axis | Required measurement |
|---|---|
| Offline | Loss by action dimension/horizon; decode error; validation split identity |
| Closed loop | Task success, progress, time, interventions, constraint violations |
| Language | Correct, paraphrased, swapped, absent, and contradictory instruction |
| Vision | Camera removal, occlusion, background, viewpoint, and lighting shifts |
| Action | Bounds, smoothness, saturation, chunk overlap, and execution delay |
| Generalization | Separate task, object, scene, embodiment, and motion shifts |
| Systems | End-to-end latency distribution, memory, dropped frames, queue age |
| Statistics | Episodes/trials, seed policy, confidence interval, and failure counts |

Offline action error can improve while task success worsens. Language-conditioned success can improve because tasks correlate with backgrounds. Report these failure paths rather than hiding them in a mean.

## 9. Failure localization

| Symptom | First checks |
|---|---|
| Good training loss, no task success | Time alignment, normalization inverse, action frame, reset distribution |
| Jerky or delayed motion | Queue age, chunk overlap, rate mismatch, network latency |
| Ignores language | Task imbalance, visual shortcut, swapped/absent-language ablation |
| Averages two strategies | Multimodal demonstrations; compare diffusion/flow head |
| Fails at contact | Camera geometry, proprio/force feedback, action rate, controller—not only VLA scale |
| Cross-robot collapse | Per-embodiment adapters, units, masks, action feasibility, pooled metrics |
| Great offline result, weak shift result | Leakage, narrow coverage, checkpoint selection, unreported OOD |

## 10. Research questions worth testing

- Does language provide causal task disambiguation beyond visual shortcuts?
- Which action representation transfers across embodiments without hiding infeasible commands?
- Does a generative action head improve closed-loop multimodal behavior at a fixed latency budget?
- Which layers should adapt under small in-domain datasets?
- Does history improve partial observability, or only leak action timing?
- Can uncertainty predict failure early enough for a safe fallback?

For each question, define a matched baseline, one changed factor, a frozen shift suite, and a result that would reject the hypothesis.

## 11. Primary sources

- [RT-1](https://arxiv.org/abs/2212.06817)
- [RT-2](https://arxiv.org/abs/2307.15818)
- [ACT](https://arxiv.org/abs/2304.13705)
- [Diffusion Policy](https://arxiv.org/abs/2303.04137)
- [Octo](https://arxiv.org/abs/2405.12213)
- [OpenVLA](https://arxiv.org/abs/2406.09246)
- [π0](https://arxiv.org/abs/2410.24164)
- [OpenVLA-OFT](https://arxiv.org/abs/2502.19645)
- [SmolVLA](https://arxiv.org/abs/2506.01844)

Reported paper results are source claims. This track provides an algorithm and evidence curriculum, not a claim that the repository reproduces those systems.
