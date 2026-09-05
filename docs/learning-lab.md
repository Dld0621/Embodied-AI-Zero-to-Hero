# Interactive Learning Lab

For smaller conceptual steps, use the [Chinese-led illustrated atlas](knowledge-atlas/index.md). It covers all 45 declared nodes with worked examples, local diagrams, and self-checks; related nodes link back to the five experiments below.

**English · [简体中文](learning-lab-cn.md) · [Start here](start-here.md)**

Turn a formula into a prediction you can test. Five small experiments connect coordinate frames, robot geometry, feedback, action timing, and the strength of experimental evidence. Change one variable at a time: predict first, operate the controls, explain the difference, then transfer the idea to code.

**[Open the interactive site](https://dld0621.github.io/Embodied-AI-Zero-to-Hero/learning-lab/)**. The controls run on the documentation site; GitHub's file preview does not execute JavaScript. The worked examples, equations, and checkpoints below also work as a complete reading companion without JavaScript. No robot connection or model training is involved.

<div class="eai-labs" data-lab-lang="en"></div>

<a id="worked-examples"></a>

## Predict → operate → explain → transfer

| Experiment | Question to answer first | Continue learning | Evidence to produce |
|---|---|---|---|
| 01 · Frames | Why does one physical point have different coordinates? | [Coordinate transforms](foundations/05-coordinate-transform.md) | Forward/inverse calculation with units |
| 02 · Kinematics | Does a reachable point allow every instantaneous velocity? | [FK / Jacobian / IK](foundations/07-fk-jacobian-ik.md) | A singular configuration and its missing direction |
| 03 · Control | Why is a larger gain not always better? | [Control foundations](foundations/08-control-basics.md) | Separate gain, delay, and saturation experiments |
| 04 · Timing | Why predict 16 actions but execute only 4? | [VLA Zero to One](specializations/vla-zero-to-one.md) | A timestamped observation–inference–action timeline |
| 05 · Evaluation | Are two 80% success rates equally convincing? | [Evaluation and reproducibility](foundations/14-evaluation-and-reproducibility.md) | A denominator, interval, and sampling assumptions |

Clicks and checkpoint answers are not proficiency certificates. Transfer tasks require calculations, code, configurations, and explanations under the repository's [evidence-based assessment](assessment.md).

<a id="frames-guide"></a>

## 01 · Frames: name the mapping before calculating

**Model and units.** Map a point from local sensor frame S to world frame W in a plane. Angle controls use degrees; trigonometric calculations use radians. Coordinates and translations are in meters. The columns of $R_{WS}$ are S's axes expressed in W, and $t_{WS}$ is S's origin expressed in W.

$$
p_W=R_{WS}p_S+t_{WS},\qquad
R(\theta)=
\begin{bmatrix}
\cos\theta&-\sin\theta\\
\sin\theta&\cos\theta
\end{bmatrix}.
$$

**Worked example.** With $p_S=(1,0)$ m, $\theta=90^\circ$, and $t_{WS}=(0.5,0.5)$ m, rotation gives $(0,1)$ m and translation gives $p_W=(0.5,1.5)$ m. For a fixed transform, the forward and inverse calculations describe the same physical point in different coordinates. Changing angle or translation sliders while keeping the local point fixed instead changes the physical placement of the sensor and point, so the world-space point moves.

1. **Predict:** Which world coordinate changes if only the x translation increases by 0.5 m?
2. **Operate:** Keep the angle and local point fixed while changing translation; then change only the angle. Watch the axes and point.
3. **Explain:** Translation changes the origin; rotation changes the basis directions. Mixing them can send a camera-derived target to the wrong robot position.
4. **Transfer:** Implement both mappings in NumPy. Check that <code>inverse(forward(p))</code> recovers the input within a stated numerical tolerance.

<details markdown="1">
<summary>Checkpoint: why is the inverse not “rotate back, then subtract the original translation”?</summary>

Rearranging $p_W=Rp_S+t$ gives $p_S=R^\mathsf{T}(p_W-t)$. First subtract a W-frame translation in W, then rotate into S. The inverse translation is $-R^\mathsf{T}t$, not simply $-t$.

For the example, $R^\mathsf{T}((0.5,1.5)-(0.5,0.5))=(1,0)$. The incorrect expression $R^\mathsf{T}p_W-t$ gives $(1,-1)$.

</details>

**Pitfalls and limits.** Rotating a frame and actively rotating a vector in a fixed frame are different questions; this lab explicitly uses the S → W coordinate mapping. It does not model camera projection, depth noise, or 3D calibration. [Modern Robotics: homogeneous transformations](https://modernrobotics.northwestern.edu/nu-gm-book-resource/3-3-1-homogeneous-transformation-matrices/) distinguishes configuration, coordinate mapping, and displacement.

<a id="kinematics-guide"></a>

## 02 · Kinematics: reachable position is not arbitrary local velocity

**Model and units.** Two revolute joints and rigid links of lengths $l_1=1$ m and $l_2=0.7$ m. Angle $q_1$ is relative to world x; $q_2$ is relative to link 1. Computation uses radians. Collision, joint limits, and dynamics are excluded.

$$
x=l_1\cos q_1+l_2\cos(q_1+q_2),\qquad
y=l_1\sin q_1+l_2\sin(q_1+q_2).
$$

**Worked example.** At $q_1=0^\circ$, $q_2=90^\circ$, the tip is $(1,0.7)$ m. The position Jacobian differentiates with respect to radians and satisfies $\dot p=J(q)\dot q$. Its determinant is $\det J=l_1l_2\sin q_2=0.7$. This is not a success rate or a universal manipulability score across robots.

1. **Predict:** When both links are straight, can the tip still move instantaneously in every direction?
2. **Operate:** Move joint 2 toward $0^\circ$ and inspect the geometry and singularity indicator. Try IK with target coordinates, then verify with FK.
3. **Explain:** The two Jacobian columns become collinear, reducing the instantaneous 2D velocity space to a line. Inverting a requested velocity in a missing direction becomes problematic.
4. **Transfer:** Compare the analytic Jacobian against central finite differences. Use a step in radians and test both a regular and near-singular configuration.

<details markdown="1">
<summary>Checkpoint: why is (0.2, 0) m unreachable? Does zero instantaneous x velocity at full extension mean the tip can never move inward?</summary>

Without limits or collisions, the reachable radius satisfies $|l_1-l_2|\leq r\leq l_1+l_2$, or $0.3\leq r\leq1.7$ m. A 0.2 m radius lies inside the inner hole; more IK iterations cannot change the geometry.

At full extension along x, the first-order x velocity vanishes. Bending the joints can still move the tip inward; its initial x displacement is second order in the angle change. A missing instantaneous first-order direction is not a permanent loss of all later positions. Singularity is defined through rank; a determinant test does not apply directly to a nonsquare Jacobian.

</details>

**Pitfalls and limits.** IK can have two elbow configurations, a boundary solution, or no solution. Geometric reachability does not establish collision freedom, safety, or dynamic feasibility. See [Modern Robotics: singularities](https://modernrobotics.northwestern.edu/nu-gm-book-resource/5-3-singularities/).

<a id="control-guide"></a>

## 03 · Control: gains, delay, and saturation interact

**Model and units.** A 1D mass $m=1$ kg has viscous damping $b=0.2$ N·s/m. Initial position and velocity are zero; the reference is $r=1$ m. Force is limited to ±10 N and the integration step is 0.005 s. Both measured position and velocity are delayed by $\tau$. Sensor noise, contact, static friction, and hardware communication are excluded.

$$
m\ddot x=u-b\dot x,\qquad
u=\operatorname{clip}\bigl(K_p[r-x(t-\tau)]-K_d\dot x(t-\tau),-10,10\bigr).
$$

$K_p$ has units N/m; $K_d$ has units N·s/m. P acts like a spring toward the target; D opposes velocity. With zero delay, no saturation, and a constant target, the error obeys:

$$
m\ddot e+(b+K_d)\dot e+K_pe=0,\qquad
\omega_n=\sqrt{K_p/m},\qquad
\zeta=\frac{b+K_d}{2\sqrt{mK_p}}.
$$

**Worked example.** For $K_p=16$, $K_d=7.8$, the linear model has $\omega_n=4$ rad/s and $\zeta=1$. However, the initial requested force is 16 N, above the 10 N limit. The plotted initial response therefore differs from that unsaturated prediction; the label “critical damping” alone is not a guarantee about the simulation. The linear quantities follow [Modern Robotics: second-order error dynamics](https://modernrobotics.northwestern.edu/nu-gm-book-resource/11-2-2-2-second-order-error-dynamics/).

1. **Predict:** With D fixed, can increasing P make recovery faster and oscillations larger at the same time?
2. **Operate:** Compare P and D at zero delay first. Then fix the gains and increase delay. Inspect trajectory, error, and force saturation separately.
3. **Explain:** Delayed feedback corrects past errors, and saturation separates requested force from applied force. Neither is covered by the simple damping-ratio guarantee.
4. **Transfer:** Rebuild the model in Python and halve the integration step to check convergence. Report peak, terminal error, and saturation duration. A finite trace without divergence is not a stability proof.

<details markdown="1">
<summary>Checkpoint: when Kp rises from 16 to 64, which Kd preserves ideal critical damping? Does that guarantee the plotted response?</summary>

Setting $\zeta=1$ gives $K_d=2\sqrt{mK_p}-b=15.8$ N·s/m. Keeping 7.8 instead gives $\zeta=0.5$, an underdamped linear system.

Only the ideal damping ratio is preserved. The initial 64 N request is clipped to 10 N, and nonzero delay changes the dynamics. Check saturation, delay, and discretization independently before applying any conclusion to hardware.

</details>

<a id="timing-guide"></a>

## 04 · Timing: fewer inference calls do not guarantee better feedback

**Model and units.** Each cycle acquires an observation, waits for inference, executes the first $K$ predicted actions, and observes again. Prediction horizon is $H$, with $1\leq K\leq H$; inference latency $L$ and action interval $\Delta t$ are in milliseconds. This is synchronous, serial execution with no new actions during inference. A real low-level controller still needs an explicit hold or safety behavior.

$$
T_{\mathrm{cycle}}=L+K\Delta t,\qquad
\eta=\frac{K\Delta t}{L+K\Delta t},\qquad
A_{\mathrm{last}}=L+(K-1)\Delta t.
$$

$\eta$ is the action-execution time fraction in this model, not GPU utilization or success rate. $A_{\mathrm{last}}$ is the age of the conditioning observation at the **start** of the last executed action; its age at that action's end is another $\Delta t$ larger.

**Worked example.** For $H=16$, $K=4$, $L=100$ ms, $\Delta t=50$ ms: the cycle is 300 ms, execution duty is 66.7%, and the last action starts with a 250 ms old observation. Changing only H to 24 changes none of these values when L and K are fixed. In a real model, H may affect measured inference latency.

1. **Predict:** Why can increasing K reduce waiting overhead but worsen reactions to changes in the scene?
2. **Operate:** Hold L and the action interval fixed; compare K=1, 4, and 8. Inspect the timeline and observation age. Then fix K and vary H alone.
3. **Explain:** Reusing an observation amortizes inference but reduces re-observation frequency. A smooth action chunk cannot substitute for feedback.
4. **Transfer:** Log camera exposure, inference start/end, action dispatch, and execution timestamps. Asynchronous queues, stale-action rejection, and temporal ensembling need their own timing model.

<details markdown="1">
<summary>Checkpoint: with K=8 and the other example values unchanged, what are the cycle, execution duty, and last-action observation age? Should you therefore use K=8?</summary>

The cycle is $100+8\times50=500$ ms, execution duty is $400/500=80\%$, and the last action starts with an observation age of $100+7\times50=450$ ms.

Waiting overhead falls, but the policy goes longer without using new observations. The choice depends on task dynamics, disturbances, action semantics, safety limits, and measured success—not duty alone.

</details>

**Source and scope.** The [Diffusion Policy authors' project page](https://diffusion-policy.cs.columbia.edu/) describes action-sequence prediction with receding-horizon decisions. The timing equations here are derived for the stated serial model, not measured performance of that paper or of a particular VLA.

<a id="evaluation-guide"></a>

## 05 · Evaluation: a percentage needs a denominator

**Model and units.** Fix the success criterion and count k successes from n independent binary trials under one evaluation distribution. The estimate is $\hat p=k/n$. The lab uses $z=1.96$ for a two-sided approximate 95% Wilson interval:

$$
\frac{\hat p+\frac{z^2}{2n}\pm
z\sqrt{\frac{\hat p(1-\hat p)}{n}+\frac{z^2}{4n^2}}}
{1+\frac{z^2}{n}}.
$$

This is a frequentist construction: over repeated sampling under the same conditions, intervals cover the true success probability approximately 95% of the time. It does not assign a 95% probability to a fixed parameter lying in this particular interval. See [NIST: confidence intervals for a proportion](https://www.itl.nist.gov/div898/handbook/prc/section2/prc241.htm).

**Worked comparison.** Both 8/10 and 80/100 give 80%, but their Wilson intervals are approximately 49.0%–94.3% and 71.1%–86.7%. A narrower interval reduces uncertainty under the sampling model; it does not repair leakage or establish generalization.

1. **Predict:** At a fixed success proportion, what happens to interval width when trials increase tenfold?
2. **Operate:** Compare 8/10 with 80/100, then try 10/10. Notice why all observed successes do not establish a true 100% success probability.
3. **Explain:** Few outcomes are compatible with many true success probabilities. Selecting only the best videos or duplicating a trajectory changes the meaning of the evidence.
4. **Transfer:** Report <code>successes / episodes</code>, interval method, tasks, seeds, failure categories, and commit. Inspect per-task and per-scene results and retain failed episodes.

<details markdown="1">
<summary>Checkpoint: A repeats one scene 100 times; B runs 10 trials in each of 10 scenes. Both succeed 80 times. Does this chart establish equivalent evidence?</summary>

No. The chart sees totals, not scene coverage, dependence, or distribution shift. Outcomes sharing a scene or seed can be correlated, and success probabilities may vary between scenes. Define the comparison target, fix task sets and budgets, report strata, and consider task- or scene-clustered resampling when appropriate.

Likewise, overlap of two intervals is not a universal significance test for model superiority. Paired evaluation needs the outcome for each model on each shared initial state.

</details>

## Turn intuition into an independent experiment

Submit one page with: question and prediction → parameters and units → plot or numerical result → explanation → missing model features → next code-level check. For every conclusion, name one condition that would invalidate it.

Continue to the [three capstones](capstone.md). These simplified browser experiments build intuition; independent research also requires data contracts, code verification, controlled comparisons, failure analysis, and evidence someone else can inspect.
