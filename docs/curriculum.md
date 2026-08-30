# Embodied AI Curriculum Contract

**English · [简体中文](curriculum_cn.md)**

This curriculum turns the repository's 45-node prerequisite graph into an actionable sequence. It defines what to understand, what to build, what to retain, and what evidence is required before advancing.

> The suggested pace is illustrative, not a promise of completion time. Advance by passing the checkpoint, not by spending a fixed number of hours.

## Choose a working mode

| Mode | Primary objective | Minimum retained artifact | Recommended entry |
|---|---|---|---|
| **Learner** | Build a correct mental model and connect concepts | Derivation notes, completed exercise, and prerequisite trace | L0, then follow the graph |
| **Engineer** | Make one closed-loop system reproducible and diagnosable | Environment receipt, configuration, logs, metrics, and failure report | L0 → L2 → one Pipeline |
| **Researcher** | Compare a hypothesis against a controlled baseline | Frozen protocol, baseline, ablation, uncertainty, and decision | Resolve missing nodes, then L3 → L5 |

## Six-stage progression

### L0 · Computing and experiment discipline

**Learn**

- Python, NumPy, tensor shapes, dtypes, units, and array semantics.
- Configuration, interfaces, deterministic seeds, tests, and source control.
- Experiment identity: code commit, data version, parameters, environment, and output paths.

**Build**

- Run one repository example from a clean environment.
- Produce a machine-readable configuration and a timestamped result directory.
- Trace every input and output shape through the example.

**Checkpoint**

- Another person can reproduce the command without guessing a dependency or parameter.
- The run records its seed, commit, environment, and artifact location.

Primary lessons: [Python for robotics](foundations/01-python-for-robotics.md) · [Evaluation and reproducibility](foundations/14-evaluation-and-reproducibility.md)

### L1 · Mathematical language for physical systems

**Learn**

- Linear algebra, least squares, probability, estimation, and optimization.
- Coordinate frames, transform composition, SO(3), SE(3), and rotation representations.
- Conditioning, discretization, finite differences, and numerical stability.

**Build**

- Derive and numerically verify a transform chain.
- Compare at least two rotation representations near a known failure case.
- Solve a constrained least-squares problem and report residuals and conditioning.

**Checkpoint**

- Every vector is labeled with frame, units, shape, and timestamp semantics.
- The numerical result agrees with an independently computed check within a declared tolerance.

Primary lessons: [Linear algebra](foundations/02-linear-algebra.md) · [Coordinate transforms](foundations/05-coordinate-transform.md) · [SO(3) and SE(3)](foundations/06-se3-and-rotation.md) · [Probability and optimization](foundations/11-probability-and-optimization.md)

### L2 · Robot modeling, sensing, control, and simulation

**Learn**

- Forward kinematics, Jacobians, inverse kinematics, rigid-body dynamics, and actuation limits.
- Contact, friction, grasp stability, feedback, impedance, saturation, and watchdogs.
- Sensor models, calibration, synchronization, fusion, and uncertainty.
- URDF/MJCF structure, scene composition, contact parameters, observability, and reset design.

**Build**

- Create or inspect a robot model and explain joints, frames, actuators, and collision geometry.
- Run a closed-loop controller with explicit rates, limits, and a stop condition.
- Build a MuJoCo scene, add an object and sensor, then verify contacts and state logging.
- Inject a timing, calibration, or sensor fault and diagnose its downstream effect.

**Checkpoint**

- A trace connects observation → state estimate → command → simulated response → metric.
- The report distinguishes model error, controller error, sensor error, and task-definition error.

Primary lessons: [FK, Jacobian, and IK](foundations/07-fk-jacobian-ik.md) · [Control](foundations/08-control-basics.md) · [MuJoCo](foundations/09-mujoco-basics.md) · [Perception and sensors](foundations/12-perception-and-sensors.md) · [Robot systems and safety](foundations/13-robot-systems-and-safety.md)

### L3 · Data, policies, prediction, and planning

**Learn**

- Episode schemas, multimodal alignment, coverage, leakage-free splits, and dataset quality.
- Neural training loops, Transformers, behavior cloning, covariate shift, and action representations.
- VLA policies, RL, MDP/POMDP formulations, world models, and trajectory optimization.

**Build**

- Collect or validate a dataset with synchronized observations, actions, task labels, and metadata.
- Train one baseline and retain the configuration, checkpoint, learning curve, and evaluation output.
- Compare open-loop prediction with closed-loop task behavior.
- Add one failure-focused split or perturbation instead of reporting only an average score.

**Checkpoint**

- Train, validation, and test identities are explicit and leakage checks pass.
- Evaluation uses a fixed protocol and reports sample count, seed policy, uncertainty, and failures.

Primary lessons: [Dataset and training](foundations/10-dataset-and-training.md) · [Transformer basics](foundations/04-transformer-basics.md) · [VLA policy Pipeline](pipelines/02-vla-policy.md) · [World-model Pipeline](pipelines/03-world-model-planning.md) · [RL Pipeline](pipelines/04-rl-post-training.md)

**VLA/WAM specialization:** after the L3 dataset and baseline checkpoint passes, use the [specialization overview](specializations/README.md) to choose a track. [VLA Zero to One](specializations/vla-zero-to-one.md) covers multimodal policy and action-generation families; [WAM Zero to One](specializations/wam-zero-to-one.md) first establishes world-model planning baselines, then introduces joint video-action learning. Do not start with a large joint model before the matched policy and model-based baselines are measured.

### L4 · Task-level embodied systems

Choose one task family and complete its full loop.

| Task family | Required stages | Minimum task evidence |
|---|---|---|
| Manipulation and imitation | observation → policy/planner → control → task metric → recovery | Success definition, object/state trace, failure taxonomy |
| Dexterity and teleoperation | human input → retargeting → command admission → contact → retention/task | Geometric error, limits, contact/retention, task result |
| Navigation and agents | state/map → plan → action → localization update → recovery | Goal completion, collision/path metrics, recovery cases |
| Humanoids and locomotion | state → motion command → whole-body control → balance → disturbance test | Stability, tracking, falls/failures, safety bounds |

**Checkpoint**

- The system has a task-level success definition, not only a plausible trajectory.
- Failures are localized to a stage and the recovery or stop behavior is observable.

Primary routes: [Research-route map](learning-paths/README.md) · [Pipeline catalog](pipelines/README.md)

### L5 · Evidence, generalization, and deployment decisions

**Learn**

- Baselines, ablations, uncertainty, negative results, robustness, and distribution shift.
- System identification, domain randomization, hardware-in-the-loop, shadow mode, and guarded rollout.
- Operational authorization, supervision, bounded commands, stop paths, and audit logs.

**Build**

- Freeze one evaluation protocol before comparing methods.
- Run a baseline and one hypothesis-driven change under matched budgets.
- Test a declared shift: object, scene, viewpoint, embodiment, latency, or disturbance.
- Write a promotion decision that names both supporting evidence and remaining blockers.

**Checkpoint**

- A lower evidence level is never presented as a higher one.
- Hardware execution requires explicit authorization and a bounded safety protocol; simulation alone is insufficient.

Primary references: [Validation policy](VALIDATION.md) · [Claim review](CLAIM_REVIEW.md) · [Benchmark contract](../BENCHMARK.md) · [Sim-to-Real Pipeline](pipelines/07-sim-to-real.md)

## Suggested twelve-block sequence

This sequence is a planning template. Repeat or split blocks until their checkpoints pass.

| Block | Focus | Deliverable |
|---:|---|---|
| 1 | Environment, Python, shapes, units | Reproducible environment receipt and traced example |
| 2 | Linear algebra and transforms | Verified frame-composition notebook |
| 3 | SO(3), SE(3), kinematics, IK | FK/IK comparison with residuals and failure cases |
| 4 | Dynamics, contact, control | Bounded closed-loop simulation and diagnosis |
| 5 | Sensors, calibration, state estimation | Synchronized state-estimation trace with uncertainty |
| 6 | MuJoCo modeling and task design | Custom scene, contact checks, reset distribution, logs |
| 7 | Dataset schema and quality | Datasheet, split manifest, leakage and coverage report |
| 8 | Behavior cloning and action representations | Trained baseline plus open/closed-loop evaluation |
| 9 | VLA, world models, or RL | One advanced policy/prediction baseline and failure report |
| 10 | Task-level Pipeline | Complete manipulation, dexterity, navigation, or locomotion loop |
| 11 | Robustness and generalization | Declared shift suite with confidence intervals or repeated trials |
| 12 | Research and deployment decision | Reproducible report, ablation, evidence grade, promotion/stop decision |

## Operating procedure for every topic

1. **Diagnose** the target with `python scripts/run_knowledge_map.py --path-to <node>`.
2. **Learn** the required document and reproduce its smallest exercise.
3. **Build** the mapped Pipeline and retain intermediate artifacts.
4. **Measure** task metrics and stage-level failures under a fixed protocol.
5. **Decide** whether the evidence supports progression, repetition, or a stop.

The machine-readable graph is [`knowledge/manifest.json`](../knowledge/manifest.json). Validate it with:

```bash
python scripts/run_knowledge_map.py --validate
python scripts/run_knowledge_map.py --summary
python scripts/run_pipeline.py --validate
python scripts/run_learning_path.py --validate
```
