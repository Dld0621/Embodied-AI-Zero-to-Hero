# Embodied AI Knowledge System

[中文](README_CN.md){ .md-button } [Inspect the graph](https://github.com/Dld0621/Embodied-AI-Zero-to-Hero/blob/master/knowledge/manifest.json){ .md-button }

This is the repository's **knowledge-point-level source of truth**. It connects 45 knowledge nodes across 9 domains and 6 stages to prerequisite nodes, primary documents, engineering Pipelines, learner evidence, and explicit assessments.

![Embodied AI knowledge system](../assets/knowledge-system.svg)

> [!IMPORTANT]
> A node is a learning contract, not a claim of mastery or research performance. Completing an explanation, derivation, or smoke path does not imply benchmark or hardware success.

## Read the system in four layers

| Layer | Question | Source of truth |
|---|---|---|
| **Knowledge node** | What must I understand and how do I prove it? | [`knowledge/manifest.json`](https://github.com/Dld0621/Embodied-AI-Zero-to-Hero/blob/master/knowledge/manifest.json) |
| **Foundation document** | Where are the concept, derivation, example, and references? | [`docs/foundations/`](../foundations/README_EN.md) and the linked deep dives |
| **Engineering Pipeline** | How do multiple nodes become a closed-loop artifact? | [Pipeline catalog](../pipelines/README.md) |
| **Research route** | Which sequence answers a research question, and what is the promotion gate? | [Seven research routes](../learning-paths/README.md) |

The previous curriculum listed lessons, Pipelines, and routes separately. The new graph makes their dependencies machine-checkable and prevents a topic label from silently standing in for implementation or evidence.

## Six-stage progression

| Stage | Purpose | Exit gate |
|---:|---|---|
| **L0 · Orientation and tools** | Run code, inspect interfaces, and retain provenance. | Re-run a deterministic task from its receipt. |
| **L1 · Mathematical models** | State assumptions and reason about geometry, uncertainty, and optimization. | Derive a result and verify it numerically. |
| **L2 · Robot closed loop** | Connect body, dynamics, sensing, state, control, and safety. | Inject a perturbation and show bounded behavior. |
| **L3 · Data and learning** | Build datasets, representations, policies, predictive models, and RL loops. | Retain data/model provenance and separate offline from closed-loop results. |
| **L4 · Task intelligence** | Compose skills into manipulation, dexterity, navigation, locomotion, and recovery. | Complete a task protocol with a stage-resolved failure report. |
| **L5 · Evidence and deployment** | Compare methods, test generalization, and decide whether risk may increase. | Make a promotion decision without crossing the available evidence boundary. |

## Start from the gap you actually have

| If you can already… | Start at | Do not skip |
|---|---|---|
| Write Python but not reason about robot frames | `robot-coordinate-frames` | Units, transform direction, round-trip checks |
| Train image models but not execute robot commands | `learning-action-representations` | Control rate, units, bounds, and actuator semantics |
| Build mechanisms but not train policies | `learning-neural-networks` | Data splits, validation, and closed-loop distribution shift |
| Run a simulator but not trust its results | `sim-scene-dynamics` | Contact model, sensor semantics, reset distribution, and evidence boundary |
| Read VLA papers but lack a reproducible baseline | `learning-vla` | Dataset provenance, ablations, task success, and latency |
| Retarget hand pose but need task-level dexterity | `task-dexterity-teleoperation` | Realizability, command admission, contact, retention, and task evidence |
| Have a policy ready for a robot | `deploy-sim-to-real` | HIL, shadow mode, stop paths, supervision, and rollback |

Resolve the exact prerequisite order instead of guessing:

```bash
python scripts/run_knowledge_map.py --validate
python scripts/run_knowledge_map.py --stats
python scripts/run_knowledge_map.py --show learning-vla
python scripts/run_knowledge_map.py --path-to task-dexterity-teleoperation
```

## Nine knowledge domains

<a id="computing"></a>
### 1. Scientific computing

| Node | Core capability | Primary document | Learner evidence |
|---|---|---|---|
| `computing-python-numpy` | Python, NumPy, tensor shape, dtype, unit, and data-flow discipline | [Python for robotics](../foundations/01-python-for-robotics.md) | Execute |
| `computing-software-contracts` | Explicit observation/action schemas, configuration, tests, and invalid-input rejection | [Robot systems and safety](../foundations/13-robot-systems-and-safety.md) | Execute |
| `computing-experiment-workflow` | Commit, environment, seed, data, checkpoint, metric, and artifact provenance | [Evaluation and reproducibility](../foundations/14-evaluation-and-reproducibility.md) | Evaluate |

**Exit question:** Can another person reproduce the artifact without relying on hidden machine state?

<a id="mathematics"></a>
### 2. Mathematics and uncertainty

| Node | Core capability | Primary document | Learner evidence |
|---|---|---|---|
| `math-linear-algebra` | Vectors, matrices, projections, decompositions, least squares, and conditioning | [Linear algebra](../foundations/02-linear-algebra.md) | Derive |
| `math-probability-statistics` | Random variables, estimation, uncertainty, confidence, and calibration | [Probability and optimization](../foundations/11-probability-and-optimization.md) | Derive |
| `math-optimization` | Objectives, constraints, regularization, infeasibility, and termination | [Probability and optimization](../foundations/11-probability-and-optimization.md) | Derive |
| `math-numerical-stability` | Discretization, singularity, damping, scaling, and solver sensitivity | [FK, Jacobian, and IK](../foundations/07-fk-jacobian-ik.md) | Execute |

**Exit question:** Can you state what is optimized, under which assumptions, and how numerical error is detected?

<a id="robot-modeling"></a>
### 3. Robot modeling and mechanics

| Node | Core capability | Primary document | Learner evidence |
|---|---|---|---|
| `robot-coordinate-frames` | Frames, units, transform direction, and composition order | [Coordinate transforms](../foundations/05-coordinate-transform.md) | Derive |
| `robot-so3-se3` | SO(3), SE(3), rotation representations, conversion, and interpolation | [SO(3) and SE(3)](../foundations/06-se3-and-rotation.md) | Derive |
| `robot-kinematics` | FK, Jacobians, numerical IK, limits, and singularity handling | [FK, Jacobian, and IK](../foundations/07-fk-jacobian-ik.md) | Execute |
| `robot-rigid-body-dynamics` | Force, acceleration, state integration, timestep, and energy/tracking behavior | [Control basics](../foundations/08-control-basics.md) | Derive |
| `robot-contact-friction` | Collision, contact constraints, friction, slip, retention, and grasp stability | [Dexterous manipulation](../pipelines/11-dexterous-manipulation.md) | Evaluate |
| `robot-actuation-transmission` | Actuators, transmissions, tendons, executable coordinates, and limits | [Concept encyclopedia](../00-concepts-encyclopedia.md) | Explain |

**Exit question:** Can you trace a command from task-space intent to bounded force or motion at the mechanism?

<a id="sensing-control"></a>
### 4. Sensing, estimation, and control

| Node | Core capability | Primary document | Learner evidence |
|---|---|---|---|
| `system-sensor-models` | Modality, frame, unit, rate, latency, noise, and missing-data semantics | [Perception and sensors](../foundations/12-perception-and-sensors.md) | Explain |
| `system-calibration-synchronization` | Spatial calibration, clocks, timestamp alignment, and measured residuals | [Perception and state estimation](../pipelines/09-perception-state-estimation.md) | Evaluate |
| `system-state-estimation` | Filtering, fusion, uncertainty, stale-data handling, and state validity | [Perception and state estimation](../pipelines/09-perception-state-estimation.md) | Execute |
| `system-feedback-control` | Feedback, trajectories, rate, saturation, anti-windup, and tracking | [Control basics](../foundations/08-control-basics.md) | Execute |
| `system-force-compliance` | Force, impedance, compliance, and bounded contact interaction | [Control basics](../foundations/08-control-basics.md) | Evaluate |
| `system-realtime-safety` | Watchdogs, limits, state machines, stop paths, logging, and human authority | [Robot systems and safety](../foundations/13-robot-systems-and-safety.md) | Deployment gate |

**Exit question:** When an observation or command becomes late, invalid, or unsafe, does the system fail into a known bounded state?

<a id="simulation-data"></a>
### 5. Simulation and data

| Node | Core capability | Primary document | Learner evidence |
|---|---|---|---|
| `sim-model-formats` | URDF/MJCF structure, assets, units, inertials, actuators, sensors, and provenance | [MuJoCo scene building](../tutorials/mujoco-scene-building.md) | Execute |
| `sim-scene-dynamics` | Scene dynamics, contacts, sensors, control inputs, logging, and deterministic stepping | [MuJoCo basics](../foundations/09-mujoco-basics.md) | Execute |
| `sim-task-randomization` | Task/reset definitions, perturbations, domain randomization, and coverage | [Simulation and data Pipeline](../pipelines/01-simulation-data.md) | Evaluate |
| `data-episode-schema` | Synchronized observation, action, language, timestamp, task, and terminal fields | [Dataset and training](../foundations/10-dataset-and-training.md) | Execute |
| `data-quality-splits` | Integrity, coverage, distribution shift, leakage-free splits, and datasheets | [Dataset and training](../foundations/10-dataset-and-training.md) | Evaluate |

**Exit question:** Can you reconstruct what the robot observed, what was commanded, why the episode ended, and which distribution it belongs to?

<a id="robot-learning"></a>
### 6. Robot learning

| Node | Core capability | Primary document | Learner evidence |
|---|---|---|---|
| `learning-neural-networks` | Forward pass, loss, gradients, optimization, validation, and overfitting | [Deep learning](../foundations/03-deep-learning-basics.md) | Execute |
| `learning-transformers-multimodal` | Tokens, images, state, masks, time, and attention shape tracing | [Transformers](../foundations/04-transformer-basics.md) | Derive |
| `learning-behavior-cloning` | Supervised imitation, covariate shift, closed-loop drift, and recovery limits | [ACT vs Diffusion Policy](../22-act-vs-diffusion-policy.md) | Evaluate |
| `learning-action-representations` | Joint/task actions, deltas, chunks, tokens, diffusion, rate, and bounds | [Action representation](../24-action-representation-and-tokenization.md) | Derive |
| `learning-vla` | Vision-language grounding, action prediction, temporal validity, and ablation | [VLA zero to one](../specializations/vla-zero-to-one.md) | Evaluate |
| `learning-reinforcement-learning` | State, action, reward, termination, exploration, post-training, and safety | [RL zero to one](../14-rl-zero-to-one.md) | Evaluate |
| `learning-cross-embodiment` | Canonical schemas, adapters, embodiment semantics, and per-robot results | [Cross-embodiment adaptation](../25-cross-embodiment-adaptation.md) | Evaluate |

**Exit question:** Does the policy output match the downstream controller's semantics, rate, units, horizon, and safety envelope?

<a id="prediction-planning"></a>
### 7. Prediction, planning, and reasoning

| Node | Core capability | Primary document | Learner evidence |
|---|---|---|---|
| `planning-mdp-pomdp` | Observed/hidden state, action, reward, transition, belief, and termination | [RL fundamentals](../06-rl-fundamentals-for-vla.md) | Derive |
| `planning-motion-trajectory` | Path, trajectory, collision, dynamics, feasibility, optimization, and feedback | [Optimization methods](../04-optimization-methods.md) | Execute |
| `planning-world-models` | One-step fit, multi-step rollout, uncertainty, and planning utility | [World model zero to one](../15-world-model-zero-to-one.md) | Evaluate |
| `planning-task-and-motion` | Typed goals, preconditions, skills, geometric feasibility, and feedback | [Embodied reasoning and planning](../27-embodied-reasoning-and-planning.md) | Evaluate |
| `planning-reasoning-recovery` | State-grounded monitoring, failure detection, bounded recovery, and replanning | [Embodied reasoning Pipeline](../pipelines/06-embodied-reasoning.md) | Evaluate |

**Exit question:** Can the system detect when its plan no longer matches the world and select a bounded recovery action?

<a id="embodied-tasks"></a>
### 8. Embodied task systems

| Node | Core capability | Primary document | Learner evidence |
|---|---|---|---|
| `task-manipulation` | Perception, approach, contact, motion, verification, and recovery | [Dexterous manipulation Pipeline](../pipelines/11-dexterous-manipulation.md) | Evaluate |
| `task-dexterity-teleoperation` | Human pose, robot realizability, command admission, contact, retention, and task evidence | [Dexterous retargeting Pipeline](../pipelines/08-dexterous-retargeting.md) | Evaluate |
| `task-navigation` | Localization assumptions, planning, control, collision, recovery, and replanning | [Navigation and locomotion Pipeline](../pipelines/10-navigation-locomotion.md) | Evaluate |
| `task-locomotion-humanoids` | Tracking, balance, contacts, falls, recovery, and embodiment limits | [Navigation and locomotion Pipeline](../pipelines/10-navigation-locomotion.md) | Deployment gate |

**Exit question:** Is task success supported by stage-level evidence, or only by a plausible command or trajectory?

<a id="evaluation-deployment"></a>
### 9. Evaluation and deployment

| Node | Core capability | Primary document | Learner evidence |
|---|---|---|---|
| `eval-task-metrics` | Task definitions, numerators, denominators, episode protocol, and failure taxonomy | [Evaluation and reproducibility](../foundations/14-evaluation-and-reproducibility.md) | Evaluate |
| `eval-generalization-robustness` | Held-out factors, perturbations, robustness, and distribution shift | [Evaluation metrics](../06-evaluation-metrics.md) | Evaluate |
| `eval-statistics-ablations` | Seeds, variability, ablations, compute, negative results, and fair comparison | [Evaluation and reproducibility](../foundations/14-evaluation-and-reproducibility.md) | Evaluate |
| `deploy-sim-to-real` | System identification, replay, randomization, HIL, shadow mode, and rollback | [Sim-to-Real guide](../19-sim-to-real-guide.md) | Deployment gate |
| `deploy-hardware-gates` | Authorization, supervision, bounded commands, stop paths, logs, and hardware evidence | [Validation policy](../VALIDATION.md) | Deployment gate |

**Exit question:** What exact evidence justifies the next increase in autonomy or physical risk—and what still blocks it?

## Learner evidence is not repository evidence

| Learner evidence | Required proof | It does not establish |
|---|---|---|
| **Explain** | Define the concept, assumptions, units, and failure modes. | A correct implementation |
| **Derive** | Produce the relation and verify a limiting or numerical case. | Closed-loop task performance |
| **Execute** | Run a deterministic example and inspect its artifacts. | Generalization or benchmark quality |
| **Evaluate** | Use a fixed protocol, metrics, seeds, and failure analysis. | Hardware readiness |
| **Deployment gate** | Pass the declared safety and operational promotion criteria. | Universal safety outside that scope |

Repository evidence labels—smoke-tested, interface-tested, benchmark, and hardware-dependent—remain governed by the [validation policy](../VALIDATION.md). The two vocabularies must not be conflated.

## Maintenance contract

When adding a knowledge point:

1. Add one bilingual node to [`knowledge/manifest.json`](https://github.com/Dld0621/Embodied-AI-Zero-to-Hero/blob/master/knowledge/manifest.json).
2. Link a real primary document and at least one registered Pipeline.
3. Declare prerequisites, learner evidence, outcome, and assessment.
4. Add or update a primary source in the [source registry](../SOURCES.md).
5. Run `python scripts/run_knowledge_map.py --validate` and the repository test suite.

The graph validator rejects missing documents, unknown Pipelines, later-stage prerequisites, duplicate IDs, and dependency cycles. It verifies repository structure; it cannot automatically certify the semantic truth of a lesson or authorize robot operation.
