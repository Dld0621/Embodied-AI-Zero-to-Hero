# Seven Research Routes

**English** · [简体中文](README_CN.md)

This layer turns a broad interest into an executable research brief. Each route connects a question to prerequisite lessons, registered Pipelines, a concrete deliverable, metrics, a promotion gate, and an explicit evidence boundary.

> A route organizes work; it does not upgrade the evidence level of any Pipeline. Check each linked Pipeline's status before making a claim.

## Pick by research outcome

| Direction | Build this | Core Pipelines |
|:---|:---|:---|
| [Foundation Models & VLA](#foundation-models-vla) | Language-conditioned policy + adapter + ablation | Data → VLA → RFM |
| [Manipulation & Imitation](#manipulation-imitation) | Closed-loop manipulation baseline + failure taxonomy | Data → VLA → RL |
| [Dexterity & Teleoperation](#dexterity-teleoperation) | Retargeted motion + separated evidence report | Retargeting → State → Sim-to-Real |
| [Navigation & Embodied Agents](#navigation-embodied-agents) | State-aware agent loop + recovery report | State → Navigation → Reasoning |
| [Humanoids & Locomotion](#humanoids-locomotion) | Motion protocol + safety and transfer gates | Locomotion → RL → Sim-to-Real |
| [Perception & World Models](#perception-world-models) | Uncertain state stream + predictive rollout | State → World model |
| [Simulation, Data & Evaluation](#simulation-data-evaluation) | Dataset datasheet + benchmark + promotion decision | Data → World model → Sim-to-Real |

Inspect the machine-readable route contract:

```bash
python scripts/run_learning_path.py --list
python scripts/run_learning_path.py --show foundation-models-vla
python scripts/run_learning_path.py --validate
```

<a id="foundation-models-vla"></a>
## 1. Foundation Models and VLA

**Research question.** How can language, vision, and robot state produce useful closed-loop actions?

- **Prerequisites:** [deep learning](../foundations/03-deep-learning-basics.md), [Transformers](../foundations/04-transformer-basics.md), [datasets and training](../foundations/10-dataset-and-training.md), [evaluation](../foundations/14-evaluation-and-reproducibility.md)
- **Pipeline sequence:** [Simulation and Data](../pipelines/01-simulation-data.md) → [VLA Policy](../pipelines/02-vla-policy.md) → [RFM and Cross-embodiment](../pipelines/05-rfm-cross-embodiment.md)
- **Deliverable:** a language-conditioned closed-loop policy, an adapter contract, and an ablation report
- **Measure:** `task_success_rate`, `language_condition_gap`, `inference_latency_ms`, `adapter_coverage`
- **Promotion gate:** beat the declared baseline under the same data and protocol, then pass action-schema and safety checks
- **Evidence boundary:** the RFM path is interface-tested; no competitive large-scale model reproduction is claimed

<a id="manipulation-imitation"></a>
## 2. Manipulation and Imitation Learning

**Research question.** How can demonstrations become a robust manipulation policy?

- **Prerequisites:** [kinematics and IK](../foundations/07-fk-jacobian-ik.md), [control](../foundations/08-control-basics.md), [datasets and training](../foundations/10-dataset-and-training.md), [evaluation](../foundations/14-evaluation-and-reproducibility.md)
- **Pipeline sequence:** [Simulation and Data](../pipelines/01-simulation-data.md) → [VLA Policy](../pipelines/02-vla-policy.md) → [RL Post-training](../pipelines/04-rl-post-training.md)
- **Deliverable:** a manipulation baseline with dataset diagnostics, closed-loop evaluation, and a failure taxonomy
- **Measure:** `task_success_rate`, `selection_accuracy`, `collision_rate`, `inference_latency_ms`
- **Promotion gate:** show a repeatable task gain across fixed seeds without increasing safety violations
- **Evidence boundary:** PushCube is a teaching-scale task, not evidence of general real-world manipulation

<a id="dexterity-teleoperation"></a>
## 3. Dexterity, Retargeting, and Teleoperation

**Research question.** How can human motion be transferred to a robot hand without confusing geometry, contact, and task evidence?

- **Prerequisites:** [SE(3)](../foundations/06-se3-and-rotation.md), [kinematics and IK](../foundations/07-fk-jacobian-ik.md), [optimization](../foundations/11-probability-and-optimization.md), [perception and sensors](../foundations/12-perception-and-sensors.md)
- **Pipeline sequence:** [Dexterous Retargeting](../pipelines/08-dexterous-retargeting.md) → [Perception and State](../pipelines/09-perception-state-estimation.md) → [Sim-to-Real](../pipelines/07-sim-to-real.md)
- **Deliverable:** a retargeted joint sequence with geometry, temporal, collision, task, and hardware evidence reported separately
- **Measure:** `retargeting_error`, `joint_limit_violation_rate`, `latency_ms`, `task_success_rate`
- **Promotion gate:** pass kinematic and temporal checks before contact simulation; use a separate gate before hardware
- **Evidence boundary:** the smoke test validates synthetic retargeting, not contact-rich task success or a real hand

<a id="navigation-embodied-agents"></a>
## 4. Navigation and Embodied Agents

**Research question.** How should an agent estimate state, plan, act, recover, and replan over long horizons?

- **Prerequisites:** [frames](../foundations/05-coordinate-transform.md), [control](../foundations/08-control-basics.md), [sensing](../foundations/12-perception-and-sensors.md), [systems and safety](../foundations/13-robot-systems-and-safety.md), [evaluation](../foundations/14-evaluation-and-reproducibility.md)
- **Pipeline sequence:** [Perception and State](../pipelines/09-perception-state-estimation.md) → [Navigation and Locomotion](../pipelines/10-navigation-locomotion.md) → [Embodied Reasoning](../pipelines/06-embodied-reasoning.md)
- **Deliverable:** a state-aware navigation loop with typed plans, safety events, recovery behavior, and a scenario report
- **Measure:** `localization_or_state_error`, `goal_success_rate`, `collision_or_fall_rate`, `recovery_success_rate`, `replan_count`
- **Promotion gate:** meet scenario-level success, collision, and recovery thresholds under fixed maps and perturbations
- **Evidence boundary:** the runnable path is synthetic grid navigation, not a SLAM, mobile-manipulation, or legged benchmark reproduction

<a id="humanoids-locomotion"></a>
## 5. Humanoids and Locomotion

**Research question.** How can motion policies track commands, recover, and remain inside safety envelopes?

- **Prerequisites:** [SE(3)](../foundations/06-se3-and-rotation.md), [kinematics](../foundations/07-fk-jacobian-ik.md), [control](../foundations/08-control-basics.md), [MuJoCo](../foundations/09-mujoco-basics.md), [systems and safety](../foundations/13-robot-systems-and-safety.md), [evaluation](../foundations/14-evaluation-and-reproducibility.md)
- **Pipeline sequence:** [Navigation and Locomotion](../pipelines/10-navigation-locomotion.md) → [RL Post-training](../pipelines/04-rl-post-training.md) → [Sim-to-Real](../pipelines/07-sim-to-real.md)
- **Deliverable:** a locomotion protocol with tracking, perturbation, recovery, safety, and transfer gates
- **Measure:** `path_or_velocity_tracking_error`, `collision_or_fall_rate`, `recovery_success_rate`, `sim_real_gap`
- **Promotion gate:** pass motion-only simulation and safety regression before hardware-in-the-loop or robot trials
- **Evidence boundary:** no reproduced humanoid policy or locally validated humanoid hardware result is claimed

<a id="perception-world-models"></a>
## 6. Perception and World Models

**Research question.** How can uncertain observations become state estimates and useful predictive rollouts?

- **Prerequisites:** [frames](../foundations/05-coordinate-transform.md), [probability](../foundations/11-probability-and-optimization.md), [perception and sensors](../foundations/12-perception-and-sensors.md), [evaluation](../foundations/14-evaluation-and-reproducibility.md)
- **Pipeline sequence:** [Perception and State](../pipelines/09-perception-state-estimation.md) → [World Model and Planning](../pipelines/03-world-model-planning.md)
- **Deliverable:** an uncertainty-aware state stream, a predictive rollout model, and calibration-to-planning error analysis
- **Measure:** `calibration_reprojection_error_px`, `sensor_sync_skew_ms`, `uncertainty_calibration_error`, `multi_step_rollout_error`, `planned_task_success_rate`
- **Promotion gate:** demonstrate calibrated uncertainty and improved downstream planning under the same observation protocol
- **Evidence boundary:** synthetic state and teaching-scale dynamics do not establish open-world visual prediction or real-sensor robustness

<a id="simulation-data-evaluation"></a>
## 7. Simulation, Data, and Evaluation

**Research question.** How can experiments produce traceable data and evidence that survives comparison?

- **Prerequisites:** [MuJoCo](../foundations/09-mujoco-basics.md), [datasets and training](../foundations/10-dataset-and-training.md), [systems and safety](../foundations/13-robot-systems-and-safety.md), [evaluation](../foundations/14-evaluation-and-reproducibility.md)
- **Pipeline sequence:** [Simulation and Data](../pipelines/01-simulation-data.md) → [World Model and Planning](../pipelines/03-world-model-planning.md) → [Sim-to-Real](../pipelines/07-sim-to-real.md)
- **Deliverable:** a versioned dataset datasheet, benchmark report, raw metrics, and an explicit deployment-promotion decision
- **Measure:** `dataset_coverage`, `data_integrity_rate`, `task_success_rate`, `robustness_gap`, `sim_real_gap`
- **Promotion gate:** retain protocol, seeds, data version, artifacts, negative results, and safety evidence before promotion
- **Evidence boundary:** repository audits verify committed contracts and artifacts, not external datasets or hardware deployment

## Research loop

For every route: freeze the question → choose one baseline → run the minimum path → retain raw artifacts → analyze failures → change one variable → rerun fixed evaluation → promote only when the declared gate passes.

The canonical route data lives in [`learning_paths/manifest.json`](../../learning_paths/manifest.json). Pipeline evidence status lives separately in [`pipelines/manifest.json`](../../pipelines/manifest.json).
