# Embodied AI Field Map

[中文版](field-map-cn.md){ .md-button }

Embodied AI is a closed-loop systems field, not a single model family. This map separates **capability coverage** from **repository evidence** so that a topic name never implies a reproduced result.

## Capability stack

| Capability | Engineering question | Best foundation | Pipeline contract |
|---|---|---|---|
| Sensing & calibration | Are observations synchronized, calibrated, and healthy? | [Perception and sensors](foundations/12-perception-and-sensors.md) | [Perception & state estimation](pipelines/09-perception-state-estimation.md) |
| State estimation | What state is safe to use, with what uncertainty? | [Probability and optimization](foundations/11-probability-and-optimization.md) | [Perception & state estimation](pipelines/09-perception-state-estimation.md) |
| Data & simulation | How are tasks, demonstrations, splits, and perturbations produced? | [MuJoCo](foundations/09-mujoco-basics.md) · [Dataset and training](foundations/10-dataset-and-training.md) | [Simulation & data](pipelines/01-simulation-data.md) |
| Policy learning | How does observation and language become action? | [Deep learning](foundations/03-deep-learning-basics.md) · [Transformers](foundations/04-transformer-basics.md) | [VLA policy](pipelines/02-vla-policy.md) |
| Predictive models | What will happen after an action? | [Probability and optimization](foundations/11-probability-and-optimization.md) | [World model planning](pipelines/03-world-model-planning.md) |
| Interactive improvement | How should behavior improve from reward and failure? | [Control](foundations/08-control-basics.md) | [RL post-training](pipelines/04-rl-post-training.md) |
| Generalist policies | How are models adapted across datasets and bodies? | [Dataset and training](foundations/10-dataset-and-training.md) | [RFM & cross-embodiment](pipelines/05-rfm-cross-embodiment.md) |
| Task reasoning | How is a long instruction decomposed and replanned? | [Robot systems and safety](foundations/13-robot-systems-and-safety.md) | [Embodied reasoning](pipelines/06-embodied-reasoning.md) |
| Manipulation & dexterity | How are geometric or learned commands mapped to constrained motion? | [FK, Jacobian, and IK](foundations/07-fk-jacobian-ik.md) | [Dexterous retargeting](pipelines/08-dexterous-retargeting.md) |
| Navigation & locomotion | How does an embodiment move while remaining localized and stable? | [Control](foundations/08-control-basics.md) · [Systems and safety](foundations/13-robot-systems-and-safety.md) | [Navigation & locomotion](pipelines/10-navigation-locomotion.md) |
| Transfer & deployment | What must pass before risk is increased? | [Evaluation and reproducibility](foundations/14-evaluation-and-reproducibility.md) | [Sim-to-Real](pipelines/07-sim-to-real.md) |

## Evidence today

| Level | Tracks | Meaning |
|---|---|---|
| **Smoke-tested** | Simulation/data, VLA, world model, RL, dexterous retargeting, perception/state, navigation | A lightweight repository path completes; synthetic wiring and performance claims remain separate. |
| **Interface-tested** | RFM/cross-embodiment, embodied reasoning | Local schemas, adapters, or planners connect without proving real weights or hardware. |
| **Documented** | Sim-to-Real | The engineering contract and gates exist; hardware deployment cannot be represented by a universal local command. |

## Choose by research goal

| Goal | Start | Then prove |
|---|---|---|
| Learn robot learning from zero | [Foundations overview](foundations/README_EN.md) | Complete one smoke-tested pipeline and retain its artifacts. |
| Build a multimodal policy | [VLA pipeline](pipelines/02-vla-policy.md) | Closed-loop success, language ablation, latency, and failure cases. |
| Study prediction and planning | [World-model pipeline](pipelines/03-world-model-planning.md) | Multi-step rollout error and planned task success separately. |
| Work across robot bodies | [RFM pipeline](pipelines/05-rfm-cross-embodiment.md) | Action semantics, adapter coverage, and per-embodiment results. |
| Study dexterous hands | [Retargeting pipeline](pipelines/08-dexterous-retargeting.md) | Geometry, temporal quality, contact/task evidence, and hardware evidence separately. |
| Build mobile or legged systems | [Navigation/locomotion contract](pipelines/10-navigation-locomotion.md) | Localization, tracking, collision/fall, recovery, and transfer evidence. |

## Deliberate non-claims

The repository does **not** currently claim a reproduced SLAM or standard navigation benchmark, legged-locomotion policy, general-purpose hardware deployment, or competitive large-scale foundation-model result. The new system smokes are deterministic synthetic fixtures, not substitutes for those results.
