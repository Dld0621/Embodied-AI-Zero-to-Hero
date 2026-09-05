# Embodied AI: Zero to Hero

**English · [简体中文](README_CN.md)**

A bilingual, evidence-aware curriculum for understanding, building, and evaluating embodied intelligence—from mathematical prerequisites to closed-loop robot systems.

[![Tests](https://img.shields.io/github/actions/workflow/status/Dld0621/Embodied-AI-Zero-to-Hero/tests.yml?branch=master&label=tests)](https://github.com/Dld0621/Embodied-AI-Zero-to-Hero/actions/workflows/tests.yml)
[![Docs](https://img.shields.io/badge/docs-online-2563eb)](https://dld0621.github.io/Embodied-AI-Zero-to-Hero/)
[![Knowledge](https://img.shields.io/badge/knowledge_nodes-45-111827)](docs/knowledge-system/README.md)
[![Pipelines](https://img.shields.io/badge/engineering_pipelines-11-111827)](docs/pipelines/README.md)
[![License](https://img.shields.io/badge/original%20content-MIT-64748b)](LICENSE)
[![Third-party assets](https://img.shields.io/badge/third--party%20assets-mixed%20licenses-94a3b8)](THIRD_PARTY_NOTICES.md)

> [!IMPORTANT]
> Running code proves execution, not task performance. This repository keeps interface checks, synthetic smoke tests, teaching benchmarks, and hardware validation visibly separate.

## What this repository gives you

**[Concept-by-concept illustrated atlas (Chinese)](docs/knowledge-atlas/index.md)**: all 45 declared knowledge nodes are broken into smaller lessons with prerequisites, causal explanations, worked examples, offline diagrams, inspectable chart data, misconceptions, and expandable self-check answers. This companion is Chinese-led with English terminology; existing English courses remain available.

**[Open the interactive learning lab](https://dld0621.github.io/Embodied-AI-Zero-to-Hero/learning-lab/)**: adjust frames, joints, controller gains, action delays, and evaluation sample sizes without installing a robotics environment. Export your predictions, parameters and explanations. The [worked examples and transfer exercises](docs/learning-lab.md) remain readable on GitHub; the interactive controls run on the documentation site.

| Learn | Build | Evaluate |
|---|---|---|
| 45 prerequisite-linked knowledge nodes across 9 domains | 11 engineering Pipelines with inputs, stages, artifacts, and failure modes | Explicit metrics, promotion gates, provenance, and evidence boundaries |
| 14 foundation lessons from Python and SE(3) to safety and reproducibility | Runnable teaching paths for simulation, VLA, world models, RL, perception, navigation, and dexterity | A shared PushCube protocol plus repository-wide accuracy and regression checks |

This is a structured learning and engineering system. It is not a claim that every included baseline is state of the art, production-ready, or validated on real hardware.

<a id="start"></a>
## Start from your goal

| Your goal | First destination | What you should produce |
|---|---|---|
| Start with no background and know what to do in the first hour | [Start here](docs/start-here.md) | A first experiment card, failure record, and personal route |
| Learn the field from first principles | [Detailed curriculum](docs/curriculum.md) | Completed exercises, derivations, and a prerequisite trace |
| Configure a robotics workstation | [Environment setup](docs/setup/README.md) | A versioned environment receipt and layered smoke checks |
| Build one complete robot-learning system | [Pipeline catalog](docs/pipelines/README.md) | Inputs, artifacts, metrics, and a stage-resolved failure report |
| Specialize in VLA or WAM | [VLA and WAM specialization](docs/specializations/README.md) | A selected algorithm family, matched baseline, ablation matrix, and closed-loop evidence |
| Enter a research direction | [Seven research routes](docs/learning-paths/README.md) | A question, baseline, ablation plan, promotion gate, and evidence boundary |
| Find one missing prerequisite | [Knowledge system](docs/knowledge-system/README.md) | A dependency-ordered study path with an assessment target |

### Run a teaching Pipeline

```bash
git clone https://github.com/Dld0621/Embodied-AI-Zero-to-Hero.git
cd Embodied-AI-Zero-to-Hero
python -m pip install numpy
python scripts/run_pipeline.py --list
python scripts/run_pipeline.py --run simulation-data
```

Use the machine-readable maps when you already know the target:

```bash
python scripts/run_knowledge_map.py --path-to task-dexterity-teleoperation
python scripts/run_pipeline.py --show dexterous-manipulation
python scripts/run_learning_path.py --show dexterity-teleoperation
```

Generate an evidence-gated route from beginner to expert practice:

```bash
python scripts/run_curriculum.py --diagnose
python scripts/run_curriculum.py --plan full-stack-expert --hours-per-week 8
```

Progress is determined by artifacts and reviewed gates, not time spent. See the [assessment standard](docs/assessment.md), [three capstones](docs/capstone.md), and [85→100 curriculum audit](docs/CURRICULUM_AUDIT.md). The 100 score means this repository's curriculum-quality contracts are implemented and structurally checked; it is not universal expert certification or a hardware-performance claim.

<a id="knowledge"></a>
## Knowledge before recipes

The [knowledge system](docs/knowledge-system/README.md) is the prerequisite-level source of truth. Every node declares a learning outcome, prerequisites, an assessment, a primary document, Pipeline mappings, and a learner-evidence type.

| Stage | Capability | Exit evidence |
|---|---|---|
| **L0 · Computing** | Run, configure, inspect, and record an experiment | Reproducible command, configuration, and environment receipt |
| **L1 · Mathematical language** | Work with frames, uncertainty, objectives, and numerical limits | Derivation plus a numerical verification |
| **L2 · Robot loop** | Model, sense, estimate, control, and simulate | Closed-loop trace with units, rates, limits, and failure diagnosis |
| **L3 · Learning and prediction** | Build datasets, policies, world models, and planners | Train/evaluate artifact with leakage and uncertainty checks |
| **L4 · Task systems** | Compose manipulation, dexterity, navigation, and locomotion | Task protocol with stage-level failures and recovery behavior |
| **L5 · Evidence and deployment** | Compare, generalize, and decide whether risk may increase | Reproducible report and an explicit promotion or stop decision |

![Knowledge dependencies across six stages](docs/assets/knowledge-system.svg)

The [detailed curriculum](docs/curriculum.md) converts these stages into learner, engineer, and researcher tracks with concrete checkpoints. The graph itself remains machine-readable in [`knowledge/manifest.json`](knowledge/manifest.json).

## VLA and WAM from zero to research

The [VLA and WAM specialization](docs/specializations/README.md) teaches the two directions as separate, dependency-ordered tracks. It covers data and action contracts, multimodal fusion, discrete and continuous action generation, diffusion and flow objectives, world-model planning baselines, joint video-action models, algorithm selection, ablations, and closed-loop evaluation.

| Track | Start with | Advance only when |
|---|---|---|
| [VLA Zero to One](docs/specializations/vla-zero-to-one.md) | Chunked behavior cloning, then language conditioning | The matched policy baseline is reproducible and language/vision ablations pass |
| [WAM Zero to One](docs/specializations/wam-zero-to-one.md) | Action-conditioned dynamics plus MPC | Rollout and planning baselines pass before joint video-action scaling |

Use the explainable selector to compare algorithm families under your actual goal, data, compute, and latency constraints:

```bash
python scripts/select_vla_wam_algorithm.py --goal language-generalization --compute single-gpu --data task-specific --latency hard
```

The selector is a learning and experiment-design aid, not a model leaderboard or deployment guarantee.

<a id="system"></a>
## The system is a closed loop

![Closed-loop embodied AI system from observation to evaluated action](assets/system_architecture.svg)

| Layer | Core question | Required output |
|---|---|---|
| Perception and state | What is happening in the world and robot? | Time-aligned observations, calibrated frames, and uncertainty |
| Policy, reasoning, and prediction | Which action should happen next, and what may follow? | Goal, plan, action representation, and predicted risk |
| Control and safety | How is the action executed within physical and operational limits? | Bounded commands, watchdogs, stop paths, and logs |
| Evaluation and learning | Did the task work, generalize, and remain safe? | Metrics, failure taxonomy, comparison, and updated policy |

<a id="pipelines"></a>
## Eleven engineering Pipelines

Each Pipeline specifies prerequisites, inputs, stages, artifacts, metrics, promotion gates, and failure modes. Status describes repository evidence only.

| Pipeline | Closed loop | Current repository evidence |
|---|---|---|
| [Simulation and data](docs/pipelines/01-simulation-data.md) | task → simulator → expert → episodes → QA | Smoke-tested |
| [VLA policy](docs/pipelines/02-vla-policy.md) | image + language + state → policy → evaluation | Teaching baseline smoke-tested |
| [World model and planning](docs/pipelines/03-world-model-planning.md) | transitions → dynamics → rollout → planning | Model smoke-tested |
| [RL post-training](docs/pipelines/04-rl-post-training.md) | MDP → reward → PPO → regression | Teaching baseline smoke-tested |
| [Robot foundation models](docs/pipelines/05-rfm-cross-embodiment.md) | canonical observation → adapter → action → safety | Interface-tested |
| [Embodied reasoning](docs/pipelines/06-embodied-reasoning.md) | instruction → plan → skills → feedback → replan | Interface-tested |
| [Sim-to-Real](docs/pipelines/07-sim-to-real.md) | robustness → HIL → shadow mode → guarded rollout | Documented; hardware-dependent |
| [Dexterous retargeting](docs/pipelines/08-dexterous-retargeting.md) | landmarks → geometry → optimization → smoothing | Synthetic smoke-tested |
| [Perception and state estimation](docs/pipelines/09-perception-state-estimation.md) | calibration → synchronization → fusion → uncertainty | Synthetic smoke-tested |
| [Navigation and locomotion](docs/pipelines/10-navigation-locomotion.md) | state → map/terrain → planning → control → recovery | Grid-navigation smoke-tested |
| [Dexterous fine manipulation](docs/pipelines/11-dexterous-manipulation.md) | state → pre-grasp → contact → lift → hold/recover | Abstract contact-dynamics smoke-tested |

```bash
python scripts/run_pipeline.py --validate
python scripts/run_pipeline.py --run vla-policy --dry-run
python scripts/run_pipeline.py --run world-model-planning
```

The executable contract lives in [`pipelines/manifest.json`](pipelines/manifest.json). Synthetic paths verify scoped behavior and wiring; they do not reproduce large-scale or hardware results.

## Seven research routes

| Direction | Pipeline sequence | Required research artifact |
|---|---|---|
| [Foundation Models and VLA](docs/learning-paths/README.md#foundation-models-vla) | Data → VLA → RFM | Policy, adapter, baseline, and ablation |
| [Manipulation and Imitation](docs/learning-paths/README.md#manipulation-imitation) | Data → VLA → RL | Closed-loop baseline and failure taxonomy |
| [Dexterity and Teleoperation](docs/learning-paths/README.md#dexterity-teleoperation) | Retargeting → State → Grasp → Sim-to-Real | Motion, contact, retention, and task evidence |
| [Navigation and Embodied Agents](docs/learning-paths/README.md#navigation-embodied-agents) | State → Navigation → Reasoning | Agent loop, recovery protocol, and report |
| [Humanoids and Locomotion](docs/learning-paths/README.md#humanoids-locomotion) | Locomotion → RL → Sim-to-Real | Motion protocol, robustness tests, and safety gates |
| [Perception and World Models](docs/learning-paths/README.md#perception-world-models) | State → World model | Uncertain state estimate and predictive rollout |
| [Simulation, Data, and Evaluation](docs/learning-paths/README.md#simulation-data-evaluation) | Data → World model → Sim-to-Real | Datasheet, benchmark, and promotion decision |

<a id="evidence"></a>
## Evidence before claims

| Evidence level | What it supports | What it does not support |
|---|---|---|
| **Import/interface check** | Modules and schemas connect | Useful behavior or correct physics |
| **Synthetic smoke test** | A scoped path executes deterministically | Generalization, benchmark quality, or real-world transfer |
| **Teaching benchmark** | A fixed protocol produced a recorded result | State-of-the-art status or transfer to another setup |
| **Hardware validation** | A named system passed a bounded physical protocol | Safety or performance outside that protocol |

The included PushCube results are a teaching snapshot with unequal training and evaluation budgets, not a controlled leaderboard. The structured-state BC baseline reaches a useful score while several vision-policy baselines remain negative results. Read the exact protocol, per-method budgets, and raw-artifact boundaries in [`BENCHMARK.md`](BENCHMARK.md) and the [benchmark report](docs/benchmark_report.md).

<a id="docs"></a>
## Documentation map

| Learn | Build | Verify |
|---|---|---|
| [Detailed curriculum](docs/curriculum.md) | [Environment setup](docs/setup/README.md) | [Validation policy](docs/VALIDATION.md) |
| [Start here](docs/start-here.md) | [Learner templates](learner/README.md) | [Curriculum audit](docs/CURRICULUM_AUDIT.md) |
| [Knowledge system](docs/knowledge-system/README.md) | [MuJoCo scene building](docs/tutorials/mujoco-scene-building.md) | [Claim review](docs/CLAIM_REVIEW.md) |
| [Foundation lessons](docs/foundations/README_EN.md) | [Pipeline catalog](docs/pipelines/README.md) | [Primary sources](docs/SOURCES.md) |
| [VLA and WAM specialization](docs/specializations/README.md) | [Algorithm-family catalog](learning_tracks/vla_wam_algorithms.json) | [VLA/WAM evidence boundaries](docs/specializations/README.md#evidence-boundary) |
| [Field map](docs/field-map.md) | [Research routes](docs/learning-paths/README.md) | [Benchmark protocol](BENCHMARK.md) |

### Verify the repository

```bash
python scripts/check_markdown_links.py
python scripts/check_markdown_format.py
python scripts/check_claims.py
python scripts/run_knowledge_map.py --validate
python scripts/run_pipeline.py --validate
python scripts/run_learning_path.py --validate
python scripts/run_curriculum.py --validate
python scripts/audit_repository.py
python -m pytest tests/ -q
```

The evidence ladder is import → smoke → deterministic test → benchmark → hardware validation. Retain the command, seed, commit, data version, checkpoint, hardware, episode count, and machine-readable metrics for every result.

## Contributing and license

Read [`CONTRIBUTING.md`](CONTRIBUTING.md) before proposing a lesson, Pipeline, benchmark claim, or robot adapter. Original project content uses the [MIT License](LICENSE); bundled upstream assets retain their own terms in [Third-Party Notices](THIRD_PARTY_NOTICES.md).

Maintained by [Gangwei Li](https://github.com/Dld0621).
