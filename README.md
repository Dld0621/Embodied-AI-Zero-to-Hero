<h1 align="center">Embodied AI · Zero to Hero</h1>

<p align="center">
  <b>English</b> · <a href="README_CN.md">简体中文</a>
</p>

<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="assets/dof-hero-dark.svg">
    <img src="assets/dof-hero.svg" alt="Learn the embodied intelligence loop and build the evidence" width="100%">
  </picture>
</p>

<p align="center">
  <a href="#start"><b>Start</b></a>&nbsp;&nbsp;·&nbsp;&nbsp;
  <a href="#knowledge"><b>Knowledge</b></a>&nbsp;&nbsp;·&nbsp;&nbsp;
  <a href="#system"><b>System</b></a>&nbsp;&nbsp;·&nbsp;&nbsp;
  <a href="#pipelines"><b>Pipelines</b></a>&nbsp;&nbsp;·&nbsp;&nbsp;
  <a href="#routes"><b>Research</b></a>&nbsp;&nbsp;·&nbsp;&nbsp;
  <a href="#evidence"><b>Evidence</b></a>&nbsp;&nbsp;·&nbsp;&nbsp;
  <a href="#docs"><b>Docs</b></a>
</p>

<p align="center">
  <a href="https://github.com/Dld0621/Embodied-AI-Zero-to-Hero/actions/workflows/tests.yml"><img src="https://img.shields.io/github/actions/workflow/status/Dld0621/Embodied-AI-Zero-to-Hero/tests.yml?branch=master&style=flat&label=build" alt="Build status"></a>
  <a href="https://github.com/Dld0621/Embodied-AI-Zero-to-Hero"><img src="https://img.shields.io/github/stars/Dld0621/Embodied-AI-Zero-to-Hero?style=flat&label=stars" alt="GitHub stars"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/original%20content-MIT-4F7CFF?style=flat" alt="Original project content uses the MIT License"></a>
  <a href="THIRD_PARTY_NOTICES.md"><img src="https://img.shields.io/badge/third--party%20assets-mixed%20licenses-6B7280?style=flat" alt="Third-party assets use mixed licenses"></a>
</p>

<p align="center">
  <b>A bilingual, evidence-aware learning and research system for embodied intelligence.</b><br>
  <sub>Understand the prerequisites. Run the loop. Measure the result. Respect the deployment boundary.</sub>
</p>

| **45** knowledge nodes | **9** domains | **14** foundation lessons | **11** engineering pipelines | **7** research routes |
|:---:|:---:|:---:|:---:|:---:|
| Dependency graph | Capability map | Concept to exercise | Data to deployment | Question to evidence |

> [!IMPORTANT]
> A script completing is evidence of execution—not proof of useful task performance. This repository keeps **interface checks**, **synthetic smoke tests**, **teaching benchmarks**, and **hardware validation** visibly separate.

<a id="start"></a>
## Choose Your Entry

| Learn | Set up | Build | Research |
|:---|:---|:---|:---|
| Resolve prerequisites and assessment targets. | Create a reviewed robotics workstation. | Run one closed-loop engineering track. | Turn a question into an experiment contract. |
| [Knowledge system →](docs/knowledge-system/README.md) | [Environment guide →](docs/setup/README.md) | [Pipeline catalog →](docs/pipelines/README.md) | [Research routes →](docs/learning-paths/README.md) |

### One-minute first run

```bash
git clone https://github.com/Dld0621/Embodied-AI-Zero-to-Hero.git
cd Embodied-AI-Zero-to-Hero
pip install numpy
python scripts/run_pipeline.py --run simulation-data
```

### Navigate by intent

```bash
# I know the capability I want to learn.
python scripts/run_knowledge_map.py --path-to task-dexterity-teleoperation

# I know the system I want to build.
python scripts/run_pipeline.py --list
python scripts/run_pipeline.py --show dexterous-manipulation

# I know the research question I want to investigate.
python scripts/run_learning_path.py --list
python scripts/run_learning_path.py --show dexterity-teleoperation
```

<a id="knowledge"></a>
## Knowledge, Connected

<p align="center">
  <img src="docs/assets/knowledge-system.svg" alt="Embodied AI knowledge system with 45 nodes, 9 domains, and 6 stages" width="100%">
</p>

The [bilingual knowledge system](docs/knowledge-system/README.md) is the prerequisite-level source of truth. Every node declares an outcome, assessment, lesson, Pipeline mapping, and learner-evidence type.

| L0 · Tools | L1 · Math | L2 · Robot loop | L3 · Learning | L4 · Tasks | L5 · Evidence |
|:---|:---|:---|:---|:---|:---|
| Run and record | Derive and verify | Sense, estimate, control | Data, policy, prediction | Compose and recover | Compare and gate risk |

The graph connects the 14 lessons, 11 Pipelines, and 7 research routes. It helps a learner move from one missing prerequisite to a measurable artifact without guessing the intermediate steps.

<a id="system"></a>
## One Closed Loop

<p align="center">
  <img src="assets/system_architecture.svg" alt="Closed-loop embodied AI system from observation to evaluated action" width="100%">
</p>

| Layer | Question | Output |
|:---|:---|:---|
| **Perception & state** | What is happening in the world and robot? | Synchronized observations with uncertainty |
| **Reasoning, policy & prediction** | Which goal, action, and consequence come next? | Plan, action chunk, predicted risk |
| **Control & safety** | How is the action executed within constraints? | Bounded command to simulator or robot |
| **Evaluation & learning** | Did it work, generalize, and remain safe? | Evidence, diagnosis, updated policy |

<a id="pipelines"></a>
## Eleven Engineering Pipelines

Each contract specifies prerequisites, inputs, stages, artifacts, metrics, promotion gates, and failure modes. Status labels describe repository evidence only.

| System track | Closed loop | Repository evidence |
|:---|:---|:---|
| [Simulation & Data](docs/pipelines/01-simulation-data.md) | task → simulator → expert → episodes → QA | Smoke-tested |
| [VLA Policy](docs/pipelines/02-vla-policy.md) | image + language + state → policy → evaluation | Teaching baseline smoke-tested |
| [World Model](docs/pipelines/03-world-model-planning.md) | transitions → dynamics → rollout → planning | Model smoke-tested |
| [RL Post-training](docs/pipelines/04-rl-post-training.md) | MDP → reward → PPO → regression | Teaching baseline smoke-tested |
| [Robot Foundation Models](docs/pipelines/05-rfm-cross-embodiment.md) | canonical observation → adapter → action → safety | Interface-tested |
| [Embodied Reasoning](docs/pipelines/06-embodied-reasoning.md) | instruction → plan → skills → feedback → replan | Interface-tested |
| [Sim-to-Real](docs/pipelines/07-sim-to-real.md) | robustness → HIL → shadow mode → guarded rollout | Documented; hardware-dependent |
| [Dexterous Retargeting](docs/pipelines/08-dexterous-retargeting.md) | landmarks → geometry → optimization → smoothing | Synthetic smoke-tested |
| [Perception & State Estimation](docs/pipelines/09-perception-state-estimation.md) | calibration → synchronization → fusion → uncertainty | Synthetic smoke-tested |
| [Navigation & Locomotion](docs/pipelines/10-navigation-locomotion.md) | state → map/terrain → planning → control → recovery | Grid-navigation smoke-tested |
| [Dexterous Fine Manipulation](docs/pipelines/11-dexterous-manipulation.md) | state → pre-grasp → contact → lift → hold/recover | Abstract contact-dynamics smoke-tested |

```bash
python scripts/run_pipeline.py --validate
python scripts/run_pipeline.py --run vla-policy --dry-run
python scripts/run_pipeline.py --run world-model-planning
```

The machine-readable contract is [`pipelines/manifest.json`](pipelines/manifest.json). Synthetic paths test wiring and scoped behavior; they do not reproduce large-scale or real-world results.

<a id="routes"></a>
## Seven Research Routes

The [route map](docs/learning-paths/README.md) turns each direction into a question, prerequisite set, Pipeline sequence, deliverable, metrics, promotion gate, and evidence boundary.

| Research direction | Pipeline sequence | Required output |
|:---|:---|:---|
| [Foundation Models & VLA](docs/learning-paths/README.md#foundation-models-vla) | Data → VLA → RFM | Policy + adapter + ablation |
| [Manipulation & Imitation](docs/learning-paths/README.md#manipulation-imitation) | Data → VLA → RL | Closed-loop baseline + failure taxonomy |
| [Dexterity & Teleoperation](docs/learning-paths/README.md#dexterity-teleoperation) | Retargeting → State → Grasp → Sim-to-Real | Motion + contact/task evidence |
| [Navigation & Embodied Agents](docs/learning-paths/README.md#navigation-embodied-agents) | State → Navigation → Reasoning | Agent loop + recovery report |
| [Humanoids & Locomotion](docs/learning-paths/README.md#humanoids-locomotion) | Locomotion → RL → Sim-to-Real | Motion protocol + safety gates |
| [Perception & World Models](docs/learning-paths/README.md#perception-world-models) | State → World model | Uncertain state + predictive rollout |
| [Simulation, Data & Evaluation](docs/learning-paths/README.md#simulation-data-evaluation) | Data → World model → Sim-to-Real | Datasheet + benchmark + promotion decision |

| 01 · Prerequisites | 02 · Foundations | 03 · Pipelines | 04 · Research |
|:---:|:---:|:---:|:---:|
| Resolve the missing node | Learn and verify | Produce artifacts and metrics | Reproduce, ablate, compare |
| [45-node graph](docs/knowledge-system/README.md) | [14-lesson roadmap](docs/foundations/00-roadmap.md) | [11 contracts](docs/pipelines/README.md) | [7 routes](docs/learning-paths/README.md) |

<a id="evidence"></a>
## Evidence Before Claims

| Label | What it proves | What it does not prove |
|:---|:---|:---|
| **Smoke-tested** | The minimum path runs to completion. | The method reaches a useful task score. |
| **Interface-tested** | Schemas, shapes, and adapters connect. | Real weights or hardware are validated. |
| **Benchmark** | A fixed protocol produced a recorded result. | The result transfers to another setup. |
| **Hardware-dependent** | The gate requires a specific robot and safety process. | Simulation is sufficient authorization. |

<details>
<summary><b>Open the teaching-scale PushCube benchmark snapshot</b></summary>

All methods share a dual-cube language-conditioned task, but budgets and evaluation counts vary. This is a teaching benchmark—not a controlled leaderboard.

| Method | Input | Data / compute | Eval episodes | Success rate |
|:---|:---|:---|---:|---:|
| Expert | State | Heuristic / CPU | 50 | **~100%** |
| State-BC | 14-D state | 100 episodes / CPU | 100 | **90%** |
| RL, BC-init PPO | 14-D state | 500 episodes / CPU | 20 | **15%** |
| VLA | RGB + language | 100 episodes / CPU | 100 | **0%** |
| WM-MPC | 14-D state | 100 episodes / CPU | 20 | **0%** |
| SmolVLA 450M | RGB + language + state | 50 episodes, 10K steps / GPU | 20 | **0%** |
| Action Chunking / Diffusion | RGB + language | 100 episodes / CPU | — | **N/A** |

The structured-state baseline shows that this task is learnable. The vision-policy gap is a negative result that motivates data, representation, and closed-loop diagnosis. Raw artifacts and scope notes live in [`BENCHMARK.md`](BENCHMARK.md) and the [benchmark report](docs/benchmark_report.md).

</details>

<details>
<summary><b>Open the shared task contract and platform boundary</b></summary>

PushCube fixes a 128×128 RGB + language observation, optional 14-D state, a 2-D end-effector delta action, and correct-cube task metrics. It lets policies change while the teaching task stays fixed.

The repository contains a verified local PushCube environment, experimental Franka and AgiBot adapters, and planned UR5e and Unitree paths. It does **not** claim locally reproduced real-robot performance for these external platforms.

Core entry points: [`environment`](examples/unified_pushcube_env.py) · [`VLA`](examples/unified_pushcube_vla.py) · [`world model`](examples/unified_pushcube_wm.py) · [`RL`](examples/unified_pushcube_rl.py)

</details>

<a id="docs"></a>
## Documentation

| Start with | Continue with | Verify with |
|:---|:---|:---|
| [Published documentation](https://dld0621.github.io/Embodied-AI-Zero-to-Hero/) | [Field map](docs/field-map.md) | [Evidence policy](docs/VALIDATION.md) |
| [Knowledge system](docs/knowledge-system/README.md) | [Foundation lessons](docs/foundations/README_EN.md) | [Claim review](docs/CLAIM_REVIEW.md) |
| [Environment setup](docs/setup/README.md) | [MuJoCo scene building](docs/tutorials/mujoco-scene-building.md) | [Primary sources](docs/SOURCES.md) |
| [Pipeline catalog](docs/pipelines/README.md) | [Robot foundation models](docs/23-robot-foundation-models.md) | [Benchmark protocol](BENCHMARK.md) |
| [Research routes](docs/learning-paths/README.md) | [Frontier paper guide](docs/18-frontier-papers-online.md) | [Security policy](SECURITY.md) |

<details>
<summary><b>Open the repository map</b></summary>

```text
Embodied-AI-Zero-to-Hero/
├─ docs/                  Lessons, guides, research routes, validation
├─ knowledge/             Machine-readable prerequisite graph
├─ pipelines/             Machine-readable engineering contracts
├─ learning_paths/        Machine-readable research-route contracts
├─ examples/              Runnable teaching and research baselines
├─ benchmarks/ + results/ Evaluation entry points and recorded artifacts
├─ tools/robotdev/        Read-only workstation checks and stack resolver
├─ scripts/ + tests/      Discovery, validation, CI, and regressions
└─ assets/                Bilingual interface and explanatory diagrams
```

</details>

### Reproduce the repository checks

```bash
python scripts/check_markdown_links.py
python scripts/check_markdown_format.py
python scripts/check_claims.py
python scripts/run_knowledge_map.py --validate
python scripts/run_pipeline.py --validate
python scripts/run_learning_path.py --validate
python scripts/audit_repository.py
python -m pytest tests/ -q
```

The evidence ladder is import → smoke → deterministic test → benchmark → hardware validation. Retain the command, seed, commit, data version, checkpoint, hardware, episode count, and machine-readable metrics for every result.

## Contribute

Read [`CONTRIBUTING.md`](CONTRIBUTING.md) before proposing a tutorial, Pipeline, benchmark claim, or robot adapter. Reproducible baselines, failing cases, bilingual improvements, new embodiments, and evidence-backed corrections are especially valuable.

Original project content is available under the [MIT License](LICENSE). Bundled upstream code, models, and assets retain their own terms; review [Third-Party Notices](THIRD_PARTY_NOTICES.md) before reuse. Citation metadata is in [`CITATION.cff`](CITATION.cff).

<p align="center">
  <b>Learn the loop. Build the evidence.</b><br>
  Maintained by <a href="https://github.com/Dld0621">Gangwei Li</a>
</p>
