# Embodied AI Pipeline Catalog

Each Pipeline composes multiple nodes from the [45-node Knowledge System](../knowledge-system/README.md). Resolve a target node when a prerequisite label is too broad; the Pipeline remains the source of truth for executable stages, artifacts, metrics, and promotion gates.

This catalog turns the repository's topic chapters into end-to-end engineering loops. Each track defines prerequisites, inputs, stages, artifacts, metrics, and a promotion gate. The commands are registered in the machine-readable [`pipelines/manifest.json`](../../pipelines/manifest.json).

Need a goal before a track? Start from the [seven research routes](../learning-paths/README.md), which combine these Pipelines into outcome-oriented experiment briefs.

## Quick start

```bash
python scripts/run_pipeline.py --list
python scripts/run_pipeline.py --validate
python scripts/run_pipeline.py --show vla-policy
python scripts/run_pipeline.py --run vla-policy --dry-run
python scripts/run_pipeline.py --run vla-policy
```

Use the smoke command to verify interfaces and data flow. Use `--full` only after the smoke run passes and the expected compute budget is available.

<div class="dof-concept" role="group" aria-label="How to read an engineering pipeline">
  <span class="dof-concept__eyebrow">Reading key</span>
  <p class="dof-concept__title">A pipeline is a contract, not just a command: follow the artifact and its promotion gate.</p>
  <div class="dof-stage-flow">
    <div class="dof-stage dof-stage--input"><span>01 · INPUT</span><strong>Task contract</strong><small>frames, units, data, limits, seed</small></div>
    <i class="dof-flow-arrow" aria-hidden="true">→</i>
    <div class="dof-stage"><span>02 · BUILD</span><strong>Executable loop</strong><small>stages, controller, model, checks</small></div>
    <i class="dof-flow-arrow" aria-hidden="true">→</i>
    <div class="dof-stage"><span>03 · RETAIN</span><strong>Artifacts</strong><small>metrics, replay, checkpoint, report</small></div>
    <i class="dof-flow-arrow" aria-hidden="true">→</i>
    <div class="dof-stage dof-stage--gate"><span>04 · DECIDE</span><strong>Promotion gate</strong><small>advance, revise, or stop with evidence</small></div>
  </div>
</div>

## Tracks

| Track | Closed loop | Current evidence | Guide |
|---|---|---|---|
| Simulation & data | task definition → simulator → expert → episodes → quality checks | smoke-tested | [Open](01-simulation-data.md) |
| VLA policy | multimodal data → policy training → closed-loop evaluation | smoke-tested teaching baseline | [Open](02-vla-policy.md) |
| World model | transitions → dynamics model → rollout → planning | smoke-tested model; planning is a separate stage | [Open](03-world-model-planning.md) |
| RL post-training | MDP → reward → PPO → evaluation → regression | smoke-tested teaching baseline | [Open](04-rl-post-training.md) |
| Robot foundation model | observation schema → adapter → action chunk → safety layer | interface-tested mock adapter | [Open](05-rfm-cross-embodiment.md) |
| Embodied reasoning | instruction → task plan → skills → feedback → replan | interface-tested rule planner | [Open](06-embodied-reasoning.md) |
| Sim-to-real | simulation → robustness → HIL → shadow → guarded deployment | documented; hardware-dependent | [Open](07-sim-to-real.md) |
| Dexterous retargeting | landmarks → geometry → IK/optimization → smoothing → evaluation | smoke-tested synthetic input | [Open](08-dexterous-retargeting.md) |
| Perception & state estimation | calibration → synchronization → fusion → uncertainty → validation | deterministic synthetic smoke test | [Open](09-perception-state-estimation.md) |
| Navigation & locomotion | state → map/terrain → planning → control → recovery | deterministic grid-navigation smoke test | [Open](10-navigation-locomotion.md) |
| Dexterous grasping & fine manipulation | state → pre-grasp → approach → contact → lift → hold/recover | abstract MuJoCo contact-dynamics smoke test | [Open](11-dexterous-manipulation.md) |

Chinese navigation: [Pipeline 总览（中文）](README_CN.md).

## Evidence labels

- **smoke-tested**: the repository contains a lightweight executable path.
- **interface-tested**: local schemas/adapters can be checked without model weights or hardware.
- **documented**: the engineering gates are defined, but no universal local command can represent the real system.
- **experimental**: an exploratory path that should not be treated as a validated baseline.

Passing a smoke test verifies wiring, not research quality or real-robot safety. Report benchmark results with the configuration, seed, hardware, checkpoint, and result artifact.
