<div class="dof-landing" markdown="1">
  <div class="dof-kicker">Embodied AI · Open research stack</div>

# From first principles to physical action.

  <p class="dof-lead">
    Learn the foundations, execute complete pipelines, and judge every result by its evidence. One bilingual system for understanding and building embodied intelligence.
  </p>

  <div class="dof-actions">
    <a class="dof-button dof-button--primary" href="foundations/README_EN/">Start learning</a>
    <a class="dof-button" href="field-map/">View field map</a>
    <a class="dof-button" href="pipelines/">Explore pipelines</a>
    <a class="dof-button" href="index_cn/">中文</a>
  </div>
</div>

<div class="dof-loop" aria-label="Embodied intelligence closed loop">
  <div><span>01</span><strong>Observe</strong><small>Sensors and state</small></div>
  <i>→</i>
  <div><span>02</span><strong>Understand</strong><small>Goals and world models</small></div>
  <i>→</i>
  <div><span>03</span><strong>Act</strong><small>Policy, control and safety</small></div>
  <i>→</i>
  <div><span>04</span><strong>Learn</strong><small>Feedback and evidence</small></div>
</div>

<div class="dof-metrics">
  <div class="dof-metric"><strong>14</strong><span>Foundation lessons</span></div>
  <div class="dof-metric"><strong>10</strong><span>Engineering pipelines</span></div>
  <div class="dof-metric"><strong>7</strong><span>Runnable smokes</span></div>
  <div class="dof-metric"><strong>EN · 中文</strong><span>Bilingual entry</span></div>
</div>

## Choose your route

<div class="dof-grid">
  <a class="dof-card" href="foundations/README_EN/">
    <span class="dof-card__index">01 · LEARN</span>
    <h3>Build the mental model</h3>
    <p>Math, learning, frames, kinematics, sensing, control, systems, safety and evaluation.</p>
  </a>
  <a class="dof-card" href="field-map/">
    <span class="dof-card__index">02 · ORIENT</span>
    <h3>Choose a research direction</h3>
    <p>See capabilities, prerequisites, pipeline contracts, evidence and deliberate non-claims in one map.</p>
  </a>
  <a class="dof-card" href="pipelines/">
    <span class="dof-card__index">03 · BUILD</span>
    <h3>Run one complete system</h3>
    <p>Start from inputs, execute every stage, retain artifacts, and check an explicit promotion gate.</p>
  </a>
  <a class="dof-card" href="benchmark_report/">
    <span class="dof-card__index">04 · MEASURE</span>
    <h3>Compare with context</h3>
    <p>Read protocol, data budget, episode count, negative results and raw-artifact boundaries together.</p>
  </a>
</div>

## System coverage today

<div class="dof-coverage">
  <a href="pipelines/01-simulation-data/"><span class="dof-status dof-status--smoke">SMOKE</span><strong>Simulation & data</strong><small>Task → expert → episodes → QA</small></a>
  <a href="pipelines/02-vla-policy/"><span class="dof-status dof-status--smoke">SMOKE</span><strong>VLA policy</strong><small>Multimodal data → policy → closed loop</small></a>
  <a href="pipelines/03-world-model-planning/"><span class="dof-status dof-status--smoke">SMOKE</span><strong>World models</strong><small>Transitions → prediction → rollout → planning</small></a>
  <a href="pipelines/04-rl-post-training/"><span class="dof-status dof-status--smoke">SMOKE</span><strong>RL post-training</strong><small>MDP → PPO → evaluation → regression</small></a>
  <a href="pipelines/05-rfm-cross-embodiment/"><span class="dof-status dof-status--interface">INTERFACE</span><strong>RFM / cross-embodiment</strong><small>Canonical schema → adapter → safety</small></a>
  <a href="pipelines/06-embodied-reasoning/"><span class="dof-status dof-status--interface">INTERFACE</span><strong>Embodied reasoning</strong><small>Instruction → subgoals → skills → replan</small></a>
  <a href="pipelines/08-dexterous-retargeting/"><span class="dof-status dof-status--smoke">SMOKE</span><strong>Dexterous retargeting</strong><small>Landmarks → geometry → optimization → time</small></a>
  <a href="pipelines/07-sim-to-real/"><span class="dof-status dof-status--documented">DOC</span><strong>Sim-to-Real</strong><small>Replay → HIL → shadow → guarded rollout</small></a>
  <a href="pipelines/09-perception-state-estimation/"><span class="dof-status dof-status--smoke">SMOKE</span><strong>Perception & state</strong><small>Calibration → sync → fusion → uncertainty</small></a>
  <a href="pipelines/10-navigation-locomotion/"><span class="dof-status dof-status--smoke">SMOKE</span><strong>Navigation & locomotion</strong><small>State → planning → control → recovery</small></a>
</div>

## Evidence before claims

<div class="dof-proof">
  <strong>7 runnable smokes · 2 interface paths · 1 hardware-dependent contract.</strong>
  <p>Execution is not performance, and synthetic simulation is not hardware validation. DoF separates import, smoke execution, deterministic tests, benchmark evidence, and hardware validation; a lower level never implies a higher one.</p>
  <p><a href="VALIDATION/">Read the validation policy →</a></p>
</div>

## Verify the repository

```bash
python scripts/run_pipeline.py --validate
python scripts/run_pipeline.py --list
python scripts/audit_repository.py
python -m pytest tests/ -q
```

The root [English README](https://github.com/Dld0621/Embodied-AI-Zero-to-Hero/blob/master/README.md) and [中文 README](https://github.com/Dld0621/Embodied-AI-Zero-to-Hero/blob/master/README_CN.md) are the concise product landing pages. This site is the deeper learning and engineering layer.
