---
hide:
  - navigation
  - toc
---

<section class="dof-landing">
  <div class="dof-landing__copy">
    <div class="dof-kicker">Embodied AI · Open research stack</div>
    <h1>From first principles<br><span>to physical action.</span></h1>
    <p class="dof-lead">
      Learn the foundations, execute complete pipelines, and judge every result by its evidence. One bilingual system for understanding and building embodied intelligence.
    </p>
    <div class="dof-actions">
      <a class="dof-button dof-button--primary" href="foundations/README_EN/">Start learning</a>
      <a class="dof-button" href="learning-paths/">Choose a route</a>
      <a class="dof-button" href="pipelines/">Explore pipelines</a>
      <a class="dof-button" href="index_cn/">中文</a>
    </div>
  </div>
  <aside class="dof-signal" aria-label="Repository evidence status">
    <div class="dof-signal__top"><span>Evidence status</span><span>LIVE</span></div>
    <strong>7 / 10</strong>
    <p>pipelines include a runnable smoke path</p>
    <div class="dof-signal__rail" aria-hidden="true">
      <span class="dof-signal__smoke"></span><span class="dof-signal__interface"></span><span class="dof-signal__documented"></span>
    </div>
    <dl>
      <div><dt>Smoke-tested</dt><dd>7</dd></div>
      <div><dt>Interface-tested</dt><dd>2</dd></div>
      <div><dt>Hardware-dependent</dt><dd>1</dd></div>
    </dl>
    <a href="VALIDATION/">Read the evidence policy <span aria-hidden="true">↗</span></a>
  </aside>
</section>

<div class="dof-section-label">Closed-loop system</div>

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
  <div class="dof-metric"><strong>7</strong><span>Research routes</span></div>
  <div class="dof-metric"><strong>EN · 中文</strong><span>Bilingual entry</span></div>
</div>

## Choose your route

<div class="dof-grid">
  <a class="dof-card" href="foundations/README_EN/">
    <span class="dof-card__index">01 · LEARN</span>
    <h3>Build the mental model</h3>
    <p>Math, learning, frames, kinematics, sensing, control, systems, safety and evaluation.</p>
  </a>
  <a class="dof-card" href="learning-paths/">
    <span class="dof-card__index">02 · ORIENT</span>
    <h3>Choose a research direction</h3>
    <p>Start from a question, then follow its prerequisites, Pipelines, deliverable, metrics and promotion gate.</p>
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

<div class="dof-section-head">
  <div><span>Goal-oriented curriculum</span><h2>Seven research routes</h2></div>
  <a class="dof-section-link" href="learning-paths/">Open full route map →</a>
</div>

<div class="dof-route-grid">
  <a class="dof-route" href="learning-paths/#foundation-models-vla"><span>01</span><strong>Foundation Models & VLA</strong><small>Policy · adapter · ablation</small></a>
  <a class="dof-route" href="learning-paths/#manipulation-imitation"><span>02</span><strong>Manipulation & Imitation</strong><small>Baseline · failures · closed loop</small></a>
  <a class="dof-route" href="learning-paths/#dexterity-teleoperation"><span>03</span><strong>Dexterity & Teleoperation</strong><small>Retargeting · evidence layers</small></a>
  <a class="dof-route" href="learning-paths/#navigation-embodied-agents"><span>04</span><strong>Navigation & Agents</strong><small>State · planning · recovery</small></a>
  <a class="dof-route" href="learning-paths/#humanoids-locomotion"><span>05</span><strong>Humanoids & Locomotion</strong><small>Motion · safety · transfer</small></a>
  <a class="dof-route" href="learning-paths/#perception-world-models"><span>06</span><strong>Perception & World Models</strong><small>Uncertainty · predictive rollout</small></a>
  <a class="dof-route" href="learning-paths/#simulation-data-evaluation"><span>07</span><strong>Simulation, Data & Evaluation</strong><small>Datasheet · benchmark · gate</small></a>
</div>

<div class="dof-section-head">
  <div><span>Pipeline status</span><h2>System coverage today</h2></div>
  <div class="dof-legend" aria-label="Pipeline evidence legend">
    <span class="dof-legend__smoke">Smoke 7</span>
    <span class="dof-legend__interface">Interface 2</span>
    <span class="dof-legend__documented">Documented 1</span>
  </div>
</div>

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
python scripts/run_learning_path.py --validate
python scripts/run_pipeline.py --list
python scripts/audit_repository.py
python -m pytest tests/ -q
```

The root [English README](https://github.com/Dld0621/Embodied-AI-Zero-to-Hero/blob/master/README.md) and [中文 README](https://github.com/Dld0621/Embodied-AI-Zero-to-Hero/blob/master/README_CN.md) are the concise product landing pages. This site is the deeper learning and engineering layer.
