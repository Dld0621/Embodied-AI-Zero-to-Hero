<div class="dof-landing" markdown="1">
  <div class="dof-kicker">Embodied AI · Open research stack</div>

# From first principles to physical action.

  <p class="dof-lead">
    Learn the foundations, execute complete pipelines, and judge every result by its evidence. One bilingual system for understanding and building embodied intelligence.
  </p>

  <div class="dof-actions">
    <a class="dof-button dof-button--primary" href="foundations/README_EN/">Start learning</a>
    <a class="dof-button" href="pipelines/">Explore pipelines</a>
    <a class="dof-button" href="benchmark_report/">View evidence</a>
    <a class="dof-button" href="https://github.com/Dld0621/Embodied-AI-Zero-to-Hero">Open GitHub</a>
  </div>
</div>

<div class="dof-metrics">
  <div class="dof-metric"><strong>14</strong><span>Foundation lessons</span></div>
  <div class="dof-metric"><strong>8</strong><span>End-to-end pipelines</span></div>
  <div class="dof-metric"><strong>5</strong><span>Evidence levels</span></div>
  <div class="dof-metric"><strong>EN · 中文</strong><span>Bilingual entry</span></div>
</div>

## Choose your route

<div class="dof-grid">
  <a class="dof-card" href="foundations/README_EN/">
    <span class="dof-card__index">01 · LEARN</span>
    <h3>Build the mental model</h3>
    <p>Math, learning, frames, kinematics, sensing, control, systems, safety and evaluation.</p>
  </a>
  <a class="dof-card" href="pipelines/">
    <span class="dof-card__index">02 · BUILD</span>
    <h3>Run one complete system</h3>
    <p>Start from inputs, execute every stage, retain artifacts, and check an explicit promotion gate.</p>
  </a>
  <a class="dof-card" href="benchmark_report/">
    <span class="dof-card__index">03 · MEASURE</span>
    <h3>Compare with context</h3>
    <p>Read protocol, data budget, episode count, negative results and raw-artifact boundaries together.</p>
  </a>
  <a class="dof-card" href="19-sim-to-real-guide/">
    <span class="dof-card__index">04 · TRANSFER</span>
    <h3>Prepare guarded deployment</h3>
    <p>Move through simulation, replay, HIL and shadow mode before any hardware-dependent claim.</p>
  </a>
</div>

## Evidence before claims

<div class="dof-proof">
  <strong>Execution is not performance, and simulation is not hardware validation.</strong>
  <p>DoF separates import, smoke execution, deterministic tests, benchmark evidence, and hardware validation. A lower level never implies a higher one.</p>
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
