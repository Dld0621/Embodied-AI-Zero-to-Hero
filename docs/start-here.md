# Start Here: From the First Hour to Expert Practice

**English · [简体中文](start-here-cn.md)**

This is not a list of articles to read from top to bottom. Work in a loop: understand → build → break → measure → explain → review. Time estimates help scheduling; only artifacts and gates support progression.

## Your first 30 minutes

```bash
git clone https://github.com/Dld0621/Embodied-AI-Zero-to-Hero.git
cd Embodied-AI-Zero-to-Hero
python -m pip install numpy
python scripts/run_curriculum.py --validate
python scripts/run_curriculum.py --diagnose
python scripts/run_pipeline.py --run simulation-data
```

You should obtain a validated curriculum contract, an evidence self-diagnostic, and a synthetic Pipeline result. These outputs show that the entry path executes. They do not demonstrate mastery or real task performance.

## The first-hour challenge

1. Locate the Pipeline inputs, stages, metrics, and output files.
2. Change one parameter, predict the effect, and rerun it.
3. Deliberately create one failure, such as a wrong shape, illegal range, or missing field.
4. Save the command, environment, seed, result, and failure explanation.
5. Write your first [experiment card](../learner/templates/experiment-card.md).

If you can copy a command but cannot explain its data flow, predict a change, or isolate a failure, start at M00.

## Place yourself with evidence

Run:

```bash
python scripts/run_curriculum.py --diagnose
```

Ask one question for every module: **Can I show the required artifact and let another person review the gate?**

- Taking a class, reading a paper, or running code is not passing evidence.
- Without evidence, use `not_started` or `in_progress`; background does not grant an automatic pass.
- Experienced learners may skip explanations, but not assessment.

## Choose the first emphasis for your background

| Background | Likely strength | Common gap | Suggested entry |
|---|---|---|---|
| Complete beginner | Willingness to run and inspect Python | Environment, units, frames, experiment identity | M00 |
| Mechanical/control | Kinematics, dynamics, and control intuition | Python, data contracts, learning evaluation | Diagnose M00, then focus on M05–M07 |
| CS/ML | Programming, training, and model implementation | Frames, contact, closed-loop control, safety | M02–M04 |
| Robotics practitioner | System integration, simulation, or hardware | Matched baselines, leakage, ablations, uncertainty | M05 and M11 |

## Generate a route

```bash
python scripts/run_curriculum.py --list-goals
python scripts/run_curriculum.py \
  --plan full-stack-expert \
  --hours-per-week 8
```

Modules with existing reviewed evidence can be omitted for planning:

```bash
python scripts/run_curriculum.py \
  --plan full-stack-expert \
  --completed M00,M01 \
  --hours-per-week 8
```

`--completed` does not validate evidence. Initialize a formal record instead:

```bash
python scripts/run_curriculum.py \
  --init-progress learner/progress.json \
  --goal full-stack-expert \
  --learner your-name
```

A `passed` module requires existing evidence, a reviewer, and a review date. Audit the record with:

```bash
python scripts/run_curriculum.py --report-progress learner/progress.json
```

## The module loop

1. **Diagnose:** resolve prerequisites and gaps.
2. **Learn:** complete the derivation or code yourself.
3. **Build:** produce the minimum module artifact.
4. **Break:** inject at least one failure, perturbation, or counterfactual.
5. **Measure:** retain raw metrics, configuration, logs, and failures.
6. **Explain:** distinguish data, model, interface, control, and task-definition failures.
7. **Review:** use the [assessment standard](assessment.md) to pass, repeat, or stop.

## What beginner-to-expert means here

| Level | Capability that must be demonstrated |
|---|---|
| L0 | Another person reproduces your experiment; you repair a deliberate fault |
| L1 | You derive and numerically verify the mathematics and catch frame/numerical traps |
| L2 | You build a bounded sensing–state–control–simulation loop and isolate faults |
| L3 | You build data, policy, RL, world-model, and planning baselines and separate offline from closed-loop quality |
| L4 | You own one task family end to end, including perturbations, recovery, and stage failures |
| L5 | You independently reproduce a baseline, test a falsifiable idea, run controlled comparisons, and survive independent review |

Expert practice is not finishing every page or lowering a training loss. See the [three capstones](capstone.md) for graduation evidence.

## When stuck

- Environment failure: use [setup troubleshooting](setup/troubleshooting.md) and preserve the first error.
- Unknown prerequisite: run `python scripts/run_knowledge_map.py --path-to <node>`.
- The system runs but the task fails: measure stage-resolved error before scaling or retraining.
- Hardware trial: first pass Sim-to-Real and safety gates; simulation never grants hardware authorization.
