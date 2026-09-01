# Assessment Standard: How Mastery Is Demonstrated

**English · [简体中文](assessment-cn.md)**

This standard separates exposure from independent capability. Passing a module means the learner supplied reviewable evidence under a declared environment and task boundary. It is not universal expert certification, a state-of-the-art result, or hardware authorization.

## Module score (100 points)

| Dimension | Points | Passing evidence |
|---|---:|---|
| Conceptual and mathematical correctness | 20 | Explains assumptions, equations, units, frames, and scope |
| Independent implementation and modification | 20 | Implements key work, predicts changes, and does more than replay output |
| Executable verification | 20 | Artifact runs or can be checked; nominal and failure cases are retained |
| Reproducibility | 15 | Command, environment, commit, configuration, seed, and data identity are complete |
| Failure diagnosis | 10 | Localizes failure to a stage and rejects at least one wrong hypothesis with evidence |
| Evidence and claim boundary | 10 | Raw and negative results remain; conclusions do not exceed evidence |
| Safety and operating boundary | 5 | Action limits, stopping, authorization, and risk match the task |

The module pass mark is **80/100**, with no critical failure below. A weak dimension requires revision; unrelated points cannot cancel it.

## Critical failures: any one means no pass

- No raw evidence; only a screenshot-like conclusion or manually entered success number.
- A reviewer cannot reproduce the key result from the record and the discrepancy is unexplained.
- An import, smoke test, fixed scene, or offline loss is described as generalization, hardware, or deployment success.
- Train/validation/test leakage or forbidden privileged state is used during evaluation.
- Data, training, evaluation, or execution budgets differ while the result claims algorithm superiority.
- Negative results are removed, or the protocol/success definition changes after seeing the test result.
- A hardware task lacks explicit authorization, bounded commands, stop paths, and site-specific risk review.

## Minimum evidence package

```text
evidence/Mxx/
├── experiment-card.md
├── environment.txt
├── config.json
├── commands.txt
├── metrics.json
├── failures.md
└── artifacts/          # curves, traces, videos, checkpoints, or data reports
```

Exact names may differ, but equivalent information must remain traceable. The [experiment card](../learner/templates/experiment-card.md) and [failure report](../learner/templates/failure-report.md) reduce omissions.

## Review workflow

1. The learner self-scores every dimension and links raw evidence.
2. A reviewer reruns one randomly selected command and inspects one failure.
3. The reviewer checks data identity, evaluation protocol, and claim boundaries.
4. The decision is `passed`, `revision_required`, or `blocked`.
5. A pass is recorded with evidence paths, reviewer, and date in progress JSON.

Reviewers should not inspect only the best episode. They should examine aggregate metrics, stage-resolved error, at least one representative failure, and uncertainty.

## Six-level promotion rule

| Level | Required before progression |
|---|---|
| L0 | M00 |
| L1 | M01 and M02, with derivations matching numerical checks |
| L2 | M03, M04, and C0; the loop includes safe stop and fault diagnosis |
| L3 | Goal-required modules from M05–M08; data, baselines, and closed-loop evaluation hold |
| L4 | Goal-required modules from M09–M10; one task family has end-to-end ownership |
| L5 | M11 and the target capstone; bounded claims survive independent review |

Time spent cannot replace a gate. When evidence does not pass, retain `in_progress` or `blocked` and name the next experiment that could change the decision.

## Required counterfactuals by specialization

- **VLA:** correct, shuffled, and missing language; visual perturbation; unseen initial state; action/chunk interface checks.
- **World model/WAM:** action swap, no action, multiple horizons; with/without lookahead; verify that planned actions actually change.
- **Dexterity:** score geometry, collision, contact, lift, retention, slip, and recovery separately.
- **Navigation/locomotion:** score goal completion, collision/fall, perturbation, recovery, replanning, and stopping separately.
- **Sim-to-Real:** separate system identification, latency, sensor shift, HIL, shadow mode, and authorized hardware gates.

## A progress record is not a certificate

```bash
python scripts/run_curriculum.py --report-progress learner/progress.json
```

This check validates record structure, evidence paths, and prerequisites. It does not determine whether a paper is correct, a model is good, or hardware motion is authorized. Human reviewers still inspect the raw artifacts.
