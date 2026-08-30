# VLA and WAM Specialization

**English · [简体中文](README_CN.md)**

This specialization is for readers who want to progress from a correct first implementation to research-grade experiment design in Vision-Language-Action models (VLA) and World Action Models (WAM). It is organized by competency gates rather than a fixed calendar.

> **Terminology boundary:** a VLA maps visual, language, and robot context to actions. A conventional world model predicts consequences under candidate actions. In the narrower recent robotics usage, a WAM jointly or aligningly models future world states—often video—and actions. An action-conditioned world model with a separate planner is an important baseline, but it is not automatically a WAM.

## Choose the correct starting point

| Your actual problem | Start here | Do not start with |
|---|---|---|
| One fixed task, limited demonstrations | Chunked behavior cloning or ACT | A large VLA merely because language can be added |
| Several language-conditioned tasks | Continuous-chunk VLA baseline | Claims of language use without ablation |
| Multiple valid action modes | Diffusion or flow-matching action model | Plain MSE without checking mode averaging |
| Counterfactual planning under candidate actions | Latent world model + MPC | WAM terminology before model/planner baselines |
| Joint future-video and action research | Small joint model, then WAM scaling | A 10B+ video backbone as the first implementation |
| Tight contact or sub-centimeter precision | Task policy + state/force feedback and safety | Visual plausibility as proof of control precision |

Use the explainable selector for a starting shortlist:

```bash
python scripts/select_vla_wam_algorithm.py \
  --goal language-generalization \
  --compute single-gpu \
  --data multi-task \
  --latency hard
```

Its source of truth is [`learning_tracks/vla_wam_algorithms.json`](../../learning_tracks/vla_wam_algorithms.json). The output is an experiment-design recommendation, not a leaderboard or deployment authorization.

## Prerequisite gates

| Gate | You must be able to do | Evidence to retain |
|---|---|---|
| G0 · Experiment contract | Pin code, environment, seed, dataset identity, and output path | Reproduction receipt |
| G1 · Robot semantics | Label frames, units, action meaning, rate, bounds, and reset/terminal rules | Interface table and round-trip checks |
| G2 · Data integrity | Synchronize observations/actions, split by episode, audit coverage and leakage | Dataset card and split manifest |
| G3 · Policy baseline | Overfit a tiny set, then evaluate a simple closed-loop policy | Curves, checkpoint, task metrics, failures |
| G4 · Predictive baseline | Measure one-step and horizon-conditioned rollout error | Rollout audit and planner comparison |
| G5 · Generalization | Freeze shifts, matched budgets, baselines, and uncertainty reporting | Reproducible comparison and decision |

Resolve missing prerequisites through the [Knowledge System](../knowledge-system/README.md). The minimum core is Python/tensors, probability and optimization, frames and kinematics, control, sensors, Transformers, episode schemas, and evaluation.

## Two primary tracks

### [VLA Zero to One](vla-zero-to-one.md)

Learn the observation contract, multimodal encoding, fusion, action representations, behavior cloning, discrete action tokens, continuous chunking, diffusion, flow matching, fine-tuning, receding-horizon execution, language ablations, and algorithm selection.

Exit artifact: one closed-loop language-conditioned policy with a matched non-language baseline, correct/swapped/absent-language ablations, latency measurements, task metrics, and a failure taxonomy.

### [WAM Zero to One](wam-zero-to-one.md)

Learn world-model baselines, future representations, inverse and forward dynamics, joint video-action modeling, autoregressive and flow-based objectives, action-video alignment, closed-loop inference, WAM selection, and the distinction between plausible video and controllable behavior.

Exit artifact: a comparison between a policy baseline, a world-model + planner baseline, and a small joint world-action model under the same data and evaluation contract.

## Recommended build order

1. Define one task, observation, action, timing, success, and safety contract.
2. Train a non-language state or vision policy and pass a tiny-set overfit test.
3. Add language only when tasks or goals vary; prove its causal use by ablation.
4. Add chunking; compare direct regression against a generative action head if the data is multimodal.
5. Train a one-step dynamics model; report rollout error by horizon.
6. Add MPC or another planner and measure downstream task improvement.
7. Build a small joint future/action model and compare it against the modular baseline.
8. Only then consider pretrained VLA or video-model backbones and larger heterogeneous datasets.
9. Freeze a generalization suite and report negative results, latency, and confidence.
10. Treat hardware rollout as a separately authorized stage with bounded commands and stop paths.

## What “familiar with the field” means

You are ready to enter research when you can:

- derive and implement the main action objectives rather than only call a checkpoint;
- explain why a task needs—or does not need—language, action multimodality, or future-video generation;
- identify whether failure came from data, representation, optimization, dynamics, planning, control, or evaluation;
- select a matched baseline and budget instead of comparing incomparable published headline numbers;
- separate offline loss, prediction quality, closed-loop task success, generalization, and hardware evidence;
- state what evidence would falsify your preferred VLA or WAM hypothesis.

## Primary reading order

| Purpose | Primary source |
|---|---|
| Scaled robot Transformer precursor | [RT-1](https://arxiv.org/abs/2212.06817) |
| VLA formulation | [RT-2](https://arxiv.org/abs/2307.15818) |
| Open discrete-action VLA | [OpenVLA](https://arxiv.org/abs/2406.09246) |
| Continuous parallel VLA fine-tuning | [OpenVLA-OFT](https://arxiv.org/abs/2502.19645) |
| Flow-matching generalist policy | [π0](https://arxiv.org/abs/2410.24164) |
| Compact VLA implementation | [SmolVLA](https://arxiv.org/abs/2506.01844) and [official LeRobot guide](https://huggingface.co/docs/lerobot/smolvla) |
| Latent model-based control baseline | [TD-MPC2](https://arxiv.org/abs/2310.16828) |
| Action-conditioned predictive planning | [V-JEPA 2](https://arxiv.org/abs/2506.09985) |
| Autoregressive image-action model | [WorldVLA](https://arxiv.org/abs/2506.21539) |
| Video-action WAM | [DreamZero](https://arxiv.org/abs/2602.15922) |

## Evidence boundary

Publication status and reported results belong to the linked sources. This repository uses them to explain algorithms; it does not claim to reproduce their large-scale or hardware results.
