# SmolVLA 10K-Step Training Summary

## Training Configuration

| Parameter | Value |
|:----------|:------|
| Model | SmolVLA 450M (`lerobot/smolvla_base`) |
| Total params | 450,046,176 |
| Trainable params | 99,880,992 (LoRA-style unfreeze) |
| Dataset | PushCube dual-cube, 50 episodes / 1788 frames |
| Action dim | 2 (`ee_delta_2d`) |
| Steps | 10,000 (resumed from 500-step checkpoint, 9500 additional) |
| Batch size | 2 |
| Precision | bf16 |
| Optimizer | AdamW |
| LR schedule | Cosine decay 1e-4 to 2.5e-6 over 9500 steps |
| Hardware | NVIDIA RTX 3060 Laptop (6.4 GB VRAM), CUDA 12.8 |
| Training time | 65.1 minutes |
| Script | `smolvla_train_10k_v2.py` (atomic checkpoint, error recovery, signal handling) |

## Training Results

| Metric | Value |
|:-------|:------|
| Starting loss (from 500-step) | 0.10 |
| Final loss | 0.031 |
| Average loss | 0.030 |
| Best loss | 0.004 |
| Checkpoint size | 399.5 MB (155 tensors) |
| Checkpoints saved | Step 5000, Step 10000 (atomic save) |

## Closed-Loop Evaluation

| Mode | Success Rate | Selection Accuracy |
|:-----|:---:|:---:|
| Correct language | 0% | 50% |
| Swapped language | 0% | 50% |
| No language (vision-only) | 0% | 50% |

- **Episodes:** 20 per language mode (60 total)
- **Analysis:** Training loss decreased 3x (0.10 to 0.03) but closed-loop success remains 0%. This is classic **BC overfitting** — the model memorizes training trajectories but cannot generalize to new initial conditions. 50 episodes (1788 frames) is insufficient for a 450M parameter VLA to generalize on a contact-rich manipulation task.

## Key Findings

1. **Reported execution:** the historical run completed train → checkpoint → resume → evaluate; raw checkpoint and per-episode files are not committed, so the current repository cannot independently re-aggregate it
2. **BC overfitting confirmed:** loss decreases but task success does not improve
3. **Data scale is the bottleneck:** 50 episodes is far too small for a 450M model
4. **Next steps:** 100+ episodes, 100K+ steps, DAgger or RL fine-tuning

## Files

| File | Committed here | Description |
|:-----|:---:|:------------|
| `training_config.json` | Yes | Checkpoint metadata (step, loss, param count, resume info) |
| `training_history.json` | No | Historical 9500-step history was not committed |
| `eval_results.json` | No | Historical per-episode evaluation file was not committed |
| `checkpoint_info.json` | No | Historical checkpoint structure file was not committed |

## Reproduce

The historical `smolvla_train_10k_v2.py`, resume checkpoint, and raw evaluation output are not part of this repository, so this snapshot is not a self-contained bitwise reproduction. For a new tracked run, use the maintained [GPU runbook](../../../docs/28-smolvla-gpu-finetuning-runbook.md), preserve the resume source and raw per-episode evaluation JSON, and only then update the canonical benchmark.
