# SmolVLA 500-Step Training Summary

## Training Configuration

| Parameter | Value |
|:----------|:------|
| Model | SmolVLA 450M (`lerobot/smolvla_base`) |
| Total params | 450,046,176 |
| Trainable params | 99,880,992 (LoRA-style unfreeze) |
| Dataset | PushCube dual-cube, 50 episodes / 1788 frames |
| Action dim | 2 (`ee_delta_2d`) |
| Steps | 500 |
| Batch size | 2 |
| Precision | bf16 |
| Optimizer | AdamW |
| Hardware | NVIDIA RTX 3060 Laptop (6.4 GB VRAM), CUDA 12.8 |

## Training Results

| Metric | Value |
|:-------|:------|
| Initial loss | 0.47 |
| Final loss | 0.103 |
| Average loss | 0.069 |
| Best loss | 0.028 |
| Checkpoint size | 399.5 MB (155 tensors) |

## Closed-Loop Evaluation

| Mode | Success Rate | Selection Accuracy |
|:-----|:---:|:---:|
| Correct language | 0% | 50% |
| Swapped language | 0% | 50% |
| No language (vision-only) | 0% | 50% |

- **Episodes:** 20 per language mode (60 total)
- **Analysis:** 500 steps is insufficient for task-level success. The original experiment report states that model loading, training, saving, reloading, and closed-loop execution completed; the current repository retains aggregate evidence but not the checkpoint or per-episode evaluation file.

## Files

| File | Committed here | Description |
|:-----|:---:|:------------|
| `training_config.json` | Yes | Checkpoint metadata (step, loss, param count) |
| `training_history.json` | Yes | Per-step loss history |
| `eval_results.json` | No | Historical per-episode evaluation file was not committed |
| `checkpoint_info.json` | No | Historical checkpoint structure file was not committed |

## Reproduce

The historical `smolvla_train_v1.py` and its checkpoint are not part of this repository, so this snapshot is not a self-contained bitwise reproduction. For a new tracked run, use the maintained [GPU runbook](../../../docs/28-smolvla-gpu-finetuning-runbook.md), set `finetune.py --steps 500 --batch_size 2`, and commit the resulting configuration plus raw evaluation JSON before updating the benchmark.
