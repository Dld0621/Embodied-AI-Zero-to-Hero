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
- **Analysis:** 500 steps is insufficient for task-level success. The pipeline is fully verified (model loads, trains, saves, reloads, runs in closed loop). Scale to 10K+ steps for meaningful success rates.

## Files

| File | Description |
|:-----|:------------|
| `training_config.json` | Checkpoint metadata (step, loss, param count) |
| `training_history.json` | Per-step loss history |
| `eval_results.json` | Full evaluation results (20 episodes x 3 modes) |
| `checkpoint_info.json` | Checkpoint structure info |

## Reproduce

```bash
# Training (requires GPU with >=6GB VRAM)
python smolvla_train_v1.py --steps 500 --batch_size 2

# Evaluation
python smolvla_eval.py --checkpoint checkpoint_final/ --n_episodes 20
```
