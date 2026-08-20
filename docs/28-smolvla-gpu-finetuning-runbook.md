# SmolVLA GPU Fine-tuning Runbook

> Step-by-step guide to fine-tune the full 450M SmolVLA model on PushCube with a GPU.

## Prerequisites

| Requirement | Minimum | Recommended |
|:------------|:--------|:------------|
| GPU | 8 GB VRAM | 16 GB+ VRAM |
| CUDA | 11.8+ | 12.1+ |
| Python | 3.10+ | 3.10/3.11 |
| Disk | 10 GB | 20 GB |

## Step 1: Install LeRobot

**Option A — From PyPI (recommended for stability):**

```bash
pip install 'lerobot[smolvla]==0.4.1'
```

This installs LeRobot 0.4.1 with the SmolVLA extra, which includes the
`lerobot.common.policies.smolvla.modeling_smolvla` module.

**Option B — From source (latest, for development):**

```bash
git clone https://github.com/huggingface/lerobot.git
cd lerobot
pip install -e ".[smolvla]"
```

Verify:
```bash
lerobot-train --help
```

## Step 2: Convert Dataset to LeRobot Format

The canonical PushCube data (50 episodes, 1788 frames) must be converted to LeRobot Parquet format:

The generated image dataset is intentionally ignored by Git. If `datasets/pushcube_canonical/` is absent, create a fresh deterministic dataset first:

```bash
cd examples/robot_foundation_models/smolvla
python collect_pushcube_dataset.py \
    --n_episodes 50 \
    --seed_start 0 \
    --output datasets/pushcube_canonical
```

Record the generated dataset manifest and checksum in your experiment report; a newly collected dataset is not byte-identical to the historical private dataset unless its full provenance matches.

```bash
cd examples/robot_foundation_models/smolvla

python -c "
import sys
sys.path.insert(0, '../common')
from canonical_dataset import load_episodes_from_dir
from to_lerobot import convert_to_lerobot

episodes = load_episodes_from_dir('datasets/pushcube_canonical/')
convert_to_lerobot(
    episodes,
    output_dir='datasets/pushcube_lerobot/',
    dataset_name='pushcube_dual_cube',
)
print(f'Converted {len(episodes)} episodes to LeRobot format')
"
```

Verify the dataset loads:
```python
from lerobot.datasets.lerobot_dataset import LeRobotDataset
dataset = LeRobotDataset(
    repo_id="local/pushcube_dual_cube",
    root="datasets/pushcube_lerobot/",
)
print(f"Dataset size: {len(dataset)} frames")
```

## Step 3: Fine-tune SmolVLA

```bash
python finetune.py \
    --dataset_dir datasets/pushcube_lerobot/ \
    --output_dir models/smolvla_pushcube/ \
    --config finetune_config.yaml
```

This calls `lerobot-train` with:
- Pretrained: `lerobot/smolvla_base` (450M params)
- Dataset: 50 episodes, 1788 frames, action_dim=2
- Steps: 20,000
- Batch size: 64
- Device: cuda
- Mixed precision: bf16

Expected training time: ~30-60 minutes on a single A100.

## Step 4: Evaluate with Real SmolVLA Checkpoint

```bash
# Closed-loop evaluation (20 episodes)
python evaluate.py \
    --mode closed_loop \
    --checkpoint models/smolvla_pushcube/checkpoints/last \
    --device cuda \
    --n_episodes 20 \
    --output ../../results/benchmarks/smolvla_real_closed_loop.json

# Offline evaluation (compare to expert)
python evaluate.py \
    --mode offline \
    --checkpoint models/smolvla_pushcube/checkpoints/last \
    --device cuda \
    --data ../../results/benchmarks/pushcube_expert.json
```

## Step 5: Update Results

After evaluation, update:
1. `results/benchmarks/smolvla_real_closed_loop.json` — closed-loop metrics
2. `README.md` model status table — change SmolVLA from `🟡` to `✅`
3. `README.md` benchmark table — add real SmolVLA success rate
4. `CHANGELOG.md` — document the real checkpoint

Do not mark `eval_results.json`, `training_history.json`, or checkpoint metadata as available unless those exact files are committed or linked from a durable artifact store. Aggregate summaries alone are reported evidence, not raw reproducibility.

## Lightweight VLA (CPU Fallback)

For environments without GPU or LeRobot, a lightweight VLA (195K params) can be trained on CPU:

```bash
python train_lightweight_vla.py --epochs 100 --batch_size 64
```

This produces a real trained checkpoint at `models/lightweight_vla/lightweight_vla_pushcube.pt` that can be used for closed-loop evaluation:

```bash
python evaluate.py \
    --mode closed_loop \
    --checkpoint models/lightweight_vla/lightweight_vla_pushcube.pt \
    --n_episodes 20
```

**Note:** The lightweight VLA is a 195K-parameter CNN+language policy, NOT the 450M SmolVLA. It demonstrates the full train→evaluate→report pipeline on CPU, but does not achieve task success (0% closed-loop success rate with 100 epochs on 50 episodes).

## Current Status

| Model | Checkpoint | Training | Closed-Loop | Status |
|:------|:-----------|:---------|:------------|:-------|
| Lightweight VLA (195K) | `lightweight_vla_pushcube.pt` | 100 epochs, CPU | 0% success, 65% selection | Checkpoint committed; result links to a committed evaluation artifact |
| SmolVLA (450M) | 155 reported state-dict entries (450,046,176 total params) | RTX 3060, bf16, 500 steps, 100M trainable | 0% success, 50% selection | **Reported aggregate**; config/history/summary committed, per-episode eval and checkpoint structure absent |
| SmolVLA (450M, 10K) | 155 reported state-dict entries (450,046,176 total params) | RTX 3060, bf16, 10K steps (resume from 500), 100M trainable | 0% success, 50% selection | **Reported aggregate**; config/summary committed, per-step history and per-episode eval absent |

### Reported GPU Run Summary (2026-08-04)

> **Evidence boundary:** the repository retains aggregate summaries and training configurations for these runs, but not the SmolVLA weights, per-episode evaluation JSON, or complete 10K-step history. The numbers below document a reported local run and cannot yet be independently re-aggregated from repository contents.

**500-step baseline run:**
- **Hardware:** NVIDIA RTX 3060 Laptop (6.4 GB VRAM), CUDA 12.8, PyTorch 2.11.0+cu128
- **Model:** `lerobot/smolvla_base` (450M params, 100M trainable after LoRA-style unfreeze)
- **Dataset:** PushCube dual-cube, 50 episodes / 1788 frames, action_dim=2
- **Training:** 500 steps, batch_size=2, bf16 mixed precision, AdamW
- **Checkpoint:** reported as per-tensor `.npy` files + `manifest.json` (155 state-dict entries, 450,046,176 total model parameters); weights are not committed
- **Loss curve:** 0.47 → 0.10 (best 0.028)
- **Closed-loop eval:** 20 episodes × 3 language modes (correct / swapped / none), 0% success, 50% selection accuracy
- **Interpretation:** this run did not achieve task-level success. The committed evidence supports the configuration and aggregate summary, not independent verification of the complete local training/reload/evaluation chain.
- **Evaluation results:** local file was reported during the run but is not committed

**10K-step scale-up run:**
- **Hardware:** Same RTX 3060 Laptop (6.4 GB VRAM)
- **Training:** Resumed from 500-step checkpoint, trained to 10K steps (9500 additional), 65.1 min total
- **Script:** `smolvla_train_10k_v2.py` was reported for the local run but is not committed
- **Checkpoint:** Atomic save at steps 5000 and 10000 (temp dir → verify → rename), 399.5 MB each, 155 saved tensors (450,046,176 total params)
- **Loss curve:** 0.10 → 0.031 (avg 0.053, best 0.004)
- **LR schedule:** Cosine decay from 1e-4 to 2.5e-6 over 9500 steps
- **Closed-loop eval:** 20 episodes × 3 language modes (correct / swapped / none), 0% success, 50% selection accuracy (all modes)
- **Observed gap:** reported training loss decreased from 0.10 to 0.03 while reported closed-loop success remained 0%. This observation alone cannot distinguish memorization from data coverage, preprocessing, action decoding, optimization, or evaluator problems.
- **Required follow-up:** commit held-out open-loop metrics and per-episode closed-loop artifacts; then test variation splits, dataset scaling, DAgger/RL post-training, and architecture changes one factor at a time.
- **Evaluation results / full training history:** reported local files are not committed

### Next Steps

1. Scale dataset to 500+ episodes (current: 50) for better generalization
2. Try DAgger or RL fine-tuning (PPO/REINFORCE) on top of BC checkpoint
3. Add action chunking (predict multi-step action sequences)
4. Compare against Diffusion Policy and ACT on the same benchmark
5. Explore data augmentation (random crop, color jitter) for vision robustness
