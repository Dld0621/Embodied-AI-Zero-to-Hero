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
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
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
| Lightweight VLA (195K) | `lightweight_vla_pushcube.pt` | 100 epochs, CPU | 0% success, 30% selection | Real checkpoint, limited capacity |
| SmolVLA (450M) | — | Requires GPU + LeRobot | — | Pipeline ready, awaiting GPU |
