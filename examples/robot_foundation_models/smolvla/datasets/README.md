# PushCube Dataset for SmolVLA

This directory contains PushCube demonstration data for fine-tuning SmolVLA.

## Dataset Structure

```
datasets/
├── pushcube_canonical/     # CanonicalEpisode format (.pkl files)
│   ├── episode_0000.pkl
│   ├── ...
│   └── dataset_info.json
└── pushcube_lerobot/       # LeRobot dataset format (for lerobot-train)
    ├── meta/
    │   └── info.json
    └── data/
```

## Generating the Dataset

### 1. Collect expert demonstrations

```bash
cd examples/robot_foundation_models/smolvla
python collect_pushcube_dataset.py --n_episodes 50 --output datasets/pushcube_canonical/
```

**Expected output:**
- 50 episodes, ~100% expert success rate
- 14-D state, 2-D action `[dx, dy]`
- 128x128 RGB images + language instruction per timestep

### 2. Convert to LeRobot format

```bash
cd ../../../..  # back to repo root
python -c "
import sys
sys.path.insert(0, 'examples/robot_foundation_models/common')
from canonical_dataset import load_episodes_from_dir
from to_lerobot import convert_to_lerobot

episodes = load_episodes_from_dir(
    'examples/robot_foundation_models/smolvla/datasets/pushcube_canonical/'
)
convert_to_lerobot(
    episodes,
    output_dir='examples/robot_foundation_models/smolvla/datasets/pushcube_lerobot/',
    dataset_name='pushcube_dual_cube',
)
"
```

> Note: Real conversion requires `pip install lerobot`. Without LeRobot, the
> converter writes Parquet files manually (requires `pip install pyarrow pandas`).

### 3. Verify LeRobot structure

```bash
ls datasets/pushcube_lerobot/meta/info.json
```

The `meta/info.json` should contain:
- `action_dim: 2` (2-D `[dx, dy]`)
- `state_dim: 14`
- `action_type: "ee_delta_2d"`
- `n_episodes: 50`

## Fine-tuning

### Mock training (CPU, no downloads)

```bash
python finetune.py --mock --dataset_dir datasets/pushcube_canonical/ --epochs 10
```

### Real SmolVLA fine-tuning (requires GPU + LeRobot)

```bash
# 1. Install LeRobot with SmolVLA support
pip install -e '.[smolvla]'  # from lerobot source

# 2. Verify lerobot-train CLI
lerobot-train --help

# 3. Run fine-tuning
python finetune.py \
  --dataset_dir datasets/pushcube_lerobot/ \
  --output_dir models/smolvla_pushcube/ \
  --config finetune_config.yaml
```

The `finetune.py` script will:
1. Validate `lerobot-train` is available on PATH
2. Check GPU availability
3. Verify the LeRobot dataset format (`meta/info.json`)
4. Construct the official `lerobot-train` command with correct hyperparameters
5. Execute training via subprocess

### Hyperparameters (from finetune_config.yaml)

| Parameter | Value | Source |
|:----------|:------|:-------|
| Pretrained | `lerobot/smolvla_base` | YAML `model.pretrained_name_or_path` |
| Batch size | 64 | YAML `training.batch_size` |
| Steps | 20,000 | YAML `training.steps` |
| Learning rate | 1e-4 | YAML `training.learning_rate` |
| Device | cuda | YAML `hardware.device` |
| Action dim | 2 (from dataset) | Not overridden on CLI |

Run `python finetune.py --test` to verify config flattening and command
construction without GPU (4 unit tests, ~1s).
