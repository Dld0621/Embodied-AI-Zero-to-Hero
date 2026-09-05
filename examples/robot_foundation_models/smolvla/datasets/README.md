# PushCube Dataset for SmolVLA

This directory documents how to generate local PushCube demonstrations. The layout below is expected output, not a claim that all data has been committed or validated for real SmolVLA training. Run every command from the **repository root** unless a different checkout is explicitly named.

## Dataset Structure

```
datasets/
├── pushcube_canonical/     # CanonicalEpisode format (.pkl files)
│   ├── episode_0000.pkl
│   ├── ...
│   └── dataset_info.json
└── pushcube_mock_parquet/  # Simplified inspection output, NOT verified LeRobotDataset
    ├── meta/
    │   └── info.json
    └── data/
```

## Generating the Dataset

### 1. Collect expert demonstrations

```bash
python examples/robot_foundation_models/smolvla/collect_pushcube_dataset.py --n_episodes 50 --output examples/robot_foundation_models/smolvla/datasets/pushcube_canonical/
```

**Expected output:**
- Requested count: 50 episodes; measure actual successes and save the collection log instead of assuming ~100%
- 14-D state, 2-D action `[dx, dy]`
- 128x128 RGB images + language instruction per timestep

### 2. Create a mock serialization fixture

This explicit mock route needs pandas/pyarrow. It stores raw image bytes and a simplified schema, not a verified dataset for `lerobot-train`. It does not load model weights.

```bash
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
    output_dir='examples/robot_foundation_models/smolvla/datasets/pushcube_mock_parquet/',
    dataset_name='pushcube_dual_cube',
    mock=True,
)
"
```

For the real route, install the repository's selected LeRobot stack and deliberately use `mock=False` with a separate output directory. The current converter can fall back to mock when LeRobot is unavailable; inspect logs and verify loading through the pinned `LeRobotDataset` API before calling the result real-format compatible.

### 3. Inspect mock metadata (not a compatibility test)

```bash
ls examples/robot_foundation_models/smolvla/datasets/pushcube_mock_parquet/meta/info.json
```

For the mock writer, `meta/info.json` is expected to contain:
- `action_dim: 2` (2-D `[dx, dy]`)
- `state_dim: 14`
- `n_episodes: 50`

The current mock metadata does **not** include `action_type`. The canonical collector labels this local task `ee_delta_2d`; preserve that contract separately and verify units, timing, image shapes and episode counts. Checking only the existence of `meta/info.json` does not validate the dataset schema or controller semantics.

## Fine-tuning

### Mock training (CPU + PyTorch, no pretrained downloads)

```bash
python examples/robot_foundation_models/smolvla/finetune.py --mock --dataset_dir examples/robot_foundation_models/smolvla/datasets/pushcube_canonical/ --config examples/robot_foundation_models/smolvla/finetune_config.yaml --epochs 10
```

### Real SmolVLA fine-tuning (requires GPU + LeRobot)

First prepare and independently validate a **real-format** dataset in `examples/robot_foundation_models/smolvla/datasets/pushcube_lerobot/`. Do not rename the mock fixture to pretend it is compatible. Check that the path exists: the current launcher can interpret a nonexistent slash-containing path as a Hub repository identifier. These commands are unexecuted training instructions, not a training receipt.

```bash
# 1. In a SEPARATE, reviewed LeRobot source checkout only:
# pip install -e '.[smolvla]'
# Then return to this repository root. See docs/setup for the selected stack.

# 2. Verify lerobot-train CLI
lerobot-train --help

# 3. Run fine-tuning
python examples/robot_foundation_models/smolvla/finetune.py \
  --dataset_dir examples/robot_foundation_models/smolvla/datasets/pushcube_lerobot/ \
  --output_dir models/smolvla_pushcube/ \
  --config examples/robot_foundation_models/smolvla/finetune_config.yaml
```

The `finetune.py` script will:
1. Validate `lerobot-train` is available on PATH
2. Check GPU availability
3. For local paths, check that a metadata file exists (not a full schema validation)
4. Construct a `lerobot-train` command from the configured fields (version compatibility still needs validation)
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

Run `python examples/robot_foundation_models/smolvla/finetune.py --test` from the repository root to verify config flattening and command construction without GPU (4 checks). This does not load actual data, verify a training batch, or reproduce a trained policy.
