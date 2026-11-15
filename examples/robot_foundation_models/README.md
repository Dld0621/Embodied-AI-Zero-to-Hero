# Robot Foundation Models

> Unified interface for integrating robot foundation models (SmolVLA, OpenVLA, Octo, GR00T) into a single experimental pipeline.

> **Safety boundary:** this is an experimental scaffold, not a hardware-ready stack. The OpenVLA adapter and SafetyFilter have unresolved executable-code defects (F002/F003); do not connect their outputs to a real robot. See the [content audit](../../docs/reviews/content-correctness-audit.md).

## Architecture

The diagram is a target architecture, not a claim that every arrow is implemented and validated. A common interface does not enforce identical action semantics or safety.

```text
User Natural Language Instruction
        ↓
Embodied Reasoner (task decomposition / spatial reasoning)
        ↓
Robot Foundation Model / VLA
Image + Language + Robot State → Action Chunk
        ↓
Robot Action Adapter
Generic Action → Robot-Specific Action Space
        ↓
Low-level Controller
End-effector pose, joint angles
        ↓
Safety Filter
Joint limits / collision / velocity limits / emergency stop
        ↓
MuJoCo or Real Robot
        ↑
World Model predictions, RL post-training
```

## Directory Structure

```text
examples/robot_foundation_models/
├── README.md                          # This file
├── common/                            # Shared interfaces (all models use these)
│   ├── observation_schema.py          # RobotObservation dataclass
│   ├── action_schema.py               # ActionChunk, ActionResult dataclasses
│   ├── model_interface.py             # RobotFoundationModel protocol
│   ├── embodiment_adapter.py          # EmbodimentAdapter ABC + GenericAction
│   ├── safety_filter.py               # SafetyFilter with joint/velocity/collision checks
│   ├── canonical_dataset.py           # CanonicalEpisode + EpisodeBuilder (unified data format)
│   ├── to_lerobot.py                  # Canonical → LeRobot dataset converter
│   └── to_rlds.py                     # Canonical → RLDS (TFRecord/NPZ) converter
├── smolvla/                           # SmolVLA (450M, first priority)
│   ├── inference.py                   # SmolVLAAdapter — wraps LeRobot SmolVLA
│   ├── evaluate.py                    # Offline + closed-loop evaluation
│   ├── closed_loop_eval.py            # PushCube closed-loop with language ablation
│   ├── collect_pushcube_dataset.py    # Expert trajectory collection → canonical format
│   ├── finetune.py                    # Fine-tuning script (real + mock mode)
│   └── finetune_config.yaml           # Fine-tuning hyperparameters
├── openvla/                           # OpenVLA-7B (second priority)
│   ├── inference.py                   # OpenVLAAdapter — wraps HuggingFace model
│   └── lora_config.yaml               # LoRA fine-tuning config
├── octo/                              # Octo (27M/93M, cross-embodiment tutorial)
│   └── inference.py                   # OctoAdapter + cross-embodiment tutorial
├── groot/                             # GR00T N1.6 (humanoid, planned)
│   ├── inference_pipeline_mock.py     # GR00TAdapter mock
│   └── observation_adapter.py         # Humanoid observation mapping reference
└── planners/                          # Embodied reasoning layer
    ├── rule_based_planner.py          # Deterministic pattern-matching planner
    └── vlm_task_planner.py            # VLM-based planner (GPT-4o / Gemini)
```

## Core Principle

All models implement the same `RobotFoundationModel` protocol:

```python
class RobotFoundationModel(Protocol):
    def reset(self) -> None: ...
    def predict_action(self, observation: RobotObservation) -> ActionChunk: ...
```

The protocol makes adapters easier to compare. Swapping a model still requires checking observation keys, normalization, action frame/units, chunk and control timing, robot mapping and controller compatibility. The outer loop may need changes too.

## Quick Start

Run each command from the **repository root** in the matching locked environment. Steps below create local data/results; mock training needs PyTorch but does not download pretrained model weights. Mock serialization also needs pandas/pyarrow. These are examples to run deliberately, not results already obtained by reading this page.

```bash
# 1. Test common interfaces
python examples/robot_foundation_models/common/canonical_dataset.py

# 2. Test SmolVLA adapter (mock mode, no GPU/download)
python examples/robot_foundation_models/smolvla/inference.py

# 3. Collect PushCube expert demonstrations
python examples/robot_foundation_models/smolvla/collect_pushcube_dataset.py --n_episodes 10 --output examples/robot_foundation_models/smolvla/datasets/pushcube_canonical/

# 4. Write a mock Parquet inspection fixture (not certified LeRobot format)
python -c "
import sys
sys.path.insert(0, 'examples/robot_foundation_models/common')
from canonical_dataset import load_episodes_from_dir
from to_lerobot import convert_to_lerobot
episodes = load_episodes_from_dir('examples/robot_foundation_models/smolvla/datasets/pushcube_canonical/')
convert_to_lerobot(episodes, 'examples/robot_foundation_models/smolvla/datasets/pushcube_mock_parquet/', mock=True)
"

# 5. Mock fine-tuning
python examples/robot_foundation_models/smolvla/finetune.py --mock --dataset_dir examples/robot_foundation_models/smolvla/datasets/pushcube_canonical/ --config examples/robot_foundation_models/smolvla/finetune_config.yaml --smoke_test

# 6. Closed-loop evaluation with ablation
python examples/robot_foundation_models/smolvla/closed_loop_eval.py --mock --ablation --n_episodes 10

# 7. Test cross-embodiment adapters
python benchmarks/robot_foundation_models/cross_embodiment_eval.py --mock --smoke-test

# 8. Test planners
python examples/robot_foundation_models/planners/rule_based_planner.py
```

## Data Pipeline

Real LeRobot/RLDS conversion must be validated against the consumer version. A mock Parquet file or NPZ fallback is not a complete version-compatible training dataset simply because the converter finished or a metadata file exists.

```text
PushCube Env + Expert Policy
        ↓
collect_pushcube_dataset.py  →  CanonicalEpisode (.pkl)
        ↓
to_lerobot.py  →  LeRobot dataset (Parquet)
to_rlds.py     →  RLDS dataset (TFRecord / NPZ)
        ↓
SmolVLA / OpenVLA / Octo fine-tuning
        ↓
closed_loop_eval.py  →  Success rate / ablation metrics
```

## Model Status

This table describes interface development, not a current rerun of real-model capabilities. Historical saved evaluation results and missing evidence are separated in [BENCHMARK.md](../../BENCHMARK.md).

| Model | Type | Scale | Status | Recommended Use |
|:------|:-----|------:|:------|:----------------|
| SmolVLA | Lightweight VLA | 450M | Mock interface: ✅<br>Real adapter: 🟡<br>Fine-tuning: ⏳<br>Closed-loop benchmark: ⏳ | Entry, fine-tuning, consumer GPU |
| OpenVLA/OFT | Generalist VLA | 7B | 🟡 Adapter | LIBERO, LoRA, standard benchmark |
| Octo | Generalist Diffusion Policy | 27M/93M | 🟡 Tutorial | Cross-embodiment learning |
| GR00T N1.6 | Humanoid Foundation Model | Large | ⏳ Planned | Humanoid, bimanual manipulation |

## Integration with Existing Tracks

The RFM module connects to the existing repository tracks:

- **VLA**: SmolVLA adapter replaces `vla_demo.py` for standardized inference
- **World Model**: World Model predicts outcomes of RFM-generated actions
- **RL**: RL can post-train RFM policies (reward-weighted fine-tuning)
- **Robot Controller**: RFM outputs target pose/joint commands → Low-level controller tracks them
- **PushCube**: Canonical evaluation environment for all RFM models

## Canonical Dataset Format

All models use the same internal dataset format (`CanonicalEpisode`):

```python
episode = {
    "task": "Push the red cube to the target",
    "robot_type": "pushcube_2d",
    "control_frequency": 20,
    "timestamps": [0.0, 0.05, ...],
    "observation": {
        "images": {"front": [(H, W, 3), ...]},
        "state": [(state_dim,), ...],
    },
    "action": [(action_dim,), ...],
    "language": ["push the red cube...", ...],
    "reward": [0.0, ...],
    "success": [False, ..., True],
}
```

Converters:
- `to_lerobot.py`: real API route or simplified mock Parquet/JSON; verify sample loading, image decoding, fields, statistics and temporal alignment with the pinned consumer before training.
- `to_rlds.py`: TFRecord route or NPZ fallback; NPZ is an inspection fallback, not automatically a standard RLDS dataset for OpenVLA/Octo.
