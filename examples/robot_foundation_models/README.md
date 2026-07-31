# Robot Foundation Models

> Unified interface for integrating robot foundation models (SmolVLA, OpenVLA, Octo, GR00T) into a single experimental pipeline.

## Architecture

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

The external control loop **never changes** when swapping models — only the adapter changes.

## Quick Start

```bash
# 1. Test common interfaces
cd examples/robot_foundation_models/common
python canonical_dataset.py

# 2. Test SmolVLA adapter (mock mode, no GPU/download)
cd ../smolvla
python inference.py

# 3. Collect PushCube expert demonstrations
python collect_pushcube_dataset.py --n_episodes 10 --output datasets/pushcube_canonical/

# 4. Convert to LeRobot format
python -c "
from common.canonical_dataset import load_episodes_from_dir
from common.to_lerobot import convert_to_lerobot
episodes = load_episodes_from_dir('datasets/pushcube_canonical/')
convert_to_lerobot(episodes, 'datasets/pushcube_lerobot/', mock=True)
"

# 5. Mock fine-tuning
python finetune.py --mock --dataset_dir datasets/pushcube_canonical/ --smoke_test

# 6. Closed-loop evaluation with ablation
python closed_loop_eval.py --mock --ablation --n_episodes 10

# 7. Test cross-embodiment adapters
cd ../../benchmarks/robot_foundation_models
python cross_embodiment_eval.py --mock --smoke-test

# 8. Test planners
cd ../../examples/robot_foundation_models/planners
python rule_based_planner.py
```

## Data Pipeline

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
- `to_lerobot.py` → SmolVLA / π0 (Parquet + JSON metadata)
- `to_rlds.py` → OpenVLA / Octo (TFRecord / NPZ fallback)
