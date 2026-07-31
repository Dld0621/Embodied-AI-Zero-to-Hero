# Changelog

> 所有值得注意的变更都将记录在此文件中。格式基于 [Keep a Changelog](https://keepachangelog.com/zh-CN/1.1.0/)。

## [Unreleased]

### SmolVLA Fine-tuning Pipeline — Config, CLI, and Dataset Alignment

**Fixed (Critical — P0):**
- **`load_config()` now flattens nested YAML sections**: The YAML config (`finetune_config.yaml`) uses nested sections (`model`, `training`, `hardware`, `logging`, `dataset`), but `load_config()` previously used `defaults.update(loaded)` which only placed these dicts at the top level without flattening. As a result, `real_train()` read flat defaults (`batch_size=8`, `epochs=30`) instead of the YAML values (`batch_size=64`, `steps=20000`). Now `load_config()` properly extracts `training.batch_size` → `config["batch_size"]`, `training.steps` → `config["steps"]`, `model.pretrained_name_or_path` → `config["pretrained_name_or_path"]`, `hardware.device` → `config["device"]`, `logging.project` → `config["job_name"]`, etc. Verified by 4 unit tests (`python finetune.py --test`).
- **`real_train()` aligned with official LeRobot CLI**: Replaced `python -m lerobot.scripts.train` with `lerobot-train` (the official entry point per [HuggingFace SmolVLA docs](https://huggingface.co/docs/lerobot/main/smolvla)). Command now uses `--policy.path=lerobot/smolvla_base`, `--dataset.repo_id=`, `--steps=` (instead of `epochs=`), `--policy.device=cuda`, and `--job_name=`. Removed unverified `policy.action_dim=2` / `policy.num_motors=2` overrides — action dimension is now determined by the dataset's `action` feature (correct LeRobot convention). Added `shutil.which("lerobot-train")` validation.
- **Benchmark JSON results synchronized with README**: Updated `results/benchmarks/pushcube_summary.json`, `rl_results.json`, `vla_results.json`, `rl_config.json` to match README claims (Expert ~100%, State-BC 90%, PPO 10–20%). Previous JSON files still contained old data (Expert 65%, REINFORCE 0%). Action-chunking and Diffusion success rates changed from `0.0` to `null` (TBD — not yet evaluated for closed-loop success). State-BC input description updated from "14-D state + language" to "14-D state with goal-color one-hot".

**Fixed (P1):**
- **Added `--steps` and `--batch_size` CLI arguments**: `main()` now supports `--steps 20000` and `--batch_size 64` for real training mode. `--epochs` retained but documented as "mock mode only; real mode uses --steps".
- **Local dataset path handling**: `build_train_command()` (new function) distinguishes HF Hub (`--dataset.repo_id=id`) from local (`--dataset.repo_id=local/<name>` + `--dataset.root=<parent_dir>`).
- **Extracted `build_train_command()` from `real_train()`**: Command construction is now a separate testable function. Added 4 unit tests (`python finetune.py --test`, no GPU needed).
- **README RL description consistency**: Learning Levels table changed from "REINFORCE on PushCube" to "PPO (main baseline) + REINFORCE (concept demo)". RL Benchmark Protocol renamed to "PushCube (PPO)".
- **Benchmark table header arrow**: Changed `Success Rate ↓` to `Success Rate ↑` in both READMEs.
- **State-BC input description**: Changed from "14-D state + language" to "14-D state with goal-color one-hot" in both READMEs and all benchmark JSON files.
- **Fixed Cyrillic character in default `--output_dir`**: `smolvlа_pushcube` (Cyrillic 'а') → `smolvla_pushcube` (Latin 'a').

**Added:**
- `examples/robot_foundation_models/smolvla/datasets/README.md`: Complete instructions for collecting 50 expert demonstrations, converting to LeRobot format, and running mock/real fine-tuning.
- `examples/robot_foundation_models/smolvla/datasets/pushcube_lerobot/meta/info.json`: LeRobot-format dataset metadata (50 episodes, 1788 frames, action_dim=2, state_dim=14, action_type="ee_delta_2d").

---

### RFM Decoupling, Action Space, and Benchmark Separation

**Fixed (Critical — P0):**
- **Robot Foundation Models no longer bound to dexterous hands**: Removed all remaining Retargeting / dexterous-hand / OmniHand references from RFM documentation and code. Architecture diagram updated from `Embodiment Adapter → Retargeting / IK / Controller` to `Robot Action Adapter → Low-level Controller`.
- **SmolVLA status corrected**: README model status table changed from `✅ Runnable` to `🟡 Adapter + Mock Pipeline`. SmolVLA will only return to ✅ after real weights load, real fine-tuning, correct action_dim, closed-loop evaluation, and real result JSON submission.
- **Real SmolVLA fine-tuning entry implemented**: `finetune.py` `real_train()` replaced placeholder with a complete LeRobot training pipeline using the official `lerobot-train` CLI.
- **Action space strictly fixed to 2-D `[dx, dy]`**: `SmolVLAAdapter` now defaults to `action_type="ee_delta_2d"` and `action_dim=2`. `collect_pushcube_dataset.py` records `action_type="ee_delta_2d"`. `action_schema.py` added `"ee_delta_2d"` to `VALID_ACTION_TYPES`.
- **Eliminated all `[:2]` action truncation**: Removed silent truncation from all evaluation and benchmark scripts. Truncation replaced with strict `ValueError` assertion.
- **Mock and real benchmark results separated**: Results saved to `results/benchmarks/rfm/mock/` (when `--mock`) or `results/benchmarks/rfm/real/` (when real).

**Changed (Cleanup):**
- `docs/25-cross-embodiment-adaptation.md`: Removed all dexterous-hand references. Replaced with `UR5eAdapter` example.
- `benchmarks/robot_foundation_models/cross_embodiment_eval.py`: Removed `OmniHandAdapter`. Replaced with `UR5eAdapter`.

---

### Real Benchmark Results: Expert 100%, State-BC 90%, PPO Non-Zero

**Fixed (Critical — P0):**
- **Expert policy success rate**: Increased from ~65% to **~100%** (50 random seeds). Three-phase heuristic: flank → behind → push.
- **State-BC baseline**: Added MLP policy with geometric feature engineering. Achieves **90% success** (100 episodes / 50 epochs).
- **RL baseline upgraded from REINFORCE to PPO**: Full PPO implementation (Actor-Critic + GAE + BC pre-training + expert-guided exploration). PPO achieves **10–20% success** (500 episodes); BC pre-training alone reaches 40%.
- **SmolVLA Adapter action chunk fix**: `_select_action_to_chunk()` now properly drains the policy's internal action queue.
- **SmolVLA Adapter camera mapping**: Added explicit `CAMERA_MAPPING` dict.

**Fixed (P1):**
- **Language ablation table logic**: Updated to single-model, same-episode, multi-condition evaluation.
- **World Model state dimension**: Fixed from 13-D to **14-D** across all documentation and code.
- **Quick Start default**: Changed to `unified_pushcube_vla.py --smoke-test`.
- **README benchmark table**: Updated with real results.

---

### Repository Restructure: Robot Foundation Models as Primary Track

**Changed (Major):**
- Repository primary focus shifted from "Dexterous Retargeting" to "Robot Foundation Models".
- README.md + README_CN.md: Removed "Dexterous Retargeting" from Core Research Tracks. New track order: RFM → VLA → World Models → RL → Embodied Reasoning.
- System overview mermaid diagram updated. Removed Retargeting node entirely.
- Supported Robots table replaced. Removed Shadow Hand / Allegro / LEAP / OmniHand. Added PushCube (2D), Franka Panda, UR5e, AgiBot X1, Unitree G1.
- 14 retargeting-specific docs removed from `docs/README.md` index.
- All docs updated: removed retargeting references, replaced with Robot Adapter / cross-embodiment concepts.

---

### Robot Foundation Models Module (Initial)

**Added:**
- `examples/robot_foundation_models/`: New upper-layer module unifying VLA, World Model, RL, and Retargeting under a single observation/action interface.
  - `common/observation_schema.py`: `RobotObservation` dataclass.
  - `common/action_schema.py`: `ActionChunk` dataclass with 5 supported action types.
  - `common/model_interface.py`: `RobotFoundationModel` Protocol.
  - `common/embodiment_adapter.py`: `EmbodimentAdapter` ABC + `GenericAction`.
  - `common/safety_filter.py`: `SafetyFilter`.
  - `smolvla/`: SmolVLAAdapter (450M, mock mode for CI).
  - `openvla/`: OpenVLAAdapter (7B, LoRA config).
  - `planners/`: Rule-based and VLM task planners.
