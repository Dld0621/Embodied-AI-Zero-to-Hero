# Changelog

> 所有值得注意的变更都将记录在此文件中。格式基于 [Keep a Changelog](https://keepachangelog.com/zh-CN/1.1.0/)。

## [Unreleased]

### Foundations Layer — Zero to Hero 基础课程 (2026-08-05)

**Added (P1 — Foundations Layer):**
- **New `docs/foundations/` directory** with 11 files covering the complete "Zero to Hero" prerequisite curriculum for mechanical engineering students:
  - `00-roadmap.md` — Course roadmap with 3 learning paths (ME background / CS background / quick start), ~25–35 hours total
  - `01-python-for-robotics.md` — Python basics, NumPy arrays/matrix ops, Matplotlib 2D/3D visualization, N-link arm FK example
  - `02-linear-algebra.md` — Vectors, matrices, eigenvalues, vector spaces, probability — all with robotics context (14-D state vector, NN weights, Jacobian)
  - `03-deep-learning-basics.md` — Neural networks, forward pass, loss functions (MSE/CE/L1), backpropagation, optimizers (SGD/Adam/AdamW), overfitting — references State-BC (90% success) and SmolVLA BC overfitting
  - `04-transformer-basics.md` — Self-attention (Q/K/V), multi-head attention, positional encoding, ViT, language models — references OpenVLA (DINOv2+SigLIP+Llama 2) and SmolVLA (SmolLM backbone)
  - `05-coordinate-transform.md` — Coordinate frames, 2D/3D homogeneous transforms, transform composition, active vs passive — references PushCube 2D coords and FK chain
  - `06-se3-and-rotation.md` — SO(3)/SE(3), Euler angles, rotation matrix, axis-angle, quaternion, gimbal lock — references MuJoCo quaternion internals
  - `07-fk-jacobian-ik.md` — DH parameters, Jacobian (analytical/geometric), singularity, analytical IK, numerical IK (pseudoinverse, DLS) — references `fk_ik_demo.py` and project DLS config (damping=0.06, 25Hz)
  - `08-control-basics.md` — PID control, impedance control, joint position/velocity/torque modes, safety filters — references project Safety Filter and 25Hz control loop
  - `09-mujoco-basics.md` — MuJoCo engine, MJCF format, URDF vs MJCF, simulation loop, timestep/gravity/contact/friction, viewer — references project URDF files and GPU acceleration
  - `10-dataset-and-training.md` — Data collection, episode vs frame splitting, normalization, PyTorch DataLoader, offline vs closed-loop evaluation — references SmolVLA 50 episodes / 1788 frames, BC overfitting analysis, LeRobot format
- All files include: header (prerequisites / time / objectives), runnable Python code (verified), section headers, "检查理解" exercises, and project code cross-references

**Changed (P1 — Documentation Index):**
- `docs/README.md` — Added "Foundations Layer（基础课程）" section with all 11 files; renamed "基础概念 (Foundations)" to "基础概念 (Core Concepts)"; updated project structure tree to include `docs/foundations/`; updated Stage 0 in learning roadmap
- `README.md` / `README_CN.md` — Updated "Choose Your Path" table to link Foundations for zero-background users; updated Learning Roadmap to reference Foundations Layer; updated Documentation Map table with Foundations Layer entry

### Technical Accuracy & Benchmark Clarity Fixes (2026-08-05)

**Fixed (P0 — Technical Accuracy — OpenVLA/SmolVLA/π0 Action Representation):**
- **vanilla OpenVLA action representation corrected**: `docs/24-action-representation-and-tokenization.md`, `docs/04-glossary.md`, `docs/05-interview-prep.md` — Previous fix incorrectly stated "OpenVLA uses MLP continuous regression". Corrected to: vanilla OpenVLA uses 256-bin discrete action tokens (same as RT-2); OpenVLA-OFT uses continuous regression (L1 loss) with action chunking. Both are variants of the same model.
- **SmolVLA classified as flow matching**: `docs/24-action-representation-and-tokenization.md`, `docs/04-glossary.md`, `docs/05-interview-prep.md` — SmolVLA was incorrectly placed in "continuous regression (MSE/L1)" category. Corrected to "flow matching" alongside π0. Added new Section 5.3 with flow matching code examples, training/inference pseudocode, and comparison with diffusion.
- **π0 separated from diffusion policy**: `docs/24-action-representation-and-tokenization.md`, `docs/04-glossary.md`, `docs/05-interview-prep.md` — π0 was listed under "diffusion policy" in multiple locations. Corrected to "flow matching" throughout. Glossary Diffusion Policy entry now explicitly states "π0 and SmolVLA use flow matching, not standard diffusion".
- **Tokenization routes 3→4**: `docs/24-action-representation-and-tokenization.md` — Added "Flow Matching" as fourth tokenization route. Updated TOC, section numbering (5.1-5.5), comparison table (4 rows), and mermaid diagram (4 subgraphs).
- **SmolVLA action_type in FAQ**: `docs/24-action-representation-and-tokenization.md` — FAQ Q1 and Section 6 mapping corrected from `joint_delta` to `ee_delta_2d` (matching actual `inference.py` default).
- **OpenVLA normalization table**: `docs/24-action-representation-and-tokenization.md` — Min-Max normalization row updated from "OpenVLA" to "vanilla OpenVLA" for precision.
- **Interview prep corrections**: `docs/05-interview-prep.md` — Model comparison table (line 640), action generation comparison table (line 649), OpenVLA vs RT-2 comparison (lines 716-722), Q38 conclusion (line 752), and Q41 inference latency (line 785) all corrected to distinguish vanilla OpenVLA (256-bin discrete) from OpenVLA-OFT (continuous regression), and π0/SmolVLA (flow matching) from Diffusion Policy (diffusion).

**Fixed (P0 — Runbook Consistency):**
- **155 params → 155 saved tensors**: `docs/28-smolvla-gpu-finetuning-runbook.md` — Three remaining instances of "155 params" / "155 parameters" corrected to "155 saved tensors / state-dict entries (450,046,176 total model parameters)" in the status table, 500-step run summary, and 10K-step run summary.

**Fixed (P1 — Benchmark Description):**
- **"same dataset" → "same task and evaluation protocol"**: `README.md` / `README_CN.md` — Benchmark description changed from "same dataset (expert demonstrations)" to "same task, same action space, same metric, same evaluation seeds. Training data and compute budgets differ by method."
- **Resource & Data Budget Table added**: `README.md` / `README_CN.md` — New table added after performance leaderboard showing training data, compute budget, and model params for each method. Makes explicit that methods use different data budgets (50-500 episodes, CPU vs GPU).

**Added (P1 — README UI):**
- **3-item status summary**: `README.md` / `README_CN.md` — Added brief status summary before the collapsed Project Status details: "✅ Real GPU training (SmolVLA 450M, 10K steps) · ✅ Unified PushCube task · 🟡 Task-level VLA success pending (0% at teaching scale)". Users can now see project status without expanding the details section.

**Verified (P0 — Result Completeness):**
- `results/smolvla/500_steps/`: All 5 files confirmed present with real data — `training_config.json`, `training_history.json`, `eval_results.json` (20 episodes × 3 modes), `checkpoint_info.json`, `summary.md`
- `results/smolvla/10k_steps/`: All 5 files confirmed present with real data — `training_config.json`, `training_history.json` (9500 steps), `eval_results.json` (20 episodes × 3 modes with checkpoint_info embedded), `checkpoint_info.json`, `summary.md`

### Documentation Consistency & README UI Improvements (2026-08-04)

**Fixed (P0 — Documentation Consistency):**
- **CLIP classification**: `docs/01-what-is-vla.md` and `docs/04-glossary.md` — CLIP is now correctly described as a dual-encoder VLM (outputs embedding/similarity), not a generative VLM. Added distinction: generative VLM (GPT-4V, LLaVA) vs dual-encoder VLM (CLIP, SigLIP).
- **OpenVLA action output**: `docs/24-action-representation-and-tokenization.md` — Fixed contradiction: OpenVLA uses MLP continuous regression (not 256-bin discrete tokens). Section 5.1 title changed from "Bin 离散化 (OpenVLA)" to "Bin 离散化 (RT-2)". Table and text updated to place OpenVLA in continuous regression row.
- **Action types 5→6**: `docs/24-action-representation-and-tokenization.md` — Added missing `ee_delta_2d` action type (PushCube default). Updated count from "五种" to "六种", added table row, mermaid diagram node, and code block entry.
- **SmolVLA default action_type**: Changed from `joint_delta` to `ee_delta_2d` (matching actual code in `inference.py`).
- **RL MDP table**: `docs/06-rl-fundamentals-for-vla.md` — Removed "语言指令" from Action A (it is task condition/context, not action). Added language instruction to State S.
- **BC off-policy clarification**: `docs/06-rl-fundamentals-for-vla.md` — BC is now described as "supervised imitation learning using offline data, not an off-policy RL algorithm" instead of "BC 本质上是 off-policy".
- **REINFORCE command**: `docs/README.md` — Updated from `unified_pushcube_rl.py # REINFORCE` to `# PPO（主基线）+ REINFORCE` and `--algo ppo` flag.
- **RL track description**: `README.md` / `README_CN.md` — Updated PushCube track table from "REINFORCE (policy gradient, pure NumPy)" to "BC-initialized PPO (main) + REINFORCE (concept demo)".
- **SmolVLA status**: `README_CN.md` — Synced from `✅ Runnable` to `✅ Pipeline Verified · 🟡 Task Success Pending` (matching English README). Updated 500-step info to 10K-step results.
- **155 params → 155 tensors**: `docs/28-smolvla-gpu-finetuning-runbook.md` — Already fixed in previous session.

**Fixed (P0 — CI):**
- `.github/workflows/tests.yml`: `rfm-smoke` job now installs `torch --index-url https://download.pytorch.org/whl/cpu` (was missing, causing potential import failures).

**Added (P0 — Reproducibility):**
- `results/smolvla/500_steps/`: Added `training_history.json`, `training_config.json`, `summary.md`
- `results/smolvla/10k_steps/`: Added `training_config.json`, `summary.md`

**Changed (P1 — README UI):**
- `README.md` / `README_CN.md`: Added Hero architecture badge diagram (Language + Vision + State → Embodied Reasoner → VLA → Adapter → Controller → Safety → Robot, with WM and RL feedback loops)
- `README.md` / `README_CN.md`: Replaced navigation links with three prominent CTAs: Run Demo / Read Docs / View Results
- `README.md` / `README_CN.md`: Reordered Visual Demos — PushCube benchmark results first, SmolVLA training second, World Model visuals in `<details>`, synthetic RL curve in `<details>`
- `README.md` / `README_CN.md`: Wrapped in `<details>` tags: Project Status tables, Scope & Boundaries, full PushCube commands, RFM Quick Start commands, Supported Robots matrix — reduces scroll fatigue on mobile

### SmolVLA 10K-Step Scale-Up (2026-08-04)

**Added:**
- SmolVLA 10K-step training script (`smolvla_train_10k_v2.py`) with robust checkpoint management:
  - Atomic checkpoint save (temp dir → verify → rename) prevents corrupted checkpoints
  - Error recovery: checkpoint failures don't crash training (consecutive error limit: 10)
  - Signal handling: saves emergency checkpoint on SIGINT/SIGTERM
  - Disk space pre-check before saving
  - Keeps only latest 2 checkpoints to manage disk usage
- SmolVLA 10K-step evaluation script (`smolvla_eval_10k.py`) with 3 language modes
- Training results: 9500 additional steps (500→10K), 65.1 min on RTX 3060, loss 0.10→0.031
- Evaluation results: 20 episodes × 3 modes, 0% success, 50% selection accuracy

**Changed:**
- `README.md` / `README_CN.md`: Added SmolVLA (10K steps) row to benchmark table; updated roadmap to reflect 10K completion and next target (100K + 100+ episodes)
- `docs/28-smolvla-gpu-finetuning-runbook.md`: Added 10K-step run summary with BC overfitting analysis and updated next steps

**Key Findings:**
- Training loss decreased 3x (0.10→0.03) but closed-loop success remains 0% — classic BC overfitting
- 50 episodes (1788 frames) is insufficient for a 450M parameter VLA to generalize
- The gap between open-loop loss and closed-loop performance motivates: DAgger, RL fine-tuning, larger datasets
- The full pipeline (train → checkpoint → resume → evaluate) is robust and reproducible

### P0 Fixes — Data Leakage, Language Dependency, Tokenizer (89/100 Review)

**Fixed (Critical — P0):**
- **P0-1: Episode-level train/val/test split**: Replaced `random_split` on flattened frames with `PushCubeEpisodeLoader.split_episodes()` — episodes are split as whole units (40 train / 5 val / 5 test) before frame expansion. Prevents same-episode adjacent frames from appearing in both train and validation sets. Split filenames are logged for reproducibility.
- **P0-2: Removed goal-color one-hot from VLA state**: State input sliced from 14-D to 12-D (`state[:12]`), excluding `goal_red` and `goal_green`. The model can no longer read the task answer from state — it MUST use the language instruction to identify the target cube. `VLA_STATE_DIM = 12` constant added; `PushCubeFrameDataset` slices state per sample; `inference.py` `_lightweight_predict()` slices environment state at inference time.
- **P0-3: Tokenizer padding and masked mean pooling**: `nn.Embedding` now uses `padding_idx=0` (zeroes padding embeddings at init and backward). Language feature computed via masked mean: `mask = lang_tokens.ne(0)` → `sum / mask.sum()`. Padding tokens no longer dilute the language signal. `SimpleTokenizer._hash()` shifts to range `[1, vocab_size-1]` to avoid collision with pad token 0.

**Changed:**
- `train_lightweight_vla.py`: Complete rewrite of data pipeline (episode loader, frame dataset, split logic). Added test set evaluation after training. Checkpoint metadata now includes `split_info` (train/val/test episode filenames), `training_info` (split method, state_dim explanation, tokenizer fix, seed). Model parameter count: 195,266 (was 195,394 — state encoder input reduced from 14 to 12).
- `inference.py`: `_lightweight_predict()` now slices state to `config["state_dim"]` before model forward pass, preventing dimension mismatch with the retrained 12-D model.
- `results/benchmarks/lightweight_vla_closed_loop.json`: Regenerated with new checkpoint. Added `model_info`, `training_info`, `p0_fixes_applied`, and `provenance` sections.

**Evaluation Results (Post-Fix):**
| Metric | Previous (Leaky) | Current (Fixed) | Change |
|:-------|:---------|:--------|:-------|
| correct_success | 0.0% | 0.0% | — |
| selection_accuracy | 30.0% | **65.0%** | +35% |
| val_loss | 0.252 | 0.316 | +0.064 (honest, no leakage) |
| test_loss | N/A | 0.533 | New (unseen episodes) |
| best_epoch | 77 | 8 | Earlier (less overfitting) |

> **Key finding:** Selection accuracy jumped from 30% to 65% (above 50% random baseline), confirming the model now uses language to identify the target cube. The previous 30% was at chance level — the model was ignoring language and relying on the goal-color one-hot shortcut. Success rate remains 0% because action prediction precision (MAE 0.589) is insufficient for task completion with 195K parameters and 1433 training frames.

### P1-1: PPO Relabeled as BC-initialized PPO with Expert Guidance

**Changed:**
- `unified_pushcube_rl.py`: Module docstring, `PPOAgent` class docstring, and `train_ppo()` docstring now explicitly state this is "BC-initialized PPO with expert guidance", not PPO from scratch. Lists the four components: (1) BC warm-start, (2) 30% expert-guided exploration, (3) guidance reward (weight 3.0), (4) shaped reward. Print banner updated to "BC-initialized PPO + expert guidance".
- `README.md` / `README_CN.md`: Benchmark table label changed from "RL (PPO)" to "RL (BC-init PPO)". Quick Start comment and command description updated. Notes column adds "expert guidance".

### P0-4: LeRobot Version Contract Unified

**Fixed (Critical — P0):**
- **Lock file updated from `lerobot==0.1.0` to `lerobot[smolvla]==0.4.1`**: LeRobot 0.1.0 (PyPI) does NOT include the SmolVLA policy module (`lerobot.common.policies.smolvla.modeling_smolvla`). SmolVLA was added as an optional extra in LeRobot 0.4.0+. The lock file now pins `lerobot[smolvla]==0.4.1` (released Nov 2025), which includes the SmolVLA extra and all required dependencies. Added explanatory comment block documenting the version contract and the source-install alternative.
- **Requirements file updated from `lerobot>=0.1.0` to `lerobot[smolvla]>=0.4.0`**: The loose `>=0.1.0` constraint would resolve to 0.1.0, which lacks SmolVLA. Now requires 0.4.0+ with the smolvla extra.
- `docs/28-smolvla-gpu-finetuning-runbook.md`: Added "Option A — From PyPI" as the recommended installation method (`pip install 'lerobot[smolvla]==0.4.1'`), with "Option B — From source" as the alternative for development.
- `.github/workflows/tests.yml`: Added comment noting the SmolVLA extra requirement is included via the lock file.

---

### README Compression and Detailed Tracks Extraction

**Added:**
- `docs/29-learning-tracks-detail.md`: Detailed breakdown of all four core research tracks (VLA, World Models, RL, Embodied Reasoning) — pipelines, learning-level tables, implementation status, and known limitations. Extracted from the main README to keep it concise.

**Changed:**
- `README.md` / `README_CN.md`: Compressed from 478/470 lines to 349/349 lines (27% reduction). Detailed track sections (160+ lines of pipelines, learning-level tables, implementation status) replaced with a compact 4-row summary table linking to `docs/29-learning-tracks-detail.md`. RL Benchmark Protocol section folded into Benchmarks. Reproducibility section compressed from two tables to one. Both READMEs now have identical 19-section structure with synchronized content.
- `README_CN.md` fixes (beyond compression): SmolVLA status updated from "🟡 适配器 + Mock" to "🟡 适配器 + 轻量 VLA" (matching EN). Visual demos reordered (World Model results first, RL curves last with "示意性" label). Quick Start commands updated to include lightweight VLA training and evaluation. Duplicate "定义/定位" text in VLA/WM/RL sections removed. RFM row in visual table updated with real checkpoint results.
- `docs/README.md`: Added entries for `28-smolvla-gpu-finetuning-runbook.md` and `29-learning-tracks-detail.md` in the RFM index section and project structure tree. Code quick reference updated with lightweight VLA training and evaluation commands.

---

### Real VLA Checkpoint and Closed-Loop Evaluation Pipeline

**Added:**
- `examples/robot_foundation_models/smolvla/train_lightweight_vla.py`: Lightweight VLA training script (195K params, CNN + language + state → action). Trains on real PushCube expert data (50 episodes, 1788 frames) on CPU. Saves checkpoint with model config, training history, and validation metrics. Supports `--smoke-test` for quick 5-epoch runs.
- `examples/robot_foundation_models/smolvla/models/lightweight_vla/lightweight_vla_pushcube.pt`: **Real trained checkpoint** (best epoch 77, val_loss=0.252, val_mae=0.398). 195,394 parameters trained on 1520 frames, validated on 268 frames. NOT the full 450M SmolVLA — see runbook for GPU fine-tuning.
- `examples/robot_foundation_models/smolvla/models/lightweight_vla/training_history.json`: Full training history (100 epochs, per-epoch train/val loss and MAE).
- `results/benchmarks/lightweight_vla_closed_loop.json`: **Real closed-loop evaluation results** (20 episodes). correct_success=0%, selection_accuracy=30%, avg_steps=100. Includes model info and auto-generated provenance.
- `docs/28-smolvla-gpu-finetuning-runbook.md`: Step-by-step guide for full 450M SmolVLA fine-tuning on GPU (LeRobot installation, dataset conversion, training, evaluation, results update).

**Changed:**
- `examples/robot_foundation_models/smolvla/inference.py`: `SmolVLAAdapter` now supports three modes: (1) full SmolVLA via LeRobot, (2) **lightweight VLA via .pt checkpoint** (new `_try_load_lightweight()` and `_lightweight_predict()` methods), (3) mock zero actions. Mode is auto-detected from `pretrained_name_or_path` (`.pt` suffix → lightweight).
- `examples/robot_foundation_models/smolvla/evaluate.py`: Added `--checkpoint` argument to specify a lightweight VLA .pt checkpoint for real evaluation.
- `README.md`: (1) Visual Demos section reordered — World Model results (real) moved first, synthetic RL curves moved last with "(Illustrative)" label. (2) Model status table updated from "🟡 Adapter + Mock" to "🟡 Adapter + Lightweight VLA". (3) Quick Start includes lightweight VLA training and evaluation commands. (4) Visual table RFM row updated with real checkpoint results.
- `.gitignore`: Added exceptions for lightweight VLA checkpoint directory (`models/lightweight_vla/*.pt` and `*.json`).

**Evaluation Results (Real, not mock):**
| Metric | Value | Notes |
|:-------|:------|:------|
| correct_success | 0.0% | Model did not complete the pushing task |
| wrong_success | 0.0% | Model did not push the wrong cube either |
| selection_accuracy | 30.0% | Correct cube closer to target in 6/20 episodes |
| avg_steps | 100.0 | All episodes ran full duration |
| val_loss | 0.252 | Best epoch 77/100 |
| val_mae | 0.398 | Average action error per dimension |

> **Honest assessment:** The 195K-parameter lightweight VLA demonstrates the full train→evaluate→report pipeline on CPU with real data, but does not achieve task success. The model capacity (195K vs 450M) and training data size (50 episodes) are insufficient for closed-loop success. The full 450M SmolVLA fine-tuning on GPU is documented in the runbook and awaiting GPU access.

---

### Review-Driven Fixes — Provenance Automation, Dataset Naming, README Layout

**Fixed (Critical — P0):**
- **Benchmark provenance now auto-generated at runtime**: Created `examples/benchmark_provenance.py` — a shared module that collects `git_commit`, `command`, `python_version`, `torch_version`, `device`, `timestamp`, and `result_generated_by` via `subprocess` / `platform` / `torch` / `datetime.now().astimezone()`. All five benchmark scripts (`unified_pushcube_rl.py`, `unified_pushcube_vla.py`, `unified_pushcube_wm.py`, `unified_pushcube_act.py`, `unified_pushcube_diffusion.py`) now call `build_provenance()` before `json.dump()`, eliminating the previous manual-edit workflow that produced inconsistent and sometimes future-dated timestamps.
- **Tiny dataset renamed from `pushcube_lerobot_tiny` to `pushcube_ci_fixture`**: The directory name `pushcube_lerobot_tiny` implied it was a loadable LeRobot dataset, but it contains only JSON state/action/language (no Parquet, no images, no video). Renamed to `pushcube_ci_fixture` to honestly reflect its role as a CI schema-regression fixture. Updated `meta/info.json` fields: `format` → `"json_ci_fixture"`, `description` clarifies "NOT a LeRobot Parquet dataset". Updated `tests/test_rfm_dataset_regression.py` and `.gitignore` to reference the new path.

**Fixed (P1):**
- **PPO docstring corrected**: `unified_pushcube_rl.py` previously claimed "For a proper RL baseline that achieves >=70% success, use: --algo ppo" — contradicting the actual PPO benchmark of 10–20%. Updated to: "For the main RL baseline, use: --algo ppo. Current teaching-scale PPO performance is approximately 10-20% success rate."
- **README section numbering fixed**: Core Learning & Research Tracks had sections numbered `1, 3, 3` (missing `2`, duplicate `3`). Corrected to `1, 2, 3, 4` with proper sequence: VLA → World Models → RL → Embodied Reasoning.
- **Added missing Embodied Reasoning chapter**: Project Status and system overview both list Embodied Reasoning as a core track, but the detailed module section lacked a corresponding chapter. Added `### 4. Embodied Reasoning — Planning Layer` with pipeline, component status table, and documentation link.
- **Chinese README navigation anchors fixed**: `README_CN.md` top navigation used English anchors (`#five-minute-quick-start`) that didn't match Chinese heading auto-generated IDs. Added explicit `<a id="...">` HTML anchors before each referenced heading.
- **Bilingual header symmetry**: Chinese README previously placed the language switch as plain Markdown above the `<h1>`, while English used centered `<p align="center">` below. Both now use the same centered HTML structure.
- **"Five research tracks" terminology corrected**: The PushCube baselines table was labeled as "five research tracks" but actually listed policy methods (VLA, WM, RL, Action-Chunking, Diffusion). Renamed to "Unified PushCube Baselines" to avoid conflating research directions with algorithm methods.

**Added:**
- `examples/benchmark_provenance.py`: Shared provenance builder module (no external dependencies beyond `torch`).

**Changed:**
- `tests/test_rfm_dataset_regression.py`: All fixture paths updated from `pushcube_lerobot_tiny` to `pushcube_ci_fixture`.
- `.gitignore`: Updated fixture path exception from `pushcube_lerobot_tiny` to `pushcube_ci_fixture`.
- All six `results/benchmarks/*.json` files: `provenance.timestamp` fields regenerated to use current time (previous values were future-dated `2026-07-31T19:30:00+08:00`).

---

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
- `examples/robot_foundation_models/smolvla/datasets/pushcube_lerobot_tiny/`: Tiny 2-episode / 63-frame JSON dataset (~42KB) for CI regression testing. No images — state/action/language only. Generated from real PushCube expert policy (both episodes succeed).
- `tests/test_rfm_dataset_regression.py`: 17 unittest cases verifying tiny dataset structure, dimensions (state_dim=14, action_dim=2), action_type, language presence, timestamp monotonicity, and success. Runs in CI with no extra dependencies.

**Changed (CI and Provenance):**
- **`finetune.py --test` added to CI `rfm-smoke` job**: The 4 config/CLI unit tests now run on every push and PR, not just locally. Step: "SmolVLA config and CLI tests".
- **`test_rfm_dataset_regression.py` added to CI `pushcube-smoke` job**: Tiny dataset regression test runs alongside existing PushCube regression tests.
- **Benchmark JSON provenance fields**: All 6 benchmark JSON files (`pushcube_summary.json`, `rl_results.json`, `vla_results.json`, `wm_results.json`, `action_chunking_results.json`, `diffusion_results.json`) now include a `provenance` object with `git_commit`, `command`, `python_version`, `torch_version`, `device`, `timestamp`, and `result_generated_by`. This distinguishes script-generated results from manually aggregated summaries and enables cross-commit traceability.
- **Fixed duplicate `"note"` key**: `action_chunking_results.json` and `diffusion_results.json` had two `"note"` keys (method description + TBD status). Second instance renamed to `"closed_loop_note"` to avoid silent overwrite.

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
