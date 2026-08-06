# PushCube Benchmark: Full Results & Reproduction Guide

> This page contains the complete benchmark tables, ablation studies, hardware specs, and reproduction commands. For a quick summary, see the [main README](README.md#benchmarks).

---

## 1. Environment Configuration

| Parameter | Value |
|:----------|:------|
| Environment | PushCube (dual-cube, language-conditioned) |
| State space | 14-D (arm x/y, red cube x/y, green cube x/y, goal x/y, goal_color one-hot[2]) |
| Action space | 2-D (`ee_delta_2d`: [dx, dy]) |
| Observation | 128×128 RGB image + 14-D state + natural language instruction |
| Task | Push the correct cube (identified by language) to the goal position |
| Success condition | Target cube within threshold distance of goal |
| Max steps per episode | 80 (all methods, including SmolVLA) |
| Evaluation episodes | Varies by method (20–100); see Section 2 |
| Evaluation seeds | 3000–3019 (SmolVLA and RL); other methods used different seeds |

### Language Conditions

| Mode | Description |
|:-----|:------------|
| **Correct** | Language instruction matches the actual target cube |
| **Swapped** | Language instruction names the wrong cube (ablation) |
| **None** | Empty language string (vision-only baseline) |

---

## 2. Unified Leaderboard

All methods evaluated on the **same environment** (dual-cube PushCube), **same task definition**, **same action space** (2-D `ee_delta_2d`), **same metric** (success rate), and **same max steps** (80). Evaluation episode counts and seeds differ by method (see table). Training data and compute budgets also differ (see Section 3). This is a teaching experiment, not a strictly controlled leaderboard.

| Method | Input | Train | Eval Eps | Success Rate ↑ | Notes |
|:-------|:------|:------|---:|:---:|:------|
| Expert | State | — | 50 | **~100%** | Three-phase heuristic (flank → behind → push) |
| State-BC | 14-D state with goal-color one-hot | 100 episodes / 50 epochs | 100 | **90%** | MLP + geometric feature engineering |
| VLA (Full) | RGB + language | 100 episodes / 50 epochs | 100 | **0%** | CNN + word embedding → MLP; needs more data |
| Action-Chunking | RGB hist + language | 100 episodes / 30 epochs | — | **N/A** | K-frame Transformer, no CVAE; trained but NOT yet evaluated |
| Diffusion Policy | RGB + language | 100 episodes / 30 epochs | — | **N/A** | DDPM, 20 steps, action horizon=10; trained but NOT yet evaluated |
| RL (BC-init PPO) | 14-D state | 500 episodes | 20 | **15%** | Actor-Critic + GAE + BC warm-start + expert guidance; BC pretrain 40% |
| WM-MPC (CEM) | 14-D state | 100 episodes WM training | 20 | **0%** | CEM planner, H=10, 500 samples, 3 iterations |
| WM-MPC (Random) | 14-D state | 100 episodes WM training | 20 | **0%** | Random shooting, H=10, 1000 samples |
| SmolVLA (500 steps) | RGB + language + state | 50 episodes / 500 steps GPU | 20 | **0%** | 450M params, bf16; loss 0.47→0.10; baseline checkpoint |
| SmolVLA (10K steps) | RGB + language + state | 50 episodes / 10K steps GPU | 20 | **0%** | 450M params, bf16; loss 0.10→0.03; 20x scale-up; BC overfitting |

> **Note:** "N/A" means the method was trained but NOT yet evaluated for closed-loop success. Previous reports showing "0%" for Action-Chunking and Diffusion were incorrect — their `success_rate` is `null` in the raw data. Source of truth: [`results/benchmarks/benchmark_v2.json`](results/benchmarks/benchmark_v2.json).

---

## 3. Resource & Data Budget Table

| Method | Training Data | Compute | Model Params |
|:-------|:-------------|:--------|:-------------|
| Expert | N/A (heuristic) | CPU | N/A |
| State-BC | 100 episodes / ~3.6K frames | CPU | ~10K |
| VLA (Full) | 100 episodes / ~3.6K frames | CPU | 195K |
| Action-Chunking | 100 episodes / ~3.6K frames | CPU | ~500K |
| Diffusion Policy | 100 episodes / ~3.6K frames | CPU | ~1M |
| RL (BC-init PPO) | 500 episodes (on-policy) | CPU | ~10K |
| WM-MPC (CEM/Random) | 100 episodes WM training | CPU | ~50K (WM) |
| SmolVLA (500 steps) | 50 episodes / 1788 frames | GPU (RTX 3060, bf16) | 450M (100M trainable) |
| SmolVLA (10K steps) | 50 episodes / 1788 frames | GPU (RTX 3060, bf16) | 450M (100M trainable) |

> **Note:** This is a multi-method teaching experiment on a unified task, not a strictly fair leaderboard. Methods use different data budgets (50–500 episodes), compute (CPU vs GPU), and model scales (10K–450M params). The comparison illustrates algorithmic differences at teaching scale, not production performance.

---

## 4. SmolVLA Experiment Details

### 4.1 Training Configuration

| Parameter | 500-step | 10K-step |
|:----------|:---------|:---------|
| Model | SmolVLA 450M (`lerobot/smolvla_base`) | Same |
| Total params | 450,046,176 | Same |
| Trainable params | 99,880,992 (LoRA-style unfreeze) | Same |
| Dataset | PushCube dual-cube, 50 episodes / 1788 frames | Same |
| Action dim | 2 (`ee_delta_2d`) | Same |
| Steps | 500 | 10,000 (resumed from 500) |
| Batch size | 2 | 2 |
| Precision | bf16 | bf16 |
| Optimizer | AdamW | AdamW |
| LR schedule | Fixed | Cosine decay 1e-4 → 2.5e-6 |
| Hardware | NVIDIA RTX 3060 Laptop (6.4 GB VRAM) | Same |
| Training time | ~15 min | 65.1 min total |
| Checkpoint size | 399.5 MB (155 saved tensors) | Same |

### 4.2 Training Results

| Metric | 500-step | 10K-step |
|:-------|:---------|:---------|
| Initial loss | 0.47 | 0.10 (resumed) |
| Final loss | 0.103 | 0.031 |
| Average loss | 0.069 | 0.030 |
| Best loss | 0.028 | 0.004 |

### 4.3 Language Ablation Results

| Mode | 500-step Success | 500-step Selection | 10K-step Success | 10K-step Selection |
|:-----|:---:|:---:|:---:|:---:|
| Correct language | 0% | 50% | 0% | 50% |
| Swapped language | 0% | 50% | 0% | 45% |
| No language | 0% | 50% | 0% | 50% |

- **Selection accuracy** = percentage of episodes where the correct cube ends up closer to the goal
- **Key finding:** Selection accuracy stays at ~50% (chance level) across all modes, indicating the model has not learned to use language to identify the target cube at this data scale

### 4.4 Result Files

| File | 500-step | 10K-step | Description |
|:-----|:---:|:---:|:------------|
| `training_config.json` | ✅ | ✅ | Checkpoint metadata (step, loss, param count) |
| `training_history.json` | ✅ | — | Per-step loss history; only the 500-step history is committed |
| `eval_results.json` | — | — | Per-episode raw evaluation files are not committed |
| `checkpoint_info.json` | — | — | Checkpoint structure files are not committed |
| `summary.md` | ✅ | ✅ | Human-readable summary |

Location: `results/smolvla/500_steps/` and `results/smolvla/10k_steps/`

The SmolVLA evaluation numbers are aggregate, **reported experiment evidence** preserved in the committed summaries and canonical JSON. They are not independently re-aggregatable from per-episode files in this repository. New experiments must save their raw evaluation JSON under `results/benchmarks/` before the canonical table is updated.

---

## 5. World Model Results

**MLP dynamics model:**
- best validation loss = **0.0409**
- Multi-step prediction error: **H=1: 0.0708, H=5: 0.2961, H=10: 0.5560**

These values come from [`results/benchmarks/wm_results.json`](results/benchmarks/wm_results.json) and are mirrored in the canonical [`benchmark_v2.json`](results/benchmarks/benchmark_v2.json). Earlier narrative values (`0.011`, `0.042/0.176/0.350`) were stale and have been removed.

---

## 6. Analysis

### Why most methods get 0%

PushCube dual-cube requires the arm to navigate behind the correct cube and push it — a contact-rich manipulation task. At teaching scale (50–200 episodes, small models), most methods cannot learn the precise contact dynamics.

- **State-BC (90%)** works because it has direct access to all positions and uses geometric feature engineering (distance-to-goal features, relative cube positions)
- **PPO (10–20%)** achieves non-zero success through BC warm-start + RL exploration, but PPO fine-tuning partially destabilizes the policy
- **VLA methods (0%)** need significantly more data (>1000 episodes) and/or larger models than the teaching-scale setup provides

### BC overfitting in SmolVLA

Training loss decreased 3× (0.10 → 0.03) but closed-loop success remains 0%. This is classic BC overfitting — the model memorizes training trajectories but cannot generalize to new initial conditions. 50 episodes (1788 frames) is insufficient for a 450M parameter VLA to generalize on a contact-rich manipulation task.

### Difficulty gap

This benchmark honestly shows the **difficulty gap** between state-based and vision-based policies:
- State-based methods (State-BC, PPO) can achieve non-zero success with small data
- Vision-based methods (VLA, SmolVLA) require 10–100× more data
- The gap motivates: larger datasets, DAgger, RL fine-tuning, better architectures

---

## 7. Reproduction Commands

```bash
# === Environment ===
cd examples
python unified_pushcube_env.py             # Environment self-test + expert baseline

# === VLA + Language Ablation ===
python unified_pushcube_vla.py             # VLA + 3-condition ablation

# === RL ===
python unified_pushcube_rl.py --algo ppo    # PPO (main RL baseline), 500 episodes
python unified_pushcube_rl.py --algo reinforce  # REINFORCE (concept demo)
python unified_pushcube_rl.py --smoke-test   # CI smoke test

# === World Model ===
python unified_pushcube_wm_mpc.py --planner cem             # CEM planner
python unified_pushcube_wm_mpc.py --planner random_shooting  # Random Shooting

# === Lightweight VLA (CPU, 195K params) — teaching-scale VLA, NOT SmolVLA ===
python train_lightweight_vla.py --epochs 100 --batch_size 64
python evaluate.py --mode closed_loop \
    --checkpoint models/lightweight_vla/lightweight_vla_pushcube.pt \
    --n_episodes 20

# === SmolVLA 450M (requires GPU, ~6GB VRAM) ===
cd robot_foundation_models/smolvla
python finetune.py \
    --dataset_dir /path/to/pushcube_dataset \
    --output_dir /path/to/output \
    --config finetune_config.yaml

# SmolVLA closed-loop evaluation (20 episodes × 3 language modes)
python closed_loop_eval.py \
    --checkpoint /path/to/checkpoint_final \
    --n_episodes 20
```

---

## 8. Evaluation Protocol

| Parameter | Value |
|:----------|:------|
| Shared conditions | Same environment, action space (2-D `ee_delta_2d`), success definition, max steps (80) |
| Evaluation episodes | **Varies by method** (20–100); see Section 2 table |
| SmolVLA / RL eval | 20 episodes × 3 language modes, seeds 3000–3019 |
| Expert / State-BC / VLA | 50–100 episodes (original evaluation) |
| Action-Chunking / Diffusion | **Not yet evaluated** for closed-loop success |
| Language modes | correct / swapped / none (SmolVLA only) |
| Max steps | 80 (all methods, including SmolVLA) |
| Success threshold | Target cube within 0.15 units of goal |
| Selection accuracy | Correct cube closer to goal than wrong cube at episode end |

> **Important:** This is a multi-method teaching experiment, not a strictly controlled leaderboard. Evaluation episode counts, training data, compute, and model scale all vary by method. Source of truth: [`results/benchmarks/benchmark_v2.json`](results/benchmarks/benchmark_v2.json).

---

> **Citation:** If you use this benchmark or results in your work, please link to this repository and reference the SmolVLA experiment configuration in `results/smolvla/`.
