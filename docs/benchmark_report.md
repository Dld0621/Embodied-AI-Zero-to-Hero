# PushCube Benchmark Report

> **Paper-style experiment report.** For the full leaderboard and reproduction commands, see [`BENCHMARK.md`](../BENCHMARK.md). For the main README summary, see [README.md](../README.md#benchmarks).

---

## Abstract

We evaluate 10 methods—ranging from heuristic expert policies to 450M-parameter vision-language-action models—on a unified dual-cube PushCube task. The task requires an agent to push the correct colored cube (identified by a language instruction) into a target zone. At teaching scale (50–500 episodes, CPU/GPU), only state-based methods achieve non-zero success rates. Vision-based VLA methods, including SmolVLA fine-tuned for 10K steps on GPU, achieve 0% closed-loop success, highlighting the difficulty gap between state-based and vision-based manipulation at limited data budgets.

---

## 1. Experiment Setup

### 1.1 Environment

| Parameter | Value |
|:----------|:------|
| Environment | PushCube (dual-cube, language-conditioned) |
| Workspace | 2D table, 1.0 × 1.0 |
| Objects | 2 cubes (red, green), 0.08 × 0.08 |
| Agent | Point pusher, velocity-controlled |
| Max steps | 80 (all methods, including SmolVLA) |

### 1.2 Observation & Action Spaces

| Space | Dimension | Description |
|:------|:---------:|:------------|
| **State** | 14-D | arm (x,y), cube1 (x,y), cube2 (x,y), goal (x,y), cube colors (r,g × 2), goal-color one-hot (2) |
| **Observation (VLA)** | 128×128 RGB | Top-down render of workspace |
| **Language** | string | "push the {red\|green} cube to the {direction}" |
| **Action** | 2-D | `[dx, dy]` — end-effector delta, bounded by arm speed (0.08/step) |

### 1.3 Metrics

| Metric | Definition |
|:-------|:-----------|
| **Success Rate ↑** | Fraction of episodes where the target cube enters the goal zone (within 0.15 units) |
| **Selection Accuracy** | Fraction of episodes where the correct cube ends up closer to the goal than the wrong cube |
| **Inference Latency** | Time per action prediction (measured for SmolVLA) |

### 1.4 Evaluation Protocol

| Parameter | Value |
|:----------|:------|
| Shared conditions | Same environment, action space (2-D `ee_delta_2d`), success definition, seed family (3000–3019) |
| Evaluation episodes | **Varies by method** — see per-method data in Section 2.1 |
| SmolVLA / RL eval | 20 episodes × 3 language modes, seeds 3000–3019 |
| Expert / State-BC / VLA | 50–100 episodes (original evaluation, seeds not strictly aligned) |
| Action-Chunking / Diffusion | **Not yet evaluated** for closed-loop success |
| Language modes | correct / swapped / none (SmolVLA only) |
| Success threshold | Target cube within 0.15 units of goal center |

> **Important:** This is NOT a strictly controlled leaderboard. Evaluation episode counts differ by method (20–100). Training data, compute, and model scale also vary. Results should be interpreted as a multi-method teaching experiment, not a controlled comparison.

---

## 2. Main Results

### 2.1 Unified Leaderboard

| Method | Input | Paradigm | Data | Compute | Params | Eval Eps | Success ↑ | Selection |
|:-------|:------|:---------|:-----|:--------|-------:|---:|:---:|:---:|
| Expert | State | Heuristic | — | CPU | — | 50 | **~100%** | — |
| State-BC | 14-D state | Regression | 100 eps / 3.6K frames | CPU | ~10K | 100 | **90%** | — |
| RL (BC-init PPO) | 14-D state | Policy Gradient | 500 eps (on-policy) | CPU | ~10K | 20 | **15%** | — |
| VLA (Full) | RGB + lang | Regression | 100 eps / 3.6K frames | CPU | 195K | 100 | **0%** | 45% |
| Action-Chunking | RGB + lang | Regression | 100 eps / 3.6K frames | CPU | ~500K | — | **N/A** | — |
| Diffusion Policy | RGB + lang | Diffusion | 100 eps / 3.6K frames | CPU | ~1M | — | **N/A** | — |
| WM-MPC (CEM) | 14-D state | Planning | 100 eps WM training | CPU | ~50K | 20 | **0%** | — |
| WM-MPC (Random) | 14-D state | Planning | 100 eps WM training | CPU | ~50K | 20 | **0%** | — |
| SmolVLA (500 step) | RGB + lang + state | Flow Matching | 50 eps / 1788 frames | GPU (RTX 3060) | 450M | 20 | **0%** | 50% |
| SmolVLA (10K step) | RGB + lang + state | Flow Matching | 50 eps / 1788 frames | GPU (RTX 3060) | 450M | 20 | **0%** | 50% |

> **Note:** "N/A" means the method was trained but NOT yet evaluated for closed-loop success. Previous reports showing "0%" for Action-Chunking and Diffusion were incorrect — their `success_rate` is `null` in the raw data (`pushcube_summary.json`). This is a multi-method teaching experiment, not a strictly fair leaderboard. Source of truth: [`results/benchmarks/benchmark_v2.json`](../results/benchmarks/benchmark_v2.json).

### 2.2 Policy Generation Paradigm Comparison

| Paradigm | Representative | Success | Key Observation |
|:---------|:---------------|:---:|:----------------|
| Heuristic | Expert | ~100% | Perfect state access + hand-crafted strategy |
| Regression (state) | State-BC | 90% | Geometric features enable learning with small data |
| Policy Gradient | PPO | 15% | BC warm-start helps, but RL destabilizes policy |
| Regression (vision) | VLA | 0% | Vision-to-action mapping needs 10× more data |
| Regression (vision) | Action-Chunking | N/A | Trained but not yet evaluated for closed-loop success |
| Diffusion | Diffusion Policy | N/A | Trained but not yet evaluated for closed-loop success |
| Flow Matching | SmolVLA | 0% | 450M model + 50 eps = severe overfitting |
| Planning (WM-MPC) | CEM, Random | 0% | Model prediction error compounds over horizon |

---

## 3. SmolVLA Experiment

### 3.1 Training Configuration

| Parameter | 500-step | 10K-step |
|:----------|:---------|:---------|
| Model | SmolVLA 450M (`lerobot/smolvla_base`) | Same |
| Total params | 450,046,176 | Same |
| Trainable params | 99,880,992 (LoRA-style) | Same |
| Dataset | PushCube, 50 episodes / 1788 frames | Same |
| Action dim | 2 (`ee_delta_2d`) | Same |
| Steps | 500 | 10,000 (resumed) |
| Batch size | 2 | 2 |
| Precision | bf16 | bf16 |
| Optimizer | AdamW | AdamW |
| LR schedule | Fixed | Cosine 1e-4 → 2.5e-6 |
| Hardware | NVIDIA RTX 3060 Laptop (6.4 GB) | Same |
| Training time | ~15 min | 65.1 min total |

### 3.2 Training Results

| Metric | 500-step | 10K-step |
|:-------|:---------|:---------|
| Initial loss | 0.47 | 0.10 (resumed) |
| Final loss | 0.103 | 0.031 |
| Average loss | 0.069 | 0.030 |
| Best loss | 0.028 | 0.004 |

### 3.3 Language Ablation

| Mode | 500-step Success | 500-step Selection | 10K-step Success | 10K-step Selection |
|:-----|:---:|:---:|:---:|:---:|
| Correct language | 0% | 50% | 0% | 50% |
| Swapped language | 0% | 50% | 0% | 45% |
| No language | 0% | 50% | 0% | 50% |

**Key finding:** Selection accuracy stays at ~50% (chance level) across all modes and training scales, indicating the model has not learned to use language to identify the target cube.

### 3.4 Checkpoint Artifacts

| File | 500-step | 10K-step | Content |
|:-----|:---:|:---:|:--------|
| `training_config.json` | ✅ | ✅ | Step, loss, param count |
| `training_history.json` | ✅ | ✅ | Per-step loss (9500 entries for 10K) |
| `eval_results.json` | ✅ | ✅ | 20 episodes × 3 language modes |
| `checkpoint_info.json` | ✅ | ✅ | Checkpoint structure |
| `summary.md` | ✅ | ✅ | Human-readable summary |

Location: `results/smolvla/500_steps/` and `results/smolvla/10k_steps/`

---

## 4. Failure Analysis

### 4.1 Why State-BC succeeds (90%) but VLA fails (0%)

| Factor | State-BC | VLA |
|:-------|:---------|:----|
| **Input dimensionality** | 14-D (compact) | 128×128×3 = 49K (high) |
| **Feature engineering** | Distance-to-goal, relative cube positions | Must learn features from pixels |
| **Data efficiency** | 100 episodes sufficient | ~1000+ episodes needed (estimated) |
| **Contact dynamics** | State directly encodes positions | Must infer positions from pixels |

### 4.2 Why SmolVLA (450M) fails despite 10K training steps

1. **Data scale insufficient:** 50 episodes (1788 frames) is ~100× too small for a 450M parameter model. Industry VLA training typically uses 10K+ episodes.
2. **BC overfitting:** Training loss decreased 3× (0.10 → 0.03) but closed-loop success remains 0%. The model memorizes training trajectories but cannot generalize to new initial conditions.
3. **Compounding error:** Small action errors accumulate over the 100-step horizon. Without DAgger or interactive data collection, the policy drifts off-distribution.
4. **No recovery behavior:** BC-trained policies have no mechanism to recover from unfamiliar states. RL post-training (e.g., PPO fine-tuning) could address this.
5. **Contact-rich manipulation:** Pushing requires precise spatial reasoning about contact points — a skill that demands either extensive data or strong inductive biases.

### 4.3 Why PPO partially works (10–20%)

PPO benefits from BC warm-start (40% expert pre-training) and on-policy exploration. However:
- BC pre-training provides a good initialization, but PPO exploration partially destabilizes the policy
- 500 episodes of on-policy data is insufficient for stable convergence
- The sparse reward signal (success only when cube reaches goal) provides weak gradients

### 4.4 Why WM-MPC fails (0%)

The world model reaches a best validation loss of 0.0409, but errors compound over the planning horizon:
- H=1: 0.0708, H=5: 0.2961, H=10: 0.5560
- At H=10, the prediction is too noisy for effective planning
- CEM with 500 samples × 3 iterations cannot find good action sequences under this noise

Source: [`results/benchmarks/wm_results.json`](../results/benchmarks/wm_results.json), mirrored in [`benchmark_v2.json`](../results/benchmarks/benchmark_v2.json).

---

## 5. Discussion

### 5.1 The State-Vision Difficulty Gap

This benchmark honestly demonstrates the **difficulty gap** between state-based and vision-based policies:

```
State-based: 10K params + 100 episodes → 90% success
Vision-based: 450M params + 50 episodes → 0% success
```

The gap motivates:
- **Larger datasets** (>1000 episodes for VLA)
- **DAgger / interactive data collection** (to address compounding error)
- **RL post-training** (to enable recovery and exploration)
- **Better architectures** (3D representations, cross-embodiment pretraining)
- **Sim-to-real transfer** (leveraging unlimited simulation data)

### 5.2 Teaching-Scale Limitations

All results are at **teaching scale** — intentionally limited to illustrate algorithmic differences, not to achieve production performance. The comparison is fair in evaluation protocol (same environment, seeds, metrics) but not in training budget (data, compute, model size vary by method).

### 5.3 What This Benchmark Does NOT Show

- It does not show that VLA methods are inherently worse than state-based methods
- It does not show the upper bound of any method's performance
- It does not control for hyperparameter tuning effort across methods

---

## 6. Reproduction

```bash
# Environment self-test + expert baseline
cd examples && python unified_pushcube_env.py

# VLA + 3-condition language ablation
python unified_pushcube_vla.py

# RL (BC-initialized PPO)
python unified_pushcube_rl.py --algo ppo

# World Model + MPC
python unified_pushcube_wm_mpc.py --planner cem

# --- Lightweight VLA (CPU, 195K params) ---
# This is the teaching-scale VLA, NOT SmolVLA 450M
cd examples
python train_lightweight_vla.py --epochs 100 --batch_size 64
python evaluate.py --mode closed_loop \
    --checkpoint models/lightweight_vla/lightweight_vla_pushcube.pt \
    --n_episodes 20

# --- SmolVLA 450M (requires GPU, ~6GB VRAM) ---
cd examples/robot_foundation_models/smolvla
python finetune.py \
    --dataset_dir /path/to/pushcube_dataset \
    --output_dir /path/to/output \
    --config finetune_config.yaml

# SmolVLA closed-loop evaluation (20 episodes × 3 language modes)
python closed_loop_eval.py \
    --checkpoint /path/to/checkpoint_final \
    --n_episodes 20
```

See [`BENCHMARK.md`](../BENCHMARK.md) for full reproduction commands and [`docs/28-smolvla-gpu-finetuning-runbook.md`](28-smolvla-gpu-finetuning-runbook.md) for GPU fine-tuning guide.

---

## 7. Citation

If you use this benchmark or results in your work, please link to this repository and reference the SmolVLA experiment configuration in `results/smolvla/`.

```bibtex
@misc{embodied-ai-zero-to-hero,
  author = {Gangwei Li},
  title = {Embodied AI: Zero to Hero — A Reproducible Learning and Experimentation Platform},
  year = {2026},
  url = {https://github.com/Dld0621/Embodied-AI-Zero-to-Hero}
}
```
