# Learning Tracks — Detailed Breakdown

> Detailed pipelines, learning levels, implementation status, and known limitations for each core research track. Extracted from the main README to keep it concise. For the high-level overview, see [README.md](../README.md#core-learning--research-tracks).

All tracks follow the same template:

| Section | Content |
|:--------|:--------|
| **Definition** | One-sentence purpose |
| **Pipeline** | Input → Core Method → Output → Evaluation |
| **Learning Levels** | Concept / Tutorial / Benchmark / Research |
| **Entry Points** | Learn · Run · Evaluate · Explore Papers |
| **Known Limitations** | Honest status of each component |

---

## 1. Vision-Language-Action — Policy Layer

> **Definition:** Generate robot actions from visual perception and natural language instructions. VLA serves as the policy layer that converts high-level human intent into executable robot commands.

**Pipeline:**

```
Multimodal Input (RGB / language / proprioception)
    → Encoding (Vision encoder + Language encoder + State encoder)
    → Fusion (Cross-attention / Token fusion / Unified transformer)
    → Action Representation (joint position / delta pose / action chunk / diffusion trajectory)
    → Training (Behavior Cloning → Pretraining / Fine-tuning)
    → Inference (Observation + Instruction → Policy → Action Chunk → Safety Filter → Controller)
```

**Input / Method / Output / Evaluation:**

| Input | Core Method | Output | Evaluation |
|:------|:------------|:-------|:-----------|
| RGB image, language instruction, proprioception, previous actions | CNN/Transformer encoder, multimodal fusion, policy head (MLP / Diffusion / Transformer) | Action chunk (T steps of joint targets / EE pose) | Task success rate, inference latency, action smoothness, generalization |

**Learning Levels:**

| Level | Content | Status | Entry |
|:------|:--------|:------:|:------|
| Concept | VLA architecture, action chunking, BC vs RL | ✅ | [`docs/01-what-is-vla.md`](01-what-is-vla.md) |
| Tutorial | Minimal VLA structure (random init, concept demo) | ✅ | [`examples/minimal_vla.py`](../examples/minimal_vla.py) |
| Tutorial | Dataset organization: episode, sync, normalization, feature mapping | ✅ | [`docs/21-vla-dataset-organization.md`](21-vla-dataset-organization.md) |
| Tutorial | ACT vs Diffusion Policy comparison with minimal implementations | ✅ | [`docs/22-act-vs-diffusion-policy.md`](22-act-vs-diffusion-policy.md) |
| Runnable | SmolVLA inference with LeRobot, OpenVLA-style loading | 🟡 | [`examples/vla_demo.py`](../examples/vla_demo.py) |
| Runnable | Unified PushCube (dual-cube): VLA + language ablation / Action-Chunking / Diffusion Policy | ✅ | [`unified_pushcube_vla.py`](../examples/unified_pushcube_vla.py) · [`unified_pushcube_act.py`](../examples/unified_pushcube_act.py) · [`unified_pushcube_diffusion.py`](../examples/unified_pushcube_diffusion.py) |
| Benchmark | LIBERO / ALOHA success rate comparison | ⏳ | See [`docs/13-vla-zero-to-one.md`](13-vla-zero-to-one.md) |
| Research | Fine-tuning, cross-embodiment adaptation, real robot | ⏳ | [`docs/02-key-papers.md`](02-key-papers.md) |

**Known Limitations:**
- `minimal_vla.py` is a structural demonstration with random weights, not a pretrained policy.
- `--mode aloha` in `vla_demo.py` requires GPU, network, and the LeRobot dataset; CPU fallback is synthetic only.
- PushCube VLA includes a 3-condition language ablation (full / shuffled / vision-only) to verify language usage. The Action-Chunking Policy omits CVAE (not full ACT). Success rates are teaching-level.
- Real-robot deployment instructions are planned but not yet included.

---

## 2. World Models — Prediction Layer

> **Definition:** Predict future observations and rewards given current state and action, supporting planning, data generation, and safe policy evaluation.

**Pipeline:**

```
Dataset (o_t, a_t, r_t, o_{t+1})
    → Representation Learning (pixel / point cloud / state → latent)
    → Dynamics Learning (p(z_{t+1} | z_t, a_t): deterministic / stochastic / RSSM / Transformer)
    → Prediction Heads (future obs / reward / termination / uncertainty)
    → Imagination (rollout candidate actions, select best)
    → Integration with VLA (action verification), RL (imaginary training), Robot Adapter (action feasibility)
```

**Input / Method / Output / Evaluation:**

| Input | Core Method | Output | Evaluation |
|:------|:------------|:-------|:-----------|
| Observation sequence, action sequence, rewards | Latent dynamics model (linear / RSSM / Transformer / Diffusion) | Predicted next observation, reward, termination, uncertainty | One-step / multi-step prediction error, visual fidelity, planning success |

**Learning Levels:**

| Level | Content | Status | Entry |
|:------|:--------|:------:|:------|
| Concept | Model-based RL, RSSM, DreamerV3, planning | ✅ | [`docs/07-world-models-for-vla.md`](07-world-models-for-vla.md) |
| Tutorial | Minimal linear world model + MPC | ✅ | [`examples/world_model_demo.py`](../examples/world_model_demo.py) |
| Runnable | DreamerV3-style RSSM depth implementation | ✅ | [`examples/dreamer_rssm.py`](../examples/dreamer_rssm.py) |
| Benchmark | Prediction error on standard control tasks | 🟡 | TBD |
| Research | WM + Policy fusion, PointWorld-style 3D flow | ⏳ | [`docs/07-world-models-for-vla.md`](07-world-models-for-vla.md) |

**Known Limitations:**
- RSSM implementation is simplified compared to full DreamerV3; image encoder/decoder is not pixel-accurate.
- Multi-step rollout accumulation error is not yet benchmarked against standard control tasks.

**Implementation Status:**

| Capability | Status |
|:-----------|:------:|
| Observation reconstruction (RSSM decoder) | ✅ |
| Latent transition (GRU + prior/posterior) | ✅ |
| Imagination rollout (prior vs posterior) | ✅ |
| Reward prediction head | ✅ (RSSM + minimal_world_model) |
| Termination prediction head | ✅ (continue_head implemented; meaningful eval requires non-trivial termination labels) |
| Uncertainty calibration | ⏳ |
| Actor–Critic imagination training | ⏳ |

---

## 3. Reinforcement Learning — Optimization Layer

> **Definition:** Optimize policies through environment interaction and reward feedback. RL serves as the fine-tuning and exploration layer that improves upon pretrained policies (VLA or BC) through trial and error.

**Pipeline:**

```
Task Definition (environment, object, goal, success/failure conditions)
    → Observation & Action Space (RGB + proprioception + object state → joint target / torque / EE delta)
    → Reward Design (task + progress + contact + smoothness - collision - energy)
    → Algorithm Selection (Q-Learning / SAC / PPO / HER / Offline RL)
    → Training (reset → rollout → buffer → update → evaluation)
    → Sim-to-Real (domain randomization, latency sim, safety constraints)
```

**Input / Method / Output / Evaluation:**

| Input | Core Method | Output | Evaluation |
|:------|:------------|:-------|:-----------|
| State observation, action space, reward function | Q-Learning, SAC, PPO, HER, Offline RL | Trained policy π(a\|s) | Success rate, sample efficiency, training stability, sim-to-real degradation |

**Learning Levels:**

| Level | Content | Status | Entry |
|:------|:--------|:------:|:------|
| Concept | MDP, value function, policy gradient, Q-Learning | ✅ | [`docs/06-rl-fundamentals-for-vla.md`](06-rl-fundamentals-for-vla.md) |
| Tutorial | Pure-numpy Q-Learning demo | ✅ | [`examples/rl_demo.py --mode demo`](../examples/rl_demo.py) |
| Runnable | PPO on PushCube (PyTorch, main baseline) + REINFORCE (concept demo) | ✅ | [`examples/unified_pushcube_rl.py`](../examples/unified_pushcube_rl.py) |
| Benchmark | Success rate vs sample count on standard tasks | 🟡 | TBD |
| Research | RL fine-tuning of VLA policies, real-robot RL | ⏳ | [`docs/14-rl-zero-to-one.md`](14-rl-zero-to-one.md) |

**Known Limitations:**
- SAC+HER training requires significant compute; CPU training is possible but slow.
- Real-robot RL safety constraints and sim-to-real transfer are documented but not yet implemented end-to-end.

---

## 4. Embodied Reasoning — Planning Layer

> **Definition:** Decompose long-horizon tasks into executable sub-goals and coordinate policy execution with failure recovery.

**Pipeline:**

```
Language instruction → Task decomposition → Subgoal sequence → VLA execution → Failure detection & recovery
```

**Current Status:**

| Component | Status | Entry |
|:----------|:-------|:------|
| Rule-based planner | ✅ Runnable | `examples/robot_foundation_models/planners/rule_based_planner.py` |
| VLM-based planner | 🟡 Mock | `examples/robot_foundation_models/planners/` |
| Failure recovery | ⏳ Planned | — |

Documentation: [`docs/27-embodied-reasoning-and-planning.md`](27-embodied-reasoning-and-planning.md)
