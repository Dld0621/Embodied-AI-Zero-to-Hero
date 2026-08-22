# Foundations Layer · English Overview

This page gives English-speaking learners a route through the 14 prerequisite documents. The [45-node Knowledge System](../knowledge-system/README.md) is the more precise prerequisite graph: it connects individual concepts to assessments, documents, and Pipelines. The linked foundation lessons currently use Chinese-first explanations, equations, runnable code, and exercises; this overview and the knowledge graph make their contracts explicit in English.

## Route

```text
Python → Linear Algebra → Deep Learning → Transformers
   └──→ Frames → SO(3)/SE(3) → FK/Jacobian/IK → Control → MuJoCo
                         └──→ Data → Probability/Optimization
                                  → Perception → Systems/Safety → Evaluation
```

| # | Lesson | What you must learn | Runnable evidence | Exit criterion |
|---:|---|---|---|---|
| 01 | [Python for robotics](01-python-for-robotics.md) | Arrays, shapes, functions, data flow, debugging | NumPy robotics examples | Explain and verify every tensor shape |
| 02 | [Linear algebra](02-linear-algebra.md) | Vectors, matrices, norms, projections, SVD | Geometry examples | Derive and test a least-squares solution |
| 03 | [Deep learning](03-deep-learning-basics.md) | Forward pass, loss, autograd, optimization, overfitting | Small PyTorch MLP | Train, validate, and diagnose a curve |
| 04 | [Transformers](04-transformer-basics.md) | Tokens, attention, masks, sequence modeling | Minimal attention example | Trace shapes through one attention block |
| 05 | [Coordinate transforms](05-coordinate-transform.md) | Frames, homogeneous transforms, composition order | Transform-chain example | State every frame and verify round trips |
| 06 | [SO(3) and SE(3)](06-se3-and-rotation.md) | Rotation representations, Lie groups, interpolation | 3D finger-chain example | Convert representations without singularity mistakes |
| 07 | [FK, Jacobian, and IK](07-fk-jacobian-ik.md) | Forward kinematics, Jacobians, numerical IK, constraints | FK/IK and retargeting examples | Reach targets while respecting limits |
| 08 | [Control basics](08-control-basics.md) | Feedback, PID, trajectories, stability, saturation | Safety-filter and retargeting examples | Tune a bounded loop and explain failure modes |
| 09 | [MuJoCo basics](09-mujoco-basics.md) · [scene building](../tutorials/mujoco-scene-building.md) | Models, modular MJCF, state, stepping, contacts, actuators, sensors, rendering, export | Runnable modular workcell and model round-trip tests | Build, inspect, step, visualize, and export a scene with explicit evidence boundaries |
| 10 | [Dataset and training](10-dataset-and-training.md) | Episode schema, synchronization, splits, normalization | Canonical dataset and LeRobot adapters | Detect leakage and reproduce a split |
| 11 | [Probability and optimization](11-probability-and-optimization.md) | Random variables, estimation, gradients, constrained objectives | RSSM and optimization examples | Define objective, constraints, and uncertainty |
| 12 | [Perception and sensors](12-perception-and-sensors.md) | Camera geometry, timing, calibration, multimodal observations | Observation-schema example | Pass calibration and synchronization gates |
| 13 | [Robot systems and safety](13-robot-systems-and-safety.md) | Middleware, rates, watchdogs, state machines, logging | Model interface and safety filter | Demonstrate bounded failure behavior |
| 14 | [Evaluation and reproducibility](14-evaluation-and-reproducibility.md) | Protocols, seeds, confidence, ablations, provenance | Benchmark runner | Reproduce a result and state its evidence level |

## How to use the lessons

1. Read the English contract above and the linked lesson.
2. Run the referenced project file rather than copying an isolated snippet.
3. Complete the “检查理解” questions.
4. Record the command, environment, seed, artifact, metric, and failure case.
5. Continue only when the lesson exit criterion is satisfied.

For a target-specific order, resolve the graph rather than assuming the table is a single linear curriculum:

```bash
python scripts/run_knowledge_map.py --path-to learning-vla
python scripts/run_knowledge_map.py --path-to task-navigation
```

Authoritative references for every lesson are maintained in the [primary-source registry](../SOURCES.md). Repository claims and experiments follow the [validation policy](../VALIDATION.md).
