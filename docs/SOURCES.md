# Primary-Source Registry / 权威来源

This registry anchors the foundations, knowledge graph, and environment layers to primary papers, official documentation, or author-maintained textbooks. It was reviewed on **2026-08-22**. External APIs and software versions can change; the repository lock files and recorded experiment environment remain the source of truth for reproduced runs.

## 01 Python and numerical computing

- [Python Tutorial](https://docs.python.org/3/tutorial/) — Python Software Foundation.
- [NumPy user guide](https://numpy.org/doc/stable/user/index.html) — NumPy project.

## 02 Linear algebra

- [Introduction to Applied Linear Algebra](https://web.stanford.edu/~boyd/vmls/) — Stephen Boyd and Lieven Vandenberghe.
- [NumPy linear algebra reference](https://numpy.org/doc/stable/reference/routines.linalg.html) — NumPy project.

## 03 Deep learning

- [PyTorch: Learn the Basics](https://docs.pytorch.org/tutorials/beginner/basics/intro.html) — PyTorch project.
- [Deep Learning](https://www.deeplearningbook.org/) — Goodfellow, Bengio, and Courville.

## 04 Transformers

- [Attention Is All You Need](https://arxiv.org/abs/1706.03762) — Vaswani et al.
- [PyTorch Transformer documentation](https://docs.pytorch.org/docs/stable/nn.html#transformer-layers) — PyTorch project.

## 05 Coordinate transforms

- [Modern Robotics, rigid-body motions](https://modernrobotics.northwestern.edu/nu-gm-book-resource/) — Lynch and Park, Chapters 2–3.

## 06 SO3 and SE3

- [Modern Robotics, Chapter 3](https://modernrobotics.northwestern.edu/nu-gm-book-resource/) — rotation matrices, exponential coordinates, and homogeneous transforms.
- [SciPy Rotation reference](https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.transform.Rotation.html) — SciPy project.

## 07 Kinematics, Jacobians, and IK

- [Modern Robotics, Chapters 4–6](https://modernrobotics.northwestern.edu/nu-gm-book-resource/) — forward kinematics, Jacobians, and numerical IK.

## 08 Control

- [Underactuated Robotics](https://underactuated.csail.mit.edu/) — Russ Tedrake, MIT.
- [Modern Robotics, Chapter 11](https://modernrobotics.northwestern.edu/nu-gm-book-resource/) — robot motion and force control.

## 09 MuJoCo

- [MuJoCo documentation](https://mujoco.readthedocs.io/en/stable/) — MuJoCo project.
- [MuJoCo Menagerie](https://github.com/google-deepmind/mujoco_menagerie) — model collection and per-model licenses.

## 10 Datasets and training

- [LeRobot documentation](https://huggingface.co/docs/lerobot/index) — Hugging Face.
- [RLDS specification and tools](https://github.com/google-research/rlds) — Google Research.

## 11 Probability and optimization

- [Convex Optimization](https://web.stanford.edu/~boyd/cvxbook/) — Boyd and Vandenberghe.
- [SciPy optimization reference](https://docs.scipy.org/doc/scipy/reference/optimize.html) — SciPy project.

## 12 Perception and sensors

- [OpenCV camera calibration and 3D reconstruction](https://docs.opencv.org/4.x/d9/d0c/group__calib3d.html) — OpenCV project.
- [ROS REP 103: Standard Units of Measure and Coordinate Conventions](https://www.ros.org/reps/rep-0103.html) — ROS community standard.

## 13 Robot systems and safety

- [ROS 2 documentation](https://docs.ros.org/en/rolling/) — Open Robotics.
- [ROS 2 Quality of Service settings](https://docs.ros.org/en/rolling/Concepts/Intermediate/About-Quality-of-Service-Settings.html) — Open Robotics.

These software references do not certify a physical robot. Applicable laws, standards, robot-manufacturer limits, site rules, and a qualified safety review remain mandatory for hardware use.

## 14 Evaluation and reproducibility

- [ACM Artifact Review and Badging](https://www.acm.org/publications/policies/artifact-review-and-badging-current) — Association for Computing Machinery.
- [DoF benchmark source of truth](../results/benchmarks/benchmark_v2.json) — machine-readable repository evidence.

## 15 Multimodal perception and state estimation

- [OpenCV Camera Calibration](https://docs.opencv.org/5.0/py_tutorials/py_calib3d/py_calibration/py_calibration.html) — intrinsics, distortion, extrinsics, and undistortion.
- [ROS 2 message_filters](https://docs.ros.org/en/ros2_packages/rolling/api/message_filters/message_filters.html) — timestamp-based exact and approximate synchronization.
- [robot_localization](https://docs.ros.org/en/noetic/api/robot_localization/html/index.html) — EKF/UKF sensor fusion and state-estimation interfaces.

## 16 Navigation and locomotion

- [Nav2 Concepts](https://docs.nav2.org/concepts/) — behavior trees, planners, controllers, smoothers, routes, and recovery servers.
- [ROS REP 105: Coordinate Frames for Mobile Platforms](https://www.ros.org/reps/rep-0105.html) — `map`, `odom`, and `base_link` frame semantics.
- [Isaac Lab environments](https://isaac-sim.github.io/IsaacLab/develop/source/overview/environments.html) — maintained navigation and legged-locomotion environment registry.

## 17 Robot development environment

- [ROS 2 Jazzy Ubuntu packages](https://docs.ros.org/en/jazzy/Installation/Ubuntu-Install-Debs.html) and [ROS 2 Humble Ubuntu packages](https://docs.ros.org/en/humble/Installation/Ubuntu-Install-Debs.html) — official ROS binary installation paths.
- [Installing Gazebo with ROS](https://gazebosim.org/docs/jetty/ros_installation/) and [ROS 2 integration](https://gazebosim.org/docs/harmonic/ros2_integration/) — default pairings and `ros_gz` boundaries.
- [MuJoCo Python](https://mujoco.readthedocs.io/en/stable/python.html) and [MuJoCo visualization](https://mujoco.readthedocs.io/en/stable/programming/visualization.html) — maintained bindings, viewer, and rendering contexts.
- [Isaac Lab local installation](https://isaac-sim.github.io/IsaacLab/main/source/setup/installation/) — current supported host, Python, Isaac Sim, driver, and installation routes.
- [Genesis World installation](https://genesis-world.readthedocs.io/en/latest/user_guide/overview/installation.html) and [minimal scene](https://genesis-world.readthedocs.io/en/latest/user_guide/getting_started/hello_genesis.html) — current package and execution sequence.
- [Python `venv`](https://docs.python.org/3/library/venv.html), [pip repeatable installs](https://pip.pypa.io/en/stable/topics/repeatable-installs/), and [PyTorch local selector](https://pytorch.org/get-started/locally/) — isolation and framework-wheel selection.
- [CUDA Linux installation](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/), [CUDA on WSL](https://docs.nvidia.com/cuda/wsl-user-guide/), and [Microsoft WSL installation](https://learn.microsoft.com/en-us/windows/wsl/install) — driver, toolkit, and WSL boundaries.

## 18 Rigid-body dynamics and contact

- [Modern Robotics, dynamics chapters](https://modernrobotics.northwestern.edu/nu-gm-book-resource/) — rigid-body dynamics, inverse dynamics, and trajectory generation.
- [MuJoCo computation](https://mujoco.readthedocs.io/en/stable/computation/) — equations of motion, actuation, constraints, contact frames, friction, and solver semantics.
- [MuJoCo modeling](https://mujoco.readthedocs.io/en/stable/modeling.html) — model parameters, contact parameter combination, and modeling guidance.

## 19 State estimation and motion planning

- [Underactuated Robotics: State Estimation](https://underactuated.mit.edu/state_estimation.html) — observers, Kalman filtering, recursive Bayesian filtering, and smoothing roadmap.
- [Underactuated Robotics: Sampling-based Motion Planning](https://underactuated.mit.edu/planning.html) — graph search, PRM, RRT, feasibility, and dynamic constraints.
- [Nav2 concepts](https://docs.nav2.org/concepts/index.html) — maintained planner, controller, behavior-tree, recovery, and navigation-system boundaries.

## 20 Imitation learning, VLA, and cross-embodiment

- [ACT: Learning Fine-Grained Bimanual Manipulation with Low-Cost Hardware](https://arxiv.org/abs/2304.13705) — original action-chunking Transformer paper and evaluation scope.
- [Diffusion Policy project and paper](https://diffusion-policy.cs.columbia.edu/) — action diffusion, receding-horizon execution, benchmarks, code, and data from the authors.
- [Open X-Embodiment / RT-X](https://robotics-transformer-x.github.io/) — original dataset-mixture, action convention, model, and cross-robot evaluation report.
- [OpenVLA](https://openvla.github.io/) — original project page with paper, code, checkpoints, architecture, adaptation protocol, and reported evaluations.
- [Octo](https://arxiv.org/abs/2405.12213) — original open generalist policy paper and reported cross-embodiment scope.
- [π0](https://arxiv.org/abs/2410.24164) — original flow-based vision-language-action model report.
- [OpenVLA-OFT](https://arxiv.org/abs/2502.19645) — original report on continuous action representation, parallel decoding, action chunking, and fine-tuning.
- [SmolVLA](https://arxiv.org/abs/2506.01844) and [official LeRobot guide](https://huggingface.co/docs/lerobot/smolvla) — original model report and maintained usage boundary.
- [LeRobot documentation](https://huggingface.co/docs/lerobot/index) — maintained policy, dataset, simulation, evaluation, and hardware interfaces; pin a release or commit for reproduction.

## 21 Reinforcement learning and post-training

- [Reinforcement Learning: An Introduction](http://incompleteideas.net/book/the-book-2nd.html) — Sutton and Barto's author-maintained textbook.
- [Proximal Policy Optimization Algorithms](https://arxiv.org/abs/1707.06347) — original PPO objective and experiments.

## 22 World models and predictive control

- [Learning Latent Dynamics for Planning from Pixels](https://arxiv.org/abs/1811.04551) — original PlaNet latent-dynamics planning paper.
- [Mastering Diverse Domains through World Models](https://arxiv.org/abs/2301.04104) — original DreamerV3 paper and reported evaluation scope.
- [TD-MPC2: Scalable, Robust World Models for Continuous Control](https://arxiv.org/abs/2310.16828) — original TD-MPC2 paper on learned models and planning.
- [V-JEPA 2](https://arxiv.org/abs/2506.09985) — original latent video representation and action-conditioned planning report; the action-conditioned planner is not labeled a WAM in this repository.
- [WorldVLA](https://arxiv.org/abs/2506.21539) — original autoregressive action-world model preprint unifying image and action understanding/generation.
- [DreamZero](https://arxiv.org/abs/2602.15922) — 2026 preprint on a video-diffusion-based World Action Model jointly predicting future video and actions.
- [Action Images](https://arxiv.org/abs/2604.06168) — emerging 2026 preprint on action-grounded video generation; treat publication status and conclusions as preprint evidence.

## 23 Manipulation, dexterity, and locomotion systems

- [Modern Robotics](https://modernrobotics.northwestern.edu/nu-gm-book-resource/) — kinematics, dynamics, motion planning, control, and manipulation foundations.
- [Underactuated Robotics](https://underactuated.mit.edu/) — dynamics, trajectory optimization, state estimation, control, and locomotion notes.
- [MuJoCo contact computation](https://mujoco.readthedocs.io/en/stable/computation/) — point-contact frames, friction dimensions, margins, gaps, and constraint-force semantics used by the simulation guides.

## 24 Sim-to-Real and deployment evidence

- [Sim-to-Real Transfer of Robotic Control with Dynamics Randomization](https://arxiv.org/abs/1710.06537) — original dynamics-randomization study and its stated robot-task scope.
- [Isaac Lab manager-based environment tutorial](https://isaac-sim.github.io/IsaacLab/main/source/tutorials/03_envs/create_manager_base_env.html) — maintained `EventManager` interface used to schedule startup, reset, interval, and other environment events, including randomization; verify the installed Isaac Lab version before use.
- [DoF validation policy](VALIDATION.md) — repository-local evidence ladder separating synthetic input, simulation, benchmark, HIL/shadow, and hardware claims.

## Citation policy

- Prefer the original paper or official project documentation over secondary summaries.
- Record the accessed version/date when an external interface can drift.
- Treat a URL check as availability evidence only; it does not prove that every local interpretation is semantically correct.
- Open a correction issue when a source and repository statement disagree.
