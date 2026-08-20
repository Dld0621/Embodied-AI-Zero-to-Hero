# Primary-Source Registry / 权威来源

This registry anchors the foundations and environment layers to primary papers, official documentation, or author-maintained textbooks. It was reviewed on **2026-08-20**. External APIs and software versions can change; the repository lock files and recorded experiment environment remain the source of truth for reproduced runs.

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

## Citation policy

- Prefer the original paper or official project documentation over secondary summaries.
- Record the accessed version/date when an external interface can drift.
- Treat a URL check as availability evidence only; it does not prove that every local interpretation is semantically correct.
- Open a correction issue when a source and repository statement disagree.
