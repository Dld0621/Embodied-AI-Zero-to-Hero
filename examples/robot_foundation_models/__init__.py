"""
Robot Foundation Models (RFM)
=============================
A unified module for integrating robot foundation models (SmolVLA, OpenVLA,
Octo, GR00T, etc.) through a single observation/action interface.

Architecture:
    Language Instruction
        ↓
    Embodied Reasoner (task decomposition)
        ↓
    Robot Foundation Model / VLA (image + lang + state → action chunk)
        ↓
    Embodiment Adapter (generic action → robot-specific action)
        ↓
    Safety Filter (joint limits / collision / velocity)
        ↓
    MuJoCo / Real Robot

All models implement the same `RobotFoundationModel` protocol, so the
external control loop never changes when swapping models — only the
adapter changes.
"""
