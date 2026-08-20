# Third-Party Notices

The repository-level [MIT License](LICENSE) applies to original DoF content. Files under `pretrained/` retain their upstream licenses; the repository MIT license does not replace those terms.

| Component | Tracked location | Upstream | License evidence |
|---|---|---|---|
| FrankMocap | `pretrained/anyteleop/frankmocap/` | [facebookresearch/frankmocap](https://github.com/facebookresearch/frankmocap) | The bundled [`LICENSE`](pretrained/anyteleop/frankmocap/LICENSE) identifies Attribution-NonCommercial 4.0 terms. Review it before reuse. |
| MuJoCo Menagerie snapshot | `pretrained/urdf/mujoco_menagerie/` | [google-deepmind/mujoco_menagerie](https://github.com/google-deepmind/mujoco_menagerie) | Menagerie is a collection with per-model terms. Consult its bundled [`LICENSE`](pretrained/urdf/mujoco_menagerie/LICENSE) and model-specific license files. |
| Franka FR3 model | `pretrained/urdf/mujoco_menagerie/franka_fr3/` | MuJoCo Menagerie | [Apache License 2.0](pretrained/urdf/mujoco_menagerie/franka_fr3/LICENSE) |
| Shadow Hand model | `pretrained/urdf/mujoco_menagerie/shadow_hand/` | MuJoCo Menagerie | [Apache License 2.0](pretrained/urdf/mujoco_menagerie/shadow_hand/LICENSE) |
| LEAP Hand simulation assets | `pretrained/urdf/leap_hand_sim/` | [leap-hand-sim](https://github.com/leap-hand/LEAP_Hand_Sim) | [MIT-style license](pretrained/urdf/leap_hand_sim/LICENSE.txt) |
| ORCA Hand description | `pretrained/urdf/orcahand_description/` | [ORCA Hand](https://github.com/orcahand) | [MIT License](pretrained/urdf/orcahand_description/LICENSE) |

Large model weights referenced by [`pretrained/README.md`](pretrained/README.md) are intentionally not tracked. Their upstream terms apply separately, including registration and non-commercial restrictions for MANO/SMPL-X assets.

## First-party repository consolidation

The former `Dld0621/RobotDev-Setup-Guide` was first-party MIT-licensed material, copyright 2025 Dld0621. It was reviewed and consolidated into [`docs/setup/`](docs/setup/); see the [migration and correction record](docs/setup/MIGRATION.md). This is provenance disclosure rather than a third-party license exception.
