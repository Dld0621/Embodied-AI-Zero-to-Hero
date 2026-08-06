# Pretrained Models and Robot Assets

This directory contains two different kinds of dependencies:

1. **tracked third-party source and robot assets** used by examples;
2. **download recipes for large or registration-gated model files** that are intentionally not stored in Git.

> Large files matching `*.tar.gz`, `*.pkl`, and `*.pth` are excluded by [`pretrained/.gitignore`](.gitignore). A path shown in the download plan is not evidence that the weight currently exists locally.

## Current repository snapshot

| Path | Tracked now | Purpose |
|---|:---:|---|
| `anyteleop/frankmocap/` | Yes | Upstream FrankMocap source snapshot; model weights are excluded |
| `urdf/mujoco_menagerie/` | Yes | Robot MJCF/mesh assets with upstream and model-specific licenses |
| `urdf/leap_hand_sim/` | Yes | LEAP Hand simulation assets |
| `urdf/orcahand_description/` | Yes | ORCA Hand description assets |
| `hamer/` | No by default | Created by `download_models.py` for HaMeR/ViTDet weights |
| `mano/` | No by default | User-created directory for registration-gated MANO files |

Run this check at any time:

```bash
python -c "from pathlib import Path; print([str(p) for p in Path('pretrained').rglob('*') if p.is_file()])"
```

## Automated downloads

From the repository root:

```bash
python pretrained/download_models.py
```

The helper downloads these upstream artifacts:

| Destination | Approximate size in the helper | Source |
|---|---:|---|
| `hamer/hamer_demo_data.tar.gz` | 1.7 GB | UT Austin HaMeR distribution |
| `hamer/model_final_f05665.pkl` | 2.6 GB | Detectron2 ViTDet model zoo |
| `anyteleop/frankmocap/.../SMPLX_HAND_INFO.pkl` | 0.1 MB | FrankMocap/EFT distribution |
| `anyteleop/frankmocap/.../mean_mano_params.pkl` | <0.1 MB | FrankMocap/EFT distribution |
| `anyteleop/frankmocap/.../pose_shape_best.pth` | 102 MB | FrankMocap/EFT distribution |

Sizes are operational estimates, not cryptographic identities. The helper supports resuming downloads but does **not** currently pin SHA-256 checksums. Before research or deployment, verify the upstream origin, final size, license, and integrity according to your organization’s policy.

## Registration-gated files

These assets cannot be redistributed or downloaded automatically:

| File | Official access | Terms | Expected destination |
|---|---|---|---|
| `MANO_RIGHT.pkl` | [MANO](https://mano.is.tue.mpg.de/) | Registration and license acceptance; non-commercial research restrictions apply | `pretrained/mano/MANO_RIGHT.pkl` |
| `SMPLX_NEUTRAL.pkl` | [SMPL-X](https://smpl-x.is.tue.mpg.de/) | Registration and license acceptance; non-commercial research restrictions apply | `pretrained/anyteleop/frankmocap/extra_data/smpl/SMPLX_NEUTRAL.pkl` |

Do not commit these files or redistribute them through project releases.

## Example placement

```bash
# HaMeR data after download
mkdir -p /path/to/hamer/_DATA
tar -xzf pretrained/hamer/hamer_demo_data.tar.gz -C /path/to/hamer/_DATA/

# Robot assets
cp -r pretrained/urdf/mujoco_menagerie/shadow_hand /your/project/assets/
cp -r pretrained/urdf/mujoco_menagerie/franka_fr3 /your/project/assets/
```

## Licensing and provenance

The repository-level MIT license does not replace third-party terms. In particular:

- the bundled FrankMocap license identifies Attribution-NonCommercial 4.0 terms, not MIT;
- MuJoCo Menagerie is a collection with per-model licenses;
- MANO and SMPL-X require separate acceptance and restrict redistribution.

See [`THIRD_PARTY_NOTICES.md`](../THIRD_PARTY_NOTICES.md) and every bundled upstream license before use.
