# FoundationPose

[![pre-commit.ci status](https://results.pre-commit.ci/badge/github/alex-bene/FoundationPose/main.svg)](https://results.pre-commit.ci/latest/github/alex-bene/FoundationPose/main)
[![Development Status](https://img.shields.io/badge/status-beta-orange)](https://github.com/alex-bene/FoundationPose)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)

[[Paper]](https://arxiv.org/abs/2312.08344) [[Original implementation]](https://github.com/NVlabs/FoundationPose) [[Website]](https://nvlabs.github.io/FoundationPose/)

This repository is a simplified fork of the original [NVLabs FoundationPose implementation](https://github.com/NVlabs/FoundationPose), refactored as an installable Python package.

The goal of this fork is packaging and simplification, not preserving the full upstream repository workflow.

## What Changed In This Fork

- The codebase has been refactored into an installable package under `src/foundationpose`.
- Installation is intended to happen directly from GitHub with `uv`.
- The original `mycpp` extension has been removed.
- Pose clustering now uses a simplified pure-Python implementation from [CARI4D](https://github.com/NVlabs/CARI4D).
- This fork is focused on the inference-oriented package code path.
- Removed debugging visualization logic

## Installation

This package is intended to be installed directly from GitHub with `uv`.

```bash
uv add git+https://github.com/alex-bene/FoundationPose.git
```

## Pretrained Weights

Download the original FoundationPose network weights from the upstream release assets [here](https://drive.google.com/drive/folders/1DFezOAD0oD1BblsXVxqDsl8fj0qzB82i?usp=sharing).

You will need:

- Refiner: `2023-10-28-18-33-37`
- Scorer: `2024-01-11-20-02-45`

Keep them in a local checkpoints directory and pass that directory when constructing the predictors.

You can download them using:
```bash
uvx gdown --folder --output [CHECKPOINTS_DIR] https://drive.google.com/drive/folders/1DFezOAD0oD1BblsXVxqDsl8fj0qzB82i
```

## Minimal Usage

```python
from foundationpose.estimater import FoundationPose
from foundationpose.learning.training.predict_pose_refine import PoseRefinePredictor
from foundationpose.learning.training.predict_score import ScorePredictor

checkpoints_dir = "/path/to/weights"

refiner = PoseRefinePredictor(checkpoints_dir=checkpoints_dir)
scorer = ScorePredictor(checkpoints_dir=checkpoints_dir)

estimator = FoundationPose(
    model_normals=model_normals,
    mesh=mesh,
    scorer=scorer,
    refiner=refiner,
)
```

## Bibtex

```bibtex
@InProceedings{foundationposewen2024,
  author        = {Bowen Wen, Wei Yang, Jan Kautz, Stan Birchfield},
  title         = {{FoundationPose}: Unified 6D Pose Estimation and Tracking of Novel Objects},
  booktitle     = {CVPR},
  year          = {2024},
}
```

If you find the model-free setup useful, please also consider cite:

```bibtex
@InProceedings{bundlesdfwen2023,
  author        = {Bowen Wen and Jonathan Tremblay and Valts Blukis and Stephen Tyree and Thomas M\"{u}ller and Alex Evans and Dieter Fox and Jan Kautz and Stan Birchfield},
  title         = {{BundleSDF}: {N}eural 6-{DoF} Tracking and {3D} Reconstruction of Unknown Objects},
  booktitle     = {CVPR},
  year          = {2023},
}
```

Also, consider citing CARI4D since we use their implementation for `cluster_poses`:

```bibtex
@inproceedings{xie2026cari4d,
  title = {CARI4D: Category Agnostic 4D Reconstruction of Human-Object Interaction},
  author = {Xie, Xianghui and Wen, Bowen and Chang, Yan and Rabeti, Hesam and Li, Jiefeng and Yuan, Ye and Pons-Moll, Gerard and Birchfield, Stan},
  booktitle = {Conference on Computer Vision and Pattern Recognition ({CVPR})},
  month = {June},
  year = {2026},
}
```

## License

This fork keeps the same license as the original repository: the NVIDIA Source Code License.

The underlying FoundationPose code originates from NVIDIA / NVLabs, and this repository is a derivative packaging and refactoring effort. We do not claim ownership of the original FoundationPose codebase or attempt to relicense it. See [LICENSE](LICENSE).
