# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

"""Pose data structures for training and inference."""

from __future__ import annotations

import contextlib
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import torch


@dataclass
class BatchPoseData:
    """Batched pose data for training.

    Attributes:
        rgbs: RGB images, shape (bsz, 3, h, w) torch tensor uint8.
        depths: Depth maps, shape (bsz, h, w) float32.
        bboxes: Bounding boxes, shape (bsz, 4) int.
        K: Camera intrinsic matrices, shape (bsz, 3, 3) float32.
    """

    rgbs: torch.Tensor | None = None
    object_datas: torch.Tensor | None = None
    bboxes: torch.Tensor | None = None
    K: torch.Tensor | None = None
    depths: torch.Tensor | None = None
    rgbAs: torch.Tensor | None = None
    rgbBs: torch.Tensor | None = None
    depthAs: torch.Tensor | None = None
    depthBs: torch.Tensor | None = None
    normalAs: torch.Tensor | None = None
    normalBs: torch.Tensor | None = None
    poseA: torch.Tensor | None = None  # (B,4,4)
    poseB: torch.Tensor | None = None
    targets: torch.Tensor | None = None  # Score targets, torch tensor (B)
    maskAs: torch.Tensor | None = None
    maskBs: torch.Tensor | None = None
    xyz_mapAs: torch.Tensor | None = None
    xyz_mapBs: torch.Tensor | None = None
    tf_to_crops: torch.Tensor | None = None
    Ks: torch.Tensor | None = None
    crop_masks: torch.Tensor | None = None
    model_pts: torch.Tensor | None = None
    mesh_diameters: torch.Tensor | None = None
    labels: torch.Tensor | None = None

    def pin_memory(self) -> BatchPoseData:
        """Pin all tensor attributes to memory for faster host-to-device transfer."""
        for k in self.__dict__:
            if self.__dict__[k] is not None:
                with contextlib.suppress(Exception):
                    self.__dict__[k] = self.__dict__[k].pin_memory()
        return self

    def cuda(self) -> BatchPoseData:
        """Move all tensor attributes to CUDA device."""
        for k in self.__dict__:
            if self.__dict__[k] is not None:
                with contextlib.suppress(BaseException):
                    self.__dict__[k] = self.__dict__[k].cuda()
        return self

    def select_by_indices(self, ids: torch.Tensor) -> BatchPoseData:
        """Select a subset of the batch by indices."""
        out = BatchPoseData()
        for k in self.__dict__:
            if self.__dict__[k] is not None:
                out.__dict__[k] = self.__dict__[k][ids.to(self.__dict__[k].device)]
        return out
