# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.


"""HDF5 dataset classes for pose estimation training."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import kornia
import numpy as np
import torch

from foundationpose.utils import depth2xyzmap_batch

if TYPE_CHECKING:
    from .pose_dataset import BatchPoseData

logger = logging.getLogger(__name__)


class PairH5Dataset(torch.utils.data.Dataset):
    def __init__(self, cfg: dict, h5_file: str, mode: str = "test") -> None:
        self.cfg = cfg
        self.h5_file = h5_file
        self.mode = mode

        logger.debug("self.h5_file:%s", self.h5_file)
        self.n_perturb = None
        self.H_ori = None
        self.W_ori = None

        if self.mode != "test":
            msg = "PairH5Dataset only supports test mode."
            raise NotImplementedError(msg)

    def __len__(self):
        return 1

    def transform_depth_to_xyzmap(self, batch: BatchPoseData, H_ori, W_ori, subtract_trans=True, crop_xyz_3d=False):
        bs = len(batch.rgbAs)
        H, W = batch.rgbAs.shape[-2:]
        mesh_radius = batch.mesh_diameters.cuda() / 2
        tf_to_crops = batch.tf_to_crops.cuda()
        crop_to_oris = batch.tf_to_crops.inverse().cuda()  # (B,3,3)
        batch.poseA = batch.poseA.cuda()
        batch.Ks = batch.Ks.cuda()
        if batch.xyz_mapAs is None and batch.depthAs is not None:
            depthAs_ori = kornia.geometry.transform.warp_perspective(
                batch.depthAs.cuda().expand(bs, -1, -1, -1),
                crop_to_oris,
                dsize=(H_ori, W_ori),
                mode="nearest",
                align_corners=False,
            )
            batch.xyz_mapAs = depth2xyzmap_batch(depthAs_ori[:, 0], batch.Ks, zfar=np.inf).permute(
                0, 3, 1, 2
            )  # (B,3,H,W)
            batch.xyz_mapAs = kornia.geometry.transform.warp_perspective(
                batch.xyz_mapAs, tf_to_crops, dsize=(H, W), mode="nearest", align_corners=False
            )
        if batch.xyz_mapAs is not None:
            batch.xyz_mapAs = batch.xyz_mapAs.cuda()
            if self.cfg["normalize_xyz"]:
                invalid = batch.xyz_mapAs[:, 2:3] < 0.001
            if batch.xyz_mapAs.shape[1] == 3:
                batch.xyz_mapAs = (
                    batch.xyz_mapAs - batch.poseA[:, :3, 3].reshape(bs, 3, 1, 1) if subtract_trans else batch.xyz_mapAs
                )
                if self.cfg["normalize_xyz"]:
                    batch.xyz_mapAs *= 1 / mesh_radius.reshape(bs, 1, 1, 1)
                    invalid = invalid.expand(bs, 3, -1, -1) | (torch.abs(batch.xyz_mapAs) >= 2)
                    batch.xyz_mapAs[invalid.expand(bs, 3, -1, -1)] = 0
            else:
                assert batch.xyz_mapAs.shape[1] == 5, f"invalid xyz_mapAs shape {batch.xyz_mapAs.shape}"
                mask_ho = batch.xyz_mapAs[:, 3:]
                batch.xyz_mapAs = (
                    batch.xyz_mapAs[:, :3] - batch.poseA[:, :3, 3].reshape(bs, 3, 1, 1)
                    if subtract_trans
                    else batch.xyz_mapAs[:, :3]
                )
                if self.cfg["normalize_xyz"]:
                    batch.xyz_mapAs *= 1 / mesh_radius.reshape(bs, 1, 1, 1)
                    invalid = invalid.expand(bs, 3, -1, -1) | (torch.abs(batch.xyz_mapAs) >= 2)
                    batch.xyz_mapAs[invalid.expand(bs, 3, -1, -1)] = 0
                batch.xyz_mapAs = torch.cat([batch.xyz_mapAs, mask_ho], dim=1)

        if batch.xyz_mapBs is None and batch.depthBs is not None:
            depthBs_ori = kornia.geometry.transform.warp_perspective(
                batch.depthBs.cuda().expand(bs, -1, -1, -1),
                crop_to_oris,
                dsize=(H_ori, W_ori),
                mode="nearest",
                align_corners=False,
            )
            batch.xyz_mapBs = depth2xyzmap_batch(depthBs_ori[:, 0], batch.Ks, zfar=np.inf).permute(
                0, 3, 1, 2
            )  # (B,3,H,W)
            batch.xyz_mapBs = kornia.geometry.transform.warp_perspective(
                batch.xyz_mapBs, tf_to_crops, dsize=(H, W), mode="nearest", align_corners=False
            )
        if batch.xyz_mapBs is not None:
            batch.xyz_mapBs = batch.xyz_mapBs.cuda()
            if self.cfg["normalize_xyz"]:
                invalid = batch.xyz_mapBs[:, 2:3] < 0.001
            # TODO: understand why here
            if batch.xyz_mapBs.shape[1] == 3:
                batch.xyz_mapBs = (
                    batch.xyz_mapBs - batch.poseA[:, :3, 3].reshape(bs, 3, 1, 1) if subtract_trans else batch.xyz_mapBs
                )
                if self.cfg["normalize_xyz"]:
                    batch.xyz_mapBs *= 1 / mesh_radius.reshape(bs, 1, 1, 1)
                    invalid = invalid.expand(bs, 3, -1, -1) | (torch.abs(batch.xyz_mapBs) >= 2)
                    batch.xyz_mapBs[invalid.expand(bs, 3, -1, -1)] = 0
            else:
                assert batch.xyz_mapBs.shape[1] == 5, f"invalid xyz_mapBs shape {batch.xyz_mapBs.shape}"
                mask_ho = batch.xyz_mapBs[:, 3:]  # TODO: understand why poseA is subtracted?
                batch.xyz_mapBs = (
                    batch.xyz_mapBs[:, :3] - batch.poseA[:, :3, 3].reshape(bs, 3, 1, 1)
                    if subtract_trans
                    else batch.xyz_mapBs[:, :3]
                )
                if self.cfg["normalize_xyz"]:
                    batch.xyz_mapBs *= 1 / mesh_radius.reshape(bs, 1, 1, 1)
                    invalid = invalid.expand(bs, 3, -1, -1) | (
                        torch.abs(batch.xyz_mapBs) >= 2
                    )  # XH: here it is cropped!!
                    batch.xyz_mapBs[invalid.expand(bs, 3, -1, -1)] = 0
                batch.xyz_mapBs = torch.cat([batch.xyz_mapBs, mask_ho], dim=1)
        if crop_xyz_3d and batch.xyz_mapBs is not None:
            assert subtract_trans
            print("cropping xyz_mapBs in 3D bbox")
            dmap_xyz = batch.xyz_mapBs[:, :3]
            bound_min, bound_max = np.array([-1, -1, -1.0]), np.array([1, 1, 1.0])
            m = (
                (dmap_xyz[:, 0] < bound_max[0])
                & (dmap_xyz[:, 0] > bound_min[0])
                & (dmap_xyz[:, 1] < bound_max[1])
                & (dmap_xyz[:, 1] > bound_min[1])
                & (dmap_xyz[:, 1] < bound_max[2])
                & (dmap_xyz[:, 1] > bound_min[2])
            )  # (B, H, W)
            dmap_xyz[~m[:, None].repeat(1, 3, 1, 1)] = 0.0
            batch.xyz_mapBs[:, :3] = dmap_xyz

        return batch

    def transform_batch(self, batch: BatchPoseData, H_ori, W_ori):
        batch.rgbAs = batch.rgbAs.cuda().float() / 255.0
        batch.rgbBs = batch.rgbBs.cuda().float() / 255.0

        return self.transform_depth_to_xyzmap(batch, H_ori, W_ori)


class TripletH5Dataset(PairH5Dataset):
    def transform_depth_to_xyzmap(self, batch: BatchPoseData, H_ori, W_ori):
        bs = len(batch.rgbAs)
        H, W = batch.rgbAs.shape[-2:]
        mesh_radius = batch.mesh_diameters.cuda() / 2
        tf_to_crops = batch.tf_to_crops.cuda()
        crop_to_oris = batch.tf_to_crops.inverse().cuda()  # (B,3,3)
        batch.poseA = batch.poseA.cuda()
        batch.Ks = batch.Ks.cuda()

        if batch.xyz_mapAs is None and batch.depthAs is not None:
            depthAs_ori = kornia.geometry.transform.warp_perspective(
                batch.depthAs.cuda().expand(bs, -1, -1, -1),
                crop_to_oris,
                dsize=(H_ori, W_ori),
                mode="nearest",
                align_corners=False,
            )
            batch.xyz_mapAs = depth2xyzmap_batch(depthAs_ori[:, 0], batch.Ks, zfar=np.inf).permute(
                0, 3, 1, 2
            )  # (B,3,H,W)
            batch.xyz_mapAs = kornia.geometry.transform.warp_perspective(
                batch.xyz_mapAs, tf_to_crops, dsize=(H, W), mode="nearest", align_corners=False
            )
        if batch.xyz_mapAs is not None:
            batch.xyz_mapAs = batch.xyz_mapAs.cuda()
            invalid = batch.xyz_mapAs[:, 2:3] < 0.1
            batch.xyz_mapAs = batch.xyz_mapAs - batch.poseA[:, :3, 3].reshape(bs, 3, 1, 1)
            if self.cfg["normalize_xyz"]:
                batch.xyz_mapAs *= 1 / mesh_radius.reshape(bs, 1, 1, 1)
                invalid = invalid.expand(bs, 3, -1, -1) | (torch.abs(batch.xyz_mapAs) >= 2)
                batch.xyz_mapAs[invalid.expand(bs, 3, -1, -1)] = 0

        if batch.xyz_mapBs is None and batch.depthBs is not None:
            # make mini batch to avoid OOM issue
            chunk_size, xyz_mapBs_list = 128, []
            for i in range(0, bs, chunk_size):
                depthBs_ori = kornia.geometry.transform.warp_perspective(
                    batch.depthBs.expand(bs, -1, -1, -1)[i : i + chunk_size],
                    crop_to_oris[i : i + chunk_size],
                    dsize=(H_ori, W_ori),
                    mode="nearest",
                    align_corners=False,
                )
                xyz_mapBs = depth2xyzmap_batch(depthBs_ori[:, 0], batch.Ks[i : i + chunk_size], zfar=np.inf).permute(
                    0, 3, 1, 2
                )  # (B,3,H,W)
                xyz_mapBs = kornia.geometry.transform.warp_perspective(
                    xyz_mapBs, tf_to_crops[i : i + chunk_size], dsize=(H, W), mode="nearest", align_corners=False
                )
                xyz_mapBs_list.append(xyz_mapBs)
            batch.xyz_mapBs = torch.cat(xyz_mapBs_list, 0)
        if batch.xyz_mapBs is not None:
            batch.xyz_mapBs = batch.xyz_mapBs.cuda()
            invalid = batch.xyz_mapBs[:, 2:3] < 0.1
            batch.xyz_mapBs = batch.xyz_mapBs - batch.poseA[:, :3, 3].reshape(bs, 3, 1, 1)
            if self.cfg["normalize_xyz"]:
                batch.xyz_mapBs *= 1 / mesh_radius.reshape(bs, 1, 1, 1)
                invalid = invalid.expand(bs, 3, -1, -1) | (torch.abs(batch.xyz_mapBs) >= 2)
                batch.xyz_mapBs[invalid.expand(bs, 3, -1, -1)] = 0

        return batch

    def transform_batch(self, batch: BatchPoseData, H_ori, W_ori):
        batch.rgbAs = batch.rgbAs.cuda().float() / 255.0
        batch.rgbBs = batch.rgbBs.cuda().float() / 255.0

        return self.transform_depth_to_xyzmap(batch, H_ori, W_ori)


class ScoreMultiPairH5Dataset(TripletH5Dataset):
    def __init__(self, cfg, h5_file, mode):
        super().__init__(cfg, h5_file, mode)
        if mode in ["train", "val"]:
            self.cfg["train_num_pair"] = self.n_perturb


class PoseRefinePairH5Dataset(PairH5Dataset):
    def __init__(self, cfg, h5_file, mode="test"):
        super().__init__(cfg=cfg, h5_file=h5_file, mode=mode)

    def transform_batch(self, batch: BatchPoseData, H_ori, W_ori, subtract_trans=True, crop_xyz_3d=False):
        batch.rgbAs = batch.rgbAs.cuda().float() / 255.0
        batch.rgbBs = batch.rgbBs.cuda().float() / 255.0

        return self.transform_depth_to_xyzmap(
            batch, H_ori, W_ori, subtract_trans=subtract_trans, crop_xyz_3d=crop_xyz_3d
        )
