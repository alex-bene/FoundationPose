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
import pickle
from pathlib import Path
from typing import TYPE_CHECKING

import h5py
import imageio
import kornia
import numpy as np
import torch
from torch.utils.data import Dataset

from foundationpose.utils import depth2xyzmap_batch

if TYPE_CHECKING:
    from .pose_dataset import BatchPoseData

logger = logging.getLogger(__name__)


class PairH5Dataset(Dataset):
    """HDF5 dataset for loading paired observation data."""

    def __init__(self, cfg: dict, h5_file: str, mode: str = "train", max_num_key: int | None = None) -> None:
        self.cfg = cfg
        self.h5_file = h5_file
        self.mode = mode

        logger.debug("self.h5_file:%s", self.h5_file)
        self.n_perturb = None
        self.H_ori = None
        self.W_ori = None

        if self.mode != "test":
            self.object_keys = []
            key_file = Path(h5_file.replace(".h5", "_keys.pkl"))
            if key_file.exists():
                with key_file.open("rb") as ff:
                    self.object_keys = pickle.load(ff)  # noqa: S301
                logger.debug("object_keys loaded#:%d from %s", len(self.object_keys), key_file)
                if max_num_key is not None:
                    self.object_keys = self.object_keys[:max_num_key]
            else:
                with h5py.File(h5_file, "r", libver="latest") as hf:
                    for k in hf:
                        self.object_keys.append(k)
                        if max_num_key is not None and len(self.object_keys) >= max_num_key:
                            logger.debug("break due to max_num_key")
                            break

            logger.debug("self.object_keys#:%d, max_num_key:%s", len(self.object_keys), max_num_key)

            with h5py.File(h5_file, "r", libver="latest") as hf:
                group = hf[self.object_keys[0]]
                cnt = 0
                for k_perturb in group:
                    if "i_perturb" in k_perturb:
                        cnt += 1
                    if "crop_ratio" in group[k_perturb]:
                        self.cfg["crop_ratio"] = float(group[k_perturb]["crop_ratio"][()])
                    if self.H_ori is None:
                        if "H_ori" in group[k_perturb]:
                            self.H_ori = int(group[k_perturb]["H_ori"][()])
                            self.W_ori = int(group[k_perturb]["W_ori"][()])
                        else:
                            self.H_ori = 540
                            self.W_ori = 720
                self.n_perturb = cnt
                logger.debug("self.n_perturb:%d", self.n_perturb)

    def __len__(self) -> int:
        """Return the number of samples in the dataset."""
        if self.mode == "test":
            return 1
        return len(self.object_keys)

    def transform_depth_to_xyzmap(self, batch: BatchPoseData, H_ori: int, W_ori: int) -> BatchPoseData:
        """Transform depth maps to XYZ coordinate maps."""
        bs = len(batch.rgbAs)
        H, W = batch.rgbAs.shape[-2:]
        mesh_radius = batch.mesh_diameters.cuda() / 2
        tf_to_crops = batch.tf_to_crops.cuda()
        crop_to_oris = batch.tf_to_crops.inverse().cuda()  # (B,3,3)
        batch.poseA = batch.poseA.cuda()
        batch.Ks = batch.Ks.cuda()

        if batch.xyz_mapAs is None:
            print("SHOULD NOT HAPPEN")  # noqa: T201
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
        batch.xyz_mapAs = batch.xyz_mapAs.cuda()
        if self.cfg["normalize_xyz"]:
            invalid = batch.xyz_mapAs[:, 2:3] < 0.001
        batch.xyz_mapAs = batch.xyz_mapAs - batch.poseA[:, :3, 3].reshape(bs, 3, 1, 1)
        if self.cfg["normalize_xyz"]:
            batch.xyz_mapAs *= 1 / mesh_radius.reshape(bs, 1, 1, 1)
            invalid = invalid.expand(bs, 3, -1, -1) | (torch.abs(batch.xyz_mapAs) >= 2)
            batch.xyz_mapAs[invalid.expand(bs, 3, -1, -1)] = 0

        if batch.xyz_mapBs is None:
            print("SHOULD NOT HAPPEN")  # noqa: T201
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
        batch.xyz_mapBs = batch.xyz_mapBs.cuda()
        if self.cfg["normalize_xyz"]:
            invalid = batch.xyz_mapBs[:, 2:3] < 0.001
        batch.xyz_mapBs = batch.xyz_mapBs - batch.poseA[:, :3, 3].reshape(bs, 3, 1, 1)
        if self.cfg["normalize_xyz"]:
            batch.xyz_mapBs *= 1 / mesh_radius.reshape(bs, 1, 1, 1)
            invalid = invalid.expand(bs, 3, -1, -1) | (torch.abs(batch.xyz_mapBs) >= 2)
            batch.xyz_mapBs[invalid.expand(bs, 3, -1, -1)] = 0

        return batch

    def transform_batch(self, batch: BatchPoseData, H_ori: int, W_ori: int) -> BatchPoseData:
        """Transform the batch before feeding to the network.

        Note: H_ori, W_ori could be different at test time from the training data, and needs to be set
        """
        batch.rgbAs = batch.rgbAs.cuda().float() / 255.0
        batch.rgbBs = batch.rgbBs.cuda().float() / 255.0

        return self.transform_depth_to_xyzmap(batch, H_ori, W_ori)


class TripletH5Dataset(PairH5Dataset):
    """HDF5 dataset for loading triplet observation data."""

    def __init__(self, cfg: dict, h5_file: str, mode: str, max_num_key: int | None = None) -> None:
        super().__init__(cfg, h5_file, mode, max_num_key)

    def transform_depth_to_xyzmap(self, batch: BatchPoseData, H_ori: int, W_ori: int) -> BatchPoseData:
        """Transform depth maps to XYZ coordinate maps."""
        bs = len(batch.rgbAs)
        H, W = batch.rgbAs.shape[-2:]
        mesh_radius = batch.mesh_diameters.cuda() / 2
        tf_to_crops = batch.tf_to_crops.cuda()
        crop_to_oris = batch.tf_to_crops.inverse().cuda()  # (B,3,3)
        batch.poseA = batch.poseA.cuda()
        batch.Ks = batch.Ks.cuda()

        if batch.xyz_mapAs is None:
            print("SHOULD NOT HAPPEN")  # noqa: T201
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
        batch.xyz_mapAs = batch.xyz_mapAs.cuda()
        invalid = batch.xyz_mapAs[:, 2:3] < 0.1
        batch.xyz_mapAs = batch.xyz_mapAs - batch.poseA[:, :3, 3].reshape(bs, 3, 1, 1)
        if self.cfg["normalize_xyz"]:
            batch.xyz_mapAs *= 1 / mesh_radius.reshape(bs, 1, 1, 1)
            invalid = invalid.expand(bs, 3, -1, -1) | (torch.abs(batch.xyz_mapAs) >= 2)
            batch.xyz_mapAs[invalid.expand(bs, 3, -1, -1)] = 0

        if batch.xyz_mapBs is None:
            print("SHOULD NOT HAPPEN")  # noqa: T201
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
        batch.xyz_mapBs = batch.xyz_mapBs.cuda()
        invalid = batch.xyz_mapBs[:, 2:3] < 0.1
        batch.xyz_mapBs = batch.xyz_mapBs - batch.poseA[:, :3, 3].reshape(bs, 3, 1, 1)
        if self.cfg["normalize_xyz"]:
            batch.xyz_mapBs *= 1 / mesh_radius.reshape(bs, 1, 1, 1)
            invalid = invalid.expand(bs, 3, -1, -1) | (torch.abs(batch.xyz_mapBs) >= 2)
            batch.xyz_mapBs[invalid.expand(bs, 3, -1, -1)] = 0

        return batch

    def transform_batch(self, batch: BatchPoseData, H_ori: int, W_ori: int) -> BatchPoseData:
        """Transform the batch before feeding to the network."""
        batch.rgbAs = batch.rgbAs.cuda().float() / 255.0
        batch.rgbBs = batch.rgbBs.cuda().float() / 255.0

        return self.transform_depth_to_xyzmap(batch, H_ori, W_ori)


class ScoreMultiPairH5Dataset(TripletH5Dataset):
    """HDF5 dataset for multi-pair scoring."""

    def __init__(self, cfg: dict, h5_file: str, mode: str, max_num_key: int | None = None) -> None:
        super().__init__(cfg, h5_file, mode, max_num_key)
        if mode in ["train", "val"]:
            self.cfg["train_num_pair"] = self.n_perturb


class PoseRefinePairH5Dataset(PairH5Dataset):
    """HDF5 dataset for pose refinement training."""

    def __init__(self, cfg: dict, h5_file: str, mode: str = "train", max_num_key: int | None = None) -> None:
        super().__init__(cfg=cfg, h5_file=h5_file, mode=mode, max_num_key=max_num_key)

        if mode != "test":
            with h5py.File(h5_file, "r", libver="latest") as hf:
                group = hf[self.object_keys[0]]
                for key_perturb in group:
                    depthA = imageio.imread(group[key_perturb]["depthA"][()])
                    depthB = imageio.imread(group[key_perturb]["depthB"][()])
                    self.cfg["n_view"] = min(self.cfg["n_view"], depthA.shape[1] // depthB.shape[1])
                    logger.debug("n_view:%d", self.cfg["n_view"])
                    self.trans_normalizer = group[key_perturb]["trans_normalizer"][()]
                    if isinstance(self.trans_normalizer, np.ndarray):
                        self.trans_normalizer = self.trans_normalizer.tolist()
                    self.rot_normalizer = group[key_perturb]["rot_normalizer"][()] / 180.0 * np.pi
                    logger.debug(
                        "self.trans_normalizer:%s, self.rot_normalizer:%s", self.trans_normalizer, self.rot_normalizer
                    )
                    break

    def transform_batch(self, batch: BatchPoseData, H_ori: int, W_ori: int) -> BatchPoseData:
        """Transform the batch before feeding to the network.

        Note: H_ori, W_ori could be different at test time from training data.
        """
        batch.rgbAs = batch.rgbAs.cuda().float() / 255.0
        batch.rgbBs = batch.rgbBs.cuda().float() / 255.0

        return self.transform_depth_to_xyzmap(batch, H_ori, W_ori)
