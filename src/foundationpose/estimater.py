# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

"""Pose registration and tracking helpers used by FoundationPose inference."""

import logging
import random
from typing import Literal, NamedTuple

import numpy as np
import nvdiffrast.torch as dr
import torch
import torch.nn.functional as F
import trimesh
from torch import nn
from transformations import euler_matrix

from foundationpose.learning.training.predict_pose_refine import PoseRefinePredictor
from foundationpose.learning.training.predict_score import ScorePredictor
from foundationpose.utils import (
    bilateral_filter_depth,
    cluster_poses,
    compute_mesh_diameter,
    depth2xyzmap,
    erode_depth,
    make_mesh_tensors,
    sample_views_icosphere,
    to_open3d_cloud,
)

logger = logging.getLogger(__name__)

TensorMap = dict[str, torch.Tensor]
RasterizeContext = dr.RasterizeCudaContext | dr.RasterizeGLContext


class FoundationPoseRegistrationOutput(NamedTuple):
    """Structured FoundationPose registration output.

    Attributes:
        best_pose: Highest-scoring pose in the original object frame. Shape: (4, 4).
        all_poses: All candidate poses sorted by descending score in the original object frame. Shape: (N, 4, 4).
        all_poses_centered: All candidate poses sorted by descending score in the centered mesh frame. Shape: (N, 4, 4).
        scores: Candidate scores sorted in descending order. Shape: (N,).
    """

    best_pose: torch.Tensor
    all_poses: torch.Tensor
    all_poses_centered: torch.Tensor
    scores: torch.Tensor


def set_seed(random_seed: int) -> None:
    """Seed Python and PyTorch RNGs for deterministic inference."""
    random.seed(random_seed)
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class FoundationPose:
    """Register and track an object's pose from RGB-D observations."""

    def __init__(
        self,
        model_normals: np.ndarray,
        symmetry_tfs: torch.Tensor | None = None,
        mesh: trimesh.Trimesh | None = None,
        scorer: ScorePredictor | None = None,
        refiner: PoseRefinePredictor | None = None,
        glctx: RasterizeContext | None = None,
        device: str | torch.device = "cuda",
    ) -> None:
        """Initialize the pose estimator with mesh geometry and predictors."""
        self.gt_pose = None
        self.ignore_normal_flip = True
        self.device = torch.device(device)

        self.reset_object(mesh, model_normals, symmetry_tfs=symmetry_tfs)
        self.make_rotation_grid(min_n_views=40, inplane_step=60)

        self.glctx = glctx
        self.scorer = scorer if scorer is not None else ScorePredictor()
        self.refiner = refiner if refiner is not None else PoseRefinePredictor()
        self.pose_last = None  # Used for tracking; per the centered mesh.

    def reset_object(
        self, mesh: trimesh.Trimesh, model_normals: np.ndarray, symmetry_tfs: torch.Tensor | None = None
    ) -> None:
        """Rebuild mesh-derived caches for a new object model."""
        max_xyz = mesh.vertices.max(axis=0)
        min_xyz = mesh.vertices.min(axis=0)
        model_center = (min_xyz + max_xyz) / 2

        self.mesh_ori = mesh.copy()
        mesh = mesh.copy()
        mesh.vertices = mesh.vertices - model_center.reshape(1, 3)

        self.model_center = torch.as_tensor(model_center, device=self.device, dtype=torch.float32)

        model_pts = mesh.vertices
        self.diameter = compute_mesh_diameter(model_pts=mesh.vertices, n_sample=10000)
        self.vox_size = max(self.diameter / 20.0, 0.003)
        logger.debug("self.diameter:%s, vox_size:%s", self.diameter, self.vox_size)
        self.dist_bin = self.vox_size / 2
        self.angle_bin = 20  # Deg
        pcd = to_open3d_cloud(model_pts, normals=model_normals)
        pcd = pcd.voxel_down_sample(self.vox_size)
        self.max_xyz = np.asarray(pcd.points).max(axis=0)
        self.min_xyz = np.asarray(pcd.points).min(axis=0)
        self.pts = torch.tensor(np.asarray(pcd.points), dtype=torch.float32, device=self.device)
        self.normals = F.normalize(
            torch.tensor(np.asarray(pcd.normals), dtype=torch.float32, device=self.device), dim=-1
        )

        self.mesh = mesh
        self.mesh_tensors: TensorMap = make_mesh_tensors(self.mesh, self.device)

        if symmetry_tfs is None:
            self.symmetry_tfs = torch.eye(4, dtype=torch.float32, device=self.device)[None]
        else:
            self.symmetry_tfs = symmetry_tfs.to(device=self.device, dtype=torch.float32)

    def get_tf_to_centered_mesh(self) -> torch.Tensor:
        """Return the transform from the original object frame to the centered mesh frame."""
        tf_to_center = torch.eye(4, dtype=torch.float32, device=self.device)
        tf_to_center[:3, 3] = -self.model_center
        return tf_to_center

    def to_device(self, device: str = "cuda:0") -> None:
        """Move cached tensors, models, and raster context to a target device."""
        for key, value in list(self.__dict__.items()):
            if torch.is_tensor(value) or isinstance(value, nn.Module):
                logger.debug("Moving %s to device %s", key, device)
                self.__dict__[key] = value.to(device)

        for key, value in self.mesh_tensors.items():
            logger.debug("Moving %s to device %s", key, device)
            self.mesh_tensors[key] = value.to(device)

        if self.refiner is not None:
            self.refiner.model.to(device)
        if self.scorer is not None:
            self.scorer.model.to(device)
        if self.glctx is not None:
            self.glctx = dr.RasterizeCudaContext(device)

    def make_rotation_grid(self, min_n_views: int = 40, inplane_step: int = 60) -> None:
        """Precompute a clustered rotation hypothesis grid for initialization."""
        cam_in_obs = sample_views_icosphere(n_views=min_n_views)
        logger.debug("cam_in_obs:%s", cam_in_obs.shape)
        rot_grid: list[np.ndarray] = []
        for cam_in_ob in cam_in_obs:
            for inplane_rot in np.deg2rad(np.arange(0, 360, inplane_step)):
                rot_inplane = euler_matrix(0, 0, inplane_rot)
                cam_in_ob_rotated = cam_in_ob @ rot_inplane
                ob_in_cam = np.linalg.inv(cam_in_ob_rotated)
                rot_grid.append(ob_in_cam)

        rot_grid = np.asarray(rot_grid)
        logger.debug("rot_grid:%s", rot_grid.shape)
        rot_grid = cluster_poses(30, 99999, rot_grid, self.symmetry_tfs.detach().cpu().numpy())
        rot_grid = np.asarray(rot_grid)
        logger.debug("after cluster, rot_grid:%s", rot_grid.shape)
        self.rot_grid = torch.as_tensor(rot_grid, device=self.device, dtype=torch.float32)
        logger.debug("self.rot_grid:%s", self.rot_grid.shape)

    def generate_random_pose_hypo(self, K: torch.Tensor, depth: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Generate rotation-grid pose hypotheses centered at the estimated translation."""
        ob_in_cams = self.rot_grid.clone()
        center = self.guess_translation(depth=depth, mask=mask, K=K)
        ob_in_cams[:, :3, 3] = center.reshape(1, 3)
        return ob_in_cams

    def guess_translation(self, depth: torch.Tensor, mask: torch.Tensor, K: torch.Tensor) -> torch.Tensor:
        """Estimate object translation from the masked depth median."""
        vs, us = torch.where(mask > 0)
        if len(us) == 0:
            logger.debug("mask is all zero")
            return depth.new_zeros(3)

        uc = (us.min() + us.max()) / 2.0
        vc = (vs.min() + vs.max()) / 2.0
        valid = mask.to(dtype=torch.bool) & (depth >= 0.001)
        if not valid.any():
            logger.debug("valid is empty")
            return depth.new_zeros(3)

        zc = torch.median(depth[valid])
        center = (K.inverse() @ depth.new_tensor([uc, vc, 1]).reshape(3, 1)) * zc
        return center.reshape(3)

    @torch.no_grad
    def register(
        self,
        K: torch.Tensor,
        rgb: torch.Tensor,
        depth: torch.Tensor,
        ob_mask: torch.Tensor,
        ob_id: int | None = None,
        glctx: RasterizeContext | None = None,
        iteration: int = 5,
        seed: int | None = 42,
        matching_mode: Literal["rgb_only", "rgbd_only", "both_rgbd_and_rgb"] = "rgbd_only",
        renderer_batch_size: int = 512,
    ) -> FoundationPoseRegistrationOutput:
        """Estimate object pose hypotheses from an RGB-D frame and object mask.

        Returns:
            FoundationPoseRegistrationOutput: Registration result sorted by descending score.
        """
        if seed is not None:
            set_seed(seed)
        if matching_mode not in ["rgb_only", "rgbd_only", "both_rgbd_and_rgb"]:
            msg = "matching_mode must be one of ['rgb_only', 'rgbd_only', 'both_rgbd_and_rgb']"
            raise ValueError(msg)

        if self.glctx is None:
            if glctx is None:
                self.glctx = dr.RasterizeCudaContext(self.device)  # dr.RasterizeGLContext()
            else:
                self.glctx = glctx

        K = torch.as_tensor(K, device=self.device, dtype=torch.float32)
        rgb = torch.as_tensor(rgb, device=self.device, dtype=torch.float32)
        depth = torch.as_tensor(depth, device=self.device, dtype=torch.float32)
        ob_mask = torch.as_tensor(ob_mask, device=self.device, dtype=torch.float32)

        depth = erode_depth(depth, radius=2)
        depth = bilateral_filter_depth(depth, radius=2)

        normal_map = None
        valid = (depth >= 0.001) & (ob_mask > 0)

        center = self.guess_translation(depth=depth, mask=ob_mask, K=K)

        if valid.sum() < 4:
            logger.warning("valid too small, return")
            pose = torch.eye(4, dtype=torch.float32, device=self.device)
            pose[:3, 3] = center
            poses = pose.reshape(1, 4, 4)
            scores = depth.new_zeros(1)
            self.pose_last = poses[0]
            poses_in_original_frame = poses @ self.get_tf_to_centered_mesh()
            return FoundationPoseRegistrationOutput(
                best_pose=poses_in_original_frame[0],
                all_poses=poses_in_original_frame,
                all_poses_centered=poses,
                scores=scores,
            )

        self.H, self.W = depth.shape[:2]
        self.K = K
        self.ob_id = ob_id
        self.ob_mask = ob_mask

        poses = self.generate_random_pose_hypo(K=K, depth=depth, mask=ob_mask)
        logger.debug("poses:%s", poses.shape)

        poses[:, :3, 3] = center.reshape(1, 3)

        xyz_map = depth2xyzmap(depth, K)
        if matching_mode == "both_rgbd_and_rgb":
            poses_list: list[torch.Tensor] = [
                self.refiner.predict(
                    mesh=self.mesh,
                    mesh_tensors=self.mesh_tensors,
                    rgb=rgb,
                    depth=depth,
                    K=K,
                    ob_in_cams=poses,
                    normal_map=normal_map,
                    xyz_map=xyz_map,
                    glctx=self.glctx,
                    mesh_diameter=self.diameter,
                    iteration=iteration,
                    rgb_only=rgb_only_i,
                    renderer_batch_size=renderer_batch_size,
                )
                for rgb_only_i in [False, True]
            ]  # [poses_depth, poses_rgb]
            poses = torch.cat(poses_list, dim=0)
        else:
            poses = self.refiner.predict(
                mesh=self.mesh,
                mesh_tensors=self.mesh_tensors,
                rgb=rgb,
                depth=depth,
                K=K,
                ob_in_cams=poses,
                normal_map=normal_map,
                xyz_map=xyz_map,
                glctx=self.glctx,
                mesh_diameter=self.diameter,
                iteration=iteration,
                rgb_only=(matching_mode == "rgb_only"),
                renderer_batch_size=renderer_batch_size,
            )

        scores = self.scorer.predict(
            mesh=self.mesh,
            rgb=rgb,
            depth=depth,
            K=K,
            ob_in_cams=poses,
            mesh_tensors=self.mesh_tensors,
            glctx=self.glctx,
            mesh_diameter=self.diameter,
            rgb_only=(matching_mode == "rgb_only"),
            renderer_batch_size=renderer_batch_size,
        )

        ids = scores.argsort(descending=True)
        logger.debug("sort ids:%s", ids)
        scores = scores[ids]
        poses = poses[ids]
        logger.debug("sorted scores:%s", scores)

        self.pose_last = poses[0]
        poses_in_original_frame = poses @ self.get_tf_to_centered_mesh()
        return FoundationPoseRegistrationOutput(
            best_pose=poses_in_original_frame[0],
            all_poses=poses_in_original_frame,
            all_poses_centered=poses,
            scores=scores,
        )

    @torch.no_grad
    def track_one(self, rgb: torch.Tensor, depth: torch.Tensor, K: torch.Tensor, iteration: int) -> torch.Tensor:
        """Refine the previous pose estimate on a subsequent RGB-D frame."""
        if self.pose_last is None:
            logger.error("Please init pose by register first")
            msg = "register must be called before track_one"
            raise RuntimeError(msg)

        depth = erode_depth(depth, radius=2)
        depth = bilateral_filter_depth(depth, radius=2)

        pose = self.refiner.predict(
            mesh=self.mesh,
            mesh_tensors=self.mesh_tensors,
            rgb=rgb,
            depth=depth,
            K=K,
            ob_in_cams=self.pose_last.reshape(1, 4, 4),
            normal_map=None,
            xyz_map=depth2xyzmap(depth, K),
            mesh_diameter=self.diameter,
            glctx=self.glctx,
            iteration=iteration,
        )

        self.pose_last = pose
        return (pose @ self.get_tf_to_centered_mesh()).reshape(4, 4)
