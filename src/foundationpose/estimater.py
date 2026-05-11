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
from pathlib import Path
from typing import Literal, NamedTuple

import numpy as np
import nvdiffrast.torch as dr
import torch
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

    def __init__(self, checkpoints_dir: str | Path, amp: bool = True, device: str | torch.device = "cuda") -> None:
        """Initialize the pose estimator with mesh geometry and predictors."""
        self.pose_last = None  # Used for tracking; per the centered mesh.
        self.device = torch.device(device)
        self.symmetry_tfs = torch.eye(4, dtype=torch.float32, device=self.device).unsqueeze(0)
        self.rot_grid = self.make_rotation_grid(min_n_views=40, inplane_step=60)

        # Load submodules
        checkpoints_dir = Path(checkpoints_dir)
        self.scorer = ScorePredictor(checkpoints_dir=checkpoints_dir, amp=amp, device=device)
        self.refiner = PoseRefinePredictor(checkpoints_dir=checkpoints_dir, amp=amp, device=device)

        self.glctx = dr.RasterizeCudaContext(self.device)  # dr.RasterizeGLContext()

    def set_object(self, mesh: trimesh.Trimesh, symmetry_tfs: torch.Tensor | None = None) -> None:
        """Rebuild mesh-derived caches for a new object model."""
        # Center mesh
        mesh_c = mesh.copy()
        mesh_verts = mesh.vertices
        mesh_center = (mesh_verts.min(axis=0) + mesh_verts.max(axis=0)) / 2
        mesh_c.apply_translation(-mesh_center)
        self.mesh_center = torch.as_tensor(mesh_center, device=self.device, dtype=torch.float32)

        # Create and cache mesh tensors
        self.mesh_tensors: TensorMap = make_mesh_tensors(mesh_c, self.device)

        # Compute mesh diameter
        self.diameter = compute_mesh_diameter(pts=self.mesh_tensors["pos"], n_sample=10000, chunk_size=4096)

        # Set up symmetry transforms
        if symmetry_tfs is None:
            self.symmetry_tfs = torch.eye(4, dtype=torch.float32, device=self.device).unsqueeze(0)
        else:
            self.symmetry_tfs = symmetry_tfs.to(device=self.device, dtype=torch.float32)
            self.rot_grid = self.make_rotation_grid(min_n_views=40, inplane_step=60)

    def get_tf_to_centered_mesh(self) -> torch.Tensor:
        """Return the transform from the original object frame to the centered mesh frame."""
        tf_to_center = torch.eye(4, dtype=torch.float32, device=self.device)
        tf_to_center[:3, 3] = -self.mesh_center
        return tf_to_center

    def to_device(self, device: str = "cuda:0") -> None:
        """Move cached tensors, models, and raster context to a target device."""
        for key, value in list(self.__dict__.items()):
            if torch.is_tensor(value) or isinstance(value, nn.Module):
                self.__dict__[key] = value.to(device)

        for key, value in self.mesh_tensors.items():
            self.mesh_tensors[key] = value.to(device)

        if self.refiner is not None:
            self.refiner.model.to(device)
        if self.scorer is not None:
            self.scorer.model.to(device)
        if self.glctx is not None:
            self.glctx = dr.RasterizeCudaContext(device)

    def make_rotation_grid(self, min_n_views: int = 40, inplane_step: int = 60) -> torch.Tensor:
        """Precompute a clustered rotation hypothesis grid for initialization."""
        cam_in_obs = sample_views_icosphere(n_views=min_n_views)
        rot_grid: list[np.ndarray] = []
        for cam_in_ob in cam_in_obs:
            for inplane_rot in np.deg2rad(np.arange(0, 360, inplane_step)):
                rot_inplane = euler_matrix(0, 0, inplane_rot)
                cam_in_ob_rotated = cam_in_ob @ rot_inplane
                ob_in_cam = np.linalg.inv(cam_in_ob_rotated)
                rot_grid.append(ob_in_cam)

        rot_grid = np.asarray(rot_grid)
        rot_grid = cluster_poses(30, 99999, rot_grid, self.symmetry_tfs.detach().cpu().numpy())
        rot_grid = np.asarray(rot_grid)
        return torch.as_tensor(rot_grid, device=self.device, dtype=torch.float32)

    def generate_random_pose_hypo(
        self, depth: torch.Tensor, mask: torch.Tensor, intrinsics_px: torch.Tensor
    ) -> torch.Tensor:
        """Generate rotation-grid pose hypotheses centered at the estimated translation."""
        ob_in_cams = self.rot_grid.clone()
        center = self.guess_translation(depth=depth, mask=mask, intrinsics_px=intrinsics_px)
        ob_in_cams[:, :3, 3] = center.reshape(1, 3)
        return ob_in_cams

    def guess_translation(self, depth: torch.Tensor, mask: torch.Tensor, intrinsics_px: torch.Tensor) -> torch.Tensor:
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
        center = (intrinsics_px.inverse() @ depth.new_tensor([uc, vc, 1]).reshape(3, 1)) * zc
        return center.reshape(3)

    @torch.no_grad()
    def register(
        self,
        image: torch.Tensor,
        depth: torch.Tensor,
        object_mask: torch.Tensor,
        intrinsics_px: torch.Tensor,
        iterations: int = 5,
        seed: int | None = 42,
        matching_mode: Literal["rgb_only", "rgbd_only", "both_rgbd_and_rgb"] = "rgbd_only",
        renderer_batch_size: int = 512,
        force_valid_depth_on_mask: bool = False,
    ) -> FoundationPoseRegistrationOutput:
        """Estimate object pose hypotheses from an RGB-D frame and object mask.

        Args:
            image (torch.Tensor): Input RGB image. Shape: (H, W, 3).
            depth (torch.Tensor): Input depth image. Shape: (H, W).
            object_mask (torch.Tensor): Binary mask of the object in the image. Shape: (H, W).
            intrinsics_px (torch.Tensor): Camera intrinsics in pixels. Shape: (3, 3).
            iterations (int, optional): Number of refinement iterations. Defaults to 5.
            seed (int | None, optional): Random seed for reproducibility. Defaults to 42.
            matching_mode (Literal["rgb_only", "rgbd_only", "both_rgbd_and_rgb"], optional): Matching mode.
                Defaults to "rgbd_only".
            renderer_batch_size (int, optional): Batch size for rendering. Defaults to 512.
            force_valid_depth_on_mask (bool, optional): Force a minumum number of valid depth values on mask.
                Defaults to False.

        Returns:
            FoundationPoseRegistrationOutput: Registration result sorted by descending score.
        """
        if seed is not None:
            set_seed(seed)
        if matching_mode not in ["rgb_only", "rgbd_only", "both_rgbd_and_rgb"]:
            msg = "matching_mode must be one of ['rgb_only', 'rgbd_only', 'both_rgbd_and_rgb']"
            raise ValueError(msg)

        image = torch.as_tensor(image, device=self.device, dtype=torch.float32)
        depth = torch.as_tensor(depth, device=self.device, dtype=torch.float32)
        object_mask = torch.as_tensor(object_mask, device=self.device, dtype=torch.float32)
        intrinsics_px = torch.as_tensor(intrinsics_px, device=self.device, dtype=torch.float32)

        depth[~depth.isfinite()] = 0
        depth = erode_depth(depth, radius=2)
        depth = bilateral_filter_depth(depth, radius=2)

        if ((object_mask > 0).sum() < 4) or (
            (((depth >= 0.001) & (object_mask > 0)).sum() < 4) and force_valid_depth_on_mask
        ):
            if (object_mask > 0).sum() < 4:
                logger.warning("Object mask has less than 4 pixels. Returning best translation guess.")
            else:
                logger.warning("Object mask has less than 4 pixels with valid depth. Returning best translation guess.")
            pose = torch.eye(4, dtype=torch.float32, device=self.device)
            pose[:3, 3] = self.guess_translation(depth=depth, mask=object_mask, intrinsics_px=intrinsics_px)
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

        poses = self.generate_random_pose_hypo(depth=depth, mask=object_mask, intrinsics_px=intrinsics_px)

        xyz_map = depth2xyzmap(depth, intrinsics_px)
        submodels_kwargs = {
            "mesh_tensors": self.mesh_tensors,
            "image": image,
            "depth": depth,
            "intrinsics_px": intrinsics_px,
            "ob_in_cams": poses,
            "xyz_map": xyz_map,
            "glctx": self.glctx,
            "mesh_diameter": self.diameter,
            "renderer_batch_size": renderer_batch_size,
        }
        if matching_mode == "both_rgbd_and_rgb":
            poses_depth = self.refiner.predict(rgb_only=False, iterations=iterations, **submodels_kwargs)
            poses_rgb = self.refiner.predict(rgb_only=True, iterations=iterations, **submodels_kwargs)
            poses = torch.cat([poses_depth, poses_rgb], dim=0)
        else:
            poses = self.refiner.predict(
                rgb_only=(matching_mode == "rgb_only"), iterations=iterations, **submodels_kwargs
            )

        submodels_kwargs["ob_in_cams"] = poses
        scores = self.scorer.predict(rgb_only=(matching_mode == "rgb_only"), **submodels_kwargs)

        ids = scores.argsort(descending=True)
        scores = scores[ids]
        poses = poses[ids]

        self.pose_last = poses[0]
        poses_in_original_frame = poses @ self.get_tf_to_centered_mesh()
        return FoundationPoseRegistrationOutput(
            best_pose=poses_in_original_frame[0],
            all_poses=poses_in_original_frame,
            all_poses_centered=poses,
            scores=scores,
        )

    @torch.no_grad()
    def track_one(
        self, image: torch.Tensor, depth: torch.Tensor, intrinsics_px: torch.Tensor, iterations: int
    ) -> torch.Tensor:
        """Refine the previous pose estimate on a subsequent RGB-D frame."""
        if self.pose_last is None:
            logger.error("Please init pose by register first")
            msg = "register must be called before track_one"
            raise RuntimeError(msg)

        depth = erode_depth(depth, radius=2)
        depth = bilateral_filter_depth(depth, radius=2)

        pose = self.refiner.predict(
            mesh_tensors=self.mesh_tensors,
            image=image,
            depth=depth,
            intrinsics_px=intrinsics_px,
            ob_in_cams=self.pose_last.reshape(1, 4, 4),
            xyz_map=depth2xyzmap(depth, intrinsics_px),
            mesh_diameter=self.diameter,
            glctx=self.glctx,
            iterations=iterations,
        )

        self.pose_last = pose
        return (pose @ self.get_tf_to_centered_mesh()).reshape(4, 4)
