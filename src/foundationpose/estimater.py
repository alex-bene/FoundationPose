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

import cv2
import imageio
import numpy as np
import nvdiffrast.torch as dr
import open3d as o3d
import torch
import torch.nn.functional as F
import trimesh
from torch import nn
from transformations import euler_matrix

from foundationpose.learning.training.predict_pose_refine import PoseRefinePredictor
from foundationpose.learning.training.predict_score import ScorePredictor
from foundationpose.Utils import (
    bilateral_filter_depth,
    cluster_poses,
    compute_mesh_diameter,
    depth2xyzmap,
    depth2xyzmap_batch,
    erode_depth,
    make_mesh_tensors,
    sample_views_icosphere,
    to_open3d_cloud,
)

logger = logging.getLogger(__name__)

TensorMap = dict[str, torch.Tensor]
ArrayTensor = np.ndarray | torch.Tensor
RasterizeContext = dr.RasterizeCudaContext | dr.RasterizeGLContext


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
        symmetry_tfs: ArrayTensor | None = None,
        mesh: trimesh.Trimesh | None = None,
        scorer: ScorePredictor | None = None,
        refiner: PoseRefinePredictor | None = None,
        glctx: RasterizeContext | None = None,
        debug: int = 0,
        debug_dir: str | Path = "/home/bowen/debug/novel_pose_debug/",
    ) -> None:
        """Initialize the pose estimator with mesh geometry and predictors."""
        self.gt_pose = None
        self.ignore_normal_flip = True
        self.debug = debug
        self.debug_dir = Path(debug_dir)
        self.debug_dir.mkdir(parents=True, exist_ok=True)

        self.reset_object(mesh, model_normals, symmetry_tfs=symmetry_tfs)
        self.make_rotation_grid(min_n_views=40, inplane_step=60)

        self.glctx = glctx
        self.scorer = scorer if scorer is not None else ScorePredictor()
        self.refiner = refiner if refiner is not None else PoseRefinePredictor()
        self.pose_last = None  # Used for tracking; per the centered mesh.

    def reset_object(
        self, model_normals: np.ndarray, mesh: trimesh.Trimesh, symmetry_tfs: ArrayTensor | None = None
    ) -> None:
        """Rebuild mesh-derived caches for a new object model."""
        max_xyz = mesh.vertices.max(axis=0)
        min_xyz = mesh.vertices.min(axis=0)
        self.model_center = (min_xyz + max_xyz) / 2

        self.mesh_ori = mesh.copy()
        mesh = mesh.copy()
        mesh.vertices = mesh.vertices - self.model_center.reshape(1, 3)

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
        self.pts = torch.tensor(np.asarray(pcd.points), dtype=torch.float32, device="cuda")
        self.normals = F.normalize(torch.tensor(np.asarray(pcd.normals), dtype=torch.float32, device="cuda"), dim=-1)
        logger.debug("self.pts:%s", self.pts.shape)

        self.mesh = mesh
        self.mesh_tensors: TensorMap = make_mesh_tensors(self.mesh)

        if symmetry_tfs is None:
            self.symmetry_tfs = torch.eye(4, dtype=torch.float, device="cuda")[None]
        else:
            self.symmetry_tfs = torch.as_tensor(symmetry_tfs, device="cuda", dtype=torch.float)

        logger.debug("reset done")

    def get_tf_to_centered_mesh(self) -> torch.Tensor:
        """Return the transform from original object frame to centered mesh frame."""
        tf_to_center = torch.eye(4, dtype=torch.float, device="cuda")
        tf_to_center[:3, 3] = -torch.as_tensor(self.model_center, device="cuda", dtype=torch.float)
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
        self.rot_grid = torch.as_tensor(rot_grid, device="cuda", dtype=torch.float)
        logger.debug("self.rot_grid:%s", self.rot_grid.shape)

    def generate_random_pose_hypo(self, K: np.ndarray, depth: np.ndarray, mask: np.ndarray) -> torch.Tensor:
        """Generate rotation-grid pose hypotheses centered at the estimated translation."""
        ob_in_cams = self.rot_grid.clone()
        center = self.guess_translation(depth=depth, mask=mask, K=K)
        ob_in_cams[:, :3, 3] = torch.tensor(center, device="cuda", dtype=torch.float).reshape(1, 3)
        return ob_in_cams

    def guess_translation(self, depth: np.ndarray, mask: np.ndarray, K: np.ndarray) -> np.ndarray:
        """Estimate object translation from the masked depth median."""
        vs, us = np.where(mask > 0)
        if len(us) == 0:
            logger.debug("mask is all zero")
            return np.zeros(3)

        uc = (us.min() + us.max()) / 2.0
        vc = (vs.min() + vs.max()) / 2.0
        valid = mask.astype(bool) & (depth >= 0.001)
        if not valid.any():
            logger.debug("valid is empty")
            return np.zeros(3)

        zc = np.median(depth[valid])
        center = (np.linalg.inv(K) @ np.asarray([uc, vc, 1]).reshape(3, 1)) * zc

        if self.debug >= 2:
            pcd = to_open3d_cloud(center.reshape(1, 3))
            o3d.io.write_point_cloud(str(self.debug_dir / "init_center.ply"), pcd)

        return center.reshape(3)

    def register(  # noqa: PLR0915
        self,
        K: np.ndarray,
        rgb: np.ndarray,
        depth: np.ndarray,
        ob_mask: np.ndarray,
        ob_id: int | None = None,
        glctx: RasterizeContext | None = None,
        iteration: int = 5,
        seed: int | None = 42,
    ) -> np.ndarray:
        """Estimate an object pose from an RGB-D frame and object mask."""
        if seed is not None:
            set_seed(seed)
        logger.debug("Welcome")

        if self.glctx is None:
            if glctx is None:
                self.glctx = dr.RasterizeCudaContext()  # dr.RasterizeGLContext()
            else:
                self.glctx = glctx

        depth = erode_depth(depth, radius=2, device="cuda")
        depth = bilateral_filter_depth(depth, radius=2, device="cuda")

        if self.debug >= 2:
            xyz_map = depth2xyzmap(depth, K)
            valid = xyz_map[..., 2] >= 0.001
            pcd = to_open3d_cloud(xyz_map[valid], rgb[valid])
            o3d.io.write_point_cloud(str(self.debug_dir / "scene_raw.ply"), pcd)
            cv2.imwrite(str(self.debug_dir / "ob_mask.png"), (ob_mask * 255.0).clip(0, 255))

        normal_map = None
        valid = (depth >= 0.001) & (ob_mask > 0)
        if valid.sum() < 4:
            logger.warning("valid too small, return")
            pose = np.eye(4)
            pose[:3, 3] = self.guess_translation(depth=depth, mask=ob_mask, K=K)
            return pose

        if self.debug >= 2:
            imageio.imwrite(self.debug_dir / "color.png", rgb)
            cv2.imwrite(str(self.debug_dir / "depth.png"), (depth * 1000).astype(np.uint16))
            valid = xyz_map[..., 2] >= 0.001
            pcd = to_open3d_cloud(xyz_map[valid], rgb[valid])
            o3d.io.write_point_cloud(str(self.debug_dir / "scene_complete.ply"), pcd)

        self.H, self.W = depth.shape[:2]
        self.K = K
        self.ob_id = ob_id
        self.ob_mask = ob_mask

        poses = self.generate_random_pose_hypo(K=K, depth=depth, mask=ob_mask)
        poses = poses.detach().cpu().numpy()
        logger.debug("poses:%s", poses.shape)
        center = self.guess_translation(depth=depth, mask=ob_mask, K=K)

        poses = torch.as_tensor(poses, device="cuda", dtype=torch.float)
        poses[:, :3, 3] = torch.as_tensor(center.reshape(1, 3), device="cuda")

        xyz_map = depth2xyzmap(depth, K)
        poses, vis = self.refiner.predict(
            mesh=self.mesh,
            mesh_tensors=self.mesh_tensors,
            rgb=rgb,
            depth=depth,
            K=K,
            ob_in_cams=poses.detach().cpu().numpy(),
            normal_map=normal_map,
            xyz_map=xyz_map,
            glctx=self.glctx,
            mesh_diameter=self.diameter,
            iteration=iteration,
            get_vis=self.debug >= 2,
        )
        if vis is not None:
            imageio.imwrite(self.debug_dir / "vis_refiner.png", vis)

        scores, vis = self.scorer.predict(
            mesh=self.mesh,
            rgb=rgb,
            depth=depth,
            K=K,
            ob_in_cams=poses.detach().cpu().numpy(),
            mesh_tensors=self.mesh_tensors,
            glctx=self.glctx,
            mesh_diameter=self.diameter,
            get_vis=self.debug >= 2,
        )
        if vis is not None:
            imageio.imwrite(self.debug_dir / "vis_score.png", vis)

        ids = torch.as_tensor(scores).argsort(descending=True)
        logger.debug("sort ids:%s", ids)
        scores = scores[ids]
        poses = poses[ids]
        logger.debug("sorted scores:%s", scores)

        best_pose = poses[0] @ self.get_tf_to_centered_mesh()
        self.pose_last = poses[0]
        self.best_id = ids[0]
        self.poses = poses
        self.scores = scores
        return best_pose.detach().cpu().numpy()

    def track_one(
        self,
        rgb: np.ndarray,
        depth: np.ndarray,
        K: np.ndarray,
        iteration: int,
        extra: dict[str, np.ndarray] | None = None,
    ) -> np.ndarray:
        """Refine the previous pose estimate on a subsequent RGB-D frame."""
        if self.pose_last is None:
            logger.error("Please init pose by register first")
            msg = "register must be called before track_one"
            raise RuntimeError(msg)
        logger.debug("Welcome")

        depth_tensor = torch.as_tensor(depth, device="cuda", dtype=torch.float)
        depth_tensor = erode_depth(depth_tensor, radius=2, device="cuda")
        depth_tensor = bilateral_filter_depth(depth_tensor, radius=2, device="cuda")
        logger.debug("depth processing done")

        xyz_map = depth2xyzmap_batch(
            depth_tensor[None], torch.as_tensor(K, dtype=torch.float, device="cuda")[None], zfar=np.inf
        )[0]

        pose, vis = self.refiner.predict(
            mesh=self.mesh,
            mesh_tensors=self.mesh_tensors,
            rgb=rgb,
            depth=depth_tensor,
            K=K,
            ob_in_cams=self.pose_last.reshape(1, 4, 4).detach().cpu().numpy(),
            normal_map=None,
            xyz_map=xyz_map,
            mesh_diameter=self.diameter,
            glctx=self.glctx,
            iteration=iteration,
            get_vis=self.debug >= 2,
        )
        logger.debug("pose done")
        if extra is None:
            extra = {}
        if self.debug >= 2 and vis is not None:
            extra["vis"] = vis
        self.pose_last = pose
        return (pose @ self.get_tf_to_centered_mesh()).detach().cpu().numpy().reshape(4, 4)
