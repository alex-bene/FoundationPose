# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

"""Inference helpers for pose-refinement training artifacts."""

import logging
from pathlib import Path

import kornia
import numpy as np
import nvdiffrast.torch as dr
import torch
import trimesh
from omegaconf import DictConfig, OmegaConf
from pytorch3d.transforms import rotation_6d_to_matrix, so3_exp_map

from foundationpose.learning.datasets.h5_dataset import PoseRefinePairH5Dataset
from foundationpose.learning.datasets.pose_dataset import BatchPoseData
from foundationpose.learning.models.refine_network import RefineNet
from foundationpose.Utils import (
    compute_crop_window_tf_batch,
    egocentric_delta_pose_to_pose,
    make_mesh_tensors,
    nvdiffrast_render,
    transform_pts,
)

logger = logging.getLogger(__name__)

TensorMap = dict[str, torch.Tensor]
ArrayTensor = np.ndarray | torch.Tensor
RasterizeContext = dr.RasterizeCudaContext | dr.RasterizeGLContext


@torch.inference_mode()
def make_crop_data_batch(
    cfg: DictConfig,
    dataset: PoseRefinePairH5Dataset,
    render_size: tuple[int, int],
    ob_in_cams: ArrayTensor,
    mesh: trimesh.Trimesh,
    rgb: ArrayTensor,
    depth: ArrayTensor,
    K: ArrayTensor,
    crop_ratio: float,
    xyz_map: ArrayTensor,
    normal_map: ArrayTensor | None = None,
    mesh_diameter: float | None = None,
    glctx: RasterizeContext | None = None,
    mesh_tensors: TensorMap | None = None,
) -> BatchPoseData:
    """Build a refinement batch from rendered and observed object crops."""
    logger.debug("Building refine crop batch")
    H, W = depth.shape[:2]
    method = "box_3d"
    tf_to_crops = compute_crop_window_tf_batch(
        pts=mesh.vertices,
        poses=ob_in_cams,
        K=K,
        crop_ratio=crop_ratio,
        out_size=(render_size[1], render_size[0]),
        method=method,
        mesh_diameter=mesh_diameter,
    )

    logger.debug("Computed crop transforms")

    B = len(ob_in_cams)
    poseA = torch.as_tensor(ob_in_cams, dtype=torch.float, device="cuda")

    bs = 512
    rgb_rs = []
    depth_rs = []
    normal_rs = []
    xyz_map_rs = []

    bbox2d_crop = torch.as_tensor(
        np.array([0, 0, cfg["input_resize"][0] - 1, cfg["input_resize"][1] - 1]).reshape(2, 2),
        device="cuda",
        dtype=torch.float,
    )
    bbox2d_ori = transform_pts(bbox2d_crop, tf_to_crops.inverse()).reshape(-1, 4)

    for b in range(0, len(poseA), bs):
        extra = {}
        rgb_r, depth_r, normal_r = nvdiffrast_render(
            ob_in_cams=poseA[b : b + bs],
            K=K,
            H=H,
            W=W,
            context="cuda",
            get_normal=cfg["use_normal"],
            glctx=glctx,
            mesh_tensors=mesh_tensors,
            output_size=cfg["input_resize"],
            bbox2d=bbox2d_ori[b : b + bs],
            use_light=True,
            extra=extra,
        )
        rgb_rs.append(rgb_r)
        depth_rs.append(depth_r[..., None])
        normal_rs.append(normal_r)
        xyz_map_rs.append(extra["xyz_map"])
    rgb_rs = torch.cat(rgb_rs, dim=0).permute(0, 3, 1, 2) * 255
    depth_rs = torch.cat(depth_rs, dim=0).permute(0, 3, 1, 2)  # (B,1,H,W)
    xyz_map_rs = torch.cat(xyz_map_rs, dim=0).permute(0, 3, 1, 2)  # (B,3,H,W)
    Ks = torch.as_tensor(K, device="cuda", dtype=torch.float).reshape(1, 3, 3)
    if cfg["use_normal"]:
        normal_rs = torch.cat(normal_rs, dim=0).permute(0, 3, 1, 2)  # (B,3,H,W)

    logger.debug("Rendered synthetic crops")

    rgbBs = kornia.geometry.transform.warp_perspective(
        torch.as_tensor(rgb, dtype=torch.float, device="cuda").permute(2, 0, 1)[None].expand(B, -1, -1, -1),
        tf_to_crops,
        dsize=render_size,
        mode="bilinear",
        align_corners=False,
    )
    if rgb_rs.shape[-2:] != cfg["input_resize"]:
        rgbAs = kornia.geometry.transform.warp_perspective(
            rgb_rs, tf_to_crops, dsize=render_size, mode="bilinear", align_corners=False
        )
    else:
        rgbAs = rgb_rs
    if xyz_map_rs.shape[-2:] != cfg["input_resize"]:
        xyz_mapAs = kornia.geometry.transform.warp_perspective(
            xyz_map_rs, tf_to_crops, dsize=render_size, mode="nearest", align_corners=False
        )
    else:
        xyz_mapAs = xyz_map_rs
    xyz_mapBs = kornia.geometry.transform.warp_perspective(
        torch.as_tensor(xyz_map, device="cuda", dtype=torch.float).permute(2, 0, 1)[None].expand(B, -1, -1, -1),
        tf_to_crops,
        dsize=render_size,
        mode="nearest",
        align_corners=False,
    )  # (B,3,H,W)

    if cfg["use_normal"]:
        normalAs = kornia.geometry.transform.warp_perspective(
            normal_rs, tf_to_crops, dsize=render_size, mode="nearest", align_corners=False
        )
        normalBs = kornia.geometry.transform.warp_perspective(
            torch.as_tensor(normal_map, dtype=torch.float, device="cuda").permute(2, 0, 1)[None].expand(B, -1, -1, -1),
            tf_to_crops,
            dsize=render_size,
            mode="nearest",
            align_corners=False,
        )
    else:
        normalAs = None
        normalBs = None

    logger.debug("Warped observed crops")

    mesh_diameters = torch.ones((len(rgbAs)), dtype=torch.float, device="cuda") * mesh_diameter
    pose_data = BatchPoseData(
        rgbAs=rgbAs,
        rgbBs=rgbBs,
        depthAs=None,
        depthBs=None,
        normalAs=normalAs,
        normalBs=normalBs,
        poseA=poseA,
        poseB=None,
        xyz_mapAs=xyz_mapAs,
        xyz_mapBs=xyz_mapBs,
        tf_to_crops=tf_to_crops,
        Ks=Ks,
        mesh_diameters=mesh_diameters,
    )
    pose_data = dataset.transform_batch(batch=pose_data, H_ori=H, W_ori=W)

    logger.debug("Prepared refinement batch data")

    return pose_data


class PoseRefinePredictor:
    """Wrapper around the pretrained pose-refinement network."""

    def __init__(self, checkpoints_dir: str | Path) -> None:  # noqa: PLR0912
        logger.debug("Initializing pose refine predictor")
        self.amp = True
        self.run_name = "2023-10-28-18-33-37"
        model_name = "model_best.pth"
        weights_dir = Path(checkpoints_dir)
        run_dir = weights_dir / self.run_name
        ckpt_dir = run_dir / model_name

        self.cfg = OmegaConf.load(run_dir / "config.yml")

        self.cfg["ckpt_dir"] = str(ckpt_dir)
        self.cfg["enable_amp"] = True

        ########## Defaults, to be backward compatible
        if "use_normal" not in self.cfg:
            self.cfg["use_normal"] = False
        if "use_mask" not in self.cfg:
            self.cfg["use_mask"] = False
        if "use_BN" not in self.cfg:
            self.cfg["use_BN"] = False
        if "c_in" not in self.cfg:
            self.cfg["c_in"] = 4
        if "crop_ratio" not in self.cfg or self.cfg["crop_ratio"] is None:
            self.cfg["crop_ratio"] = 1.2
        if "n_view" not in self.cfg:
            self.cfg["n_view"] = 1
        if "trans_rep" not in self.cfg:
            self.cfg["trans_rep"] = "tracknet"
        if "rot_rep" not in self.cfg:
            self.cfg["rot_rep"] = "axis_angle"
        if "zfar" not in self.cfg:
            self.cfg["zfar"] = 3
        if "normalize_xyz" not in self.cfg:
            self.cfg["normalize_xyz"] = False
        if isinstance(self.cfg["zfar"], str) and "inf" in self.cfg["zfar"].lower():
            self.cfg["zfar"] = np.inf
        if "normal_uint8" not in self.cfg:
            self.cfg["normal_uint8"] = False
        logger.info("self.cfg:\n%s", OmegaConf.to_yaml(self.cfg))

        self.dataset = PoseRefinePairH5Dataset(cfg=self.cfg, h5_file="", mode="test")
        self.model = RefineNet(
            use_batch_norm=self.cfg["use_BN"], rotation_representation=self.cfg["rot_rep"], c_in=self.cfg["c_in"]
        ).cuda()

        logger.info("Using pretrained model from %s", ckpt_dir)
        ckpt = torch.load(ckpt_dir)
        if "model" in ckpt:
            ckpt = ckpt["model"]
        self.model.load_state_dict(ckpt)

        self.model.cuda().eval()
        logger.info("Initialized pose refine predictor")
        self.last_trans_update = None
        self.last_rot_update = None

    @torch.inference_mode()
    def predict(  # noqa: PLR0912, PLR0915
        self,
        rgb: ArrayTensor,
        depth: ArrayTensor,
        K: ArrayTensor,
        ob_in_cams: ArrayTensor,
        xyz_map: ArrayTensor,
        normal_map: ArrayTensor | None = None,
        mesh: trimesh.Trimesh | None = None,
        mesh_tensors: TensorMap | None = None,
        glctx: RasterizeContext | None = None,
        mesh_diameter: float | None = None,
        iteration: int = 5,
    ) -> torch.Tensor:
        """Refine candidate object poses for a single RGB-D observation."""
        torch.set_default_tensor_type("torch.cuda.FloatTensor")
        logger.debug("ob_in_cams shape: %s", ob_in_cams.shape)
        tf_to_center = np.eye(4)
        ob_centered_in_cams = ob_in_cams
        mesh_centered = mesh

        logger.debug("self.cfg.use_normal: %s", self.cfg.use_normal)
        if not self.cfg.use_normal:
            normal_map = None

        crop_ratio = self.cfg["crop_ratio"]
        logger.debug(
            "trans_normalizer: %s, rot_normalizer: %s", self.cfg["trans_normalizer"], self.cfg["rot_normalizer"]
        )
        bs = 1024

        B_in_cams = torch.as_tensor(ob_centered_in_cams, device="cuda", dtype=torch.float)

        if mesh_tensors is None:
            mesh_tensors = make_mesh_tensors(mesh_centered)

        rgb_tensor = torch.as_tensor(rgb, device="cuda", dtype=torch.float)
        depth_tensor = torch.as_tensor(depth, device="cuda", dtype=torch.float)
        xyz_map_tensor = torch.as_tensor(xyz_map, device="cuda", dtype=torch.float)
        trans_normalizer = self.cfg["trans_normalizer"]
        if not isinstance(trans_normalizer, float):
            trans_normalizer = torch.as_tensor(list(trans_normalizer), device="cuda", dtype=torch.float).reshape(1, 3)

        for _ in range(iteration):
            logger.debug("Building cropped refinement inputs")
            pose_data = make_crop_data_batch(
                self.cfg,
                self.dataset,
                self.cfg.input_resize,
                B_in_cams,
                mesh_centered,
                rgb_tensor,
                depth_tensor,
                K,
                crop_ratio=crop_ratio,
                normal_map=normal_map,
                xyz_map=xyz_map_tensor,
                glctx=glctx,
                mesh_tensors=mesh_tensors,
                mesh_diameter=mesh_diameter,
            )
            B_in_cams = []
            for b in range(0, pose_data.rgbAs.shape[0], bs):
                A = torch.cat(
                    [pose_data.rgbAs[b : b + bs].cuda(), pose_data.xyz_mapAs[b : b + bs].cuda()], dim=1
                ).float()
                B = torch.cat(
                    [pose_data.rgbBs[b : b + bs].cuda(), pose_data.xyz_mapBs[b : b + bs].cuda()], dim=1
                ).float()
                logger.debug("Starting refine forward pass")
                with torch.amp.autocast("cuda", enabled=self.amp):
                    output = self.model(A, B)
                for k in output:
                    output[k] = output[k].float()
                logger.debug("Completed refine forward pass")
                if self.cfg["trans_rep"] == "tracknet":
                    if not self.cfg["normalize_xyz"]:
                        trans_delta = torch.tanh(output["trans"]) * trans_normalizer
                    else:
                        trans_delta = output["trans"]

                elif self.cfg["trans_rep"] == "deepim":
                    ks_batch = pose_data.Ks[b : b + bs]
                    tf_to_crops_batch = pose_data.tf_to_crops[b : b + bs]

                    def project_and_transform_to_crop(
                        centers: torch.Tensor,
                        ks_batch: torch.Tensor = ks_batch,
                        tf_to_crops_batch: torch.Tensor = tf_to_crops_batch,
                    ) -> torch.Tensor:
                        uvs = (ks_batch @ centers.reshape(-1, 3, 1)).reshape(-1, 3)
                        uvs = uvs / uvs[:, 2:3]
                        uvs = (tf_to_crops_batch @ uvs.reshape(-1, 3, 1)).reshape(-1, 3)
                        return uvs[:, :2]

                    rot_delta = output["rot"]
                    z_pred = output["trans"][:, 2] * pose_data.poseA[b : b + bs][..., 2, 3]
                    uvA_crop = project_and_transform_to_crop(pose_data.poseA[b : b + bs][..., :3, 3])
                    uv_pred_crop = uvA_crop + output["trans"][:, :2] * self.cfg["input_resize"][0]
                    uv_pred = transform_pts(uv_pred_crop, pose_data.tf_to_crops[b : b + bs].inverse().cuda())
                    center_pred = torch.cat(
                        [uv_pred, torch.ones((len(rot_delta), 1), dtype=torch.float, device="cuda")], dim=-1
                    )
                    center_pred = (
                        pose_data.Ks[b : b + bs].inverse().cuda() @ center_pred.reshape(len(rot_delta), 3, 1)
                    ).reshape(len(rot_delta), 3) * z_pred.reshape(len(rot_delta), 1)
                    trans_delta = center_pred - pose_data.poseA[b : b + bs][..., :3, 3]

                else:
                    trans_delta = output["trans"]

                if self.cfg["rot_rep"] == "axis_angle":
                    rot_mat_delta = torch.tanh(output["rot"]) * self.cfg["rot_normalizer"]
                    rot_mat_delta = so3_exp_map(rot_mat_delta).permute(0, 2, 1)
                elif self.cfg["rot_rep"] == "6d":
                    rot_mat_delta = rotation_6d_to_matrix(output["rot"]).permute(0, 2, 1)
                else:
                    raise RuntimeError

                if self.cfg["normalize_xyz"]:
                    trans_delta *= mesh_diameter / 2

                B_in_cam = egocentric_delta_pose_to_pose(
                    pose_data.poseA[b : b + bs], trans_delta=trans_delta, rot_mat_delta=rot_mat_delta
                )
                B_in_cams.append(B_in_cam)

            B_in_cams = torch.cat(B_in_cams, dim=0).reshape(len(ob_in_cams), 4, 4)

        B_in_cams_out = B_in_cams @ torch.tensor(tf_to_center[None], device="cuda", dtype=torch.float)
        torch.cuda.empty_cache()
        self.last_trans_update = trans_delta
        self.last_rot_update = rot_mat_delta
        return B_in_cams_out
