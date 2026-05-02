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

import nvdiffrast.torch as dr
import torch
import trimesh
from omegaconf import OmegaConf

from foundationpose.learning.datasets.h5_dataset import PoseRefinePairH5Dataset
from foundationpose.learning.models.refine_network import RefineNet
from foundationpose.pytorch3d.transforms import rotation_6d_to_matrix, so3_exp_map
from foundationpose.utils import egocentric_delta_pose_to_pose, make_mesh_tensors, transform_pts

from .predict_score import make_crop_data_batch

logger = logging.getLogger(__name__)

TensorMap = dict[str, torch.Tensor]
RasterizeContext = dr.RasterizeCudaContext | dr.RasterizeGLContext


class PoseRefinePredictor:
    """Wrapper around the pretrained pose-refinement network."""

    def __init__(self, checkpoints_dir: str | Path, amp: bool = True, device: str | torch.device = "cuda") -> None:
        self.amp = amp
        self.device = torch.device(device)
        self.run_name = "2023-10-28-18-33-37"

        model_name = "model_best.pth"
        weights_dir = Path(checkpoints_dir)
        run_dir = weights_dir / self.run_name
        ckpt_dir = run_dir / model_name

        self.cfg = OmegaConf.load(run_dir / "config.yml")

        ########## Defaults, to be backward compatible
        if "use_normal" not in self.cfg:
            self.cfg["use_normal"] = False
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
        if "normalize_xyz" not in self.cfg:
            self.cfg["normalize_xyz"] = False
        logger.info("[PoseRefinePredictor] self.cfg:\n%s", OmegaConf.to_yaml(self.cfg))

        self.dataset = PoseRefinePairH5Dataset(cfg=self.cfg, h5_file="", mode="test")
        self.model = RefineNet(
            use_batch_norm=self.cfg["use_BN"], rotation_representation=self.cfg["rot_rep"], c_in=self.cfg["c_in"]
        ).to(self.device)

        ckpt = torch.load(ckpt_dir)
        if "model" in ckpt:
            ckpt = ckpt["model"]
        self.model.load_state_dict(ckpt)
        self.model.to(self.device).eval()

        logger.info("[PoseRefinePredictor] Initialized using pretrained model from %s", ckpt_dir)

    @torch.inference_mode()
    def predict(  # noqa: PLR0912, PLR0915
        self,
        rgb: torch.Tensor,
        depth: torch.Tensor,
        K: torch.Tensor,
        ob_in_cams: torch.Tensor,
        xyz_map: torch.Tensor,
        normal_map: torch.Tensor | None = None,
        mesh: trimesh.Trimesh | None = None,
        mesh_tensors: TensorMap | None = None,
        glctx: RasterizeContext | None = None,
        mesh_diameter: float | None = None,
        iteration: int = 5,
        rgb_only: bool = False,
        renderer_batch_size: int = 512,
    ) -> torch.Tensor:
        """Refine candidate object poses for a single RGB-D observation."""
        bs = 1024
        mesh_centered = mesh
        crop_ratio = self.cfg["crop_ratio"]
        use_normal = self.cfg["use_normal"]
        trans_normalizer = self.cfg["trans_normalizer"]
        normalize_xyz = self.cfg["normalize_xyz"]
        rot_normalizer = self.cfg["rot_normalizer"]
        rot_rep = self.cfg["rot_rep"]
        trans_rep = self.cfg["trans_rep"]
        input_resize = self.cfg["input_resize"]

        # move to device/dtype
        rgb_tensor = rgb.to(device=self.device, dtype=torch.float32)
        depth_tensor = depth.to(device=self.device, dtype=torch.float32)
        K = K.to(device=self.device, dtype=torch.float32)
        B_in_cams = ob_in_cams.to(device=self.device, dtype=torch.float32)
        xyz_map_tensor = xyz_map.to(device=self.device, dtype=torch.float32)
        normal_map = normal_map.to(device=self.device, dtype=torch.float32) if normal_map is not None else None

        if not use_normal:
            normal_map = None
        if mesh_tensors is None:
            mesh_tensors = make_mesh_tensors(mesh_centered)

        if rgb_only:
            depth_tensor = torch.zeros_like(depth_tensor)
            xyz_map_tensor = torch.zeros_like(xyz_map_tensor)
            normal_map = None

        if not isinstance(trans_normalizer, float):
            trans_normalizer = torch.as_tensor(list(trans_normalizer), device=self.device, dtype=torch.float32).reshape(
                1, 3
            )

        for _ in range(iteration):
            pose_data = make_crop_data_batch(
                mesh=mesh_centered,
                mesh_diameter=mesh_diameter,
                ob_in_cams=B_in_cams,
                rgb=rgb_tensor,
                K=K,
                use_normal=use_normal,
                render_size=input_resize,
                crop_ratio=crop_ratio,
                dataset=self.dataset,
                xyz_map=xyz_map_tensor,
                normal_map=normal_map,
                glctx=glctx,
                mesh_tensors=mesh_tensors,
                device=self.device,
                renderer_batch_size=renderer_batch_size,
            )
            B_in_cams = []
            for b in range(0, pose_data.rgbAs.shape[0], bs):
                A = torch.cat([pose_data.rgbAs[b : b + bs], pose_data.xyz_mapAs[b : b + bs]], dim=1)
                B = torch.cat([pose_data.rgbBs[b : b + bs], pose_data.xyz_mapBs[b : b + bs]], dim=1)
                with torch.amp.autocast(self.device.type, enabled=self.amp):
                    output = self.model(A, B)
                if trans_rep == "tracknet":
                    if not normalize_xyz:
                        trans_delta = torch.tanh(output["trans"]) * trans_normalizer
                    else:
                        trans_delta = output["trans"]
                elif trans_rep == "deepim":
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
                    uv_pred_crop = uvA_crop + output["trans"][:, :2] * input_resize[0]
                    uv_pred = transform_pts(uv_pred_crop, pose_data.tf_to_crops[b : b + bs].inverse())
                    center_pred = torch.cat(
                        [uv_pred, torch.ones((len(rot_delta), 1), dtype=torch.float32, device=self.device)], dim=-1
                    )
                    center_pred = (
                        pose_data.Ks[b : b + bs].inverse() @ center_pred.reshape(len(rot_delta), 3, 1)
                    ).reshape(len(rot_delta), 3) * z_pred.reshape(len(rot_delta), 1)
                    trans_delta = center_pred - pose_data.poseA[b : b + bs][..., :3, 3]
                else:
                    trans_delta = output["trans"]

                if rot_rep == "axis_angle":
                    rot_mat_delta = torch.tanh(output["rot"]) * rot_normalizer
                    rot_mat_delta = so3_exp_map(rot_mat_delta).permute(0, 2, 1)
                elif rot_rep == "6d":
                    rot_mat_delta = rotation_6d_to_matrix(output["rot"]).permute(0, 2, 1)
                else:
                    raise RuntimeError

                if normalize_xyz:
                    trans_delta *= mesh_diameter / 2

                B_in_cams.append(
                    egocentric_delta_pose_to_pose(
                        pose_data.poseA[b : b + bs], trans_delta=trans_delta, rot_mat_delta=rot_mat_delta
                    )
                )
            B_in_cams = torch.cat(B_in_cams, dim=0).reshape(len(ob_in_cams), 4, 4)

        torch.cuda.empty_cache()
        return B_in_cams
