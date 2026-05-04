# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

"""Inference helpers for score-model training artifacts."""

import logging
from pathlib import Path

import kornia
import nvdiffrast.torch as dr
import torch
from omegaconf import OmegaConf

from foundationpose.learning.datasets.h5_dataset import ScoreMultiPairH5Dataset, TripletH5Dataset
from foundationpose.learning.datasets.pose_dataset import BatchPoseData
from foundationpose.learning.models.score_network import ScoreNetMultiPair
from foundationpose.utils import compute_crop_window_tf_batch, nvdiffrast_render, transform_pts

logger = logging.getLogger(__name__)

TensorMap = dict[str, torch.Tensor]
RasterizeContext = dr.RasterizeCudaContext | dr.RasterizeGLContext


@torch.inference_mode()
def make_crop_data_batch(
    mesh_diameter: float,
    ob_in_cams: torch.Tensor,
    image: torch.Tensor,
    intrinsics_px: torch.Tensor,
    use_normal: bool,
    render_size: tuple[int, int],
    crop_ratio: float,
    dataset: TripletH5Dataset,
    mesh_tensors: TensorMap,
    depth: torch.Tensor | None = None,
    xyz_map: torch.Tensor | None = None,
    glctx: RasterizeContext | None = None,
    renderer_batch_size: int = 512,
) -> BatchPoseData:
    """Build a score-model batch from image crops and rendered views."""
    B = len(ob_in_cams)
    H, W = image.shape[:2]
    method = "box_3d"

    tf_to_crops = compute_crop_window_tf_batch(
        poses=ob_in_cams,
        intrinsics_px=intrinsics_px,
        crop_ratio=crop_ratio,
        out_size=(render_size[1], render_size[0]),
        method=method,
        mesh_diameter=mesh_diameter,
    )

    bbox2d_crop = ob_in_cams.new_tensor([0, 0, render_size[0] - 1, render_size[1] - 1]).reshape(2, 2)
    bbox2d_ori = transform_pts(bbox2d_crop, tf_to_crops.inverse()[:, None]).reshape(-1, 4)

    image_rs, depth_rs, normal_rs, xyz_map_rs = [], [], [], []
    for b in range(0, B, renderer_batch_size):
        image_r, depth_r, normal_r, xyz_map_r = nvdiffrast_render(
            ob_in_cams=ob_in_cams[b : b + renderer_batch_size],
            intrinsics_px=intrinsics_px,
            H=H,
            W=W,
            context="cuda",
            get_normal=use_normal,
            glctx=glctx,
            mesh_tensors=mesh_tensors,
            output_size=render_size,
            bbox2d=bbox2d_ori[b : b + renderer_batch_size],
            use_light=True,
        )
        image_rs.append(image_r)
        depth_rs.append(depth_r[..., None])
        normal_rs.append(normal_r)
        xyz_map_rs.append(xyz_map_r)

    image_rs = torch.cat(image_rs, dim=0).permute(0, 3, 1, 2) * 255
    depth_rs = torch.cat(depth_rs, dim=0).permute(0, 3, 1, 2)
    xyz_map_rs = torch.cat(xyz_map_rs, dim=0).permute(0, 3, 1, 2)  # (B,3,H,W)
    if use_normal:
        normal_rs = torch.cat(normal_rs, dim=0).permute(0, 3, 1, 2)  # (B,3,H,W)

    ## RGB
    imageAs = image_rs
    if image_rs.shape[-2:] != render_size:
        imageAs = kornia.geometry.transform.warp_perspective(
            image_rs, tf_to_crops, dsize=render_size, mode="bilinear", align_corners=False
        )
    imageBs = kornia.geometry.transform.warp_perspective(
        image.permute(2, 0, 1)[None].expand(B, -1, -1, -1),
        tf_to_crops,
        dsize=render_size,
        mode="bilinear",
        align_corners=False,
    )

    ## Depth
    depthAs = depth_rs if depth is not None else None
    depthBs = None
    if depth is not None:
        if depth_rs.shape[-2:] != render_size:
            depthAs = kornia.geometry.transform.warp_perspective(
                depth_rs, tf_to_crops, dsize=render_size, mode="nearest", align_corners=False
            )
        depthBs = kornia.geometry.transform.warp_perspective(
            depth[None, None].expand(B, -1, -1, -1), tf_to_crops, dsize=render_size, mode="nearest", align_corners=False
        )

    ## XYZ Map
    xyz_mapAs = xyz_map_rs if xyz_map is not None else None
    xyz_mapBs = None
    if xyz_map is not None:
        if xyz_map_rs.shape[-2:] != render_size:
            xyz_mapAs = kornia.geometry.transform.warp_perspective(
                xyz_map_rs, tf_to_crops, dsize=render_size, mode="nearest", align_corners=False
            )
        xyz_mapBs = kornia.geometry.transform.warp_perspective(
            xyz_map.permute(2, 0, 1)[None].expand(B, -1, -1, -1),
            tf_to_crops,
            dsize=render_size,
            mode="nearest",
            align_corners=False,
        )  # (B,3,H,W)

    ## Normal
    normalAs = None
    normalBs = None
    if use_normal:
        if normal_rs.shape[-2:] != render_size:
            normalAs = kornia.geometry.transform.warp_perspective(
                normal_rs, tf_to_crops, dsize=render_size, mode="nearest", align_corners=False
            )
        normalBs = kornia.geometry.transform.warp_perspective(
            mesh_tensors["vnormals"].permute(2, 0, 1)[None].expand(B, -1, -1, -1),
            tf_to_crops,
            dsize=render_size,
            mode="nearest",
            align_corners=False,
        )

    pose_data = BatchPoseData(
        rgbAs=imageAs,
        rgbBs=imageBs,
        depthAs=depthAs,
        depthBs=depthBs,
        normalAs=normalAs,
        normalBs=normalBs,
        poseA=ob_in_cams,
        xyz_mapAs=xyz_mapAs,
        xyz_mapBs=xyz_mapBs,
        tf_to_crops=tf_to_crops,
        Ks=intrinsics_px.reshape(1, 3, 3).expand(B, 3, 3),
        mesh_diameters=ob_in_cams.new_ones(B) * mesh_diameter,
    )
    return dataset.transform_batch(pose_data, H_ori=H, W_ori=W)


class ScorePredictor:
    """Wrapper around the pretrained score network used during inference."""

    def __init__(self, checkpoints_dir: str | Path, amp: bool = True, device: str | torch.device = "cuda") -> None:
        self.amp = amp
        self.device = torch.device(device)
        self.run_name = "2024-01-11-20-02-45"

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
        if "normalize_xyz" not in self.cfg:
            self.cfg["normalize_xyz"] = False
        if "crop_ratio" not in self.cfg or self.cfg["crop_ratio"] is None:
            self.cfg["crop_ratio"] = 1.2

        logger.info("[ScorePredictor] self.cfg:\n%s", OmegaConf.to_yaml(self.cfg))

        self.dataset = ScoreMultiPairH5Dataset(cfg=self.cfg, mode="test", h5_file=None)
        self.model = ScoreNetMultiPair(use_batch_norm=self.cfg["use_BN"], c_in=self.cfg["c_in"]).to(self.device)

        ckpt = torch.load(ckpt_dir)
        if "model" in ckpt:
            ckpt = ckpt["model"]
        self.model.load_state_dict(ckpt)
        self.model.to(self.device).eval()

        logger.info("[ScorePredictor] Initialized using pretrained model from %s", ckpt_dir)

    @torch.inference_mode()
    def predict(
        self,
        image: torch.Tensor,
        depth: torch.Tensor,
        intrinsics_px: torch.Tensor,
        ob_in_cams: torch.Tensor,
        xyz_map: torch.Tensor,
        mesh_tensors: TensorMap,
        mesh_diameter: float,
        glctx: RasterizeContext | None = None,
        rgb_only: bool = False,
        renderer_batch_size: int = 512,
    ) -> torch.Tensor:
        """Score candidate object poses for a single RGB-D observation."""
        # move to device/dtype
        image = image.to(device=self.device, dtype=torch.float32)
        depth = depth.to(device=self.device, dtype=torch.float32)
        xyz_map = xyz_map.to(device=self.device, dtype=torch.float32)
        intrinsics_px = intrinsics_px.to(device=self.device, dtype=torch.float32)
        ob_in_cams = ob_in_cams.to(device=self.device, dtype=torch.float32)

        if rgb_only:
            depth = None
            xyz_map = None

        pose_data = make_crop_data_batch(
            mesh_diameter=mesh_diameter,
            ob_in_cams=ob_in_cams,
            image=image,
            intrinsics_px=intrinsics_px,
            use_normal=False,
            render_size=self.cfg["input_resize"],
            crop_ratio=self.cfg["crop_ratio"],
            dataset=self.dataset,
            depth=depth,
            xyz_map=xyz_map,
            glctx=glctx,
            mesh_tensors=mesh_tensors,
            renderer_batch_size=renderer_batch_size,
        )

        if rgb_only:
            pose_data.xyz_mapAs = torch.zeros_like(pose_data.rgbAs)
            pose_data.xyz_mapBs = torch.zeros_like(pose_data.rgbBs)

        def find_best_among_pairs(pose_data: BatchPoseData) -> tuple[torch.Tensor, torch.Tensor]:
            ids = []
            scores = []
            bs = pose_data.rgbAs.shape[0]
            for b in range(0, pose_data.rgbAs.shape[0], bs):
                A = torch.cat([pose_data.rgbAs[b : b + bs], pose_data.xyz_mapAs[b : b + bs]], dim=1)
                B = torch.cat([pose_data.rgbBs[b : b + bs], pose_data.xyz_mapBs[b : b + bs]], dim=1)
                if pose_data.normalAs is not None:
                    A = torch.cat([A, pose_data.normalAs], dim=1)
                    B = torch.cat([B, pose_data.normalBs], dim=1)
                with torch.amp.autocast(self.device.type, enabled=self.amp):
                    output = self.model(A, B, L=len(A))
                scores_cur = output["score_logit"].reshape(-1).to(dtype=torch.float32)
                ids.append(scores_cur.argmax() + b)
                scores.append(scores_cur)
            ids = torch.stack(ids, dim=0).reshape(-1)
            scores = torch.cat(scores, dim=0).reshape(-1)
            return ids, scores

        pose_data_iter = pose_data
        global_ids = torch.arange(len(ob_in_cams), device=self.device, dtype=torch.long)
        scores_global = torch.zeros((len(ob_in_cams)), dtype=torch.float32, device=self.device)

        while 1:
            ids, scores = find_best_among_pairs(pose_data_iter)
            if len(ids) == 1:
                scores_global[global_ids] = scores + 100
                break
            global_ids = global_ids[ids]
            pose_data_iter = pose_data.select_by_indices(global_ids)

        torch.cuda.empty_cache()
        return scores_global
