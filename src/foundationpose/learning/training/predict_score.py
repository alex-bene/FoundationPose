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
import numpy as np
import nvdiffrast.torch as dr
import torch
import trimesh
from omegaconf import DictConfig, OmegaConf

from foundationpose.learning.datasets.h5_dataset import ScoreMultiPairH5Dataset, TripletH5Dataset
from foundationpose.learning.datasets.pose_dataset import BatchPoseData
from foundationpose.learning.models.score_network import ScoreNetMultiPair
from foundationpose.Utils import compute_crop_window_tf_batch, make_mesh_tensors, nvdiffrast_render, transform_pts

logger = logging.getLogger(__name__)

TensorMap = dict[str, torch.Tensor]
ArrayTensor = np.ndarray | torch.Tensor
RasterizeContext = dr.RasterizeCudaContext | dr.RasterizeGLContext


@torch.no_grad()
def make_crop_data_batch(
    cfg: DictConfig,
    dataset: TripletH5Dataset,
    render_size: tuple[int, int],
    ob_in_cams: ArrayTensor,
    mesh: trimesh.Trimesh,
    rgb: ArrayTensor,
    depth: ArrayTensor,
    K: ArrayTensor,
    crop_ratio: float,
    normal_map: ArrayTensor | None = None,
    mesh_diameter: float | None = None,
    glctx: RasterizeContext | None = None,
    mesh_tensors: TensorMap | None = None,
) -> BatchPoseData:
    """Build a score-model batch from image crops and rendered views."""
    _ = normal_map
    logger.debug("Building score crop batch")
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
    poseAs = torch.as_tensor(ob_in_cams, dtype=torch.float, device="cuda")

    bs = 512
    rgb_rs = []
    depth_rs = []
    xyz_map_rs = []

    bbox2d_crop = torch.as_tensor(
        np.array([0, 0, cfg["input_resize"][0] - 1, cfg["input_resize"][1] - 1]).reshape(2, 2),
        device="cuda",
        dtype=torch.float,
    )
    bbox2d_ori = transform_pts(bbox2d_crop, tf_to_crops.inverse()[:, None]).reshape(-1, 4)

    for b in range(0, len(ob_in_cams), bs):
        extra = {}
        rgb_r, depth_r, _ = nvdiffrast_render(
            ob_in_cams=poseAs[b : b + bs],
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
        xyz_map_rs.append(extra["xyz_map"])

    rgb_rs = torch.cat(rgb_rs, dim=0).permute(0, 3, 1, 2) * 255
    depth_rs = torch.cat(depth_rs, dim=0).permute(0, 3, 1, 2)
    xyz_map_rs = torch.cat(xyz_map_rs, dim=0).permute(0, 3, 1, 2)  # (B,3,H,W)
    logger.debug("Rendered synthetic crops")

    rgbBs = kornia.geometry.transform.warp_perspective(
        torch.as_tensor(rgb, dtype=torch.float, device="cuda").permute(2, 0, 1)[None].expand(B, -1, -1, -1),
        tf_to_crops,
        dsize=render_size,
        mode="bilinear",
        align_corners=False,
    )
    depthBs = kornia.geometry.transform.warp_perspective(
        torch.as_tensor(depth, dtype=torch.float, device="cuda")[None, None].expand(B, -1, -1, -1),
        tf_to_crops,
        dsize=render_size,
        mode="nearest",
        align_corners=False,
    )
    if rgb_rs.shape[-2:] != cfg["input_resize"]:
        rgbAs = kornia.geometry.transform.warp_perspective(
            rgb_rs, tf_to_crops, dsize=render_size, mode="bilinear", align_corners=False
        )
        depthAs = kornia.geometry.transform.warp_perspective(
            depth_rs, tf_to_crops, dsize=render_size, mode="nearest", align_corners=False
        )
    else:
        rgbAs = rgb_rs
        depthAs = depth_rs

    if xyz_map_rs.shape[-2:] != cfg["input_resize"]:
        xyz_mapAs = kornia.geometry.transform.warp_perspective(
            xyz_map_rs, tf_to_crops, dsize=render_size, mode="nearest", align_corners=False
        )
    else:
        xyz_mapAs = xyz_map_rs

    normalAs = None
    normalBs = None

    Ks = torch.as_tensor(K, dtype=torch.float).reshape(1, 3, 3).expand(len(rgbAs), 3, 3)
    mesh_diameters = torch.ones((len(rgbAs)), dtype=torch.float, device="cuda") * mesh_diameter

    pose_data = BatchPoseData(
        rgbAs=rgbAs,
        rgbBs=rgbBs,
        depthAs=depthAs,
        depthBs=depthBs,
        normalAs=normalAs,
        normalBs=normalBs,
        poseA=poseAs,
        xyz_mapAs=xyz_mapAs,
        tf_to_crops=tf_to_crops,
        Ks=Ks,
        mesh_diameters=mesh_diameters,
    )
    pose_data = dataset.transform_batch(pose_data, H_ori=H, W_ori=W)

    logger.debug("Prepared score batch data")

    return pose_data


class ScorePredictor:
    """Wrapper around the pretrained score network used during inference."""

    def __init__(self, checkpoints_dir: str | Path, amp: bool = True, device: str = "cuda") -> None:
        self.amp = amp
        self.device = device
        self.run_name = "2024-01-11-20-02-45"

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
        if "use_BN" not in self.cfg:
            self.cfg["use_BN"] = False
        if "zfar" not in self.cfg:
            self.cfg["zfar"] = np.inf
        if "c_in" not in self.cfg:
            self.cfg["c_in"] = 4
        if "normalize_xyz" not in self.cfg:
            self.cfg["normalize_xyz"] = False
        if "crop_ratio" not in self.cfg or self.cfg["crop_ratio"] is None:
            self.cfg["crop_ratio"] = 1.2

        logger.info("self.cfg:\n%s", OmegaConf.to_yaml(self.cfg))

        self.dataset = ScoreMultiPairH5Dataset(cfg=self.cfg, mode="test", h5_file=None, max_num_key=1)
        self.model = ScoreNetMultiPair(use_batch_norm=self.cfg["use_BN"], c_in=self.cfg["c_in"]).cuda()

        logger.info("Using pretrained model from %s", ckpt_dir)
        ckpt = torch.load(ckpt_dir)
        if "model" in ckpt:
            ckpt = ckpt["model"]
        self.model.load_state_dict(ckpt)

        self.model.cuda().eval()
        logger.info("Initialized score predictor")

    @torch.inference_mode()
    def predict(
        self,
        rgb: ArrayTensor,
        depth: ArrayTensor,
        K: ArrayTensor,
        ob_in_cams: ArrayTensor,
        mesh: trimesh.Trimesh | None = None,
        mesh_tensors: TensorMap | None = None,
        glctx: RasterizeContext | None = None,
        mesh_diameter: float | None = None,
    ) -> torch.Tensor:
        """Score candidate object poses for a single RGB-D observation."""
        logger.debug("ob_in_cams shape: %s", ob_in_cams.shape)
        ob_in_cams = torch.as_tensor(ob_in_cams, dtype=torch.float, device="cuda")

        logger.debug("self.cfg.use_normal: %s", self.cfg.use_normal)
        logger.debug("Building cropped score inputs")

        if mesh_tensors is None:
            mesh_tensors = make_mesh_tensors(mesh)

        rgb = torch.as_tensor(rgb, device="cuda", dtype=torch.float)
        depth = torch.as_tensor(depth, device="cuda", dtype=torch.float)

        pose_data = make_crop_data_batch(
            self.cfg,
            self.dataset,
            self.cfg.input_resize,
            ob_in_cams,
            mesh,
            rgb,
            depth,
            K,
            crop_ratio=self.cfg["crop_ratio"],
            glctx=glctx,
            mesh_tensors=mesh_tensors,
            mesh_diameter=mesh_diameter,
        )

        def find_best_among_pairs(pose_data: BatchPoseData) -> tuple[torch.Tensor, torch.Tensor]:
            logger.debug("pose_data.rgbAs batch size: %s", pose_data.rgbAs.shape[0])
            ids = []
            scores = []
            bs = pose_data.rgbAs.shape[0]
            for b in range(0, pose_data.rgbAs.shape[0], bs):
                A = torch.cat(
                    [pose_data.rgbAs[b : b + bs].cuda(), pose_data.xyz_mapAs[b : b + bs].cuda()], dim=1
                ).float()
                B = torch.cat(
                    [pose_data.rgbBs[b : b + bs].cuda(), pose_data.xyz_mapBs[b : b + bs].cuda()], dim=1
                ).float()
                if pose_data.normalAs is not None:
                    A = torch.cat([A, pose_data.normalAs.cuda().float()], dim=1)
                    B = torch.cat([B, pose_data.normalBs.cuda().float()], dim=1)
                with torch.amp.autocast("cuda", enabled=self.amp):
                    output = self.model(A, B, L=len(A))
                scores_cur = output["score_logit"].float().reshape(-1)
                ids.append(scores_cur.argmax() + b)
                scores.append(scores_cur)
            ids = torch.stack(ids, dim=0).reshape(-1)
            scores = torch.cat(scores, dim=0).reshape(-1)
            return ids, scores

        pose_data_iter = pose_data
        global_ids = torch.arange(len(ob_in_cams), device="cuda", dtype=torch.long)
        scores_global = torch.zeros((len(ob_in_cams)), dtype=torch.float, device="cuda")

        while 1:
            ids, scores = find_best_among_pairs(pose_data_iter)
            if len(ids) == 1:
                scores_global[global_ids] = scores + 100
                break
            global_ids = global_ids[ids]
            pose_data_iter = pose_data.select_by_indices(global_ids)

        scores = scores_global

        logger.debug("Completed score forward pass")
        torch.cuda.empty_cache()

        return scores
