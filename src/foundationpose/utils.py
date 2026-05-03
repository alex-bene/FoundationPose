# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

"""Utility helpers for rendering, geometry, and depth-map processing."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Literal, TypeAlias

import cv2
import numpy as np
import nvdiffrast.torch as dr
import torch
import torch.nn.functional as F
import trimesh
import warp as wp

if TYPE_CHECKING:
    from collections.abc import Sequence

wp.init()

logger = logging.getLogger(__name__)

DeviceLike: TypeAlias = str | torch.device
MeshTensorMap: TypeAlias = dict[str, torch.Tensor]
ColorBgr: TypeAlias = tuple[int, int, int]
RasterizeContext: TypeAlias = dr.RasterizeCudaContext | dr.RasterizeGLContext

DEFAULT_LIGHT_DIR = np.array([0.0, 0.0, 1.0], dtype=float)
DEFAULT_LIGHT_POS = np.array([0.0, 0.0, 0.0], dtype=float)

glcam_in_cvcam = np.array([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]]).astype(float)


def make_mesh_tensors(
    mesh: trimesh.Trimesh, device: DeviceLike = "cuda", max_tex_size: int | None = None
) -> MeshTensorMap:
    """Convert a trimesh mesh into tensor buffers suitable for rendering."""
    mesh_tensors: MeshTensorMap = {}
    if isinstance(mesh.visual, trimesh.visual.texture.TextureVisuals):
        img = np.array(mesh.visual.material.image.convert("RGB"))
        img = img[..., :3]
        if max_tex_size is not None:
            max_size = max(img.shape[0], img.shape[1])
            if max_size > max_tex_size:
                scale = 1 / max_size * max_tex_size
                img = cv2.resize(img, fx=scale, fy=scale, dsize=None)
        mesh_tensors["tex"] = torch.as_tensor(img, device=device, dtype=torch.float32)[None] / 255.0
        mesh_tensors["uv_idx"] = torch.as_tensor(mesh.faces, device=device, dtype=torch.int32)  # nvdiffrast needs int32
        uv = torch.as_tensor(mesh.visual.uv, device=device, dtype=torch.float32)
        uv[:, 1] = 1 - uv[:, 1]
        mesh_tensors["uv"] = uv
    else:
        if mesh.visual.vertex_colors is None:
            logger.warning("mesh doesn't have vertex_colors, assigning a pure color")
            mesh.visual.vertex_colors = np.tile(np.array([128, 128, 128]).reshape(1, 3), (len(mesh.vertices), 1))
        mesh_tensors["vertex_color"] = (
            torch.as_tensor(mesh.visual.vertex_colors[..., :3], device=device, dtype=torch.float32) / 255.0
        )

    mesh_tensors.update(
        {
            "pos": torch.tensor(mesh.vertices, device=device, dtype=torch.float32),
            "faces": torch.tensor(mesh.faces, device=device, dtype=torch.int32),  # nvdiffrast needs int32
            "vnormals": torch.tensor(mesh.vertex_normals, device=device, dtype=torch.float32),
        }
    )
    return mesh_tensors


@torch.no_grad()
def nvdiffrast_render(  # noqa: PLR0912, PLR0915
    ob_in_cams: torch.Tensor,
    mesh_tensors: MeshTensorMap,
    intrinsics_px: torch.Tensor | None = None,
    H: int | None = None,
    W: int | None = None,
    glctx: RasterizeContext | None = None,
    context: Literal["cuda", "gl"] = "cuda",
    get_normal: bool = False,
    projection_mat: torch.Tensor | None = None,
    bbox2d: torch.Tensor | None = None,
    output_size: Sequence[int] | np.ndarray | None = None,
    use_light: bool = False,
    light_color: np.ndarray | torch.Tensor | None = None,
    light_dir: np.ndarray | None = None,
    light_pos: np.ndarray | None = None,
    w_ambient: float = 0.8,
    w_diffuse: float = 0.5,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
    """Render a mesh with nvdiffrast without gradient support.

    Args:
        intrinsics_px: Camera intrinsics with shape `(3, 3)`.
        H: Output image height.
        W: Output image width.
        ob_in_cams: Object poses in OpenCV camera coordinates with shape `(N, 4, 4)`.
        glctx: Optional pre-created rasterization context.
        context: Backend used when creating a rasterization context.
        get_normal: Whether to return an interpolated normal map.
        mesh_tensors: Optional prebuilt mesh tensor buffers.
        mesh: Mesh source used when `mesh_tensors` is not provided.
        projection_mat: Optional OpenGL projection matrix with shape `(4, 4)`.
        output_size: Optional `(height, width)` render size override.
        use_light: Whether to apply a simple diffuse lighting model.
        light_color: Optional RGB light color.
        bbox2d: Optional per-pose ROI boxes as `(umin, vmin, umax, vmax)`.
        light_dir: Optional light direction in camera space.
        light_pos: Optional light position in camera space.
        w_ambient: Ambient lighting weight.
        w_diffuse: Diffuse lighting weight.
    """
    device = ob_in_cams.device

    if glctx is None:
        if context == "gl":
            glctx = dr.RasterizeGLContext(device=device)
        elif "cuda" in str(context):
            glctx = dr.RasterizeCudaContext(device=device)
        else:
            raise NotImplementedError

    pos = mesh_tensors["pos"]
    vnormals = mesh_tensors["vnormals"]
    pos_idx = mesh_tensors["faces"]
    has_tex = "tex" in mesh_tensors

    if projection_mat is None:
        if intrinsics_px is None or H is None or W is None:
            msg = "`intrinsics_px`, `H`, and `W` are required when `projection_mat` is not provided."
            raise ValueError(msg)
        projection_mat = projection_matrix_from_intrinsics(intrinsics_px, height=H, width=W, znear=0.001, zfar=100)
    if output_size is None:
        if H is None or W is None:
            msg = "`H` and `W` are required when `output_size` is not provided."
            raise ValueError(msg)
        output_size = (H, W)
    if light_dir is None:
        light_dir = DEFAULT_LIGHT_DIR
    if light_pos is None:
        light_pos = DEFAULT_LIGHT_POS

    ob_in_glcams = torch.as_tensor(glcam_in_cvcam, device=device, dtype=torch.float32)[None] @ ob_in_cams
    projection_mat = projection_mat.reshape(-1, 4, 4)
    mtx = projection_mat @ ob_in_glcams

    pts_cam = transform_pts(pos, ob_in_cams)
    pos_homo = to_homo_torch(pos)
    pos_clip = (mtx[:, None] @ pos_homo[None, ..., None])[..., 0]
    if bbox2d is not None:
        if H is None or W is None:
            msg = "`H` and `W` are required when `bbox2d` is provided."
            raise ValueError(msg)
        left = bbox2d[:, 0]
        t = H - bbox2d[:, 1]
        r = bbox2d[:, 2]
        b = H - bbox2d[:, 3]
        tf = (
            torch.eye(4, dtype=torch.float32, device=device).reshape(1, 4, 4).expand(len(ob_in_cams), 4, 4).contiguous()
        )
        tf[:, 0, 0] = W / (r - left)
        tf[:, 1, 1] = H / (t - b)
        tf[:, 3, 0] = (W - r - left) / (r - left)
        tf[:, 3, 1] = (H - t - b) / (t - b)
        pos_clip = pos_clip @ tf
    rast_out, _ = dr.rasterize(glctx, pos_clip, pos_idx, resolution=output_size)
    xyz_map, _ = dr.interpolate(pts_cam, rast_out, pos_idx)
    depth = xyz_map[..., 2]
    if has_tex:
        texc, _ = dr.interpolate(mesh_tensors["uv"], rast_out, mesh_tensors["uv_idx"])
        color = dr.texture(mesh_tensors["tex"], texc, filter_mode="linear")
    else:
        color, _ = dr.interpolate(mesh_tensors["vertex_color"], rast_out, pos_idx)

    if use_light:
        get_normal = True
    if get_normal:
        vnormals_cam = transform_dirs(vnormals, ob_in_cams)
        normal_map, _ = dr.interpolate(vnormals_cam, rast_out, pos_idx)
        normal_map = F.normalize(normal_map, dim=-1)
        normal_map = torch.flip(normal_map, dims=[1])
    else:
        normal_map = None

    if use_light:
        if light_dir is not None:
            light_dir_neg = -torch.as_tensor(light_dir, dtype=torch.float32, device=device)
        else:
            light_dir_neg = torch.as_tensor(light_pos, dtype=torch.float32, device=device).reshape(1, 1, 3) - pts_cam
        diffuse_intensity = (
            (F.normalize(vnormals_cam, dim=-1) * F.normalize(light_dir_neg, dim=-1)).sum(dim=-1).clip(0, 1)[..., None]
        )
        diffuse_intensity_map, _ = dr.interpolate(diffuse_intensity, rast_out, pos_idx)  # (N_pose, H, W, 1)
        light_color = color if light_color is None else torch.as_tensor(light_color, device=device, dtype=torch.float32)
        color = color * w_ambient + diffuse_intensity_map * light_color * w_diffuse

    color = color.clip(0, 1)
    color = color * torch.clamp(rast_out[..., -1:], 0, 1)  # Mask out background using alpha
    color = torch.flip(color, dims=[1])  # Flip Y coordinates
    depth = torch.flip(depth, dims=[1])
    return color, depth, normal_map, torch.flip(xyz_map, dims=[1])


if wp is not None:

    @wp.kernel(enable_backward=False)
    def bilateral_filter_depth_kernel(  # noqa: PLR0912
        depth: wp.array(dtype=float, ndim=2),  # pyright: ignore reportInvalidType
        out: wp.array(dtype=float, ndim=2),  # pyright: ignore reportInvalidType
        radius: int,
        zfar: float,
        sigmaD: float,
        sigmaR: float,
    ) -> None:
        """Warp kernel for bilateral filtering a depth map."""
        h, w = wp.tid()
        H = depth.shape[0]
        W = depth.shape[1]
        if w >= W or h >= H:
            return
        out[h, w] = 0.0
        mean_depth = float(0)
        num_valid = int(0.0)
        for u in range(w - radius, w + radius + 1):
            if u < 0 or u >= W:
                continue
            for v in range(h - radius, h + radius + 1):
                if v < 0 or v >= H:
                    continue
                cur_depth = depth[v, u]
                if cur_depth >= 0.001 and cur_depth < zfar:
                    num_valid += 1
                    mean_depth += cur_depth
        if num_valid == 0:
            return
        mean_depth /= float(num_valid)

        depthCenter = depth[h, w]
        sum_weight = float(0)
        depth_sum = float(0)
        for u in range(w - radius, w + radius + 1):
            if u < 0 or u >= W:
                continue
            for v in range(h - radius, h + radius + 1):
                if v < 0 or v >= H:
                    continue
                cur_depth = depth[v, u]
                if cur_depth >= 0.001 and cur_depth < zfar and abs(cur_depth - mean_depth) < 0.01:
                    weight = wp.exp(
                        -float((u - w) * (u - w) + (h - v) * (h - v)) / (2.0 * sigmaD * sigmaD)
                        - (depthCenter - cur_depth) * (depthCenter - cur_depth) / (2.0 * sigmaR * sigmaR)
                    )
                    sum_weight += weight
                    depth_sum += weight * cur_depth
        if sum_weight > 0 and num_valid > 0:
            out[h, w] = depth_sum / sum_weight

    def bilateral_filter_depth(
        depth: torch.Tensor, radius: int = 2, zfar: float = 100, sigmaD: float = 2, sigmaR: float = 100000
    ) -> torch.Tensor:
        """Apply a bilateral filter to a depth map using Warp."""
        depth_wp = wp.from_torch(depth)
        out_wp = wp.zeros(depth.shape, dtype=float, device=str(depth.device))
        wp.launch(
            kernel=bilateral_filter_depth_kernel,
            device=str(depth.device),
            dim=[depth.shape[0], depth.shape[1]],
            inputs=[depth_wp, out_wp, radius, zfar, sigmaD, sigmaR],
        )
        return wp.to_torch(out_wp)

    @wp.kernel(enable_backward=False)
    def erode_depth_kernel(
        depth: wp.array(dtype=float, ndim=2),  # pyright: ignore reportInvalidType
        out: wp.array(dtype=float, ndim=2),  # pyright: ignore reportInvalidType
        radius: int,
        depth_diff_thres: float,
        ratio_thres: float,
        zfar: float,
    ) -> None:
        """Warp kernel that erodes depth discontinuities."""
        h, w = wp.tid()
        H = depth.shape[0]
        W = depth.shape[1]
        if w >= W or h >= H:
            return
        d_ori = depth[h, w]
        if d_ori < 0.001 or d_ori >= zfar:
            out[h, w] = 0.0
        bad_cnt = float(0)
        total = float(0)
        for u in range(w - radius, w + radius + 1):
            if u < 0 or u >= W:
                continue
            for v in range(h - radius, h + radius + 1):
                if v < 0 or v >= H:
                    continue
                cur_depth = depth[v, u]
                total += 1.0
                if cur_depth < 0.001 or cur_depth >= zfar or abs(cur_depth - d_ori) > depth_diff_thres:
                    bad_cnt += 1.0
        if bad_cnt / total > ratio_thres:
            out[h, w] = 0.0
        else:
            out[h, w] = d_ori

    def erode_depth(
        depth: torch.Tensor,
        radius: int = 2,
        depth_diff_thres: float = 0.001,
        ratio_thres: float = 0.8,
        zfar: float = 100,
    ) -> torch.Tensor:
        """Erode unstable depth pixels using local agreement."""
        depth_wp = wp.from_torch(depth)
        out_wp = wp.zeros(depth.shape, dtype=float, device=str(depth.device))
        wp.launch(
            kernel=erode_depth_kernel,
            device=str(depth.device),
            dim=[depth.shape[0], depth.shape[1]],
            inputs=[depth_wp, out_wp, radius, depth_diff_thres, ratio_thres, zfar],
        )
        return wp.to_torch(out_wp)


def depth2xyzmap(depth: torch.Tensor, intrinsics_px: torch.Tensor) -> torch.Tensor:
    """Project a depth map into an XYZ map using camera intrinsics."""
    invalid_mask = depth < 0.001
    H, W = depth.shape[:2]
    vs, us = torch.meshgrid(
        torch.arange(0, H, device=depth.device), torch.arange(0, W, device=depth.device), indexing="ij"
    )
    vs = vs.reshape(-1)
    us = us.reshape(-1)

    zs = depth[vs, us]
    xs = (us - intrinsics_px[0, 2]) * zs / intrinsics_px[0, 0]
    ys = (vs - intrinsics_px[1, 2]) * zs / intrinsics_px[1, 1]
    pts = torch.stack((xs.reshape(-1), ys.reshape(-1), zs.reshape(-1)), 1)  # (N,3)
    xyz_map = depth.new_zeros((H, W, 3))
    xyz_map[vs, us] = pts
    xyz_map[invalid_mask] = 0
    return xyz_map


def depth2xyzmap_batch(depths: torch.Tensor, Ks: torch.Tensor, zfar: float) -> torch.Tensor:
    """Project a batch of depth maps into XYZ maps."""
    bs = depths.shape[0]
    invalid_mask = (depths < 0.001) | (depths > zfar)
    H, W = depths.shape[-2:]
    vs, us = torch.meshgrid(
        torch.arange(0, H, device=depths.device), torch.arange(0, W, device=depths.device), indexing="ij"
    )
    vs = vs.reshape(-1).float()[None].expand(bs, -1)
    us = us.reshape(-1).float()[None].expand(bs, -1)
    zs = depths.reshape(bs, -1)
    Ks = Ks[:, None].expand(bs, zs.shape[-1], 3, 3)
    xs = (us - Ks[..., 0, 2]) * zs / Ks[..., 0, 0]  # (B,N)
    ys = (vs - Ks[..., 1, 2]) * zs / Ks[..., 1, 1]
    pts = torch.stack([xs, ys, zs], dim=-1)  # (B,N,3)
    xyz_maps = pts.reshape(bs, H, W, 3)
    xyz_maps[invalid_mask] = 0
    return xyz_maps


def sample_views_icosphere(n_views: int, subdivisions: int | None = None, radius: float = 1) -> np.ndarray:
    """Sample camera poses from an icosphere."""
    if subdivisions is not None:
        mesh = trimesh.creation.icosphere(subdivisions=subdivisions, radius=radius)
    else:
        subdivision = 1
        while 1:
            mesh = trimesh.creation.icosphere(subdivisions=subdivision, radius=radius)
            if mesh.vertices.shape[0] >= n_views:
                break
            subdivision += 1
    cam_in_obs = np.tile(np.eye(4)[None], (len(mesh.vertices), 1, 1))
    cam_in_obs[:, :3, 3] = mesh.vertices
    up = np.array([0, 0, 1])
    z_axis = -cam_in_obs[:, :3, 3]  # (N,3)
    z_axis /= np.linalg.norm(z_axis, axis=-1).reshape(-1, 1)
    x_axis = np.cross(up.reshape(1, 3), z_axis)
    invalid = (x_axis == 0).all(axis=-1)
    x_axis[invalid] = [1, 0, 0]
    x_axis /= np.linalg.norm(x_axis, axis=-1).reshape(-1, 1)
    y_axis = np.cross(z_axis, x_axis)
    y_axis /= np.linalg.norm(y_axis, axis=-1).reshape(-1, 1)
    cam_in_obs[:, :3, 0] = x_axis
    cam_in_obs[:, :3, 1] = y_axis
    cam_in_obs[:, :3, 2] = z_axis
    return cam_in_obs


def to_homo_torch(pts: torch.Tensor) -> torch.Tensor:
    """Append a homogeneous coordinate to the last dimension of a tensor."""
    ones = torch.ones((*pts.shape[:-1], 1), dtype=torch.float32, device=pts.device)
    return torch.cat((pts, ones), dim=-1)


def transform_pts(pts: torch.Tensor, tf: torch.Tensor) -> torch.Tensor:
    """Transform 2D or 3D points with a homogeneous transform."""
    if len(tf.shape) >= 3 and tf.shape[-3] != pts.shape[-2]:
        tf = tf[..., None, :, :]
    return (tf[..., :-1, :-1] @ pts[..., None] + tf[..., :-1, -1:])[..., 0]


def transform_dirs(dirs: torch.Tensor, tf: torch.Tensor) -> torch.Tensor:
    """Rotate direction vectors using the linear part of a transform."""
    if len(tf.shape) >= 3 and tf.shape[-3] != dirs.shape[-2]:
        tf = tf[..., None, :, :]
    return (tf[..., :3, :3] @ dirs[..., None])[..., 0]


def compute_mesh_diameter(pts: torch.Tensor, n_sample: int | None = 1000, chunk_size: int = 4096) -> float:
    """Estimate the diameter of a point cloud."""
    if len(pts) < 2:
        return 0.0

    if n_sample is not None and n_sample < len(pts):
        ids = torch.randperm(len(pts), device=pts.device)[:n_sample]
        pts = pts[ids]

    max_distance = pts.new_tensor(0.0)
    chunk_size = min(len(pts), chunk_size)
    for start in range(0, len(pts), chunk_size):
        dists = torch.cdist(pts[start : start + chunk_size], pts)
        max_distance = torch.maximum(max_distance, dists.amax())
    return max_distance.item()


def compute_crop_window_tf_batch(
    pts: torch.Tensor | None = None,
    poses: torch.Tensor | None = None,
    intrinsics_px: torch.Tensor | None = None,
    crop_ratio: float = 1.2,
    out_size: Sequence[int] | torch.Tensor | np.ndarray | None = None,
    method: str = "min_box",
    mesh_diameter: float | None = None,
) -> torch.Tensor:
    """Compute crop transforms for a batch of object poses."""

    def compute_tf_batch(
        left: torch.Tensor, right: torch.Tensor, top: torch.Tensor, bottom: torch.Tensor
    ) -> torch.Tensor:
        B = len(left)
        left = left.round()
        right = right.round()
        top = top.round()
        bottom = bottom.round()

        if out_size is None:
            msg = "`out_size` is required."
            raise ValueError(msg)
        tf = torch.eye(3, device=left.device, dtype=left.dtype)[None].expand(B, -1, -1).contiguous()
        tf[:, 0, 2] = -left
        tf[:, 1, 2] = -top
        new_tf = torch.eye(3, device=left.device, dtype=left.dtype)[None].expand(B, -1, -1).contiguous()
        new_tf[:, 0, 0] = out_size[0] / (right - left)
        new_tf[:, 1, 1] = out_size[1] / (bottom - top)
        return new_tf @ tf

    if poses is None or intrinsics_px is None:
        msg = "`poses` and `intrinsics_px` are required."
        raise ValueError(msg)
    B = len(poses)
    if method != "box_3d":
        msg_0 = f"Unknown method: {method}"
        raise RuntimeError(msg_0)

    if mesh_diameter is None:
        msg = "`mesh_diameter` is required for `box_3d` crops."
        raise ValueError(msg)
    radius = mesh_diameter * crop_ratio / 2
    offsets = torch.tensor(
        [0, 0, 0, radius, 0, 0, -radius, 0, 0, 0, radius, 0, 0, -radius, 0], device=poses.device, dtype=poses.dtype
    ).reshape(-1, 3)
    pts = poses[:, :3, 3].reshape(-1, 1, 3) + offsets.reshape(1, -1, 3)
    projected = (intrinsics_px @ pts.reshape(-1, 3).T).T
    uvs = projected[:, :2] / projected[:, 2:3]
    uvs = uvs.reshape(B, -1, 2)
    center = uvs[:, 0]  # (B,2)
    radius = torch.abs(uvs - center.reshape(-1, 1, 2)).reshape(B, -1).max(axis=-1)[0].reshape(-1)  # (B)
    left = center[:, 0] - radius
    right = center[:, 0] + radius
    top = center[:, 1] - radius
    bottom = center[:, 1] + radius
    return compute_tf_batch(left, right, top, bottom)


def projection_matrix_from_intrinsics(
    intrinsics_px: torch.Tensor,
    height: int,
    width: int,
    znear: float,
    zfar: float,
    window_coords: Literal["y_up", "y_down"] = "y_down",
) -> torch.Tensor:
    """Conversion of Hartley-Zisserman intrinsic matrix to OpenGL proj. matrix.

    Ref:
    1) https://strawlab.org/2011/11/05/augmented-reality-with-OpenGL
    2) https://github.com/strawlab/opengl-hz/blob/master/src/calib_test_utils.py

    :param intrinsics_px: 3x3 ndarray with the intrinsic camera matrix.
    :param x0 The X coordinate of the camera image origin (typically 0).
    :param y0: The Y coordinate of the camera image origin (typically 0).
    :param w: Image width.
    :param h: Image height.
    :param nc: Near clipping plane.
    :param fc: Far clipping plane.
    :param window_coords: 'y_up' or 'y_down'.
    :return: 4x4 ndarray with the OpenGL projection matrix.
    """
    x0 = 0
    y0 = 0
    w = width
    h = height
    nc = znear
    fc = zfar

    depth = float(fc - nc)
    q = -(fc + nc) / depth
    qn = -2 * (fc * nc) / depth

    # Draw our images upside down, so that all the pixel-based coordinate
    # systems are the same.
    K = intrinsics_px
    if window_coords == "y_up":
        proj = K.new_tensor(
            [
                [2 * K[0, 0] / w, -2 * K[0, 1] / w, (-2 * K[0, 2] + w + 2 * x0) / w, 0],
                [0, -2 * K[1, 1] / h, (-2 * K[1, 2] + h + 2 * y0) / h, 0],
                [0, 0, q, qn],  # Sets near and far planes (glPerspective).
                [0, 0, -1, 0],
            ]
        )

    # Draw the images upright and modify the projection matrix so that OpenGL
    # will generate window coords that compensate for the flipped image coords.
    elif window_coords == "y_down":
        proj = K.new_tensor(
            [
                [2 * K[0, 0] / w, -2 * K[0, 1] / w, (-2 * K[0, 2] + w + 2 * x0) / w, 0],
                [0, 2 * K[1, 1] / h, (2 * K[1, 2] - h + 2 * y0) / h, 0],
                [0, 0, q, qn],  # Sets near and far planes (glPerspective).
                [0, 0, -1, 0],
            ]
        )
    else:
        raise NotImplementedError

    return proj


def egocentric_delta_pose_to_pose(
    A_in_cam: torch.Tensor, trans_delta: torch.Tensor, rot_mat_delta: torch.Tensor
) -> torch.Tensor:
    """Apply egocentric pose deltas to a batch of camera-frame poses."""
    B_in_cam = (
        torch.eye(4, dtype=A_in_cam.dtype, device=A_in_cam.device)[None].expand(len(A_in_cam), -1, -1).contiguous()
    )
    B_in_cam[:, :3, 3] = A_in_cam[:, :3, 3] + trans_delta
    B_in_cam[:, :3, :3] = rot_mat_delta @ A_in_cam[:, :3, :3]
    return B_in_cam


def matrix_distance(R1: np.ndarray, R2: np.ndarray) -> float:
    """Return the angular distance between two rotation matrices."""
    R_rel = R1.T @ R2
    trace = np.trace(R_rel)
    cos_theta = (trace - 1) / 2
    cos_theta = np.clip(cos_theta, -1.0, 1.0)
    return float(np.arccos(cos_theta))


def cluster_poses(
    angle_diff: float, dist_diff: float, poses_in: Sequence[np.ndarray], symmetry_tfs: Sequence[np.ndarray]
) -> list[np.ndarray]:
    """Cluster similar poses under translational and rotational thresholds."""
    poses_out = [poses_in[0]]
    radian_thres = angle_diff / 180.0 * np.pi

    for i in range(1, len(poses_in)):
        isnew = True
        cur_pose = poses_in[i]
        for cluster in poses_out:
            t0 = cluster[0:3, 3]
            t1 = cur_pose[0:3, 3]

            if np.linalg.norm(t0 - t1) >= dist_diff:
                continue

            for tf in symmetry_tfs:
                cur_pose_tmp = np.dot(cur_pose, tf)
                rot_diff = matrix_distance(cur_pose_tmp[0:3, 0:3], cluster[0:3, 0:3])
                if rot_diff < radian_thres:
                    isnew = False
                    break

            if not isnew:
                break

        if isnew:
            poses_out.append(poses_in[i])

    return poses_out
