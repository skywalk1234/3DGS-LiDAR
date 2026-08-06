#----------------------------------------------------------------#
# ReconDrive - sweep-time intermediate evaluation                 #
#                                                                  #
# For each two-keyframe window (frame 0 + frame N), enumerate the  #
# raw nuScenes sweeps strictly between the two keyframe timestamps #
# (per camera ~12 Hz, LIDAR_TOP ~20 Hz), render the model's        #
# gaussians at every sweep time via the per-gaussian vehicle flow  #
# (continuous-time temporal propagation), and compare against the  #
# sweep ground truth.                                              #
#                                                                  #
# Metrics:                                                         #
#   camera: PSNR / SSIM / LPIPS per (sample, cam, sweep)           #
#   lidar : depth_l2 / depth_median_l2 / delta_1-3 /               #
#           intensity_rmse / ray_drop_acc / chamfer_distance       #
#                                                                  #
# The sweep GT is fully independent of the model input (frame 0    #
# and frame N only), so this is a legitimate novel-time evaluation.#
#----------------------------------------------------------------#

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image as PILImage
from PIL import Image

# ------------------------------------------------------------------
# Path setup (project root + models dir, mirroring trainer.py)
# ------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
for entry in (PROJECT_ROOT, PROJECT_ROOT / "models"):
    if str(entry) not in sys.path:
        sys.path.insert(0, str(entry))

import yaml

from gsplat.rendering import rasterization, lidar_rasterization
from dataset.data_util import (
    build_lidar_range_image,
    LIDAR_NUM_RINGS,
    LIDAR_AZIMUTH_RESOLUTION,
)
from dataset.vggt3dgs_scene_data_module import VGGT3DGS_SceneDataModule
from models.recondrive_model import ReconDrive_LITModelModule

CAMERA_CHANNELS = (
    "CAM_FRONT",
    "CAM_FRONT_LEFT",
    "CAM_FRONT_RIGHT",
    "CAM_BACK_LEFT",
    "CAM_BACK_RIGHT",
    "CAM_BACK",
)
LIDAR_CHANNEL = "LIDAR_TOP"


# ------------------------------------------------------------------
# Quaternion / transform helpers (nuScenes stores [w, x, y, z])
# ------------------------------------------------------------------
def quat_wxyz_to_matrix(wxyz: Sequence[float]) -> np.ndarray:
    w, x, y, z = (float(v) for v in wxyz)
    norm = float(np.sqrt(w * w + x * x + y * y + z * z))
    if norm <= 0:
        raise ValueError("zero-norm quaternion")
    w, x, y, z = w / norm, x / norm, y / norm, z / norm
    return np.asarray(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def pose_matrix(translation: Sequence[float], rotation: Sequence[float]) -> np.ndarray:
    result = np.eye(4, dtype=np.float64)
    result[:3, :3] = quat_wxyz_to_matrix(rotation)
    result[:3, 3] = np.asarray(translation, dtype=np.float64)
    return result


def to_device(data: Any, device: torch.device) -> Any:
    """Move tensors to device; keep vehicle annotation payloads untouched."""
    if isinstance(data, dict):
        return {
            key: (value if key == "vehicle_annotations" else to_device(value, device))
            for key, value in data.items()
        }
    if isinstance(data, (list, tuple)):
        return type(data)(to_device(item, device) for item in data)
    if torch.is_tensor(data):
        return data.to(device)
    return data


def _collate_window(window: Mapping[str, Any]) -> dict[str, Any]:
    """Add the batch dimension to a single window, mirroring the behaviour of
    vggt4dgs_dataset.custom_collate_fn for a batch of size 1 (the model
    expects tensors shaped [1, views, C, H, W])."""
    result: dict[str, Any] = {}
    for key, value in window.items():
        if isinstance(key, tuple):
            try:
                result[key] = torch.stack([value], dim=0)
            except Exception:
                result[key] = [value]
        elif isinstance(key, str) and (
            key == "vehicle_annotations" or key.startswith("vehicle_annotations_frame_")
        ):
            result[key] = [value]
        elif key in ("all_dict", "context_frames"):
            result[key] = _collate_window(value)
        elif isinstance(key, str) and key == "lidar":
            result[key] = {}
            for lidar_key, lidar_value in value.items():
                try:
                    result[key][lidar_key] = torch.stack([lidar_value], dim=0)
                except Exception:
                    result[key][lidar_key] = [lidar_value]
        else:
            try:
                result[key] = torch.utils.data.default_collate([value])
            except Exception:
                result[key] = [value]
    return result


# ------------------------------------------------------------------
# Raw nuScenes table access (independent of the devkit)
# ------------------------------------------------------------------
class NuscTables:
    """Lightweight indexed access to the raw nuScenes JSON tables."""

    def __init__(self, data_root: str | Path, version: str = "v1.0-mini"):
        meta_root = Path(data_root) / version
        self.data_root = Path(data_root)
        self.version = version
        self.sample_data = {
            record["token"]: record
            for record in json.loads((meta_root / "sample_data.json").read_text())
        }
        self.sample = {
            record["token"]: record
            for record in json.loads((meta_root / "sample.json").read_text())
        }
        self.scene = {
            record["token"]: record
            for record in json.loads((meta_root / "scene.json").read_text())
        }
        self.ego_pose = {
            record["token"]: record
            for record in json.loads((meta_root / "ego_pose.json").read_text())
        }
        self.calibrated_sensor = {
            record["token"]: record
            for record in json.loads((meta_root / "calibrated_sensor.json").read_text())
        }
        self.sensor = {
            record["token"]: record
            for record in json.loads((meta_root / "sensor.json").read_text())
        }
        self._channel_by_calibration = {
            token: self.sensor[record["sensor_token"]]["channel"]
            for token, record in self.calibrated_sensor.items()
        }
        # Keyframe sample_data per (sample_token, channel).
        # Note: sample["data"] is NOT stored in the raw JSON; the devkit
        # synthesizes it, so we index keyframes ourselves.
        self.keyframe_by_sample_channel: dict[str, dict[str, dict[str, Any]]] = {}
        for record in self.sample_data.values():
            if not record.get("is_key_frame"):
                continue
            self.keyframe_by_sample_channel.setdefault(
                record["sample_token"], {}
            )[self.channel_of(record)] = record

    def channel_of(self, sample_data_record: Mapping[str, Any]) -> str:
        calib_token = sample_data_record["calibrated_sensor_token"]
        return self._channel_by_calibration[calib_token]

    def keyframe_for(self, sample_token: str, channel: str) -> dict[str, Any]:
        return self.keyframe_by_sample_channel[sample_token][channel]

    def sample_data_by_sample_token(self, channel: str) -> list[dict[str, Any]]:
        return [
            record
            for record in self.sample_data.values()
            if self.channel_of(record) == channel
        ]


# ------------------------------------------------------------------
# Window sweep context: one two-keyframe window + its sweeps
# ------------------------------------------------------------------
class SweepContext:
    """Enumerate per-sensor sweeps strictly between two keyframes."""

    def __init__(
        self,
        tables: NuscTables,
        frame0_sample_token: str,
        frameN_sample_token: str,
    ) -> None:
        self.tables = tables
        self.frame0_token = frame0_sample_token
        self.frameN_token = frameN_sample_token
        sample0 = tables.sample[frame0_sample_token]
        sampleN = tables.sample[frameN_sample_token]
        # keyframe sample_data per channel (sensor -> keyframe record)
        self.keyframes: dict[str, dict[str, Any]] = {}
        for channel in (*CAMERA_CHANNELS, LIDAR_CHANNEL):
            sd0 = tables.keyframe_for(frame0_sample_token, channel)
            sdN = tables.keyframe_for(frameN_sample_token, channel)
            self.keyframes[channel] = {"start": sd0, "end": sdN}

    def _sweeps(self, channel: str) -> list[dict[str, Any]]:
        start_ts = int(self.keyframes[channel]["start"]["timestamp"])
        end_ts = int(self.keyframes[channel]["end"]["timestamp"])
        if end_ts <= start_ts:
            return []
        out = []
        for record in self.tables.sample_data.values():
            if self.tables.channel_of(record) != channel:
                continue
            if record.get("is_key_frame"):
                continue
            timestamp = int(record["timestamp"])
            if start_ts < timestamp < end_ts:
                out.append(record)
        out.sort(key=lambda record: int(record["timestamp"]))
        return out

    def camera_sweeps(self, camera: str) -> list[dict[str, Any]]:
        return self._sweeps(camera)

    def lidar_sweeps(self) -> list[dict[str, Any]]:
        return self._sweeps(LIDAR_CHANNEL)

    def keyframe_timestamps(self, channel: str) -> tuple[int, int]:
        return (
            int(self.keyframes[channel]["start"]["timestamp"]),
            int(self.keyframes[channel]["end"]["timestamp"]),
        )

    def keyframe_ego_pose(self, channel: str) -> np.ndarray:
        """ego0 -> world using the frame-0 keyframe's own ego pose (per sensor)."""
        record = self.keyframes[channel]["start"]
        pose = self.tables.ego_pose[record["ego_pose_token"]]
        return pose_matrix(pose["translation"], pose["rotation"])

    def sweep_pose_in_ego0(
        self, sweep_record: Mapping[str, Any], channel: str
    ) -> np.ndarray:
        """Sensor -> ego0 pose of a sweep capture (4x4)."""
        ego0_to_world = self.keyframe_ego_pose(channel)
        sweep_ego = self.tables.ego_pose[sweep_record["ego_pose_token"]]
        sweep_ego_to_world = pose_matrix(sweep_ego["translation"], sweep_ego["rotation"])
        calibration = self.tables.calibrated_sensor[
            sweep_record["calibrated_sensor_token"]
        ]
        sensor_to_ego = pose_matrix(
            calibration["translation"], calibration["rotation"]
        )
        return np.linalg.inv(ego0_to_world) @ sweep_ego_to_world @ sensor_to_ego


# ------------------------------------------------------------------
# Gaussian motion to an arbitrary intermediate time
# ------------------------------------------------------------------
def move_gaussians_to_t(
    recontrast_data: Mapping[str, Any],
    t_seconds: float,
    span_seconds: float,
) -> torch.Tensor:
    """Linear per-gaussian motion (vehicle flow), mirroring render_splating_imgs.

    xyz layout: [B, N, 3] with the first half = frame-0 views and the second
    half = frame-N views (already aligned to ego_0 via xyz_transformed).
    """
    xyz = recontrast_data.get("xyz_transformed", recontrast_data["xyz"])
    flow = recontrast_data["forward_flow"]
    if t_seconds is None:
        return xyz
    t = float(np.clip(t_seconds, 0.0, float(span_seconds)))
    xyz_t = xyz.clone()
    mid_point = xyz_t.shape[1] // 2
    xyz_t[:, :mid_point] += flow[:, :mid_point] * t
    xyz_t[:, mid_point:] -= flow[:, mid_point:] * (float(span_seconds) - t)
    return xyz_t


# ------------------------------------------------------------------
# Camera sweep rendering
# ------------------------------------------------------------------
def load_sweep_image(data_root: Path, sweep_record: Mapping[str, Any]) -> np.ndarray:
    path = data_root / sweep_record["filename"]
    with PILImage.open(path) as image:
        return np.asarray(image.convert("RGB"), dtype=np.uint8)


def render_sweep_camera(
    model: ReconDrive_LITModelModule,
    recontrast_data: Mapping[str, Any],
    sweep_record: Mapping[str, Any],
    sweep_pose_ego0: np.ndarray,
    *,
    t_seconds: float,
    span_seconds: float,
    tables: NuscTables,
) -> dict[str, torch.Tensor]:
    """Render the gaussians (already moved to the sweep time) from the sweep
    camera.  All gaussians are used (both context frames), like the novel-view
    branch of render_splating_imgs.
    """
    device = recontrast_data["xyz"].device
    calibration = tables.calibrated_sensor[sweep_record["calibrated_sensor_token"]]

    cam_pose_ego0 = torch.as_tensor(
        sweep_pose_ego0, dtype=torch.float32, device=device
    )
    viewmat = torch.linalg.inv(cam_pose_ego0).unsqueeze(0)  # [1, 4, 4]

    native_w = int(sweep_record["width"])
    native_h = int(sweep_record["height"])
    render_h, render_w = model.render_height, model.render_width
    intrinsics = np.asarray(calibration["camera_intrinsic"], dtype=np.float64)
    k_scaled = np.eye(3, dtype=np.float64)
    k_scaled[0, 0] = intrinsics[0, 0] * (render_w / native_w)
    k_scaled[1, 1] = intrinsics[1, 1] * (render_h / native_h)
    k_scaled[0, 2] = intrinsics[0, 2] * (render_w / native_w)
    k_scaled[1, 2] = intrinsics[1, 2] * (render_h / native_h)
    Ks = torch.as_tensor(k_scaled, dtype=torch.float32, device=device).unsqueeze(0)

    xyz = move_gaussians_to_t(recontrast_data, t_seconds, span_seconds)
    if "rot_maps_transformed" in recontrast_data and getattr(
        model, "translate_3dgs", False
    ):
        rot = recontrast_data["rot_maps_transformed"]
        sh = recontrast_data["sh_maps_transformed"]
    else:
        rot = recontrast_data["rot_maps"]
        sh = recontrast_data["sh_maps"]

    colors, alpha, _ = rasterization(
        xyz[0],
        rot[0],
        recontrast_data["scale_maps"][0],
        recontrast_data["opacity_maps"][0].squeeze(-1),
        sh[0],
        velocities=None,
        viewmats=viewmat,
        Ks=Ks,
        width=render_w,
        height=render_h,
        sh_degree=model.sh_degree,
        render_mode="RGB",
    )
    rgb = colors[..., :3].permute(0, 3, 1, 2)  # [1, 3, H, W]
    return {"image": rgb.clamp(0, 1)}


# ------------------------------------------------------------------
# LiDAR sweep rendering
# ------------------------------------------------------------------
def render_sweep_lidar(
    model: ReconDrive_LITModelModule,
    recontrast_data: Mapping[str, Any],
    sweep_record: Mapping[str, Any],
    data_root: Path,
    *,
    t_seconds: float,
    span_seconds: float,
    tables: NuscTables,
    ego0_to_world: np.ndarray,
) -> dict[str, Any]:
    """Render the gaussians (moved to the sweep time) onto the sweep's own
    physical 32x3600 firing grid, in the sweep's ego frame."""
    device = recontrast_data["xyz"].device

    points = np.fromfile(
        data_root / sweep_record["filename"], dtype=np.float32
    ).reshape(-1, 5)
    calibration = tables.calibrated_sensor[sweep_record["calibrated_sensor_token"]]
    lidar_to_ego = pose_matrix(calibration["translation"], calibration["rotation"])
    sweep_ego = tables.ego_pose[sweep_record["ego_pose_token"]]
    sweep_ego_to_world = pose_matrix(sweep_ego["translation"], sweep_ego["rotation"])
    # viewmat must map ego_0 -> lidar (like the model's render_lidar, which
    # uses ego_to_lidar = inv(lidar_to_ego) for the frame-0 grid).  For a
    # sweep the lidar sits at the sweep ego pose, so the full chain is:
    #   lidar -> ego_sweep -> world -> ego_0 , then invert to get ego_0 -> lidar.
    lidar_to_ego0 = (
        np.linalg.inv(ego0_to_world) @ sweep_ego_to_world @ lidar_to_ego
    ).astype(np.float32)
    lidar_to_ego0_t = (
        torch.from_numpy(lidar_to_ego0).float().to(device)
    )
    viewmat = (
        torch.from_numpy(np.linalg.inv(lidar_to_ego0)).float().to(device).unsqueeze(0)
    )

    raster_pts, gt_depth, gt_intensity, gt_ray_drop, el_boundaries = (
        build_lidar_range_image(
            points, lidar_to_ego, sweep_ego_to_world, sweep_record["timestamp"]
        )
    )
    raster_pts = torch.from_numpy(raster_pts).float().to(device)
    el_boundaries_tensor = torch.from_numpy(np.ascontiguousarray(el_boundaries)).float().to(device)
    gt_depth_t = torch.from_numpy(np.ascontiguousarray(gt_depth)).float().to(device)
    gt_intensity_t = torch.from_numpy(np.ascontiguousarray(gt_intensity)).float().to(device)
    gt_ray_drop_t = torch.from_numpy(np.ascontiguousarray(gt_ray_drop)).float().to(device)

    xyz = move_gaussians_to_t(recontrast_data, t_seconds, span_seconds)
    if "rot_maps_transformed" in recontrast_data and getattr(
        model, "translate_3dgs", False
    ):
        rot = recontrast_data["rot_maps_transformed"]
    else:
        rot = recontrast_data["rot_maps"]

    render, alpha, _, _ = lidar_rasterization(
        means=xyz[0],
        quats=rot[0],
        scales=recontrast_data["scale_maps"][0],
        opacities=recontrast_data["opacity_maps"][0].squeeze(-1),
        lidar_features=recontrast_data["lidar_feat_maps"][0].unsqueeze(0),
        velocities=None,
        viewmats=viewmat,
        raster_pts=raster_pts,
        tile_elevation_boundaries=el_boundaries_tensor,
        n_elevation_channels=int(LIDAR_NUM_RINGS),
        azimuth_resolution=float(LIDAR_AZIMUTH_RESOLUTION),
        near_plane=0.2,
        far_plane=300,
        compute_alpha_sum_until_points=False,
    )
    pred_depth = render[..., -1:]
    features = render[..., :-1]
    rp_deg = torch.deg2rad(raster_pts[..., :2])
    ray_dir = torch.cat(
        [
            torch.cos(rp_deg[..., 0:1]) * torch.cos(rp_deg[..., 1:2]),
            torch.sin(rp_deg[..., 0:1]) * torch.cos(rp_deg[..., 1:2]),
            torch.sin(rp_deg[..., 1:2]),
        ],
        dim=-1,
    )
    intensity, ray_drop_logits = model.lidar_decoder(features, ray_dir)
    if os.environ.get("DEBUG_LIDAR"):
        gt_valid = gt_depth_t > 2.5
        pred_valid = pred_depth > 0
        joint = pred_valid & gt_valid
        print(
            f"[DEBUG] sweep ts={sweep_record['timestamp']} t={t_seconds:.2f}s "
            f"gt_valid={int(gt_valid.sum().item())} pred_valid={int(pred_valid.sum().item())} "
            f"joint_valid={int(joint.sum().item())} "
            f"gt_depth_mean={float(gt_depth_t[gt_valid].mean().item()):.2f} "
            f"pred_depth_mean(joint)={float(pred_depth[joint].mean().item()):.2f} "
            f"viewmat_trans={viewmat[0, :3, 3].tolist()}"
        )
    return {
        "depth": pred_depth,
        "intensity": intensity.sigmoid(),
        "ray_drop_logits": ray_drop_logits,
        "gt_depth": gt_depth_t,
        "gt_intensity": gt_intensity_t,
        "gt_ray_drop": gt_ray_drop_t,
        "viewmat": viewmat,
        "lidar_to_ego0": lidar_to_ego0_t,
        "raster_pts": raster_pts,
        "points": points,
        "lidar_to_ego": lidar_to_ego,
    }


# ------------------------------------------------------------------
# Metrics
# ------------------------------------------------------------------
def compute_lidar_sweep_metrics(pred: Mapping[str, Any]) -> dict[str, float]:
    def _squeeze(value: torch.Tensor) -> torch.Tensor:
        return value[0] if value.dim() == 4 else value

    gt_depth = _squeeze(pred["gt_depth"])
    pred_depth = _squeeze(pred["depth"])
    gt_intensity = _squeeze(pred["gt_intensity"])
    pred_intensity = _squeeze(pred["intensity"])
    gt_ray_drop = _squeeze(pred["gt_ray_drop"])
    pred_ray_drop = _squeeze(pred["ray_drop_logits"].sigmoid())
    raster_pts = _squeeze(pred["raster_pts"])

    valid = gt_depth > 2.5
    metrics: dict[str, float] = {}
    if not bool(valid.any()):
        return {
            "depth_l2": float("nan"),
            "depth_median_l2": float("nan"),
            "delta_1": float("nan"),
            "delta_2": float("nan"),
            "delta_3": float("nan"),
            "intensity_rmse": float("nan"),
            "ray_drop_acc": float("nan"),
            "chamfer_distance": float("nan"),
        }
    depth_sq = ((pred_depth[valid] - gt_depth[valid]) ** 2)
    metrics["depth_l2"] = float(depth_sq.mean().item())
    metrics["depth_median_l2"] = float(depth_sq.median().item())
    pred_d = pred_depth[valid]
    gt_d = gt_depth[valid]
    depth_ratio = torch.max(pred_d / gt_d, gt_d / pred_d)
    metrics["delta_1"] = float((depth_ratio < 1.25).float().mean().item())
    metrics["delta_2"] = float((depth_ratio < 1.25**2).float().mean().item())
    metrics["delta_3"] = float((depth_ratio < 1.25**3).float().mean().item())
    metrics["intensity_rmse"] = float(
        ((pred_intensity[valid] - gt_intensity[valid]) ** 2).mean().sqrt().item()
    )
    metrics["ray_drop_acc"] = float(
        ((pred_ray_drop > 0.5) == (gt_ray_drop > 0.5)).float().mean().item()
    )

    # Chamfer distance on 3D points (subsampled to keep cdist tractable).
    az_rad = torch.deg2rad(raster_pts[..., 0:1])
    el_rad = torch.deg2rad(raster_pts[..., 1:2])
    cos_el = torch.cos(el_rad)
    gt_d = gt_depth.expand_as(az_rad)
    pd_d = pred_depth.expand_as(az_rad)
    x_gt = gt_d * cos_el * torch.cos(az_rad)
    y_gt = gt_d * cos_el * torch.sin(az_rad)
    z_gt = gt_d * torch.sin(el_rad)
    x_pd = pd_d * cos_el * torch.cos(az_rad)
    y_pd = pd_d * cos_el * torch.sin(az_rad)
    z_pd = pd_d * torch.sin(el_rad)
    pts_gt = torch.stack([x_gt, y_gt, z_gt], dim=-1)
    pts_pd = torch.stack([x_pd, y_pd, z_pd], dim=-1)
    valid_flat = valid.squeeze(-1)
    pts_gt_v = pts_gt[valid_flat]
    pts_pd_v = pts_pd[valid_flat]
    count = pts_gt_v.shape[0]
    max_pts = 2048
    if count > max_pts:
        idx = torch.randperm(count, device=pts_gt_v.device)[:max_pts]
        pts_gt_v = pts_gt_v[idx]
        pts_pd_v = pts_pd_v[idx]
    if pts_gt_v.shape[0] > 0:
        dist_g2p = torch.cdist(pts_gt_v, pts_pd_v).min(dim=1).values.mean().item()
        dist_p2g = torch.cdist(pts_pd_v, pts_gt_v).min(dim=1).values.mean().item()
        metrics["chamfer_distance"] = float(dist_g2p + dist_p2g)
    else:
        metrics["chamfer_distance"] = float("nan")
    return metrics


# ------------------------------------------------------------------
# Main evaluation loop
# ------------------------------------------------------------------
def run_sweep_evaluation(
    model,
    scene_batch,
    tables: NuscTables,
    *,
    data_root: Path,
    device: torch.device,
    frame_skip: int,
    context_span: int,
    scene_idx: int,
    save_renders: bool,
    output_dir: Path,
    sanity_zero: bool = False,
    save_views: bool = False,
) -> dict[str, Any]:
    dataset = scene_batch["dataset"]
    scene_length = scene_batch["scene_length"]
    sample_indices = list(range(0, scene_length, frame_skip))
    scene_name = scene_batch["scene_name"]

    camera_samples: list[dict[str, Any]] = []
    lidar_samples: list[dict[str, Any]] = []
    model.eval()

    with torch.no_grad():
        for sample_idx in sample_indices:
            window = dataset.__getitem__(sample_idx, scene_idx)
            if not window.get("context_frames"):
                continue
            window = _collate_window(window)
            window = to_device(window, device)
            model.set_normal_params(window)
            # re-extract the two keyframe sample tokens from the dataset tables
            frame0_token = dataset.scenes_data[scene_idx][sample_idx]
            frameN_token = frame0_token
            current = tables.sample[frame0_token]
            for _ in range(context_span):
                nxt = current.get("next")
                if nxt:
                    current = tables.sample[nxt]
                    frameN_token = current["token"]
                else:
                    frameN_token = None
                    break
            if frameN_token is None:
                continue

            recontrast = model.get_recontrast_data(window)
            span_seconds = float(getattr(model, "time_delta", 0.5))

            ctx = SweepContext(tables, frame0_token, frameN_token)

            # ---- camera sweeps ----
            for cam_idx, cam in enumerate(CAMERA_CHANNELS):
                kf0_ts, kfN_ts = ctx.keyframe_timestamps(cam)
                span_cam = (kfN_ts - kf0_ts) / 1e6 if kfN_ts > kf0_ts else span_seconds
                for sweep in ctx.camera_sweeps(cam):
                    t_seconds = (int(sweep["timestamp"]) - kf0_ts) / 1e6
                    sweep_pose = ctx.sweep_pose_in_ego0(sweep, cam)
                    pred = render_sweep_camera(
                        model,
                        recontrast,
                        sweep,
                        sweep_pose,
                        t_seconds=t_seconds,
                        span_seconds=span_cam,
                        tables=tables,
                    )["image"]
                    gt_np = np.ascontiguousarray(load_sweep_image(data_root, sweep))
                    gt = (
                        torch.from_numpy(gt_np)
                        .permute(2, 0, 1)
                        .unsqueeze(0)
                        .float()
                        .to(device)
                        / 255.0
                    )
                    gt = F.interpolate(
                        gt,
                        size=(model.render_height, model.render_width),
                        mode="bilinear",
                        align_corners=False,
                    )
                    pred_eval = pred.clamp(0, 1)
                    gt_eval = gt.clamp(0, 1)
                    sample = {
                        "sample_idx": int(sample_idx),
                        "camera": cam,
                        "cam_idx": int(cam_idx),
                        "sweep_token": str(sweep["token"]),
                        "sweep_timestamp_us": int(sweep["timestamp"]),
                        "t_seconds": float(t_seconds),
                        "t_normalized": float(t_seconds / max(span_cam, 1e-9)),
                        "psnr": float(
                            model.compute_psnr(gt_eval, pred_eval).mean().item()
                        ),
                        "ssim": float(
                            model.compute_ssim(gt_eval, pred_eval).mean().item()
                        ),
                        "lpips": float(
                            model.compute_lpips(gt_eval, pred_eval).mean().item()
                        ),
                    }
                    camera_samples.append(sample)
                    if save_renders:
                        _save_sweep_render(
                            output_dir, scene_name, sample_idx, cam, sweep, pred_eval, gt_eval
                        )
                    if save_views:
                        _save_sweep_gt_views(
                            output_dir, scene_name, sample_idx, sweep,
                            pred_eval, gt_eval, cam, cam_idx,
                        )

            # ---- lidar sweeps ----
            lid_kf0_ts, lid_kfN_ts = ctx.keyframe_timestamps(LIDAR_CHANNEL)
            span_lid = (
                (lid_kfN_ts - lid_kf0_ts) / 1e6 if lid_kfN_ts > lid_kf0_ts else span_seconds
            )
            lidar_ego0_to_world = ctx.keyframe_ego_pose(LIDAR_CHANNEL)
            lidar_sweep_data: list[tuple[dict[str, Any], dict[str, Any]]] = []
            camera_sweep_records: dict[str, list[dict[str, Any]]] = {
                cam: list(ctx.camera_sweeps(cam)) for cam in CAMERA_CHANNELS
            }
            for sweep in ctx.lidar_sweeps():
                t_seconds = (int(sweep["timestamp"]) - lid_kf0_ts) / 1e6
                pred = render_sweep_lidar(
                    model,
                    recontrast,
                    sweep,
                    data_root,
                    t_seconds=t_seconds,
                    span_seconds=span_lid,
                    tables=tables,
                    ego0_to_world=lidar_ego0_to_world,
                )
                metrics = compute_lidar_sweep_metrics(pred)
                sample = {
                    "sample_idx": int(sample_idx),
                    "sweep_token": str(sweep["token"]),
                    "sweep_timestamp_us": int(sweep["timestamp"]),
                    "t_seconds": float(t_seconds),
                    "t_normalized": float(t_seconds / max(span_lid, 1e-9)),
                    **metrics,
                }
                lidar_samples.append(sample)
                if save_views:
                    lidar_sweep_data.append((dict(sweep), pred))

            # ---- save lidar PLY + lidar_cam per sweep ----
            if save_views:
                for sweep_record, lidar_pred in lidar_sweep_data:
                    _save_sweep_lidar_ply(
                        output_dir, scene_name, sample_idx, sweep_record, lidar_pred,
                    )
                    _save_sweep_lidar_cam(
                        lidar_pred,
                        camera_sweep_records,
                        lidar_timestamp_us=int(sweep_record["timestamp"]),
                        data_root=data_root,
                        output_dir=output_dir,
                        scene_name=scene_name,
                        sample_idx=sample_idx,
                        tables=tables,
                        render_h=model.render_height,
                        render_w=model.render_width,
                    )

            # ---- sanity check: render frame-0 keyframes (t=0) with the same
            # pipeline, to compare against the model's own frame-0 baseline ----
            if sanity_zero:
                for cam_idx, cam in enumerate(CAMERA_CHANNELS):
                    kf = ctx.keyframes[cam]["start"]
                    pose = ctx.sweep_pose_in_ego0(kf, cam)  # == c2e_extr at t=0
                    pred = render_sweep_camera(
                        model,
                        recontrast,
                        kf,
                        pose,
                        t_seconds=0.0,
                        span_seconds=span_seconds,
                        tables=tables,
                    )["image"]
                    gt_np = np.ascontiguousarray(load_sweep_image(data_root, kf))
                    gt = (
                        torch.from_numpy(gt_np)
                        .permute(2, 0, 1)
                        .unsqueeze(0)
                        .float()
                        .to(device)
                        / 255.0
                    )
                    gt = F.interpolate(
                        gt,
                        size=(model.render_height, model.render_width),
                        mode="bilinear",
                        align_corners=False,
                    )
                    camera_samples.append(
                        {
                            "sample_idx": int(sample_idx),
                            "camera": cam,
                            "cam_idx": int(cam_idx),
                            "sweep_token": "sanity_frame0",
                            "sweep_timestamp_us": int(kf["timestamp"]),
                            "t_seconds": 0.0,
                            "t_normalized": 0.0,
                            "psnr": float(
                                model.compute_psnr(gt.clamp(0, 1), pred.clamp(0, 1))
                                .mean()
                                .item()
                            ),
                            "ssim": float(
                                model.compute_ssim(gt.clamp(0, 1), pred.clamp(0, 1))
                                .mean()
                                .item()
                            ),
                            "lpips": float(
                                model.compute_lpips(gt.clamp(0, 1), pred.clamp(0, 1))
                                .mean()
                                .item()
                            ),
                            "sanity_zero": True,
                        }
                    )
                kf_lidar = ctx.keyframes[LIDAR_CHANNEL]["start"]
                pred_lidar = render_sweep_lidar(
                    model,
                    recontrast,
                    kf_lidar,
                    data_root,
                    t_seconds=0.0,
                    span_seconds=span_seconds,
                    tables=tables,
                    ego0_to_world=lidar_ego0_to_world,
                )
                lidar_samples.append(
                    {
                        "sample_idx": int(sample_idx),
                        "sweep_token": "sanity_frame0_lidar",
                        "sweep_timestamp_us": int(kf_lidar["timestamp"]),
                        "t_seconds": 0.0,
                        "t_normalized": 0.0,
                        **compute_lidar_sweep_metrics(pred_lidar),
                        "sanity_zero": True,
                    }
                )

    result = {
        "scene_idx": int(scene_idx),
        "scene_name": scene_name,
        "camera_sweeps": camera_samples,
        "lidar_sweeps": lidar_samples,
    }
    return result


def _save_sweep_render(
    output_dir: Path,
    scene_name: str,
    sample_idx: int,
    cam: str,
    sweep: Mapping[str, Any],
    pred: torch.Tensor,
    gt: torch.Tensor,
) -> None:
    folder = (
        output_dir
        / scene_name
        / f"sample_{int(sample_idx):04d}"
        / "sweeps"
        / cam
        / str(sweep["timestamp"])
    )
    folder.mkdir(parents=True, exist_ok=True)
    for name, tensor in (("pred.png", pred), ("gt.png", gt)):
        img = tensor[0].detach().cpu().numpy().transpose(1, 2, 0)
        img = np.clip(img * 255.0, 0, 255).astype(np.uint8)
        PILImage.fromarray(img).save(folder / name)


# ------------------------------------------------------------------
#  Save helpers: gt_views / lidar / lidar_cam (matching inference.py)
# ------------------------------------------------------------------
def _save_sweep_gt_views(
    output_dir: Path,
    scene_name: str,
    sample_idx: int,
    sweep_record: Mapping[str, Any],
    pred: torch.Tensor,
    gt: torch.Tensor,
    cam: str,
    cam_idx: int,
) -> None:
    """Save pred/gt camera images in gt_views/ format (same as inference.py)."""
    folder = (
        output_dir
        / scene_name
        / f"sample_{int(sample_idx):04d}"
        / "camera"
        / cam
        / str(sweep_record["timestamp"])
        / "gt_views"
    )
    folder.mkdir(parents=True, exist_ok=True)
    for name, tensor in (("pred.png", pred), ("gt.png", gt)):
        img = tensor[0].detach().cpu().numpy().transpose(1, 2, 0)
        img = np.clip(img * 255.0, 0, 255).astype(np.uint8)
        PILImage.fromarray(img).save(folder / name)


def _save_sweep_lidar_ply(
    output_dir: Path,
    scene_name: str,
    sample_idx: int,
    sweep_record: Mapping[str, Any],
    lidar_pred: Mapping[str, Any],
    min_depth: float = 2.5,
) -> list[str]:
    """Save GT/pred LiDAR point clouds as PLY, same format as inference.py."""
    folder = (
        output_dir
        / scene_name
        / f"sample_{int(sample_idx):04d}"
        / "lidar"
        / str(sweep_record["timestamp"])
        / "lidar"
    )
    folder.mkdir(parents=True, exist_ok=True)

    raster_pts = lidar_pred["raster_pts"]  # [B, H, W, 4]
    az_rad = torch.deg2rad(raster_pts[..., 0:1])
    el_rad = torch.deg2rad(raster_pts[..., 1:2])
    cos_el = torch.cos(el_rad)
    lidar_to_ego0 = lidar_pred["lidar_to_ego0"]  # [4, 4]

    def _ego_points(depth: torch.Tensor) -> torch.Tensor:
        pts_lidar = torch.cat([
            depth * cos_el * torch.cos(az_rad),
            depth * cos_el * torch.sin(az_rad),
            depth * torch.sin(el_rad),
        ], dim=-1)  # [B, H, W, 3]
        ones = torch.ones_like(pts_lidar[..., :1])
        pts_h = torch.cat([pts_lidar, ones], dim=-1)
        l2e = lidar_to_ego0.reshape(1, 4, 4).to(pts_h.device)
        return torch.matmul(pts_h, l2e.transpose(1, 2))[..., :3]

    def _write_ply(path: str, pts: torch.Tensor, depth: torch.Tensor,
                   intensity: torch.Tensor, valid_extra=None) -> int:
        pts_np = pts[0].reshape(-1, 3).detach().cpu().numpy()
        d_np = depth[0].reshape(-1).detach().cpu().numpy()
        i_np = intensity[0].reshape(-1).detach().cpu().numpy()
        valid = d_np > min_depth
        if valid_extra is not None:
            valid = valid & valid_extra
        pts_np, d_np, i_np = pts_np[valid], d_np[valid], i_np[valid]
        n = pts_np.shape[0]
        with open(path, "w") as f:
            f.write("ply\n")
            f.write("format ascii 1.0\n")
            f.write(f"element vertex {n}\n")
            f.write("property float x\n")
            f.write("property float y\n")
            f.write("property float z\n")
            f.write("property float intensity\n")
            f.write("property float depth\n")
            f.write("end_header\n")
            for j in range(n):
                f.write(
                    f"{pts_np[j, 0]:.4f} {pts_np[j, 1]:.4f} {pts_np[j, 2]:.4f} "
                    f"{i_np[j]:.4f} {d_np[j]:.4f}\n"
                )
        return n

    saved = []
    gt_pts = _ego_points(lidar_pred["gt_depth"])
    gt_path = str(folder / "gt_points.ply")
    _write_ply(gt_path, gt_pts, lidar_pred["gt_depth"], lidar_pred["gt_intensity"])
    saved.append(gt_path)

    pred_pts = _ego_points(lidar_pred["depth"])
    pred_path = str(folder / "pred_points.ply")
    _write_ply(pred_path, pred_pts, lidar_pred["depth"], lidar_pred["intensity"])
    saved.append(pred_path)

    gt_valid = (lidar_pred["gt_depth"][0].reshape(-1).detach().cpu().numpy() > min_depth)
    pred_mask_path = str(folder / "pred_points_mask.ply")
    _write_ply(pred_mask_path, pred_pts, lidar_pred["depth"], lidar_pred["intensity"],
               valid_extra=gt_valid)
    saved.append(pred_mask_path)

    return saved


def _save_sweep_lidar_cam(
    lidar_pred: Mapping[str, Any],
    camera_sweep_records: dict[str, list[dict[str, Any]]],
    *,
    lidar_timestamp_us: int,
    data_root: Path,
    output_dir: Path,
    scene_name: str,
    sample_idx: int,
    tables: NuscTables,
    render_h: int,
    render_w: int,
    max_depth: float = 80.0,
    point_radius: int = 1,
    alpha: float = 0.8,
) -> None:
    """Overlay LiDAR sweep points onto nearby camera sweep GT images.

    For each camera channel, finds the sweep nearest to the lidar timestamp,
    loads its GT image, and projects GT/pred LiDAR points onto it.
    """
    folder = (
        output_dir
        / scene_name
        / f"sample_{int(sample_idx):04d}"
        / "lidar"
        / str(lidar_timestamp_us)
        / "lidar_cam"
    )
    folder.mkdir(parents=True, exist_ok=True)

    # Build 3D points in LiDAR frame from range image
    raster_pts = lidar_pred["raster_pts"]  # [B, H, W, 4]
    az_rad = torch.deg2rad(raster_pts[..., 0:1])
    el_rad = torch.deg2rad(raster_pts[..., 1:2])
    cos_el = torch.cos(el_rad)
    gt_depth = lidar_pred["gt_depth"]
    pred_depth = lidar_pred["depth"]
    gt_pts_3d = torch.cat([
        gt_depth * cos_el * torch.cos(az_rad),
        gt_depth * cos_el * torch.sin(az_rad),
        gt_depth * torch.sin(el_rad),
    ], dim=-1)
    pred_pts_3d = torch.cat([
        pred_depth * cos_el * torch.cos(az_rad),
        pred_depth * cos_el * torch.sin(az_rad),
        pred_depth * torch.sin(el_rad),
    ], dim=-1)

    device = gt_depth.device
    lidar_to_ego0 = lidar_pred["lidar_to_ego0"].reshape(1, 4, 4)

    for cam_idx, cam in enumerate(CAMERA_CHANNELS):
        camera_sweeps = camera_sweep_records.get(cam, [])
        if not camera_sweeps:
            continue
        # Find camera sweep closest to this lidar timestamp
        nearest = min(camera_sweeps, key=lambda s: abs(int(s["timestamp"]) - lidar_timestamp_us))

        # Load GT image
        gt_np = np.ascontiguousarray(load_sweep_image(data_root, nearest))
        gt = (
            torch.from_numpy(gt_np)
            .permute(2, 0, 1)
            .unsqueeze(0)
            .float()
            .to(device)
            / 255.0
        )
        gt = F.interpolate(
            gt, size=(render_h, render_w), mode="bilinear", align_corners=False,
        )
        gt_np_img = gt[0].permute(1, 2, 0).clamp(0, 1).cpu().numpy()

        # Camera pose at the lidar time (using lidar's ego, camera's calibration)
        sweep_ego = tables.ego_pose[nearest["ego_pose_token"]]
        sweep_ego_to_world = pose_matrix(sweep_ego["translation"], sweep_ego["rotation"])
        calibration = tables.calibrated_sensor[nearest["calibrated_sensor_token"]]
        camera_to_ego = pose_matrix(calibration["translation"], calibration["rotation"])
        # ego0_to_world from the nearest camera sweep's context is not available
        # here; derive ego0 from the lidar sweep: work in ego_sweep frame instead.
        camera_to_ego_sweep = np.linalg.inv(sweep_ego_to_world) @ sweep_ego_to_world @ camera_to_ego
        e2c_extr = torch.from_numpy(np.linalg.inv(camera_to_ego_sweep)).float().to(device).unsqueeze(0)

        # Intrinsics
        intrinsics = np.asarray(calibration["camera_intrinsic"], dtype=np.float64)
        native_w, native_h = int(nearest["width"]), int(nearest["height"])
        k_scaled = np.eye(3, dtype=np.float64)
        k_scaled[0, 0] = intrinsics[0, 0] * (render_w / native_w)
        k_scaled[1, 1] = intrinsics[1, 1] * (render_h / native_h)
        k_scaled[0, 2] = intrinsics[0, 2] * (render_w / native_w)
        k_scaled[1, 2] = intrinsics[1, 2] * (render_h / native_h)
        K = torch.as_tensor(k_scaled, dtype=torch.float32, device=device)

        def _project(pts_3d: torch.Tensor, e2c: torch.Tensor, K_mat: torch.Tensor):
            pts_flat = pts_3d[0].reshape(-1, 3)
            e2c_4x4 = e2c[0]
            l2e_4x4 = lidar_to_ego0[0]
            cam_t_lidar = torch.mm(e2c_4x4, l2e_4x4)
            ones = torch.ones(pts_flat.shape[0], 1, device=device, dtype=pts_flat.dtype)
            pts_h = torch.cat([pts_flat, ones], dim=-1)
            pts_cam = torch.mm(pts_h, cam_t_lidar.T)[:, :3]
            front = pts_cam[:, 2] > 0
            pts_front = pts_cam[front]
            if pts_front.shape[0] == 0:
                return np.zeros((0, 2)), np.zeros(0)
            uv_h = torch.mm(pts_front, K_mat.T)
            uv_h[:, :2] /= uv_h[:, 2:3]
            u, v, z = uv_h[:, 0], uv_h[:, 1], uv_h[:, 2]
            in_bounds = (u >= 0) & (u < render_w) & (v >= 0) & (v < render_h)
            res = torch.stack([u[in_bounds], v[in_bounds]], dim=-1)
            return res.cpu().numpy(), z[in_bounds].cpu().numpy()

        # Only keep points with depth > 2.5m
        gt_valid = gt_depth > 2.5  # [B, H, W, 1] -> broadcasts with [B, H, W, 3]
        pred_valid = pred_depth > 2.5

        gt_uv, gt_z = _project(gt_pts_3d * gt_valid.float(), e2c_extr, K)
        pred_uv, pred_z = _project(pred_pts_3d * pred_valid.float(), e2c_extr, K)

        def _make_overlay(base: np.ndarray, uv: np.ndarray, z_vals: np.ndarray, path: str):
            overlay = base.copy()
            h, w = overlay.shape[:2]
            if len(uv) == 0:
                Image.fromarray((overlay * 255).astype(np.uint8)).save(path)
                return
            depth_norm = np.clip(z_vals / max_depth, 0, 1)
            colors = np.zeros((len(z_vals), 3))
            colors[:, 0] = depth_norm
            colors[:, 1] = 1.0 - np.abs(depth_norm - 0.5) * 2
            colors[:, 2] = 1.0 - depth_norm
            colors = np.clip(colors, 0, 1)
            u_c = np.round(uv[:, 0]).astype(int)
            v_c = np.round(uv[:, 1]).astype(int)
            order = np.argsort(-z_vals)
            u_c, v_c, colors = u_c[order], v_c[order], colors[order]
            for i in range(len(u_c)):
                col = colors[i]
                for dy in range(-point_radius, point_radius + 1):
                    for dx in range(-point_radius, point_radius + 1):
                        if dx * dx + dy * dy <= point_radius * point_radius:
                            ud, vd = u_c[i] + dx, v_c[i] + dy
                            if 0 <= ud < w and 0 <= vd < h:
                                overlay[vd, ud] = (1 - alpha) * overlay[vd, ud] + alpha * col
            Image.fromarray((np.clip(overlay, 0, 1) * 255).astype(np.uint8)).save(path)

        _make_overlay(gt_np_img, gt_uv, gt_z, str(folder / f"cam_{cam_idx}_gt_lidar_overlay.png"))
        _make_overlay(gt_np_img, pred_uv, pred_z, str(folder / f"cam_{cam_idx}_pred_lidar_overlay.png"))


def _mean_std(values: Sequence[float]) -> dict[str, float]:
    if not values:
        return {"mean": float("nan"), "std": float("nan"), "count": 0}
    arr = np.asarray(values, dtype=np.float64)
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return {"mean": float("nan"), "std": float("nan"), "count": len(values)}
    return {
        "mean": float(finite.mean()),
        "std": float(finite.std()),
        "count": int(len(values)),
    }


def aggregate_scene_metrics(scene_result: Mapping[str, Any]) -> dict[str, Any]:
    cam = [item for item in scene_result["camera_sweeps"] if not item.get("sanity_zero")]
    lid = [item for item in scene_result["lidar_sweeps"] if not item.get("sanity_zero")]
    aggregate = {
        "scene_name": scene_result["scene_name"],
        "camera_sweep_count": len(cam),
        "lidar_sweep_count": len(lid),
    }
    for key in ("psnr", "ssim", "lpips"):
        aggregate[f"camera_{key}"] = _mean_std([float(item[key]) for item in cam])
    for key in (
        "depth_l2",
        "depth_median_l2",
        "delta_1",
        "delta_2",
        "delta_3",
        "intensity_rmse",
        "ray_drop_acc",
        "chamfer_distance",
    ):
        aggregate[f"lidar_{key}"] = _mean_std([float(item[key]) for item in lid])
    return aggregate


def main() -> int:
    parser = argparse.ArgumentParser(description="ReconDrive sweep-time evaluation")
    parser.add_argument("--cfg_path", type=str, required=True)
    parser.add_argument("--restore_ckpt", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--scene", type=str, default=None)
    parser.add_argument("--max_scenes", type=int, default=None)
    parser.add_argument("--frame_skip", type=int, default=6)
    parser.add_argument("--save_renders", action="store_true")
    parser.add_argument(
        "--sanity_zero", action="store_true",
        help="also render frame-0 keyframes (t=0) for baseline comparison",
    )
    parser.add_argument(
        "--save_views", action="store_true",
        help="save gt_views/, lidar/ PLY, and lidar_cam/ overlays (like inference.py)",
    )
    args = parser.parse_args()

    with open(args.cfg_path) as handle:
        config = yaml.load(handle, Loader=yaml.FullLoader)

    config["model_cfg"]["batch_size"] = 1
    config["data_cfg"]["batch_size"] = 1
    if "context_span" in config["data_cfg"]:
        config["model_cfg"]["context_span"] = config["data_cfg"]["context_span"]
    if "nuscenes_version" in config["data_cfg"]:
        config["model_cfg"]["nuscenes_version"] = config["data_cfg"]["nuscenes_version"]

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    model = ReconDrive_LITModelModule(cfg=config["model_cfg"], save_dir=".", logger=None)
    model.load_pretrained_checkpoint(args.restore_ckpt)
    model.to(device)
    model.eval()

    data_module = VGGT3DGS_SceneDataModule(cfg=config["data_cfg"])
    data_module.setup(stage="test")
    scene_dataloader = data_module.test_scene_dataloader()

    data_root = Path(config["data_cfg"]["data_path"])
    version = str(config["data_cfg"].get("nuscenes_version", "v1.0-mini"))
    tables = NuscTables(data_root, version)
    context_span = int(config["data_cfg"].get("context_span", 1))
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    all_results = []
    started = time.time()
    for scene_idx, scene_batch in enumerate(scene_dataloader):
        if args.max_scenes is not None and len(all_results) >= args.max_scenes:
            break
        if args.scene is not None and scene_batch["scene_name"] != args.scene:
            continue
        print(f"\n=== Sweep evaluation: {scene_batch['scene_name']} ===", flush=True)
        result = run_sweep_evaluation(
            model,
            scene_batch,
            tables,
            data_root=data_root,
            device=device,
            frame_skip=args.frame_skip,
            context_span=context_span,
            scene_idx=scene_idx,
            save_renders=args.save_renders,
            output_dir=output_dir,
            sanity_zero=args.sanity_zero,
            save_views=args.save_views,
        )
        all_results.append(result)
        summary = aggregate_scene_metrics(result)
        print(
            f"  cam_sweeps={summary['camera_sweep_count']} "
            f"lidar_sweeps={summary['lidar_sweep_count']} "
            f"psnr={summary['camera_psnr']['mean']:.3f} "
            f"depth_l2={summary['lidar_depth_l2']['mean']:.3f}",
            flush=True,
        )
        with open(
            output_dir / f"{result['scene_name']}_sweep_evaluation.json", "w"
        ) as handle:
            json.dump(result, handle, indent=2, sort_keys=True)

    overall: dict[str, Any] = {"scenes": [], "overall_metrics": {}}
    pooled_keys = {
        "camera_psnr",
        "camera_ssim",
        "camera_lpips",
        "lidar_depth_l2",
        "lidar_depth_median_l2",
        "lidar_delta_1",
        "lidar_delta_2",
        "lidar_delta_3",
        "lidar_intensity_rmse",
        "lidar_ray_drop_acc",
        "lidar_chamfer_distance",
    }
    for scene_result in all_results:
        scene_summary = aggregate_scene_metrics(scene_result)
        overall["scenes"].append(scene_summary)
    for key in pooled_keys:
        values = [
            item[key]["mean"]
            for item in overall["scenes"]
            if np.isfinite(item[key]["mean"])
        ]
        overall["overall_metrics"][key] = _mean_std(values)
    overall["total_camera_sweeps"] = sum(
        item["camera_sweep_count"] for item in overall["scenes"]
    )
    overall["total_lidar_sweeps"] = sum(
        item["lidar_sweep_count"] for item in overall["scenes"]
    )
    overall["elapsed_seconds"] = round(time.time() - started, 1)
    with open(output_dir / "sweep_inference_summary.json", "w") as handle:
        json.dump(overall, handle, indent=2, sort_keys=True)

    print("\n" + "=" * 60)
    print("SWEEP-TIME EVALUATION COMPLETED")
    print(json.dumps(overall["overall_metrics"], indent=2, sort_keys=True))
    print(f"Output: {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
