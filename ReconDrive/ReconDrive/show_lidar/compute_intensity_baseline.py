#!/usr/bin/env python3
"""
Compute a naive intensity baseline: train-set mean GT intensity → test-set per-sample RMSE.

This script:
  1. Iterates over all training set samples, loads LiDAR GT, builds intensity range images,
     and computes the per-pixel mean intensity (only at valid pixels).
  2. Iterates over all test set samples, and for each sample computes the RMSE between
     the test GT intensity and the training-set mean intensity.
  3. Reports the average intensity RMSE across the test set as a baseline reference.

Usage:
    python show_lidar/compute_intensity_baseline.py

Requirements:
    - nuScenes dataset at the default path (configurable via --data_root)
    - nuscenes-devkit installed (pip install nuscenes-devkit)
"""

import argparse
import json
import os
import sys
import numpy as np
from pathlib import Path

from nuscenes.nuscenes import NuScenes
from nuscenes.utils import splits
from pyquaternion import Quaternion
from PIL import Image


# ──────────────────────────────────────────────────────────────────────
#  Inline constants & helpers from dataset/data_util.py
#  (copied here so the script is self-contained)
# ──────────────────────────────────────────────────────────────────────
LIDAR_NUM_RINGS = 32
LIDAR_AZIMUTH_BINS = 3600
LIDAR_ELEVATIONS = np.linspace(-30, 10, LIDAR_NUM_RINGS)
LIDAR_AZIMUTH_RESOLUTION = 360.0 / LIDAR_AZIMUTH_BINS


def build_lidar_range_image_simple(lidar_points, lidar_to_ego):
    """Simplified version that computes only depth, intensity, ray_drop range images.

    Args:
        lidar_points: (N, 5) = [x, y, z, intensity, ring]
        lidar_to_ego: (4, 4) LiDAR→ego transform

    Returns:
        gt_intensity: (H, W) float32, values in [0, 1], 0 = no return
    """
    xyz = lidar_points[:, :3]
    intensity_raw = lidar_points[:, 3]  # 0-255 raw
    ring = lidar_points[:, 4].astype(int)

    depth = np.linalg.norm(xyz, axis=1)
    azimuth = np.degrees(np.arctan2(xyz[:, 1], xyz[:, 0]))  # [-180, 180)
    # Note: elevation is only needed if we wanted to verify ring mapping,
    # but the ring channel from the sensor is authoritative.

    H, W = LIDAR_NUM_RINGS, LIDAR_AZIMUTH_BINS

    gt_intensity = np.zeros((H, W), dtype=np.float32)

    for r in range(LIDAR_NUM_RINGS):
        mask = ring == r
        if not mask.any():
            continue
        pts_az = azimuth[mask]
        pts_d = depth[mask]
        pts_int = intensity_raw[mask]
        col = ((pts_az + 180) / LIDAR_AZIMUTH_RESOLUTION).astype(int) % W
        for ci in range(W):
            bin_mask = col == ci
            if bin_mask.any():
                nearest = np.argmin(pts_d[bin_mask])
                gt_intensity[r, ci] = pts_int[bin_mask][nearest] / 255.0

    return gt_intensity


# ──────────────────────────────────────────────────────────────────────
#  Test scenes (from vggt4dgs_scene_dataset.py lines 86-90)
# ──────────────────────────────────────────────────────────────────────
TEST_SCENE_NAMES = [
    'scene-0061', 'scene-0103', 'scene-0553', 'scene-0655',
    'scene-0757', 'scene-0796', 'scene-0916', 'scene-1077',
    'scene-1094', 'scene-1100',
]


def load_lidar_intensity(nusc, sample_token, data_root):
    """Load LiDAR GT and return the intensity range image (H, W) in [0, 1]."""
    sample = nusc.get('sample', sample_token)
    lidar_sample = nusc.get('sample_data', sample['data']['LIDAR_TOP'])
    lidar_file = os.path.join(data_root, lidar_sample['filename'])
    lidar_points = np.fromfile(lidar_file, dtype=np.float32).reshape(-1, 5)

    calib = nusc.get('calibrated_sensor', lidar_sample['calibrated_sensor_token'])
    calib_rot = Quaternion(calib['rotation']).rotation_matrix
    calib_trans = np.array(calib['translation']).reshape(1, 3)
    lidar_to_ego = np.eye(4)
    lidar_to_ego[:3, :3] = calib_rot
    lidar_to_ego[:3, 3] = calib_trans

    return build_lidar_range_image_simple(lidar_points, lidar_to_ego)


def collect_intensities(nusc, data_root, scene_names, verbose=True):
    """Collect GT intensity range images for all samples in the given scenes.

    Returns:
        intensities: list of (H, W) float32 arrays, intensity in [0, 1]
    """
    intensities = []
    total_samples = 0

    for scene in nusc.scene:
        if scene['name'] not in scene_names:
            continue
        if verbose:
            print(f"  Scene: {scene['name']}")

        sample_token = scene['first_sample_token']
        visited = set()
        scene_count = 0
        while sample_token:
            if sample_token in visited:
                break
            visited.add(sample_token)
            try:
                gt_intensity = load_lidar_intensity(nusc, sample_token, data_root)
                intensities.append(gt_intensity)
                scene_count += 1
            except Exception as e:
                print(f"    [WARN] Failed to load sample {sample_token}: {e}")

            sample_obj = nusc.get('sample', sample_token)
            sample_token = sample_obj['next']

        total_samples += scene_count
        if verbose:
            print(f"    Loaded {scene_count} samples")

    if verbose:
        print(f"  Total: {total_samples} samples from {len(scene_names)} scenes")
    return intensities


def compute_mean_intensity(intensity_list):
    """Compute per-pixel mean intensity across all samples.

    For each pixel location (r, c), the mean is computed only over samples
    where the intensity > 0 (valid LiDAR return). Pixels that are never
    valid across the training set remain 0.

    Returns:
        mean_intensity: (H, W) float32
        count_map:      (H, W) int32 — number of valid observations per pixel
    """
    H, W = intensity_list[0].shape
    sum_intensity = np.zeros((H, W), dtype=np.float64)
    count_map = np.zeros((H, W), dtype=np.int32)

    for img in intensity_list:
        valid = img > 0
        sum_intensity[valid] += img[valid]
        count_map[valid] += 1

    mean_intensity = np.where(count_map > 0, sum_intensity / count_map, 0.0).astype(np.float32)
    return mean_intensity, count_map


def compute_rmse_vs_mean(intensity_list, mean_intensity):
    """For each sample, compute RMSE of GT intensity vs. the training-set mean.

    RMSE is computed per sample only over valid (gt > 0) pixels.

    Returns:
        per_sample_rmse: list of float — one per sample
    """
    rmses = []
    for img in intensity_list:
        valid = img > 0
        if not valid.any():
            rmses.append(0.0)
            continue
        diff = img[valid] - mean_intensity[valid]
        rmse = np.sqrt(np.mean(diff ** 2))
        rmses.append(float(rmse))
    return rmses


def main():
    parser = argparse.ArgumentParser(
        description='Compute training-set mean intensity baseline and test-set intensity RMSE'
    )
    parser.add_argument(
        '--data_root', type=str,
        default=str(Path(__file__).parent.parent / 'data' / 'nuscenes'),
        help='NuScenes dataset root directory'
    )
    parser.add_argument(
        '--version', type=str, default='v1.0-mini',
        help='NuScenes dataset version (default: v1.0-mini)'
    )
    args = parser.parse_args()

    print("=" * 60)
    print("LiDAR GT Intensity Baseline (Train Mean -> Test RMSE)")
    print("=" * 60)
    print(f"Data root: {args.data_root}")
    print(f"Version:   {args.version}")

    # ── Load nuScenes ──────────────────────────────────────────────
    print("\n[1] Loading nuScenes dataset...")
    nusc = NuScenes(version=args.version, dataroot=args.data_root, verbose=True)

    # ── Determine train / test scene names ─────────────────────────
    train_scene_names = set(splits.train)
    test_scene_names = set(TEST_SCENE_NAMES)

    # Filter to only scenes that actually exist in this dataset
    available_scenes = {s['name'] for s in nusc.scene}
    train_scene_names = [s for s in train_scene_names if s in available_scenes]
    test_scene_names = [s for s in test_scene_names if s in available_scenes]

    print(f"\n[2] Training scenes ({len(train_scene_names)} available): {train_scene_names}")

    # ── Collect training intensities ───────────────────────────────
    print("\n[3] Collecting training-set GT intensities...")
    train_intensities = collect_intensities(nusc, args.data_root, train_scene_names)

    if len(train_intensities) == 0:
        print("[ERROR] No training samples found. Check data_root and version.")
        sys.exit(1)

    # ── Compute training-set mean intensity ────────────────────────
    print("\n[4] Computing per-pixel mean intensity over training set...")
    train_mean, count_map = compute_mean_intensity(train_intensities)
    valid_pixel_ratio = (count_map > 0).mean() * 100
    print(f"  Mean intensity shape: {train_mean.shape}")
    print(f"  Pixels with >=1 valid observation: {valid_pixel_ratio:.1f}%")
    print(f"  Global mean intensity (over valid pixels): "
          f"{train_mean[train_mean > 0].mean():.4f}")

    # ── Save mean intensity for visualization ──────────────────────
    save_dir = Path(__file__).parent
    os.makedirs(save_dir, exist_ok=True)
    mean_img_uint8 = (train_mean * 255).clip(0, 255).astype(np.uint8)
    Image.fromarray(mean_img_uint8).save(save_dir / 'train_mean_intensity.png')
    np.save(save_dir / 'train_mean_intensity.npy', train_mean)
    print(f"\n  Saved mean intensity image: {save_dir / 'train_mean_intensity.png'}")
    print(f"  Saved mean intensity array: {save_dir / 'train_mean_intensity.npy'}")

    # ── Collect test intensities ───────────────────────────────────
    print(f"\n[5] Test scenes ({len(test_scene_names)}): {test_scene_names}")
    print("    Collecting test-set GT intensities...")
    test_intensities = collect_intensities(nusc, args.data_root, test_scene_names)

    if len(test_intensities) == 0:
        print("[WARN] No test samples found. Dataset might not contain test scenes.")
        print("       This is expected for v1.0-mini (only has 2 scenes).")
        print("       The script will still work with a full nuScenes dataset.")
    else:
        # ── Compute per-sample RMSE against train mean ─────────────
        print(f"\n[6] Computing per-sample intensity RMSE vs. training-set mean...")
        per_sample_rmse = compute_rmse_vs_mean(test_intensities, train_mean)

        avg_rmse = np.mean(per_sample_rmse)
        std_rmse = np.std(per_sample_rmse)
        min_rmse = np.min(per_sample_rmse)
        max_rmse = np.max(per_sample_rmse)

        print(f"\n{'=' * 60}")
        print(f"RESULTS: Intensity RMSE Baseline (Train Mean -> Test)")
        print(f"{'=' * 60}")
        print(f"  Number of test samples:    {len(per_sample_rmse)}")
        print(f"  Mean intensity RMSE:        {avg_rmse:.6f}  ({avg_rmse * 255:.3f} / 255)")
        print(f"  Std intensity RMSE:         {std_rmse:.6f}  ({std_rmse * 255:.3f} / 255)")
        print(f"  Min intensity RMSE:         {min_rmse:.6f}  ({min_rmse * 255:.3f} / 255)")
        print(f"  Max intensity RMSE:         {max_rmse:.6f}  ({max_rmse * 255:.3f} / 255)")
        print(f"\n  This is the naive baseline: predicting every pixel's")
        print(f"  intensity as the training-set mean at that position.")
        print(f"  Any learned model should beat this.")
        print(f"{'=' * 60}")

        # Save results
        results = {
            'num_train_samples': len(train_intensities),
            'num_test_samples': len(test_intensities),
            'mean_intensity_rmse': float(avg_rmse),
            'std_intensity_rmse': float(std_rmse),
            'min_intensity_rmse': float(min_rmse),
            'max_intensity_rmse': float(max_rmse),
            'per_sample_rmse': per_sample_rmse,
        }
        with open(save_dir / 'intensity_baseline_results.json', 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\n  Results saved: {save_dir / 'intensity_baseline_results.json'}")

    print("\nDone.")


if __name__ == '__main__':
    main()
