#!/usr/bin/env python3
"""
LiDAR Range Image Visualization Script

Displays GT vs Predicted LiDAR range images (depth, intensity, ray_drop)
as a side-by-side 2x3 grid with proper color mapping.

Usage:
  # Single lidar folder
  python show_lidar/visualize_lidar.py work_dirs/.../scene-0061/sample_0000/lidar

  # Multiple folders
  python show_lidar/visualize_lidar.py <folder1> <folder2> ...

  # Batch all scenes under a directory
  python show_lidar/visualize_lidar.py work_dirs/inference_trained_results_v3 --batch

  # Batch with a specific scene filter
  python show_lidar/visualize_lidar.py work_dirs/inference_trained_results_v3 --batch --scene scene-0061
"""

import argparse
import glob
import os
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


def load_lidar_png(path):
    """Load a LiDAR PNG, return as numpy array."""
    img = np.array(Image.open(path))
    return img


def normalize_depth(depth, clip_percentile=98):
    """
    Normalize depth (uint16 mm) to [0, 1] for visualization.
    Clips outliers at the given percentile, then min-max to [0,1].
    """
    depth = depth.astype(np.float32)
    valid = depth > 0
    if not valid.any():
        return np.zeros_like(depth, dtype=np.float32)

    d_valid = depth[valid]
    vmax = np.percentile(d_valid, clip_percentile)
    vmin = d_valid.min()

    result = np.zeros_like(depth, dtype=np.float32)
    clipped = np.clip(depth, vmin, vmax)
    # Avoid division by zero
    if vmax - vmin > 1e-6:
        result[valid] = (clipped[valid] - vmin) / (vmax - vmin)
    return result


def normalize_intensity(intensity):
    """Normalize intensity (uint8) to [0, 1], masking zero pixels."""
    img = intensity.astype(np.float32) / 255.0
    # Where GT is 0 (no return), set to NaN so it shows as transparent/white
    return img


def visualize_lidar(lidar_dir, output_path=None, dpi=150):
    """
    Create a 2x3 visualization grid from a lidar directory.

    The six panels:
      GT Depth         |  Pred Depth
      GT Intensity     |  Pred Intensity
      GT RayDrop       |  Pred RayDrop
    """
    files = {
        'gt_depth.png':      ('GT Depth', 'depth'),
        'pred_depth.png':    ('Pred Depth', 'depth'),
        'gt_intensity.png':  ('GT Intensity', 'intensity'),
        'pred_intensity.png':('Pred Intensity', 'intensity'),
        'gt_ray_drop.png':   ('GT RayDrop', 'ray_drop'),
        'pred_ray_drop.png': ('Pred RayDrop', 'ray_drop'),
    }

    data = {}
    for fname, (title, kind) in files.items():
        fpath = os.path.join(lidar_dir, fname)
        if not os.path.isfile(fpath):
            print(f"  [WARN] Missing {fname}, skipping")
            continue
        img = load_lidar_png(fpath)

        if kind == 'depth':
            vis = normalize_depth(img, clip_percentile=98)
            cmap = 'viridis'
        elif kind == 'intensity':
            # GT intensity: zero pixels = no return, show as NaN
            # Pred intensity: nearly constant (broken), still show
            vis = normalize_intensity(img)
            if fname.startswith('gt_'):
                vis[img == 0] = np.nan
            cmap = 'gray'
        elif kind == 'ray_drop':
            vis = img.astype(np.float32) / 255.0
            # ray_drop is binary: 0 = hit (no drop), 255 = drop
            # Show as-is, no NaN masking
            cmap = 'gray'
        else:
            vis = img
            cmap = 'gray'

        data[fname] = {
            'title': title,
            'vis': vis,
            'cmap': cmap,
        }

    if not data:
        print(f"  [ERROR] No valid PNG files found in {lidar_dir}")
        return

    # Sort into order: gt_depth, pred_depth, gt_intensity, pred_intensity, gt_raydrop, pred_raydrop
    order = ['gt_depth.png', 'pred_depth.png',
             'gt_intensity.png', 'pred_intensity.png',
             'gt_ray_drop.png', 'pred_ray_drop.png']

    fig, axes = plt.subplots(3, 2, figsize=(14, 8))

    for ax, fname in zip(axes.flat, order):
        if fname in data:
            d = data[fname]
            cmap = d['cmap']

            if cmap in ('viridis', 'turbo', 'plasma', 'inferno', 'magma'):
                # Depth colormap: NaN is valid=0, so use set_bad for zero pixels
                masked = np.ma.masked_where(d['vis'] == 0, d['vis'])
                im = ax.imshow(masked, cmap=cmap, aspect='auto', vmin=0, vmax=1)
            else:
                # Grayscale: NaN for missing, Bad for zero
                # For GT: zero = no return (transparent), for pred show everything
                if 'gt_' in fname:
                    # GT: zero pixels are no LiDAR return → make transparent
                    masked = d['vis']  # NaN already set where GT==0
                    im = ax.imshow(masked, cmap=cmap, aspect='auto', vmin=0, vmax=1)
                else:
                    im = ax.imshow(d['vis'], cmap=cmap, aspect='auto', vmin=0, vmax=1)

            ax.set_title(d['title'], fontsize=11, fontweight='bold')
        else:
            ax.set_title('(missing)', fontsize=11, fontweight='bold')
            ax.text(0.5, 0.5, 'N/A', ha='center', va='center', transform=ax.transAxes, fontsize=14, color='gray')

        ax.set_xlabel('Azimuth bin')
        ax.set_ylabel('Elevation (ring)')
        ax.tick_params(labelsize=7)

    plt.suptitle(f'LiDAR Range Image Comparison\n{lidar_dir}', fontsize=13, fontweight='bold')
    plt.tight_layout()

    if output_path is None:
        output_path = os.path.join(lidar_dir, 'lidar_visualization.png')

    plt.savefig(output_path, dpi=dpi, bbox_inches='tight')
    plt.close(fig)
    print(f"  [OK] Saved visualization: {output_path}")


def find_lidar_dirs(root_dir, scene_filter=None):
    """Recursively find all 'lidar' directories under root_dir."""
    lidar_dirs = []
    for root, dirs, files in os.walk(root_dir):
        if os.path.basename(root) == 'lidar':
            # Check if it contains the typical lidar PNGs
            has_files = any(f.endswith('.png') for f in files)
            if has_files:
                # Apply scene filter if specified
                if scene_filter:
                    if scene_filter in root:
                        lidar_dirs.append(root)
                else:
                    lidar_dirs.append(root)
    return sorted(lidar_dirs)


def main():
    parser = argparse.ArgumentParser(description='Visualize LiDAR range images')
    parser.add_argument('paths', nargs='+', help='Lidar directories or result root dir')
    parser.add_argument('--batch', action='store_true',
                        help='Batch mode: treat paths as result root dirs, find all lidar/ subdirs')
    parser.add_argument('--scene', type=str, default=None,
                        help='Scene name filter (only in batch mode)')
    parser.add_argument('--dpi', type=int, default=150, help='Output image DPI')

    args = parser.parse_args()

    if args.batch:
        all_dirs = []
        for p in args.paths:
            all_dirs.extend(find_lidar_dirs(p, scene_filter=args.scene))
        if not all_dirs:
            print(f"[ERROR] No 'lidar' subdirectories found under {args.paths}")
            sys.exit(1)
        print(f"Found {len(all_dirs)} lidar directories to process")
        for i, d in enumerate(all_dirs):
            print(f"  [{i+1}/{len(all_dirs)}] {d}")
            visualize_lidar(d, dpi=args.dpi)
    else:
        for p in args.paths:
            if not os.path.isdir(p):
                print(f"[ERROR] Directory not found: {p}")
                continue
            output = os.path.join(p, 'lidar_visualization.png')
            print(f"Processing: {p}")
            visualize_lidar(p, output_path=output, dpi=args.dpi)


if __name__ == '__main__':
    main()
