#!/usr/bin/env python3
"""
将 ReconDrive 推理输出的 LiDAR Range Image PNG 恢复为 3D 点云 (.ply)

# 处理单个 lidar 文件夹
python show_lidar/range_image_to_ply.py work_dirs/.../scene-0103/sample_0000/lidar

# 处理多个
python show_lidar/range_image_to_ply.py <folder1> <folder2>

# 批量处理整个目录下的所有场景
python show_lidar/range_image_to_ply.py work_dirs/inference_trained_results --batch

# 批量处理指定场景
python show_lidar/range_image_to_ply.py work_dirs/inference_trained_results --batch --scene scene-0103
"""

import argparse
import numpy as np
from PIL import Image
from pathlib import Path

# ---- 与 dataset/data_util.py 保持一致 ----
LIDAR_NUM_RINGS = 32                # nuScenes 32线激光雷达
LIDAR_AZIMUTH_BINS = 3600           # 方位角分辨率 0.1°
LIDAR_ELEVATIONS = np.linspace(-30, 10, LIDAR_NUM_RINGS)   # 俯仰角范围
LIDAR_AZIMUTH_RESOLUTION = 360.0 / LIDAR_AZIMUTH_BINS      # 0.1°
RAY_DROP_THRESHOLD = 128            # 二值化阈值 (0-255)


def spherical_to_cartesian(az_deg, el_deg, depth):
    """球坐标 → 笛卡尔坐标 (x-forward, y-left, z-up)

    Args:
        az_deg: 方位角 (度),  0°=前方, 顺时针增加
        el_deg: 俯仰角 (度), 正=向上
        depth:  距离 (米)

    Returns:
        (x, y, z) 笛卡尔坐标
    """
    az = np.deg2rad(az_deg)
    el = np.deg2rad(el_deg)
    cos_el = np.cos(el)

    x = depth * cos_el * np.cos(az)
    y = depth * cos_el * np.sin(az)
    z = depth * np.sin(el)
    return x, y, z


def load_png(path, dtype, scale=1.0):
    """读取 PNG 并转为 numpy 数组"""
    img = Image.open(path)
    arr = np.array(img).astype(np.float32)
    if scale != 1.0:
        arr = arr * scale
    return arr.astype(dtype)


def read_lidar_images(lidar_dir):
    """读取 lidar 文件夹中的 6 张 PNG，返回 dict"""
    lidar_dir = Path(lidar_dir)
    data = {}

    pred_depth_path = lidar_dir / "pred_depth.png"
    if pred_depth_path.exists():
        # uint16, 单位 mm → 转为米
        depth_mm = load_png(pred_depth_path, np.float32)
        data["pred_depth"] = depth_mm / 1000.0

    pred_intensity_path = lidar_dir / "pred_intensity.png"
    if pred_intensity_path.exists():
        # uint8, 0-255 → 归一化到 0-1
        data["pred_intensity"] = load_png(pred_intensity_path, np.float32) / 255.0

    pred_ray_drop_path = lidar_dir / "pred_ray_drop.png"
    if pred_ray_drop_path.exists():
        # uint8, 0-255 → 二值化
        data["pred_ray_drop"] = load_png(pred_ray_drop_path, np.float32) / 255.0

    gt_depth_path = lidar_dir / "gt_depth.png"
    if gt_depth_path.exists():
        depth_mm = load_png(gt_depth_path, np.float32)
        data["gt_depth"] = depth_mm / 1000.0

    gt_intensity_path = lidar_dir / "gt_intensity.png"
    if gt_intensity_path.exists():
        data["gt_intensity"] = load_png(gt_intensity_path, np.float32) / 255.0

    gt_ray_drop_path = lidar_dir / "gt_ray_drop.png"
    if gt_ray_drop_path.exists():
        data["gt_ray_drop"] = load_png(gt_ray_drop_path, np.float32) / 255.0

    return data


def build_spherical_grid():
    """构建 azimuth / elevation 网格

    Returns:
        az_grid: (32, 3600) 方位角 (度)
        el_grid: (32, 3600) 俯仰角 (度)
    """
    # 与 data_util.py 中的 raster_pts 一致
    az_grid, el_grid = np.meshgrid(
        np.linspace(0, 360, LIDAR_AZIMUTH_BINS, endpoint=False),
        LIDAR_ELEVATIONS,
    )
    return az_grid, el_grid


def extract_point_cloud(depth, intensity, ray_drop, az_grid, el_grid):
    """从 range image 提取有效点云

    Args:
        depth:    (H, W) 深度图 (米)
        intensity: (H, W) 反射强度 [0, 1]
        ray_drop:  (H, W) 有效返回概率 [0, 1]
        az_grid:  (H, W) 方位角 (度)
        el_grid:  (H, W) 俯仰角 (度)

    Returns:
        points:    (N, 3) 点云坐标
        intensity: (N,)   强度值
        depth:     (N,)   深度值
    """
    valid = (depth > 0) & (ray_drop > 0.5)

    az = az_grid[valid]
    el = el_grid[valid]
    d = depth[valid]
    inten = intensity[valid]

    x, y, z = spherical_to_cartesian(az, el, d)

    points = np.stack([x, y, z], axis=-1)
    return points, inten, d


def save_ply(filepath, points, intensities=None, depths=None):
    """保存为 PLY 点云文件

    Args:
        filepath:  输出路径
        points:    (N, 3) 点云坐标
        intensities: (N,) 可选强度
        depths:    (N,) 可选深度
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)

    n = points.shape[0]
    has_intensity = intensities is not None
    has_depth = depths is not None

    with open(filepath, "w") as f:
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {n}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        if has_intensity:
            f.write("property float intensity\n")
        if has_depth:
            f.write("property float depth\n")
        f.write("end_header\n")

        for i in range(n):
            line = f"{points[i, 0]:.4f} {points[i, 1]:.4f} {points[i, 2]:.4f}"
            if has_intensity:
                line += f" {intensities[i]:.4f}"
            if has_depth:
                line += f" {depths[i]:.4f}"
            f.write(line + "\n")

    print(f"  ✓ 保存 {n} 个点到 {filepath}")


def process_lidar_folder(lidar_dir):
    """处理一个 lidar 文件夹，生成 .ply 文件"""
    lidar_dir = Path(lidar_dir)
    if not lidar_dir.is_dir():
        print(f"✗ 目录不存在: {lidar_dir}")
        return

    print(f"\n处理: {lidar_dir}")
    data = read_lidar_images(lidar_dir)
    if not data:
        print("✗ 没有找到 PNG 文件")
        return

    # 验证图片尺寸并打印值范围
    sample_key = list(data.keys())[0]
    h, w = data[sample_key].shape
    print(f"  Range Image 尺寸: {h}×{w} (期望: {LIDAR_NUM_RINGS}×{LIDAR_AZIMUTH_BINS})")
    for key in sorted(data.keys()):
        arr = data[key]
        print(f"    {key}: range [{arr[arr > 0].min() if (arr > 0).any() else 0:.4f}, {arr.max():.4f}], "
              f"非零像素 {(arr > 0).sum()}/{arr.size}")

    # 构建球坐标网格
    az_grid, el_grid = build_spherical_grid()

    # 处理预测数据
    if all(k in data for k in ["pred_depth", "pred_intensity", "pred_ray_drop"]):
        print("  预测点云:")
        points, intensities, depths = extract_point_cloud(
            data["pred_depth"], data["pred_intensity"], data["pred_ray_drop"],
            az_grid, el_grid,
        )
        save_ply(lidar_dir / "pred_points.ply", points, intensities, depths)
    else:
        print("  ⚠ 缺少预测数据 (需要 pred_depth.png, pred_intensity.png, pred_ray_drop.png)")

    # 处理真值数据
    if all(k in data for k in ["gt_depth", "gt_intensity", "gt_ray_drop"]):
        print("  真值点云:")
        points, intensities, depths = extract_point_cloud(
            data["gt_depth"], data["gt_intensity"], data["gt_ray_drop"],
            az_grid, el_grid,
        )
        save_ply(lidar_dir / "gt_points.ply", points, intensities, depths)


def main():
    parser = argparse.ArgumentParser(
        description="将 ReconDrive LiDAR Range Image PNG 恢复为 3D 点云 (.ply)"
    )
    parser.add_argument(
        "lidar_folder",
        type=str,
        nargs="+",
        help="lidar 文件夹路径，可指定多个；或使用 --batch 指定场景根目录批量处理",
    )
    parser.add_argument(
        "--batch",
        action="store_true",
        help="批量模式: lidar_folder 是场景根目录 (如 work_dirs/.../scene-0103/)，"
             "会自动查找所有 sample_*/lidar 子文件夹",
    )
    parser.add_argument(
        "--scene",
        type=str,
        default=None,
        help="在 --batch 模式下，只处理指定场景名称 (如 scene-0103)",
    )
    args = parser.parse_args()

    if args.batch:
        root = Path(args.lidar_folder[0])
        if args.scene:
            scene_dirs = [root / args.scene]
        else:
            scene_dirs = sorted(root.glob("scene-*/"))
        print(f"批量模式: 扫描 {len(scene_dirs)} 个场景...")
        for scene_dir in scene_dirs:
            lidar_dirs = sorted(scene_dir.glob("sample_*/lidar"))
            for ld in lidar_dirs:
                process_lidar_folder(ld)
    else:
        for folder in args.lidar_folder:
            process_lidar_folder(folder)


if __name__ == "__main__":
    main()
