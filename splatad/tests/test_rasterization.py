"""Tests for the functions in the CUDA extension.

Usage:
```bash
pytest <THIS_PY_FILE> -s
```
"""

import math
from typing import Optional

import pytest
import torch

device = torch.device("cuda:0")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="No CUDA device")
@pytest.mark.parametrize("per_view_color", [True, False])
@pytest.mark.parametrize("sh_degree", [None, 3])
@pytest.mark.parametrize("render_mode", ["RGB", "RGB+D", "D"])
@pytest.mark.parametrize("packed", [True, False])
def test_rasterization(per_view_color: bool, sh_degree: Optional[int], render_mode: str, packed: bool):
    from gsplat.rendering import rasterization

    torch.manual_seed(42)

    C, N = 2, 10_000
    means = torch.rand(N, 3, device=device)
    velocities = torch.randn(N, 3, device=device) * 0.01
    quats = torch.randn(N, 4, device=device)
    scales = torch.rand(N, 3, device=device)
    opacities = torch.rand(N, device=device)
    if per_view_color:
        if sh_degree is None:
            colors = torch.rand(C, N, 3, device=device)
        else:
            colors = torch.rand(C, N, (sh_degree + 1) ** 2, 3, device=device)
    else:
        if sh_degree is None:
            colors = torch.rand(N, 3, device=device)
        else:
            colors = torch.rand(N, (sh_degree + 1) ** 2, 3, device=device)

    width, height = 300, 200
    focal = 300.0
    Ks = torch.tensor(
        [[focal, 0.0, width / 2.0], [0.0, focal, height / 2.0], [0.0, 0.0, 1.0]],
        device=device,
    ).expand(C, -1, -1)
    viewmats = torch.eye(4, device=device).expand(C, -1, -1)

    linear_velocity = torch.randn(C, 3, device=device) * 0.01
    angular_velocity = torch.randn(C, 3, device=device) * 0.01
    rolling_shutter_time = torch.rand(C, device=device) * 0.1

    colors_out, _, _ = rasterization(
        means=means,
        quats=quats,
        scales=scales,
        opacities=opacities,
        colors=colors,
        velocities=velocities,
        viewmats=viewmats,
        Ks=Ks,
        width=width,
        height=height,
        linear_velocity=linear_velocity,
        angular_velocity=angular_velocity,
        rolling_shutter_time=rolling_shutter_time,
        sh_degree=sh_degree,
        render_mode=render_mode,
        packed=packed,
    )

    if render_mode == "D":
        assert colors_out.shape == (C, height, width, 1)
    elif render_mode == "RGB":
        assert colors_out.shape == (C, height, width, 3)
    elif render_mode == "RGB+D":
        assert colors_out.shape == (C, height, width, 4)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="No CUDA device")
@pytest.mark.parametrize("channels", [3, 32, 128])
def test_lidar_rasterization(channels: int):
    from gsplat.rendering import lidar_rasterization

    torch.manual_seed(42)

    C, N = 2, 10_000
    means = torch.rand(N, 3, device=device)
    quats = torch.randn(N, 4, device=device)
    scales = torch.rand(N, 3, device=device)
    opacities = torch.rand(N, device=device)
    velocities = torch.randn(N, 3, device=device) * 0.01

    min_azimuth = -180
    max_azimuth = 180
    min_elevation = -45
    max_elevation = 45
    n_elevation_channels = 32
    azimuth_resolution = 0.2

    tile_width = 64
    tile_height = 4
    tile_elevation_boundaries = torch.linspace(
        min_elevation, max_elevation, n_elevation_channels // tile_height + 1, device=means.device
    )

    viewmats = torch.eye(4, device=device).expand(C, -1, -1)
    lidar_features = torch.randn(C, len(means), channels, device=device)

    image_width = math.ceil((max_azimuth - min_azimuth) / azimuth_resolution)
    raster_pts_azim = torch.linspace(
        min_azimuth + azimuth_resolution / 2, max_azimuth - azimuth_resolution / 2, image_width, device=means.device
    )
    raster_pts_elev = torch.linspace(
        min_elevation + (max_elevation - min_elevation) / n_elevation_channels / 2,
        max_elevation - (max_elevation - min_elevation) / n_elevation_channels / 2,
        n_elevation_channels,
        device=means.device,
    )
    raster_pts = torch.stack(torch.meshgrid(raster_pts_elev, raster_pts_azim), dim=-1)[..., [1, 0]]
    ranges = torch.rand(n_elevation_channels, image_width, 1, device=device) * 10
    keep_range_mask = torch.rand(n_elevation_channels, image_width, device=device) > 0.1
    raster_pts = torch.cat([raster_pts, ranges], dim=-1)
    raster_pts = raster_pts.unsqueeze(0).repeat(C, 1, 1, 1)
    # add randomness
    raster_pts += torch.randn_like(raster_pts) * 0.01 * keep_range_mask[None, ..., None]

    linear_velocity = torch.randn(C, 3, device=device) * 0.01
    angular_velocity = torch.randn(C, 3, device=device) * 0.01
    rolling_shutter_time = torch.rand(C, device=device) * 0.1

    # add timestamps
    raster_pts = torch.cat(
        [raster_pts, rolling_shutter_time.max() * torch.randn(raster_pts[..., 0:1].shape, device=raster_pts.device)],
        dim=-1,
    )

    render_lidar_features, _, _, _ = lidar_rasterization(
        means=means,
        quats=quats,
        scales=scales,
        opacities=opacities,
        lidar_features=lidar_features,
        velocities=velocities,
        linear_velocity=linear_velocity,
        angular_velocity=angular_velocity,
        rolling_shutter_time=rolling_shutter_time,
        viewmats=viewmats,
        min_azimuth=min_azimuth,
        max_azimuth=max_azimuth,
        min_elevation=min_elevation,
        max_elevation=max_elevation,
        n_elevation_channels=n_elevation_channels,
        azimuth_resolution=azimuth_resolution,
        raster_pts=raster_pts,
        tile_width=tile_width,
        tile_height=tile_height,
        tile_elevation_boundaries=tile_elevation_boundaries,
    )

    n_azimuth_pixels = math.ceil((max_azimuth - min_azimuth) / azimuth_resolution)
    assert render_lidar_features.shape == (C, n_elevation_channels, n_azimuth_pixels, channels + 1)
