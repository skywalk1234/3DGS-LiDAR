#----------------------------------------------------------------#
# ReconDrive - LiDAR conditioning modules                          #
# Depth residual head + Lidar feature residual head               #
#----------------------------------------------------------------#

import torch
import torch.nn as nn


class LidarFeatureBuilder(nn.Module):
    """
    Build 4-channel LiDAR feature map from projected LiDAR data.
    
    Input channels (per camera view):
        depth_maps: predicted depth (normalized to [0,1])
        proj_depth: LiDAR projected depth (normalized to [0,1])
        proj_intensity: LiDAR projected intensity (normalized)
        proj_mask: binary valid mask
    
    Output:
        4-channel feature: [lidar_depth_norm, lidar_intensity_norm, 
                           valid_mask, depth_diff]
    """
    def __init__(self, min_depth=1.5, max_depth=110.0):
        super().__init__()
        self.min_depth = min_depth
        self.max_depth = max_depth

    def forward(self, depth_maps, proj_depth, proj_intensity, proj_mask):
        """
        Args:
            depth_maps: [B*V, 1, H, W] predicted depth in meters
            proj_depth: [B*V, 1, H, W] LiDAR projected depth
            proj_intensity: [B*V, 1, H, W] LiDAR projected intensity
            proj_mask: [B*V, 1, H, W] binary mask
        Returns:
            feat: [B*V, 4, H, W] LiDAR feature
        """
        # Normalize depth to [0, 1]
        depth_norm = (proj_depth - self.min_depth) / (self.max_depth - self.min_depth)
        depth_norm = depth_norm.clamp(0, 1) * proj_mask

        # Normalize intensity (raw intensity 0-255)
        intensity_norm = (proj_intensity / 255.0) * proj_mask

        # Depth difference signal (LiDAR - pred, signed)
        with torch.no_grad():
            depth_diff = (proj_depth - depth_maps) / (self.max_depth - self.min_depth)
            depth_diff = depth_diff.clamp(-1, 1) * proj_mask

        return torch.cat([depth_norm, intensity_norm, proj_mask, depth_diff], dim=1)


class DepthResidualHead(nn.Module):
    """
    Predict per-pixel depth residual, zero-initialized, bounded output.
    """
    def __init__(self, in_channels=4, hidden=32, max_correction=10.0):
        super().__init__()
        self.max_correction = max_correction

        self.net = nn.Sequential(
            nn.Conv2d(in_channels + 1, hidden, 3, padding=1),  # +1 for depth_maps
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, hidden, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, hidden, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, 1, 3, padding=1),
        )
        # Zero init: start with no correction
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, depth_maps, lidar_feat_4ch):
        """
        Args:
            depth_maps: [B*V, 1, H, W] predicted depth in meters
            lidar_feat_4ch: [B*V, 4, H, W] LiDAR features
        Returns:
            residual: [B*V, 1, H, W] bounded depth correction in meters
        """
        x = torch.cat([depth_maps, lidar_feat_4ch], dim=1)
        residual = self.net(x)
        residual = torch.tanh(residual) * self.max_correction
        return residual


class LidarFeatResidualHead(nn.Module):
    """
    Predict per-pixel 16D lidar_feat residual, zero-initialized.
    """
    def __init__(self, lidar_feat_dim=16, lidar_feat_channels=4, hidden=64):
        super().__init__()
        self.lidar_feat_dim = lidar_feat_dim

        self.net = nn.Sequential(
            nn.Conv2d(lidar_feat_channels + 1, hidden, 3, padding=1),  # +1 for depth_maps
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, hidden, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, lidar_feat_dim, 3, padding=1),
        )
        # Zero init: start with no correction
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, depth_maps, lidar_feat_4ch):
        """
        Args:
            depth_maps: [B*V, 1, H, W] predicted depth in meters
            lidar_feat_4ch: [B*V, 4, H, W] LiDAR features
        Returns:
            residual: [B*V, H, W, 16] 16D feature residual
        """
        x = torch.cat([depth_maps, lidar_feat_4ch], dim=1)
        residual = self.net(x)  # [B*V, 16, H, W]
        return residual.permute(0, 2, 3, 1)  # [B*V, H, W, 16]
