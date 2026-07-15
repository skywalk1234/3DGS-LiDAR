import torch
import torch.nn as nn


class LidarDecoder(nn.Module):
    def __init__(self, feature_dim: int = 16):
        super().__init__()
        self.out_dim = 2
        self.net = nn.Sequential(
            nn.Linear(feature_dim + 3, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 2),
        )

    def forward(self, features, ray_dirs):
        x = torch.cat([features, ray_dirs], dim=-1)
        x = self.net(x)
        intensity, ray_drop = x.split(1, dim=-1)
        return intensity, ray_drop
