"""Training-only lightweight relative-depth estimator."""

from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class _ConvBlock(nn.Sequential):
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1) -> None:
        super().__init__(
            nn.Conv2d(in_channels, out_channels, 3, stride=stride, padding=1, bias=False),
            nn.GroupNorm(1, out_channels),
            nn.GELU(),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.GroupNorm(1, out_channels),
            nn.GELU(),
        )


class DepthEstimationNet(nn.Module):
    """Estimate bounded relative depth from a clean RGB image.

    Input shape: ``[B,3,H,W]``. Output shape: ``[B,1,H,W]`` in
    ``[depth_min, depth_max]``. This network is used only during training.
    """

    def __init__(
        self,
        depth_min: float = 0.2,
        depth_max: float = 5.0,
        channels: Tuple[int, int, int] = (32, 64, 96),
    ) -> None:
        super().__init__()
        if not 0.0 <= depth_min < depth_max:
            raise ValueError("depth range must satisfy 0 <= min < max")
        if len(channels) != 3 or any(channel <= 0 for channel in channels):
            raise ValueError("channels must contain three positive values")
        self.depth_min = float(depth_min)
        self.depth_max = float(depth_max)
        c0, c1, c2 = channels
        self.enc0 = _ConvBlock(3, c0)
        self.enc1 = _ConvBlock(c0, c1, stride=2)
        self.enc2 = _ConvBlock(c1, c2, stride=2)
        self.bottleneck = _ConvBlock(c2, c2)
        self.dec1 = _ConvBlock(c2 + c1, c1)
        self.dec0 = _ConvBlock(c1 + c0, c0)
        self.output = nn.Conv2d(c0, 1, kernel_size=3, padding=1)

    def set_depth_range(self, depth_min: float, depth_max: float) -> None:
        """Switch the output range between indoor and real joint batches."""
        if not 0.0 <= depth_min < depth_max:
            raise ValueError("depth range must satisfy 0 <= min < max")
        self.depth_min = float(depth_min)
        self.depth_max = float(depth_max)

    def forward(self, clean: torch.Tensor) -> torch.Tensor:
        if clean.ndim != 4 or clean.shape[1] != 3:
            raise ValueError(f"clean must have shape [B,3,H,W], got {tuple(clean.shape)}")
        enc0 = self.enc0(clean)
        enc1 = self.enc1(enc0)
        enc2 = self.bottleneck(self.enc2(enc1))
        dec1 = F.interpolate(enc2, size=enc1.shape[-2:], mode="bilinear", align_corners=False)
        dec1 = self.dec1(torch.cat([dec1, enc1], dim=1))
        dec0 = F.interpolate(dec1, size=enc0.shape[-2:], mode="bilinear", align_corners=False)
        dec0 = self.dec0(torch.cat([dec0, enc0], dim=1))
        normalized = torch.sigmoid(self.output(dec0))
        return self.depth_min + normalized * (self.depth_max - self.depth_min)
