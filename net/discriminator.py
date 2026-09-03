"""Training-only PatchGAN discriminators for clear and hazy domains."""

from typing import Tuple

import torch
import torch.nn as nn


class PatchDiscriminator(nn.Module):
    """Return an unconstrained LSGAN patch score for ``[B,3,H,W]`` images."""

    def __init__(self, in_channels: int = 3, base_channels: int = 64, layers: int = 3) -> None:
        super().__init__()
        if layers < 2:
            raise ValueError("layers must be at least 2")
        modules = []
        current = in_channels
        for index in range(layers):
            output = min(base_channels * (2**index), base_channels * 8)
            stride = 2 if index < layers - 1 else 1
            conv = nn.Conv2d(current, output, 4, stride=stride, padding=1)
            modules.extend([conv, nn.LeakyReLU(0.2, inplace=True)])
            current = output
        modules.append(nn.Conv2d(current, 1, 4, stride=1, padding=1))
        self.body = nn.Sequential(*modules)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        if image.ndim != 4 or image.shape[1] != 3:
            raise ValueError(f"image must have shape [B,3,H,W], got {tuple(image.shape)}")
        return self.body(image)


def build_dehaze_discriminators(
    base_channels: int = 64, layers: int = 3
) -> Tuple[PatchDiscriminator, PatchDiscriminator]:
    """Build ``(D_clear, D_hazy)`` with independent parameters."""
    return (
        PatchDiscriminator(3, base_channels, layers),
        PatchDiscriminator(3, base_channels, layers),
    )
