"""Training-only physical haze rendering and bounded residual refinement."""

import torch
import torch.nn as nn


class PhysicalHazeRenderer(nn.Module):
    """Render ``H=J*exp(-beta*d)+A*(1-exp(-beta*d))``.

    ``clean`` is ``[B,3,H,W]``, ``depth`` is ``[B,1,H,W]``, ``beta`` is
    ``[B,1,1,1]``, and ``atmosphere`` is ``[B,3,1,1]``.
    """

    def __init__(self, transmission_min: float = 0.05, transmission_max: float = 0.95) -> None:
        super().__init__()
        if not 0.0 < transmission_min < transmission_max <= 1.0:
            raise ValueError("transmission range must satisfy 0 < min < max <= 1")
        self.transmission_min = transmission_min
        self.transmission_max = transmission_max

    def forward(
        self,
        clean: torch.Tensor,
        depth: torch.Tensor,
        beta: torch.Tensor,
        atmosphere: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if clean.ndim != 4 or clean.shape[1] != 3:
            raise ValueError("clean must have shape [B,3,H,W]")
        expected_depth = (clean.shape[0], 1, clean.shape[2], clean.shape[3])
        if depth.shape != expected_depth:
            raise ValueError(f"depth must have shape {expected_depth}, got {tuple(depth.shape)}")
        if beta.shape != (clean.shape[0], 1, 1, 1):
            raise ValueError("beta must have shape [B,1,1,1]")
        if atmosphere.shape != (clean.shape[0], 3, 1, 1):
            raise ValueError("atmosphere must have shape [B,3,1,1]")
        transmission = torch.exp(-beta * depth).clamp(
            self.transmission_min, self.transmission_max
        )
        coarse = clean * transmission + atmosphere * (1.0 - transmission)
        return coarse.clamp(0.0, 1.0), transmission


class HazeRefineNet(nn.Module):
    """Refine coarse haze with a residual bounded by ``residual_scale``."""

    def __init__(self, channels: int = 32, residual_scale: float = 0.1) -> None:
        super().__init__()
        if channels <= 0:
            raise ValueError("channels must be positive")
        if not 0.0 < residual_scale <= 0.1:
            raise ValueError("residual_scale must be in (0, 0.1]")
        self.residual_scale = residual_scale
        self.body = nn.Sequential(
            nn.Conv2d(7, channels, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(channels, 3, kernel_size=3, padding=1),
        )

    def forward(
        self, clean: torch.Tensor, coarse: torch.Tensor, transmission: torch.Tensor
    ) -> torch.Tensor:
        if clean.shape != coarse.shape or clean.ndim != 4 or clean.shape[1] != 3:
            raise ValueError("clean and coarse must share shape [B,3,H,W]")
        if transmission.shape != (clean.shape[0], 1, clean.shape[2], clean.shape[3]):
            raise ValueError("transmission must have shape [B,1,H,W]")
        residual = self.body(torch.cat([clean, coarse, transmission], dim=1))
        return (coarse + self.residual_scale * torch.tanh(residual)).clamp(0.0, 1.0)
