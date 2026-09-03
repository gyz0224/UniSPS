"""SPS-based dehazing parameter prediction and atmospheric physics."""

from dataclasses import asdict, dataclass, fields
from typing import Any, Mapping, Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from net.blocks import OverlapPatchEmbed, SGA, TransformerBlock
from net.pafm import PAFM


@dataclass(frozen=True)
class DehazeConfig:
    """Configuration for the inference-time dehazing branch."""

    stage_channels: Tuple[int, int, int] = (64, 96, 128)
    blocks_per_stage: Tuple[int, int, int] = (1, 1, 2)
    decoder_blocks: int = 1
    shared_channels: int = 64
    num_heads: int = 1
    ffn_expansion_factor: float = 2.66
    bias: bool = False
    layer_norm_type: str = "WithBias"
    transmission_min: float = 0.05
    transmission_max: float = 0.95
    beta_min: float = 0.6
    beta_max: float = 1.8
    atmosphere_mode: str = "max"
    use_semantics: bool = True
    dcp_window: int = 15
    dcp_top_percent: float = 0.001
    dcp_omega: float = 0.95
    eps: float = 1e-6

    def __post_init__(self) -> None:
        if len(self.stage_channels) != 3 or any(c <= 0 for c in self.stage_channels):
            raise ValueError("stage_channels must contain three positive channel counts")
        if len(self.blocks_per_stage) != 3 or any(n < 0 for n in self.blocks_per_stage):
            raise ValueError("blocks_per_stage must contain three non-negative counts")
        if self.decoder_blocks < 0:
            raise ValueError("decoder_blocks must be non-negative")
        if not 0.0 < self.transmission_min < self.transmission_max <= 1.0:
            raise ValueError("transmission range must satisfy 0 < min < max <= 1")
        if not 0.0 < self.beta_min < self.beta_max:
            raise ValueError("beta range must satisfy 0 < min < max")
        if self.atmosphere_mode not in {"max", "dcp"}:
            raise ValueError("atmosphere_mode must be 'max' or 'dcp'")
        if self.dcp_window <= 0 or self.dcp_window % 2 == 0:
            raise ValueError("dcp_window must be a positive odd integer")
        if not 0.0 < self.dcp_top_percent <= 1.0:
            raise ValueError("dcp_top_percent must be in (0, 1]")

    @classmethod
    def from_value(
        cls, value: Optional[Union["DehazeConfig", Mapping[str, Any]]]
    ) -> "DehazeConfig":
        if value is None:
            return cls()
        if isinstance(value, cls):
            return value
        if not isinstance(value, Mapping):
            raise TypeError("dehaze_config must be a DehazeConfig, mapping, or None")
        allowed = {field.name for field in fields(cls)}
        unknown = sorted(set(value) - allowed)
        if unknown:
            raise ValueError(f"Unknown dehaze configuration keys: {unknown}")
        normalized = dict(value)
        for key in ("stage_channels", "blocks_per_stage"):
            if key in normalized:
                normalized[key] = tuple(normalized[key])
        return cls(**normalized)

    def to_dict(self) -> Mapping[str, Any]:
        return asdict(self)


class AtmosphereEstimator(nn.Module):
    """Parameter-free per-image atmospheric-light estimator.

    Args:
        mode: ``"max"`` for per-channel global maxima or ``"dcp"`` for
            dark-channel-prior airlight selection.

    Input:
        image: Float tensor ``[B, 3, H, W]`` in ``[0, 1]``.

    Returns:
        Atmospheric light tensor ``[B, 3, 1, 1]``.
    """

    def __init__(
        self,
        mode: str = "max",
        window_size: int = 15,
        top_percent: float = 0.001,
        omega: float = 0.95,
        eps: float = 1e-6,
    ) -> None:
        super().__init__()
        if mode not in {"max", "dcp"}:
            raise ValueError("AtmosphereEstimator mode must be 'max' or 'dcp'")
        if window_size <= 0 or window_size % 2 == 0:
            raise ValueError("window_size must be a positive odd integer")
        if not 0.0 < top_percent <= 1.0:
            raise ValueError("top_percent must be in (0, 1]")
        self.mode = mode
        self.window_size = window_size
        self.top_percent = top_percent
        self.omega = omega
        self.eps = eps

    def _validate_image(self, image: torch.Tensor) -> None:
        if image.ndim != 4 or image.shape[1] != 3:
            raise ValueError(
                f"Expected image shape [B,3,H,W], received {tuple(image.shape)}"
            )
        if image.shape[-2] == 0 or image.shape[-1] == 0:
            raise ValueError("Image spatial dimensions must be non-empty")

    def dark_channel(self, image: torch.Tensor) -> torch.Tensor:
        """Return the local dark channel with shape ``[B,1,H,W]``."""
        self._validate_image(image)
        channel_min = image.amin(dim=1, keepdim=True)
        pad = self.window_size // 2
        return -F.max_pool2d(
            -channel_min,
            kernel_size=self.window_size,
            stride=1,
            padding=pad,
        )

    def _dcp_atmosphere(self, image: torch.Tensor) -> torch.Tensor:
        dark = self.dark_channel(image).flatten(2)
        pixels = image.flatten(2)
        count = max(1, int(dark.shape[-1] * self.top_percent))
        indices = dark.topk(k=count, dim=-1, largest=True, sorted=False).indices
        candidates = pixels.gather(2, indices.expand(-1, 3, -1))
        return candidates.amax(dim=2, keepdim=True).unsqueeze(-1).clamp(0.0, 1.0)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        self._validate_image(image)
        if self.mode == "max":
            return image.amax(dim=(-2, -1), keepdim=True)
        return self._dcp_atmosphere(image)

    def estimate_transmission(
        self, image: torch.Tensor, atmosphere: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Estimate DCP transmission ``[B,1,H,W]`` for pseudo-depth targets."""
        self._validate_image(image)
        atmosphere = self(image) if atmosphere is None else atmosphere
        if atmosphere.shape != (image.shape[0], 3, 1, 1):
            raise ValueError(
                "atmosphere must have shape [B,3,1,1], received "
                f"{tuple(atmosphere.shape)}"
            )
        normalized = image / atmosphere.clamp_min(self.eps)
        return (1.0 - self.omega * self.dark_channel(normalized)).clamp(0.0, 1.0)


def recover_clean_image(
    hazy: torch.Tensor,
    transmission: torch.Tensor,
    atmosphere: torch.Tensor,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Recover clean radiance from ``H=Jt+A(1-t)``."""
    return ((hazy - atmosphere) / transmission.clamp_min(eps) + atmosphere).clamp(0.0, 1.0)


def depth_from_transmission(
    transmission: torch.Tensor, beta: torch.Tensor, eps: float = 1e-6
) -> torch.Tensor:
    """Convert transmission ``[B,1,H,W]`` and beta ``[B,1,1,1]`` to depth."""
    return torch.log(transmission.clamp_min(eps)) / (-beta.clamp_min(eps))


def _transformer_stack(channels: int, count: int, cfg: DehazeConfig) -> nn.Sequential:
    return nn.Sequential(
        *[
            TransformerBlock(
                dim=channels,
                num_heads=cfg.num_heads,
                ffn_expansion_factor=cfg.ffn_expansion_factor,
                bias=cfg.bias,
                LayerNorm_type=cfg.layer_norm_type,
            )
            for _ in range(count)
        ]
    )


class DehazeParameterHead(nn.Module):
    """Predict transmission and global scattering density from SPS features.

    Inputs:
        hazy: Hazy RGB tensor ``[B,3,H,W]``.
        x_feat: Shared SPS feature tensor ``[B,shared_channels,H,W]``.
        atmosphere: Atmospheric light ``[B,3,1,1]``.
        sem_feats: Optional ``[RGB feature, CLIP 768-channel deep feature]``.

    Returns:
        ``(transmission, beta)`` with shapes ``[B,1,H,W]`` and
        ``[B,1,1,1]``, both bounded by configuration.
    """

    def __init__(
        self, config: Optional[Union[DehazeConfig, Mapping[str, Any]]] = None
    ) -> None:
        super().__init__()
        self.config = DehazeConfig.from_value(config)
        self.transmission_min = self.config.transmission_min
        self.transmission_max = self.config.transmission_max
        self.beta_min = self.config.beta_min
        self.beta_max = self.config.beta_max
        c0, c1, c2 = self.config.stage_channels

        self.shared_projection = (
            nn.Identity()
            if self.config.shared_channels == c0
            else nn.Conv2d(self.config.shared_channels, c0, kernel_size=1, bias=False)
        )
        self.physical_embed = OverlapPatchEmbed(6, c0, bias=self.config.bias)
        self.pafm = PAFM(c0)

        self.stage0 = _transformer_stack(c0, self.config.blocks_per_stage[0], self.config)
        self.down1 = nn.Conv2d(c0, c1, kernel_size=3, stride=2, padding=1, bias=self.config.bias)
        self.stage1 = _transformer_stack(c1, self.config.blocks_per_stage[1], self.config)
        self.down2 = nn.Conv2d(c1, c2, kernel_size=3, stride=2, padding=1, bias=self.config.bias)
        self.stage2 = _transformer_stack(c2, self.config.blocks_per_stage[2], self.config)

        if self.config.use_semantics:
            self.sga_stage0 = SGA(c0, 3, self.config.num_heads, self.config.bias)
            self.sga_stage2 = SGA(c2, 768, self.config.num_heads, self.config.bias)
        else:
            self.sga_stage0 = None
            self.sga_stage2 = None

        self.decode1 = nn.Conv2d(c2 + c1, c1, kernel_size=3, padding=1, bias=self.config.bias)
        self.decoder_stage1 = _transformer_stack(c1, self.config.decoder_blocks, self.config)
        self.decode0 = nn.Conv2d(c1 + c0, c0, kernel_size=3, padding=1, bias=self.config.bias)
        self.decoder_stage0 = _transformer_stack(c0, self.config.decoder_blocks, self.config)
        self.transmission_out = nn.Conv2d(c0, 1, kernel_size=3, padding=1, bias=True)

        beta_channels = c0 + c1 + c2
        beta_hidden = max(16, c1)
        self.beta_head = nn.Sequential(
            nn.Conv2d(beta_channels, beta_hidden, kernel_size=1, bias=True),
            nn.GELU(),
            nn.Conv2d(beta_hidden, 1, kernel_size=1, bias=True),
        )

    def set_physical_ranges(
        self,
        beta_min: float,
        beta_max: float,
        transmission_min: float,
        transmission_max: float,
    ) -> None:
        """Switch configured physical ranges for sequential joint-task batches."""
        if not 0.0 < beta_min < beta_max:
            raise ValueError("beta range must satisfy 0 < min < max")
        if not 0.0 < transmission_min < transmission_max <= 1.0:
            raise ValueError("transmission range must satisfy 0 < min < max <= 1")
        self.beta_min = float(beta_min)
        self.beta_max = float(beta_max)
        self.transmission_min = float(transmission_min)
        self.transmission_max = float(transmission_max)

    def _semantic(
        self, sem_feats: Optional[Sequence[Optional[torch.Tensor]]], index: int
    ) -> Optional[torch.Tensor]:
        if sem_feats is None or index >= len(sem_feats):
            return None
        return sem_feats[index]

    def forward(
        self,
        hazy: torch.Tensor,
        x_feat: torch.Tensor,
        atmosphere: torch.Tensor,
        sem_feats: Optional[Sequence[Optional[torch.Tensor]]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if hazy.ndim != 4 or hazy.shape[1] != 3:
            raise ValueError(f"hazy must have shape [B,3,H,W], got {tuple(hazy.shape)}")
        if x_feat.ndim != 4 or x_feat.shape[0] != hazy.shape[0]:
            raise ValueError("x_feat must be a batched 4D tensor aligned with hazy")
        if x_feat.shape[-2:] != hazy.shape[-2:]:
            raise ValueError("x_feat and hazy must share spatial dimensions")
        if atmosphere.shape != (hazy.shape[0], 3, 1, 1):
            raise ValueError("atmosphere must have shape [B,3,1,1]")

        atmosphere_map = atmosphere.expand(-1, -1, hazy.shape[-2], hazy.shape[-1])
        physical_input = torch.cat([hazy, atmosphere_map], dim=1)
        physical_feat = self.physical_embed(physical_input)
        shared_feat = self.shared_projection(x_feat)

        stage0 = self.stage0(self.pafm(shared_feat, physical_feat))
        semantic0 = self._semantic(sem_feats, 0)
        if self.sga_stage0 is not None and semantic0 is not None:
            stage0 = self.sga_stage0(stage0, semantic0)

        stage1 = self.stage1(self.down1(stage0))
        stage2 = self.stage2(self.down2(stage1))
        semantic2 = self._semantic(sem_feats, 1)
        if self.sga_stage2 is not None and semantic2 is not None:
            stage2 = self.sga_stage2(stage2, semantic2)

        pooled = torch.cat(
            [F.adaptive_avg_pool2d(stage, 1) for stage in (stage0, stage1, stage2)],
            dim=1,
        )
        beta_raw = self.beta_head(pooled)
        beta = self.beta_min + torch.sigmoid(beta_raw) * (
            self.beta_max - self.beta_min
        )

        decoded1 = F.interpolate(stage2, size=stage1.shape[-2:], mode="bilinear", align_corners=False)
        decoded1 = self.decoder_stage1(self.decode1(torch.cat([decoded1, stage1], dim=1)))
        decoded0 = F.interpolate(decoded1, size=stage0.shape[-2:], mode="bilinear", align_corners=False)
        decoded0 = self.decoder_stage0(self.decode0(torch.cat([decoded0, stage0], dim=1)))

        transmission_raw = self.transmission_out(decoded0)
        transmission = self.transmission_min + torch.sigmoid(transmission_raw) * (
            self.transmission_max - self.transmission_min
        )
        transmission = transmission.clamp(
            self.transmission_min, self.transmission_max
        )
        return transmission, beta


class SharedSpsEncoder(nn.Module):
    """Deployment view containing only ``N_net.patch_embed`` and ``encoder``."""

    def __init__(self, source: nn.Module) -> None:
        super().__init__()
        self.patch_embed = source.patch_embed
        self.encoder = source.encoder

    def encode(self, image: torch.Tensor) -> torch.Tensor:
        return self.encoder(self.patch_embed(image))

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        return self.encode(image)


class DehazeInferenceModel(nn.Module):
    """Deployment-only view of the shared SPS encoder and dehaze head."""

    def __init__(
        self,
        semantic_net: nn.Module,
        shared_encoder: nn.Module,
        atmosphere_estimator: AtmosphereEstimator,
        parameter_head: DehazeParameterHead,
        eps: float = 1e-6,
    ) -> None:
        super().__init__()
        self.sem_net = semantic_net
        self.N_net = SharedSpsEncoder(shared_encoder)
        self.atmosphere_estimator = atmosphere_estimator
        self.dehaze_head = parameter_head
        self.eps = eps

    def forward(
        self,
        hazy: torch.Tensor,
        sem_feats: Optional[Sequence[Optional[torch.Tensor]]] = None,
        atmosphere: Optional[torch.Tensor] = None,
        return_aux: bool = False,
    ):
        """Dehaze RGB, optionally reusing a supplied global atmosphere value."""
        if hazy.ndim != 4 or hazy.shape[1] != 3:
            raise ValueError(f"hazy must have shape [B,3,H,W], got {tuple(hazy.shape)}")
        sem_feats = self.sem_net(hazy) if sem_feats is None else sem_feats
        feature = self.N_net.encode(hazy)
        atmosphere = (
            self.atmosphere_estimator(hazy)
            if atmosphere is None
            else atmosphere
        )
        transmission, beta = self.dehaze_head(
            hazy, feature, atmosphere, sem_feats
        )
        clean = recover_clean_image(hazy, transmission, atmosphere, self.eps)
        depth = depth_from_transmission(transmission, beta, self.eps)
        if return_aux:
            return {
                "clean": clean,
                "transmission": transmission,
                "beta": beta,
                "depth_from_haze": depth,
                "atmosphere": atmosphere,
            }
        return clean
