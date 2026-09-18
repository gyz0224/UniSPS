"""Numerically safe losses for D4+-style unpaired dehazing."""

from dataclasses import dataclass
from typing import Callable, Dict, Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass(frozen=True)
class DehazeLossWeights:
    cycle: float = 1.0
    gan: float = 0.2
    scattering: float = 1.0
    contrast: float = 1e-4
    semantic: float = 0.05


def cycle_loss(
    clean: torch.Tensor,
    clean_cycle: torch.Tensor,
    hazy: torch.Tensor,
    hazy_cycle: torch.Tensor,
    clean_mask: Optional[torch.Tensor] = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return total, clean-domain, and hazy-domain L1 cycle losses."""
    if clean_mask is not None:
        valid = 1.0 - clean_mask.detach()
        clean = clean * valid
        clean_cycle = clean_cycle * valid
    clean_term = F.l1_loss(clean_cycle, clean)
    hazy_term = F.l1_loss(hazy_cycle, hazy)
    return clean_term + hazy_term, clean_term, hazy_term


def lsgan_discriminator_loss(real_scores: torch.Tensor, fake_scores: torch.Tensor) -> torch.Tensor:
    """Least-squares discriminator loss for one image domain."""
    return 0.5 * (F.mse_loss(real_scores, torch.ones_like(real_scores)) + F.mse_loss(
        fake_scores, torch.zeros_like(fake_scores)
    ))


def lsgan_generator_loss(*fake_scores: torch.Tensor) -> torch.Tensor:
    """Mean least-squares generator loss over one or more domains."""
    if not fake_scores:
        raise ValueError("At least one fake score tensor is required")
    return torch.stack(
        [F.mse_loss(scores, torch.ones_like(scores)) for scores in fake_scores]
    ).mean()


def scattering_loss(
    predicted_beta: torch.Tensor,
    target_beta: torch.Tensor,
    beta_min: float,
    beta_max: float,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Normalized beta pseudo-supervision loss."""
    denominator = max(float(beta_max - beta_min), eps)
    return F.mse_loss(
        predicted_beta.float(), target_beta.detach().float()
    ) / denominator


def depth_pseudo_loss(
    predicted_depth: torch.Tensor,
    pseudo_depth: torch.Tensor,
    depth_min: float,
    depth_max: float,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Normalized training-only DepthNet pseudo-supervision loss."""
    denominator = max(float(depth_max - depth_min), eps)
    return F.l1_loss(
        predicted_depth.float(), pseudo_depth.detach().float()
    ) / denominator


def semantic_consistency_loss(
    semantic_encoder: Callable[[torch.Tensor], torch.Tensor],
    hazy: torch.Tensor,
    clean_prediction: torch.Tensor,
) -> torch.Tensor:
    """CLIP semantic consistency while retaining gradients to the prediction."""
    with torch.no_grad():
        hazy_embedding = semantic_encoder(hazy)
    clean_embedding = semantic_encoder(clean_prediction)
    hazy_embedding = F.normalize(hazy_embedding.float(), dim=-1, eps=1e-6)
    clean_embedding = F.normalize(clean_embedding.float(), dim=-1, eps=1e-6)
    return (1.0 - F.cosine_similarity(hazy_embedding, clean_embedding, dim=-1)).mean()


class VGG19FeatureExtractor(nn.Module):
    """Frozen five-slice VGG19 feature extractor used only during training.

    Set ``pretrained=False`` for offline structural tests. Production training
    defaults to ImageNet weights and reports a clear torchvision/download error
    if those weights are unavailable.
    """

    def __init__(self, pretrained: bool = True) -> None:
        super().__init__()
        try:
            from torchvision import models

            weights = models.VGG19_Weights.IMAGENET1K_V1 if pretrained else None
            features = models.vgg19(weights=weights).features
        except Exception as exc:
            raise RuntimeError(
                "Unable to construct VGG19 perceptual features. Ensure torchvision "
                "is installed and ImageNet weights are cached or downloadable."
            ) from exc

        boundaries = ((0, 2), (2, 7), (7, 12), (12, 21), (21, 30))
        self.slices = nn.ModuleList(
            [nn.Sequential(*[features[index] for index in range(start, end)]) for start, end in boundaries]
        )
        self.eval()
        self.requires_grad_(False)

    def train(self, mode: bool = True):
        # VGG is always frozen/eval even when a parent module enters train mode.
        return super().train(False)

    def forward(self, image: torch.Tensor) -> Sequence[torch.Tensor]:
        outputs = []
        feature = image
        for block in self.slices:
            feature = block(feature)
            outputs.append(feature)
        return outputs


class DualContrastivePerceptualLoss(nn.Module):
    """D4+ dual-domain perceptual ratios over five frozen feature levels."""

    def __init__(
        self,
        feature_extractor: Optional[nn.Module] = None,
        weights: Sequence[float] = (0.0, 0.4, 0.6, 0.0, 1.0),
        eps: float = 1e-7,
    ) -> None:
        super().__init__()
        if len(weights) != 5:
            raise ValueError("Dual contrastive VGG weights must contain five values")
        self.feature_extractor = feature_extractor or VGG19FeatureExtractor()
        self.feature_extractor.requires_grad_(False)
        self.feature_extractor.eval()
        self.register_buffer("level_weights", torch.tensor(weights, dtype=torch.float32))
        self.eps = eps

    def train(self, mode: bool = True):
        super().train(mode)
        self.feature_extractor.eval()
        return self

    @staticmethod
    def _distance(first: torch.Tensor, second: torch.Tensor) -> torch.Tensor:
        return F.l1_loss(first, second)

    def forward(
        self,
        clean_prediction: torch.Tensor,
        haze_prediction: torch.Tensor,
        clean_reference: torch.Tensor,
        hazy_reference: torch.Tensor,
    ) -> torch.Tensor:
        clean_pred_features = self.feature_extractor(clean_prediction)
        haze_pred_features = self.feature_extractor(haze_prediction)
        with torch.no_grad():
            clean_ref_features = self.feature_extractor(clean_reference)
            hazy_ref_features = self.feature_extractor(hazy_reference)

        groups = (
            clean_pred_features,
            haze_pred_features,
            clean_ref_features,
            hazy_ref_features,
        )
        if any(len(group) != len(self.level_weights) for group in groups):
            raise ValueError("Feature extractor must return exactly five feature levels")

        loss = clean_prediction.new_zeros(())
        for index, weight in enumerate(self.level_weights):
            clean_ratio = self._distance(
                clean_pred_features[index], clean_ref_features[index]
            ) / (
                self._distance(clean_pred_features[index], hazy_ref_features[index])
                + self.eps
            )
            haze_ratio = self._distance(
                haze_pred_features[index], hazy_ref_features[index]
            ) / (
                self._distance(haze_pred_features[index], clean_ref_features[index])
                + self.eps
            )
            loss = loss + weight.to(loss) * (clean_ratio + haze_ratio)
        return loss


def generator_total_loss(
    cycle: torch.Tensor,
    gan: torch.Tensor,
    scattering: torch.Tensor,
    contrast: torch.Tensor,
    semantic: torch.Tensor,
    weights: Optional[DehazeLossWeights] = None,
) -> tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """Combine the five required generator objectives and return weighted terms."""
    weights = weights or DehazeLossWeights()
    terms = {
        "cycle": weights.cycle * cycle,
        "gan_G": weights.gan * gan,
        "scattering": weights.scattering * scattering,
        "contrast": weights.contrast * contrast,
        "semantic": weights.semantic * semantic,
    }
    return sum(terms.values(), cycle.new_zeros(())), terms
