"""Losses used by the SPS-Netpro training entrypoints."""

from .dehaze_loss import (
    DehazeLossWeights,
    DualContrastivePerceptualLoss,
    VGG19FeatureExtractor,
    cycle_loss,
    depth_pseudo_loss,
    generator_total_loss,
    lsgan_discriminator_loss,
    lsgan_generator_loss,
    scattering_loss,
    semantic_consistency_loss,
)
from .lowlight_clip import CLIPLoss
from .lowlight_loss import C_loss, L_TV, L_color, L_exp, L_spa, P_loss, R_loss
from .sasw_loss import CosineLoss, SASWLoss, WHTBlock

__all__ = [
    "DehazeLossWeights",
    "DualContrastivePerceptualLoss",
    "VGG19FeatureExtractor",
    "cycle_loss",
    "depth_pseudo_loss",
    "generator_total_loss",
    "lsgan_discriminator_loss",
    "lsgan_generator_loss",
    "scattering_loss",
    "semantic_consistency_loss",
    "CLIPLoss",
    "C_loss",
    "CosineLoss",
    "L_TV",
    "L_color",
    "L_exp",
    "L_spa",
    "P_loss",
    "R_loss",
    "SASWLoss",
    "WHTBlock",
]
