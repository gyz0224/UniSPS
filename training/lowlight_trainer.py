"""Shared, import-safe low-light training and validation operations."""

import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

from loss.lowlight_loss import C_loss, L_TV, L_color, L_exp, L_spa, P_loss, R_loss
from loss.sasw_loss import SASWLoss
from metrics.image_quality import calculate_psnr, calculate_ssim
from training.lowlight_sampling import (
    gamma_correction,
    generate_mask_pair,
    generate_subimages,
)
from training.precision import TrainingPrecision


@dataclass(frozen=True)
class LowlightTrainerConfig:
    """Weights for the unchanged SPS low-light objective."""

    loss_weights: Sequence[float] = (1.0, 0.1, 0.1, 0.5)
    light_patch: int = 64
    mean_val: float = 0.5
    w_sem: float = 0.1
    w_iqa: float = 0.01
    prior_weight: float = 500.0
    sasw_weight: float = 0.5

    def __post_init__(self) -> None:
        if len(self.loss_weights) != 4:
            raise ValueError("loss_weights must contain four values")
        if self.light_patch <= 0:
            raise ValueError("light_patch must be positive")


def seed_torch(seed: int) -> None:
    """Apply the original Python/NumPy/PyTorch seed policy."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def resolve_lowlight_device(
    requested: Optional[str] = None, gpu_mode: bool = True
) -> torch.device:
    """Resolve legacy ``gpu_mode`` plus the new explicit device override."""
    if requested:
        device = torch.device(requested)
    elif gpu_mode:
        if not torch.cuda.is_available():
            raise RuntimeError("No GPU found; pass --device cpu to run on CPU")
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"CUDA requested but unavailable: {device}")
    return device


def parse_epoch_list(spec: str, max_epoch: int) -> set[int]:
    """Parse comma-separated epochs and inclusive ranges."""
    epochs: set[int] = set()
    if not spec:
        return epochs
    for part in spec.split(","):
        token = part.strip()
        if not token:
            continue
        if "-" in token:
            try:
                start_text, end_text = token.split("-", 1)
                start, end = int(start_text), int(end_text)
            except ValueError:
                print(f"Warning: invalid range '{token}' in save_epochs; skipped")
                continue
            if start > end:
                start, end = end, start
            epochs.update(range(max(1, start), min(max_epoch, end) + 1))
            continue
        try:
            epoch = int(token)
        except ValueError:
            print(f"Warning: invalid epoch '{token}' in save_epochs; skipped")
            continue
        if 1 <= epoch <= max_epoch:
            epochs.add(epoch)
    return epochs


def lowlight_state_dict(model: nn.Module) -> Dict[str, torch.Tensor]:
    """Return legacy checkpoint contents without frozen semantic weights."""
    return {
        key: value
        for key, value in model.state_dict().items()
        if "sem_net" not in key
    }


def save_lowlight_checkpoint(path: os.PathLike | str, model: nn.Module) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    torch.save(lowlight_state_dict(model), target)


def load_lowlight_checkpoint(
    path: os.PathLike | str,
    model: nn.Module,
    map_location: Any = "cpu",
) -> torch.nn.modules.module._IncompatibleKeys:
    state = torch.load(path, map_location=map_location)
    if isinstance(state, Mapping) and "model" in state:
        state = state["model"]
    if not isinstance(state, Mapping):
        raise TypeError(f"Low-light checkpoint must contain a state mapping: {path}")
    return model.load_state_dict(state, strict=False)


class LowlightTrainer:
    """Execute the original neighboring-pixel SPS low-light update."""

    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        clip_criterion: nn.Module,
        config: LowlightTrainerConfig,
        device: torch.device,
        amp_enabled: bool = False,
        amp_dtype: str = "bfloat16",
    ) -> None:
        self.model = model
        self.optimizer = optimizer
        self.clip_criterion = clip_criterion
        self.config = config
        self.device = device
        self.precision = TrainingPrecision(
            device=device, enabled=amp_enabled, dtype=amp_dtype
        )
        self.sasw = SASWLoss().to(device)
        self.exposure = L_exp(config.light_patch, config.mean_val).to(device)
        self.color = L_color().to(device)
        self.spatial = L_spa().to(device)
        self.tv = L_TV().to(device)

    def train_step(self, input_image: torch.Tensor) -> Dict[str, float]:
        input_image = input_image.to(self.device, non_blocking=True)
        self.model.train()
        self.model.sem_net.eval()
        with torch.no_grad():
            semantic_features = self.model.extract_semantics(input_image)
        mask1, mask2 = generate_mask_pair(input_image)
        image1 = generate_subimages(input_image, mask1)
        image2 = gamma_correction(generate_subimages(input_image, mask2))

        with self.precision.autocast():
            l1, r1, x1, i1 = self.model(image1, sem_feats=semantic_features)
            _, r2, _, _ = self.model(image2, sem_feats=semantic_features)
            _, _, _, full_output = self.model(
                input_image, sem_feats=semantic_features
            )
        sub_r1 = generate_subimages(full_output, mask1)
        sub_r2 = generate_subimages(full_output, mask2)

        consistency = C_loss(r1, r2) + 0.5 * C_loss(r1 - r2, sub_r1 - sub_r2)
        reconstruction = R_loss(l1, r1, image1, x1)
        prior = P_loss(image1, x1)
        with self.precision.autocast():
            sasw = self.sasw(x1, image1)
            lowlight = (
                self.config.loss_weights[0] * self.exposure(i1)
                + self.config.loss_weights[1] * torch.mean(self.spatial(x1, i1))
                + self.config.loss_weights[2] * self.tv(i1)
                + self.config.loss_weights[3] * torch.mean(self.color(i1))
            )
            semantic, iqa = self.clip_criterion(input_image, full_output)
        total = (
            consistency
            + reconstruction
            + self.config.prior_weight * prior
            + lowlight
            + self.config.sasw_weight * sasw
            + self.config.w_sem * semantic
            + self.config.w_iqa * iqa
        )
        if not torch.isfinite(total):
            raise FloatingPointError(
                "Non-finite low-light loss: "
                f"consistency={consistency.item()} reconstruction={reconstruction.item()} "
                f"prior={prior.item()} lowlight={lowlight.item()} sasw={sasw.item()} "
                f"semantic={semantic.item()} iqa={iqa.item()}"
            )
        self.optimizer.zero_grad(set_to_none=True)
        self.precision.backward_step(total, self.optimizer)
        self.precision.update()
        return {
            "total": total.detach().item(),
            "consistency": consistency.detach().item(),
            "reconstruction": reconstruction.detach().item(),
            "prior": prior.detach().item(),
            "lowlight": lowlight.detach().item(),
            "sasw": sasw.detach().item(),
            "semantic": semantic.detach().item(),
            "iqa": iqa.detach().item(),
        }

    def train_epoch(self, loader, epoch: int, log_interval: int = 100) -> float:
        accumulated = 0.0
        for iteration, batch in enumerate(loader, 1):
            metrics = self.train_step(batch[0])
            accumulated += metrics["total"]
            if log_interval and iteration % log_interval == 0:
                print(
                    f"===> Epoch[{epoch}]({iteration}/{len(loader)}): "
                    f"Loss: {accumulated:.4f} || Learning rate: "
                    f"lr={self.optimizer.param_groups[0]['lr']}."
                )
                accumulated = 0.0
        return accumulated


def evaluate_paired_lowlight(
    model: nn.Module,
    loader,
    reference_dir: os.PathLike | str,
    device: torch.device,
    lpips_model: Optional[nn.Module] = None,
) -> Dict[str, float]:
    """Compute the original PSNR/SSIM/LPIPS validation metrics."""
    if lpips_model is None:
        try:
            import lpips
        except ImportError as exc:
            raise RuntimeError("Paired validation requires the lpips package") from exc
        lpips_model = lpips.LPIPS(net="alex").to(device)
    else:
        import lpips

    model.eval()
    totals = {"psnr": 0.0, "ssim": 0.0, "lpips": 0.0}
    count = 0
    print(f"  Validating {len(loader)} images...")
    for batch in loader:
        count += 1
        input_image, names = batch[0].to(device, non_blocking=True), batch[1]
        with torch.no_grad():
            _, _, _, enhanced = model(input_image)
            enhanced = torch.clamp(enhanced, 0, 1)
        base, extension = os.path.splitext(names[0])
        clean_base = base.split("_")[0]
        label_path = Path(reference_dir) / f"{clean_base}{extension}"
        if not label_path.is_file():
            raise FileNotFoundError(f"Validation reference does not exist: {label_path}")
        with Image.open(label_path) as image:
            reference = np.array(image.convert("RGB"))
        prediction = (
            enhanced.squeeze(0).permute(1, 2, 0).cpu().numpy() * 255.0
        ).astype(np.uint8)
        totals["psnr"] += float(calculate_psnr(prediction, reference))
        totals["ssim"] += float(calculate_ssim(prediction, reference))
        prediction_lpips = (enhanced * 2) - 1
        reference_lpips = lpips.im2tensor(reference).to(device)
        with torch.no_grad():
            score = lpips_model.forward(reference_lpips, prediction_lpips)
        totals["lpips"] += float(score.detach().mean().item())
    if count == 0:
        raise ValueError("Paired validation loader is empty")
    model.train()
    return {key: value / count for key, value in totals.items()}


def resize_for_no_reference(
    input_image: torch.Tensor, max_dimension: int = 1024
) -> torch.Tensor:
    """Preserve the no-reference validation resize policy."""
    _, _, height, width = input_image.shape
    if height <= max_dimension and width <= max_dimension:
        return input_image
    scale = max_dimension / max(height, width)
    new_height = max(8, (int(height * scale) // 8) * 8)
    new_width = max(8, (int(width * scale) // 8) * 8)
    return F.interpolate(
        input_image,
        size=(new_height, new_width),
        mode="bilinear",
        align_corners=False,
    )


def evaluate_no_reference_lowlight(
    model: nn.Module,
    loader,
    metrics: Mapping[str, nn.Module],
    device: torch.device,
) -> Dict[str, float]:
    """Compute NIQE, BRISQUE, and MUSIQ over enhanced validation images."""
    totals = {name: 0.0 for name in metrics}
    count = 0
    model.eval()
    with torch.no_grad():
        for batch in loader:
            input_image = resize_for_no_reference(
                batch[0].to(device, non_blocking=True)
            )
            enhanced = torch.clamp(model(input_image)[-1], 0, 1)
            for name, metric in metrics.items():
                totals[name] += float(metric(enhanced).item())
            count += 1
    if count == 0:
        raise ValueError("No-reference validation loader is empty")
    model.train()
    return {name: value / count for name, value in totals.items()}


__all__ = [
    "LowlightTrainer",
    "LowlightTrainerConfig",
    "evaluate_no_reference_lowlight",
    "evaluate_paired_lowlight",
    "load_lowlight_checkpoint",
    "lowlight_state_dict",
    "parse_epoch_list",
    "resolve_lowlight_device",
    "resize_for_no_reference",
    "save_lowlight_checkpoint",
    "seed_torch",
]
