"""Directory-level PSNR/SSIM/LPIPS evaluation used by ``measure.py``."""

import glob
from pathlib import Path
from typing import Mapping, Optional, Sequence, Union

import cv2
import numpy as np
import torch
from PIL import Image

from metrics.image_quality import calculate_psnr, calculate_ssim


IMAGE_SUFFIXES = {".bmp", ".jpeg", ".jpg", ".png"}


def build_lpips_model(
    device: Union[str, torch.device],
) -> torch.nn.Module:
    """Build the shared AlexNet LPIPS evaluator on the requested device."""
    try:
        import lpips
    except ImportError as exc:
        raise RuntimeError("LPIPS evaluation requires the lpips package") from exc
    model = lpips.LPIPS(net="alex").to(torch.device(device))
    model.eval()
    return model


def calculate_lpips(
    prediction: np.ndarray,
    reference: np.ndarray,
    model: torch.nn.Module,
    device: Union[str, torch.device],
) -> float:
    """Calculate LPIPS for two same-size uint8 RGB images."""
    first = np.asarray(prediction)
    second = np.asarray(reference)
    if first.shape != second.shape:
        raise ValueError("LPIPS inputs must have the same dimensions.")
    if first.ndim != 3 or first.shape[2] != 3:
        raise ValueError("LPIPS expects RGB images with three channels.")
    if first.dtype != np.uint8 or second.dtype != np.uint8:
        raise ValueError("LPIPS expects uint8 RGB images in [0,255].")

    resolved_device = torch.device(device)

    def tensor(image: np.ndarray) -> torch.Tensor:
        value = torch.from_numpy(np.ascontiguousarray(image))
        value = value.permute(2, 0, 1).unsqueeze(0)
        return value.to(resolved_device, dtype=torch.float32).div(127.5).sub(1.0)

    with torch.no_grad():
        score = model.forward(tensor(second), tensor(first))
    return float(score.mean().item())


def resolve_image_files(source: Union[str, Sequence[str]]) -> list[str]:
    if isinstance(source, (list, tuple)):
        candidates = []
        for item in source:
            matches = glob.glob(str(item))
            candidates.extend(matches or [str(item)])
    else:
        candidates = glob.glob(source)
    files = sorted(
        {
            str(path)
            for item in candidates
            if (path := Path(item)).is_file()
            and path.suffix.casefold() in IMAGE_SUFFIXES
        }
    )
    if not files:
        raise ValueError(f"No images found for pattern: {source}")
    return files


def _reference_index(reference_dir: Union[str, Path]) -> dict[str, list[Path]]:
    root = Path(reference_dir)
    if not root.is_dir():
        raise FileNotFoundError(f"Reference directory does not exist: {root}")
    index: dict[str, list[Path]] = {}
    for path in root.iterdir():
        if path.is_file() and path.suffix.casefold() in IMAGE_SUFFIXES:
            index.setdefault(path.stem.casefold(), []).append(path)
    return index


def resolve_lowlight_reference(
    prediction: Union[str, Path],
    reference_dir: Union[str, Path],
    pairing: str = "same-name",
    reference_index: Optional[Mapping[str, Sequence[Path]]] = None,
) -> Path:
    """Resolve one low-light prediction to its paired reference image."""
    prediction = Path(prediction)
    reference_dir = Path(reference_dir)
    if pairing == "same-name":
        reference = reference_dir / prediction.name
        if not reference.is_file():
            raise FileNotFoundError(f"Reference image does not exist: {reference}")
        return reference
    if pairing != "sice":
        raise ValueError(f"Unsupported low-light pairing rule: {pairing}")

    scene_stem, separator, exposure = prediction.stem.rpartition("_")
    if not separator or not scene_stem or not exposure.isdigit():
        raise ValueError(
            "SICE prediction name must end in an exposure suffix such as "
            f"'10_1.JPG': {prediction.name}"
        )
    index = reference_index or _reference_index(reference_dir)
    matches = list(index.get(scene_stem.casefold(), ()))
    if not matches:
        raise FileNotFoundError(
            f"No SICE reference for prediction {prediction.name} in {reference_dir}"
        )
    if len(matches) > 1:
        names = ", ".join(sorted(path.name for path in matches))
        raise ValueError(
            f"Ambiguous SICE references for {prediction.name}: {names}"
        )
    return matches[0]


def evaluate_lpips_directory(
    image_source: Union[str, Sequence[str]],
    label_dir: Union[str, Path],
    device: Union[str, torch.device],
    pairing: str = "same-name",
):
    """Return average PSNR, SSIM, and LPIPS for enhanced images."""
    device = torch.device(device)
    loss_fn = build_lpips_model(device)
    totals = {"psnr": 0.0, "ssim": 0.0, "lpips": 0.0}
    files = resolve_image_files(image_source)
    reference_index = _reference_index(label_dir) if pairing == "sice" else None
    for item in files:
        with Image.open(item) as image:
            prediction_image = image.convert("RGB")
        reference_path = resolve_lowlight_reference(
            item,
            label_dir,
            pairing,
            reference_index,
        )
        with Image.open(reference_path) as image:
            reference_image = image.convert("RGB")
        width, height = reference_image.size
        prediction_lpips = cv2.resize(
            np.array(prediction_image),
            (width, height),
        )
        prediction_image = prediction_image.resize((width, height))
        prediction = np.array(prediction_image, dtype=np.uint8)
        reference = np.array(reference_image, dtype=np.uint8)
        totals["psnr"] += float(calculate_psnr(prediction, reference))
        totals["ssim"] += float(calculate_ssim(prediction, reference))
        totals["lpips"] += calculate_lpips(
            prediction_lpips,
            reference,
            loss_fn,
            device,
        )
    count = len(files)
    return tuple(totals[name] / count for name in ("psnr", "ssim", "lpips"))


__all__ = [
    "build_lpips_model",
    "calculate_lpips",
    "evaluate_lpips_directory",
    "resolve_image_files",
    "resolve_lowlight_reference",
]
