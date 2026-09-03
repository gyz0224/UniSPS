"""Full-reference and FADE directory evaluation for dehazing results."""

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

import cv2
import numpy as np
from PIL import Image

from metrics.fade import calculate_fade
from metrics.image_quality import calculate_psnr, calculate_ssim
from metrics.lpips_evaluator import build_lpips_model, calculate_lpips


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
PAIRING_MODES = {"ihaze", "same-stem", "sots", "sots-outdoor"}
ProgressCallback = Callable[[int, int, Path], None]


@dataclass(frozen=True)
class DehazeMetrics:
    samples: int
    psnr_db: float
    ssim: float
    ciede2000: float
    lpips: Optional[float] = None


@dataclass(frozen=True)
class FadeMetrics:
    samples: int
    fade: float


def _image_paths(directory: Path) -> list[Path]:
    if not directory.is_dir():
        raise FileNotFoundError(f"Image directory does not exist: {directory}")
    paths = sorted(
        path
        for path in directory.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )
    if not paths:
        raise FileNotFoundError(f"No supported images found in: {directory}")
    return paths


def _reference_index(directory: Path) -> dict[str, Path]:
    index: dict[str, Path] = {}
    for path in _image_paths(directory):
        if path.stem in index:
            raise ValueError(
                f"Reference stem is ambiguous: {path.stem!r} maps to both "
                f"{index[path.stem]} and {path}"
            )
        index[path.stem] = path
    return index


def _prediction_key(path: Path, pairing: str) -> str:
    if pairing == "same-stem":
        return path.stem
    if pairing == "ihaze":
        if not path.stem.endswith("_hazy"):
            raise ValueError(
                f"Invalid I-HAZE prediction name {path.name!r}; expected "
                "'<scene_id>_indoor_hazy.<ext>'"
            )
        return f"{path.stem[:-len('_hazy')]}_GT"
    if pairing == "sots-outdoor":
        fields = path.stem.split("_")
        if len(fields) != 3 or not fields[0].isdigit():
            raise ValueError(
                f"Invalid SOTS Outdoor prediction name {path.name!r}; "
                "expected '<scene_id>_<atmospheric_light>_<beta>.<ext>'"
            )
        try:
            float(fields[1])
            float(fields[2])
        except ValueError as error:
            raise ValueError(
                f"Invalid SOTS Outdoor prediction name {path.name!r}; "
                "atmospheric light and beta must be numeric"
            ) from error
        return fields[0]
    if pairing == "sots":
        scene_id, separator, haze_level = path.stem.rpartition("_")
        if not separator or not scene_id or not haze_level.isdigit():
            raise ValueError(
                f"Invalid SOTS prediction name {path.name!r}; expected "
                "'<scene_id>_<haze_level>.<ext>'"
            )
        return scene_id
    raise ValueError(
        f"Unknown pairing mode {pairing!r}; expected one of "
        f"{sorted(PAIRING_MODES)}"
    )


def pair_dehaze_images(
    prediction_dir: Path,
    reference_dir: Path,
    pairing: str,
) -> list[tuple[Path, Path]]:
    """Pair predictions with clean references using a strict filename policy."""
    predictions = _image_paths(Path(prediction_dir))
    references = _reference_index(Path(reference_dir))
    pairs: list[tuple[Path, Path]] = []
    missing: list[tuple[Path, str]] = []
    for prediction in predictions:
        key = _prediction_key(prediction, pairing)
        reference = references.get(key)
        if reference is None:
            missing.append((prediction, key))
        else:
            pairs.append((prediction, reference))
    if missing:
        examples = ", ".join(
            f"{prediction.name}->{key}" for prediction, key in missing[:5]
        )
        raise FileNotFoundError(
            f"{len(missing)} predictions have no matching clean reference "
            f"in {reference_dir}; examples: {examples}"
        )
    return pairs


def calculate_ciede2000_lab(lab1: np.ndarray, lab2: np.ndarray) -> np.ndarray:
    """Return per-element CIEDE2000 (Delta E 00) for two CIELAB arrays."""
    first = np.asarray(lab1, dtype=np.float64)
    second = np.asarray(lab2, dtype=np.float64)
    if first.shape != second.shape or first.shape[-1] != 3:
        raise ValueError("CIELAB inputs must have the same shape ending in 3")

    l1, a1, b1 = np.moveaxis(first, -1, 0)
    l2, a2, b2 = np.moveaxis(second, -1, 0)

    c1 = np.hypot(a1, b1)
    c2 = np.hypot(a2, b2)
    c_bar = (c1 + c2) / 2.0
    c_bar7 = c_bar**7
    g = 0.5 * (1.0 - np.sqrt(c_bar7 / (c_bar7 + 25.0**7)))

    a1_prime = (1.0 + g) * a1
    a2_prime = (1.0 + g) * a2
    c1_prime = np.hypot(a1_prime, b1)
    c2_prime = np.hypot(a2_prime, b2)
    h1_prime = np.mod(np.degrees(np.arctan2(b1, a1_prime)), 360.0)
    h2_prime = np.mod(np.degrees(np.arctan2(b2, a2_prime)), 360.0)

    delta_l_prime = l2 - l1
    delta_c_prime = c2_prime - c1_prime
    hue_difference = h2_prime - h1_prime
    chroma_product = c1_prime * c2_prime
    delta_h_prime = np.where(
        chroma_product == 0.0,
        0.0,
        np.where(
            np.abs(hue_difference) <= 180.0,
            hue_difference,
            np.where(
                hue_difference > 180.0,
                hue_difference - 360.0,
                hue_difference + 360.0,
            ),
        ),
    )
    delta_big_h_prime = (
        2.0
        * np.sqrt(chroma_product)
        * np.sin(np.radians(delta_h_prime / 2.0))
    )

    l_bar_prime = (l1 + l2) / 2.0
    c_bar_prime = (c1_prime + c2_prime) / 2.0
    hue_sum = h1_prime + h2_prime
    h_bar_prime = np.where(
        chroma_product == 0.0,
        hue_sum,
        np.where(
            np.abs(hue_difference) <= 180.0,
            hue_sum / 2.0,
            np.where(
                hue_sum < 360.0,
                (hue_sum + 360.0) / 2.0,
                (hue_sum - 360.0) / 2.0,
            ),
        ),
    )

    t = (
        1.0
        - 0.17 * np.cos(np.radians(h_bar_prime - 30.0))
        + 0.24 * np.cos(np.radians(2.0 * h_bar_prime))
        + 0.32 * np.cos(np.radians(3.0 * h_bar_prime + 6.0))
        - 0.20 * np.cos(np.radians(4.0 * h_bar_prime - 63.0))
    )
    delta_theta = 30.0 * np.exp(-((h_bar_prime - 275.0) / 25.0) ** 2)
    c_bar_prime7 = c_bar_prime**7
    r_c = 2.0 * np.sqrt(
        c_bar_prime7 / (c_bar_prime7 + 25.0**7)
    )
    l_offset = l_bar_prime - 50.0
    s_l = 1.0 + (0.015 * l_offset**2) / np.sqrt(20.0 + l_offset**2)
    s_c = 1.0 + 0.045 * c_bar_prime
    s_h = 1.0 + 0.015 * c_bar_prime * t
    r_t = -np.sin(np.radians(2.0 * delta_theta)) * r_c

    l_term = delta_l_prime / s_l
    c_term = delta_c_prime / s_c
    h_term = delta_big_h_prime / s_h
    return np.sqrt(
        l_term**2 + c_term**2 + h_term**2 + r_t * c_term * h_term
    )


def calculate_ciede2000(
    prediction: np.ndarray,
    reference: np.ndarray,
) -> float:
    """Return mean pixel CIEDE2000 after sRGB-to-CIELAB D65 conversion."""
    first = np.asarray(prediction)
    second = np.asarray(reference)
    if first.shape != second.shape:
        raise ValueError("Input images must have the same dimensions.")
    if first.ndim != 3 or first.shape[2] != 3:
        raise ValueError("CIEDE2000 expects RGB images with three channels.")
    first_lab = cv2.cvtColor(
        first.astype(np.float32) / 255.0, cv2.COLOR_RGB2LAB
    ).astype(np.float64)
    second_lab = cv2.cvtColor(
        second.astype(np.float32) / 255.0, cv2.COLOR_RGB2LAB
    ).astype(np.float64)
    return float(calculate_ciede2000_lab(first_lab, second_lab).mean())


def _read_rgb(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        return np.array(image.convert("RGB"), dtype=np.uint8, copy=True)


def evaluate_dehaze_directory(
    prediction_dir: Path,
    reference_dir: Path,
    pairing: str,
    progress: Optional[ProgressCallback] = None,
    lpips_device: Optional[str] = None,
) -> DehazeMetrics:
    """Compute paired metrics, with optional AlexNet LPIPS."""
    pairs = pair_dehaze_images(
        Path(prediction_dir), Path(reference_dir), pairing
    )
    totals = {
        "psnr_db": 0.0,
        "ssim": 0.0,
        "ciede2000": 0.0,
        "lpips": 0.0,
    }
    lpips_model = (
        build_lpips_model(lpips_device)
        if lpips_device is not None
        else None
    )
    count = len(pairs)
    for index, (prediction_path, reference_path) in enumerate(pairs, start=1):
        prediction = _read_rgb(prediction_path)
        reference = _read_rgb(reference_path)
        if prediction.shape != reference.shape:
            raise ValueError(
                f"Prediction/reference dimensions differ for "
                f"{prediction_path.name}: {prediction.shape} vs "
                f"{reference.shape}"
            )
        totals["psnr_db"] += float(calculate_psnr(prediction, reference))
        totals["ssim"] += float(calculate_ssim(prediction, reference))
        totals["ciede2000"] += calculate_ciede2000(
            prediction, reference
        )
        if lpips_model is not None:
            totals["lpips"] += calculate_lpips(
                prediction,
                reference,
                lpips_model,
                lpips_device,
            )
        if progress is not None:
            progress(index, count, prediction_path)

    return DehazeMetrics(
        samples=count,
        psnr_db=totals["psnr_db"] / count,
        ssim=totals["ssim"] / count,
        ciede2000=totals["ciede2000"] / count,
        lpips=(
            totals["lpips"] / count
            if lpips_model is not None
            else None
        ),
    )


def evaluate_fade_directory(
    prediction_dir: Path,
    progress: Optional[ProgressCallback] = None,
    workers: int = 1,
) -> FadeMetrics:
    """Compute the image-wise mean no-reference FADE score."""
    if workers < 1:
        raise ValueError("workers must be at least 1")
    paths = _image_paths(Path(prediction_dir))

    def score(path: Path) -> float:
        value = float(calculate_fade(_read_rgb(path)))
        if not np.isfinite(value):
            raise ValueError(f"FADE produced a non-finite score for {path}")
        return value

    if workers == 1:
        values = map(score, paths)
        executor = None
    else:
        executor = ThreadPoolExecutor(max_workers=workers)
        values = executor.map(score, paths)

    total = 0.0
    try:
        for index, (path, value) in enumerate(
            zip(paths, values),
            start=1,
        ):
            total += value
            if progress is not None:
                progress(index, len(paths), path)
    finally:
        if executor is not None:
            executor.shutdown(wait=True, cancel_futures=True)

    return FadeMetrics(samples=len(paths), fade=total / len(paths))


__all__ = [
    "DehazeMetrics",
    "FadeMetrics",
    "calculate_ciede2000",
    "calculate_ciede2000_lab",
    "evaluate_dehaze_directory",
    "evaluate_fade_directory",
    "pair_dehaze_images",
]
