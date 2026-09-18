#!/usr/bin/env python3
"""Export detail-enhanced low-light illumination maps for figure use."""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, Tuple

import cv2
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont, ImageOps

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from net.model import net
from training.lowlight_trainer import load_lowlight_checkpoint
from training.runtime import resolve_device


DEFAULT_CHECKPOINT = Path("runs/lolv1/checkpoints/stage4_joint/latest.pth")
DEFAULT_INPUT = Path("dataset/LOLv1/Test/input/22.png")
DEFAULT_OUTPUT = Path("figs/lolv1_22/lowlight_clear")

INFERNO_POSITIONS = np.array([0.0, 0.18, 0.38, 0.58, 0.78, 1.0])
INFERNO_COLORS = np.array(
    [
        [0, 0, 4],
        [50, 10, 94],
        [126, 30, 110],
        [203, 62, 79],
        [247, 142, 14],
        [252, 255, 164],
    ],
    dtype=np.float32,
) / 255.0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--device", default="cuda:0")
    return parser


def _load_rgb(path: Path, device: torch.device) -> torch.Tensor:
    with Image.open(path) as image:
        array = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
    return torch.from_numpy(array.copy()).permute(2, 0, 1).unsqueeze(0).to(device)


def _robust_channel(value: np.ndarray) -> Tuple[np.ndarray, float, float]:
    low, high = np.percentile(value, [1.0, 99.0])
    if high <= low:
        low, high = float(value.min()), float(value.max())
    if high <= low:
        return np.zeros_like(value, dtype=np.float32), float(low), float(high)
    normalized = np.clip((value - low) / (high - low), 0.0, 1.0)
    return normalized.astype(np.float32), float(low), float(high)


def _robust_channels(value: np.ndarray) -> Tuple[np.ndarray, list]:
    channels = []
    ranges = []
    for index in range(value.shape[0]):
        normalized, low, high = _robust_channel(value[index])
        channels.append(normalized)
        ranges.append([low, high])
    return np.stack(channels), ranges


def _clahe_gray(value: np.ndarray, clip_limit: float = 2.5) -> np.ndarray:
    gray = (np.clip(value, 0.0, 1.0) * 255.0).round().astype(np.uint8)
    enhanced = cv2.createCLAHE(
        clipLimit=clip_limit,
        tileGridSize=(8, 8),
    ).apply(gray)
    return enhanced.astype(np.float32) / 255.0


def _clahe_rgb(value: np.ndarray) -> np.ndarray:
    rgb = (np.clip(value, 0.0, 1.0) * 255.0).round().astype(np.uint8)
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)
    lab[:, :, 0] = cv2.createCLAHE(
        clipLimit=2.0,
        tileGridSize=(8, 8),
    ).apply(lab[:, :, 0])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2RGB).astype(np.float32) / 255.0


def _inferno(value: np.ndarray) -> np.ndarray:
    return np.stack(
        [
            np.interp(value, INFERNO_POSITIONS, INFERNO_COLORS[:, channel])
            for channel in range(3)
        ],
        axis=-1,
    )


def _pca_rgb(feature: np.ndarray) -> Tuple[np.ndarray, list]:
    channels, height, width = feature.shape
    matrix = feature.transpose(1, 2, 0).reshape(-1, channels).astype(np.float64)
    centered = matrix - matrix.mean(axis=0, keepdims=True)
    covariance = centered.T @ centered / max(1, centered.shape[0] - 1)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    order = np.argsort(eigenvalues)[::-1][:3]
    components = eigenvectors[:, order]
    for index in range(components.shape[1]):
        pivot = np.argmax(np.abs(components[:, index]))
        if components[pivot, index] < 0:
            components[:, index] *= -1
    projected = (centered @ components).reshape(height, width, 3)
    normalized = []
    ranges = []
    for index in range(3):
        channel, low, high = _robust_channel(projected[:, :, index])
        normalized.append(channel)
        ranges.append([low, high])
    rgb = np.stack(normalized, axis=-1)
    explained = eigenvalues[order] / max(eigenvalues.clip(min=0).sum(), 1e-12)
    return _clahe_rgb(rgb), explained.tolist()


def _font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    path = Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf")
    return ImageFont.truetype(str(path), size) if path.is_file() else ImageFont.load_default()


def _save_rgb(value: np.ndarray, path: Path) -> None:
    array = (np.clip(value, 0.0, 1.0) * 255.0).round().astype(np.uint8)
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(array).save(path)


def _save_gray(value: np.ndarray, path: Path) -> None:
    array = (np.clip(value, 0.0, 1.0) * 255.0).round().astype(np.uint8)
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(array).save(path)


def _save_feature_grid(feature: np.ndarray, path: Path) -> list:
    variance = feature.reshape(feature.shape[0], -1).var(axis=1)
    indices = np.argsort(variance)[::-1][:12]
    tile_width, tile_height, label_height = 240, 160, 28
    canvas = Image.new("RGB", (4 * tile_width, 3 * (tile_height + label_height)), "white")
    draw = ImageDraw.Draw(canvas)
    font = _font(17)
    for position, channel_index in enumerate(indices):
        normalized, _, _ = _robust_channel(feature[channel_index])
        normalized = _clahe_gray(normalized)
        tile = Image.fromarray(
            (_inferno(normalized) * 255.0).round().astype(np.uint8)
        ).resize((tile_width, tile_height), Image.Resampling.BILINEAR)
        row, column = divmod(position, 4)
        left = column * tile_width
        top = row * (tile_height + label_height)
        canvas.paste(tile, (left, top + label_height))
        label = f"channel {int(channel_index)}"
        box = draw.textbbox((0, 0), label, font=font)
        draw.text(
            (left + (tile_width - (box[2] - box[0])) // 2, top + 3),
            label,
            font=font,
            fill=(20, 20, 20),
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(path)
    return [int(index) for index in indices]


def _save_panel(output: Path) -> None:
    items = (
        ("L.png", "Illumination L"),
        ("L_inferno.png", "L (heatmap)"),
        ("illu_map.png", "Illumination map"),
        ("illu_fea.png", "Illumination feature (PCA)"),
    )
    cell_width, cell_height, title_height = 360, 240, 40
    panel = Image.new("RGB", (4 * cell_width, cell_height + title_height), "white")
    draw = ImageDraw.Draw(panel)
    font = _font(21)
    for column, (filename, title) in enumerate(items):
        with Image.open(output / filename) as source:
            thumbnail = ImageOps.contain(
                source.convert("RGB"),
                (cell_width - 12, cell_height - 12),
                method=Image.Resampling.LANCZOS,
            )
        left = column * cell_width
        panel.paste(
            thumbnail,
            (
                left + (cell_width - thumbnail.width) // 2,
                title_height + (cell_height - thumbnail.height) // 2,
            ),
        )
        box = draw.textbbox((0, 0), title, font=font)
        draw.text(
            (left + (cell_width - (box[2] - box[0])) // 2, 7),
            title,
            font=font,
            fill=(15, 15, 15),
        )
    panel.save(output / "lowlight_maps_clear_panel.png")


def _stats(value: np.ndarray) -> Dict[str, object]:
    return {
        "shape": list(value.shape),
        "min": float(value.min()),
        "max": float(value.max()),
        "mean": float(value.mean()),
        "std": float(value.std()),
    }


def main() -> int:
    args = build_parser().parse_args()
    checkpoint = args.checkpoint.expanduser().resolve()
    input_path = args.input.expanduser().resolve()
    output = args.output.expanduser().resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Checkpoint does not exist: {checkpoint}")
    if not input_path.is_file():
        raise FileNotFoundError(f"Input does not exist: {input_path}")

    device = resolve_device(args.device)
    model = net().to(device)
    incompatible = load_lowlight_checkpoint(checkpoint, model, map_location=device)
    unexpected = list(incompatible.unexpected_keys)
    if unexpected:
        raise ValueError(f"Unexpected checkpoint keys: {unexpected[:5]}")
    model.eval()

    image = _load_rgb(input_path, device)
    with torch.inference_mode():
        x_img, _ = model.N_net(image)
        illu_fea, illu_map = model.illp(x_img)
        illumination = model.L_net(x_img)

    tensors = {
        "L": illumination.detach().float().cpu().numpy()[0],
        "illu_map": illu_map.detach().float().cpu().numpy()[0],
        "illu_fea": illu_fea.detach().float().cpu().numpy()[0],
    }
    output.mkdir(parents=True, exist_ok=True)
    for name, value in tensors.items():
        np.save(output / f"{name}.npy", value)

    l_normalized, l_low, l_high = _robust_channel(tensors["L"][0])
    l_clear = _clahe_gray(l_normalized)
    _save_gray(np.clip(tensors["L"][0], 0.0, 1.0), output / "L_raw.png")
    _save_gray(l_normalized, output / "L_percentile.png")
    _save_gray(l_clear, output / "L.png")
    _save_rgb(_inferno(l_clear), output / "L_inferno.png")

    illu_map_normalized, illu_map_ranges = _robust_channels(tensors["illu_map"])
    illu_map_rgb = _clahe_rgb(illu_map_normalized.transpose(1, 2, 0))
    _save_rgb(illu_map_rgb, output / "illu_map.png")
    _save_rgb(
        np.clip(tensors["illu_map"].transpose(1, 2, 0), 0.0, 1.0),
        output / "illu_map_raw_clamped.png",
    )

    illu_fea_rgb, explained_variance = _pca_rgb(tensors["illu_fea"])
    _save_rgb(illu_fea_rgb, output / "illu_fea.png")
    selected_channels = _save_feature_grid(
        tensors["illu_fea"],
        output / "illu_fea_top_variance_grid.png",
    )
    _save_panel(output)

    manifest = {
        "checkpoint": str(checkpoint),
        "input": str(input_path),
        "device": str(device),
        "visualization_only": True,
        "L_visualization": {
            "percentile_range": [l_low, l_high],
            "final": "1%-99% stretch followed by CLAHE",
        },
        "illu_map_visualization": {
            "per_channel_percentile_ranges": illu_map_ranges,
            "final": "per-channel 1%-99% stretch followed by luminance CLAHE",
        },
        "illu_fea_visualization": {
            "final": "64-to-3 PCA followed by luminance CLAHE",
            "pca_explained_variance_ratio": explained_variance,
            "top_variance_grid_channels": selected_channels,
        },
        "raw_tensors": {name: _stats(value) for name, value in tensors.items()},
    }
    with (output / "manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, ensure_ascii=False, indent=2)

    print(f"[export] clear low-light maps: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
