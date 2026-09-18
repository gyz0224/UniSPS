#!/usr/bin/env python3
"""Export SPS-NetPro semantic and dehazing intermediates for figure use."""

import argparse
import hashlib
import json
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Dict, Mapping, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFont, ImageOps

# Support both ``python -m scripts.export_architecture_maps`` and direct use.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from net.dehaze import DehazeConfig
from net.model import net
from training.dehaze_trainer import load_training_checkpoint
from training.runtime import dehaze_model_config, load_yaml_config, resolve_device


DEFAULT_CHECKPOINT = Path("runs/lolv1/checkpoints/stage4_joint/latest.pth")
DEFAULT_OUTPUT = Path("figs")
DEFAULT_SAMPLES = (
    (
        "lolv1_22",
        Path("dataset/LOLv1/Test/input/22.png"),
        Path("configs/joint.yaml"),
        "low-light image explicitly probed through the dehazing branch",
    ),
    (
        "hsts_syn_8180",
        Path("dataset/eval/HSTS/synthetic/hazy/8180.jpg"),
        Path("configs/dehaze_real.yaml"),
        "Stage 4 HSTS Synthetic dehazing preset",
    ),
)

SEMANTIC_PALETTE = torch.tensor(
    [
        [68, 1, 84],
        [59, 82, 139],
        [33, 145, 140],
        [94, 201, 98],
        [253, 231, 37],
        [239, 71, 111],
    ],
    dtype=torch.float32,
) / 255.0

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
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--device", default="cuda:0")
    return parser


def _load_rgb(path: Path, device: torch.device) -> torch.Tensor:
    with Image.open(path) as image:
        array = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
    return torch.from_numpy(array.copy()).permute(2, 0, 1).unsqueeze(0).to(device)


def _uint8_rgb(value: torch.Tensor) -> np.ndarray:
    value = value.detach().float().cpu().clamp(0.0, 1.0)
    if value.ndim == 4:
        value = value[0]
    return (
        value.permute(1, 2, 0).numpy().clip(0.0, 1.0) * 255.0
    ).round().astype(np.uint8)


def _save_rgb(value: torch.Tensor, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(_uint8_rgb(value)).save(path)


def _save_gray(value: torch.Tensor, path: Path) -> None:
    value = value.detach().float().cpu().squeeze().clamp(0.0, 1.0)
    array = (value.numpy() * 255.0).round().astype(np.uint8)
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(array).save(path)


def _inferno(value: torch.Tensor) -> torch.Tensor:
    """Apply a compact black-red-yellow colormap similar to the D4+ figure."""
    array = value.detach().float().cpu().squeeze().clamp(0.0, 1.0).numpy()
    colored = np.stack(
        [
            np.interp(array, INFERNO_POSITIONS, INFERNO_COLORS[:, channel])
            for channel in range(3)
        ],
        axis=0,
    )
    return torch.from_numpy(colored.astype(np.float32)).unsqueeze(0)


def _font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    path = Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf")
    if path.is_file():
        return ImageFont.truetype(str(path), size=size)
    return ImageFont.load_default()


def _centered_text(
    draw: ImageDraw.ImageDraw,
    canvas_width: int,
    y: int,
    text: str,
    font: ImageFont.FreeTypeFont | ImageFont.ImageFont,
    fill: Tuple[int, int, int] = (20, 20, 20),
) -> None:
    box = draw.textbbox((0, 0), text, font=font)
    width = box[2] - box[0]
    draw.text(((canvas_width - width) // 2, y), text, font=font, fill=fill)


def _save_beta_card(
    beta: torch.Tensor,
    image_size: Tuple[int, int],
    path: Path,
) -> None:
    width, height = image_size
    canvas = Image.new("RGB", image_size, "white")
    draw = ImageDraw.Draw(canvas)
    border = max(3, min(width, height) // 80)
    draw.rectangle(
        (border, border, width - border - 1, height - border - 1),
        outline=(25, 25, 25),
        width=border,
    )
    beta_value = float(beta.detach().float().cpu().item())
    symbol_font = _font(max(48, height // 3))
    value_font = _font(max(20, height // 13))
    _centered_text(draw, width, height // 8, "β", symbol_font)
    _centered_text(
        draw,
        width,
        int(height * 0.66),
        f"β = {beta_value:.5f}",
        value_font,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(path)


def _save_atmosphere_card(
    atmosphere: torch.Tensor,
    image_size: Tuple[int, int],
    path: Path,
) -> None:
    width, height = image_size
    canvas = Image.new("RGB", image_size, "white")
    draw = ImageDraw.Draw(canvas)
    border = max(3, min(width, height) // 80)
    draw.rectangle(
        (border, border, width - border - 1, height - border - 1),
        outline=(25, 25, 25),
        width=border,
    )
    values = atmosphere.detach().float().cpu().flatten().clamp(0.0, 1.0)
    color = tuple(int(round(float(value) * 255.0)) for value in values)
    label_font = _font(max(40, height // 5))
    value_font = _font(max(18, height // 16))
    _centered_text(draw, width, height // 18, "A", label_font)
    swatch_left, swatch_right = int(width * 0.18), int(width * 0.82)
    swatch_top, swatch_bottom = int(height * 0.34), int(height * 0.67)
    draw.rectangle(
        (swatch_left, swatch_top, swatch_right, swatch_bottom),
        fill=color,
        outline=(25, 25, 25),
        width=max(2, border // 2),
    )
    value_text = "A = (" + ", ".join(f"{float(value):.3f}" for value in values) + ")"
    _centered_text(draw, width, int(height * 0.78), value_text, value_font)
    path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(path)


def _robust_normalize(value: torch.Tensor) -> Tuple[torch.Tensor, float, float]:
    value = value.detach().float().cpu()
    flat = value.flatten()
    low = float(torch.quantile(flat, 0.01))
    high = float(torch.quantile(flat, 0.99))
    if high <= low:
        low = float(flat.min())
        high = float(flat.max())
    if high <= low:
        return torch.zeros_like(value), low, high
    return ((value - low) / (high - low)).clamp(0.0, 1.0), low, high


def _save_array(value: torch.Tensor, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, value.detach().float().cpu().numpy())


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _pca_rgb(feature: torch.Tensor) -> torch.Tensor:
    """Map one CxHxW feature tensor to PCA RGB for visible spatial detail."""
    _, channels, height, width = feature.shape
    matrix = (
        feature.detach()
        .float()
        .cpu()
        .squeeze(0)
        .permute(1, 2, 0)
        .reshape(-1, channels)
    )
    centered = matrix - matrix.mean(dim=0, keepdim=True)
    _, _, vh = torch.linalg.svd(centered, full_matrices=False)
    components = vh[:3].transpose(0, 1)

    # Fix SVD sign ambiguity so reruns keep the same channel orientation.
    for index in range(components.shape[1]):
        component = components[:, index]
        pivot = int(component.abs().argmax())
        if component[pivot] < 0:
            components[:, index] *= -1

    projected = centered @ components
    for index in range(3):
        channel = projected[:, index]
        low = torch.quantile(channel, 0.01)
        high = torch.quantile(channel, 0.99)
        projected[:, index] = (
            (channel - low) / (high - low).clamp_min(1e-8)
        ).clamp(0.0, 1.0)

    return projected.reshape(height, width, 3).permute(2, 0, 1).unsqueeze(0)


def _joint_semantic_clusters(
    features: Sequence[torch.Tensor], cluster_count: int = 6
) -> Sequence[torch.Tensor]:
    """Cluster true CLIP patch features into deterministic semantic regions."""
    matrices = [
        F.normalize(
            feature.detach()
            .float()
            .cpu()
            .squeeze(0)
            .permute(1, 2, 0)
            .reshape(-1, feature.shape[1]),
            dim=1,
        )
        for feature in features
    ]
    combined = torch.cat(matrices, dim=0)
    center_indices = [
        int(((combined - combined.mean(dim=0)) ** 2).sum(dim=1).argmax())
    ]
    centers = combined[center_indices]
    for _ in range(1, cluster_count):
        nearest_distance = 1.0 - (combined @ centers.transpose(0, 1)).amax(dim=1)
        center_indices.append(int(nearest_distance.argmax()))
        centers = combined[center_indices]

    labels = torch.full((combined.shape[0],), -1, dtype=torch.long)
    for _ in range(50):
        new_labels = (combined @ centers.transpose(0, 1)).argmax(dim=1)
        if torch.equal(new_labels, labels):
            break
        labels = new_labels
        new_centers = []
        for index in range(cluster_count):
            members = combined[labels == index]
            new_centers.append(
                centers[index]
                if members.shape[0] == 0
                else F.normalize(members.mean(dim=0), dim=0)
            )
        centers = torch.stack(new_centers)

    outputs = []
    offset = 0
    for feature, matrix in zip(features, matrices):
        height, width = feature.shape[-2:]
        count = matrix.shape[0]
        outputs.append(labels[offset : offset + count].reshape(height, width))
        offset += count
    return outputs


def _save_semantic_regions(
    labels: torch.Tensor,
    image_size: Tuple[int, int],
    path: Path,
) -> None:
    colors = SEMANTIC_PALETTE[labels].permute(2, 0, 1).unsqueeze(0)
    colors = F.interpolate(
        colors,
        size=(image_size[1], image_size[0]),
        mode="nearest",
    )
    _save_rgb(colors, path)


def _save_panel(sample_dir: Path) -> None:
    items = (
        ("input.png", "Input"),
        ("semantic_aware_map.png", "Semantic-aware map"),
        ("deep_semantic_feature.png", "Deep semantic feature"),
        ("atmosphere_A.png", "Atmosphere A"),
        ("transmission_t.png", "Transmission t"),
        ("scattering_beta.png", "Scattering β"),
        ("depth_from_haze_d.png", "Depth-from-haze d"),
        ("dehazed_J.png", "Dehazed J"),
    )
    cell_width, cell_height, title_height = 320, 220, 42
    panel = Image.new(
        "RGB",
        (4 * cell_width, 2 * (cell_height + title_height)),
        "white",
    )
    draw = ImageDraw.Draw(panel)
    font = _font(22)
    for index, (filename, title) in enumerate(items):
        row, column = divmod(index, 4)
        left = column * cell_width
        top = row * (cell_height + title_height)
        with Image.open(sample_dir / filename) as source:
            thumbnail = ImageOps.contain(
                source.convert("RGB"),
                (cell_width - 12, cell_height - 12),
                method=Image.Resampling.LANCZOS,
            )
        image_left = left + (cell_width - thumbnail.width) // 2
        image_top = top + title_height + (cell_height - thumbnail.height) // 2
        panel.paste(thumbnail, (image_left, image_top))
        text_box = draw.textbbox((0, 0), title, font=font)
        text_width = text_box[2] - text_box[0]
        draw.text(
            (left + (cell_width - text_width) // 2, top + 8),
            title,
            font=font,
            fill=(15, 15, 15),
        )
    panel.save(sample_dir / "architecture_maps_panel.png")


def _stats(value: torch.Tensor) -> Mapping[str, object]:
    detached = value.detach().float().cpu()
    return {
        "shape": list(detached.shape),
        "min": float(detached.min()),
        "max": float(detached.max()),
        "mean": float(detached.mean()),
    }


def main() -> int:
    args = build_parser().parse_args()
    checkpoint_path = args.checkpoint.expanduser().resolve()
    output_root = args.output.expanduser().resolve()
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint does not exist: {checkpoint_path}")

    device = resolve_device(args.device)
    sample_specs = []
    for name, image_path, config_path, routing_note in DEFAULT_SAMPLES:
        resolved_image = image_path.resolve()
        resolved_config = config_path.resolve()
        if not resolved_image.is_file():
            raise FileNotFoundError(f"Input image does not exist: {resolved_image}")
        values = load_yaml_config(str(resolved_config))
        physics = DehazeConfig.from_value(dehaze_model_config(values))
        sample_specs.append(
            (name, resolved_image, resolved_config, routing_note, physics)
        )

    # Topology is shared across the two physics profiles; ranges and atmosphere
    # mode are switched before each forward pass.
    model = net(dehaze_config=asdict(sample_specs[0][4])).to(device)
    checkpoint = load_training_checkpoint(
        str(checkpoint_path), model=model, map_location=device
    )
    model.eval()

    records: Dict[str, Dict[str, object]] = {}
    deep_features = []
    input_sizes = []
    with torch.inference_mode():
        for name, image_path, config_path, routing_note, physics in sample_specs:
            model.set_dehaze_physics(
                physics.beta_min,
                physics.beta_max,
                physics.transmission_min,
                physics.transmission_max,
                atmosphere_mode=physics.atmosphere_mode,
            )
            image = _load_rgb(image_path, device)
            semantic_features = model.extract_semantics(image)
            auxiliary = model(
                image,
                sem_feats=semantic_features,
                task="dehaze",
                return_aux=True,
            )
            sample_dir = output_root / name
            height, width = image.shape[-2:]

            semantic_map = semantic_features[0]
            deep = semantic_features[1]
            atmosphere = auxiliary["atmosphere"]
            transmission = auxiliary["transmission"]
            beta = auxiliary["beta"]
            depth = auxiliary["depth_from_haze"]

            _save_rgb(image, sample_dir / "input.png")
            _save_rgb(
                semantic_map,
                sample_dir / "semantic_aware_map_raw_rgb.png",
            )
            _save_rgb(auxiliary["clean"], sample_dir / "dehazed_J.png")

            atmosphere_map = atmosphere.expand(-1, -1, height, width)
            _save_rgb(atmosphere_map, sample_dir / "atmosphere_A_field.png")
            _save_atmosphere_card(
                atmosphere,
                (width, height),
                sample_dir / "atmosphere_A.png",
            )
            _save_gray(transmission, sample_dir / "transmission_t_gray.png")
            _save_rgb(_inferno(transmission), sample_dir / "transmission_t.png")

            beta_display = (
                (beta - physics.beta_min)
                / max(physics.beta_max - physics.beta_min, 1e-8)
            ).expand(-1, -1, height, width)
            _save_rgb(
                _inferno(beta_display),
                sample_dir / "scattering_beta_field.png",
            )
            _save_beta_card(
                beta,
                (width, height),
                sample_dir / "scattering_beta.png",
            )

            depth_display, depth_low, depth_high = _robust_normalize(depth)
            _save_gray(
                depth_display,
                sample_dir / "depth_from_haze_d_gray.png",
            )
            _save_rgb(
                _inferno(depth_display),
                sample_dir / "depth_from_haze_d.png",
            )

            _save_array(semantic_map, sample_dir / "semantic_aware_map.npy")
            _save_array(deep, sample_dir / "deep_semantic_feature.npy")
            _save_array(atmosphere, sample_dir / "atmosphere_A.npy")
            _save_array(transmission, sample_dir / "transmission_t.npy")
            _save_array(beta, sample_dir / "scattering_beta.npy")
            _save_array(depth, sample_dir / "depth_from_haze_d.npy")

            deep_features.append(deep)
            input_sizes.append((height, width))
            records[name] = {
                "input": str(image_path),
                "physics_config": str(config_path),
                "routing_note": routing_note,
                "physics": asdict(physics),
                "semantic_aware_map_note": (
                    "semantic_aware_map.png is a deterministic six-region clustering "
                    "visualization of the true CLIP block-9 patch features. The "
                    "implemented sem_feats[0] remains input RGB and is preserved as "
                    "semantic_aware_map_raw_rgb.png and semantic_aware_map.npy."
                ),
                "deep_semantic_visualization": (
                    "Per-image PCA of the 768x14x14 CLIP block-9 tensor; PCA RGB "
                    "is bilinearly resized to the source image size."
                ),
                "depth_visualization_range_p01_p99": [depth_low, depth_high],
                "tensors": {
                    "semantic_aware_map": _stats(semantic_map),
                    "deep_semantic_feature": _stats(deep),
                    "atmosphere_A": _stats(atmosphere),
                    "transmission_t": _stats(transmission),
                    "scattering_beta": _stats(beta),
                    "depth_from_haze_d": _stats(depth),
                    "dehazed_J": _stats(auxiliary["clean"]),
                },
            }

    semantic_regions = _joint_semantic_clusters(deep_features)
    for (name, *_), deep, labels, size in zip(
        sample_specs,
        deep_features,
        semantic_regions,
        input_sizes,
    ):
        pca_map = _pca_rgb(deep)
        pca_map = F.interpolate(
            pca_map, size=size, mode="bilinear", align_corners=False
        )
        _save_rgb(pca_map, output_root / name / "deep_semantic_feature.png")
        sample_dir = output_root / name
        _save_semantic_regions(
            labels,
            (size[1], size[0]),
            sample_dir / "semantic_aware_map.png",
        )
        np.save(
            sample_dir / "semantic_aware_labels.npy",
            labels.numpy().astype(np.uint8),
        )
        records[name]["semantic_region_cluster_count"] = int(
            torch.unique(labels).numel()
        )
        _save_panel(sample_dir)

    manifest = {
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": _sha256(checkpoint_path),
        "checkpoint_stage": checkpoint.get("stage") if isinstance(checkpoint, dict) else None,
        "checkpoint_iteration": checkpoint.get("iteration") if isinstance(checkpoint, dict) else None,
        "device": str(device),
        "samples": records,
    }
    output_root.mkdir(parents=True, exist_ok=True)
    with (output_root / "architecture_maps_manifest.json").open(
        "w", encoding="utf-8"
    ) as handle:
        json.dump(manifest, handle, ensure_ascii=False, indent=2)

    for name in records:
        print(f"[export] {name}: {output_root / name}", flush=True)
    print(f"[export] manifest: {output_root / 'architecture_maps_manifest.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
