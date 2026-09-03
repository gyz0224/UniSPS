#!/usr/bin/env python3
"""Run SPS dehazing inference without constructing training-only modules."""

import argparse
import contextlib
import gc
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from net.model import net
from training.dehaze_trainer import load_training_checkpoint
from training.experiment import (
    DEHAZE_PRESETS,
    EXPERIMENT_CHOICES,
    ExperimentLayout,
)
from training.runtime import dehaze_model_config, load_yaml_config, resolve_device


EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
DEFAULT_AMP_PIXEL_THRESHOLD = 3_000_000


@dataclass(frozen=True)
class InferenceSpec:
    experiment: str
    stage: str
    config: Path
    checkpoint: Path
    input_path: Path
    output: Path
    tile_size: int
    tile_overlap: int


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--experiment",
        required=True,
        choices=EXPERIMENT_CHOICES,
        help="isolated run identity",
    )
    parser.add_argument(
        "--stage",
        required=True,
        choices=tuple(DEHAZE_PRESETS),
        help="checkpoint/dataset preset",
    )
    parser.add_argument("--config", help="override preset model/physics YAML")
    parser.add_argument("--checkpoint", help="override preset checkpoint")
    parser.add_argument("--input", help="override preset image or directory")
    parser.add_argument("--output", help="override canonical result directory")
    parser.add_argument("--device", help="torch device")
    parser.add_argument(
        "--amp-pixel-threshold",
        type=int,
        default=DEFAULT_AMP_PIXEL_THRESHOLD,
        help=(
            "use CUDA AMP directly at or above this pixel count; smaller images "
            "stay FP32 unless they raise CUDA OOM"
        ),
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="recompute valid existing outputs instead of resuming",
    )
    parser.add_argument(
        "--tile-size",
        type=int,
        default=None,
        help=(
            "override preset overlapping tile size; 0 keeps full-frame inference"
        ),
    )
    parser.add_argument(
        "--tile-overlap",
        type=int,
        default=128,
        help="tile overlap in pixels; default: 128",
    )
    return parser


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    return build_parser().parse_args(argv)


def resolve_inference(args: argparse.Namespace) -> InferenceSpec:
    layout = ExperimentLayout(args.experiment)
    preset = DEHAZE_PRESETS[args.stage]
    return InferenceSpec(
        experiment=args.experiment,
        stage=args.stage,
        config=Path(args.config) if args.config else preset.config,
        checkpoint=(
            Path(args.checkpoint)
            if args.checkpoint
            else layout.checkpoint(preset.train_stage)
        ),
        input_path=Path(args.input) if args.input else preset.input_dir,
        output=Path(args.output) if args.output else layout.dehaze_result(args.stage),
        tile_size=preset.tile_size if args.tile_size is None else args.tile_size,
        tile_overlap=args.tile_overlap,
    )


def _images(source: Path) -> list[Path]:
    if source.is_file() and source.suffix.lower() in EXTENSIONS:
        return [source]
    if source.is_dir():
        images = sorted(
            path for path in source.rglob("*") if path.is_file() and path.suffix.lower() in EXTENSIONS
        )
        if images:
            return images
    raise FileNotFoundError(f"No supported input images found at {source}")


def _image_size(path: Path) -> tuple[int, int]:
    with Image.open(path) as image:
        return image.size


def _tensor(path: Path, device: torch.device) -> torch.Tensor:
    with Image.open(path) as image:
        array = np.array(image.convert("RGB"), dtype=np.float32, copy=True) / 255.0
    return torch.from_numpy(array).permute(2, 0, 1).unsqueeze(0).to(device)


def _save(image: torch.Tensor, path: Path) -> None:
    array = (
        image.detach().squeeze(0).clamp(0, 1).permute(1, 2, 0).cpu().numpy() * 255.0
    ).round().astype(np.uint8)
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(array).save(path)


def _target_path(source: Path, output: Path, image_path: Path) -> Path:
    relative = (
        Path(image_path.name)
        if source.is_file()
        else image_path.relative_to(source)
    )
    return (output / relative).with_suffix(".png")


def _valid_output(image_path: Path, target: Path) -> bool:
    if not target.is_file():
        return False
    try:
        with Image.open(image_path) as source_image:
            source_size = source_image.size
        with Image.open(target) as output_image:
            output_size = output_image.size
            output_image.verify()
    except (OSError, ValueError):
        return False
    return output_size == source_size


def _pending_jobs(
    images: Sequence[Path],
    source: Path,
    output: Path,
    overwrite: bool,
) -> tuple[list[tuple[Path, Path]], int]:
    jobs: list[tuple[Path, Path]] = []
    skipped = 0
    for image_path in images:
        target = _target_path(source, output, image_path)
        if not overwrite and _valid_output(image_path, target):
            skipped += 1
        else:
            jobs.append((image_path, target))
    return jobs, skipped


def _forward(
    model: torch.nn.Module,
    image: torch.Tensor,
    device: torch.device,
    use_amp: bool,
    atmosphere: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    amp_context = (
        torch.autocast(device_type="cuda", dtype=torch.float16)
        if use_amp
        else contextlib.nullcontext()
    )
    with torch.inference_mode(), amp_context:
        if atmosphere is None:
            return model(image)
        return model(image, atmosphere=atmosphere)


def _tile_starts(length: int, tile_size: int, overlap: int) -> list[int]:
    if length <= tile_size:
        return [0]
    stride = tile_size - overlap
    starts = list(range(0, length - tile_size + 1, stride))
    final_start = length - tile_size
    if starts[-1] != final_start:
        starts.append(final_start)
    return starts


def _tile_weight(
    height: int,
    width: int,
    overlap: int,
    *,
    top: bool,
    bottom: bool,
    left: bool,
    right: bool,
) -> torch.Tensor:
    y_weight = torch.ones(height, dtype=torch.float32)
    x_weight = torch.ones(width, dtype=torch.float32)

    y_overlap = min(overlap, height)
    if y_overlap:
        y_ramp = torch.linspace(0.0, 1.0, y_overlap + 2)[1:-1]
        if top:
            y_weight[:y_overlap] = y_ramp
        if bottom:
            y_weight[-y_overlap:] = y_ramp.flip(0)

    x_overlap = min(overlap, width)
    if x_overlap:
        x_ramp = torch.linspace(0.0, 1.0, x_overlap + 2)[1:-1]
        if left:
            x_weight[:x_overlap] = x_ramp
        if right:
            x_weight[-x_overlap:] = x_ramp.flip(0)
    return (y_weight[:, None] * x_weight[None, :]).unsqueeze(0).unsqueeze(0)


def _tiled_inference(
    model: torch.nn.Module,
    image_path: Path,
    device: torch.device,
    tile_size: int,
    tile_overlap: int,
) -> tuple[torch.Tensor, str]:
    image = _tensor(image_path, torch.device("cpu"))
    height, width = image.shape[-2:]
    y_starts = _tile_starts(height, tile_size, tile_overlap)
    x_starts = _tile_starts(width, tile_size, tile_overlap)

    # Estimate one atmosphere value for the whole image so overlapping tiles
    # do not receive different physical illumination. Downsampling only this
    # global statistic keeps very large source images off the GPU.
    atmosphere_input = image
    longest_side = max(height, width)
    if longest_side > tile_size:
        scale = tile_size / float(longest_side)
        atmosphere_input = F.interpolate(
            image,
            size=(
                max(1, round(height * scale)),
                max(1, round(width * scale)),
            ),
            mode="bilinear",
            align_corners=False,
        )
    with torch.inference_mode():
        atmosphere = model.atmosphere_estimator(
            atmosphere_input.to(device)
        ).detach()

    output_sum = torch.zeros_like(image, dtype=torch.float32)
    weight_sum = torch.zeros(
        (1, 1, height, width),
        dtype=torch.float32,
    )
    use_amp = device.type == "cuda"
    total_tiles = len(y_starts) * len(x_starts)
    tile_index = 0
    for y in y_starts:
        for x in x_starts:
            tile_index += 1
            tile = image[
                ...,
                y : min(y + tile_size, height),
                x : min(x + tile_size, width),
            ].to(device)
            try:
                prediction = _forward(
                    model,
                    tile,
                    device,
                    use_amp=use_amp,
                    atmosphere=atmosphere,
                )
            except torch.OutOfMemoryError as exc:
                _clear_cuda_memory(model)
                raise torch.OutOfMemoryError(
                    f"CUDA OOM for tile {tile_size}x{tile_size} in "
                    f"{image_path}; retry with a smaller --tile-size"
                ) from exc
            prediction = prediction.detach().float().cpu()
            tile_height, tile_width = prediction.shape[-2:]
            weight = _tile_weight(
                tile_height,
                tile_width,
                tile_overlap,
                top=y > 0,
                bottom=y + tile_height < height,
                left=x > 0,
                right=x + tile_width < width,
            )
            output_sum[
                ...,
                y : y + tile_height,
                x : x + tile_width,
            ] += prediction * weight
            weight_sum[
                ...,
                y : y + tile_height,
                x : x + tile_width,
            ] += weight
            if (
                tile_index == 1
                or tile_index % 10 == 0
                or tile_index == total_tiles
            ):
                print(
                    f"[eval] tiles={tile_index}/{total_tiles} "
                    f"image={image_path.name}",
                    flush=True,
                )
            del tile, prediction

    result = output_sum / weight_sum.clamp_min(1e-8)
    mode = "amp-tiled" if use_amp else "fp32-tiled"
    return result, mode


def _clear_cuda_memory(model: torch.nn.Module) -> None:
    semantic_net = getattr(model, "sem_net", None)
    cached_features = getattr(semantic_net, "_hook_feats", None)
    if isinstance(cached_features, dict):
        cached_features.clear()
    gc.collect()
    torch.cuda.empty_cache()


def _adaptive_inference(
    model: torch.nn.Module,
    image_path: Path,
    device: torch.device,
    image_pixels: int,
    amp_pixel_threshold: int,
    tile_size: int = 0,
    tile_overlap: int = 128,
) -> tuple[torch.Tensor, str]:
    if tile_size:
        return _tiled_inference(
            model,
            image_path,
            device,
            tile_size,
            tile_overlap,
        )
    image = _tensor(image_path, device)
    use_amp = (
        device.type == "cuda" and image_pixels >= amp_pixel_threshold
    )
    if use_amp:
        try:
            return _forward(model, image, device, use_amp=True), "amp-large"
        except torch.OutOfMemoryError as exc:
            _clear_cuda_memory(model)
            raise torch.OutOfMemoryError(
                f"CUDA OOM for {image_path} ({image_pixels} pixels) even with AMP"
            ) from exc

    try:
        return _forward(model, image, device, use_amp=False), "fp32"
    except torch.OutOfMemoryError as exc:
        if device.type != "cuda":
            raise
        _clear_cuda_memory(model)
        print(
            f"[eval] FP32 OOM for {image_path}; retrying this image with AMP",
            flush=True,
        )
        try:
            return _forward(model, image, device, use_amp=True), "amp-oom"
        except torch.OutOfMemoryError as amp_exc:
            _clear_cuda_memory(model)
            raise torch.OutOfMemoryError(
                f"CUDA OOM for {image_path} ({image_pixels} pixels) even with AMP"
            ) from amp_exc


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    spec = resolve_inference(args)
    if args.amp_pixel_threshold <= 0:
        raise ValueError("--amp-pixel-threshold must be positive")
    if spec.tile_size < 0:
        raise ValueError("--tile-size must be non-negative")
    if spec.tile_overlap < 0:
        raise ValueError("--tile-overlap must be non-negative")
    if spec.tile_size and spec.tile_overlap >= spec.tile_size:
        raise ValueError("--tile-overlap must be smaller than --tile-size")
    config = load_yaml_config(str(spec.config))
    device = resolve_device(args.device or config.get("device"))
    model = net(dehaze_config=dehaze_model_config(config)).to(device)
    load_training_checkpoint(str(spec.checkpoint), model=model, map_location=device)
    model = model.to_dehaze_inference().to(device)
    model.eval()

    source = spec.input_path.expanduser()
    output = spec.output.expanduser()
    images = _images(source)
    jobs, skipped = _pending_jobs(
        images,
        source,
        output,
        overwrite=args.overwrite,
    )
    print(
        f"[eval] experiment={spec.experiment} stage={spec.stage} "
        f"checkpoint={spec.checkpoint} inputs={len(images)} skipped={skipped} "
        f"pending={len(jobs)} amp_threshold={args.amp_pixel_threshold} "
        f"tile_size={spec.tile_size} tile_overlap={spec.tile_overlap} output={output}",
        flush=True,
    )
    for index, (image_path, target) in enumerate(jobs, start=1):
        width, height = _image_size(image_path)
        clean, mode = _adaptive_inference(
            model,
            image_path,
            device,
            image_pixels=width * height,
            amp_pixel_threshold=args.amp_pixel_threshold,
            tile_size=spec.tile_size,
            tile_overlap=spec.tile_overlap,
        )
        _save(clean, target)
        print(
            f"[eval] {index}/{len(jobs)} mode={mode} size={width}x{height} "
            f"{target}",
            flush=True,
        )
        del clean
        if device.type == "cuda" and mode.startswith("amp"):
            _clear_cuda_memory(model)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
