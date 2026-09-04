"""Shared construction helpers for dehaze and joint command-line entrypoints."""

from dataclasses import fields
from pathlib import Path
from typing import Any, Dict, Iterator, Mapping, Optional

import torch
import yaml
from torch.utils.data import DataLoader

from datasets.dehaze import UnpairedDehazeDataset
from loss.dehaze_loss import DualContrastivePerceptualLoss
from net.dehaze import DehazeConfig
from net.depth import DepthEstimationNet
from net.discriminator import build_dehaze_discriminators
from net.model import net
from net.rehaze import HazeRefineNet
from training.dehaze_trainer import (
    DehazeTrainer,
    TrainerConfig,
    build_dehaze_optimizers,
    configure_stage,
    load_training_checkpoint,
)


def load_yaml_config(path: str) -> Dict[str, Any]:
    config_path = Path(path).expanduser()
    if not config_path.is_file():
        raise FileNotFoundError(f"Configuration file does not exist: {config_path}")
    with config_path.open("r", encoding="utf-8") as handle:
        values = yaml.safe_load(handle)
    if not isinstance(values, dict):
        raise ValueError(f"Configuration root must be a mapping: {config_path}")
    values["_config_path"] = str(config_path.resolve())
    return values


def resolve_device(requested: Optional[str]) -> torch.device:
    if requested:
        device = torch.device(requested)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"CUDA device requested but CUDA is unavailable: {device}")
    return device


def dehaze_model_config(values: Mapping[str, Any]) -> Dict[str, Any]:
    allowed = {field.name for field in fields(DehazeConfig)}
    return {key: value for key, value in values.items() if key in allowed}


def make_unpaired_dataset(
    values: Mapping[str, Any], return_paths: bool = True
) -> UnpairedDehazeDataset:
    for key in ("clean_flist", "hazy_flist"):
        if not values.get(key):
            raise ValueError(f"Configuration must set a non-empty {key}")
    return UnpairedDehazeDataset(
        clean_source=values["clean_flist"],
        hazy_source=values["hazy_flist"],
        crop_size=int(values.get("crop_size", 256)),
        clean_mask_dir=values.get("clean_mask_dir"),
        hazy_mask_dir=values.get("hazy_mask_dir"),
        is_real=bool(values.get("is_real", False)),
        iteration_length=values.get("iteration_length"),
        test_gt_source=values.get("test_gt_flist"),
        augment=bool(values.get("augment", True)),
        seed=values.get("seed"),
        return_paths=return_paths,
    )


def make_dehaze_loader(values: Mapping[str, Any], return_paths: bool = True) -> DataLoader:
    dataset = make_unpaired_dataset(values, return_paths=return_paths)
    batch_size = int(values.get("batch_size", 2))
    if len(dataset) < batch_size:
        raise ValueError(
            f"Dataset length {len(dataset)} is smaller than batch_size {batch_size}"
        )
    generator = None
    if values.get("seed") is not None:
        generator = torch.Generator()
        generator.manual_seed(int(values["seed"]))
    return DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=int(values.get("num_workers", 0)),
        shuffle=False,
        drop_last=True,
        pin_memory=bool(values.get("pin_memory", torch.cuda.is_available())),
        generator=generator,
    )


def infinite_batches(loader: DataLoader) -> Iterator[Mapping[str, Any]]:
    while True:
        yield from loader


def batch_to_device(batch: Mapping[str, Any], device: torch.device) -> Dict[str, Any]:
    return {
        key: value.to(device, non_blocking=True) if torch.is_tensor(value) else value
        for key, value in batch.items()
    }


def build_training_stack(
    values: Mapping[str, Any],
    device: torch.device,
    include_lowlight: bool = False,
) -> Dict[str, Any]:
    print("[setup] Initializing SPS model and semantic encoder...", flush=True)
    model = net(dehaze_config=dehaze_model_config(values)).to(device)
    print("[setup] SPS model is ready.", flush=True)
    if values.get("pretrained_sps"):
        print(
            f"[setup] Loading pretrained SPS weights from "
            f"{values['pretrained_sps']}...",
            flush=True,
        )
        load_training_checkpoint(
            values["pretrained_sps"], model=model, map_location=device
        )

    stage = int(values.get("stage", 1))
    configure_stage(model, stage)
    print(f"[setup] Stage {stage} parameter freezing is configured.", flush=True)
    trainer_config = TrainerConfig.from_mapping(values)
    print("[setup] Initializing depth, refinement, and discriminator networks...", flush=True)
    depth_net = DepthEstimationNet(
        depth_min=trainer_config.depth_min,
        depth_max=trainer_config.depth_max,
        channels=tuple(values.get("depth_channels", (32, 64, 96))),
    ).to(device)
    refine_net = HazeRefineNet(
        channels=int(values.get("refine_channels", 32)),
        residual_scale=float(values.get("residual_scale", 0.1)),
    ).to(device)
    d_clear, d_hazy = build_dehaze_discriminators(
        base_channels=int(values.get("discriminator_channels", 64)),
        layers=int(values.get("discriminator_layers", 3)),
    )
    d_clear, d_hazy = d_clear.to(device), d_hazy.to(device)
    print("[setup] Auxiliary dehazing networks are ready.", flush=True)
    if values.get("initial_checkpoint"):
        print(
            f"[setup] Loading prior-stage checkpoint from "
            f"{values['initial_checkpoint']}...",
            flush=True,
        )
        load_training_checkpoint(
            values["initial_checkpoint"],
            model=model,
            depth_net=depth_net,
            refine_net=refine_net,
            d_clear=d_clear,
            d_hazy=d_hazy,
            map_location=device,
        )
    print("[setup] Building optimizers...", flush=True)
    optimizers = build_dehaze_optimizers(
        model,
        depth_net,
        refine_net,
        d_clear,
        d_hazy,
        trainer_config,
        include_lowlight=include_lowlight,
    )

    contrast_loss = None
    if trainer_config.contrast > 0 and bool(values.get("enable_contrast", True)):
        print("[setup] Initializing VGG19 contrastive perceptual loss...", flush=True)
        contrast_loss = DualContrastivePerceptualLoss().to(device)
        print("[setup] Contrastive perceptual loss is ready.", flush=True)
    semantic_encoder = (
        model.extract_global_semantics
        if trainer_config.semantic > 0 and bool(values.get("enable_semantic", True))
        else None
    )
    trainer = DehazeTrainer(
        model,
        depth_net,
        refine_net,
        d_clear,
        d_hazy,
        optimizers,
        trainer_config,
        contrast_loss=contrast_loss,
        semantic_encoder=semantic_encoder,
    )
    max_iterations = int(values.get("max_iterations", 150000))
    lowlight_steps = int(values.get("lowlight_steps_per_cycle", 1)) if include_lowlight else 0
    dehaze_steps = int(values.get("dehaze_steps_per_cycle", 1))
    schedulers = {}
    for name, optimizer in optimizers.items():
        steps_per_iteration = (
            lowlight_steps + dehaze_steps if name == "generator" else dehaze_steps
        )
        schedulers[name] = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=max(1, max_iterations * steps_per_iteration)
        )
    print("[setup] Training stack is ready.", flush=True)
    return {
        "model": model,
        "depth_net": depth_net,
        "refine_net": refine_net,
        "D_clear": d_clear,
        "D_hazy": d_hazy,
        "optimizers": optimizers,
        "schedulers": schedulers,
        "trainer": trainer,
        "trainer_config": trainer_config,
        "stage": stage,
    }


def format_metrics(metrics: Mapping[str, float]) -> str:
    preferred = (
        "generator_total",
        "cycle_clean",
        "cycle_hazy",
        "gan_G",
        "D_clear",
        "D_hazy",
        "scattering",
        "depth",
        "contrast",
        "semantic",
        "t_mean",
        "t_std",
        "t_low_ratio",
        "t_high_ratio",
        "beta_mean",
        "beta_std",
        "depth_mean",
        "depth_std",
    )
    keys = [key for key in preferred if key in metrics]
    keys.extend(sorted(key for key in metrics if key.startswith("lr_")))
    return " ".join(f"{key}={metrics[key]:.6g}" for key in keys)


def format_duration(seconds: float) -> str:
    """Format a non-negative duration for training progress logs."""
    total_seconds = max(0, int(round(seconds)))
    hours, remainder = divmod(total_seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    if hours:
        return f"{hours}h{minutes:02d}m{seconds:02d}s"
    if minutes:
        return f"{minutes}m{seconds:02d}s"
    return f"{seconds}s"
