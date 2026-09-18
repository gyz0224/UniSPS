#!/usr/bin/env python3
"""Stage 4 joint low-light retention and indoor/real dehaze fine-tuning."""

import argparse
import random
import time
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from datasets.loaders import get_training_set
from net.model import net
from training.dehaze_trainer import (
    TrainerConfig,
    load_training_checkpoint,
    save_training_checkpoint,
    set_requires_grad,
)
from training.experiment import (
    EXPERIMENT_CHOICES,
    ExperimentLayout,
    ensure_new_training_output,
)
from training.runtime import (
    batch_to_device,
    build_training_stack,
    dehaze_model_config,
    format_duration,
    format_metrics,
    infinite_batches,
    load_yaml_config,
    make_dehaze_loader,
    resolve_device,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--experiment",
        required=True,
        choices=EXPERIMENT_CHOICES,
        help="isolated run identity and low-light retention teacher",
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--resume")
    parser.add_argument(
        "--init-checkpoint", help="Stage 3 checkpoint with fresh Stage 4 optimizers"
    )
    parser.add_argument(
        "--teacher-checkpoint",
        help="override the experiment's frozen low-light retention teacher",
    )
    parser.add_argument(
        "--lowlight-data",
        help="override the experiment's native low-light retention training data",
    )
    parser.add_argument("--device")
    parser.add_argument("--max-iterations", type=int)
    parser.add_argument("--output")
    amp_group = parser.add_mutually_exclusive_group()
    amp_group.add_argument(
        "--amp", dest="amp", action="store_true", help="enable CUDA AMP"
    )
    amp_group.add_argument(
        "--no-amp", dest="amp", action="store_false", help="disable AMP"
    )
    parser.set_defaults(amp=None)
    parser.add_argument(
        "--amp-dtype", choices=("bfloat16", "float16"), help="override AMP dtype"
    )
    return parser


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    return build_parser().parse_args(argv)


def _merged(base: Mapping[str, Any], section: str) -> Dict[str, Any]:
    section_values = base.get(section)
    if not isinstance(section_values, dict):
        raise ValueError(f"Joint configuration requires a {section!r} mapping")
    merged = dict(base)
    merged.update(section_values)
    return merged


def configure_experiment(
    args: argparse.Namespace,
    config: Dict[str, Any],
) -> ExperimentLayout:
    """Apply isolated Stage 4 initializer, teacher, and output defaults."""
    layout = ExperimentLayout(args.experiment)
    config["experiment"] = args.experiment
    config["output_dir"] = str(args.output or layout.checkpoint_dir(4))
    config["teacher_checkpoint"] = str(
        args.teacher_checkpoint or layout.initial_weight
    )
    config["lowlight_data"] = str(
        args.lowlight_data or layout.lowlight_dataset.train_input
    )
    config["pretrained_sps"] = None
    config["initial_checkpoint"] = (
        None
        if args.resume
        else str(args.init_checkpoint or layout.checkpoint(3))
    )
    ensure_new_training_output(Path(config["output_dir"]), args.resume)
    return layout


def _lowlight_loader(config: Mapping[str, Any]) -> DataLoader:
    path = config.get("lowlight_data")
    if not path:
        raise ValueError("Joint configuration must set lowlight_data")
    dataset = get_training_set(path)
    return DataLoader(
        dataset,
        batch_size=int(config.get("lowlight_batch_size", 1)),
        num_workers=int(config.get("lowlight_num_workers", 0)),
        shuffle=True,
        drop_last=True,
    )


def _retain_step(
    batch,
    model,
    teacher,
    optimizer,
    device: torch.device,
    weight: float,
    precision,
) -> float:
    lowlight = batch[0].to(device, non_blocking=True)
    optimizer.zero_grad(set_to_none=True)
    with precision.autocast():
        with torch.no_grad():
            semantics = model.extract_semantics(lowlight)
            teacher_output = teacher(lowlight, sem_feats=semantics, task="lowlight")[-1]
        student_output = model(lowlight, sem_feats=semantics, task="lowlight")[-1]
        retain = F.l1_loss(student_output, teacher_output)
        weighted = weight * retain
    if not torch.isfinite(weighted):
        names = batch[1] if len(batch) > 1 else "unavailable"
        raise FloatingPointError(
            f"Non-finite low-light retain loss for inputs {names}: {retain.item()}"
        )
    precision.backward_step(weighted, optimizer)
    precision.update()
    return retain.detach().item()


def _format_progress(
    *,
    joint_iteration: int,
    maximum: int,
    starting_iteration: int,
    epoch: int,
    step_seconds: float,
    elapsed_seconds: float,
    source_name: str,
    retain: float,
    metrics: Mapping[str, float],
) -> str:
    completed = joint_iteration - starting_iteration
    average_seconds = elapsed_seconds / max(1, completed)
    eta_seconds = average_seconds * max(0, maximum - joint_iteration)
    progress = 100.0 * joint_iteration / maximum
    return (
        f"[train] iter={joint_iteration}/{maximum} ({progress:.2f}%) "
        f"epoch={epoch} cycle={step_seconds:.1f}s "
        f"avg={average_seconds:.1f}s/iter "
        f"eta={format_duration(eta_seconds)} "
        f"source={source_name} retain={retain:.6g} "
        f"{format_metrics(metrics)}"
    ).rstrip()


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    config = load_yaml_config(args.config)
    if args.max_iterations is not None:
        if args.max_iterations <= 0:
            raise ValueError("--max-iterations must be positive")
        config["max_iterations"] = args.max_iterations
    if args.amp is not None:
        config["amp"] = args.amp
    if args.amp_dtype is not None:
        config["amp_dtype"] = args.amp_dtype
    layout = configure_experiment(args, config)

    device = resolve_device(args.device or config.get("device"))
    print(
        f"[setup] experiment={layout.experiment} config={config['_config_path']} "
        f"stage=4 device={device} lowlight_data={config['lowlight_data']}",
        flush=True,
    )
    stack = build_training_stack(config, device=device, include_lowlight=True)
    trainer = stack["trainer"]
    if stack["stage"] != 4:
        raise ValueError("train_joint.py requires stage: 4")

    teacher_checkpoint = config.get("teacher_checkpoint")
    if not teacher_checkpoint:
        raise ValueError("Joint configuration must set teacher_checkpoint")
    # Sharing the already loaded frozen semantic prior avoids a third CLIP copy.
    teacher = net(
        dehaze_config=dehaze_model_config(config), semantic_net=stack["model"].sem_net
    ).to(device)
    load_training_checkpoint(teacher_checkpoint, model=teacher, map_location=device)
    set_requires_grad(teacher, False)
    teacher.eval()

    indoor_config = _merged(config, "indoor")
    real_config = _merged(config, "real")
    indoor_loader = make_dehaze_loader(indoor_config, return_paths=True)
    real_loader = make_dehaze_loader(real_config, return_paths=True)
    lowlight_loader = _lowlight_loader(config)
    indoor_batches = infinite_batches(indoor_loader)
    real_batches = infinite_batches(real_loader)
    lowlight_batches = infinite_batches(lowlight_loader)

    joint_iteration = 0
    epoch = 0
    if args.resume:
        checkpoint = load_training_checkpoint(
            args.resume,
            model=stack["model"],
            depth_net=stack["depth_net"],
            refine_net=stack["refine_net"],
            d_clear=stack["D_clear"],
            d_hazy=stack["D_hazy"],
            optimizers=stack["optimizers"],
            schedulers=stack["schedulers"],
            map_location=device,
            precision=trainer.precision,
        )
        joint_iteration = int(checkpoint.get("iteration", 0))
        trainer.iteration = joint_iteration
        epoch = int(checkpoint.get("epoch", 0))

    lowlight_steps = int(config.get("lowlight_steps_per_cycle", 1))
    dehaze_steps = int(config.get("dehaze_steps_per_cycle", 1))
    if lowlight_steps <= 0 or dehaze_steps <= 0:
        raise ValueError("Joint task step counts must be positive")
    real_probability = float(config.get("real_dehaze_probability", 0.7))
    if not 0.0 <= real_probability <= 1.0:
        raise ValueError("real_dehaze_probability must be in [0,1]")
    retain_weight = float(config.get("lambda_retain", 0.1))
    rng = random.Random(int(config.get("seed", 10)))
    maximum = int(config.get("max_iterations", 100000))
    log_interval = int(config.get("log_interval", 100))
    save_interval = int(config.get("save_interval", 2000))
    output_dir = Path(config["output_dir"])
    starting_iteration = joint_iteration
    started_at = time.monotonic()
    print(
        f"[train] Started: iteration={starting_iteration}/{maximum} "
        f"log_interval={log_interval} save_interval={save_interval} "
        f"output={output_dir}",
        flush=True,
    )

    while joint_iteration < maximum:
        step_started_at = time.monotonic()
        retain = 0.0
        for _ in range(lowlight_steps):
            retain += _retain_step(
                next(lowlight_batches),
                stack["model"],
                teacher,
                stack["optimizers"]["generator"],
                device,
                retain_weight,
                trainer.precision,
            )
            stack["schedulers"]["generator"].step()
        retain /= lowlight_steps

        last_metrics = {}
        source_name = "indoor"
        for _ in range(dehaze_steps):
            if rng.random() < real_probability:
                source_name, source_config, source_batches = (
                    "real",
                    real_config,
                    real_batches,
                )
            else:
                source_name, source_config, source_batches = (
                    "indoor",
                    indoor_config,
                    indoor_batches,
                )
            trainer.set_physics(
                TrainerConfig.from_mapping(source_config),
                atmosphere_mode=source_config.get("atmosphere_mode"),
            )
            last_metrics = trainer.train_step(
                batch_to_device(next(source_batches), device)
            )
            stack["schedulers"]["generator"].step()
            stack["schedulers"]["depth"].step()
            stack["schedulers"]["discriminator"].step()

        joint_iteration += 1
        epoch = joint_iteration // max(1, len(lowlight_loader))
        should_log = (
            joint_iteration == starting_iteration + 1
            or joint_iteration == maximum
            or (log_interval and joint_iteration % log_interval == 0)
        )
        if should_log:
            now = time.monotonic()
            print(
                _format_progress(
                    joint_iteration=joint_iteration,
                    maximum=maximum,
                    starting_iteration=starting_iteration,
                    epoch=epoch,
                    step_seconds=now - step_started_at,
                    elapsed_seconds=now - started_at,
                    source_name=source_name,
                    retain=retain,
                    metrics=last_metrics,
                ),
                flush=True,
            )
        if save_interval and joint_iteration % save_interval == 0:
            save_training_checkpoint(
                str(output_dir / f"iter_{joint_iteration:07d}.pth"),
                stack["model"],
                stack["depth_net"],
                stack["refine_net"],
                stack["D_clear"],
                stack["D_hazy"],
                stack["optimizers"],
                stack["schedulers"],
                4,
                joint_iteration,
                epoch,
                config,
                precision=trainer.precision,
            )

    save_training_checkpoint(
        str(output_dir / "latest.pth"),
        stack["model"],
        stack["depth_net"],
        stack["refine_net"],
        stack["D_clear"],
        stack["D_hazy"],
        stack["optimizers"],
        stack["schedulers"],
        4,
        joint_iteration,
        epoch,
        config,
        precision=trainer.precision,
    )
    print(
        f"[train] Complete: iteration={joint_iteration}/{maximum} "
        f"elapsed={format_duration(time.monotonic() - started_at)} "
        f"checkpoint={output_dir / 'latest.pth'}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
