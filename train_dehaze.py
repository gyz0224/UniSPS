#!/usr/bin/env python3
"""Train the SPS-based dehazing branch on unpaired clear and hazy images."""

import argparse
import time
from pathlib import Path
from typing import Optional, Sequence

from training.dehaze_trainer import load_training_checkpoint, save_training_checkpoint
from training.experiment import (
    EXPERIMENT_CHOICES,
    ExperimentLayout,
    ensure_new_training_output,
)
from training.runtime import (
    batch_to_device,
    build_training_stack,
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
        help="isolated run identity and low-light initializer",
    )
    parser.add_argument("--config", required=True, help="dehaze YAML configuration")
    parser.add_argument("--resume", help="full dehaze checkpoint to resume")
    parser.add_argument("--pretrained-sps", help="legacy SPS checkpoint (model weights only)")
    parser.add_argument(
        "--init-checkpoint",
        help="prior-stage checkpoint (all module weights, fresh optimizers)",
    )
    parser.add_argument("--device", help="torch device, e.g. cuda:0 or cpu")
    parser.add_argument("--stage", type=int, choices=(1, 2, 3), help="override stage")
    parser.add_argument("--max-iterations", type=int, help="override iteration limit")
    parser.add_argument(
        "--log-interval",
        type=int,
        help="print metrics every N iterations (0 disables periodic metrics)",
    )
    parser.add_argument("--output", help="override checkpoint directory")
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


def configure_experiment(
    args: argparse.Namespace,
    config: dict,
) -> ExperimentLayout:
    """Apply one experiment's stage chain without trusting stale YAML paths."""
    layout = ExperimentLayout(args.experiment)
    stage = int(config.get("stage", 1))
    if stage not in (1, 2, 3):
        raise ValueError("train_dehaze.py requires stage 1, 2, or 3")
    if args.pretrained_sps and args.init_checkpoint:
        raise ValueError("--pretrained-sps and --init-checkpoint are mutually exclusive")
    if stage > 1 and args.pretrained_sps:
        raise ValueError("--pretrained-sps is only valid for Stage 1")

    config["experiment"] = args.experiment
    config["output_dir"] = str(args.output or layout.checkpoint_dir(stage))
    if args.resume:
        config["pretrained_sps"] = None
        config["initial_checkpoint"] = None
    elif args.init_checkpoint:
        config["pretrained_sps"] = None
        config["initial_checkpoint"] = args.init_checkpoint
    elif stage == 1:
        config["pretrained_sps"] = args.pretrained_sps or str(layout.initial_weight)
        config["initial_checkpoint"] = None
    else:
        config["pretrained_sps"] = None
        config["initial_checkpoint"] = str(layout.checkpoint(stage - 1))

    ensure_new_training_output(Path(config["output_dir"]), args.resume)
    return layout


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    config = load_yaml_config(args.config)
    if args.stage is not None:
        config["stage"] = args.stage
    if args.max_iterations is not None:
        if args.max_iterations <= 0:
            raise ValueError("--max-iterations must be positive")
        config["max_iterations"] = args.max_iterations
    if args.log_interval is not None:
        if args.log_interval < 0:
            raise ValueError("--log-interval must be non-negative")
        config["log_interval"] = args.log_interval
    if args.amp is not None:
        config["amp"] = args.amp
    if args.amp_dtype is not None:
        config["amp_dtype"] = args.amp_dtype
    layout = configure_experiment(args, config)

    device = resolve_device(args.device or config.get("device"))
    print(
        f"[setup] experiment={layout.experiment} config={config['_config_path']} "
        f"stage={config.get('stage', 1)} device={device}",
        flush=True,
    )
    print("[setup] Reading the training dataset...", flush=True)
    loader = make_dehaze_loader(config, return_paths=True)
    print(
        f"[setup] Dataset ready: samples={len(loader.dataset)} "
        f"batches/epoch={len(loader)} batch_size={loader.batch_size} "
        f"workers={loader.num_workers}",
        flush=True,
    )
    batches = infinite_batches(loader)
    print("[setup] Building the training stack...", flush=True)
    stack = build_training_stack(config, device=device)
    trainer = stack["trainer"]
    epoch = 0

    if args.resume:
        print(f"[setup] Restoring training state from {args.resume}...", flush=True)
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
        trainer.iteration = int(checkpoint.get("iteration", 0))
        epoch = int(checkpoint.get("epoch", 0))
        print(
            f"[setup] Resume complete: iteration={trainer.iteration} epoch={epoch}",
            flush=True,
        )

    maximum = int(config.get("max_iterations", 150000))
    log_interval = int(config.get("log_interval", 100))
    save_interval = int(config.get("save_interval", 2000))
    output_dir = Path(config["output_dir"])
    starting_iteration = trainer.iteration
    started_at = time.monotonic()
    print(
        f"[train] Started: iteration={starting_iteration}/{maximum} "
        f"log_interval={log_interval} save_interval={save_interval} "
        f"output={output_dir}",
        flush=True,
    )

    while trainer.iteration < maximum:
        is_first_iteration = trainer.iteration == starting_iteration
        if is_first_iteration:
            print(
                "[train] Running the first iteration; data workers and CUDA kernels "
                "can make this step slower than later ones...",
                flush=True,
            )
        step_started_at = time.monotonic()
        batch = batch_to_device(next(batches), device)
        progress_callback = None
        if is_first_iteration:
            progress_callback = lambda message: print(
                f"[train] First iteration — {message}...", flush=True
            )
        metrics = trainer.train_step(batch, progress_callback=progress_callback)
        for scheduler in stack["schedulers"].values():
            scheduler.step()
        if trainer.iteration % max(1, len(loader)) == 0:
            epoch += 1
        should_log = (
            trainer.iteration == starting_iteration + 1
            or trainer.iteration == maximum
            or (log_interval and trainer.iteration % log_interval == 0)
        )
        if should_log:
            now = time.monotonic()
            completed = trainer.iteration - starting_iteration
            elapsed = now - started_at
            average_seconds = elapsed / max(1, completed)
            eta_seconds = average_seconds * max(0, maximum - trainer.iteration)
            progress = 100.0 * trainer.iteration / maximum
            print(
                f"[train] iter={trainer.iteration}/{maximum} ({progress:.2f}%) "
                f"epoch={epoch} step={now - step_started_at:.1f}s "
                f"avg={average_seconds:.1f}s/iter eta={format_duration(eta_seconds)} "
                f"{format_metrics(metrics)}",
                flush=True,
            )
        if save_interval and trainer.iteration % save_interval == 0:
            checkpoint_path = output_dir / f"iter_{trainer.iteration:07d}.pth"
            print(f"[checkpoint] Saving {checkpoint_path}...", flush=True)
            save_training_checkpoint(
                str(checkpoint_path),
                stack["model"],
                stack["depth_net"],
                stack["refine_net"],
                stack["D_clear"],
                stack["D_hazy"],
                stack["optimizers"],
                stack["schedulers"],
                stack["stage"],
                trainer.iteration,
                epoch,
                config,
                precision=trainer.precision,
            )
            print(f"[checkpoint] Saved {checkpoint_path}", flush=True)

    latest_path = output_dir / "latest.pth"
    print(f"[checkpoint] Saving final checkpoint to {latest_path}...", flush=True)
    save_training_checkpoint(
        str(latest_path),
        stack["model"],
        stack["depth_net"],
        stack["refine_net"],
        stack["D_clear"],
        stack["D_hazy"],
        stack["optimizers"],
        stack["schedulers"],
        stack["stage"],
        trainer.iteration,
        epoch,
        config,
        precision=trainer.precision,
    )
    print(
        f"[train] Complete: iteration={trainer.iteration}/{maximum} "
        f"elapsed={format_duration(time.monotonic() - started_at)} "
        f"checkpoint={latest_path}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
