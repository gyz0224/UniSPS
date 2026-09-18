#!/usr/bin/env python3
"""Train SPS-Net low-light enhancement with paired-reference validation."""

import argparse
from pathlib import Path
from typing import Optional, Sequence

import torch
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.optim.lr_scheduler as lrs
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from datasets.loaders import get_eval_set, get_training_set
from loss.lowlight_clip import CLIPLoss
from net.model import net
from training.lowlight_trainer import (
    LowlightTrainer,
    LowlightTrainerConfig,
    evaluate_paired_lowlight,
    load_lowlight_checkpoint,
    parse_epoch_list,
    resolve_lowlight_device,
    save_lowlight_checkpoint,
    seed_torch,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="PairLIE")
    parser.add_argument("--batchSize", type=int, default=1, help="training batch size")
    parser.add_argument("--nEpochs", type=int, default=300, help="number of epochs")
    parser.add_argument("--snapshots", type=int, default=1, help="validation interval")
    parser.add_argument("--start_iter", type=int, default=1, help="starting epoch")
    parser.add_argument("--lr", type=float, default=1e-4, help="learning rate")
    parser.add_argument("--gpu_mode", type=bool, default=True)
    parser.add_argument("--device", help="explicit torch device, e.g. cuda:0 or cpu")
    parser.add_argument("--threads", type=int, default=0, help="data-loader workers")
    parser.add_argument("--decay", type=int, default=300, help="LR decay period")
    parser.add_argument("--gamma", type=float, default=0.5, help="LR decay factor")
    parser.add_argument("--seed", type=int, default=123, help="random seed")
    parser.add_argument("--data_train", default="dataset/LOLv1/Train/input")
    parser.add_argument("--data_val", default="dataset/LOLv1/Test/input")
    parser.add_argument("--reference_val", default="dataset/LOLv1/Test/target")
    parser.add_argument("--rgb_range", type=int, default=1)
    parser.add_argument("--save_folder", default="weights/LOLv1")
    parser.add_argument("--logroot", default="logs/LOLv1")
    parser.add_argument(
        "--loss_weights", type=float, nargs=4, default=[1, 0.1, 0.1, 0.5]
    )
    parser.add_argument("--light_patch", type=int, default=64)
    parser.add_argument("--w_sem", type=float, default=0.1)
    parser.add_argument("--w_iqa", type=float, default=0.01)
    parser.add_argument(
        "--save_epochs",
        default="",
        help="comma-separated epochs or ranges, e.g. 50,60-80",
    )
    amp_group = parser.add_mutually_exclusive_group()
    amp_group.add_argument(
        "--amp", dest="amp", action="store_true", help="enable CUDA AMP"
    )
    amp_group.add_argument(
        "--no-amp", dest="amp", action="store_false", help="disable AMP"
    )
    parser.set_defaults(amp=True)
    parser.add_argument(
        "--amp-dtype", choices=("bfloat16", "float16"), default="bfloat16"
    )
    return parser


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    return build_parser().parse_args(argv)


def _write_paired_summary(
    output: Path,
    epoch: int,
    maximum: int,
    records,
) -> None:
    lines = ["=" * 60, "Best Models Metrics Summary", "=" * 60, ""]
    for label, filename in (
        ("Best PSNR Model", "best_psnr.pth"),
        ("Best SSIM Model", "best_ssim.pth"),
        ("Best LPIPS Model", "best_lpips.pth"),
    ):
        item = records[filename]
        lines.extend(
            [
                label,
                "-" * 60,
                f"Model:  {filename}",
                f"Epoch:  {item['epoch']}",
                f"PSNR:   {item['psnr']:.4f} dB",
                f"SSIM:   {item['ssim']:.4f}",
                f"LPIPS:  {item['lpips']:.4f}",
                "",
            ]
        )
    lines.extend(
        ["=" * 60, f"Current Training Status: Epoch {epoch}/{maximum}", "=" * 60]
    )
    (output / "best_models_metrics.txt").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    if args.nEpochs <= 0 or args.snapshots <= 0 or args.decay <= 0:
        raise ValueError("nEpochs, snapshots, and decay must be positive")
    device = resolve_lowlight_device(args.device, args.gpu_mode)
    seed_torch(args.seed)
    cudnn.benchmark = device.type == "cuda"
    custom_epochs = parse_epoch_list(args.save_epochs, args.nEpochs)
    if custom_epochs:
        print(f"===> Will save checkpoints at epochs: {sorted(custom_epochs)}")

    print("===> Loading datasets")
    train_set = get_training_set(args.data_train)
    validation_set = get_eval_set(args.data_val)
    train_loader = DataLoader(
        train_set,
        num_workers=args.threads,
        batch_size=args.batchSize,
        shuffle=True,
    )
    validation_loader = DataLoader(
        validation_set, num_workers=args.threads, batch_size=1, shuffle=False
    )
    print(f"Training set: {len(train_set)} samples, Validation set: {len(validation_set)} samples")

    print("===> Building model")
    model = net().to(device)
    optimizer = optim.Adam(
        model.parameters(), lr=args.lr, betas=(0.9, 0.999), eps=1e-8
    )
    if args.start_iter > 1:
        resume = Path(args.save_folder) / f"epoch_{args.start_iter - 1}.pth"
        if resume.is_file():
            print(f"===> Resuming from checkpoint: {resume}")
            load_lowlight_checkpoint(resume, model, map_location=device)
            print(f"===> Resume training from epoch {args.start_iter}")
    milestones = [
        epoch for epoch in range(1, args.nEpochs + 1) if epoch % args.decay == 0
    ]
    scheduler = lrs.MultiStepLR(optimizer, milestones, args.gamma)
    output = Path(args.save_folder)
    output.mkdir(parents=True, exist_ok=True)
    Path(args.logroot).mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(args.logroot)
    clip_criterion = CLIPLoss(device=str(device)).to(device)
    trainer = LowlightTrainer(
        model,
        optimizer,
        clip_criterion,
        LowlightTrainerConfig(
            loss_weights=tuple(args.loss_weights),
            light_patch=args.light_patch,
            w_sem=args.w_sem,
            w_iqa=args.w_iqa,
        ),
        device,
        amp_enabled=args.amp,
        amp_dtype=args.amp_dtype,
    )
    print(f"===> Training precision: {trainer.precision.mode}")

    best = {"psnr": 0.0, "ssim": 0.0, "lpips": float("inf")}
    records = {
        "best_psnr.pth": {"psnr": 0.0, "ssim": 0.0, "lpips": float("inf"), "epoch": 0},
        "best_ssim.pth": {"psnr": 0.0, "ssim": 0.0, "lpips": float("inf"), "epoch": 0},
        "best_lpips.pth": {"psnr": 0.0, "ssim": 0.0, "lpips": float("inf"), "epoch": 0},
    }
    best_epochs = {"psnr": 0, "ssim": 0, "lpips": 0}
    print("===> Start training...")
    try:
        for epoch in range(args.start_iter, args.nEpochs + 1):
            print(f"\n===> Epoch {epoch}/{args.nEpochs}")
            trainer.train_epoch(train_loader, epoch, log_interval=100)
            scheduler.step()
            if epoch % args.snapshots == 0:
                values = evaluate_paired_lowlight(
                    model, validation_loader, args.reference_val, device
                )
                for name, value in values.items():
                    writer.add_scalar(name, value, epoch)
                print(f"===> Avg.PSNR: {values['psnr']:.4f} dB ")
                print(f"===> Avg.SSIM: {values['ssim']:.4f} ")
                print(f"===> Avg.LPIPS: {values['lpips']:.4f} ")
                rules = {
                    "psnr": values["psnr"] > best["psnr"],
                    "ssim": values["ssim"] > best["ssim"],
                    "lpips": values["lpips"] < best["lpips"],
                }
                for metric, improved in rules.items():
                    if not improved:
                        continue
                    best[metric] = values[metric]
                    best_epochs[metric] = epoch
                    filename = f"best_{metric}.pth"
                    records[filename] = {**values, "epoch": epoch}
                    save_lowlight_checkpoint(output / filename, model)
                    print(
                        f"    ★ New best {metric.upper()}! Saved to {filename} "
                        f"(PSNR={values['psnr']:.4f}, SSIM={values['ssim']:.4f}, "
                        f"LPIPS={values['lpips']:.4f})"
                    )
                _write_paired_summary(output, epoch, args.nEpochs, records)
                print(
                    "    Best so far - "
                    f"PSNR: {best['psnr']:.4f} (epoch {best_epochs['psnr']}), "
                    f"SSIM: {best['ssim']:.4f} (epoch {best_epochs['ssim']}), "
                    f"LPIPS: {best['lpips']:.4f} (epoch {best_epochs['lpips']})"
                )
            if custom_epochs and epoch in custom_epochs:
                target = output / f"epoch_{epoch}.pth"
                save_lowlight_checkpoint(target, model)
                print(f"Checkpoint saved to {target}")
    finally:
        writer.close()
    print("\n===> Training complete!")
    print(f"Best PSNR: {best['psnr']:.4f} at epoch {best_epochs['psnr']}")
    print(f"Best SSIM: {best['ssim']:.4f} at epoch {best_epochs['ssim']}")
    print(f"Best LPIPS: {best['lpips']:.4f} at epoch {best_epochs['lpips']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
