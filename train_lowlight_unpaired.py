#!/usr/bin/env python3
"""Train SPS low-light enhancement with no-reference IQA validation."""

import argparse
import shutil
from pathlib import Path
from typing import Dict, Optional, Sequence

import torch
import torch.backends.cudnn as cudnn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from datasets.loaders import get_eval_set, get_training_set
from loss.lowlight_clip import CLIPLoss
from net.model import net
from training.lowlight_trainer import (
    LowlightTrainer,
    LowlightTrainerConfig,
    evaluate_no_reference_lowlight,
    load_lowlight_checkpoint,
    resolve_lowlight_device,
    save_lowlight_checkpoint,
    seed_torch,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="SPS-Net", conflict_handler="resolve"
    )
    parser.add_argument("--batchSize", type=int, default=1)
    parser.add_argument("--nEpochs", type=int, default=300)
    parser.add_argument("--snapshots", type=int, default=1)
    parser.add_argument("--start_iter", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--gpu_mode", type=bool, default=True)
    parser.add_argument("--device", help="explicit torch device, e.g. cuda:0 or cpu")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--decay", type=int, default=100)
    parser.add_argument("--gamma", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--data_train", default="dataset/LOLv1/Train/input")
    parser.add_argument("--data_val", default="dataset/LOLv1/Test/input")
    parser.add_argument("--save_folder", default="weights/LOLv1_Unpaired")
    parser.add_argument("--logroot", default="logs/LOLv1_Unpaired")
    parser.add_argument(
        "--loss_weights", type=float, nargs=4, default=[1, 0.1, 0.1, 0.5]
    )
    parser.add_argument("--light_patch", type=int, default=64)
    parser.add_argument("--mean_val", type=float, default=0.5)
    parser.add_argument("--w_sem", type=float, default=0.1)
    parser.add_argument("--w_iqa", type=float, default=0.01)
    parser.add_argument("--weights_dir", default="weights/metrics_weights")
    return parser


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    return build_parser().parse_args(argv)


def build_no_reference_metrics(
    weights_dir: Path, device: torch.device
) -> Dict[str, torch.nn.Module]:
    """Build NIQE/BRISQUE/MUSIQ with the original local-weight policy."""
    try:
        import pyiqa
    except ImportError as exc:
        raise RuntimeError("No-reference validation requires the pyiqa package") from exc
    cache = Path.home() / ".cache/torch/hub/pyiqa"
    cache.mkdir(parents=True, exist_ok=True)
    filenames = (
        "niqe_modelparameters.mat",
        "brisque_svm_weights.pth",
        "musiq_koniq_ckpt-2d880026.pth",
    )
    for filename in filenames:
        source = weights_dir / filename
        if source.is_file():
            shutil.copy2(source, cache / filename)
        elif filename == "brisque_svm_weights.pth":
            print(f"[Warn] Missing {filename}. BRISQUE might try to download it.")
    niqe = pyiqa.create_metric("niqe", device=device)
    brisque = pyiqa.create_metric("brisque", device=device)
    musiq = pyiqa.create_metric("musiq", device=device, pretrained=False)
    musiq_checkpoint = weights_dir / "musiq_koniq_ckpt-2d880026.pth"
    if musiq_checkpoint.is_file():
        state = torch.load(musiq_checkpoint, map_location=device)
        if "params" in state:
            state = state["params"]
        musiq.net.load_state_dict(state, strict=True)
        musiq.eval()
    return {"niqe": niqe, "brisque": brisque, "musiq": musiq}


def _write_no_reference_summary(
    output: Path, epoch: int, maximum: int, records
) -> None:
    lines = ["=" * 60, "Best Models Metrics Summary", "=" * 60, ""]
    definitions = (
        ("Best NIQE Model (lower is better)", "best_niqe.pth"),
        ("Best BRISQUE Model (lower is better)", "best_brisque.pth"),
        ("Best MUSIQ Model (higher is better)", "best_musiq.pth"),
    )
    for label, filename in definitions:
        item = records[filename]
        lines.extend(
            [
                label,
                "-" * 60,
                f"Model:  {filename}",
                f"Epoch:  {item['epoch']}",
                f"NIQE:   {item['niqe']:.4f}",
                f"BRISQUE:{item['brisque']:.4f}",
                f"MUSIQ:  {item['musiq']:.4f}",
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
    if args.nEpochs <= 0 or args.snapshots <= 0:
        raise ValueError("nEpochs and snapshots must be positive")
    device = resolve_lowlight_device(args.device, args.gpu_mode)
    seed_torch(args.seed)
    cudnn.benchmark = device.type == "cuda"
    output = Path(args.save_folder)
    output.mkdir(parents=True, exist_ok=True)
    Path(args.logroot).mkdir(parents=True, exist_ok=True)

    print("===> Loading datasets")
    train_loader = DataLoader(
        get_training_set(args.data_train),
        num_workers=args.threads,
        batch_size=args.batchSize,
        shuffle=True,
    )
    validation_loader = DataLoader(
        get_eval_set(args.data_val),
        num_workers=args.threads,
        batch_size=1,
        shuffle=False,
    )
    print("===> Building model & metrics")
    model = net().to(device)
    optimizer = optim.Adam(
        model.parameters(), lr=args.lr, betas=(0.9, 0.999), eps=1e-8
    )
    clip_criterion = CLIPLoss(device=str(device)).to(device)
    trainer = LowlightTrainer(
        model,
        optimizer,
        clip_criterion,
        LowlightTrainerConfig(
            loss_weights=tuple(args.loss_weights),
            light_patch=args.light_patch,
            mean_val=args.mean_val,
            w_sem=args.w_sem,
            w_iqa=args.w_iqa,
        ),
        device,
    )
    metrics = build_no_reference_metrics(Path(args.weights_dir), device)
    if args.start_iter > 1:
        resume = output / f"epoch_{args.start_iter - 1}.pth"
        if resume.is_file():
            load_lowlight_checkpoint(resume, model, map_location=device)
    writer = SummaryWriter(args.logroot)
    best = {"niqe": float("inf"), "brisque": float("inf"), "musiq": float("-inf")}
    best_epochs = {"niqe": 0, "brisque": 0, "musiq": 0}
    initial = {"niqe": float("inf"), "brisque": float("inf"), "musiq": float("-inf"), "epoch": 0}
    records = {
        "best_niqe.pth": dict(initial),
        "best_brisque.pth": dict(initial),
        "best_musiq.pth": dict(initial),
    }
    print("===> Start training...")
    try:
        for epoch in range(args.start_iter, args.nEpochs + 1):
            trainer.train_epoch(train_loader, epoch, log_interval=5)
            if epoch % args.snapshots == 0:
                if device.type == "cuda":
                    torch.cuda.empty_cache()
                values = evaluate_no_reference_lowlight(
                    model, validation_loader, metrics, device
                )
                print(
                    f"Epoch {epoch}: NIQE={values['niqe']:.4f}, "
                    f"BRISQUE={values['brisque']:.4f}, MUSIQ={values['musiq']:.4f}"
                )
                for name, value in values.items():
                    writer.add_scalar(name.upper(), value, epoch)
                improved = {
                    "niqe": values["niqe"] < best["niqe"],
                    "brisque": values["brisque"] < best["brisque"],
                    "musiq": values["musiq"] > best["musiq"],
                }
                for name, is_improved in improved.items():
                    if not is_improved:
                        continue
                    best[name] = values[name]
                    best_epochs[name] = epoch
                    filename = f"best_{name}.pth"
                    records[filename] = {**values, "epoch": epoch}
                    save_lowlight_checkpoint(output / filename, model)
                    print(
                        f"    ★ New best {name.upper()}! Saved to {filename} "
                        f"(NIQE={values['niqe']:.4f}, "
                        f"BRISQUE={values['brisque']:.4f}, "
                        f"MUSIQ={values['musiq']:.4f})"
                    )
                _write_no_reference_summary(output, epoch, args.nEpochs, records)
                print(
                    "    Best so far - "
                    f"NIQE: {best['niqe']:.4f} (epoch {best_epochs['niqe']}), "
                    f"BRISQUE: {best['brisque']:.4f} (epoch {best_epochs['brisque']}), "
                    f"MUSIQ: {best['musiq']:.4f} (epoch {best_epochs['musiq']})"
                )
            save_lowlight_checkpoint(output / f"epoch_{epoch}.pth", model)
    finally:
        writer.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
