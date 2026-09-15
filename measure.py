#!/usr/bin/env python3
"""Compute PSNR, SSIM, LPIPS, and NIQE for paired low-light results."""

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence

import torch

from metrics.image_quality import calculate_psnr, calculate_ssim, ssim
from metrics.lpips_evaluator import evaluate_lowlight_directory, resolve_image_files
from training.experiment import EXPERIMENT_CHOICES, ExperimentLayout


@dataclass(frozen=True)
class MeasurementSpec:
    experiment: str
    stage: str
    dataset_key: str
    dataset_name: str
    image_source: object
    label_dir: Path
    pairing: str
    metrics_output: Path


def metrics(im_dir, label_dir, device=None, pairing="same-name"):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    return evaluate_lowlight_directory(
        im_dir,
        label_dir,
        device,
        pairing=pairing,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Compute PSNR/SSIM/LPIPS/NIQE")
    parser.add_argument(
        "--experiment",
        required=True,
        choices=EXPERIMENT_CHOICES,
        help="isolated run identity",
    )
    parser.add_argument(
        "--stage",
        choices=("pretrained", "stage4"),
        default="pretrained",
    )
    parser.add_argument(
        "--im_dir",
        nargs="+",
        default=None,
        help="glob (quoted) or list of enhanced images",
    )
    parser.add_argument(
        "--label_dir",
        help="override the experiment's native reference directory",
    )
    parser.add_argument(
        "--pairing",
        choices=("same-name", "sice"),
        help="override the experiment's prediction/reference pairing rule",
    )
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument(
        "--metrics-output",
        type=Path,
        help="override the canonical JSON metrics path",
    )
    return parser


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    return build_parser().parse_args(argv)


def resolve_measurement(args: argparse.Namespace) -> MeasurementSpec:
    layout = ExperimentLayout(args.experiment)
    dataset = layout.lowlight_dataset
    image_source = (
        args.im_dir
        if args.im_dir
        else [str(layout.lowlight_result(args.stage) / "I" / "*")]
    )
    resolved_source = image_source if len(image_source) > 1 else image_source[0]
    return MeasurementSpec(
        experiment=args.experiment,
        stage=args.stage,
        dataset_key=dataset.dataset_key,
        dataset_name=dataset.dataset_name,
        image_source=resolved_source,
        label_dir=Path(args.label_dir) if args.label_dir else dataset.reference_dir,
        pairing=args.pairing or dataset.pairing,
        metrics_output=args.metrics_output or layout.lowlight_metrics(args.stage),
    )


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    spec = resolve_measurement(args)
    samples = len(resolve_image_files(spec.image_source))
    psnr, ssim_value, lpips_value, niqe_value = metrics(
        spec.image_source,
        spec.label_dir,
        args.device,
        pairing=spec.pairing,
    )
    print(f"===> Avg.PSNR: {psnr:.4f} dB ")
    print(f"===> Avg.SSIM: {ssim_value:.4f} ")
    print(f"===> Avg.LPIPS: {lpips_value:.4f} ")
    print(f"===> Avg.NIQE: {niqe_value:.4f} ")
    payload = {
        "experiment": spec.experiment,
        "stage": spec.stage,
        "dataset": spec.dataset_key,
        "dataset_name": spec.dataset_name,
        "prediction": str(spec.image_source),
        "reference": str(spec.label_dir),
        "pairing": spec.pairing,
        "samples": samples,
        "psnr_db": psnr,
        "ssim": ssim_value,
        "lpips": lpips_value,
        "niqe": niqe_value,
    }
    spec.metrics_output.parent.mkdir(parents=True, exist_ok=True)
    spec.metrics_output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"===> Metrics: {spec.metrics_output}")
    return 0


__all__ = [
    "build_parser",
    "calculate_psnr",
    "calculate_ssim",
    "main",
    "metrics",
    "parse_args",
    "ssim",
]


if __name__ == "__main__":
    raise SystemExit(main())
