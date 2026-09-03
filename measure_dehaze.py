#!/usr/bin/env python3
"""Compute paired metrics, optional LPIPS, or no-reference FADE."""

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence

from metrics.dehaze_evaluator import (
    evaluate_dehaze_directory,
    evaluate_fade_directory,
)
from training.experiment import (
    DEHAZE_PRESETS,
    EXPERIMENT_CHOICES,
    ExperimentLayout,
)


@dataclass(frozen=True)
class EvaluationSpec:
    experiment: str
    stage: str
    dataset: str
    prediction: Path
    reference: Optional[Path]
    pairing: str
    mode: str
    lpips: bool
    metrics_output: Path


STAGE_PRESETS = DEHAZE_PRESETS


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
        choices=tuple(STAGE_PRESETS),
        help=(
            "plain stages use paired SOTS Indoor, *-ihaze use paired I-HAZE, "
            "*-outdoor use paired SOTS Outdoor, *-hsts-synthetic use paired "
            "HSTS, *-hsts-real use HSTS FADE, and stage3/4-real use RTTS FADE"
        ),
    )
    parser.add_argument(
        "--prediction",
        type=Path,
        help="override the preset prediction directory",
    )
    parser.add_argument(
        "--reference",
        type=Path,
        help=(
            "override/provide a genuine paired clean reference directory; "
            "supplying this switches a real preset from FADE to paired metrics"
        ),
    )
    parser.add_argument(
        "--pairing",
        choices=("ihaze", "sots", "sots-outdoor", "same-stem"),
        help="override the preset filename pairing rule",
    )
    parser.add_argument(
        "--print-every",
        type=int,
        default=25,
        help="print progress every N images; use 0 to disable progress",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="parallel image workers for FADE only; default: 1",
    )
    parser.add_argument(
        "--lpips-device",
        help=(
            "LPIPS device for HSTS Synthetic; default: CUDA when available, "
            "otherwise CPU"
        ),
    )
    parser.add_argument(
        "--metrics-output",
        type=Path,
        help="override the canonical JSON metrics path",
    )
    return parser


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    return build_parser().parse_args(argv)


def resolve_evaluation(args: argparse.Namespace) -> EvaluationSpec:
    preset = STAGE_PRESETS[args.stage]
    layout = ExperimentLayout(args.experiment)
    reference = args.reference or preset.reference_dir
    return EvaluationSpec(
        experiment=args.experiment,
        stage=args.stage,
        dataset=preset.dataset_name,
        prediction=args.prediction or layout.dehaze_result(args.stage),
        reference=reference,
        pairing=args.pairing or preset.pairing,
        mode="fade" if reference is None else "full-reference",
        lpips=preset.lpips and reference is not None,
        metrics_output=args.metrics_output or layout.dehaze_metrics(args.stage),
    )


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    if args.print_every < 0:
        print("error: --print-every must be non-negative", file=sys.stderr)
        return 2
    if args.workers < 1:
        print("error: --workers must be at least 1", file=sys.stderr)
        return 2
    spec = resolve_evaluation(args)
    lpips_device = None
    if spec.lpips:
        if args.lpips_device:
            lpips_device = args.lpips_device
        else:
            import torch

            lpips_device = "cuda" if torch.cuda.is_available() else "cpu"
    if spec.mode == "fade":
        print(
            f"[metrics] stage={spec.stage} dataset={spec.dataset} "
            f"prediction={spec.prediction} metric=FADE "
            f"workers={args.workers}",
            flush=True,
        )
    else:
        print(
            f"[metrics] stage={spec.stage} dataset={spec.dataset} "
            f"prediction={spec.prediction} reference={spec.reference} "
            f"pairing={spec.pairing} lpips_device={lpips_device or 'disabled'}",
            flush=True,
        )

    def report_progress(index: int, total: int, path: Path) -> None:
        if args.print_every and (
            index == 1 or index % args.print_every == 0 or index == total
        ):
            print(
                f"[metrics] {index}/{total} {path}",
                flush=True,
            )

    if spec.mode == "fade":
        fade_values = evaluate_fade_directory(
            spec.prediction,
            progress=report_progress,
            workers=args.workers,
        )
        print(f"Samples:    {fade_values.samples}")
        print(f"FADE:       {fade_values.fade:.4f}")
        metric_values = {
            "samples": fade_values.samples,
            "fade": fade_values.fade,
        }
    else:
        assert spec.reference is not None
        paired_values = evaluate_dehaze_directory(
            spec.prediction,
            spec.reference,
            pairing=spec.pairing,
            progress=report_progress,
            lpips_device=lpips_device,
        )
        print(f"Samples:    {paired_values.samples}")
        print(f"PSNR:       {paired_values.psnr_db:.4f} dB")
        print(f"SSIM:       {paired_values.ssim:.4f}")
        print(f"CIEDE2000:  {paired_values.ciede2000:.4f}")
        if paired_values.lpips is not None:
            print(f"LPIPS:      {paired_values.lpips:.4f}")
        metric_values = {
            "samples": paired_values.samples,
            "psnr_db": paired_values.psnr_db,
            "ssim": paired_values.ssim,
            "ciede2000": paired_values.ciede2000,
            "lpips": paired_values.lpips,
        }
    payload = {
        "experiment": spec.experiment,
        "stage": spec.stage,
        "dataset": spec.dataset,
        "mode": spec.mode,
        "prediction": str(spec.prediction),
        "reference": str(spec.reference) if spec.reference else None,
        "pairing": spec.pairing,
        **metric_values,
    }
    spec.metrics_output.parent.mkdir(parents=True, exist_ok=True)
    spec.metrics_output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"Metrics:    {spec.metrics_output}")
    return 0


__all__ = [
    "EvaluationSpec",
    "STAGE_PRESETS",
    "build_parser",
    "main",
    "parse_args",
    "resolve_evaluation",
]


if __name__ == "__main__":
    raise SystemExit(main())
