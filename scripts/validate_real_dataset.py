#!/usr/bin/env python3
"""Validate the canonical D4+ real-outdoor image and sky-mask layout."""

import argparse
from pathlib import Path
from typing import Optional, Sequence

from datasets.dehaze import validate_real_dataset_layout


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Check D4+ real-outdoor image/mask basenames, readability, dimensions, "
            "and expected archive counts."
        )
    )
    parser.add_argument("--clean", type=Path, default=Path("dataset/real/clear"))
    parser.add_argument("--hazy", type=Path, default=Path("dataset/real/hazy"))
    parser.add_argument(
        "--clean-masks",
        type=Path,
        default=Path("dataset/real/masks/clear"),
    )
    parser.add_argument(
        "--hazy-masks",
        type=Path,
        default=Path("dataset/real/masks/hazy"),
    )
    parser.add_argument(
        "--expected-clean",
        type=int,
        default=3577,
        help="expected clean pair count; use 0 to disable the count check",
    )
    parser.add_argument(
        "--expected-hazy",
        type=int,
        default=2902,
        help="expected hazy pair count; use 0 to disable the count check",
    )
    return parser


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    return build_parser().parse_args(argv)


def _check_expected(actual: int, expected: int, domain: str) -> None:
    if expected < 0:
        raise ValueError(f"expected {domain} count must be non-negative")
    if expected and actual != expected:
        raise ValueError(
            f"Unexpected {domain} pair count: expected {expected}, found {actual}"
        )


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    try:
        counts = validate_real_dataset_layout(
            args.clean,
            args.hazy,
            args.clean_masks,
            args.hazy_masks,
        )
        _check_expected(counts["clean"], args.expected_clean, "clean")
        _check_expected(counts["hazy"], args.expected_hazy, "hazy")
    except (OSError, ValueError) as exc:
        raise SystemExit(f"real dataset validation error: {exc}") from exc

    print(
        "Validated D4+ real dataset: "
        f"clean={counts['clean']} pairs, hazy={counts['hazy']} pairs"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
