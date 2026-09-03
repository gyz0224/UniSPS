#!/usr/bin/env python3
"""Create deterministic recursive image file lists."""

import argparse
import os
from pathlib import Path
from typing import Optional, Sequence


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def build_flist(source: Path, absolute: bool = False) -> list[str]:
    """Return sorted image paths under an existing directory."""
    source = source.expanduser()
    if not source.exists():
        raise FileNotFoundError(f"Input directory does not exist: {source}")
    if not source.is_dir():
        raise NotADirectoryError(f"Input path is not a directory: {source}")

    images = sorted(
        path.resolve()
        for path in source.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )
    if not images:
        raise ValueError(
            f"No supported images found under {source}; expected jpg/jpeg/png/bmp/tif/tiff"
        )
    if absolute:
        return [str(path) for path in images]
    working_directory = Path.cwd().resolve()
    return [os.path.relpath(path, working_directory) for path in images]


def write_flist(entries: Sequence[str], output: Path) -> None:
    """Write one path per line, creating the output parent directory."""
    output = output.expanduser()
    try:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text("".join(f"{entry}\n" for entry in entries), encoding="utf-8")
    except OSError as exc:
        raise OSError(f"Unable to write flist {output}: {exc}") from exc


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Recursively list supported dataset images in deterministic order."
    )
    parser.add_argument("--path", required=True, type=Path, help="dataset directory")
    parser.add_argument("--output", required=True, type=Path, help="output .flist path")
    parser.add_argument(
        "--absolute", action="store_true", help="write resolved absolute image paths"
    )
    return parser


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    return build_parser().parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    try:
        entries = build_flist(args.path, absolute=args.absolute)
        write_flist(entries, args.output)
    except (OSError, ValueError) as exc:
        raise SystemExit(f"flist error: {exc}") from exc
    print(f"Wrote {len(entries)} images to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
