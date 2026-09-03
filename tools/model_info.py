"""Quick script to report parameter counts for PairLIEpro net."""
import argparse
import sys
from pathlib import Path
from typing import Optional, Sequence

import torch
import torch.nn as nn

# Ensure repo root is on sys.path so "net" can be imported when run as a script.
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from net.model import net


def count_params(model: nn.Module, trainable_only: bool = True) -> int:
    """Return total or trainable parameter count."""
    if trainable_only:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    return sum(p.numel() for p in model.parameters())


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Model parameter counter for PairLIEpro")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu", help="Device to init the model on")
    return parser


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    return build_parser().parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)

    device = torch.device(args.device)
    model = net().to(device)

    total = count_params(model, trainable_only=False)
    trainable = count_params(model, trainable_only=True)

    print(f"Device: {device}")
    print(f"Total params   : {total:,}")
    print(f"Trainable params: {trainable:,}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
