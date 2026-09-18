"""Shared automatic mixed-precision helpers for SPS-Net training."""

from contextlib import nullcontext
from typing import Iterable, Optional

import torch
import torch.nn as nn


AMP_DTYPE_ALIASES = {
    "bf16": "bfloat16",
    "bfloat16": "bfloat16",
    "fp16": "float16",
    "float16": "float16",
}


def normalize_amp_dtype(value: str) -> str:
    """Return one canonical AMP dtype name or raise a clear config error."""
    normalized = AMP_DTYPE_ALIASES.get(str(value).lower())
    if normalized is None:
        choices = ", ".join(sorted(AMP_DTYPE_ALIASES))
        raise ValueError(f"amp_dtype must be one of: {choices}")
    return normalized


class TrainingPrecision:
    """Own autocast and optional FP16 gradient scaling for one training stack."""

    def __init__(
        self,
        device: torch.device,
        enabled: bool = False,
        dtype: str = "bfloat16",
    ) -> None:
        self.device = torch.device(device)
        self.requested = bool(enabled)
        self.dtype_name = normalize_amp_dtype(dtype)
        self.enabled = self.requested and self.device.type == "cuda"
        if (
            self.enabled
            and self.dtype_name == "bfloat16"
            and not torch.cuda.is_bf16_supported()
        ):
            raise RuntimeError(
                "bfloat16 AMP was requested, but this CUDA device does not support it; "
                "set amp_dtype: float16 or disable AMP"
            )
        self.dtype = (
            torch.bfloat16 if self.dtype_name == "bfloat16" else torch.float16
        )
        self.scaler = torch.amp.GradScaler(
            "cuda", enabled=self.enabled and self.dtype == torch.float16
        )

    @property
    def mode(self) -> str:
        if self.enabled:
            return f"AMP-{self.dtype_name}"
        if self.requested:
            return "FP32 (AMP inactive on non-CUDA device)"
        return "FP32"

    def autocast(self):
        if not self.enabled:
            return nullcontext()
        return torch.autocast(
            device_type="cuda",
            dtype=self.dtype,
            enabled=True,
        )

    def backward_step(
        self,
        loss: torch.Tensor,
        optimizer: torch.optim.Optimizer,
        *,
        parameters: Optional[Iterable[nn.Parameter]] = None,
        max_grad_norm: float = 0.0,
    ) -> None:
        """Backpropagate, optionally unscale/clip, and step one optimizer."""
        if self.scaler.is_enabled():
            self.scaler.scale(loss).backward()
            if max_grad_norm > 0:
                self.scaler.unscale_(optimizer)
            self._clip(parameters, max_grad_norm)
            self.scaler.step(optimizer)
            return

        loss.backward()
        self._clip(parameters, max_grad_norm)
        optimizer.step()

    def update(self) -> None:
        """Update FP16's dynamic scale after all optimizer steps in one phase."""
        if self.scaler.is_enabled():
            self.scaler.update()

    def state_dict(self) -> dict:
        return self.scaler.state_dict()

    def load_state_dict(self, state: dict) -> None:
        if self.scaler.is_enabled() and state:
            self.scaler.load_state_dict(state)

    @staticmethod
    def _clip(
        parameters: Optional[Iterable[nn.Parameter]], max_grad_norm: float
    ) -> None:
        if parameters is not None and max_grad_norm > 0:
            nn.utils.clip_grad_norm_(list(parameters), max_grad_norm)


__all__ = ["TrainingPrecision", "normalize_amp_dtype"]
