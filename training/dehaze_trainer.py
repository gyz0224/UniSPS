"""Three-phase optimizer loop for unpaired SPS-Net dehazing."""

from contextlib import contextmanager
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any, Callable, Dict, Mapping, Optional, Sequence

import torch
import torch.nn as nn

from loss.dehaze_loss import (
    DehazeLossWeights,
    cycle_loss,
    depth_pseudo_loss,
    generator_total_loss,
    lsgan_discriminator_loss,
    lsgan_generator_loss,
    scattering_loss,
    semantic_consistency_loss,
)
from net.dehaze import depth_from_transmission
from net.rehaze import PhysicalHazeRenderer
from training.precision import TrainingPrecision, normalize_amp_dtype


@dataclass(frozen=True)
class TrainerConfig:
    beta_min: float = 0.6
    beta_max: float = 1.8
    depth_min: float = 0.2
    depth_max: float = 5.0
    transmission_min: float = 0.05
    transmission_max: float = 0.95
    cycle: float = 1.0
    gan: float = 0.2
    scattering: float = 1.0
    contrast: float = 1e-4
    semantic: float = 0.05
    dehaze_lr: float = 1e-4
    depth_lr: float = 1e-4
    discriminator_lr: float = 1e-5
    shared_lr: float = 1e-6
    lowlight_lr: float = 5e-6
    weight_decay: float = 0.0
    grad_clip: float = 20.0
    use_dcp_pseudo_depth: bool = False
    eps: float = 1e-6

    amp: bool = False
    amp_dtype: str = "bfloat16"

    def __post_init__(self) -> None:
        if not 0.0 < self.beta_min < self.beta_max:
            raise ValueError("Trainer beta range must satisfy 0 < min < max")
        if not 0.0 <= self.depth_min < self.depth_max:
            raise ValueError("Trainer depth range must satisfy 0 <= min < max")
        if not 0.0 < self.transmission_min < self.transmission_max <= 1.0:
            raise ValueError("Trainer transmission range must satisfy 0 < min < max <= 1")
        for name in (
            "dehaze_lr",
            "depth_lr",
            "discriminator_lr",
            "shared_lr",
            "lowlight_lr",
        ):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")

        normalize_amp_dtype(self.amp_dtype)

    @classmethod
    def from_mapping(cls, values: Mapping[str, Any]) -> "TrainerConfig":
        allowed = {field.name for field in fields(cls)}
        return cls(**{key: value for key, value in values.items() if key in allowed})

    @property
    def loss_weights(self) -> DehazeLossWeights:
        return DehazeLossWeights(
            cycle=self.cycle,
            gan=self.gan,
            scattering=self.scattering,
            contrast=self.contrast,
            semantic=self.semantic,
        )


def set_requires_grad(module: nn.Module, enabled: bool) -> None:
    """Uniformly enable or freeze all parameters in ``module``."""
    for parameter in module.parameters():
        parameter.requires_grad_(enabled)


@contextmanager
def temporarily_frozen(*modules: nn.Module):
    """Freeze module parameters temporarily while preserving input gradients."""
    states = [[parameter.requires_grad for parameter in module.parameters()] for module in modules]
    try:
        for module in modules:
            set_requires_grad(module, False)
        yield
    finally:
        for module, module_states in zip(modules, states):
            for parameter, enabled in zip(module.parameters(), module_states):
                parameter.requires_grad_(enabled)


def configure_stage(model: nn.Module, stage: int) -> None:
    """Apply the Stage 1-4 SPS freeze policy without renaming any modules."""
    if stage not in {1, 2, 3, 4}:
        raise ValueError("stage must be one of 1, 2, 3, or 4")

    set_requires_grad(model, False)
    set_requires_grad(model.dehaze_head, True)
    set_requires_grad(model.sem_net, False)

    if stage in {2, 3}:
        if len(model.N_net.encoder) == 0:
            raise ValueError("N_net.encoder has no TransformerBlock to unfreeze")
        set_requires_grad(model.N_net.encoder[-1], True)
    elif stage == 4:
        for module in (
            model.N_net.patch_embed,
            model.N_net.encoder,
            model.N_net.output,
            model.L_net,
            model.R_net,
            model.Gamma_Predictor,
            model.illp,
            model.dehaze_head,
        ):
            set_requires_grad(module, True)


def _unique_trainable(modules: Sequence[nn.Module]) -> list[nn.Parameter]:
    parameters = []
    seen = set()
    for module in modules:
        for parameter in module.parameters():
            identifier = id(parameter)
            if parameter.requires_grad and identifier not in seen:
                parameters.append(parameter)
                seen.add(identifier)
    return parameters


def assert_disjoint_optimizers(optimizers: Mapping[str, torch.optim.Optimizer]) -> None:
    """Raise when a parameter appears in two optimizers or twice in one optimizer."""
    owners: Dict[int, str] = {}
    for optimizer_name, optimizer in optimizers.items():
        local = set()
        for group in optimizer.param_groups:
            for parameter in group["params"]:
                identifier = id(parameter)
                if identifier in local:
                    raise ValueError(
                        f"Parameter appears more than once in optimizer {optimizer_name!r}"
                    )
                local.add(identifier)
                if identifier in owners:
                    raise ValueError(
                        "Parameter is shared by optimizers "
                        f"{owners[identifier]!r} and {optimizer_name!r}"
                    )
                owners[identifier] = optimizer_name


def build_dehaze_optimizers(
    model: nn.Module,
    depth_net: nn.Module,
    refine_net: nn.Module,
    d_clear: nn.Module,
    d_hazy: nn.Module,
    config: TrainerConfig,
    include_lowlight: bool = False,
) -> Dict[str, torch.optim.Optimizer]:
    """Build disjoint generator, depth, and discriminator Adam optimizers."""
    generator_groups = []
    shared = _unique_trainable([model.N_net.patch_embed, model.N_net.encoder])
    if shared:
        generator_groups.append({"params": shared, "lr": config.shared_lr, "name": "shared"})

    dehaze = _unique_trainable([model.dehaze_head, refine_net])
    if not dehaze:
        raise ValueError("No trainable dehaze/refinement parameters were found")
    generator_groups.append({"params": dehaze, "lr": config.dehaze_lr, "name": "dehaze"})

    if include_lowlight:
        lowlight = _unique_trainable(
            [
                model.N_net.output,
                model.L_net,
                model.R_net,
                model.Gamma_Predictor,
                model.illp,
            ]
        )
        if lowlight:
            generator_groups.append(
                {"params": lowlight, "lr": config.lowlight_lr, "name": "lowlight"}
            )

    optimizers = {
        "generator": torch.optim.Adam(
            generator_groups,
            betas=(0.9, 0.999),
            eps=1e-8,
            weight_decay=config.weight_decay,
        ),
        "depth": torch.optim.Adam(
            _unique_trainable([depth_net]),
            lr=config.depth_lr,
            betas=(0.9, 0.999),
            eps=1e-8,
            weight_decay=config.weight_decay,
        ),
        "discriminator": torch.optim.Adam(
            _unique_trainable([d_clear, d_hazy]),
            lr=config.discriminator_lr,
            betas=(0.9, 0.999),
            eps=1e-8,
            weight_decay=config.weight_decay,
        ),
    }
    assert_disjoint_optimizers(optimizers)
    return optimizers


def _masked(image: torch.Tensor, mask: Optional[torch.Tensor]) -> torch.Tensor:
    return image if mask is None else image * (1.0 - mask.detach())


def _finite_diagnostics(
    phase: str,
    values: Mapping[str, torch.Tensor],
    batch: Mapping[str, Any],
    physical: Optional[Mapping[str, torch.Tensor]] = None,
) -> None:
    if all(torch.isfinite(value).all() for value in values.values()) and all(
        torch.isfinite(value).all() for value in (physical or {}).values()
    ):
        return
    paths = {
        key: batch[key]
        for key in ("clean_path", "hazy_path", "clean_ref_path", "hazy_ref_path")
        if key in batch
    }
    def ranges(items: Mapping[str, torch.Tensor]) -> Dict[str, tuple[float, float]]:
        return {
            key: (value.detach().amin().item(), value.detach().amax().item())
            for key, value in items.items()
        }

    raise FloatingPointError(
        f"Non-finite {phase}; input_paths={paths or 'unavailable'}; "
        f"loss_ranges={ranges(values)}; physical_ranges={ranges(physical or {})}"
    )


class DehazeTrainer:
    """Execute exactly three updates per unpaired dehazing iteration."""

    def __init__(
        self,
        model: nn.Module,
        depth_net: nn.Module,
        refine_net: nn.Module,
        d_clear: nn.Module,
        d_hazy: nn.Module,
        optimizers: Mapping[str, torch.optim.Optimizer],
        config: TrainerConfig,
        contrast_loss: Optional[nn.Module] = None,
        semantic_encoder: Optional[Any] = None,
    ) -> None:
        required = {"generator", "depth", "discriminator"}
        missing = sorted(required - set(optimizers))
        if missing:
            raise ValueError(f"Missing optimizers: {missing}")
        assert_disjoint_optimizers(optimizers)
        self.model = model
        self.depth_net = depth_net
        self.refine_net = refine_net
        self.d_clear = d_clear
        self.d_hazy = d_hazy
        self.optimizers = dict(optimizers)
        self.config = config
        self.contrast_loss = contrast_loss
        self.semantic_encoder = semantic_encoder
        self.renderer = PhysicalHazeRenderer(
            config.transmission_min, config.transmission_max
        )
        device = next(model.parameters()).device
        self.precision = TrainingPrecision(
            device=device,
            enabled=config.amp,
            dtype=config.amp_dtype,
        )
        self.iteration = 0

    def set_physics(self, config: TrainerConfig, atmosphere_mode: Optional[str] = None) -> None:
        """Switch physical ranges for sequential indoor/real joint batches."""
        self.config = config
        self.renderer = PhysicalHazeRenderer(
            config.transmission_min, config.transmission_max
        )
        self.model.set_dehaze_physics(
            config.beta_min,
            config.beta_max,
            config.transmission_min,
            config.transmission_max,
            atmosphere_mode=atmosphere_mode,
        )
        if hasattr(self.depth_net, "set_depth_range"):
            self.depth_net.set_depth_range(config.depth_min, config.depth_max)

    def _random_beta(self, clean: torch.Tensor) -> torch.Tensor:
        shape = (clean.shape[0], 1, 1, 1)
        return self.config.beta_min + torch.rand(
            shape, device=clean.device, dtype=clean.dtype
        ) * (self.config.beta_max - self.config.beta_min)

    def _cycles(self, batch: Mapping[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        clean = batch["clean"]
        hazy = batch["hazy"]

        hazy_aux = self.model(hazy, task="dehaze", return_aux=True)
        clean_prediction = hazy_aux["clean"]
        clean_prediction_depth = self.depth_net(clean_prediction)
        coarse_hazy_cycle, haze_cycle_t = self.renderer(
            clean_prediction,
            clean_prediction_depth,
            hazy_aux["beta"],
            hazy_aux["atmosphere"],
        )
        hazy_cycle = self.refine_net(
            clean_prediction, coarse_hazy_cycle, haze_cycle_t
        )

        clean_depth = self.depth_net(clean)
        sampled_beta = self._random_beta(clean)
        clean_atmosphere = self.model.atmosphere_estimator(clean)
        coarse_haze_prediction, haze_prediction_t = self.renderer(
            clean, clean_depth, sampled_beta, clean_atmosphere
        )
        haze_prediction = self.refine_net(
            clean, coarse_haze_prediction, haze_prediction_t
        )
        clean_cycle_aux = self.model(
            haze_prediction, task="dehaze", return_aux=True
        )

        return {
            "clean_prediction": clean_prediction,
            "clean_prediction_depth": clean_prediction_depth,
            "hazy_cycle": hazy_cycle,
            "haze_prediction": haze_prediction,
            "clean_cycle": clean_cycle_aux["clean"],
            "sampled_beta": sampled_beta,
            "predicted_sample_beta": clean_cycle_aux["beta"],
            "hazy_transmission": hazy_aux["transmission"],
            "hazy_beta": hazy_aux["beta"],
            "hazy_depth": hazy_aux["depth_from_haze"],
            "hazy_atmosphere": hazy_aux["atmosphere"],
        }

    def _discriminator_phase(self, batch: Mapping[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        optimizer = self.optimizers["discriminator"]
        set_requires_grad(self.d_clear, True)
        set_requires_grad(self.d_hazy, True)
        optimizer.zero_grad(set_to_none=True)
        with self.precision.autocast(), torch.no_grad():
            outputs = self._cycles(batch)

        clean_mask = batch.get("clean_mask")
        hazy_mask = batch.get("hazy_mask")
        real_clear = _masked(batch["clean"], clean_mask)
        fake_clear = _masked(outputs["clean_prediction"], hazy_mask)
        with self.precision.autocast():
            d_clear_loss = lsgan_discriminator_loss(
                self.d_clear(real_clear), self.d_clear(fake_clear.detach())
            )
            d_hazy_loss = lsgan_discriminator_loss(
                self.d_hazy(batch["hazy"]),
                self.d_hazy(outputs["haze_prediction"].detach()),
            )
            total = d_clear_loss + d_hazy_loss
        _finite_diagnostics(
            "discriminator phase",
            {"D_clear": d_clear_loss, "D_hazy": d_hazy_loss, "total": total},
            batch,
            {
                "t": outputs["hazy_transmission"],
                "beta": outputs["hazy_beta"],
                "depth": outputs["hazy_depth"],
            },
        )
        self.precision.backward_step(total, optimizer)
        return {"D_clear": d_clear_loss.detach(), "D_hazy": d_hazy_loss.detach()}

    def _generator_phase(
        self, batch: Mapping[str, torch.Tensor]
    ) -> tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        optimizer = self.optimizers["generator"]
        optimizer.zero_grad(set_to_none=True)
        with temporarily_frozen(self.depth_net, self.d_clear, self.d_hazy):
            with self.precision.autocast():
                outputs = self._cycles(batch)
            total_cycle, clean_cycle, hazy_cycle = cycle_loss(
                batch["clean"],
                outputs["clean_cycle"],
                batch["hazy"],
                outputs["hazy_cycle"],
                clean_mask=batch.get("clean_mask"),
            )
            with self.precision.autocast():
                gan = lsgan_generator_loss(
                    self.d_clear(
                        _masked(outputs["clean_prediction"], batch.get("hazy_mask"))
                    ),
                    self.d_hazy(outputs["haze_prediction"]),
                )
            scatter = scattering_loss(
                outputs["predicted_sample_beta"],
                outputs["sampled_beta"],
                self.config.beta_min,
                self.config.beta_max,
                self.config.eps,
            )
            contrast = total_cycle.new_zeros(())
            if self.contrast_loss is not None:
                with self.precision.autocast():
                    contrast = self.contrast_loss(
                        outputs["clean_prediction"],
                        outputs["haze_prediction"],
                        batch["clean_ref"],
                        batch["hazy_ref"],
                    )
            semantic = total_cycle.new_zeros(())
            if self.semantic_encoder is not None:
                with self.precision.autocast():
                    semantic = semantic_consistency_loss(
                        self.semantic_encoder,
                        batch["hazy"],
                        outputs["clean_prediction"],
                    )
            total, weighted = generator_total_loss(
                total_cycle,
                gan,
                scatter,
                contrast,
                semantic,
                self.config.loss_weights,
            )
            _finite_diagnostics(
                "generator phase",
                {
                    "cycle_clean": clean_cycle,
                    "cycle_hazy": hazy_cycle,
                    "gan_G": gan,
                    "scattering": scatter,
                    "contrast": contrast,
                    "semantic": semantic,
                    "total": total,
                },
                batch,
                {
                    "t": outputs["hazy_transmission"],
                    "beta": outputs["hazy_beta"],
                    "depth": outputs["hazy_depth"],
                },
            )
            parameters = [
                parameter
                for group in optimizer.param_groups
                for parameter in group["params"]
                if parameter.requires_grad
            ]
            self.precision.backward_step(
                total,
                optimizer,
                parameters=parameters,
                max_grad_norm=self.config.grad_clip,
            )

        metrics = {
            "cycle_clean": clean_cycle.detach(),
            "cycle_hazy": hazy_cycle.detach(),
            "gan_G": gan.detach(),
            "scattering": scatter.detach(),
            "contrast": contrast.detach(),
            "semantic": semantic.detach(),
            "generator_total": total.detach(),
        }
        metrics.update({f"weighted_{key}": value.detach() for key, value in weighted.items()})
        return outputs, metrics

    def _depth_phase(
        self, batch: Mapping[str, torch.Tensor], outputs: Mapping[str, torch.Tensor]
    ) -> torch.Tensor:
        optimizer = self.optimizers["depth"]
        set_requires_grad(self.depth_net, True)
        optimizer.zero_grad(set_to_none=True)

        if self.config.use_dcp_pseudo_depth:
            with torch.no_grad():
                dcp_t = self.model.atmosphere_estimator.estimate_transmission(
                    batch["hazy"], outputs["hazy_atmosphere"]
                ).clamp(self.config.transmission_min, self.config.transmission_max)
                pseudo_depth = depth_from_transmission(
                    dcp_t, outputs["hazy_beta"].detach(), self.config.eps
                )
        else:
            pseudo_depth = outputs["hazy_depth"].detach()

        with self.precision.autocast():
            predicted_depth = self.depth_net(outputs["clean_prediction"].detach())
        loss = depth_pseudo_loss(
            predicted_depth,
            pseudo_depth,
            self.config.depth_min,
            self.config.depth_max,
            self.config.eps,
        )
        _finite_diagnostics(
            "depth phase",
            {"depth": loss},
            batch,
            {
                "t": outputs["hazy_transmission"],
                "beta": outputs["hazy_beta"],
                "pseudo_depth": pseudo_depth,
                "predicted_depth": predicted_depth,
            },
        )
        self.precision.backward_step(loss, optimizer)
        return loss.detach()

    @staticmethod
    def _stats(prefix: str, tensor: torch.Tensor) -> Dict[str, float]:
        detached = tensor.detach()
        return {
            f"{prefix}_mean": detached.mean().item(),
            f"{prefix}_std": detached.std(unbiased=False).item(),
        }

    def train_step(
        self,
        batch: Mapping[str, torch.Tensor],
        progress_callback: Optional[Callable[[str], None]] = None,
    ) -> Dict[str, float]:
        """Run discriminator, generator, then one DepthNet update."""
        required = {"clean", "hazy", "clean_ref", "hazy_ref"}
        missing = sorted(required - set(batch))
        if missing:
            raise KeyError(f"Dehaze batch is missing required keys: {missing}")
        self.model.train()
        self.model.sem_net.eval()
        self.depth_net.train()
        self.refine_net.train()
        self.d_clear.train()
        self.d_hazy.train()

        if progress_callback is not None:
            progress_callback("phase 1/3: updating discriminators")
        discriminator_metrics = self._discriminator_phase(batch)
        if progress_callback is not None:
            progress_callback("phase 2/3: updating generator")
        outputs, generator_metrics = self._generator_phase(batch)
        if progress_callback is not None:
            progress_callback("phase 3/3: updating depth network")
        depth_loss = self._depth_phase(batch, outputs)
        self.precision.update()
        self.iteration += 1

        tensors: Dict[str, torch.Tensor] = {
            **discriminator_metrics,
            **generator_metrics,
            "depth": depth_loss,
        }
        metrics = {key: value.item() for key, value in tensors.items()}
        t = outputs["hazy_transmission"]
        beta = outputs["hazy_beta"]
        depth = outputs["hazy_depth"]
        metrics.update(self._stats("t", t))
        metrics["t_low_ratio"] = (t <= self.config.transmission_min + 0.01).float().mean().item()
        metrics["t_high_ratio"] = (t >= self.config.transmission_max - 0.01).float().mean().item()
        metrics.update(self._stats("beta", beta))
        metrics.update(self._stats("depth", depth))
        for optimizer_name, optimizer in self.optimizers.items():
            for index, group in enumerate(optimizer.param_groups):
                group_name = group.get("name", str(index))
                metrics[f"lr_{optimizer_name}_{group_name}"] = float(group["lr"])
        return metrics


def _model_state_without_clip(model: nn.Module) -> Dict[str, torch.Tensor]:
    return {
        key: value
        for key, value in model.state_dict().items()
        if not key.startswith("sem_net.")
    }


def _print_key_summary(label: str, keys: Sequence[str]) -> None:
    """Print checkpoint incompatibilities without flooding the terminal."""
    if not keys:
        print(f"[checkpoint] {label}: 0", flush=True)
        return
    groups: Dict[str, int] = {}
    for key in keys:
        group = key.split(".", 1)[0]
        groups[group] = groups.get(group, 0) + 1
    group_summary = ", ".join(
        f"{name}={count}" for name, count in sorted(groups.items())
    )
    examples = ", ".join(keys[:3])
    suffix = ", ..." if len(keys) > 3 else ""
    print(
        f"[checkpoint] {label}: {len(keys)} ({group_summary}); "
        f"examples: {examples}{suffix}",
        flush=True,
    )


def save_training_checkpoint(
    path: str,
    model: nn.Module,
    depth_net: nn.Module,
    refine_net: nn.Module,
    d_clear: nn.Module,
    d_hazy: nn.Module,
    optimizers: Mapping[str, torch.optim.Optimizer],
    schedulers: Optional[Mapping[str, Any]],
    stage: int,
    iteration: int,
    epoch: int,
    config: Mapping[str, Any],
    precision: Optional[TrainingPrecision] = None,
) -> None:
    """Save complete train state without duplicating frozen CLIP weights."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": _model_state_without_clip(model),
            "depth_net": depth_net.state_dict(),
            "refine_net": refine_net.state_dict(),
            "D_clear": d_clear.state_dict(),
            "D_hazy": d_hazy.state_dict(),
            "optimizers": {name: item.state_dict() for name, item in optimizers.items()},
            "schedulers": {
                name: item.state_dict() for name, item in (schedulers or {}).items()
            },
            "stage": stage,
            "iteration": iteration,
            "epoch": epoch,
            "config": dict(config),
            "amp_scaler": precision.state_dict() if precision is not None else {},
        },
        target,
    )


def load_training_checkpoint(
    path: str,
    model: nn.Module,
    depth_net: Optional[nn.Module] = None,
    refine_net: Optional[nn.Module] = None,
    d_clear: Optional[nn.Module] = None,
    d_hazy: Optional[nn.Module] = None,
    optimizers: Optional[Mapping[str, torch.optim.Optimizer]] = None,
    schedulers: Optional[Mapping[str, Any]] = None,
    map_location: Any = "cpu",
    precision: Optional[TrainingPrecision] = None,
) -> Dict[str, Any]:
    """Load new full checkpoints or legacy SPS state dicts with diagnostics."""
    checkpoint = torch.load(path, map_location=map_location)
    model_state = checkpoint.get("model", checkpoint) if isinstance(checkpoint, dict) else checkpoint
    incompatible = model.load_state_dict(model_state, strict=False)
    _print_key_summary("Missing model keys", list(incompatible.missing_keys))
    _print_key_summary("Unexpected model keys", list(incompatible.unexpected_keys))

    if isinstance(checkpoint, dict) and "model" in checkpoint:
        modules = {
            "depth_net": depth_net,
            "refine_net": refine_net,
            "D_clear": d_clear,
            "D_hazy": d_hazy,
        }
        for key, module in modules.items():
            if module is not None and key in checkpoint:
                module.load_state_dict(checkpoint[key])
        for name, optimizer in (optimizers or {}).items():
            if name in checkpoint.get("optimizers", {}):
                optimizer.load_state_dict(checkpoint["optimizers"][name])
        for name, scheduler in (schedulers or {}).items():
            if name in checkpoint.get("schedulers", {}):
                scheduler.load_state_dict(checkpoint["schedulers"][name])
        if precision is not None and checkpoint.get("amp_scaler"):
            precision.load_state_dict(checkpoint["amp_scaler"])
    return checkpoint if isinstance(checkpoint, dict) else {"model": checkpoint}
