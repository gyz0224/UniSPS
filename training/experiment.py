"""Canonical experiment identities and artifact paths.

All Stage 1--4 training, inference, and metrics entrypoints resolve paths
through this module so changing the low-light initializer cannot overwrite a
different experiment.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Optional


EXPERIMENT_INITIAL_WEIGHTS = {
    "lolv1": Path("weights/lolv1.pth"),
    "lolv2real": Path("weights/lolv2real.pth"),
    "sice": Path("weights/sice.pth"),
}
EXPERIMENT_CHOICES = tuple(EXPERIMENT_INITIAL_WEIGHTS)

STAGE_DIRECTORY_NAMES = {
    1: "stage1_warmup",
    2: "stage2_its",
    3: "stage3_real",
    4: "stage4_joint",
}


@dataclass(frozen=True)
class LowlightDatasetPreset:
    """Native low-light data and evaluation convention for one experiment."""

    dataset_key: str
    dataset_name: str
    train_input: Path
    test_input: Path
    reference_dir: Path
    pairing: str


LOWLIGHT_DATASETS = {
    "lolv1": LowlightDatasetPreset(
        dataset_key="lolv1_test",
        dataset_name="LOL-v1 Test",
        train_input=Path("dataset/LOLv1/Train/input"),
        test_input=Path("dataset/LOLv1/Test/input"),
        reference_dir=Path("dataset/LOLv1/Test/target"),
        pairing="same-name",
    ),
    "lolv2real": LowlightDatasetPreset(
        dataset_key="lolv2real_test",
        dataset_name="LOL-v2 Real-captured Test",
        train_input=Path("dataset/LOLv2/Real_captured/Train/Low"),
        test_input=Path("dataset/LOLv2/Real_captured/Test/Low"),
        reference_dir=Path("dataset/LOLv2/Real_captured/Test/Normal"),
        pairing="same-name",
    ),
    "sice": LowlightDatasetPreset(
        dataset_key="sice_test",
        dataset_name="SICE Test",
        train_input=Path("dataset/SICE/Train"),
        test_input=Path("dataset/SICE/Test/image"),
        reference_dir=Path("dataset/SICE/Test/label"),
        pairing="sice",
    ),
}


@dataclass(frozen=True)
class DehazePreset:
    """One reproducible dehaze inference/metric combination."""

    train_stage: int
    dataset_key: str
    dataset_name: str
    config: Path
    input_dir: Path
    reference_dir: Optional[Path]
    pairing: str
    lpips: bool = False
    tile_size: int = 0
    tile_overlap: int = 128


DEHAZE_PRESETS = {
    "stage2": DehazePreset(
        train_stage=2,
        dataset_key="sots_indoor",
        dataset_name="SOTS Indoor",
        config=Path("configs/dehaze_its.yaml"),
        input_dir=Path("dataset/eval/SOTS/indoor/hazy"),
        reference_dir=Path("dataset/eval/SOTS/indoor/gt"),
        pairing="sots",
        lpips=True,
    ),
    "stage2-outdoor": DehazePreset(
        train_stage=2,
        dataset_key="sots_outdoor",
        dataset_name="SOTS Outdoor (Stage 2)",
        config=Path("configs/dehaze_real.yaml"),
        input_dir=Path("dataset/eval/SOTS/outdoor/hazy"),
        reference_dir=Path("dataset/eval/SOTS/outdoor/gt"),
        pairing="sots-outdoor",
        lpips=True,
    ),
    "stage2-hsts-synthetic": DehazePreset(
        train_stage=2,
        dataset_key="hsts_synthetic",
        dataset_name="HSTS Synthetic (Stage 2)",
        config=Path("configs/dehaze_real.yaml"),
        input_dir=Path("dataset/eval/HSTS/synthetic/hazy"),
        reference_dir=Path("dataset/eval/HSTS/synthetic/gt"),
        pairing="same-stem",
        lpips=True,
    ),
    "stage2-hsts-real": DehazePreset(
        train_stage=2,
        dataset_key="hsts_real",
        dataset_name="HSTS Real-world (Stage 2)",
        config=Path("configs/dehaze_real.yaml"),
        input_dir=Path("dataset/eval/HSTS/real-world"),
        reference_dir=None,
        pairing="same-stem",
        tile_size=1024,
    ),
    "stage2-ihaze": DehazePreset(
        train_stage=2,
        dataset_key="ihaze",
        dataset_name="I-HAZE (Stage 2)",
        config=Path("configs/dehaze_its.yaml"),
        input_dir=Path("dataset/eval/I-HAZE/hazy"),
        reference_dir=Path("dataset/eval/I-HAZE/gt"),
        pairing="ihaze",
        lpips=True,
        tile_size=1024,
    ),
    "stage3": DehazePreset(
        train_stage=3,
        dataset_key="sots_indoor",
        dataset_name="SOTS Indoor (Stage 3 cross-evaluation)",
        config=Path("configs/dehaze_real.yaml"),
        input_dir=Path("dataset/eval/SOTS/indoor/hazy"),
        reference_dir=Path("dataset/eval/SOTS/indoor/gt"),
        pairing="sots",
        lpips=True,
    ),
    "stage3-outdoor": DehazePreset(
        train_stage=3,
        dataset_key="sots_outdoor",
        dataset_name="SOTS Outdoor (Stage 3 cross-evaluation)",
        config=Path("configs/dehaze_real.yaml"),
        input_dir=Path("dataset/eval/SOTS/outdoor/hazy"),
        reference_dir=Path("dataset/eval/SOTS/outdoor/gt"),
        pairing="sots-outdoor",
        lpips=True,
    ),
    "stage3-hsts-synthetic": DehazePreset(
        train_stage=3,
        dataset_key="hsts_synthetic",
        dataset_name="HSTS Synthetic (Stage 3)",
        config=Path("configs/dehaze_real.yaml"),
        input_dir=Path("dataset/eval/HSTS/synthetic/hazy"),
        reference_dir=Path("dataset/eval/HSTS/synthetic/gt"),
        pairing="same-stem",
        lpips=True,
    ),
    "stage3-hsts-real": DehazePreset(
        train_stage=3,
        dataset_key="hsts_real",
        dataset_name="HSTS Real-world (Stage 3)",
        config=Path("configs/dehaze_real.yaml"),
        input_dir=Path("dataset/eval/HSTS/real-world"),
        reference_dir=None,
        pairing="same-stem",
        tile_size=1024,
    ),
    "stage3-ihaze": DehazePreset(
        train_stage=3,
        dataset_key="ihaze",
        dataset_name="I-HAZE (Stage 3)",
        config=Path("configs/dehaze_real.yaml"),
        input_dir=Path("dataset/eval/I-HAZE/hazy"),
        reference_dir=Path("dataset/eval/I-HAZE/gt"),
        pairing="ihaze",
        lpips=True,
        tile_size=1024,
    ),
    "stage3-real": DehazePreset(
        train_stage=3,
        dataset_key="rtts",
        dataset_name="RTTS",
        config=Path("configs/dehaze_real.yaml"),
        input_dir=Path("dataset/eval/RTTS/JPEGImages"),
        reference_dir=None,
        pairing="same-stem",
    ),
    "stage4": DehazePreset(
        train_stage=4,
        dataset_key="sots_indoor",
        dataset_name="SOTS Indoor",
        config=Path("configs/dehaze_its.yaml"),
        input_dir=Path("dataset/eval/SOTS/indoor/hazy"),
        reference_dir=Path("dataset/eval/SOTS/indoor/gt"),
        pairing="sots",
        lpips=True,
    ),
    "stage4-outdoor": DehazePreset(
        train_stage=4,
        dataset_key="sots_outdoor",
        dataset_name="SOTS Outdoor (Stage 4)",
        config=Path("configs/dehaze_real.yaml"),
        input_dir=Path("dataset/eval/SOTS/outdoor/hazy"),
        reference_dir=Path("dataset/eval/SOTS/outdoor/gt"),
        pairing="sots-outdoor",
        lpips=True,
    ),
    "stage4-hsts-synthetic": DehazePreset(
        train_stage=4,
        dataset_key="hsts_synthetic",
        dataset_name="HSTS Synthetic (Stage 4)",
        config=Path("configs/dehaze_real.yaml"),
        input_dir=Path("dataset/eval/HSTS/synthetic/hazy"),
        reference_dir=Path("dataset/eval/HSTS/synthetic/gt"),
        pairing="same-stem",
        lpips=True,
    ),
    "stage4-hsts-real": DehazePreset(
        train_stage=4,
        dataset_key="hsts_real",
        dataset_name="HSTS Real-world (Stage 4)",
        config=Path("configs/dehaze_real.yaml"),
        input_dir=Path("dataset/eval/HSTS/real-world"),
        reference_dir=None,
        pairing="same-stem",
        tile_size=1024,
    ),
    "stage4-ihaze": DehazePreset(
        train_stage=4,
        dataset_key="ihaze",
        dataset_name="I-HAZE (Stage 4)",
        config=Path("configs/dehaze_real.yaml"),
        input_dir=Path("dataset/eval/I-HAZE/hazy"),
        reference_dir=Path("dataset/eval/I-HAZE/gt"),
        pairing="ihaze",
        lpips=True,
        tile_size=1024,
    ),
    "stage4-real": DehazePreset(
        train_stage=4,
        dataset_key="rtts",
        dataset_name="RTTS",
        config=Path("configs/dehaze_real.yaml"),
        input_dir=Path("dataset/eval/RTTS/JPEGImages"),
        reference_dir=None,
        pairing="same-stem",
    ),
}


@dataclass(frozen=True)
class ExperimentLayout:
    """Resolve every generated path for one initializer experiment."""

    experiment: str
    runs_root: Path = Path("runs")

    def __post_init__(self) -> None:
        if self.experiment not in EXPERIMENT_INITIAL_WEIGHTS:
            choices = ", ".join(EXPERIMENT_CHOICES)
            raise ValueError(
                f"Unknown experiment {self.experiment!r}; choose one of {choices}"
            )

    @property
    def root(self) -> Path:
        return self.runs_root / self.experiment

    @property
    def initial_weight(self) -> Path:
        return EXPERIMENT_INITIAL_WEIGHTS[self.experiment]

    @property
    def lowlight_dataset(self) -> LowlightDatasetPreset:
        return LOWLIGHT_DATASETS[self.experiment]

    def checkpoint_dir(self, stage: int) -> Path:
        try:
            name = STAGE_DIRECTORY_NAMES[stage]
        except KeyError as error:
            raise ValueError(f"Checkpoint stage must be 1, 2, 3, or 4: {stage}") from error
        return self.root / "checkpoints" / name

    def checkpoint(self, stage: int, filename: str = "latest.pth") -> Path:
        return self.checkpoint_dir(stage) / filename

    def dehaze_result(self, preset_name: str) -> Path:
        try:
            preset = DEHAZE_PRESETS[preset_name]
        except KeyError as error:
            raise ValueError(f"Unknown dehaze preset: {preset_name}") from error
        return (
            self.root
            / "results"
            / "dehaze"
            / f"stage{preset.train_stage}"
            / preset.dataset_key
        )

    def lowlight_result(self, stage: str) -> Path:
        if stage not in {"pretrained", "stage4"}:
            raise ValueError("Low-light result stage must be pretrained or stage4")
        return (
            self.root
            / "results"
            / "lowlight"
            / stage
            / self.lowlight_dataset.dataset_key
        )

    def dehaze_metrics(self, preset_name: str) -> Path:
        if preset_name not in DEHAZE_PRESETS:
            raise ValueError(f"Unknown dehaze preset: {preset_name}")
        return self.root / "metrics" / "dehaze" / f"{preset_name}.json"

    def lowlight_metrics(self, stage: str) -> Path:
        if stage not in {"pretrained", "stage4"}:
            raise ValueError("Low-light metric stage must be pretrained or stage4")
        return self.root / "metrics" / "lowlight" / f"{stage}.json"


def ensure_new_training_output(output_dir: Path, resume: Optional[str]) -> None:
    """Refuse an accidental fresh run over an existing checkpoint directory."""
    if resume or not output_dir.is_dir():
        return
    checkpoint_files = sorted(output_dir.glob("*.pth"))
    if checkpoint_files:
        examples = ", ".join(path.name for path in checkpoint_files[:3])
        raise FileExistsError(
            f"Checkpoint directory already contains {len(checkpoint_files)} .pth "
            f"files: {output_dir} ({examples}). Use --resume for the same run or "
            "choose another --experiment."
        )


__all__ = [
    "DEHAZE_PRESETS",
    "EXPERIMENT_CHOICES",
    "EXPERIMENT_INITIAL_WEIGHTS",
    "LOWLIGHT_DATASETS",
    "DehazePreset",
    "ExperimentLayout",
    "LowlightDatasetPreset",
    "ensure_new_training_output",
]
