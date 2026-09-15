#!/usr/bin/env python3
"""Evaluate SPS low-light enhancement and export intermediate feature images."""

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Sequence

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.utils import save_image

from datasets.loaders import get_eval_set
from net.model import net
from training.experiment import (
    EXPERIMENT_CHOICES,
    ExperimentLayout,
)
from training.lowlight_sampling import (
    gamma_correction,
    generate_mask_pair,
    generate_subimages,
)
from training.lowlight_trainer import load_lowlight_checkpoint, resolve_lowlight_device


OUTPUT_GROUPS = (
    "L",
    "R",
    "I",
    "X",
    "im1",
    "im2",
    "X_feats",
    "illu_fea",
    "ill_map",
    "noisy_R",
    "sem_feat0",
    "sem_feat1",
)


@dataclass(frozen=True)
class LowlightInferenceSpec:
    experiment: str
    stage: str
    dataset_key: str
    dataset_name: str
    data_test: Path
    model: Path
    output: Path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="PairLIE")
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
        help="evaluate the initial low-light weight or final joint checkpoint",
    )
    parser.add_argument("--testBatchSize", type=int, default=1)
    parser.add_argument("--gpu_mode", type=bool, default=True)
    parser.add_argument("--device", help="explicit torch device, e.g. cuda:0 or cpu")
    parser.add_argument("--threads", type=int, default=0)
    parser.add_argument("--rgb_range", type=int, default=1)
    parser.add_argument(
        "--data_test",
        help="override the experiment's native low-light test input",
    )
    parser.add_argument("--model", help="override the preset checkpoint")
    parser.add_argument(
        "--output_folder",
        "--output-folder",
        dest="output_folder",
        help="override the canonical output directory",
    )
    return parser


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    return build_parser().parse_args(argv)


def resolve_inference(args: argparse.Namespace) -> LowlightInferenceSpec:
    layout = ExperimentLayout(args.experiment)
    dataset = layout.lowlight_dataset
    model = (
        Path(args.model)
        if args.model
        else (
            layout.initial_weight
            if args.stage == "pretrained"
            else layout.checkpoint(4)
        )
    )
    output = (
        Path(args.output_folder)
        if args.output_folder
        else layout.lowlight_result(args.stage)
    )
    return LowlightInferenceSpec(
        experiment=args.experiment,
        stage=args.stage,
        dataset_key=dataset.dataset_key,
        dataset_name=dataset.dataset_name,
        data_test=Path(args.data_test) if args.data_test else dataset.test_input,
        model=model,
        output=output,
    )


def ensure_output_dirs(base_path: Path) -> None:
    base_path.mkdir(parents=True, exist_ok=True)
    for name in OUTPUT_GROUPS:
        (base_path / name).mkdir(parents=True, exist_ok=True)


def save_tensor_image(tensor: torch.Tensor, path: Path) -> None:
    image = torch.clamp(tensor.detach().cpu(), 0, 1)
    transforms.ToPILImage()(image.squeeze(0)).save(path)


def save_feature_grid(
    tensor: Optional[torch.Tensor], path: Path, max_channels: int = 8, nrow: int = 4
) -> None:
    if tensor is None or tensor.dim() != 4:
        return
    tensor = tensor.detach().cpu()
    take = min(max_channels, tensor.shape[1])
    feature = tensor[0, :take]
    minimum = feature.amin(dim=(1, 2), keepdim=True)
    maximum = feature.amax(dim=(1, 2), keepdim=True)
    feature = (feature - minimum) / (maximum - minimum + 1e-8)
    if feature.shape[0] == 1:
        grid = feature
    elif feature.shape[0] == 2:
        grid = torch.cat([feature, feature[:1]], dim=0)
    else:
        grid = feature[:3]
    save_image(grid, path, nrow=nrow)


def prepare_lowlight_input(
    input_image: torch.Tensor, max_dimension: int = 1024
) -> torch.Tensor:
    """Sanitize zero-valued pixels, then apply the evaluation size policy.

    Exact zero channel values carry no usable low-light signal and can create
    black artifacts in the Retinex division.  Replace uint8-equivalent zeros
    with 1/255 before any resize so every evaluation and dataset-generation
    entry point uses the same safe input convention.
    """
    input_image = input_image.masked_fill(input_image == 0, 1.0 / 255.0)
    _, _, height, width = input_image.shape
    new_height = height - (height % 2)
    new_width = width - (width % 2)
    if new_height != height or new_width != width:
        input_image = input_image[:, :, :new_height, :new_width]
    if height > max_dimension or width > max_dimension:
        scale = max_dimension / max(height, width)
        resized_height = max(8, (int(height * scale) // 8) * 8)
        resized_width = max(8, (int(width * scale) // 8) * 8)
        input_image = F.interpolate(
            input_image,
            size=(resized_height, resized_width),
            mode="bilinear",
            align_corners=False,
        )
    return input_image


def evaluate_lowlight_features(
    model: torch.nn.Module, input_image: torch.Tensor
) -> Dict[str, object]:
    """Return final low-light tensors and visualization-only intermediates."""
    input_image = prepare_lowlight_input(input_image)
    semantic_features = model.extract_semantics(input_image)
    mask1, mask2 = generate_mask_pair(input_image)
    image1 = generate_subimages(input_image, mask1)
    image2 = gamma_correction(generate_subimages(input_image, mask2))
    x_img, x_features = model.N_net(input_image)
    illumination_feature, illumination_map = model.illp(x_img)
    illumination = model.L_net(x_img)
    noisy_reflectance = input_image / illumination
    reflectance = model.R_net(
        feature_x=x_features,
        noisy_R=noisy_reflectance.detach(),
        L=illumination,
        fea=illumination_feature,
        pre_R=x_img,
        sem_feats=semantic_features,
    )
    alpha = model.Gamma_Predictor(illumination)
    enhanced = torch.pow(illumination, alpha) * reflectance
    return {
        "input": input_image,
        "L": illumination,
        "R": reflectance,
        "I": enhanced,
        "X": x_img,
        "im1": image1,
        "im2": image2,
        "X_feats": x_features,
        "illu_fea": illumination_feature,
        "ill_map": illumination_map,
        "noisy_R": noisy_reflectance,
        "sem_feats": semantic_features,
    }


def save_feature_outputs(outputs: Dict[str, object], output: Path, name: str) -> None:
    stem, extension = Path(name).stem, Path(name).suffix
    for key in ("L", "R", "I", "X", "ill_map", "noisy_R"):
        save_tensor_image(torch.clamp(outputs[key], 0, 1), output / key / name)
    save_tensor_image(
        torch.clamp(outputs["im1"], 0, 1), output / "im1" / f"{stem}_im1{extension}"
    )
    save_tensor_image(
        torch.clamp(outputs["im2"], 0, 1), output / "im2" / f"{stem}_im2{extension}"
    )
    save_feature_grid(outputs["X_feats"], output / "X_feats" / f"{stem}.png")
    save_feature_grid(outputs["illu_fea"], output / "illu_fea" / f"{stem}.png")
    semantic_features = outputs["sem_feats"]
    if semantic_features and semantic_features[0] is not None:
        save_feature_grid(
            semantic_features[0], output / "sem_feat0" / f"{stem}.png"
        )
    if len(semantic_features) > 1 and semantic_features[1] is not None:
        save_feature_grid(
            semantic_features[1], output / "sem_feat1" / f"{stem}.png"
        )


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    spec = resolve_inference(args)
    device = resolve_lowlight_device(args.device, args.gpu_mode)
    print(
        f"===> Experiment: {spec.experiment} stage={spec.stage} "
        f"checkpoint={spec.model}"
    )
    print(
        f"===> Loading dataset: {spec.dataset_name} input={spec.data_test}"
    )
    test_set = get_eval_set(str(spec.data_test))
    loader = DataLoader(
        test_set, num_workers=args.threads, batch_size=1, shuffle=False
    )
    print("===> Building model")
    model = net().to(device)
    incompatible = load_lowlight_checkpoint(
        str(spec.model), model, map_location=device
    )
    if incompatible.unexpected_keys:
        print(f"Unexpected checkpoint keys: {list(incompatible.unexpected_keys)}")
    print("Pre-trained model is loaded.")
    model.eval()
    output = spec.output
    ensure_output_dirs(output)
    print("\nEvaluation:")
    with torch.no_grad():
        for batch in loader:
            input_image, names = batch[0].to(device), batch[1]
            print(names)
            outputs = evaluate_lowlight_features(model, input_image)
            save_feature_outputs(outputs, output, names[0])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
