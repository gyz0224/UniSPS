"""Strict unpaired clear/hazy dataset used by D4+-style training."""

import os
import random
from pathlib import Path
from typing import Optional, Sequence, Union

import numpy as np
import torch
import torch.utils.data as data
from PIL import Image, ImageOps


DEHAZE_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def read_dehaze_paths(
    source: Union[str, os.PathLike, Sequence[Union[str, os.PathLike]]],
    domain_name: str,
) -> list[Path]:
    """Resolve a directory, image, flist, or explicit sequence to image paths."""
    if isinstance(source, (list, tuple)):
        raw_paths = [Path(item).expanduser() for item in source]
    else:
        source_path = Path(source).expanduser()
        if source_path.is_dir():
            raw_paths = [
                path
                for path in source_path.rglob("*")
                if path.is_file()
                and path.suffix.lower() in DEHAZE_IMAGE_EXTENSIONS
            ]
        elif (
            source_path.is_file()
            and source_path.suffix.lower() in DEHAZE_IMAGE_EXTENSIONS
        ):
            raw_paths = [source_path]
        elif source_path.is_file():
            raw_paths = []
            for line_number, line in enumerate(
                source_path.read_text(encoding="utf-8").splitlines(), start=1
            ):
                entry = line.strip()
                if not entry or entry.startswith("#"):
                    continue
                entry_path = Path(entry).expanduser()
                if not entry_path.is_absolute():
                    entry_path = source_path.parent / entry_path
                if entry_path.suffix.lower() not in DEHAZE_IMAGE_EXTENSIONS:
                    raise ValueError(
                        f"Unsupported image extension in {domain_name} flist "
                        f"{source_path}:{line_number}: {entry_path}"
                    )
                raw_paths.append(entry_path)
        else:
            raise FileNotFoundError(
                f"{domain_name} data source does not exist: {source_path}"
            )

    paths = sorted(path.resolve() for path in raw_paths)
    if not paths:
        raise ValueError(f"No supported images found for {domain_name}")
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        preview = missing[:3]
        raise FileNotFoundError(
            f"{domain_name} contains missing files: {preview}"
            + (f" (and {len(missing) - 3} more)" if len(missing) > 3 else "")
        )
    return paths


def find_matching_mask(image_path: Path, directory: Path) -> Path:
    """Resolve the same-name or same-stem mask used by the training loader."""
    exact = directory / image_path.name
    if exact.is_file():
        return exact
    matches = sorted(
        path
        for path in directory.rglob(f"{image_path.stem}.*")
        if path.suffix.lower() in DEHAZE_IMAGE_EXTENSIONS
    )
    if not matches:
        raise FileNotFoundError(
            f"No mask matching image {image_path.name} in {directory}"
        )
    return matches[0]


def _unique_stem_index(paths: Sequence[Path], label: str) -> dict[str, Path]:
    index: dict[str, Path] = {}
    duplicates: list[str] = []
    for path in paths:
        if path.stem in index:
            duplicates.append(path.stem)
        else:
            index[path.stem] = path
    if duplicates:
        raise ValueError(
            f"{label} contains duplicate basenames: {sorted(set(duplicates))[:3]}"
        )
    return index


def validate_image_mask_pairs(
    image_source: Union[str, os.PathLike, Sequence[Union[str, os.PathLike]]],
    mask_source: Union[str, os.PathLike, Sequence[Union[str, os.PathLike]]],
    domain_name: str,
) -> int:
    """Validate one real-image domain against masks by basename and dimensions."""
    image_paths = read_dehaze_paths(image_source, domain_name)
    mask_paths = read_dehaze_paths(mask_source, f"{domain_name} masks")
    images = _unique_stem_index(image_paths, domain_name)
    masks = _unique_stem_index(mask_paths, f"{domain_name} masks")
    missing = sorted(set(images) - set(masks))
    extra = sorted(set(masks) - set(images))
    if missing or extra:
        raise ValueError(
            f"{domain_name} image/mask basenames differ: "
            f"missing masks={missing[:3]}, extra masks={extra[:3]}"
        )

    for stem, image_path in images.items():
        mask_path = masks[stem]
        try:
            with Image.open(image_path) as image, Image.open(mask_path) as mask:
                image_size = image.size
                mask_size = mask.size
        except (OSError, ValueError) as exc:
            raise OSError(
                f"Failed to read {domain_name} image/mask pair "
                f"{image_path} / {mask_path}: {exc}"
            ) from exc
        if image_size != mask_size:
            raise ValueError(
                f"Mask/image size mismatch for {image_path.name}: "
                f"{mask_size} vs {image_size}"
            )
    return len(images)


def validate_real_dataset_layout(
    clean_source: Union[str, os.PathLike, Sequence[Union[str, os.PathLike]]],
    hazy_source: Union[str, os.PathLike, Sequence[Union[str, os.PathLike]]],
    clean_mask_source: Union[
        str, os.PathLike, Sequence[Union[str, os.PathLike]]
    ],
    hazy_mask_source: Union[
        str, os.PathLike, Sequence[Union[str, os.PathLike]]
    ],
) -> dict[str, int]:
    """Validate both domains required by D4+ real-outdoor training."""
    return {
        "clean": validate_image_mask_pairs(
            clean_source, clean_mask_source, "clean"
        ),
        "hazy": validate_image_mask_pairs(hazy_source, hazy_mask_source, "hazy"),
    }


class UnpairedDehazeDataset(data.Dataset):
    """Independently sample clear/hazy images and contrastive references.

    Default items contain exactly ``clean``, ``hazy``, ``clean_ref``,
    ``hazy_ref``, ``clean_mask``, ``hazy_mask``, and ``is_real``.
    """

    def __init__(
        self,
        clean_source: Union[str, os.PathLike, Sequence[Union[str, os.PathLike]]],
        hazy_source: Union[str, os.PathLike, Sequence[Union[str, os.PathLike]]],
        crop_size: int = 256,
        clean_mask_dir: Optional[Union[str, os.PathLike]] = None,
        hazy_mask_dir: Optional[Union[str, os.PathLike]] = None,
        is_real: bool = False,
        iteration_length: Optional[int] = None,
        test_gt_source: Optional[
            Union[str, os.PathLike, Sequence[Union[str, os.PathLike]]]
        ] = None,
        augment: bool = True,
        seed: Optional[int] = None,
        return_paths: bool = False,
    ) -> None:
        super().__init__()
        if crop_size <= 0:
            raise ValueError("crop_size must be positive")
        if iteration_length is not None and iteration_length <= 0:
            raise ValueError("iteration_length must be positive when provided")
        self.clean_paths = read_dehaze_paths(clean_source, "clean")
        self.hazy_paths = read_dehaze_paths(hazy_source, "hazy")
        self.crop_size = int(crop_size)
        self.is_real = bool(is_real)
        self.iteration_length = iteration_length
        self.augment = bool(augment)
        self.return_paths = bool(return_paths)
        self._rng = random.Random(seed) if seed is not None else random
        self.clean_mask_dir = (
            Path(clean_mask_dir).expanduser().resolve() if clean_mask_dir else None
        )
        self.hazy_mask_dir = (
            Path(hazy_mask_dir).expanduser().resolve() if hazy_mask_dir else None
        )
        if self.is_real:
            for label, directory in (
                ("clean_mask_dir", self.clean_mask_dir),
                ("hazy_mask_dir", self.hazy_mask_dir),
            ):
                if directory is None or not directory.is_dir():
                    raise FileNotFoundError(
                        f"Real-outdoor data requires an existing {label}: {directory}"
                    )

        if test_gt_source is not None:
            test_gt = set(read_dehaze_paths(test_gt_source, "test GT"))
            overlap = sorted(test_gt.intersection(self.clean_paths + self.hazy_paths))
            if overlap:
                raise ValueError(
                    "Test ground truth must not enter dehaze training pools; overlap: "
                    f"{[str(path) for path in overlap[:3]]}"
                )

    def __len__(self) -> int:
        if self.iteration_length is not None:
            return self.iteration_length
        return max(len(self.clean_paths), len(self.hazy_paths))

    def _sample_index(self, paths: Sequence[Path]) -> int:
        return self._rng.randrange(len(paths))

    @staticmethod
    def _load_rgb(path: Path) -> Image.Image:
        try:
            with Image.open(path) as image:
                return image.convert("RGB")
        except (OSError, ValueError) as exc:
            raise OSError(f"Failed to read RGB image {path}: {exc}") from exc

    def _load_mask(
        self, image_path: Path, directory: Optional[Path], size
    ) -> Image.Image:
        if directory is None:
            return Image.new("L", size, color=0)
        mask_path = find_matching_mask(image_path, directory)
        try:
            with Image.open(mask_path) as mask:
                loaded = mask.convert("L")
        except (OSError, ValueError) as exc:
            raise OSError(f"Failed to read mask {mask_path}: {exc}") from exc
        if loaded.size != size:
            raise ValueError(
                f"Mask/image size mismatch for {image_path.name}: {loaded.size} vs {size}"
            )
        return loaded

    def _joint_transform(
        self, image: Image.Image, mask: Optional[Image.Image] = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        mask = mask if mask is not None else Image.new("L", image.size, color=0)
        width, height = image.size
        if min(width, height) < self.crop_size:
            scale = self.crop_size / min(width, height)
            resized = (
                max(self.crop_size, round(width * scale)),
                max(self.crop_size, round(height * scale)),
            )
            image = image.resize(resized, Image.Resampling.BICUBIC)
            mask = mask.resize(resized, Image.Resampling.NEAREST)
            width, height = resized

        left = self._rng.randint(0, width - self.crop_size)
        top = self._rng.randint(0, height - self.crop_size)
        box = (left, top, left + self.crop_size, top + self.crop_size)
        image = image.crop(box)
        mask = mask.crop(box)
        if self.augment and self._rng.random() < 0.5:
            image = ImageOps.mirror(image)
            mask = ImageOps.mirror(mask)

        image_array = np.array(image, dtype=np.float32, copy=True) / 255.0
        mask_array = np.array(mask, dtype=np.float32, copy=True) / 255.0
        image_tensor = torch.from_numpy(image_array).permute(2, 0, 1).contiguous()
        mask_tensor = (torch.from_numpy(mask_array).unsqueeze(0) >= 0.5).float()
        return image_tensor, mask_tensor

    def __getitem__(self, index):
        del index
        # Match D4+: all four unpaired-domain samples are drawn independently.
        clean_index = self._sample_index(self.clean_paths)
        hazy_index = self._sample_index(self.hazy_paths)
        clean_ref_index = self._sample_index(self.clean_paths)
        hazy_ref_index = self._sample_index(self.hazy_paths)
        clean_path = self.clean_paths[clean_index]
        hazy_path = self.hazy_paths[hazy_index]
        clean_ref_path = self.clean_paths[clean_ref_index]
        hazy_ref_path = self.hazy_paths[hazy_ref_index]

        clean_image = self._load_rgb(clean_path)
        hazy_image = self._load_rgb(hazy_path)
        clean_mask = self._load_mask(
            clean_path, self.clean_mask_dir, clean_image.size
        )
        hazy_mask = self._load_mask(hazy_path, self.hazy_mask_dir, hazy_image.size)
        clean, clean_mask_tensor = self._joint_transform(clean_image, clean_mask)
        hazy, hazy_mask_tensor = self._joint_transform(hazy_image, hazy_mask)
        clean_ref, _ = self._joint_transform(self._load_rgb(clean_ref_path))
        hazy_ref, _ = self._joint_transform(self._load_rgb(hazy_ref_path))

        item = {
            "clean": clean,
            "hazy": hazy,
            "clean_ref": clean_ref,
            "hazy_ref": hazy_ref,
            "clean_mask": clean_mask_tensor,
            "hazy_mask": hazy_mask_tensor,
            "is_real": self.is_real,
        }
        if self.return_paths:
            item.update(
                {
                    "clean_path": str(clean_path),
                    "hazy_path": str(hazy_path),
                    "clean_ref_path": str(clean_ref_path),
                    "hazy_ref_path": str(hazy_ref_path),
                }
            )
        return item


__all__ = [
    "DEHAZE_IMAGE_EXTENSIONS",
    "UnpairedDehazeDataset",
    "find_matching_mask",
    "read_dehaze_paths",
    "validate_image_mask_pairs",
    "validate_real_dataset_layout",
]
