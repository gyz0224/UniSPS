"""Neighboring-pixel sampling and gamma augmentation for low-light training."""

import random
from typing import Optional, Union

import torch
import torch.nn.functional as F


operation_seed_counter = 0


def get_generator(
    device: Optional[Union[str, torch.device]] = None,
) -> torch.Generator:
    """Return the next deterministically seeded generator on ``device``."""
    global operation_seed_counter
    operation_seed_counter += 1
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    generator = torch.Generator(device=torch.device(device))
    generator.manual_seed(operation_seed_counter)
    return generator


def space_to_depth(x, block_size):
    n, c, h, w = x.size()
    unfolded_x = F.unfold(x, block_size, stride=block_size)
    return unfolded_x.view(
        n, c * block_size**2, h // block_size, w // block_size
    )


def generate_mask_pair(img):
    """Generate one randomly selected pixel pair per non-overlapping 2x2 block."""
    n, _, h, w = img.shape
    if h % 2 or w % 2:
        raise ValueError(
            f"generate_mask_pair requires even H/W, received {(h, w)}"
        )
    size = n * h // 2 * w // 2
    mask1 = torch.zeros(size=(size * 4,), dtype=torch.bool, device=img.device)
    mask2 = torch.zeros(size=(size * 4,), dtype=torch.bool, device=img.device)
    idx_pair = torch.tensor(
        [[0, 1], [0, 2], [1, 3], [2, 3], [1, 0], [2, 0], [3, 1], [3, 2]],
        dtype=torch.int64,
        device=img.device,
    )
    rd_idx = torch.zeros(size=(size,), dtype=torch.int64, device=img.device)
    torch.randint(
        low=0,
        high=8,
        size=(size,),
        generator=get_generator(img.device),
        out=rd_idx,
    )
    rd_pair_idx = idx_pair[rd_idx]
    rd_pair_idx += torch.arange(
        start=0,
        end=size * 4,
        step=4,
        dtype=torch.int64,
        device=img.device,
    ).reshape(-1, 1)
    mask1[rd_pair_idx[:, 0]] = 1
    mask2[rd_pair_idx[:, 1]] = 1
    return mask1, mask2


def pair_downsampler(img):
    channels = img.shape[1]
    filter1 = img.new_tensor([[[[0, 0.5], [0.5, 0]]]]).repeat(
        channels, 1, 1, 1
    )
    filter2 = img.new_tensor([[[[0.5, 0], [0, 0.5]]]]).repeat(
        channels, 1, 1, 1
    )
    output1 = F.conv2d(img, filter1, stride=2, groups=channels)
    output2 = F.conv2d(img, filter2, stride=2, groups=channels)
    return output1, output2


def generate_subimages(img, mask):
    n, c, h, w = img.shape
    subimage = torch.zeros(
        n,
        c,
        h // 2,
        w // 2,
        dtype=img.dtype,
        layout=img.layout,
        device=img.device,
    )
    for index in range(c):
        per_channel = space_to_depth(img[:, index : index + 1], block_size=2)
        per_channel = per_channel.permute(0, 2, 3, 1).reshape(-1)
        subimage[:, index : index + 1] = (
            per_channel[mask]
            .reshape(n, h // 2, w // 2, 1)
            .permute(0, 3, 1, 2)
        )
    return subimage


def gamma_correction(images):
    gamma = random.uniform(1.2, 1.5)
    return torch.clamp(images.pow(1 / gamma), 0, 1)


__all__ = [
    "gamma_correction",
    "generate_mask_pair",
    "generate_subimages",
    "get_generator",
    "pair_downsampler",
    "space_to_depth",
]
