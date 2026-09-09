"""Fog Aware Density Evaluator (FADE), ported from the LIVE MATLAB release.

Python port for UniSPS. The algorithm and embedded reference-model assets
come from the FADE 1.0 software release. The implementation deliberately keeps
the original 8-bit input, 8x8 patch, feature-order, and aggregation semantics.

-----------COPYRIGHT NOTICE STARTS WITH THIS LINE------------
Copyright (c) 2015 The University of Texas at Austin
All rights reserved.

Permission is hereby granted, without written agreement and without license or
royalty fees, to use, copy, modify, and distribute this code (the source files)
and its documentation for any purpose, provided that the copyright notice in
its entirety appear in all copies of this code, and the original source of this
code, Laboratory for Image and Video Engineering
(LIVE, http://live.ece.utexas.edu) at The University of Texas at Austin
(UT Austin, http://www.utexas.edu), is acknowledged in any publication that
reports research using this code. The research is to be cited in the
bibliography as:

1. L. K. Choi, J. You, and A. C. Bovik, "Referenceless Prediction of
   Perceptual Fog Density and Perceptual Image Defogging," IEEE Transactions
   on Image Processing, to appear (2015).
2. L. K. Choi, J. You, and A. C. Bovik, "Referenceless perceptual fog density
   prediction model," in Proc. SPIE Human Vis. Electron. Imag., Feb. 2014,
   90140H.
3. L. K. Choi, J. You, and A. C. Bovik, "FADE Software Release,"
   URL: http://live.ece.utexas.edu/research/fog/FADE_release.zip, 2015

IN NO EVENT SHALL THE UNIVERSITY OF TEXAS AT AUSTIN BE LIABLE TO ANY PARTY FOR
DIRECT, INDIRECT, SPECIAL, INCIDENTAL, OR CONSEQUENTIAL DAMAGES ARISING OUT OF
THE USE OF THIS DATABASE AND ITS DOCUMENTATION, EVEN IF THE UNIVERSITY OF TEXAS
AT AUSTIN HAS BEEN ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

THE UNIVERSITY OF TEXAS AT AUSTIN SPECIFICALLY DISCLAIMS ANY WARRANTIES,
INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND
FITNESS FOR A PARTICULAR PURPOSE. THE DATABASE PROVIDED HEREUNDER IS ON AN
"AS IS" BASIS, AND THE UNIVERSITY OF TEXAS AT AUSTIN HAS NO OBLIGATION TO
PROVIDE MAINTENANCE, SUPPORT, UPDATES, ENHANCEMENTS, OR MODIFICATIONS.
-----------COPYRIGHT NOTICE ENDS WITH THIS LINE------------
"""

import base64
import io
from dataclasses import dataclass
from functools import lru_cache
from typing import Union

import numpy as np
from scipy import ndimage
from scipy.io import loadmat
from scipy.linalg import solve


PATCH_SIZE = 8

# Exact model files from the official FADE_release.zip. Embedding the small
# assets keeps evaluation offline and avoids introducing a binary-data loader
# path outside this module.
# natural_fogfree_image_features_ps8.mat SHA256:
# 9a1812bcb2318908d864a53163e32cb430e92dddb88579ea02ee944f020d69cb
_FOGFREE_MODEL_B64 = """
TUFUTEFCIDUuMCBNQVQtZmlsZSwgUGxhdGZvcm06IFBDV0lONjQsIENyZWF0ZWQgb246IFNhdCBB
cHIgMDQgMTc6Mzc6NTcgMjAxNSAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAg
ICAgICAgICAgIAABSU0PAAAAQwMAAHicPZN9LNRxHMdPYidUc2OLPNwppUidHLa6j811FnFCmh1J
jovzcKRMNSm1VS7zNCYPWTUq/MN0Z+R7RzHzkOKE5m63SXOUpxXLoWv5+G3ffff7bb/33nt9Xp/d
FApFvpNCMdbfVMr/+99jtPVutnUM9Gev/sSm3oiOTxXGp8fFiWPSY5JN9N9y9P9LaPSM5TkxZDzx
TTK+wAf/jcA+97pIsD9bPFq1kAh7Nfk5mSwWaVHe+SA9KSAUhWPwJXsx8e1emh/fFJOZPPWxixNn
IOTHZP7jIgBLDtf9WjKdaAeCY0vVsQRzGdV9MD2SBN6FMsWvgl9sq+VWei0/BDrztTLuJU+yHskd
62KEE2kujTcwxiPth1NGzd08CdOhezCl8igkZGcmjLgdhdBvQoPfgzq2tcraOS/UmWBfzO0t1bHP
OcWDr+W98LXKUHj+2f+cvNABmAbdVnVF/uTtUqvkpsF5YuPqWNDvGUVUNmKORGYLX9mrTik0UyKw
3POtS7UfHC0OsRPm/QhywL6Yy7D2Lw+jt0OudeEnL8sSEM1UtVeYvgABmzF73a8GhqnkoI5VDn5z
YexFdTYEGSmLot9nwfRQFix3HCQVkV3mRbzabb7IAftibkyIsM5uUQJCJqNMKssB5UOvq/suCuHA
wneWopcHk3cDeL+pdpBeXevSFC4k76Z9IjaJK6mT1ZjNb0i254Z8kQP2xdzEof5dpf3PwKqe4Xn8
TzUoq/hMyCiHI3tuiVaHr8DktEgbpgkGUeDrlmRGIDRPNXiNWTwF9AHnhnyRA/bF3AafwdCpYgUw
h6gsTqwUTr3KCyoZi4Kzo7c9jMuSiLrBO1Hg8Qgam/srOW/kgJ6hDzg35IscsC/mDlOb4rgqBUzs
75woXQqB+g53mjPfEcw9TDK0JAc6XgoYdBc5oL/oGfqAc0O+yAH7Yu799YHEcZcImJXqtD7vAqCp
bWEHU2NHVgpWDD8Xp23vBfqLnqEPODfkixywL+YarnFpFlFpQLpUD3o0IiLQOVMGfl7e3jfcC/QX
PUMfcG7IFzlgX8wlnEbWCdtsuG54urtZVwa4x7hvuBfoL3qGPuDckC9ywL6Y+6WNv67q+Qh/ASBh
NdIPAAAAmQAAAHic42NgYFgAxGxAzAGlQYAVymcEYh4ozQ/EuaXxafnpaUWpqQWJRYm5DJxAsQSQ
GY3W0/k/3bCP7XY6YzB7n/01BZXnn59ssn/PwP5QYBeng+fj3S7z9lyyj824P/f86b/27zQNonrS
P9n/tja77tn4wT7rRGtRUNlfey3WNXGnPh21D/pp/zk56o79ins/GLnzuBwA0i04bQ==
"""

# natural_foggy_image_features_ps8.mat SHA256:
# 1de98cec133b367f6d2d9cfa039a2d53892a1606ea9f4b216d5b08f73b346afd
_FOGGY_MODEL_B64 = """
TUFUTEFCIDUuMCBNQVQtZmlsZSwgUGxhdGZvcm06IFBDV0lONjQsIENyZWF0ZWQgb246IFNhdCBBcHIgMDQgMTc6Mzg6NTEgMjAxNSAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgIAABSU0PAAAARQMAAHicPZN9LNRxHMd/lmtsHhI9EKK4PGwiPxK5TzoPlc7TKHNhHB2dy+OM3ZSrPCSHlIfUqFCXYxVak91+ytMfIuRExKJS2bCJMaFr9fn9tu+++/2233vvvT6vjxZBEG2qBLFVeasR/+6/D+P/u8b/o6I8WsrDi08NPx/P56cnRCRFCAlCXfktU/l/yas12X1hIjAzxtQKsjxBbhZYpiFjAY8XGmIyIgG9hg9eb0VMGE4NGKWupIC7eOFSckwM5PRnftd7KoSQ7FOk6psCcKyjPprEBVFeywOkmtQSzMLEca89xXTuzsrkPvm0BawK2ZMCti4EK3LSghyiIaXGsNWyXQW0R6bHmEb+0MuZZ/S4uMEC310zso4NzwzOOB/sjQRbh8SMuRZbKq0Cwj/7qICDpsvnAfsgui/mlsw8jvX5bQp2HLehGScudMV2DN2uXGIFD7Af7Dh0HCZJ8wqt2cPQ6dTJ0E91gWvPLcS1eWeBcdcnt3yESRXeaxCvcldY+T/87nRU+NEcsC/mOhenG0TWNQPjwvgX8ok/kEXrq3yLWvAOaMhbqyqEPe4eFxaTb0KjiUqZrqQJ9rtG2mm/v0Ld8FrPb9HxhVE9e5bc+wHNFzlgX8wt3ri8/2voIitLqqnhaMqG5n2bmVUCK+gpMhGdXbCGRNmu1tUDHGBK5ojeF2aU47awV42ibyzhww1mvJ4jPTfkixywL+auCAYna0sfwfCRtHJpQx5Mddvwsl2ug/4WvkAxVQ2cOa7NN1kSVfbxdOfPZQfwe6lmcDI/n/YB54Z8kQP2xVzJ0bFR//AKSACF77uEYqivN5y4yL0FEdUPdY6QXGrWqs+sKMMTfg11d8RKqmnP0AecG/JFDtgXc4cmZEulujWwfffVCK5HKQwq+pdMBdGUqrs6uXnrHJwQ9FoH857R/qJn6APODfkiB+yLuZpFPt/bPsnB3DXqBmkspuYnh43SyTMwECWPzxFJ6b1Af9Ez9AHnhnyRA/bF3Bjpy70i42TIHS/tc2vyoKbWSm4HdonpfcO9QH/RM/QB54Z8kQP2xdysi3UTKSf9oH1T65hxSyG9x7hvuBfoL3qGPuDckC9ywL6Yu8TpWCyYV8AfPfsckQ8AAACYAAAAeJzjY2BgWADEbEDMAaVBgBXKZwRiHijNC8S5pfFp+enplQWJRYm5QD4nECcA8dI9XvsTp++0/3m1ZcGvLa32Dnz3zOb+bbDXUDqsZ1T62H66toGB4aZie6HtEaIe/CfsH81nsOM7f8C+apFYcn3AOXuD6DU31Uze2i/9d2OLve4j+6pZFvyLM5fa/66e9s6E56s9AL/qM6E=
"""


@dataclass(frozen=True)
class _ReferenceModel:
    mean: np.ndarray
    half_covariance: np.ndarray
    half_covariance_solve_ones: np.ndarray
    half_covariance_ones_quadratic: float


def _decode_model(encoded: str, mean_key: str, covariance_key: str) -> _ReferenceModel:
    data = loadmat(io.BytesIO(base64.b64decode(encoded)))
    mean = np.asarray(data[mean_key], dtype=np.float64).reshape(-1)
    covariance = np.asarray(data[covariance_key], dtype=np.float64)
    half_covariance = covariance / 2.0
    ones = np.ones(mean.shape[0], dtype=np.float64)
    solved_ones = solve(
        half_covariance,
        ones,
        assume_a="sym",
        check_finite=False,
    )
    return _ReferenceModel(
        mean=mean,
        half_covariance=half_covariance,
        half_covariance_solve_ones=solved_ones,
        half_covariance_ones_quadratic=float(ones @ solved_ones),
    )


@lru_cache(maxsize=1)
def _reference_models() -> tuple[_ReferenceModel, _ReferenceModel]:
    fogfree = _decode_model(
        _FOGFREE_MODEL_B64,
        "mu_fogfreeparam",
        "cov_fogfreeparam",
    )
    foggy = _decode_model(
        _FOGGY_MODEL_B64,
        "mu_foggyparam",
        "cov_foggyparam",
    )
    return fogfree, foggy


def _split_blocks(image: np.ndarray) -> np.ndarray:
    rows, columns = image.shape
    return image.reshape(
        rows // PATCH_SIZE,
        PATCH_SIZE,
        columns // PATCH_SIZE,
        PATCH_SIZE,
    ).transpose(0, 2, 1, 3)


def _row_nanvar(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    finite = np.isfinite(values)
    counts = finite.sum(axis=1)
    totals = np.where(finite, values, 0.0).sum(axis=1)
    means = np.divide(
        totals,
        counts,
        out=np.zeros_like(totals),
        where=counts > 0,
    )
    centered = np.where(finite, values - means[:, None], 0.0)
    sum_squares = np.square(centered).sum(axis=1)
    result = np.full(values.shape[0], np.nan, dtype=np.float64)
    result[counts == 1] = 0.0
    valid = counts > 1
    result[valid] = sum_squares[valid] / (counts[valid] - 1.0)
    return result


def _matlab_gray(image: np.ndarray) -> np.ndarray:
    weights = np.array(
        [0.298936021293775, 0.587043074451121, 0.114020904255103],
        dtype=np.float64,
    )
    gray = np.tensordot(
        image.astype(np.float64, copy=False),
        weights,
        axes=([2], [0]),
    )
    return np.clip(np.floor(gray + 0.5), 0, 255)


def _saturation(image: np.ndarray) -> np.ndarray:
    rgb = image.astype(np.float64, copy=False) / 255.0
    maximum = rgb.max(axis=2)
    minimum = rgb.min(axis=2)
    return np.divide(
        maximum - minimum,
        maximum,
        out=np.zeros_like(maximum),
        where=maximum > 0,
    )


@lru_cache(maxsize=1)
def _mscn_kernel() -> tuple[np.ndarray, np.ndarray]:
    axis = np.arange(-3.0, 4.0, dtype=np.float64)
    vector = np.exp(-(axis * axis) / (2.0 * (7.0 / 6.0) ** 2))
    vector /= vector.sum()
    return vector, vector


def _local_mean(image: np.ndarray) -> np.ndarray:
    column_kernel, row_kernel = _mscn_kernel()
    filtered = ndimage.correlate1d(
        image,
        column_kernel,
        axis=0,
        mode="nearest",
    )
    return ndimage.correlate1d(
        filtered,
        row_kernel,
        axis=1,
        mode="nearest",
    )


def _border_in(image: np.ndarray, size: int) -> np.ndarray:
    upper = size // 2
    lower = upper - 1 if size % 2 == 0 else upper
    vertical = np.concatenate(
        [image[:upper, :], image, image[-(lower + 1) :, :]],
        axis=0,
    )
    return np.concatenate(
        [
            vertical[:, :upper],
            vertical,
            vertical[:, -(lower + 1) :],
        ],
        axis=1,
    )


def _border_out(image: np.ndarray, size: int) -> np.ndarray:
    upper = size // 2
    lower = upper - 1 if size % 2 == 0 else upper
    return image[
        upper : image.shape[0] - (lower + 1),
        upper : image.shape[1] - (lower + 1),
    ]


@lru_cache(maxsize=1)
def _contrast_kernel() -> np.ndarray:
    sigma = 3.25
    axis = -9.75 + np.arange(20, dtype=np.float64)
    gaussian = (
        np.exp((axis * axis) / (-2.0 * sigma * sigma))
        / (np.sqrt(2.0 * np.pi) * sigma)
    )
    gaussian /= gaussian.sum()
    kernel = ((axis * axis) / sigma**4 - 1.0 / sigma**2) * gaussian
    kernel -= kernel.mean()
    kernel /= np.sum(0.5 * axis * axis * kernel)
    return kernel


def _contrast_energy(channel: np.ndarray, threshold: float) -> np.ndarray:
    padded = _border_in(channel, 20)
    kernel = _contrast_kernel()
    horizontal = ndimage.convolve1d(
        padded,
        kernel,
        axis=1,
        mode="constant",
        cval=0.0,
        origin=0,
    )
    vertical = ndimage.convolve1d(
        padded,
        kernel,
        axis=0,
        mode="constant",
        cval=0.0,
        origin=0,
    )
    contrast = _border_out(np.hypot(horizontal, vertical), 20)
    maximum = float(np.max(contrast))
    with np.errstate(divide="ignore", invalid="ignore"):
        response = (contrast * maximum) / (contrast + maximum * 0.1)
    response -= threshold
    return np.where(response > 1e-7, response, 0.0)


def _entropy_blocks(gray: np.ndarray) -> np.ndarray:
    blocks = _split_blocks(gray.astype(np.uint8)).reshape(-1, 64)
    sorted_blocks = np.sort(blocks, axis=1)
    starts = np.ones_like(sorted_blocks, dtype=bool)
    starts[:, 1:] = sorted_blocks[:, 1:] != sorted_blocks[:, :-1]
    rows, columns = np.nonzero(starts)
    next_columns = np.empty_like(columns)
    next_columns[:-1] = columns[1:]
    next_columns[-1] = 64
    row_ends = np.empty_like(rows, dtype=bool)
    row_ends[:-1] = rows[1:] != rows[:-1]
    row_ends[-1] = True
    next_columns[row_ends] = 64
    run_lengths = next_columns - columns
    probabilities = run_lengths.astype(np.float64) / 64.0
    terms = -(probabilities * np.log2(probabilities))
    return np.bincount(
        rows,
        weights=terms,
        minlength=blocks.shape[0],
    )


def _feature_matrix(image: np.ndarray) -> tuple[np.ndarray, tuple[int, int]]:
    rows = image.shape[0] // PATCH_SIZE * PATCH_SIZE
    columns = image.shape[1] // PATCH_SIZE * PATCH_SIZE
    image = image[:rows, :columns, :3]
    if rows == 0 or columns == 0:
        raise ValueError("FADE requires an image of at least 8x8 pixels")

    rgb = image.astype(np.float64, copy=False)
    red, green, blue = np.moveaxis(rgb, -1, 0)
    gray = _matlab_gray(image)
    dark_channel = rgb.min(axis=2) / 255.0
    saturation = _saturation(image)

    local_mean = _local_mean(gray)
    second_moment = _local_mean(gray * gray)
    sigma = np.sqrt(np.abs(second_moment - local_mean * local_mean))
    mscn = (gray - local_mean) / (sigma + 1.0)
    with np.errstate(divide="ignore", invalid="ignore"):
        coefficient_of_variation = sigma / local_mean

    red_green = red - green
    blue_yellow = 0.5 * (red + green) - blue
    vertical_pair = mscn * np.roll(mscn, shift=1, axis=0)
    negative_pair = np.where(vertical_pair > 0, np.nan, vertical_pair)
    positive_pair = np.where(vertical_pair < 0, np.nan, vertical_pair)

    def flattened_blocks(values: np.ndarray) -> np.ndarray:
        return _split_blocks(values).reshape(-1, 64)

    mscn_variance = _row_nanvar(flattened_blocks(mscn))
    positive_pair_variance = _row_nanvar(flattened_blocks(positive_pair))
    negative_pair_variance = _row_nanvar(flattened_blocks(negative_pair))

    gray_ce = _contrast_energy(
        0.299 * red + 0.587 * green + 0.114 * blue,
        9.225496406318721e-4 * 255.0,
    )
    blue_yellow_ce = _contrast_energy(
        blue_yellow,
        8.969246659629488e-4 * 255.0,
    )
    red_green_ce = _contrast_energy(
        red_green,
        2.069284034165411e-4 * 255.0,
    )

    red_green_blocks = flattened_blocks(red_green)
    blue_yellow_blocks = flattened_blocks(blue_yellow)
    colorfulness = np.hypot(
        np.std(red_green_blocks, axis=1, ddof=1),
        np.std(blue_yellow_blocks, axis=1, ddof=1),
    )
    colorfulness += 0.3 * np.hypot(
        red_green_blocks.mean(axis=1),
        blue_yellow_blocks.mean(axis=1),
    )

    feature_columns = [
        mscn_variance,
        positive_pair_variance,
        negative_pair_variance,
        flattened_blocks(sigma).mean(axis=1),
        flattened_blocks(coefficient_of_variation).mean(axis=1),
        flattened_blocks(gray_ce).mean(axis=1),
        flattened_blocks(blue_yellow_ce).mean(axis=1),
        flattened_blocks(red_green_ce).mean(axis=1),
        _entropy_blocks(gray),
        flattened_blocks(dark_channel).mean(axis=1),
        flattened_blocks(saturation).mean(axis=1),
        colorfulness,
    ]
    # The block arrays above are row-major. MATLAB im2col/reshape and (:)
    # enumerate the patch grid column-major, so reorder every feature map.
    patch_grid = (rows // PATCH_SIZE, columns // PATCH_SIZE)
    reordered = [
        values.reshape(patch_grid).reshape(-1, order="F")
        for values in feature_columns
    ]
    return np.log1p(np.column_stack(reordered)), patch_grid


def _distance_to_model(
    features: np.ndarray,
    model: _ReferenceModel,
) -> np.ndarray:
    patch_variance = _row_nanvar(features)
    delta = model.mean[None, :] - features
    solved_delta = solve(
        model.half_covariance,
        delta.T,
        assume_a="sym",
        check_finite=False,
    ).T
    base_distance_squared = np.einsum("ij,ij->i", delta, solved_delta)

    # The original MATLAB code expands each patch's scalar variance to a
    # constant 12x12 matrix. Apply Sherman-Morrison to its rank-one update.
    alpha = patch_variance / 2.0
    delta_ones = delta @ model.half_covariance_solve_ones
    correction = (
        alpha
        * np.square(delta_ones)
        / (1.0 + alpha * model.half_covariance_ones_quadratic)
    )
    distance_squared = base_distance_squared - correction
    tiny_negative = (
        (distance_squared < 0)
        & (np.abs(distance_squared) < 1e-12)
    )
    distance_squared[tiny_negative] = 0.0
    with np.errstate(invalid="ignore"):
        return np.sqrt(distance_squared)


def calculate_fade(
    image: np.ndarray,
    *,
    return_map: bool = False,
) -> Union[float, tuple[float, np.ndarray]]:
    """Calculate FADE for one 8-bit RGB image; lower is less perceptual fog."""
    array = np.asarray(image)
    if array.ndim != 3 or array.shape[2] != 3:
        raise ValueError("FADE expects an RGB image with shape (H, W, 3)")
    if array.dtype != np.uint8:
        raise TypeError("FADE expects an 8-bit uint8 RGB image")

    features, patch_grid = _feature_matrix(array)
    fogfree, foggy = _reference_models()
    distance_fogfree = _distance_to_model(features, fogfree)
    distance_foggy = _distance_to_model(features, foggy)
    score = float(
        np.nanmean(distance_fogfree)
        / (np.nanmean(distance_foggy) + 1.0)
    )
    if not return_map:
        return score
    density_map = (
        distance_fogfree / (distance_foggy + 1.0)
    ).reshape(patch_grid, order="F")
    return score, density_map


__all__ = ["PATCH_SIZE", "calculate_fade"]
