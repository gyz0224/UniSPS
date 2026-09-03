"""PSNR and MATLAB-style SSIM calculations used by SPS-Net scripts."""

import cv2
import numpy as np


def ssim(prediction, target):
    c1 = (0.01 * 255) ** 2
    c2 = (0.03 * 255) ** 2
    image1 = prediction.astype(np.float64)
    image2 = target.astype(np.float64)
    kernel = cv2.getGaussianKernel(11, 1.5)
    window = np.outer(kernel, kernel.transpose())
    mu1 = cv2.filter2D(image1, -1, window)[5:-5, 5:-5]
    mu2 = cv2.filter2D(image2, -1, window)[5:-5, 5:-5]
    mu1_sq = mu1**2
    mu2_sq = mu2**2
    mu1_mu2 = mu1 * mu2
    sigma1_sq = cv2.filter2D(image1**2, -1, window)[5:-5, 5:-5] - mu1_sq
    sigma2_sq = cv2.filter2D(image2**2, -1, window)[5:-5, 5:-5] - mu2_sq
    sigma12 = cv2.filter2D(image1 * image2, -1, window)[5:-5, 5:-5] - mu1_mu2
    ssim_map = ((2 * mu1_mu2 + c1) * (2 * sigma12 + c2)) / (
        (mu1_sq + mu2_sq + c1) * (sigma1_sq + sigma2_sq + c2)
    )
    return ssim_map.mean()


def calculate_ssim(target, reference):
    image1 = np.array(target, dtype=np.float64)
    image2 = np.array(reference, dtype=np.float64)
    if image1.shape != image2.shape:
        raise ValueError("Input images must have the same dimensions.")
    if image1.ndim == 2:
        return ssim(image1, image2)
    if image1.ndim == 3:
        if image1.shape[2] == 3:
            return np.array(
                [ssim(image1[:, :, index], image2[:, :, index]) for index in range(3)]
            ).mean()
        if image1.shape[2] == 1:
            return ssim(np.squeeze(image1), np.squeeze(image2))
    raise ValueError("Wrong input image dimensions.")


def calculate_psnr(target, reference):
    image1 = np.array(target, dtype=np.float32)
    image2 = np.array(reference, dtype=np.float32)
    difference = image1 - image2
    return 10.0 * np.log10(255.0 * 255.0 / np.mean(np.square(difference)))


__all__ = ["calculate_psnr", "calculate_ssim", "ssim"]
