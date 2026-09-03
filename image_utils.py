"""Small PIL image composition helpers retained from the original utilities."""

from PIL import Image


def joint_RGB_horizontal(im1, im2):
    if im1.size == im2.size:
        width, height = im1.size
        result = Image.new("RGB", (width * 2, height))
        result.paste(im1, box=(0, 0))
        result.paste(im2, box=(width, 0))
        return result
    return None


def joint_L_horizontal(im1, im2):
    if im1.size == im2.size:
        width, height = im1.size
        result = Image.new("L", (width * 2, height))
        result.paste(im1, box=(0, 0))
        result.paste(im2, box=(width, 0))
        return result
    return None


__all__ = ["joint_L_horizontal", "joint_RGB_horizontal"]
