"""Retinex, exposure, spatial, TV, and color losses for SPS low-light training."""

import torch
import torch.nn as nn
import torch.nn.functional as F


def gradient(img):
    height = img.size(2)
    width = img.size(3)
    gradient_h = (img[:, :, 2:, :] - img[:, :, : height - 2, :]).abs()
    gradient_w = (img[:, :, :, 2:] - img[:, :, :, : width - 2]).abs()
    return gradient_h, gradient_w


def tv_loss(illumination):
    gradient_illu_h, gradient_illu_w = gradient(illumination)
    return gradient_illu_h.mean() + gradient_illu_w.mean()


class L_TV(nn.Module):
    def __init__(self, TVLoss_weight=1):
        super().__init__()
        self.TVLoss_weight = TVLoss_weight

    def forward(self, x):
        batch_size = x.size(0)
        height = x.size(2)
        width = x.size(3)
        count_h = (height - 1) * width
        count_w = height * (width - 1)
        h_tv = torch.pow(x[:, :, 1:, :] - x[:, :, : height - 1, :], 2).sum()
        w_tv = torch.pow(x[:, :, :, 1:] - x[:, :, :, : width - 1], 2).sum()
        return (
            self.TVLoss_weight
            * 2
            * (h_tv / count_h + w_tv / count_w)
            / batch_size
        )


class L_color(nn.Module):
    def forward(self, x):
        mean_rgb = torch.mean(x, [2, 3], keepdim=True)
        mr, mg, mb = torch.split(mean_rgb, 1, dim=1)
        drg = torch.pow(mr - mg, 2)
        drb = torch.pow(mr - mb, 2)
        dgb = torch.pow(mb - mg, 2)
        return torch.pow(torch.pow(drg, 2) + torch.pow(drb, 2) + torch.pow(dgb, 2), 0.5)


class L_exp(nn.Module):
    def __init__(self, patch_size, mean_val):
        super().__init__()
        self.pool = nn.AvgPool2d(patch_size)
        self.mean_val = mean_val

    def forward(self, x):
        x = torch.mean(x, 1, keepdim=True)
        mean = self.pool(x)
        return torch.mean(torch.pow(mean - mean.new_tensor(self.mean_val), 2))


class L_spa(nn.Module):
    def __init__(self):
        super().__init__()
        kernels = {
            "left": [[0, 0, 0], [-1, 1, 0], [0, 0, 0]],
            "right": [[0, 0, 0], [0, 1, -1], [0, 0, 0]],
            "up": [[0, -1, 0], [0, 1, 0], [0, 0, 0]],
            "down": [[0, 0, 0], [0, 1, 0], [0, -1, 0]],
        }
        for name, values in kernels.items():
            parameter = nn.Parameter(
                torch.tensor(values, dtype=torch.float32).unsqueeze(0).unsqueeze(0),
                requires_grad=False,
            )
            setattr(self, f"weight_{name}", parameter)
        self.pool = nn.AvgPool2d(4)

    def forward(self, org, enhance):
        org_pool = self.pool(torch.mean(org, 1, keepdim=True))
        enhance_pool = self.pool(torch.mean(enhance, 1, keepdim=True))
        d_org_left = F.conv2d(org_pool, self.weight_left, padding=1)
        d_org_right = F.conv2d(org_pool, self.weight_right, padding=1)
        d_org_up = F.conv2d(org_pool, self.weight_up, padding=1)
        d_org_down = F.conv2d(org_pool, self.weight_down, padding=1)
        d_enhance_left = F.conv2d(enhance_pool, self.weight_left, padding=1)
        d_enhance_right = F.conv2d(enhance_pool, self.weight_right, padding=1)
        d_enhance_up = F.conv2d(enhance_pool, self.weight_up, padding=1)
        d_enhance_down = F.conv2d(enhance_pool, self.weight_down, padding=1)
        return (
            torch.pow(d_org_left - d_enhance_left, 2)
            + torch.pow(d_org_right - d_enhance_right, 2)
            + torch.pow(d_org_up - d_enhance_up, 2)
            + torch.pow(d_org_down - d_enhance_down, 2)
        )


def C_loss(R1, R2):
    return nn.MSELoss()(R1, R2)


def R_loss(L1, R1, im1, X1):
    max_rgb1, _ = torch.max(im1, 1)
    max_rgb1 = max_rgb1.unsqueeze(1)
    loss1 = nn.MSELoss()(L1 * R1, X1) + nn.MSELoss()(R1, X1 / L1.detach())
    loss2 = nn.MSELoss()(L1, max_rgb1) + tv_loss(L1)
    return loss1 + loss2


def P_loss(im1, X1):
    return nn.MSELoss()(im1, X1)


__all__ = [
    "C_loss",
    "L_TV",
    "L_color",
    "L_exp",
    "L_spa",
    "P_loss",
    "R_loss",
    "gradient",
    "tv_loss",
]
