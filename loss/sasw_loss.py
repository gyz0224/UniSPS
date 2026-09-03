"""Walsh-Hadamard spectral adaptive spatial weighting loss."""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class WHTBlock(nn.Module):
    def __init__(
        self,
        block_size=16,
        thresh=0.1,
        calc_iwht=True,
        isdiff=True,
        final_level="wht",
        normalized=True,
        updown_flg=False,
    ):
        super().__init__()
        self.walsh_matrix = self.generate_walsh_matrix(block_size)
        self.block_size = block_size
        self.normalized = normalized
        self.calc_iwht = calc_iwht
        self.threshold = thresh
        self.isdiff = isdiff
        self.final_level = final_level
        self.updown_flg = updown_flg

    @staticmethod
    def generate_walsh_matrix(n):
        assert (n & (n - 1)) == 0, "n必须是2的幂次"
        if n == 1:
            return torch.ones((1, 1))
        matrix = WHTBlock.generate_walsh_matrix(n // 2)
        return torch.cat(
            [
                torch.cat([matrix, matrix], dim=1),
                torch.cat([matrix, -matrix], dim=1),
            ],
            dim=0,
        )

    def wht_1d(self, x):
        n = x.shape[-1]
        matrix = self.walsh_matrix.to(x.device)
        if self.normalized:
            matrix = matrix / np.sqrt(n)
        return torch.matmul(x, matrix.T)

    def iwht_1d(self, x):
        n = x.shape[-1]
        matrix = self.walsh_matrix.to(x.device)
        if self.normalized:
            matrix = matrix.T / np.sqrt(n)
        else:
            matrix = matrix.T / n
        return torch.matmul(x, matrix)

    def wht_2d(self, x):
        x = self.wht_1d(x)
        return self.wht_1d(x.transpose(-1, -2)).transpose(-1, -2)

    def iwht_2d(self, x):
        x = self.iwht_1d(x)
        return self.iwht_1d(x.transpose(-1, -2)).transpose(-1, -2)

    def call0(self, input, thresh=0.1):
        _, _, height, width = input.shape
        block_size = self.block_size
        h_pad = ((height + block_size - 1) // block_size) * block_size - height
        w_pad = ((width + block_size - 1) // block_size) * block_size - width
        padded = F.pad(input, (0, w_pad, 0, h_pad), "reflect")
        reshaped = padded.unfold(2, block_size, block_size).unfold(
            3, block_size, block_size
        )
        coefficients = self.wht_2d(reshaped)
        coefficient_abs = torch.abs(coefficients)
        coefficients[coefficient_abs < 0.02] = 0
        reconstructed = self.iwht_2d(coefficients)
        return coefficients, reconstructed

    def cal_wht(self, input, threshold):
        _, _, height, width = input.shape
        block_size = self.block_size
        h_pad = ((height + block_size - 1) // block_size) * block_size - height
        w_pad = ((width + block_size - 1) // block_size) * block_size - width
        padded = F.pad(input, (0, w_pad, 0, h_pad), "reflect")
        reshaped = padded.unfold(2, block_size, block_size).unfold(
            3, block_size, block_size
        )
        coefficients = self.wht_2d(reshaped)
        coefficient_abs = torch.abs(coefficients)
        outputs = []
        if isinstance(threshold, (int, float)):
            coefficients[coefficient_abs < threshold] = 0
            outputs.append(coefficients)
            return outputs
        if isinstance(threshold, (np.ndarray, list, torch.Tensor)):
            if isinstance(threshold, torch.Tensor):
                threshold = threshold.numpy()
            for value in threshold:
                copied = coefficients.clone()
                copied[coefficient_abs < value] = 0
                outputs.append(copied)
            return outputs
        raise ValueError("Threshold should be either a number or an array.")

    def cal_wht_2d_diff(self, input, threshold, isdiff=True, final_level="ori"):
        _, _, height, width = input.shape
        block_size = self.block_size
        h_pad = ((height + block_size - 1) // block_size) * block_size - height
        w_pad = ((width + block_size - 1) // block_size) * block_size - width
        padded = F.pad(input, (0, w_pad, 0, h_pad), "reflect")
        reshaped = padded.unfold(2, block_size, block_size).unfold(
            3, block_size, block_size
        )
        coefficients = self.wht_2d(reshaped)
        coefficient_abs = torch.abs(coefficients)
        outputs = []
        values = threshold
        if isinstance(values, torch.Tensor):
            values = values.numpy()
        if isinstance(values, (int, float)):
            values = [values]
        if not isinstance(values, (np.ndarray, list)):
            raise ValueError("Threshold should be either a number or an array.")
        for value in values:
            copied = coefficients.clone()
            copied[coefficient_abs < value] = 0
            reconstructed = self.iwht_2d(copied)
            outputs.append(
                torch.abs(reshaped - reconstructed) if isdiff else reconstructed
            )
        if final_level == "ori":
            outputs.append(reshaped)
        elif final_level == "wht":
            outputs.append(coefficients)
        return outputs

    def call_half_wht(self, input):
        _, _, height, width = input.shape
        block_size = self.block_size
        h_pad = ((height + block_size - 1) // block_size) * block_size - height
        w_pad = ((width + block_size - 1) // block_size) * block_size - width
        padded = F.pad(input, (0, w_pad, 0, h_pad), "reflect")
        reshaped = padded.unfold(2, block_size, block_size).unfold(
            3, block_size, block_size
        )
        coefficients = self.wht_2d(reshaped)
        upper = coefficients.clone()
        lower = coefficients.clone()
        lower[:, :, :, :, : block_size // 2, : block_size // 2] = 0
        upper[:, :, :, :, block_size // 2 :, : block_size // 2] = 0
        return [
            self.iwht_2d(upper),
            self.iwht_2d(lower),
            coefficients,
        ]

    def forward(self, input):
        if self.updown_flg:
            return self.call_half_wht(input)
        if self.calc_iwht:
            return self.cal_wht(input, self.threshold)
        return self.cal_wht_2d_diff(
            input=input,
            threshold=self.threshold,
            isdiff=self.isdiff,
            final_level=self.final_level,
        )


class CosineLoss(nn.Module):
    def __init__(self, reduction="mean", eps=1e-8):
        super().__init__()
        self.reduction = reduction
        self.eps = eps

    def forward(self, input, target):
        input_norm = F.normalize(input, dim=-1, eps=self.eps)
        target_norm = F.normalize(target, dim=-1, eps=self.eps)
        cos_loss = 1.0 - (input_norm * target_norm).sum(dim=-1)
        if self.reduction == "mean":
            return cos_loss.mean()
        if self.reduction == "sum":
            return cos_loss.sum()
        return cos_loss


class SASWLoss(nn.Module):
    def __init__(self, loss_weight=1.0, reduction="mean"):
        super().__init__()
        self.block_size = 32
        self.thresh = [0.2, 0.1, 0.05]
        self.wht = WHTBlock(
            block_size=self.block_size,
            thresh=self.thresh,
            calc_iwht=True,
            isdiff=True,
            final_level="wht",
            normalized=True,
            updown_flg=True,
        )
        self.lambda_value = [1, 2, 1, 1]
        self.l1_loss = nn.L1Loss(reduction="mean")
        self.loss1 = CosineLoss()
        if reduction not in ["none", "mean", "sum"]:
            raise ValueError(
                f"Unsupported reduction mode: {reduction}. "
                "Supported ones are: none, mean, sum"
            )

    def forward(self, pred, gt, weight=None, **kwargs):
        pred_list = self.wht(pred)
        gt_list = self.wht(gt)
        assert len(pred_list) == len(gt_list)
        loss_val = 0.0
        for index in range(len(pred_list) - 1):
            if index == 1:
                pred_level = pred_list[index]
                gt_level = gt_list[index]
                assert pred_level.shape == gt_level.shape, (
                    f"Layer {index} shape mismatch"
                )
                loss_val += self.lambda_value[index] * self.loss1(
                    pred_level.reshape(pred_level.size(0), -1),
                    gt_level.reshape(pred_level.size(0), -1),
                )
        pred_level = pred_list[-1]
        gt_level = gt_list[-1]
        loss_val += self.lambda_value[-1] * torch.mean(
            self.l1_loss(pred_level, gt_level)
        )
        return loss_val


__all__ = ["CosineLoss", "SASWLoss", "WHTBlock"]
