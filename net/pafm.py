"""Physical-aware cross-channel and spatial feature fusion."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class PAFM(nn.Module):
    """Fuse two aligned ``[B,C,H,W]`` feature maps with cross attention."""

    def __init__(self, channel=64):
        super().__init__()
        self.conv1_spatial = nn.Conv2d(2, 1, 3, stride=1, padding=1, groups=1)
        self.conv2_spatial = nn.Conv2d(1, 1, 3, stride=1, padding=1, groups=1)
        self.avg1 = nn.Conv2d(channel, channel // 2, 1, stride=1, padding=0)
        self.avg2 = nn.Conv2d(channel, channel // 2, 1, stride=1, padding=0)
        self.max1 = nn.Conv2d(channel, channel // 2, 1, stride=1, padding=0)
        self.max2 = nn.Conv2d(channel, channel // 2, 1, stride=1, padding=0)
        self.avg11 = nn.Conv2d(channel // 2, channel, 1, stride=1, padding=0)
        self.avg22 = nn.Conv2d(channel // 2, channel, 1, stride=1, padding=0)
        self.max11 = nn.Conv2d(channel // 2, channel, 1, stride=1, padding=0)
        self.max22 = nn.Conv2d(channel // 2, channel, 1, stride=1, padding=0)
        self.channel = channel
        self.fusion = nn.Conv2d(channel * 2, channel, 1, 1, 0)

    def forward(self, f1, f2):
        if f1.shape != f2.shape or f1.ndim != 4:
            raise ValueError(
                f"PAFM inputs must share [B,C,H,W] shape, got {tuple(f1.shape)} "
                f"and {tuple(f2.shape)}"
            )
        batch, channels, height, width = f1.size()
        f1_flat = f1.view(batch, channels, -1)
        f2_flat = f2.view(batch, channels, -1)

        avg_1 = torch.mean(f1_flat, dim=-1, keepdim=True).unsqueeze(-1)
        max_1, _ = torch.max(f1_flat, dim=-1, keepdim=True)
        max_1 = max_1.unsqueeze(-1)
        avg_1 = self.avg11(F.relu(self.avg1(avg_1))).squeeze(-1)
        max_1 = self.max11(F.relu(self.max1(max_1))).squeeze(-1)
        attention_1 = avg_1 + max_1

        avg_2 = torch.mean(f2_flat, dim=-1, keepdim=True).unsqueeze(-1)
        max_2, _ = torch.max(f2_flat, dim=-1, keepdim=True)
        max_2 = max_2.unsqueeze(-1)
        avg_2 = self.avg22(F.relu(self.avg2(avg_2))).squeeze(-1)
        max_2 = self.max22(F.relu(self.max2(max_2))).squeeze(-1)
        attention_2 = avg_2 + max_2

        cross = torch.matmul(attention_1, attention_2.transpose(1, 2))
        attended_1 = torch.matmul(F.softmax(cross, dim=-1), f1_flat).view(
            batch, channels, height, width
        )
        attended_2 = torch.matmul(
            F.softmax(cross.transpose(1, 2), dim=-1), f2_flat
        ).view(batch, channels, height, width)

        avg_out1 = torch.mean(attended_1, dim=1, keepdim=True)
        max_out1, _ = torch.max(attended_1, dim=1, keepdim=True)
        spatial_1 = self.conv2_spatial(
            F.relu(self.conv1_spatial(torch.cat([avg_out1, max_out1], dim=1)))
        )
        spatial_1 = F.softmax(spatial_1.view(batch, 1, -1), dim=-1).view(
            batch, 1, height, width
        )

        avg_out2 = torch.mean(attended_2, dim=1, keepdim=True)
        max_out2, _ = torch.max(attended_2, dim=1, keepdim=True)
        spatial_2 = self.conv2_spatial(
            F.relu(self.conv1_spatial(torch.cat([avg_out2, max_out2], dim=1)))
        )
        spatial_2 = F.softmax(spatial_2.view(batch, 1, -1), dim=-1).view(
            batch, 1, height, width
        )
        out1 = f1 * spatial_1 + f1
        out2 = f2 * spatial_2 + f2
        return self.fusion(torch.cat((out1, out2), dim=1))


__all__ = ["PAFM"]
