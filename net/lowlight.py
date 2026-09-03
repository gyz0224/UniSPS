"""Checkpoint-compatible SPS-Net low-light subnetworks."""

import torch
import torch.nn as nn

from net.blocks import IGABlock, OverlapPatchEmbed, SGA, TransformerBlock
from net.pafm import PAFM


class L_net(nn.Module):
    def __init__(
        self,
        num=48,
        num_heads=1,
        num_blocks=2,
        inp_channels=3,
        out_channels=1,
        ffn_expansion_factor=2.66,
        bias=False,
        LayerNorm_type="WithBias",
    ):
        super().__init__()
        self.patch_embed = OverlapPatchEmbed(inp_channels, num)
        self.encoder = nn.Sequential(
            *[
                TransformerBlock(
                    dim=num,
                    num_heads=num_heads,
                    ffn_expansion_factor=ffn_expansion_factor,
                    bias=bias,
                    LayerNorm_type=LayerNorm_type,
                )
                for _ in range(num_blocks)
            ]
        )
        self.output = nn.Conv2d(
            num, out_channels, kernel_size=3, stride=1, padding=1, bias=bias
        )

    def forward(self, input):
        out = self.patch_embed(input)
        out = self.encoder(out)
        out = self.output(out)
        return torch.sigmoid(out) + torch.mean(input, dim=1, keepdim=True)


class R_net(nn.Module):
    def __init__(
        self,
        num=64,
        num_heads=1,
        num_blocks=2,
        inp_channels=68,
        out_channels=3,
        ffn_expansion_factor=2.66,
        bias=False,
        LayerNorm_type="WithBias",
    ):
        super().__init__()
        self.phys_embed = OverlapPatchEmbed(in_c=4, embed_dim=num, bias=bias)
        self.pafm = PAFM(channel=num)
        self.encoder = nn.ModuleList(
            [
                IGABlock(
                    dim=num,
                    num_heads=num_heads,
                    ffn_expansion_factor=ffn_expansion_factor,
                    bias=bias,
                    LayerNorm_type=LayerNorm_type,
                )
                for _ in range(num_blocks)
            ]
        )
        self.sga = nn.ModuleList(
            [
                SGA(img_channels=num, sem_channels=3, num_heads=num_heads, bias=bias),
                SGA(
                    img_channels=num,
                    sem_channels=768,
                    num_heads=num_heads,
                    bias=bias,
                ),
            ]
        )
        self.output = nn.Conv2d(
            num, out_channels, kernel_size=3, stride=1, padding=1, bias=bias
        )

    def forward(self, feature_x, noisy_R, L, fea, pre_R, sem_feats=None):
        sem_feats = sem_feats or []
        physical_input = torch.cat([noisy_R, L.detach()], dim=1)
        physical_feature = self.phys_embed(physical_input)
        feature = self.pafm(feature_x, physical_feature)
        for index, block in enumerate(self.encoder):
            feature, fea = block([feature, fea])
            if index < len(self.sga) and index < len(sem_feats):
                feature = self.sga[index](feature, sem_feats[index])
        return torch.sigmoid(self.output(feature)) + pre_R


class Illumination_Estimator(nn.Module):
    def __init__(self, n_fea_middle, n_fea_in=4, n_fea_out=3):
        super().__init__()
        self.conv1 = nn.Conv2d(n_fea_in, n_fea_middle, kernel_size=1, bias=True)
        self.depth_conv = nn.Conv2d(
            n_fea_middle,
            n_fea_middle,
            kernel_size=5,
            padding=2,
            bias=True,
            groups=n_fea_in,
        )
        self.conv2 = nn.Conv2d(n_fea_middle, n_fea_out, kernel_size=1, bias=True)

    def forward(self, img):
        mean_c = img.mean(dim=1).unsqueeze(1)
        input = torch.cat([img, mean_c], dim=1)
        x_1 = self.conv1(input)
        illu_fea = self.depth_conv(x_1)
        illu_map = self.conv2(illu_fea)
        return illu_fea, illu_map


class N_net(nn.Module):
    def __init__(
        self,
        num=64,
        num_heads=1,
        num_blocks=2,
        inp_channels=3,
        out_channels=3,
        ffn_expansion_factor=2.66,
        bias=False,
        LayerNorm_type="WithBias",
    ):
        super().__init__()
        self.patch_embed = OverlapPatchEmbed(inp_channels, num)
        self.encoder = nn.Sequential(
            *[
                TransformerBlock(
                    dim=num,
                    num_heads=num_heads,
                    ffn_expansion_factor=ffn_expansion_factor,
                    bias=bias,
                    LayerNorm_type=LayerNorm_type,
                )
                for _ in range(num_blocks)
            ]
        )
        self.output = nn.Conv2d(
            num, out_channels, kernel_size=3, stride=1, padding=1, bias=bias
        )

    def encode(self, image):
        """Encode ``[B,3,H,W]`` into shared ``[B,64,H,W]`` SPS features."""
        return self.encoder(self.patch_embed(image))

    def restore_initial(self, image, feat):
        """Restore the legacy initial image from RGB input and shared features."""
        return torch.sigmoid(self.output(feat)) + image

    def forward(self, input):
        """Return the legacy ``(x_img, x_feat)`` pair unchanged."""
        x_feat = self.encode(input)
        return self.restore_initial(input, x_feat), x_feat


class Gamma_Predictor(nn.Module):
    def __init__(
        self,
        in_channel=1,
        num=32,
        num_heads=1,
        num_blocks=2,
        ffn_expansion_factor=2.66,
        bias=False,
        LayerNorm_type="WithBias",
    ):
        super().__init__()
        self.head = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(1, num, 3, 1, 0),
            nn.ReLU(),
            nn.ReflectionPad2d(1),
            nn.Conv2d(num, num, 3, 1, 0),
            nn.ReLU(),
            nn.ReflectionPad2d(1),
            nn.Conv2d(num, num, 3, 1, 0),
            nn.ReLU(),
        )
        self.lin = nn.AdaptiveAvgPool2d(1)
        self.tail = nn.Sequential(
            nn.Conv2d(num, num // 2, kernel_size=1, padding=0, stride=1),
            nn.ReLU(),
            nn.Conv2d(num // 2, 1, kernel_size=1, padding=0, stride=1),
            nn.ReLU(),
        )

    def forward(self, input):
        return self.tail(self.lin(self.head(input)))


__all__ = [
    "Gamma_Predictor",
    "Illumination_Estimator",
    "L_net",
    "N_net",
    "R_net",
]
