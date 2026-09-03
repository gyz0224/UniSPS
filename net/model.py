"""Top-level checkpoint-compatible SPS-Net multi-task model."""

import torch
import torch.nn as nn

from net.lowlight import (
    Gamma_Predictor,
    Illumination_Estimator,
    L_net,
    N_net,
    R_net,
)
from net.semantic import SemanticPriorNet


class net(nn.Module):
    """Preserve the historical class name while exposing explicit task routing."""

    def __init__(self, dehaze_config=None, semantic_net=None):
        super().__init__()
        self.L_net = L_net(num=48)
        self.R_net = R_net(num=64, inp_channels=68)
        self.N_net = N_net(num=64)
        self.sem_net = semantic_net if semantic_net is not None else SemanticPriorNet()
        self.illp = Illumination_Estimator(n_fea_middle=64)
        self.Gamma_Predictor = Gamma_Predictor(in_channel=1, num=32)

        from net.dehaze import AtmosphereEstimator, DehazeConfig, DehazeParameterHead

        self.dehaze_config = DehazeConfig.from_value(dehaze_config)
        self.atmosphere_estimator = AtmosphereEstimator(
            mode=self.dehaze_config.atmosphere_mode,
            window_size=self.dehaze_config.dcp_window,
            top_percent=self.dehaze_config.dcp_top_percent,
            omega=self.dehaze_config.dcp_omega,
            eps=self.dehaze_config.eps,
        )
        self.dehaze_head = DehazeParameterHead(self.dehaze_config)

    def extract_semantics(self, input):
        return self.sem_net(input)

    def extract_global_semantics(self, input):
        """Return normalized global CLIP embeddings without another CLIP copy."""
        if not hasattr(self.sem_net, "encode_global"):
            raise TypeError("Configured semantic_net does not implement encode_global")
        return self.sem_net.encode_global(input)

    def set_dehaze_physics(
        self,
        beta_min,
        beta_max,
        transmission_min,
        transmission_max,
        atmosphere_mode=None,
    ):
        """Switch indoor/real ranges between sequential joint-training batches."""
        self.dehaze_head.set_physical_ranges(
            beta_min, beta_max, transmission_min, transmission_max
        )
        if atmosphere_mode is not None:
            if atmosphere_mode not in {"max", "dcp"}:
                raise ValueError("atmosphere_mode must be 'max' or 'dcp'")
            self.atmosphere_estimator.mode = atmosphere_mode

    def to_dehaze_inference(self):
        """Return a deployment wrapper containing no low-light auxiliaries."""
        from net.dehaze import DehazeInferenceModel

        return DehazeInferenceModel(
            semantic_net=self.sem_net,
            shared_encoder=self.N_net,
            atmosphere_estimator=self.atmosphere_estimator,
            parameter_head=self.dehaze_head,
            eps=self.dehaze_config.eps,
        )

    def _forward_lowlight(self, input, sem_feats=None):
        if sem_feats is None:
            sem_feats = self.sem_net(input)
        x_img, x_feat = self.N_net(input)
        y, _ = self.illp(x_img)
        L = self.L_net(x_img)
        noisy_R = input / L
        R = self.R_net(
            feature_x=x_feat,
            noisy_R=noisy_R.detach(),
            L=L,
            fea=y,
            pre_R=x_img,
            sem_feats=sem_feats,
        )
        alpha = self.Gamma_Predictor(L)
        I = torch.pow(L, alpha) * R
        return L, R, x_img, I

    def _forward_dehaze(self, input, sem_feats=None, return_aux=False):
        """Run dehazing for an RGB tensor ``[B,3,H,W]`` in ``[0,1]``."""
        from net.dehaze import depth_from_transmission, recover_clean_image

        if input.ndim != 4 or input.shape[1] != 3:
            raise ValueError(
                f"dehaze input must have shape [B,3,H,W], got {tuple(input.shape)}"
            )
        if sem_feats is None:
            sem_feats = self.sem_net(input)
        x_feat = self.N_net.encode(input)
        atmosphere = self.atmosphere_estimator(input)
        transmission, beta = self.dehaze_head(
            input, x_feat, atmosphere, sem_feats
        )
        clean = recover_clean_image(
            input, transmission, atmosphere, eps=self.dehaze_config.eps
        )
        depth = depth_from_transmission(
            transmission, beta, eps=self.dehaze_config.eps
        )
        if return_aux:
            return {
                "clean": clean,
                "transmission": transmission,
                "beta": beta,
                "depth_from_haze": depth,
                "atmosphere": atmosphere,
            }
        return clean

    def forward(self, input, sem_feats=None, task="lowlight", return_aux=False):
        """Run the legacy low-light task or explicit dehaze task."""
        if task == "lowlight":
            return self._forward_lowlight(input, sem_feats=sem_feats)
        if task == "dehaze":
            return self._forward_dehaze(
                input, sem_feats=sem_feats, return_aux=return_aux
            )
        raise ValueError(
            f"Unsupported task {task!r}; expected 'lowlight' or 'dehaze'"
        )


SPSNet = net

__all__ = ["SPSNet", "net"]
