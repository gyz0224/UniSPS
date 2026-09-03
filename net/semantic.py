"""Frozen CLIP semantic prior shared by low-light and dehaze tasks."""

from typing import Optional

import clip
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


class SemanticPriorNet(nn.Module):
    """Expose spatial and global ViT-B/16 semantics without training CLIP."""

    def __init__(self, device: Optional[str] = None):
        super().__init__()
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model, _ = clip.load("ViT-B/16", device=self.device)
        self.model.eval()
        for parameter in self.model.parameters():
            parameter.requires_grad = False
        self._hook_feats = {}
        self._register_hooks()
        self.register_buffer(
            "mean",
            torch.tensor([0.48145466, 0.4578275, 0.40821073]).view(1, 3, 1, 1),
            persistent=False,
        )
        self.register_buffer(
            "std",
            torch.tensor([0.26862954, 0.26130258, 0.27577711]).view(1, 3, 1, 1),
            persistent=False,
        )

    def _register_hooks(self):
        def save_activation(name):
            def hook(_, __, output):
                self._hook_feats[name] = output

            return hook

        self.model.visual.transformer.resblocks[9].register_forward_hook(
            save_activation("deep")
        )

    def forward(self, x):
        model_device = self.model.visual.conv1.weight.device
        x = x.to(model_device)
        x_resized = F.interpolate(
            x, size=(224, 224), mode="bilinear", align_corners=False
        )
        dtype = self.model.visual.conv1.weight.dtype
        x_resized = ((x_resized - self.mean) / self.std).to(dtype)
        self._hook_feats.clear()
        with torch.no_grad():
            self.model.encode_image(x_resized)

        batch = x.shape[0]
        deep = self._hook_feats.get("deep")

        def prepare_feat(feat):
            if feat is None:
                return None
            if feat.dim() == 3 and feat.shape[0] != batch and feat.shape[1] == batch:
                feat = feat.permute(1, 0, 2)
            feat = feat[:, 1:, :]
            tokens = feat.shape[1]
            height = width = int(tokens**0.5)
            if height * width != tokens:
                raise ValueError(
                    f"Unexpected token count {tokens} for ViT-B/16 feature map"
                )
            return rearrange(
                feat, "b (h w) c -> b c h w", h=height, w=width
            ).float()

        return [x.float(), prepare_feat(deep)]

    def encode_global(self, x, normalize=True):
        """Return global ``[B,D]`` CLIP features with gradients to ``x``."""
        model_device = self.model.visual.conv1.weight.device
        x = x.to(model_device)
        x = F.interpolate(x, size=(224, 224), mode="bilinear", align_corners=False)
        x = (x - self.mean) / self.std
        dtype = self.model.visual.conv1.weight.dtype
        embedding = self.model.encode_image(x.to(dtype)).float()
        if normalize:
            embedding = F.normalize(embedding, dim=-1, eps=1e-6)
        return embedding


__all__ = ["SemanticPriorNet"]
