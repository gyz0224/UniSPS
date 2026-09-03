"""Frozen CLIP semantic and image-quality objectives for low-light training."""

import clip
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms


class CLIPLoss(nn.Module):
    def __init__(self, device=None):
        super().__init__()
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model, _ = clip.load("ViT-B/32", device=self.device)
        self.model.eval()
        for parameter in self.model.parameters():
            parameter.requires_grad = False
        self.register_buffer(
            "mean",
            torch.tensor([0.48145466, 0.4578275, 0.40821073]).view(1, 3, 1, 1),
        )
        self.register_buffer(
            "std",
            torch.tensor([0.26862954, 0.26130258, 0.27577711]).view(1, 3, 1, 1),
        )
        self.transform = transforms.Compose([])
        good_prompts = [
            "High quality image",
            "Clear photo",
            "Well lit",
            "Sharp details",
            "Realistic colors",
        ]
        bad_prompts = [
            "Noisy image",
            "Dark photo",
            "Blurry",
            "Low quality",
            "Grainy",
            "Underexposed",
        ]
        with torch.no_grad():
            good_tokens = clip.tokenize(good_prompts).to(self.device)
            bad_tokens = clip.tokenize(bad_prompts).to(self.device)
            good_features = self.model.encode_text(good_tokens)
            bad_features = self.model.encode_text(bad_tokens)
            good_features = good_features / good_features.norm(dim=-1, keepdim=True)
            bad_features = bad_features / bad_features.norm(dim=-1, keepdim=True)
            average_good = good_features.mean(dim=0, keepdim=True)
            average_bad = bad_features.mean(dim=0, keepdim=True)
            average_good = average_good / average_good.norm(dim=-1, keepdim=True)
            average_bad = average_bad / average_bad.norm(dim=-1, keepdim=True)
            text_features = torch.cat([average_good, average_bad], dim=0)
        self.register_buffer("text_features", text_features)

    def _preprocess(self, image):
        image = F.interpolate(
            image, size=(224, 224), mode="bicubic", align_corners=False
        )
        return (image - self.mean) / self.std

    def forward(self, img_low, img_en):
        model_device = self.model.visual.conv1.weight.device
        img_low = img_low.to(model_device)
        img_en = img_en.to(model_device)
        image_low_features = self.model.encode_image(self._preprocess(img_low))
        image_enhanced_features = self.model.encode_image(self._preprocess(img_en))
        image_low_features = image_low_features / image_low_features.norm(
            dim=-1, keepdim=True
        )
        image_enhanced_features = image_enhanced_features / image_enhanced_features.norm(
            dim=-1, keepdim=True
        )
        cosine = torch.clamp(
            F.cosine_similarity(
                image_low_features, image_enhanced_features, dim=-1
            ),
            -1.0,
            1.0,
        )
        loss_sem = (1 - cosine).mean()
        logits = (
            self.model.logit_scale.exp()
            * image_enhanced_features
            @ self.text_features.t()
        )
        loss_iqa = (1 - logits.softmax(dim=-1)[:, 0]).mean()
        return loss_sem, loss_iqa


__all__ = ["CLIPLoss"]
