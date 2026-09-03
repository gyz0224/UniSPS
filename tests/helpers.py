import torch
import torch.nn as nn

from net.model import net


class DummySemanticPrior(nn.Module):
    """Download-free semantic prior with the production feature contracts."""

    def forward(self, image):
        batch = image.shape[0]
        deep = image.new_zeros((batch, 768, 2, 2))
        return [image, deep]

    def encode_global(self, image):
        return image.mean(dim=(-2, -1))


def tiny_model(use_semantics=False):
    return net(
        dehaze_config={
            "stage_channels": (16, 24, 32),
            "blocks_per_stage": (0, 0, 0),
            "decoder_blocks": 0,
            "use_semantics": use_semantics,
        },
        semantic_net=DummySemanticPrior(),
    )


def random_batch(batch_size=1, size=16):
    return {
        "clean": torch.rand(batch_size, 3, size, size),
        "hazy": torch.rand(batch_size, 3, size, size),
        "clean_ref": torch.rand(batch_size, 3, size, size),
        "hazy_ref": torch.rand(batch_size, 3, size, size),
        "clean_mask": torch.zeros(batch_size, 1, size, size),
        "hazy_mask": torch.zeros(batch_size, 1, size, size),
    }
