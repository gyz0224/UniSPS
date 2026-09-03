import unittest

import torch

from tests.helpers import tiny_model


class LowlightBackwardCompatibilityTest(unittest.TestCase):
    def test_default_and_explicit_task_are_identical(self):
        torch.manual_seed(3)
        model = tiny_model().eval()
        image = torch.rand(1, 3, 16, 16)
        with torch.no_grad():
            old_call = model(image)
            explicit_call = model(image, task="lowlight")
        self.assertEqual(len(old_call), 4)
        for old, explicit in zip(old_call, explicit_call):
            self.assertLess((old - explicit).abs().max().item(), 1e-6)

    def test_legacy_state_names_load_without_unexpected_keys(self):
        source = tiny_model().eval()
        legacy_state = {
            key: value
            for key, value in source.state_dict().items()
            if not key.startswith("dehaze_head.")
        }
        target = tiny_model().eval()
        incompatible = target.load_state_dict(legacy_state, strict=False)
        self.assertEqual(incompatible.unexpected_keys, [])
        self.assertTrue(incompatible.missing_keys)
        self.assertTrue(
            all(key.startswith("dehaze_head.") for key in incompatible.missing_keys)
        )
        for required in (
            "N_net.patch_embed.proj.weight",
            "N_net.encoder.0.attn.qkv.weight",
            "N_net.output.weight",
        ):
            self.assertIn(required, target.state_dict())


if __name__ == "__main__":
    unittest.main()
