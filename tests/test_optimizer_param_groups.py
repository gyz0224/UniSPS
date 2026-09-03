import unittest

import torch

from net.depth import DepthEstimationNet
from net.discriminator import build_dehaze_discriminators
from net.rehaze import HazeRefineNet
from tests.helpers import tiny_model
from training.dehaze_trainer import (
    TrainerConfig,
    assert_disjoint_optimizers,
    build_dehaze_optimizers,
    configure_stage,
)


class OptimizerParamGroupsTest(unittest.TestCase):
    def _components(self):
        model = tiny_model()
        depth = DepthEstimationNet(channels=(8, 12, 16))
        refine = HazeRefineNet(channels=8)
        discriminators = build_dehaze_discriminators(base_channels=8, layers=2)
        return model, depth, refine, discriminators

    def test_stages_and_optimizer_ownership(self):
        model, depth, refine, (d_clear, d_hazy) = self._components()
        configure_stage(model, 1)
        self.assertFalse(any(p.requires_grad for p in model.N_net.parameters()))
        self.assertTrue(all(p.requires_grad for p in model.dehaze_head.parameters()))
        optimizers = build_dehaze_optimizers(
            model, depth, refine, d_clear, d_hazy, TrainerConfig()
        )
        assert_disjoint_optimizers(optimizers)

        configure_stage(model, 2)
        self.assertTrue(any(p.requires_grad for p in model.N_net.encoder[-1].parameters()))
        self.assertFalse(any(p.requires_grad for p in model.N_net.output.parameters()))

        configure_stage(model, 4)
        self.assertTrue(any(p.requires_grad for p in model.L_net.parameters()))
        self.assertFalse(any(p.requires_grad for p in model.sem_net.parameters()))

    def test_duplicate_parameter_is_rejected(self):
        parameter = torch.nn.Parameter(torch.tensor(1.0))
        first = torch.optim.SGD([parameter], lr=0.1)
        second = torch.optim.SGD([parameter], lr=0.1)
        with self.assertRaisesRegex(ValueError, "shared by optimizers"):
            assert_disjoint_optimizers({"first": first, "second": second})


if __name__ == "__main__":
    unittest.main()
