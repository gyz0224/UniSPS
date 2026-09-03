import unittest

import torch

from net.dehaze import AtmosphereEstimator
from net.rehaze import HazeRefineNet
from tests.helpers import tiny_model


class DehazeForwardTest(unittest.TestCase):
    def test_forward_shapes_ranges_and_finiteness(self):
        torch.manual_seed(4)
        model = tiny_model().eval()
        hazy = torch.rand(2, 3, 17, 19)
        with torch.no_grad():
            clean = model(hazy, task="dehaze")
            aux = model(hazy, task="dehaze", return_aux=True)
        self.assertEqual(clean.shape, hazy.shape)
        self.assertEqual(aux["transmission"].shape, (2, 1, 17, 19))
        self.assertEqual(aux["beta"].shape, (2, 1, 1, 1))
        self.assertEqual(aux["depth_from_haze"].shape, (2, 1, 17, 19))
        self.assertEqual(aux["atmosphere"].shape, (2, 3, 1, 1))
        for value in aux.values():
            self.assertTrue(torch.isfinite(value).all())
        self.assertGreaterEqual(aux["clean"].min().item(), 0.0)
        self.assertLessEqual(aux["clean"].max().item(), 1.0)
        self.assertGreaterEqual(aux["transmission"].min().item(), 0.05)
        self.assertLessEqual(aux["transmission"].max().item(), 0.95)
        self.assertGreaterEqual(aux["beta"].min().item(), 0.6)
        self.assertLessEqual(aux["beta"].max().item(), 1.8)

    def test_atmosphere_modes_and_invalid_task(self):
        image = torch.rand(1, 3, 8, 9)
        for mode in ("max", "dcp"):
            estimator = AtmosphereEstimator(mode=mode, window_size=3)
            atmosphere = estimator(image)
            transmission = estimator.estimate_transmission(image, atmosphere)
            self.assertEqual(atmosphere.shape, (1, 3, 1, 1))
            self.assertEqual(transmission.shape, (1, 1, 8, 9))
            self.assertTrue(torch.isfinite(transmission).all())
        with self.assertRaisesRegex(ValueError, "Unsupported task"):
            tiny_model()(image, task="unknown")

    def test_refinement_residual_is_bounded(self):
        network = HazeRefineNet(channels=8, residual_scale=0.1)
        clean = torch.rand(1, 3, 8, 8)
        coarse = torch.full_like(clean, 0.5)
        transmission = torch.rand(1, 1, 8, 8)
        refined = network(clean, coarse, transmission)
        self.assertLessEqual((refined - coarse).abs().max().item(), 0.1 + 1e-6)

    def test_deployment_wrapper_excludes_task_and_training_only_modules(self):
        deployment = tiny_model().to_dehaze_inference().eval()
        self.assertFalse(hasattr(deployment.N_net, "output"))
        for excluded in ("L_net", "R_net", "Gamma_Predictor", "depth_net", "refine_net"):
            self.assertFalse(hasattr(deployment, excluded))
        with torch.no_grad():
            hazy = torch.rand(1, 3, 9, 11)
            clean = deployment(hazy)
            atmosphere = deployment.atmosphere_estimator(hazy)
            clean_with_atmosphere = deployment(
                hazy,
                atmosphere=atmosphere,
            )
        self.assertEqual(clean.shape, (1, 3, 9, 11))
        torch.testing.assert_close(clean_with_atmosphere, clean)


if __name__ == "__main__":
    unittest.main()
