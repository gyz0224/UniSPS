import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
from PIL import Image


class LowlightMetricsTest(unittest.TestCase):
    def test_niqe_receives_zero_to_one_rgb_tensor(self):
        import torch

        from metrics.lpips_evaluator import calculate_niqe

        class RecordingNiqe(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.seen = None

            def forward(self, image):
                self.seen = image.detach().cpu()
                return image.mean()

        model = RecordingNiqe()
        image = np.full((16, 12, 3), 255, dtype=np.uint8)

        score = calculate_niqe(image, model, "cpu")

        self.assertEqual(tuple(model.seen.shape), (1, 3, 16, 12))
        self.assertEqual(model.seen.dtype, torch.float32)
        self.assertAlmostEqual(float(model.seen.min()), 1.0)
        self.assertAlmostEqual(float(model.seen.max()), 1.0)
        self.assertAlmostEqual(score, 1.0)

    def test_lowlight_directory_adds_niqe_at_native_prediction_size(self):
        import torch

        from metrics.lpips_evaluator import evaluate_lowlight_directory

        class ZeroLpips(torch.nn.Module):
            def forward(self, reference, prediction):
                return (reference - prediction).abs().mean(
                    dim=(1, 2, 3), keepdim=True
                )

        class RecordingNiqe(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.shapes = []

            def forward(self, image):
                self.shapes.append(tuple(image.shape))
                return image.mean()

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            predictions = root / "predictions"
            references = root / "references"
            predictions.mkdir()
            references.mkdir()
            Image.new("RGB", (20, 18), color=(128, 128, 128)).save(
                predictions / "sample.png"
            )
            Image.new("RGB", (16, 16), color=(100, 100, 100)).save(
                references / "sample.png"
            )
            niqe_model = RecordingNiqe()

            with mock.patch(
                "metrics.lpips_evaluator.build_lpips_model",
                return_value=ZeroLpips(),
            ), mock.patch(
                "metrics.lpips_evaluator.build_niqe_model",
                return_value=niqe_model,
            ):
                psnr, ssim, lpips, niqe = evaluate_lowlight_directory(
                    str(predictions / "*"), references, "cpu"
                )

        self.assertTrue(
            all(np.isfinite(value) for value in (psnr, ssim, lpips, niqe))
        )
        self.assertEqual(niqe_model.shapes, [(1, 3, 18, 20)])
        self.assertAlmostEqual(niqe, 128.0 / 255.0, places=6)

    def test_measure_json_contains_niqe(self):
        from measure import main

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            prediction = root / "sample.png"
            reference_dir = root / "references"
            output = root / "metrics.json"
            reference_dir.mkdir()
            Image.new("RGB", (16, 16)).save(prediction)

            with mock.patch(
                "measure.metrics",
                return_value=(20.0, 0.8, 0.2, 3.5),
            ):
                result = main(
                    [
                        "--experiment",
                        "lolv1",
                        "--im_dir",
                        str(prediction),
                        "--label_dir",
                        str(reference_dir),
                        "--device",
                        "cpu",
                        "--metrics-output",
                        str(output),
                    ]
                )

            payload = json.loads(output.read_text(encoding="utf-8"))

        self.assertEqual(result, 0)
        self.assertEqual(payload["niqe"], 3.5)


if __name__ == "__main__":
    unittest.main()
