import math
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image


class DehazeMetricsTest(unittest.TestCase):
    @staticmethod
    def _fade_regression_image() -> np.ndarray:
        rows, cols = np.indices((16, 16))
        image = np.empty((16, 16, 3), dtype=np.uint8)
        image[:, :, 0] = (rows * 17 + cols * 11) % 256
        image[:, :, 1] = (rows * 7 + cols * 19 + 23) % 256
        image[:, :, 2] = (rows * 29 + cols * 3 + 101) % 256
        return image

    def test_ciede2000_matches_published_reference_pairs(self):
        from metrics.dehaze_evaluator import calculate_ciede2000_lab

        lab1 = np.array(
            [
                [50.0000, 2.6772, -79.7751],
                [50.0000, 3.1571, -77.2803],
                [50.0000, 2.8361, -74.0200],
                [50.0000, -1.3802, -84.2814],
            ],
            dtype=np.float64,
        )
        lab2 = np.array(
            [
                [50.0000, 0.0000, -82.7485],
                [50.0000, 0.0000, -82.7485],
                [50.0000, 0.0000, -82.7485],
                [50.0000, 0.0000, -82.7485],
            ],
            dtype=np.float64,
        )

        actual = calculate_ciede2000_lab(lab1, lab2)
        np.testing.assert_allclose(
            actual,
            np.array([2.0425, 2.8615, 3.4412, 1.0000]),
            rtol=0,
            atol=5e-5,
        )

    def test_lpips_uses_minus_one_to_one_rgb_normalization(self):
        import torch

        from metrics.lpips_evaluator import calculate_lpips

        class MeanAbsoluteDistance(torch.nn.Module):
            def forward(self, reference, prediction):
                return (reference - prediction).abs().mean(
                    dim=(1, 2, 3),
                    keepdim=True,
                )

        prediction = np.zeros((16, 16, 3), dtype=np.uint8)
        reference = np.full((16, 16, 3), 255, dtype=np.uint8)
        score = calculate_lpips(
            prediction,
            reference,
            MeanAbsoluteDistance(),
            "cpu",
        )

        self.assertAlmostEqual(score, 2.0)

    def test_sice_lowlight_pairing_maps_each_exposure_to_scene_reference(self):
        from metrics.lpips_evaluator import resolve_lowlight_reference

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            references = root / "references"
            references.mkdir()
            Image.new("RGB", (16, 16)).save(references / "10.JPG")

            first = resolve_lowlight_reference(
                root / "10_1.JPG", references, pairing="sice"
            )
            second = resolve_lowlight_reference(
                root / "10_2.JPG", references, pairing="sice"
            )

        self.assertEqual(first.name, "10.JPG")
        self.assertEqual(second.name, "10.JPG")

    def test_lowlight_image_glob_includes_sice_jpg_outputs(self):
        from metrics.lpips_evaluator import resolve_image_files

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            Image.new("RGB", (16, 16)).save(root / "10_1.JPG")
            (root / "not-an-image.txt").write_text("ignore", encoding="utf-8")

            files = resolve_image_files(str(root / "*"))

        self.assertEqual([Path(item).name for item in files], ["10_1.JPG"])

    def test_sots_pairing_maps_haze_suffix_to_scene_gt(self):
        from metrics.dehaze_evaluator import pair_dehaze_images

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            predictions = root / "predictions"
            references = root / "references"
            predictions.mkdir()
            references.mkdir()
            for name in ("1400_1.png", "1400_10.png", "1401_2.jpg"):
                Image.new("RGB", (16, 16)).save(predictions / name)
            Image.new("RGB", (16, 16)).save(references / "1400.png")
            Image.new("RGB", (16, 16)).save(references / "1401.bmp")

            pairs = pair_dehaze_images(predictions, references, pairing="sots")

        self.assertEqual(
            [(prediction.name, reference.name) for prediction, reference in pairs],
            [
                ("1400_1.png", "1400.png"),
                ("1400_10.png", "1400.png"),
                ("1401_2.jpg", "1401.bmp"),
            ],
        )

    def test_sots_outdoor_pairing_maps_physical_parameters_to_scene_gt(self):
        from metrics.dehaze_evaluator import pair_dehaze_images

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            predictions = root / "predictions"
            references = root / "references"
            predictions.mkdir()
            references.mkdir()
            for name in (
                "0001_0.8_0.2.jpg",
                "0051_0.85_0.08.png",
                "0051_0.9_0.16.jpg",
            ):
                Image.new("RGB", (16, 16)).save(predictions / name)
            Image.new("RGB", (16, 16)).save(references / "0001.png")
            Image.new("RGB", (16, 16)).save(references / "0051.bmp")

            pairs = pair_dehaze_images(
                predictions,
                references,
                pairing="sots-outdoor",
            )

        self.assertEqual(
            [(prediction.name, reference.name) for prediction, reference in pairs],
            [
                ("0001_0.8_0.2.jpg", "0001.png"),
                ("0051_0.85_0.08.png", "0051.bmp"),
                ("0051_0.9_0.16.jpg", "0051.bmp"),
            ],
        )

    def test_ihaze_pairing_maps_hazy_suffix_to_gt_suffix(self):
        from metrics.dehaze_evaluator import pair_dehaze_images

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            predictions = root / "predictions"
            references = root / "references"
            predictions.mkdir()
            references.mkdir()
            Image.new("RGB", (16, 16)).save(
                predictions / "01_indoor_hazy.png"
            )
            Image.new("RGB", (16, 16)).save(
                references / "01_indoor_GT.jpg"
            )

            pairs = pair_dehaze_images(
                predictions,
                references,
                pairing="ihaze",
            )

        self.assertEqual(
            [(prediction.name, reference.name) for prediction, reference in pairs],
            [("01_indoor_hazy.png", "01_indoor_GT.jpg")],
        )

    def test_directory_metrics_are_finite_and_reject_size_mismatch(self):
        from metrics.dehaze_evaluator import evaluate_dehaze_directory

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            predictions = root / "predictions"
            references = root / "references"
            predictions.mkdir()
            references.mkdir()
            reference = np.full((16, 16, 3), 100, dtype=np.uint8)
            prediction = np.full((16, 16, 3), 110, dtype=np.uint8)
            Image.fromarray(reference).save(references / "1400.png")
            Image.fromarray(prediction).save(predictions / "1400_1.png")

            metrics = evaluate_dehaze_directory(
                predictions, references, pairing="sots"
            )
            self.assertEqual(metrics.samples, 1)
            self.assertIsNone(metrics.lpips)
            self.assertTrue(math.isfinite(metrics.psnr_db))
            self.assertTrue(math.isfinite(metrics.ssim))
            self.assertGreater(metrics.ciede2000, 0.0)

            Image.new("RGB", (15, 16)).save(predictions / "1400_1.png")
            with self.assertRaisesRegex(ValueError, "dimensions"):
                evaluate_dehaze_directory(
                    predictions, references, pairing="sots"
                )

    def test_fade_matches_matlab_aligned_regression_value(self):
        from metrics.fade import calculate_fade

        score, density_map = calculate_fade(
            self._fade_regression_image(),
            return_map=True,
        )

        self.assertEqual(density_map.shape, (2, 2))
        self.assertAlmostEqual(score, 0.1813003412312952, places=10)
        np.testing.assert_allclose(
            density_map,
            np.array(
                [
                    [0.215179855687064, 0.171280844331513],
                    [0.222215457146475, 0.134828067185145],
                ]
            ),
            rtol=0,
            atol=1e-10,
        )

    def test_fade_directory_metrics_need_no_reference(self):
        from metrics.dehaze_evaluator import evaluate_fade_directory

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            first = self._fade_regression_image()
            second = np.flip(first, axis=1)
            Image.fromarray(first).save(root / "a.png")
            Image.fromarray(second).save(root / "b.png")
            progress = []

            metrics = evaluate_fade_directory(
                root,
                progress=lambda index, total, path: progress.append(
                    (index, total, path.name)
                ),
            )

        self.assertEqual(metrics.samples, 2)
        self.assertTrue(math.isfinite(metrics.fade))
        self.assertGreater(metrics.fade, 0.0)
        self.assertEqual(progress, [(1, 2, "a.png"), (2, 2, "b.png")])

    def test_stage_presets_select_paired_or_fade_metrics(self):
        from measure_dehaze import parse_args, resolve_evaluation
        from training.experiment import DEHAZE_PRESETS

        def resolve(stage, *extra):
            return resolve_evaluation(
                parse_args(
                    ["--experiment", "lolv1", "--stage", stage, *extra]
                )
            )

        for stage, preset in DEHAZE_PRESETS.items():
            with self.subTest(stage=stage, invariant="paired-lpips"):
                evaluation = resolve(stage)
                has_reference = preset.reference_dir is not None
                self.assertEqual(preset.lpips, has_reference)
                self.assertEqual(evaluation.lpips, has_reference)

        stage2 = resolve("stage2")
        self.assertEqual(
            stage2.prediction,
            Path("runs/lolv1/results/dehaze/stage2/sots_indoor"),
        )
        self.assertEqual(
            stage2.reference, Path("dataset/eval/SOTS/indoor/gt")
        )
        self.assertEqual(stage2.pairing, "sots")
        self.assertEqual(stage2.mode, "full-reference")
        self.assertTrue(stage2.lpips)

        outdoor_predictions = {
            "stage2-outdoor": "runs/lolv1/results/dehaze/stage2/sots_outdoor",
            "stage3-outdoor": "runs/lolv1/results/dehaze/stage3/sots_outdoor",
            "stage4-outdoor": "runs/lolv1/results/dehaze/stage4/sots_outdoor",
        }
        for stage, prediction in outdoor_predictions.items():
            with self.subTest(stage=stage):
                outdoor = resolve(stage)
                self.assertEqual(outdoor.prediction, Path(prediction))
                self.assertEqual(
                    outdoor.reference,
                    Path("dataset/eval/SOTS/outdoor/gt"),
                )
                self.assertEqual(outdoor.pairing, "sots-outdoor")
                self.assertEqual(outdoor.mode, "full-reference")
                self.assertTrue(outdoor.lpips)

        hsts_predictions = {
            "stage2-hsts-synthetic": (
                "runs/lolv1/results/dehaze/stage2/hsts_synthetic",
                "full-reference",
            ),
            "stage2-hsts-real": (
                "runs/lolv1/results/dehaze/stage2/hsts_real",
                "fade",
            ),
            "stage3-hsts-synthetic": (
                "runs/lolv1/results/dehaze/stage3/hsts_synthetic",
                "full-reference",
            ),
            "stage3-hsts-real": (
                "runs/lolv1/results/dehaze/stage3/hsts_real",
                "fade",
            ),
            "stage4-hsts-synthetic": (
                "runs/lolv1/results/dehaze/stage4/hsts_synthetic",
                "full-reference",
            ),
            "stage4-hsts-real": (
                "runs/lolv1/results/dehaze/stage4/hsts_real",
                "fade",
            ),
        }
        for stage, (prediction, mode) in hsts_predictions.items():
            with self.subTest(stage=stage):
                hsts = resolve(stage)
                self.assertEqual(hsts.prediction, Path(prediction))
                self.assertEqual(hsts.pairing, "same-stem")
                self.assertEqual(hsts.mode, mode)
                if mode == "full-reference":
                    self.assertTrue(hsts.lpips)
                    self.assertEqual(
                        hsts.reference,
                        Path("dataset/eval/HSTS/synthetic/gt"),
                    )
                else:
                    self.assertFalse(hsts.lpips)
                    self.assertIsNone(hsts.reference)

        stage3 = resolve("stage3")
        self.assertEqual(
            stage3.prediction,
            Path("runs/lolv1/results/dehaze/stage3/sots_indoor"),
        )
        self.assertEqual(
            stage3.reference, Path("dataset/eval/SOTS/indoor/gt")
        )
        self.assertEqual(stage3.pairing, "sots")
        self.assertTrue(stage3.lpips)

        stage3_ihaze = resolve("stage3-ihaze")
        self.assertEqual(
            stage3_ihaze.prediction,
            Path("runs/lolv1/results/dehaze/stage3/ihaze"),
        )
        self.assertEqual(
            stage3_ihaze.reference,
            Path("dataset/eval/I-HAZE/gt"),
        )
        self.assertEqual(stage3_ihaze.pairing, "ihaze")
        self.assertEqual(stage3_ihaze.mode, "full-reference")
        self.assertTrue(stage3_ihaze.lpips)

        stage3_real = resolve("stage3-real")
        self.assertEqual(
            stage3_real.prediction,
            Path("runs/lolv1/results/dehaze/stage3/rtts"),
        )
        self.assertIsNone(stage3_real.reference)
        self.assertEqual(stage3_real.mode, "fade")
        self.assertFalse(stage3_real.lpips)

        stage3_real_with_reference = resolve(
            "stage3-real", "--reference", "custom/gt"
        )
        self.assertEqual(stage3_real_with_reference.mode, "full-reference")
        self.assertTrue(stage3_real_with_reference.lpips)

        stage3_real_custom = resolve(
            "stage3-real",
            "--reference",
            "/data/custom-rtts-gt",
        )
        self.assertEqual(
            stage3_real_custom.reference, Path("/data/custom-rtts-gt")
        )
        self.assertEqual(stage3_real_custom.pairing, "same-stem")
        self.assertEqual(stage3_real_custom.mode, "full-reference")


if __name__ == "__main__":
    unittest.main()
