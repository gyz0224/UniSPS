import contextlib
import importlib
import io
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import torch
import torch.nn as nn
from PIL import Image


class OrganizationTest(unittest.TestCase):
    def test_canonical_packages_export_owned_symbols(self):
        from datasets import UnpairedDehazeDataset as PackageUnpaired
        from datasets import get_training_set as package_training_set
        from datasets import validate_real_dataset_layout as package_real_validator
        from datasets.dehaze import UnpairedDehazeDataset
        from datasets.dehaze import validate_real_dataset_layout
        from datasets.loaders import get_training_set
        from loss import CLIPLoss as PackageClipLoss
        from loss import C_loss as PackageCLoss
        from loss.lowlight_clip import CLIPLoss
        from loss.lowlight_loss import C_loss
        from net import net as PackageNet
        from net.model import net as CanonicalNet

        self.assertIs(PackageNet, CanonicalNet)
        self.assertIs(PackageUnpaired, UnpairedDehazeDataset)
        self.assertIs(package_training_set, get_training_set)
        self.assertIs(package_real_validator, validate_real_dataset_layout)
        self.assertIs(PackageCLoss, C_loss)
        self.assertIs(PackageClipLoss, CLIPLoss)

    def test_all_canonical_entrypoints_are_import_safe(self):
        names = (
            "train_lowlight",
            "train_lowlight_unpaired",
            "eval_lowlight",
            "measure",
            "measure_dehaze",
            "train_dehaze",
            "train_joint",
            "eval_dehaze",
            "scripts.flist",
            "scripts.validate_real_dataset",
            "tools.model_info",
        )
        for name in names:
            with self.subTest(name=name), contextlib.redirect_stdout(io.StringIO()) as output:
                module = importlib.import_module(name)
                self.assertTrue(callable(module.main))
                self.assertEqual(output.getvalue(), "")

    def test_joint_progress_reports_cycle_speed_and_eta(self):
        from train_joint import _format_progress

        message = _format_progress(
            joint_iteration=25,
            maximum=100,
            starting_iteration=20,
            epoch=2,
            step_seconds=2.5,
            elapsed_seconds=50.0,
            source_name="real",
            retain=0.125,
            metrics={"generator_total": 1.25},
        )

        self.assertIn("iter=25/100 (25.00%)", message)
        self.assertIn("epoch=2", message)
        self.assertIn("cycle=2.5s", message)
        self.assertIn("avg=10.0s/iter", message)
        self.assertIn("eta=12m30s", message)
        self.assertIn("source=real", message)
        self.assertIn("retain=0.125", message)
        self.assertIn("generator_total=1.25", message)

    def test_real_dataset_validator_defaults_match_canonical_layout(self):
        from scripts.validate_real_dataset import parse_args

        args = parse_args([])
        self.assertEqual(args.clean, Path("dataset/real/clear"))
        self.assertEqual(args.hazy, Path("dataset/real/hazy"))
        self.assertEqual(args.clean_masks, Path("dataset/real/masks/clear"))
        self.assertEqual(args.hazy_masks, Path("dataset/real/masks/hazy"))
        self.assertEqual((args.expected_clean, args.expected_hazy), (3577, 2902))

    def test_lowlight_parser_defaults_match_local_dataset_tree(self):
        from eval_lowlight import parse_args as parse_eval
        from eval_lowlight import resolve_inference as resolve_eval
        from measure import parse_args as parse_measure
        from measure import resolve_measurement
        from train_lowlight import parse_args as parse_paired
        from train_lowlight_unpaired import parse_args as parse_unpaired

        paired = parse_paired([])
        self.assertEqual(
            (paired.batchSize, paired.nEpochs, paired.threads, paired.decay),
            (1, 300, 0, 300),
        )
        self.assertEqual(paired.loss_weights, [1, 0.1, 0.1, 0.5])
        self.assertEqual(paired.data_train, "dataset/LOLv1/Train/input")
        self.assertEqual(paired.data_val, "dataset/LOLv1/Test/input")
        self.assertEqual(paired.reference_val, "dataset/LOLv1/Test/target")
        self.assertEqual(paired.save_folder, "weights/LOLv1")
        self.assertEqual(paired.logroot, "logs/LOLv1")

        unpaired = parse_unpaired([])
        self.assertEqual(
            (unpaired.batchSize, unpaired.nEpochs, unpaired.threads, unpaired.decay),
            (1, 300, 4, 100),
        )
        self.assertEqual(unpaired.data_train, "dataset/LOLv1/Train/input")
        self.assertEqual(unpaired.data_val, "dataset/LOLv1/Test/input")
        self.assertEqual(unpaired.weights_dir, "weights/metrics_weights")

        native = {
            "lolv1": (
                "dataset/LOLv1/Test/input",
                "dataset/LOLv1/Test/target",
                "lolv1_test",
                "same-name",
            ),
            "lolv2real": (
                "dataset/LOLv2/Real_captured/Test/Low",
                "dataset/LOLv2/Real_captured/Test/Normal",
                "lolv2real_test",
                "same-name",
            ),
            "sice": (
                "dataset/SICE/Test/image",
                "dataset/SICE/Test/label",
                "sice_test",
                "sice",
            ),
        }
        for experiment, (data_test, reference, dataset_key, pairing) in native.items():
            with self.subTest(experiment=experiment):
                evaluation_args = parse_eval(["--experiment", experiment])
                evaluation = resolve_eval(evaluation_args)
                self.assertEqual(evaluation_args.testBatchSize, 1)
                self.assertIsNone(evaluation_args.data_test)
                self.assertEqual(evaluation.data_test, Path(data_test))
                self.assertEqual(
                    evaluation.model, Path(f"weights/{experiment}.pth")
                )
                self.assertEqual(
                    evaluation.output,
                    Path(
                        f"runs/{experiment}/results/lowlight/pretrained/"
                        f"{dataset_key}"
                    ),
                )

                measurement_args = parse_measure(["--experiment", experiment])
                measurement = resolve_measurement(measurement_args)
                self.assertEqual(
                    measurement.image_source,
                    (
                        f"runs/{experiment}/results/lowlight/pretrained/"
                        f"{dataset_key}/I/*"
                    ),
                )
                self.assertEqual(measurement.label_dir, Path(reference))
                self.assertEqual(measurement.pairing, pairing)
                self.assertEqual(
                    measurement.metrics_output,
                    Path(f"runs/{experiment}/metrics/lowlight/pretrained.json"),
                )

        for path in (
            paired.data_train,
            paired.data_val,
            paired.reference_val,
            "weights/lolv1.pth",
            "weights/lolv2real.pth",
            "weights/sice.pth",
        ):
            self.assertTrue(Path(path).exists(), path)

    def test_experiment_layout_isolates_stage_chain_and_teacher(self):
        from training.experiment import (
            ExperimentLayout,
            ensure_new_training_output,
        )
        from train_dehaze import configure_experiment as configure_dehaze
        from train_dehaze import parse_args as parse_dehaze
        from train_joint import configure_experiment as configure_joint
        from train_joint import parse_args as parse_joint

        lolv2 = ExperimentLayout("lolv2real")
        self.assertEqual(lolv2.initial_weight, Path("weights/lolv2real.pth"))
        self.assertEqual(
            lolv2.checkpoint(3),
            Path("runs/lolv2real/checkpoints/stage3_real/latest.pth"),
        )
        self.assertEqual(
            lolv2.dehaze_result("stage4-hsts-real"),
            Path("runs/lolv2real/results/dehaze/stage4/hsts_real"),
        )

        dehaze_args = parse_dehaze(
            [
                "--experiment",
                "sice",
                "--config",
                "configs/dehaze_its.yaml",
                "--stage",
                "2",
            ]
        )
        dehaze_config = {"stage": 2}
        with mock.patch("train_dehaze.ensure_new_training_output"):
            configure_dehaze(dehaze_args, dehaze_config)
        self.assertEqual(
            dehaze_config["initial_checkpoint"],
            "runs/sice/checkpoints/stage1_warmup/latest.pth",
        )
        self.assertEqual(
            dehaze_config["output_dir"],
            "runs/sice/checkpoints/stage2_its",
        )

        joint_args = parse_joint(
            ["--experiment", "sice", "--config", "configs/joint.yaml"]
        )
        joint_config = {"stage": 4}
        with mock.patch("train_joint.ensure_new_training_output"):
            configure_joint(joint_args, joint_config)
        self.assertEqual(joint_config["teacher_checkpoint"], "weights/sice.pth")
        self.assertEqual(
            joint_config["initial_checkpoint"],
            "runs/sice/checkpoints/stage3_real/latest.pth",
        )
        self.assertEqual(
            joint_config["output_dir"],
            "runs/sice/checkpoints/stage4_joint",
        )
        self.assertEqual(joint_config["lowlight_data"], "dataset/SICE/Train")

        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            (output / "iter_0002000.pth").touch()
            with self.assertRaisesRegex(FileExistsError, "Use --resume"):
                ensure_new_training_output(output, resume=None)
            ensure_new_training_output(output, resume=str(output / "resume.pth"))

    def test_dehaze_eval_preset_uses_experiment_paths_and_large_image_tiling(self):
        from eval_dehaze import parse_args, resolve_inference

        spec = resolve_inference(
            parse_args(
                [
                    "--experiment",
                    "lolv2real",
                    "--stage",
                    "stage4-hsts-real",
                ]
            )
        )
        self.assertEqual(spec.config, Path("configs/dehaze_real.yaml"))
        self.assertEqual(
            spec.checkpoint,
            Path("runs/lolv2real/checkpoints/stage4_joint/latest.pth"),
        )
        self.assertEqual(
            spec.output,
            Path("runs/lolv2real/results/dehaze/stage4/hsts_real"),
        )
        self.assertEqual((spec.tile_size, spec.tile_overlap), (1024, 128))

    def test_dehaze_eval_resumes_and_recomputes_invalid_outputs(self):
        from eval_dehaze import _pending_jobs

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "input"
            output = root / "output"
            source.mkdir()
            output.mkdir()
            first = source / "first.png"
            second = source / "second.png"
            Image.new("RGB", (12, 8)).save(first)
            Image.new("RGB", (10, 6)).save(second)
            Image.new("RGB", (12, 8)).save(output / first.name)
            Image.new("RGB", (9, 6)).save(output / second.name)

            jobs, skipped = _pending_jobs(
                [first, second], source, output, overwrite=False
            )
            self.assertEqual(skipped, 1)
            self.assertEqual(jobs, [(second, output / second.name)])

            jobs, skipped = _pending_jobs(
                [first, second], source, output, overwrite=True
            )
            self.assertEqual(skipped, 0)
            self.assertEqual(
                jobs,
                [(first, output / first.name), (second, output / second.name)],
            )

    def test_dehaze_eval_uses_amp_only_for_large_images_or_oom(self):
        from eval_dehaze import _adaptive_inference

        model = nn.Identity()
        device = torch.device("cuda")
        tensor = torch.rand(1, 3, 4, 4)
        prediction = torch.rand_like(tensor)

        with mock.patch(
            "eval_dehaze._tensor", return_value=tensor
        ), mock.patch(
            "eval_dehaze._forward", return_value=prediction
        ) as forward:
            result, mode = _adaptive_inference(
                model,
                Path("large.png"),
                device,
                image_pixels=3_000_000,
                amp_pixel_threshold=3_000_000,
            )
        self.assertIs(result, prediction)
        self.assertEqual(mode, "amp-large")
        forward.assert_called_once_with(model, tensor, device, use_amp=True)

        oom = torch.OutOfMemoryError("test OOM")
        with mock.patch(
            "eval_dehaze._tensor", return_value=tensor
        ), mock.patch(
            "eval_dehaze._forward", side_effect=[oom, prediction]
        ) as forward, mock.patch("eval_dehaze._clear_cuda_memory") as clear:
            result, mode = _adaptive_inference(
                model,
                Path("small.png"),
                device,
                image_pixels=100,
                amp_pixel_threshold=3_000_000,
            )
        self.assertIs(result, prediction)
        self.assertEqual(mode, "amp-oom")
        self.assertEqual(
            [call.kwargs["use_amp"] for call in forward.call_args_list],
            [False, True],
        )
        clear.assert_called_once_with(model)

    def test_dehaze_eval_tiling_preserves_size_and_blends_identity(self):
        from eval_dehaze import _adaptive_inference, _tensor

        class IdentityTiledModel(nn.Module):
            def atmosphere_estimator(self, image):
                return image.mean(dim=(-2, -1), keepdim=True)

            def forward(self, image, atmosphere=None):
                self.assert_atmosphere(atmosphere)
                return image

            @staticmethod
            def assert_atmosphere(atmosphere):
                if atmosphere is None or atmosphere.shape != (1, 3, 1, 1):
                    raise AssertionError("global atmosphere was not supplied")

        with tempfile.TemporaryDirectory() as directory:
            image_path = Path(directory) / "large.png"
            Image.new("RGB", (19, 13), color=(17, 83, 191)).save(image_path)
            expected = _tensor(image_path, torch.device("cpu"))
            result, mode = _adaptive_inference(
                IdentityTiledModel(),
                image_path,
                torch.device("cpu"),
                image_pixels=19 * 13,
                amp_pixel_threshold=3_000_000,
                tile_size=8,
                tile_overlap=2,
            )

        self.assertEqual(mode, "fp32-tiled")
        self.assertEqual(result.shape, expected.shape)
        torch.testing.assert_close(result, expected)

    def test_cpu_sampling_and_losses_follow_original_contracts(self):
        from loss.lowlight_loss import L_exp, L_spa
        from training import lowlight_sampling

        image = torch.rand(2, 3, 16, 18)
        lowlight_sampling.operation_seed_counter = 0
        first, second = lowlight_sampling.generate_mask_pair(image)
        self.assertEqual(first.sum().item(), 2 * 8 * 9)
        self.assertEqual(second.sum().item(), 2 * 8 * 9)
        self.assertFalse(torch.logical_and(first, second).any())
        self.assertEqual(
            lowlight_sampling.generate_subimages(image, first).shape,
            (2, 3, 8, 9),
        )
        self.assertTrue(torch.isfinite(L_exp(4, 0.5)(image)))
        self.assertTrue(torch.isfinite(L_spa()(image, image * 0.9)).all())

    def test_shared_lowlight_trainer_runs_one_complete_update(self):
        from tests.helpers import tiny_model
        from training.lowlight_trainer import LowlightTrainer, LowlightTrainerConfig

        class DummyClipLoss(nn.Module):
            def forward(self, low, enhanced):
                return (low.mean() - enhanced.mean()).abs(), enhanced.new_zeros(())

        torch.manual_seed(71)
        model = tiny_model()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)
        trainer = LowlightTrainer(
            model,
            optimizer,
            DummyClipLoss(),
            LowlightTrainerConfig(light_patch=16),
            torch.device("cpu"),
        )
        metrics = trainer.train_step(torch.rand(1, 3, 64, 64))
        self.assertEqual(
            set(metrics),
            {
                "total",
                "consistency",
                "reconstruction",
                "prior",
                "lowlight",
                "sasw",
                "semantic",
                "iqa",
            },
        )
        self.assertTrue(
            all(torch.isfinite(torch.tensor(value)) for value in metrics.values())
        )


if __name__ == "__main__":
    unittest.main()
