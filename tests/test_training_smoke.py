import io
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path

import torch

from net.depth import DepthEstimationNet
from net.discriminator import build_dehaze_discriminators
from net.rehaze import HazeRefineNet
from tests.helpers import random_batch, tiny_model
from training.dehaze_trainer import (
    DehazeTrainer,
    TrainerConfig,
    build_dehaze_optimizers,
    configure_stage,
    load_training_checkpoint,
    save_training_checkpoint,
)


class TrainingSmokeTest(unittest.TestCase):
    def test_two_iterations_have_exactly_one_depth_step_each(self):
        torch.manual_seed(5)
        model = tiny_model()
        configure_stage(model, 1)
        depth = DepthEstimationNet(channels=(8, 12, 16))
        refine = HazeRefineNet(channels=8)
        d_clear, d_hazy = build_dehaze_discriminators(base_channels=8, layers=2)
        config = TrainerConfig()
        optimizers = build_dehaze_optimizers(
            model, depth, refine, d_clear, d_hazy, config
        )
        trainer = DehazeTrainer(
            model, depth, refine, d_clear, d_hazy, optimizers, config
        )

        depth_steps = 0
        original_step = optimizers["depth"].step

        def counted_step(*args, **kwargs):
            nonlocal depth_steps
            depth_steps += 1
            return original_step(*args, **kwargs)

        optimizers["depth"].step = counted_step
        batch = random_batch()
        for _ in range(2):
            metrics = trainer.train_step(batch)
            self.assertTrue(
                all(torch.isfinite(torch.tensor(value)) for value in metrics.values())
            )
        self.assertEqual(trainer.iteration, 2)
        self.assertEqual(depth_steps, 2)

    def test_checkpoint_round_trip_restores_complete_train_state_without_clip(self):
        torch.manual_seed(9)
        model = tiny_model()
        configure_stage(model, 2)
        depth = DepthEstimationNet(channels=(8, 12, 16))
        refine = HazeRefineNet(channels=8)
        d_clear, d_hazy = build_dehaze_discriminators(base_channels=8, layers=2)
        config = TrainerConfig()
        optimizers = build_dehaze_optimizers(
            model, depth, refine, d_clear, d_hazy, config
        )
        schedulers = {
            name: torch.optim.lr_scheduler.StepLR(optimizer, step_size=1)
            for name, optimizer in optimizers.items()
        }
        trainer = DehazeTrainer(
            model, depth, refine, d_clear, d_hazy, optimizers, config
        )
        trainer.train_step(random_batch())
        for scheduler in schedulers.values():
            scheduler.step()

        with tempfile.TemporaryDirectory() as directory:
            checkpoint_path = Path(directory) / "round_trip.pth"
            save_training_checkpoint(
                checkpoint_path,
                model,
                depth,
                refine,
                d_clear,
                d_hazy,
                optimizers,
                schedulers,
                stage=2,
                iteration=trainer.iteration,
                epoch=3,
                config={"name": "round-trip"},
            )
            serialized = torch.load(checkpoint_path, map_location="cpu")
            self.assertFalse(
                any(key.startswith("sem_net.") for key in serialized["model"])
            )

            restored_model = tiny_model()
            configure_stage(restored_model, 2)
            restored_depth = DepthEstimationNet(channels=(8, 12, 16))
            restored_refine = HazeRefineNet(channels=8)
            restored_d_clear, restored_d_hazy = build_dehaze_discriminators(
                base_channels=8, layers=2
            )
            restored_optimizers = build_dehaze_optimizers(
                restored_model,
                restored_depth,
                restored_refine,
                restored_d_clear,
                restored_d_hazy,
                config,
            )
            restored_schedulers = {
                name: torch.optim.lr_scheduler.StepLR(optimizer, step_size=1)
                for name, optimizer in restored_optimizers.items()
            }
            with redirect_stdout(io.StringIO()):
                loaded = load_training_checkpoint(
                    checkpoint_path,
                    model=restored_model,
                    depth_net=restored_depth,
                    refine_net=restored_refine,
                    d_clear=restored_d_clear,
                    d_hazy=restored_d_hazy,
                    optimizers=restored_optimizers,
                    schedulers=restored_schedulers,
                )

        self.assertEqual(loaded["stage"], 2)
        self.assertEqual(loaded["iteration"], 1)
        self.assertEqual(loaded["epoch"], 3)
        self.assertEqual(loaded["config"], {"name": "round-trip"})
        for expected, actual in zip(
            model.dehaze_head.parameters(), restored_model.dehaze_head.parameters()
        ):
            self.assertTrue(torch.equal(expected, actual))
        for name in optimizers:
            expected_optimizer = optimizers[name].state_dict()
            actual_optimizer = restored_optimizers[name].state_dict()
            self.assertEqual(
                expected_optimizer["param_groups"], actual_optimizer["param_groups"]
            )
            self.assertEqual(
                set(expected_optimizer["state"]), set(actual_optimizer["state"])
            )
            for parameter_id, expected_state in expected_optimizer["state"].items():
                actual_state = actual_optimizer["state"][parameter_id]
                self.assertEqual(set(expected_state), set(actual_state))
                for key, expected_value in expected_state.items():
                    actual_value = actual_state[key]
                    if torch.is_tensor(expected_value):
                        self.assertTrue(torch.equal(expected_value, actual_value))
                    else:
                        self.assertEqual(expected_value, actual_value)
            self.assertEqual(
                schedulers[name].state_dict(), restored_schedulers[name].state_dict()
            )


if __name__ == "__main__":
    unittest.main()
