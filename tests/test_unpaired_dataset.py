import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, call

import numpy as np
from PIL import Image

from datasets.dehaze import UnpairedDehazeDataset, validate_real_dataset_layout
from scripts.flist import build_flist, write_flist


class UnpairedDatasetTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.clean = self.root / "clean"
        self.hazy = self.root / "hazy"
        self.clean_masks = self.root / "clean_masks"
        self.hazy_masks = self.root / "hazy_masks"
        for directory in (
            self.clean,
            self.hazy,
            self.clean_masks,
            self.hazy_masks,
        ):
            directory.mkdir()
        for stem in ("scene1", "scene2"):
            image = np.zeros((12, 12, 3), dtype=np.uint8)
            image[:, :6, 0] = 255
            mask = np.zeros((12, 12), dtype=np.uint8)
            mask[:, :6] = 255
            Image.fromarray(image).save(self.clean / f"{stem}.png")
            Image.fromarray(mask).save(self.clean_masks / f"{stem}.png")
        for stem in ("scene1_1", "scene3_1"):
            image = np.zeros((12, 12, 3), dtype=np.uint8)
            image[:6, :, 1] = 255
            mask = np.zeros((12, 12), dtype=np.uint8)
            mask[:6, :] = 255
            Image.fromarray(image).save(self.hazy / f"{stem}.jpg")
            Image.fromarray(mask).save(self.hazy_masks / f"{stem}.jpg")

    def tearDown(self):
        self.temporary.cleanup()

    def test_d4plus_independent_sampling_and_mask_alignment(self):
        flist = self.root / "lists" / "clean.flist"
        entries = build_flist(self.clean, absolute=True)
        write_flist(entries, flist)
        dataset = UnpairedDehazeDataset(
            flist,
            self.hazy,
            crop_size=8,
            clean_mask_dir=self.clean_masks,
            hazy_mask_dir=self.hazy_masks,
            is_real=True,
            seed=4,
            return_paths=True,
        )
        self.assertEqual(len(dataset), 2)
        dataset._sample_index = Mock(side_effect=(0, 0, 0, 0))
        item = dataset[0]
        self.assertEqual(
            dataset._sample_index.call_args_list,
            [
                call(dataset.clean_paths),
                call(dataset.hazy_paths),
                call(dataset.clean_paths),
                call(dataset.hazy_paths),
            ],
        )
        self.assertEqual(item["clean_path"], item["clean_ref_path"])
        self.assertEqual(item["hazy_path"], item["hazy_ref_path"])
        self.assertEqual(item["clean"].shape, (3, 8, 8))
        self.assertEqual(item["clean_mask"].shape, (1, 8, 8))
        expected_mask = (item["clean"][0] > 0.5).float()
        self.assertTrue((expected_mask == item["clean_mask"][0]).all())

    def test_indoor_masks_are_zero_and_test_gt_overlap_is_rejected(self):
        dataset = UnpairedDehazeDataset(
            self.clean, self.hazy, crop_size=8, is_real=False, seed=1
        )
        item = dataset[0]
        self.assertEqual(set(item), {
            "clean", "hazy", "clean_ref", "hazy_ref",
            "clean_mask", "hazy_mask", "is_real",
        })
        self.assertEqual(item["clean_mask"].count_nonzero().item(), 0)
        with self.assertRaisesRegex(ValueError, "Test ground truth"):
            UnpairedDehazeDataset(
                self.clean,
                self.hazy,
                crop_size=8,
                test_gt_source=[next(self.clean.glob("*.png"))],
            )

    def test_real_layout_validation_checks_pair_counts_and_dimensions(self):
        counts = validate_real_dataset_layout(
            self.clean, self.hazy, self.clean_masks, self.hazy_masks
        )
        self.assertEqual(counts, {"clean": 2, "hazy": 2})

        mismatched_mask = self.hazy_masks / "scene1_1.jpg"
        Image.fromarray(np.zeros((10, 12), dtype=np.uint8)).save(mismatched_mask)
        with self.assertRaisesRegex(ValueError, "size mismatch"):
            validate_real_dataset_layout(
                self.clean, self.hazy, self.clean_masks, self.hazy_masks
            )


if __name__ == "__main__":
    unittest.main()
