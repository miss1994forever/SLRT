import gzip
import importlib.util
import pickle
import tempfile
import unittest
from pathlib import Path

import numpy as np


CSLR_ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = CSLR_ROOT / "tools/run_phoenix_train_dense_shards.py"
SPEC = importlib.util.spec_from_file_location("train_dense_shards", MODULE_PATH)
SHARDS = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(SHARDS)


class TrainDenseShardTests(unittest.TestCase):
    def test_source_video_groups_do_not_cross_fit_calibration(self):
        records = [
            {"name": "train/video-a-1"},
            {"name": "train/video-a-2"},
            {"name": "train/video-b-1"},
            {"name": "train/video-c-1"},
        ]
        split = SHARDS.build_fit_calibration(records, calibration_fraction=0.5)
        assignment = split["assignments"]
        self.assertEqual(assignment["train/video-a-1"], assignment["train/video-a-2"])
        fit_groups = {
            SHARDS.source_video_id(name) for name, value in assignment.items() if value == "fit"
        }
        calibration_groups = {
            SHARDS.source_video_id(name)
            for name, value in assignment.items()
            if value == "calibration"
        }
        self.assertTrue(fit_groups.isdisjoint(calibration_groups))

    def test_train_metadata_is_accepted_and_test_path_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            train_path = Path(directory) / "train.pkl.gz"
            with gzip.open(train_path, "wb") as handle:
                pickle.dump([{"name": "train/example-1"}], handle)
            self.assertEqual(SHARDS.load_train_meta(train_path)[0]["name"], "train/example-1")
            test_path = Path(directory) / "test" / "meta.pkl.gz"
            test_path.parent.mkdir()
            with gzip.open(test_path, "wb") as handle:
                pickle.dump([{"name": "train/example-1"}], handle)
            with self.assertRaisesRegex(ValueError, "test paths are forbidden"):
                SHARDS.load_train_meta(test_path)

    def test_completed_shard_requires_dense_coordinates_and_reference(self):
        with tempfile.TemporaryDirectory() as directory:
            shard = Path(directory)
            records = [{"name": "train/a-1", "num_frames": 3, "gloss": "A"}]
            results = {
                "train/a-1": {
                    "gls_ref": "A",
                    "adaptive_stride_metadata": [
                        {"start": 0},
                        {"start": 1},
                        {"start": 2},
                    ],
                }
            }
            logits = {"train/a-1": np.zeros((3, 2), dtype=np.float32)}
            with (shard / "train_results.pkl").open("wb") as handle:
                pickle.dump(results, handle)
            with (shard / "train_logits.pkl").open("wb") as handle:
                pickle.dump(logits, handle)
            (shard / "train_runtime_profile.json").write_text("{}\n")
            summary = SHARDS.validate_completed_shard(shard, records)
            self.assertEqual(summary["samples"], 1)
            self.assertEqual(summary["windows"], 3)


if __name__ == "__main__":
    unittest.main()
