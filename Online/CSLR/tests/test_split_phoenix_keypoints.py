import importlib.util
import json
from pathlib import Path
import pickle
import tempfile
import unittest

import numpy as np


PATH = Path(__file__).resolve().parents[1] / "tools/split_phoenix_keypoints.py"
SPEC = importlib.util.spec_from_file_location("split_phoenix_keypoints", PATH)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class SplitPhoenixKeypointsTest(unittest.TestCase):
    def mapping(self):
        return {
            "train/a": np.asarray([[[1.0, np.nan, 0.5]]], dtype=np.float16),
            "dev/b": np.ones((2, 1, 3), dtype=np.float16),
            "test/c": np.zeros((3, 1, 3), dtype=np.float16),
        }

    def test_partition_rejects_unqualified_and_unknown_keys(self):
        for key in ("video", "validation/video", "train/"):
            with self.subTest(key=key), self.assertRaises(ValueError):
                MODULE.partition_mapping({key: np.zeros((1, 1, 3))})

    def test_partition_is_complete_and_disjoint(self):
        source = self.mapping()
        parts = MODULE.partition_mapping(source)
        self.assertEqual(set().union(*(set(value) for value in parts.values())), set(source))
        self.assertTrue(set(parts["train"]).isdisjoint(parts["dev"]))
        self.assertTrue(set(parts["train"]).isdisjoint(parts["test"]))
        self.assertTrue(set(parts["dev"]).isdisjoint(parts["test"]))

    def test_split_file_round_trip_and_existing_outputs_are_validated(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "whole.pkl"
            manifest_path = root / "manifest.json"
            with source.open("wb") as handle:
                pickle.dump(self.mapping(), handle)
            first = MODULE.split_file(source, root, manifest_path)
            self.assertTrue(all(status == "created" for status in first["run_write_status"].values()))
            saved_manifest = json.loads(manifest_path.read_text())
            self.assertTrue(saved_manifest["integrity"]["union_equals_source"])
            for split, expected_frames in (("train", 1), ("dev", 2), ("test", 3)):
                output = root / f"whole.{split}.pkl"
                with output.open("rb") as handle:
                    mapping = pickle.load(handle)
                self.assertEqual(set(mapping), {f"{split}/{chr(97 + MODULE.SPLITS.index(split))}"})
                self.assertEqual(first["outputs"][split]["total_frames"], expected_frames)

            second = MODULE.split_file(source, root, manifest_path)
            self.assertTrue(all(status == "existing" for status in second["run_write_status"].values()))

    def test_existing_corrupt_output_is_not_overwritten(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "whole.pkl"
            with source.open("wb") as handle:
                pickle.dump(self.mapping(), handle)
            with (root / "whole.dev.pkl").open("wb") as handle:
                pickle.dump({"dev/wrong": np.zeros((1, 1, 3))}, handle)
            with self.assertRaisesRegex(ValueError, "dev key mismatch"):
                MODULE.split_file(source, root, root / "manifest.json")
            with (root / "whole.dev.pkl").open("rb") as handle:
                self.assertEqual(set(pickle.load(handle)), {"dev/wrong"})


if __name__ == "__main__":
    unittest.main()
