import gzip
import importlib.util
import json
import pickle
import tempfile
import unittest
from pathlib import Path

import numpy as np


CSLR_ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = CSLR_ROOT / "tools/build_phoenix_train_counterfactual_utility_dataset.py"
SPEC = importlib.util.spec_from_file_location("train_counterfactual_dataset", MODULE_PATH)
TRAIN = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(TRAIN)


class TrainCounterfactualUtilityDatasetTests(unittest.TestCase):
    def make_dense_shard(self, root):
        shard = root / "shards/shard-00000-of-00001"
        shard.mkdir(parents=True)
        name = "train/source-a-1"
        logits = {name: np.asarray([[2.0, 1.0], [0.1, 1.9], [1.8, 0.2]], dtype=np.float32)}
        results = {
            name: {
                "gls_ref": "A",
                "adaptive_stride_metadata": [{"start": index} for index in range(3)],
            }
        }
        for filename, value in (("train_results.pkl", results), ("train_logits.pkl", logits)):
            with (shard / filename).open("wb") as handle:
                pickle.dump(value, handle)
        completion = {
            "shard_index": 0,
            "samples": 1,
            "windows": 3,
            "artifacts": {
                filename: {
                    "bytes": (shard / filename).stat().st_size,
                    "sha256": TRAIN.sha256_file(shard / filename),
                }
                for filename in ("train_results.pkl", "train_logits.pkl")
            },
        }
        (shard / "complete.json").write_text(json.dumps(completion))
        return shard, results, logits

    def test_dense_shard_hashes_coordinates_and_train_split(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _, expected_results, expected_logits = self.make_dense_shard(root)
            results, logits, completion = TRAIN.validate_dense_shard(root, 0, 1)
            self.assertEqual(list(results), list(expected_results))
            self.assertEqual(list(logits), list(expected_logits))
            self.assertEqual(completion["windows"], 3)

    def test_non_dense_coordinates_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            shard, results, logits = self.make_dense_shard(root)
            name = next(iter(results))
            results[name]["adaptive_stride_metadata"][1]["start"] = 2
            with (shard / "train_results.pkl").open("wb") as handle:
                pickle.dump(results, handle)
            completion = json.loads((shard / "complete.json").read_text())
            completion["artifacts"]["train_results.pkl"] = {
                "bytes": (shard / "train_results.pkl").stat().st_size,
                "sha256": TRAIN.sha256_file(shard / "train_results.pkl"),
            }
            (shard / "complete.json").write_text(json.dumps(completion))
            with self.assertRaisesRegex(ValueError, "non-dense"):
                TRAIN.validate_dense_shard(root, 0, 1)

    def test_source_video_partition_must_be_mutually_exclusive(self):
        names = ["train/source-a-1", "train/source-a-2"]
        good = {"assignments": {name: "fit" for name in names}}
        self.assertEqual(TRAIN.validate_train_partition(good, names)["fit"], 2)
        bad = {"assignments": {names[0]: "fit", names[1]: "calibration"}}
        with self.assertRaisesRegex(ValueError, "crosses"):
            TRAIN.validate_train_partition(bad, names)

    def test_generated_states_cover_all_candidates_and_keep_inputs_clean(self):
        rng = np.random.default_rng(41)
        name = "train/source-a-1"
        logits = {name: rng.normal(size=(10, 4)).astype(np.float32)}
        results = {name: {"gls_ref": "A B"}}
        vocab = ["<blank>", "A", "B", "C"]
        samples, statistics = TRAIN.process_dense_shard(results, logits, vocab, workers=1)
        self.assertEqual(statistics["samples"], 1)
        for state in samples[0]["states"]:
            selected = set(state["oracle_state"]["initial_skeleton"])
            selected.update(state["oracle_state"]["selected_bonus_before"])
            candidates = [row["candidate_window_start"] for row in state["candidates"]]
            self.assertEqual(set(candidates), set(range(10)) - selected)
            self.assertEqual(len(candidates), len(set(candidates)))
            for row in state["candidates"]:
                TRAIN.BASE.assert_predictor_inputs_clean(row["predictor_inputs"])
                self.assertNotIn("logits", row["predictor_inputs"])

    def test_test_paths_are_forbidden(self):
        with self.assertRaisesRegex(ValueError, "test paths are forbidden"):
            TRAIN.reject_test_path(Path("/tmp/test/utility"))


if __name__ == "__main__":
    unittest.main()
