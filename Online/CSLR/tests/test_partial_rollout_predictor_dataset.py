import importlib.util
import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
PATH = ROOT / "tools/build_phoenix_partial_rollout_predictor_dataset.py"
SPEC = importlib.util.spec_from_file_location("rollout_predictor_dataset", PATH)
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


class RolloutPredictorDatasetTests(unittest.TestCase):
    def probabilities(self):
        rng = np.random.default_rng(81)
        x = rng.random((30, 8), dtype=np.float32)
        return x / x.sum(axis=1, keepdims=True)

    def test_prefix_ignores_current_and_future_logits(self):
        values = self.probabilities(); selected = [0, 3, 4, 7]
        before = MODULE.prefix_features(values, selected, 8, 0)
        changed = values.copy(); changed[8:] = changed[8:, ::-1]
        self.assertEqual(before, MODULE.prefix_features(changed, selected, 8, 0))

    def test_prefix_changes_when_past_selected_logit_changes(self):
        values = self.probabilities(); selected = [0, 3, 4, 7]
        before = MODULE.prefix_features(values, selected, 8, 0)
        changed = values.copy(); changed[7] = np.roll(changed[7], 1)
        self.assertNotEqual(before, MODULE.prefix_features(changed, selected, 8, 0))

    def test_source_video_removes_only_numeric_segment(self):
        self.assertEqual(MODULE.source_video("train/foo-bar-17"), "train/foo-bar")
        with self.assertRaises(ValueError): MODULE.source_video("train/foo-bar")

    def test_autonomous_rows_exclude_forced_states(self):
        vocab = ["<blank>", "A", "B", "C"]
        probabilities = self.probabilities()[:, :4]
        probabilities /= probabilities.sum(axis=1, keepdims=True)
        logits = np.log(probabilities)
        result = {"gls_ref": "A B", "adaptive_stride_metadata": [{"start": i} for i in range(30)]}
        rows, counts = MODULE.sample_records("train/foo-1", result, logits, vocab, 0, "fit")
        self.assertTrue(rows)
        self.assertTrue(all(r["autonomous_training_row"] == (not r["forced_by_token_capacity"]) for r in rows))
        self.assertTrue(all("total_windows" not in r["bookkeeping"] for r in rows))
        self.assertGreater(counts["forced"], 0)


if __name__ == "__main__": unittest.main()
