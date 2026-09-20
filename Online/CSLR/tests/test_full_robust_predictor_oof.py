import importlib.util
import sys
import unittest
from pathlib import Path

import numpy as np


PATH = Path(__file__).resolve().parents[1] / "tools/analyze_phoenix_full_robust_predictor_oof.py"
SPEC = importlib.util.spec_from_file_location("analyze_phoenix_full_robust_predictor_oof", PATH)
OOF = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = OOF
SPEC.loader.exec_module(OOF)


class FullRobustPredictorOOFTest(unittest.TestCase):
    def test_frozen_counts_and_gate(self):
        config = OOF.frozen_config()
        self.assertEqual(config["scope"]["samples"], 6378)
        self.assertEqual(config["scope"]["sources"], 578)
        self.assertEqual(config["rows"], 366802)
        self.assertEqual(config["class_counts"], {
            "beneficial": 769, "harmful": 2015, "neutral": 364018,
        })
        self.assertEqual(config["strong_headroom_errors"], 249)

    def test_fold_percentiles_are_fold_local(self):
        scores = np.asarray([0.1, 0.2, 20.0, 10.0], dtype=np.float32)
        folds = np.asarray([0, 0, 1, 1], dtype=np.int8)
        actual = OOF.fold_percentiles(scores, folds)
        np.testing.assert_allclose(actual, [0.0, 1.0, 1.0, 0.0])

    def test_source_bootstrap_delta_direction(self):
        rewards = np.asarray([1, -1, 1, -1])
        sources = np.asarray([0, 0, 1, 1])
        better = np.asarray([4.0, 1.0, 3.0, 0.0])
        worse = np.asarray([1.0, 4.0, 0.0, 3.0])
        interval = OOF.source_bootstrap_delta(rewards, sources, better, worse, k=2, reps=100)
        self.assertGreater(interval[0], 0)


if __name__ == "__main__":
    unittest.main()
