import importlib.util
import sys
import unittest
from pathlib import Path

import numpy as np


PATH = Path(__file__).resolve().parents[1] / "tools/analyze_phoenix_robust_predictability_oof.py"
SPEC = importlib.util.spec_from_file_location("analyze_phoenix_robust_predictability_oof", PATH)
OOF = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = OOF
SPEC.loader.exec_module(OOF)


class RobustPredictabilityOOFTest(unittest.TestCase):
    def test_source_fold_is_deterministic_and_bounded(self):
        self.assertEqual(OOF.source_fold("source-a"), OOF.source_fold("source-a"))
        self.assertIn(OOF.source_fold("source-a"), range(OOF.FOLDS))

    def test_visual_summary(self):
        history = np.asarray([[1.0, 2.0], [3.0, 6.0]], dtype=np.float32)
        actual = OOF.visual_summary(history)
        expected = np.asarray([3, 6, 2, 4, 1, 2, 2, 4], dtype=np.float32)
        np.testing.assert_allclose(actual, expected)

    def test_metrics_penalizes_harmful_top_k(self):
        labels = np.asarray([1, 0, 1, 0], dtype=np.float32)
        rewards = np.asarray([1, -1, 1, 0], dtype=np.int64)
        scores = np.asarray([0.9, 0.8, 0.2, 0.1], dtype=np.float32)
        result = OOF.metrics(labels, rewards, scores)
        self.assertEqual(result["K"], 2)
        self.assertEqual(result["top_K_signed_robust_utility"], 0)
        self.assertEqual(result["top_K_harmful_actions"], 1)
        self.assertAlmostEqual(result["positive_recall_at_K"], 0.5)


if __name__ == "__main__":
    unittest.main()
