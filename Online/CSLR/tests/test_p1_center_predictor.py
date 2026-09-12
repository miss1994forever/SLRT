import importlib.util
from pathlib import Path
import unittest

import numpy as np


PATH = Path(__file__).resolve().parents[1] / "tools/train_phoenix_p1_center_predictor.py"
SPEC = importlib.util.spec_from_file_location("p1_center_predictor", PATH)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class P1CenterPredictorTest(unittest.TestCase):
    def test_rejects_dev_and_test_paths(self):
        for path in ("/data/dev/file.pkl", "/data/test/file.pkl", "/x/test_features.npz"):
            with self.subTest(path=path), self.assertRaises(ValueError):
                MODULE.reject_dev_or_test_path(path)

    def test_event_is_upward_crossing_with_refractory(self):
        scores = [0.1, 0.6, 0.7, 0.2, 0.8, 0.1, 0.9]
        self.assertEqual(MODULE.events_from_scores(scores, 0.5, refractory=4), [1, 6])

    def test_event_metrics_reports_delay_direction(self):
        metrics = MODULE.event_metrics({"v": [0.1, 0.1, 0.8, 0.1]}, {"v": [1]}, 0.5)
        self.assertEqual(metrics["center_recall"], 1.0)
        self.assertEqual(metrics["positive_delay_fraction"], 1.0)
        self.assertEqual(metrics["matched_signed_error_mean"], 1.0)

    def test_logistic_fits_separable_toy_data(self):
        features = np.asarray([[-2.0], [-1.0], [1.0], [2.0]])
        target = np.asarray([0, 0, 1, 1])
        coef, intercept, status = MODULE.fit_logistic(features, target)
        probability = MODULE.expit(features @ coef + intercept)
        self.assertTrue(status["success"])
        self.assertGreater(probability[2], probability[1])

    def test_binary_metrics_perfect_ranking(self):
        metrics = MODULE.binary_metrics([0, 0, 1, 1], [0.1, 0.2, 0.8, 0.9])
        self.assertAlmostEqual(metrics["auroc"], 1.0)
        self.assertAlmostEqual(metrics["auprc"], 1.0)


if __name__ == "__main__":
    unittest.main()
