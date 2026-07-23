import unittest

import torch

from tools.reliability_dev_diagnostic import keypoint_quality
from utils.reliability_analysis import (
    binary_auc,
    expected_calibration_error,
    selective_accuracy,
)


class ReliabilityAnalysisTests(unittest.TestCase):
    def test_auc_perfect_and_reversed(self):
        self.assertEqual(binary_auc([0.1, 0.2, 0.8, 0.9], [0, 0, 1, 1]), 1.0)
        self.assertEqual(binary_auc([0.9, 0.8, 0.2, 0.1], [0, 0, 1, 1]), 0.0)

    def test_auc_handles_ties(self):
        self.assertEqual(binary_auc([0.5, 0.5], [0, 1]), 0.5)

    def test_auc_requires_both_classes(self):
        self.assertIsNone(binary_auc([0.1, 0.2], [1, 1]))

    def test_ece_is_zero_for_matching_groups(self):
        self.assertAlmostEqual(expected_calibration_error([0.0, 1.0], [0, 1], bins=2), 0.0)

    def test_selective_accuracy_uses_highest_confidence(self):
        result = selective_accuracy([0.1, 0.9, 0.8, 0.2], [0, 1, 1, 0], coverages=(0.5, 1.0))
        self.assertEqual(result["0.5"]["accuracy"], 1.0)
        self.assertEqual(result["1.0"]["accuracy"], 0.5)

    def test_keypoint_quality_preserves_float_dtype(self):
        keypoints = torch.zeros(2, 3, 4, 3, dtype=torch.float32)
        keypoints[..., 2] = 1.0
        visible, confidence, motion = keypoint_quality([keypoints], 0.2)
        self.assertEqual(visible.dtype, torch.float32)
        self.assertEqual(confidence.dtype, torch.float32)
        self.assertEqual(motion.dtype, torch.float32)
        self.assertEqual(visible.tolist(), [1.0, 1.0])


if __name__ == "__main__":
    unittest.main()
