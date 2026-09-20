import importlib.util
import sys
import unittest
from pathlib import Path


PATH = Path(__file__).resolve().parents[1] / "tools/audit_phoenix_full_robust_label_stability.py"
SPEC = importlib.util.spec_from_file_location("audit_phoenix_full_robust_label_stability", PATH)
AUDIT = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = AUDIT
SPEC.loader.exec_module(AUDIT)


class FullRobustLabelStabilityTest(unittest.TestCase):
    def test_center_is_the_stable_tie_break(self):
        self.assertEqual(AUDIT.stable_best([4, 4, 4]), 1)
        self.assertEqual(AUDIT.stable_best([3, 4, 3]), 0)
        self.assertEqual(AUDIT.stable_best([4, 3, 3]), 1)

    def test_side_signs_are_relative_to_center_error(self):
        self.assertEqual(AUDIT.side_signs([2, 3, 4]), (1, -1))
        self.assertEqual(AUDIT.side_signs([3, 3, 3]), (0, 0))

    def test_frozen_complete_fit_counts(self):
        config = AUDIT.frozen_config()
        self.assertEqual(config["scope"]["samples"], 6378)
        self.assertEqual(config["scope"]["sources"], 578)
        self.assertEqual(config["blocks"], 183401)
        self.assertEqual(config["expected_robust_positive_sides"], 769)


if __name__ == "__main__":
    unittest.main()
