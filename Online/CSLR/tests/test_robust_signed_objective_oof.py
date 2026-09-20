import importlib.util
import sys
import unittest
from pathlib import Path

import numpy as np


PATH = Path(__file__).resolve().parents[1] / "tools/analyze_phoenix_robust_signed_objective_oof.py"
SPEC = importlib.util.spec_from_file_location("analyze_phoenix_robust_signed_objective_oof", PATH)
SIGNED = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = SIGNED
SPEC.loader.exec_module(SIGNED)


class RobustSignedObjectiveOOFTest(unittest.TestCase):
    def test_signed_classes(self):
        np.testing.assert_array_equal(
            SIGNED.signed_classes(np.asarray([-2, -1, 0, 1, 2])),
            np.asarray([0, 0, 1, 2, 2]),
        )

    def test_best_prefix(self):
        result = SIGNED.best_prefix(
            np.asarray([1, -1, 1]), np.asarray([0.9, 0.8, 0.7])
        )
        self.assertEqual(result, {"signed_utility": 1, "K": 1})


if __name__ == "__main__":
    unittest.main()
