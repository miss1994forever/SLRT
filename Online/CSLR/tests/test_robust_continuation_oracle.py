import importlib.util
import sys
import unittest
from pathlib import Path


PATH = Path(__file__).resolve().parents[1] / "tools/analyze_phoenix_robust_continuation_oracle.py"
SPEC = importlib.util.spec_from_file_location("analyze_phoenix_robust_continuation_oracle", PATH)
ORACLE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = ORACLE
SPEC.loader.exec_module(ORACLE)


class RobustContinuationOracleTest(unittest.TestCase):
    def test_continuation_offsets(self):
        self.assertEqual(ORACLE.continuation(12, 4, 2), [4, 6, 8, 10])
        self.assertEqual(ORACLE.continuation(12, 4, 3), [4, 7, 8, 11])

    def test_center_fallback_without_strict_benefit(self):
        error_table = {
            (1, 10): 3, (2, 10): 3, (3, 10): 4,
            (1, 20): 2, (2, 20): 2, (3, 20): 2,
        }
        original = ORACLE.decode_error
        ORACLE.decode_error = lambda _p, selected, _r, _v, _b: error_table[(selected[-2], selected[-1])]
        try:
            choice, errors, robust = ORACLE.robust_choice(
                object(), [], [1, 2, 3], [[10], [20]], object(), object(), object()
            )
        finally:
            ORACLE.decode_error = original
        self.assertEqual(choice, 1)
        self.assertEqual(errors, [[3, 3, 4], [2, 2, 2]])
        self.assertEqual(robust, [0, 0, -1])

    def test_side_requires_strict_benefit_under_both_futures(self):
        error_table = {
            (1, 10): 2, (2, 10): 4, (3, 10): 4,
            (1, 20): 3, (2, 20): 4, (3, 20): 1,
        }
        original = ORACLE.decode_error
        ORACLE.decode_error = lambda _p, selected, _r, _v, _b: error_table[(selected[-2], selected[-1])]
        try:
            choice, _, robust = ORACLE.robust_choice(
                object(), [], [1, 2, 3], [[10], [20]], object(), object(), object()
            )
        finally:
            ORACLE.decode_error = original
        self.assertEqual(choice, 0)
        self.assertEqual(robust, [1, 0, 0])


if __name__ == "__main__":
    unittest.main()
