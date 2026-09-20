import importlib.util
import sys
import unittest
from pathlib import Path


PATH = Path(__file__).resolve().parents[1] / "tools/audit_phoenix_terminal_label_stability.py"
SPEC = importlib.util.spec_from_file_location("audit_phoenix_terminal_label_stability", PATH)
AUDIT = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = AUDIT
SPEC.loader.exec_module(AUDIT)


class TerminalLabelStabilityTest(unittest.TestCase):
    def test_late_continuation_keeps_one_bonus_in_complete_blocks(self):
        self.assertEqual(AUDIT.late_continuation(11, 4), [4, 7, 8])
        self.assertEqual(AUDIT.late_continuation(12, 4), [4, 7, 8, 11])

    def test_center_tie_break(self):
        self.assertEqual(AUDIT.stable_best([3, 3, 4]), 1)
        self.assertEqual(AUDIT.stable_best([2, 3, 2]), 0)
        self.assertEqual(AUDIT.stable_best([4, 3, 2]), 2)

    def test_side_signs_are_relative_to_center(self):
        self.assertEqual(AUDIT.signs([2, 3, 4]), (1, -1))
        self.assertEqual(AUDIT.signs([3, 3, 3]), (0, 0))


if __name__ == "__main__":
    unittest.main()
