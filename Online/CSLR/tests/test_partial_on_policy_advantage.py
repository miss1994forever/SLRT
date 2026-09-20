import importlib.util
import sys
import unittest
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
PATH = ROOT / "tools/analyze_phoenix_partial_on_policy_advantage.py"
SPEC = importlib.util.spec_from_file_location("on_policy_advantage", PATH)
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


class OnPolicyAdvantageTests(unittest.TestCase):
    def test_sign(self):
        self.assertEqual(MODULE.sign(-2), -1)
        self.assertEqual(MODULE.sign(0), 0)
        self.assertEqual(MODULE.sign(3), 1)

    def test_summary_exposes_policy_mismatch(self):
        rows = [
            {"eager_advantage": 1, "on_policy_advantage": 0, "action_execute": True},
            {"eager_advantage": 0, "on_policy_advantage": 1, "action_execute": False},
            {"eager_advantage": -1, "on_policy_advantage": -1, "action_execute": True},
        ]
        value = MODULE.summarize(rows)
        self.assertEqual(value["eager_positive_onpolicy_nonpositive"], 1)
        self.assertEqual(value["eager_nonpositive_onpolicy_positive"], 1)
        self.assertEqual(value["predictor_action"]["harmful_execute_count"], 1)
        self.assertEqual(value["predictor_action"]["on_policy_action_regret"], 2)

    def test_branch_does_not_mutate_selected(self):
        selected = [0, 4]
        original = list(selected)
        # Exercise the mutation contract without requiring a model: horizon 0.
        result = MODULE.branch_future(
            "unused", None, None, 0.0, None, "unused", np.zeros((10, 2)),
            0, selected, 1.0, 5, True, horizon=0,
        )
        self.assertEqual(selected, original)
        self.assertEqual(result, [0, 4, 5])


if __name__ == "__main__":
    unittest.main()
