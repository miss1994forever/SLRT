import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "analyze_phoenix_partial_terminal_advantage_audit",
    ROOT / "tools/analyze_phoenix_partial_terminal_advantage_audit.py",
)
TERMINAL = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = TERMINAL
SPEC.loader.exec_module(TERMINAL)


class TerminalAdvantageAuditTests(unittest.TestCase):
    def test_hash_subset_is_deterministic(self):
        names = ["sample-c", "sample-a", "sample-b"]
        first = sorted(names, key=lambda x: (TERMINAL.stable_rank(x), x))
        second = sorted(reversed(names), key=lambda x: (TERMINAL.stable_rank(x), x))
        self.assertEqual(first, second)

    def test_branch_to_end_does_not_mutate_selected_and_uses_same_policy(self):
        selected = [0, 4]
        policy = ("kind", mock.Mock(), {"x": 1}, 0.25)
        with mock.patch.object(TERMINAL.AUDIT, "branch_future", return_value=[0, 4, 5]) as branch:
            result = TERMINAL.branch_to_end(
                policy, mock.Mock(), "name", np.zeros((10, 2)), selected,
                1.0, 5, True, 0,
            )
        self.assertEqual(selected, [0, 4])
        self.assertEqual(result, [0, 4, 5])
        args, kwargs = branch.call_args
        self.assertIs(args[1], policy[1])
        self.assertEqual(args[3], policy[3])
        self.assertEqual(kwargs["horizon"], 10)

    def test_future_can_change_label_without_changing_current_input(self):
        selected = [0]
        probabilities = np.asarray([[0.8, 0.2], [0.8, 0.2], [0.8, 0.2]])
        changed = probabilities.copy(); changed[2] = [0.1, 0.9]
        with mock.patch.object(TERMINAL, "branch_to_end", side_effect=[[0, 1, 2], [0, 2], [0, 1, 2], [0, 2]]), \
             mock.patch.object(TERMINAL, "advantage_from_selections", side_effect=[(0, 0, 0), (1, 1, 0)]):
            before = TERMINAL.terminal_advantage(
                mock.Mock(), mock.Mock(), "x", probabilities, selected, 1., 1,
                ["A"], ["<blank>", "A"], 0)
            after = TERMINAL.terminal_advantage(
                mock.Mock(), mock.Mock(), "x", changed, selected, 1., 1,
                ["A"], ["<blank>", "A"], 0)
        self.assertEqual(selected, [0])
        self.assertNotEqual(before[0], after[0])

    def test_preregistered_state_action_excludes_eos(self):
        with mock.patch.object(TERMINAL, "sha256_file", return_value="hash"):
            cfg = TERMINAL.frozen_config(["x"])
        self.assertTrue(cfg["protocol"]["unknown_T_EOS_for_actions"])
        self.assertIn("T/EOS in state or deployable action", cfg["forbidden"])
        self.assertIn("true sample end", cfg["terminal_label"])

    def test_label_summary_reports_zero_nonzero_and_opposites(self):
        rows = [
            {"local16_advantage": 0, "terminal_advantage": 1},
            {"local16_advantage": 1, "terminal_advantage": 0},
            {"local16_advantage": 1, "terminal_advantage": -1},
            {"local16_advantage": 1, "terminal_advantage": 1},
        ]
        value = TERMINAL.label_summary(rows)
        self.assertEqual(value["local_zero_terminal_nonzero"], 1)
        self.assertEqual(value["local_nonzero_terminal_zero"], 1)
        self.assertEqual(value["direct_opposite_signs"], 1)
        self.assertEqual(value["union_nonzero_states"], 4)

    def test_output_root_rejects_dev_or_test(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(ValueError):
                TERMINAL.preregister(Path(directory) / "dev-output")


if __name__ == "__main__":
    unittest.main()
