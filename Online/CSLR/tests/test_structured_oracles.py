import importlib.util
import unittest
from pathlib import Path

import numpy as np


CSLR_ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = CSLR_ROOT / "tools/analyze_phoenix_structured_oracles.py"
SPEC = importlib.util.spec_from_file_location("structured_oracles", MODULE_PATH)
ORACLE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(ORACLE)


class StructuredOracleTests(unittest.TestCase):
    def test_uniform_skeleton_is_exact_unique_and_endpoint_covering(self):
        values = ORACLE.uniform_positions(20, 11)
        self.assertEqual(len(values), 11)
        self.assertEqual(values, sorted(set(values)))
        self.assertEqual((values[0], values[-1]), (0, 19))

    def test_offline_schedule_preserves_skeleton_and_exact_budget(self):
        scores = np.asarray([0, 0, 9, 0, 0, 8, 0, 0, 7, 0, 0, 6], dtype=float)
        schedule = ORACLE.offline_event_bonus_schedule(scores, 8)
        skeleton = ORACLE.uniform_positions(12, ORACLE.skeleton_count(12, 8))
        self.assertEqual(len(schedule), 8)
        self.assertEqual(schedule, sorted(set(schedule)))
        self.assertTrue(set(skeleton).issubset(schedule))
        self.assertEqual((schedule[0], schedule[-1]), (0, 11))

    def test_causal_schedule_is_deterministic_exact_and_endpoint_covering(self):
        scores = np.linspace(0.0, 1.0, 30)
        first = ORACLE.causal_paced_schedule(scores, 19)
        second = ORACLE.causal_paced_schedule(scores, 19)
        self.assertEqual(first, second)
        self.assertEqual(len(first), 19)
        self.assertEqual(first, sorted(set(first)))
        self.assertEqual((first[0], first[-1]), (0, 29))

    def test_signal_computation_reads_only_current_and_previous_rows(self):
        rng = np.random.default_rng(7)
        logits = rng.normal(size=(20, 6))
        full = ORACLE.compute_past_only_signals(logits, blank_id=0)
        prefix = ORACLE.compute_past_only_signals(logits[:11], blank_id=0)
        for signal in ORACLE.SIGNALS:
            np.testing.assert_allclose(full[signal][:11], prefix[signal], rtol=0, atol=0)

    def test_causal_policy_prefix_is_unchanged_when_future_scores_change(self):
        rng = np.random.default_rng(11)
        scores = rng.random(40)
        changed = scores.copy()
        changed[21:] = rng.random(19) * 1000
        first = ORACLE.causal_paced_schedule(scores, 24)
        second = ORACLE.causal_paced_schedule(changed, 24)
        self.assertEqual([x for x in first if x <= 20], [x for x in second if x <= 20])

    def test_coordinate_mapping_remains_fixed_b0_compatible(self):
        aligned, dense_pad, target_pad = ORACLE.REPLAY.align_starts_to_dense_inputs(
            list(range(100)), [0, 3, 6, 97]
        )
        self.assertEqual((dense_pad, target_pad), (7, 6))
        self.assertEqual(aligned, [1, 4, 7, 98])

    def test_test_paths_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "test paths are forbidden"):
            ORACLE.alignment_audit(Path("/tmp/test/meta"), Path("/tmp/dev/labels"))


if __name__ == "__main__":
    unittest.main()
