import importlib.util
import gzip
import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
PATH = ROOT / "tools/analyze_phoenix_partial_chronological_oracle.py"
SPEC = importlib.util.spec_from_file_location("partial_chronological_oracle", PATH)
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


class PartialChronologicalOracleTests(unittest.TestCase):
    def test_dense_coordinate_and_eos_flush_semantics(self):
        first = MODULE.window_input_coordinates(0, 20)
        self.assertEqual(first["unclamped_original_range"], [-7, 8])
        self.assertEqual(first["left_repeats"], 7)
        self.assertEqual(first["normal_available_frame"], 8)
        last_normal = MODULE.window_input_coordinates(11, 20)
        self.assertEqual(last_normal["clamped_original_range"], [4, 19])
        self.assertFalse(last_normal["requires_eos_flush"])
        tail = MODULE.window_input_coordinates(12, 20)
        self.assertTrue(tail["requires_eos_flush"])
        self.assertIsNone(tail["normal_available_frame"])

    def test_bucket_has_frozen_unknown_eos_rate_and_capacity(self):
        selected, balance = MODULE.online_uniform_schedule(400)
        self.assertEqual(selected[:6], [0, 3, 4, 7, 8, 11])
        self.assertAlmostEqual(len(selected) / 400, 0.5)
        self.assertGreaterEqual(balance, -MODULE.EPSILON)
        self.assertLessEqual(balance, MODULE.TOKEN_CAPACITY + MODULE.EPSILON)

    def test_skeleton_bounds_coverage_without_total_length(self):
        for length in range(1, 50):
            selected = [start for start in range(length) if MODULE.is_skeleton(start)]
            metrics = MODULE.coverage_metrics(selected, length)
            self.assertLessEqual(metrics["max_gap_including_endpoints"], 4)

    def test_prefix_state_ignores_candidate_and_future_logits(self):
        rng = np.random.default_rng(9)
        probabilities = rng.random((12, 7), dtype=np.float32)
        probabilities /= probabilities.sum(axis=1, keepdims=True)
        selected = [0, 3, 4]
        before = MODULE.prefix_state(probabilities, selected, 6)
        changed = probabilities.copy()
        changed[6:] = rng.random(changed[6:].shape, dtype=np.float32)
        changed[6:] /= changed[6:].sum(axis=1, keepdims=True)
        after = MODULE.prefix_state(changed, selected, 6)
        self.assertEqual(before, after)

        # Cheap future frames are not arguments to this minimal state at all;
        # perturbing them therefore cannot alter any serialized causal field.
        cheap_frames = rng.random((12, 4), dtype=np.float32)
        future_changed = cheap_frames.copy()
        future_changed[9:] += 100.0
        self.assertTrue(np.array_equal(cheap_frames[:9], future_changed[:9]))
        self.assertEqual(before, MODULE.prefix_state(probabilities, selected, 6))

    def test_immediate_label_can_change_when_candidate_logit_changes(self):
        vocab = ["<blank>", "A", "B"]
        probabilities = np.asarray([
            [0.05, 0.90, 0.05],
            [0.05, 0.90, 0.05],
            [0.90, 0.05, 0.05],
        ], dtype=np.float32)
        selected = [0]
        value_a = MODULE.immediate_utility(probabilities, selected, 2, "A", vocab, 0)[0]
        changed = probabilities.copy()
        changed[2] = [0.05, 0.05, 0.90]
        value_b = MODULE.immediate_utility(changed, selected, 2, "A", vocab, 0)[0]
        self.assertNotEqual(value_a, value_b)
        self.assertEqual(
            MODULE.prefix_state(probabilities, selected, 2),
            MODULE.prefix_state(changed, selected, 2),
        )

    def test_policy_never_selects_future_or_duplicate_start(self):
        rng = np.random.default_rng(12)
        probabilities = rng.random((17, 5), dtype=np.float32)
        probabilities /= probabilities.sum(axis=1, keepdims=True)
        vocab = ["<blank>", "A", "B", "C", "D"]
        for policy in ("myopic", "rollout16"):
            output = MODULE.run_policy(probabilities, "A B", vocab, 0, policy)
            self.assertEqual(output["selected"], sorted(set(output["selected"])))
            for record in output["trace"]:
                current = record["candidate_window_start"]
                prefix = record["predictor_inputs"]["decoder_prefix_token_ids"]
                self.assertIsInstance(prefix, list)
                self.assertNotIn("reference", record["predictor_inputs"])
                self.assertLessEqual(record["predictor_inputs"]["selected_past_count"], current)

    def test_each_decision_has_exactly_one_immediate_utility_count(self):
        rng = np.random.default_rng(14)
        probabilities = rng.random((21, 5), dtype=np.float32)
        probabilities /= probabilities.sum(axis=1, keepdims=True)
        vocab = ["<blank>", "A", "B", "C", "D"]
        for policy in ("myopic", "rollout16"):
            output = MODULE.run_policy(probabilities, "A B", vocab, 0, policy)
            self.assertEqual(
                len(output["trace"]),
                sum(output["immediate_utility_counts"].values()),
            )

    def test_trace_action_counts(self):
        rows = [
            {"policy_trajectory": "myopic", "forced_by_token_capacity": False},
            {"policy_trajectory": "myopic", "forced_by_token_capacity": True},
            {"policy_trajectory": "rollout16", "forced_by_token_capacity": True},
        ]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "trace.jsonl.gz"
            with gzip.open(path, "wt") as handle:
                for row in rows:
                    handle.write(json.dumps(row) + "\n")
            counts = MODULE.trace_action_counts(path)
        self.assertEqual(counts["decision"]["myopic"], 2)
        self.assertEqual(counts["forced"]["myopic"], 1)
        self.assertEqual(counts["forced"]["rollout16"], 1)


if __name__ == "__main__":
    unittest.main()
