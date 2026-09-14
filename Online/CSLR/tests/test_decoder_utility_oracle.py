import importlib.util
import unittest
from pathlib import Path

import numpy as np


CSLR_ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = CSLR_ROOT / "tools/analyze_phoenix_decoder_utility_oracle.py"
SPEC = importlib.util.spec_from_file_location("decoder_utility_oracle", MODULE_PATH)
ORACLE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(ORACLE)


class DecoderUtilityOracleTests(unittest.TestCase):
    def test_half_budget_uses_nearest_integer_and_minimum_endpoints(self):
        self.assertEqual(ORACLE.dense_half_budget(16), 8)
        self.assertEqual(ORACLE.dense_half_budget(17), 9)
        self.assertEqual(ORACLE.dense_half_budget(1), 1)

    def test_skeleton_ratios_are_exact_and_endpoint_covering(self):
        budget = 20
        self.assertEqual(ORACLE.skeleton_size(budget, 0.0), 0)
        self.assertEqual(ORACLE.skeleton_size(budget, 0.25), 5)
        self.assertEqual(ORACLE.skeleton_size(budget, 0.5), 10)
        self.assertEqual(ORACLE.skeleton_size(budget, 0.75), 15)
        self.assertEqual(ORACLE.skeleton_size(budget, 1.0), 20)
        values = ORACLE.uniform_positions(41, 10)
        self.assertEqual(len(values), 10)
        self.assertEqual(values, sorted(set(values)))
        self.assertEqual((values[0], values[-1]), (0, 40))

    def test_greedy_schedule_is_exact_and_preserves_skeleton(self):
        probabilities = np.asarray(
            [
                [0.9, 0.1],
                [0.1, 0.9],
                [0.8, 0.2],
                [0.2, 0.8],
                [0.9, 0.1],
                [0.1, 0.9],
            ],
            dtype=np.float32,
        )
        vocab = ["<blank>", "A"]
        skeleton = [0, 5]
        selected, trace = ORACLE.greedy_schedule(
            probabilities, "A", 4, skeleton, vocab, blank_id=0
        )
        self.assertEqual(len(selected), 4)
        self.assertEqual(selected, sorted(set(selected)))
        self.assertTrue(set(skeleton).issubset(selected))
        self.assertEqual(len(trace), 2)

    def test_local_pair_enumerates_ten_pairs_for_pool_of_five(self):
        probabilities = np.tile(np.asarray([[0.6, 0.4]], dtype=np.float32), (8, 1))
        result = ORACLE.local_pair_diagnostic(
            probabilities, "A", [0, 7], ["<blank>", "A"], blank_id=0, pool_size=5
        )
        self.assertEqual(result["candidate_pool_size"], 5)
        self.assertEqual(result["pair_count"], 10)

    def test_incremental_candidate_decode_matches_full_decode(self):
        rng = np.random.default_rng(17)
        probabilities = rng.random((30, 7), dtype=np.float32)
        probabilities /= probabilities.sum(axis=1, keepdims=True)
        starts = [0, 4, 9, 15, 22, 29]
        selected, raw_scores, token_ids = ORACLE.decoder_state(probabilities, starts)
        vocab = ["<blank>", "A", "B", "C", "D", "E", "F"]
        for candidate in [2, 7, 14, 20, 27]:
            incremental = ORACLE.candidate_hypothesis(
                probabilities,
                selected,
                raw_scores,
                token_ids,
                candidate,
                vocab,
                blank_id=0,
            )
            full = ORACLE.decode_probabilities(
                probabilities, sorted([*starts, candidate]), vocab, blank_id=0
            )
            self.assertEqual(incremental, full)


if __name__ == "__main__":
    unittest.main()
