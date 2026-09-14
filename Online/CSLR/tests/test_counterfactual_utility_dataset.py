import importlib.util
import unittest
from pathlib import Path

import numpy as np


CSLR_ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = CSLR_ROOT / "tools/build_phoenix_counterfactual_utility_dataset.py"
SPEC = importlib.util.spec_from_file_location("counterfactual_dataset", MODULE_PATH)
DATASET = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(DATASET)


class CounterfactualUtilityDatasetTests(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(31)
        self.probabilities = rng.random((10, 4), dtype=np.float32)
        self.probabilities /= self.probabilities.sum(axis=1, keepdims=True)
        self.vocab = ["<blank>", "A", "B", "C"]
        self.skeleton = [0, 4, 9]

    def make_record(self):
        return DATASET.state_record(
            "dev/artificial",
            self.probabilities,
            "A B",
            self.skeleton,
            [6],
            decision_step=1,
            total_budget=6,
            vocab=self.vocab,
            blank_id=0,
            pair_auxiliary_available=False,
        )[0]

    def test_every_unselected_candidate_is_present_once(self):
        record = self.make_record()
        expected = set(range(10)) - set(self.skeleton) - {6}
        observed = [row["candidate_window_start"] for row in record["candidates"]]
        self.assertEqual(set(observed), expected)
        self.assertEqual(len(observed), len(set(observed)))
        self.assertEqual(record["candidate_count"], len(expected))

    def test_utility_matches_independent_full_decode(self):
        record = self.make_record()
        selected = sorted([*self.skeleton, 6])
        before = DATASET.ORACLE.error_count(
            "A B",
            DATASET.ORACLE.decode_probabilities(
                self.probabilities, selected, self.vocab, blank_id=0
            ),
        )
        for candidate in record["candidates"]:
            start = candidate["candidate_window_start"]
            after = DATASET.ORACLE.error_count(
                "A B",
                DATASET.ORACLE.decode_probabilities(
                    self.probabilities, sorted([*selected, start]), self.vocab, blank_id=0
                ),
            )
            self.assertEqual(candidate["label"]["utility"], before - after)
            self.assertEqual(candidate["label"]["error_after"], after)

    def test_budget_and_selected_state_are_reconstructable(self):
        record = self.make_record()
        state = record["oracle_state"]
        reconstructed = sorted(
            [*state["initial_skeleton"], *state["selected_bonus_before"]]
        )
        self.assertEqual(state["selected_count"], len(reconstructed))
        self.assertEqual(
            state["selected_set_sha256"], DATASET.selected_set_hash(reconstructed)
        )
        self.assertEqual(state["exact_bonus_slots_remaining_known_only_offline"], 2)
        self.assertIsNone(state["token_bucket_balance"])

    def test_predictor_inputs_exclude_oracle_information(self):
        record = self.make_record()
        for candidate in record["candidates"]:
            DATASET.assert_predictor_inputs_clean(candidate["predictor_inputs"])
            serialized = str(candidate["predictor_inputs"]).lower()
            for token in ("reference", "logit", "future", "utility", "error"):
                self.assertNotIn(token, serialized)
        with self.assertRaisesRegex(ValueError, "oracle information leaked"):
            DATASET.assert_predictor_inputs_clean({"candidate_logits": [1.0]})

    def test_test_paths_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "test paths are forbidden"):
            DATASET.reject_test_path(Path("/tmp/test/forbidden.pkl"))


if __name__ == "__main__":
    unittest.main()
