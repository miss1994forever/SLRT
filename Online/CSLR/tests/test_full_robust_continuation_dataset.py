import importlib.util
import sys
import unittest
from pathlib import Path

import numpy as np


PATH = Path(__file__).resolve().parents[1] / "tools/build_phoenix_full_robust_continuation_dataset.py"
SPEC = importlib.util.spec_from_file_location("build_phoenix_full_robust_continuation_dataset", PATH)
DATASET = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = DATASET
SPEC.loader.exec_module(DATASET)


class FullRobustContinuationDatasetTest(unittest.TestCase):
    def test_frozen_config_matches_completed_oracle(self):
        config = DATASET.frozen_config()
        self.assertEqual(config["scope"]["samples"], 6378)
        self.assertEqual(config["scope"]["sources"], 578)
        self.assertTrue(config["teacher"]["uses_reference_future_and_EOS"])
        self.assertFalse(config["predictor_inputs"]["candidate_expensive_logits"])

    def test_shared_prefix_optimization_is_numerically_equivalent(self):
        rng = np.random.default_rng(7)
        probabilities = rng.random((12, 6))
        probabilities /= probabilities.sum(axis=1, keepdims=True)
        selected = [0, 2, 4]
        candidates = [5, 6, 7]
        expected = DATASET.BLOCK.candidate_features(
            probabilities, selected, candidates, blank_id=0
        )
        actual = DATASET.shared_prefix_candidate_features(
            probabilities, selected, candidates, blank=0
        )
        for observed, reference in zip(actual, expected):
            self.assertEqual(observed["bookkeeping"], reference["bookkeeping"])
            np.testing.assert_allclose(observed["prefix"], reference["prefix"], rtol=1e-6, atol=1e-6)
            # Lengths, token-state flags and hashed token identities must be exact;
            # only entropy reductions may differ by float summation order.
            for index in [0, 1, 5, 6, 7, *range(8, len(reference["prefix"]))]:
                self.assertEqual(observed["prefix"][index], reference["prefix"][index])
        old_starts, old_scores, old_tokens = DATASET.BUILDER.ORACLE.decoder_state(
            probabilities, selected
        )
        new_starts, new_scores, new_tokens = DATASET.banded_decoder_state(
            probabilities, selected
        )
        np.testing.assert_array_equal(new_starts, old_starts)
        np.testing.assert_allclose(new_scores, old_scores, rtol=1e-6, atol=1e-6)
        np.testing.assert_array_equal(new_tokens, old_tokens)


if __name__ == "__main__":
    unittest.main()
