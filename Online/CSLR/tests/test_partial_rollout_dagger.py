import importlib.util
import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
PATH = ROOT / "tools/build_phoenix_partial_rollout_dagger_self_dataset.py"
SPEC = importlib.util.spec_from_file_location("rollout_dagger_builder", PATH)
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)

ANALYZE_PATH = ROOT / "tools/analyze_phoenix_partial_rollout_dagger_da1.py"
ANALYZE_SPEC = importlib.util.spec_from_file_location("rollout_dagger_analyzer", ANALYZE_PATH)
ANALYZER = importlib.util.module_from_spec(ANALYZE_SPEC)
sys.modules[ANALYZE_SPEC.name] = ANALYZER
ANALYZE_SPEC.loader.exec_module(ANALYZER)


class RolloutDaggerBuilderTests(unittest.TestCase):
    def test_only_one_aggregation_round_and_frozen_variants(self):
        self.assertEqual(MODULE.KINDS, (
            "B2_bookkeeping_prefix", "B3_bookkeeping_visual_tcn_prefix"))

    def test_self_prefix_uses_own_past_and_not_candidate(self):
        rng = np.random.default_rng(17)
        probabilities = rng.random((24, 8)); probabilities /= probabilities.sum(1, keepdims=True)
        selected = [0, 3, 4, 7]
        before = MODULE.ANALYZE.online_feature_row(probabilities, selected, 8, 1.0, 0)
        changed = probabilities.copy(); changed[8:] = changed[8:, ::-1]
        after = MODULE.ANALYZE.online_feature_row(changed, selected, 8, 1.0, 0)
        self.assertEqual(before, after)

    def test_future_changes_label_but_not_input(self):
        rng = np.random.default_rng(22)
        probabilities = rng.random((30, 5)); probabilities /= probabilities.sum(1, keepdims=True)
        selected, current = [0, 3, 4], 7
        before = MODULE.ANALYZE.online_feature_row(probabilities, selected, current, 1.0, 0)
        changed = probabilities.copy(); changed[current:] = np.roll(changed[current:], 1, axis=1)
        after = MODULE.ANALYZE.online_feature_row(changed, selected, current, 1.0, 0)
        self.assertEqual(before, after)
        # The test establishes the permitted asymmetry: label code may inspect
        # changed future probabilities, while the serialized input cannot.
        self.assertFalse(np.array_equal(probabilities[current:], changed[current:]))

    def test_unknown_total_absent_from_online_row(self):
        rng = np.random.default_rng(3)
        probabilities = rng.random((12, 4)); probabilities /= probabilities.sum(1, keepdims=True)
        row = MODULE.ANALYZE.online_feature_row(probabilities, [0, 3, 4], 7, 1.0, 0)
        self.assertNotIn("total_windows", row["bookkeeping"])
        self.assertNotIn("eos", row["bookkeeping"])

    def test_teacher_self_rows_have_exact_equal_source_weight(self):
        weights = ANALYZER.source_balanced_row_weights(3, 11)
        self.assertAlmostEqual(weights[:3].sum(), .5)
        self.assertAlmostEqual(weights[3:].sum(), .5)

    def test_weighted_threshold_reads_only_supplied_fit_scores(self):
        teacher = np.arange(10, dtype=float)
        self_values = np.arange(100, 110, dtype=float)
        weights = ANALYZER.source_balanced_row_weights(len(teacher), len(self_values))
        threshold = ANALYZER.weighted_quantile(np.r_[teacher, self_values], weights, .9)
        self.assertGreaterEqual(threshold, 100)

    def test_preregistered_analyzer_runs_only_da1(self):
        text = ANALYZE_PATH.read_text()
        self.assertIn('"dataset_aggregation_round":1', text)
        self.assertNotIn("DA2", text)


if __name__ == "__main__": unittest.main()
