import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
PATH = ROOT / "tools/analyze_phoenix_partial_rollout_predictor.py"
SPEC = importlib.util.spec_from_file_location("partial_rollout_predictor", PATH)
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


class PartialRolloutPredictorTests(unittest.TestCase):
    def sequence(self, frames):
        directory = tempfile.TemporaryDirectory()
        path = Path(directory.name) / "train_sequence.npz"
        frames = np.asarray(frames, dtype=np.float32)
        frames = np.repeat(frames, len(MODULE.TEMPORAL_FEATURE_NAMES), axis=1)
        np.savez(path, features=frames,
                 video_names=np.asarray(["train/foo-1"]), video_offsets=np.asarray([0, len(frames)]),
                 feature_names=np.asarray(MODULE.TEMPORAL_FEATURE_NAMES))
        return directory, MODULE.SequenceFeatures(path)

    def test_visual_history_ends_at_window_availability_and_is_causal(self):
        directory, sequence = self.sequence(np.arange(50)[:, None])
        try:
            history = sequence.history("train/foo-1", 20)
            self.assertEqual(history.shape, (31, len(MODULE.TEMPORAL_FEATURE_NAMES)))
            self.assertEqual(history[-1, 0], 28)
            self.assertEqual(history[0, 0], 0)
        finally:
            directory.cleanup()

    def test_visual_history_clamps_after_observed_eos(self):
        directory, sequence = self.sequence(np.arange(20)[:, None])
        try:
            history = sequence.history("train/foo-1", 12)
            self.assertEqual(history[-1, 0], 19)
        finally:
            directory.cleanup()

    def test_variants_have_frozen_input_mapping(self):
        self.assertEqual(MODULE.VARIANTS, {
            "B0_bookkeeping": (False, False),
            "B1_bookkeeping_visual_tcn": (False, True),
            "B2_bookkeeping_prefix": (True, False),
            "B3_bookkeeping_visual_tcn_prefix": (True, True),
        })

    def test_regression_score_metrics_rank_known_advantage(self):
        advantages = np.asarray([-1, 0, 0, 1, 2] * 20, dtype=float)
        labels = advantages > 0
        metrics = MODULE.evaluation_metrics(labels, advantages, advantages, labels.astype(float))
        self.assertAlmostEqual(metrics["pr_auc_regression_score"], 1.0)
        self.assertGreater(metrics["score_decile_spearman"], 0)
        self.assertGreater(metrics["highest_minus_lowest_decile_mean_advantage"], 0)

    def test_gate_requires_all_preregistered_checks(self):
        base = {"pr_auc_regression_score": .1, "positive_recall_at_top_10pct": .2,
                "ndcg_at_top_10pct": .3, "score_decile_spearman": .1,
                "highest_minus_lowest_decile_mean_advantage": .1}
        better = dict(base, pr_auc_regression_score=.2, positive_recall_at_top_10pct=.3,
                      ndcg_at_top_10pct=.4)
        values = {"B0_bookkeeping": base, "B3_bookkeeping_visual_tcn_prefix": better}
        self.assertTrue(MODULE.preregistered_gate(values)["passed"])
        values["B3_bookkeeping_visual_tcn_prefix"] = dict(better, ndcg_at_top_10pct=.2)
        self.assertFalse(MODULE.preregistered_gate(values)["passed"])

    def test_model_heads_have_expected_shape(self):
        model = MODULE.RolloutPredictor(9, 40, 3, True, True)
        regression, positive = model(
            np_to_tensor(np.zeros((2, 9), dtype=np.float32)),
            np_to_tensor(np.zeros((2, 40), dtype=np.float32)),
            np_to_tensor(np.zeros((2, 31, 3), dtype=np.float32)),
        )
        self.assertEqual(tuple(regression.shape), (2,))
        self.assertEqual(tuple(positive.shape), (2,))


def np_to_tensor(value):
    import torch
    return torch.from_numpy(value)


if __name__ == "__main__":
    unittest.main()
