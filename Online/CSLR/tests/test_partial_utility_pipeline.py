import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
PATH = ROOT / "tools/analyze_phoenix_partial_utility_pipeline.py"
SPEC = importlib.util.spec_from_file_location("partial_utility_pipeline", PATH)
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


class PartialUtilityPipelineTests(unittest.TestCase):
    def test_raw_features_use_only_clean_predictor_inputs(self):
        inputs = {
            "candidate_window_start": 8,
            "selected_past_count": 2,
            "frames_since_last_selected_past": 3,
            "has_selected_past": True,
        }
        value = MODULE.raw_features(inputs)
        self.assertEqual(value.shape, (4,))
        with self.assertRaisesRegex(ValueError, "oracle information leaked"):
            MODULE.raw_features({**inputs, "candidate_logits": [1.0]})

    def test_average_precision_known_example(self):
        labels = [1, 0, 1, 0]
        scores = [0.9, 0.8, 0.7, 0.1]
        self.assertAlmostEqual(MODULE.average_precision(labels, scores), (1.0 + 2 / 3) / 2)

    def test_logistic_fit_ranks_separable_rows(self):
        x = np.asarray([[0, 0, 0, 0], [0.1, 0, 0, 0], [2, 0, 0, 0], [3, 0, 0, 0]])
        y = np.asarray([0, 0, 1, 1], dtype=float)
        model = MODULE.train_logistic(x, y)
        scores = MODULE.predict(model, x)
        self.assertGreater(scores[2:].min(), scores[:2].max())

    def test_learned_schedule_is_exact_budget_and_contains_skeleton(self):
        matrix = np.asarray(
            [[0, 0, 0, 0], [1, 0, 0, 0], [2, 1, 1, 1], [3, 1, 1, 1]], dtype=float
        )
        target = np.asarray([0, 0, 1, 1], dtype=float)
        model = MODULE.train_logistic(matrix, target)
        total_windows = 12
        schedule = MODULE.learned_schedule(total_windows, model)
        budget = MODULE.ORACLE.dense_half_budget(total_windows)
        skeleton_count = MODULE.ORACLE.skeleton_size(budget, MODULE.ORACLE.PRIMARY_RATIO)
        skeleton = MODULE.ORACLE.uniform_positions(total_windows, skeleton_count)
        self.assertEqual(len(schedule), budget)
        self.assertEqual(len(set(schedule)), budget)
        self.assertTrue(set(skeleton).issubset(schedule))

    def test_partial_partition_summary_is_source_disjoint(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "shards").mkdir()
            _, complete = MODULE.BUILDER.output_paths(root, 0, 1)
            complete.write_text(json.dumps({
                "sample_ids": ["train/source-a-1", "train/source-a-2", "train/source-b-1"]
            }))
            assignments = {
                "train/source-a-1": "fit", "train/source-a-2": "fit",
                "train/source-b-1": "calibration",
            }
            summary = MODULE.partial_partition_summary(root, [0], assignments, 1)
            self.assertEqual(summary["source_video_overlap"], 0)
            self.assertEqual(summary["source_videos"], {"fit": 1, "calibration": 1})


if __name__ == "__main__":
    unittest.main()
