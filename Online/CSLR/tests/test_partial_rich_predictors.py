import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
PATH = ROOT / "tools/analyze_phoenix_partial_rich_predictors.py"
SPEC = importlib.util.spec_from_file_location("partial_rich_predictors", PATH)
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


class PartialRichPredictorTests(unittest.TestCase):
    def test_descriptor_is_causal_and_uses_window_availability_frame(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "train_sequences.npz"
            frames = np.arange(40, dtype=np.float32)[:, None]
            np.savez(path, features=frames, video_names=np.asarray(["train/source-1"]),
                     video_offsets=np.asarray([0, 40]), feature_names=np.asarray(["value"]))
            sequence = MODULE.SequenceFeatures(path)
            value = sequence.candidate("train/source-1", 10)
            # centered 16-frame window start 10 is available after original frame 18
            self.assertEqual(value[0], 18)
            self.assertEqual(value[1], np.arange(19).mean())

    def test_descriptor_never_changes_when_future_changes(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "train_sequences.npz"
            frames = np.arange(40, dtype=np.float32)[:, None]
            np.savez(path, features=frames, video_names=np.asarray(["train/source-1"]),
                     video_offsets=np.asarray([0, 40]), feature_names=np.asarray(["value"]))
            first = MODULE.SequenceFeatures(path).candidate("train/source-1", 10).copy()
            frames[19:] = 9999
            np.savez(path, features=frames, video_names=np.asarray(["train/source-1"]),
                     video_offsets=np.asarray([0, 40]), feature_names=np.asarray(["value"]))
            second = MODULE.SequenceFeatures(path).candidate("train/source-1", 10)
            np.testing.assert_array_equal(first, second)

    def test_mlp_learns_separable_rows(self):
        MODULE.set_determinism(7)
        x = np.asarray([[0], [0.1], [0.2], [2], [2.1], [2.2]], dtype=np.float32)
        y = np.asarray([0, 0, 0, 1, 1, 1], dtype=np.float32)
        model = MODULE.train_mlp(x, y, epochs=30, batch_size=6, seed=7)
        score = MODULE.predict_mlp(model, x)
        self.assertGreater(score[3:].min(), score[:3].max())

    def test_feature_ablation_dimensions(self):
        bookkeeping = np.zeros((2, 4), dtype=np.float32)
        visual = np.zeros((2, 6), dtype=np.float32)
        self.assertEqual(MODULE.feature_matrix("bookkeeping_mlp", bookkeeping, visual).shape, (2, 4))
        self.assertEqual(MODULE.feature_matrix("bookkeeping_linear", bookkeeping, visual).shape, (2, 4))
        self.assertEqual(
            MODULE.feature_matrix("causal_pose_handshape_mlp", bookkeeping, visual).shape, (2, 10)
        )


if __name__ == "__main__":
    unittest.main()
