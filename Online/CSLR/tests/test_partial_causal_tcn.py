import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
PATH = ROOT / "tools/analyze_phoenix_partial_causal_tcn.py"
SPEC = importlib.util.spec_from_file_location("partial_causal_tcn", PATH)
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


class PartialCausalTCNTests(unittest.TestCase):
    def archive(self, path, frames):
        names = np.asarray(MODULE.TEMPORAL_FEATURE_NAMES)
        np.savez(path, features=frames, video_names=np.asarray(["train/source-1"]),
                 video_offsets=np.asarray([0, len(frames)]), feature_names=names)

    def test_history_ends_at_window_availability_and_keeps_order(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "train_sequences.npz"
            frames = np.tile(np.arange(40, dtype=np.float32)[:, None],
                             (1, len(MODULE.TEMPORAL_FEATURE_NAMES)))
            self.archive(path, frames)
            history = MODULE.TemporalSequenceFeatures(path).history("train/source-1", 10)
            self.assertEqual(history.shape, (31, len(MODULE.TEMPORAL_FEATURE_NAMES)))
            self.assertEqual(history[-1, 0], 18)
            np.testing.assert_array_equal(history[-19:, 0], np.arange(19))
            np.testing.assert_array_equal(history[:12, 0], 0)

    def test_future_perturbation_cannot_change_history_or_prediction(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "train_sequences.npz"
            frames = np.tile(np.arange(40, dtype=np.float32)[:, None],
                             (1, len(MODULE.TEMPORAL_FEATURE_NAMES)))
            self.archive(path, frames)
            first = MODULE.TemporalSequenceFeatures(path).history("train/source-1", 10).copy()
            frames[19:] = 9999
            self.archive(path, frames)
            second = MODULE.TemporalSequenceFeatures(path).history("train/source-1", 10)
            np.testing.assert_array_equal(first, second)
            MODULE.set_determinism(3)
            network = MODULE.TinyCausalTCN(len(MODULE.TEMPORAL_FEATURE_NAMES))
            bookkeeping = torch.zeros((1, 4))
            with torch.no_grad():
                score1 = network(bookkeeping, torch.from_numpy(first[None])).item()
                score2 = network(bookkeeping, torch.from_numpy(second[None])).item()
            self.assertEqual(score1, score2)

    def test_fixed_shuffle_changes_order_but_not_values(self):
        permutation = MODULE.fixed_time_permutation()
        self.assertEqual(sorted(permutation.tolist()), list(range(MODULE.HISTORY_FRAMES)))
        self.assertFalse(np.array_equal(permutation, np.arange(MODULE.HISTORY_FRAMES)))
        history = np.arange(MODULE.HISTORY_FRAMES * 2).reshape(MODULE.HISTORY_FRAMES, 2)
        np.testing.assert_array_equal(np.sort(history[permutation], axis=0),
                                      np.sort(history, axis=0))
        self.assertFalse(np.array_equal(history[permutation], history))

    def test_tcn_shape_and_order_sensitivity(self):
        MODULE.set_determinism(5)
        network = MODULE.TinyCausalTCN(3)
        bookkeeping = torch.zeros((2, 4))
        history = torch.arange(2 * 31 * 3, dtype=torch.float32).reshape(2, 31, 3)
        output = network(bookkeeping, history)
        reversed_output = network(bookkeeping, history.flip(1))
        self.assertEqual(tuple(output.shape), (2,))
        self.assertFalse(torch.equal(output, reversed_output))

    def test_prefix_audit_refuses_nonchronological_state(self):
        audit = MODULE.audit_decoder_prefix(Path("train_partial_utility"))
        self.assertFalse(audit["included"])
        self.assertIn("future", audit["reason"])
        self.assertIn("frame-arrival order", audit["required_reconstruction"])


if __name__ == "__main__":
    unittest.main()
