import importlib.util
import inspect
import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
PATH = ROOT / "tools/analyze_phoenix_partial_sparse_decoder_mismatch.py"
SPEC = importlib.util.spec_from_file_location("sparse_decoder_mismatch_tested", PATH)
M = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = M
SPEC.loader.exec_module(M)


class SparseDecoderMismatchTests(unittest.TestCase):
    def test_scope_excludes_fresh_replication(self):
        source = PATH.read_text()
        self.assertNotIn("fresh_disjoint", source)
        self.assertTrue(M.config()["old_fixed128_only"])

    def test_split_is_source_disjoint(self):
        split = M.config()["split"]
        self.assertFalse(set(split["tune_source_videos"]) & set(split["audit_source_videos"]))
        self.assertEqual(len(split["tune"]) + len(split["audit"]), 128)

    def test_identity_calibration_matches_softmax(self):
        logits = np.asarray([[1.0, 2.0, -1.0], [0.0, 0.0, 0.0]], dtype=np.float32)
        got = M.calibrated_probabilities(logits, 1.0, 0.0, 0)
        expected = np.exp(logits - logits.max(1, keepdims=True))
        expected /= expected.sum(1, keepdims=True)
        np.testing.assert_allclose(got, expected, rtol=1e-6, atol=1e-7)

    def test_blank_bias_only_changes_blank_logit(self):
        logits = np.zeros((1, 3), dtype=np.float32)
        got = M.calibrated_probabilities(logits, 1.0, 1.0, 1)[0]
        self.assertGreater(got[1], got[0])
        self.assertAlmostEqual(float(got[0]), float(got[2]), places=7)

    def test_schedule_is_frozen_before_calibration(self):
        source = inspect.getsource(M.evaluate)
        self.assertIn('sample["schedules"][schedule]', source)
        self.assertNotIn("full_nonblank_schedule", source)

    def test_cpu_only(self):
        self.assertNotIn("cuda()", PATH.read_text())
        self.assertTrue(M.config()["cpu_only"])


if __name__ == "__main__":
    unittest.main()
