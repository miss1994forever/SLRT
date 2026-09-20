import importlib.util
import inspect
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PATH = ROOT / "tools/analyze_phoenix_partial_hog_preview_oof.py"
SPEC = importlib.util.spec_from_file_location("hog_preview_oof_tested", PATH)
M = importlib.util.module_from_spec(SPEC); sys.modules[SPEC.name] = M; SPEC.loader.exec_module(M)


class HogPreviewOOFTests(unittest.TestCase):
    def test_frozen_subset(self):
        c = M.config(); self.assertEqual(c["subset"]["samples"], 512); self.assertEqual(c["subset"]["sources"], 211)

    def test_representation_is_low_complexity(self):
        r = M.config()["candidate_representation"]
        self.assertEqual(r["HOG_model_width"], 189); self.assertEqual(r["no_HOG_model_width"], 117); self.assertEqual(r["PCA"], "none")

    def test_fold_matches_selective_residual(self):
        self.assertEqual(M.FOLD_SEED, 261022); self.assertEqual(M.FOLDS, 5)

    def test_cache_has_no_temporal_fill(self):
        c = M.config()["cache"]; self.assertIn("invalid is zeros", c["crop"]); self.assertIn("no future fill", c["causality"])

    def test_fold_internal_normalization(self):
        src = inspect.getsource(M.oof_predictions); self.assertIn("train_one(data, train_mask", src)

    def test_gate_compares_baseline(self):
        src = inspect.getsource(M.run_oof); self.assertIn('"reward_gt_no_HOG"', src); self.assertIn('"regret_lt_no_HOG"', src)

    def test_no_calibration_or_gpu(self):
        text = PATH.read_text(); self.assertNotIn("cuda()", text); self.assertNotIn("calibration_started", text); self.assertIn('os.environ["CUDA_VISIBLE_DEVICES"] = ""', text)


if __name__ == "__main__": unittest.main()
