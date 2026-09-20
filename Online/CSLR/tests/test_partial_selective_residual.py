import importlib.util
import inspect
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PATH = ROOT / "tools/analyze_phoenix_partial_selective_residual.py"
SPEC = importlib.util.spec_from_file_location("selective_residual_tested", PATH)
M = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = M
SPEC.loader.exec_module(M)


class SelectiveResidualTests(unittest.TestCase):
    def test_protocol_is_safe_center_residual(self):
        p = M.config()["protocol"]
        self.assertEqual(p["default_bonus_offset"], 2)
        self.assertEqual(p["eligible_override_offsets"], [1, 3])
        self.assertEqual(p["max_gap"], 3)
        self.assertTrue(p["unknown_EOS"])

    def test_oof_is_source_grouped(self):
        o = M.config()["oof_selection"]
        self.assertEqual(o["folds"], 5)
        self.assertEqual(o["group"], "source_video_id")
        self.assertEqual(o["candidate_override_coverages"], [0.005, 0.01, 0.02, 0.05, 0.1])

    def test_loss_penalizes_false_override(self):
        src = inspect.getsource(M.train_one)
        self.assertIn("(target <= 0) & (pred > 0)", src)
        self.assertEqual(M.FALSE_OVERRIDE_COST, 4.0)

    def test_all_fold_gate_and_minimum_overrides(self):
        src = inspect.getsource(M.oof_select)
        self.assertIn("all(v >= 0", src)
        self.assertIn('report["overrides"] >= 20', src)

    def test_calibration_cannot_change_threshold(self):
        src = inspect.getsource(M.calibrate)
        self.assertIn('saved["threshold"]', src)
        self.assertNotIn("candidate_threshold", src)

    def test_evaluation_is_gated(self):
        src = inspect.getsource(M.evaluate)
        self.assertIn('calibration_passed_model_frozen', src)
        self.assertIn("evaluation_started.marker", src)

    def test_empty_policy_is_reported(self):
        self.assertIn("empty_policy", inspect.getsource(M.train))
        self.assertIn("empty_policy", inspect.getsource(M.calibrate))

    def test_no_gpu_or_dev_test(self):
        text = PATH.read_text()
        self.assertNotIn("cuda()", text)
        self.assertIn('os.environ["CUDA_VISIBLE_DEVICES"] = ""', text)


if __name__ == "__main__":
    unittest.main()
