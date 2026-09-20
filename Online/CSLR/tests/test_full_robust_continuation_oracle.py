import importlib.util
import sys
import unittest
from pathlib import Path


PATH = Path(__file__).resolve().parents[1] / "tools/analyze_phoenix_full_robust_continuation_oracle.py"
SPEC = importlib.util.spec_from_file_location("analyze_phoenix_full_robust_continuation_oracle", PATH)
FULL = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = FULL
SPEC.loader.exec_module(FULL)


class FullRobustContinuationOracleTest(unittest.TestCase):
    def test_frozen_scope_and_hashes(self):
        config = FULL.frozen_config()
        self.assertEqual(config["scope"]["samples"], 6378)
        self.assertEqual(config["scope"]["sources"], 578)
        self.assertEqual(config["checkpoint_sha256"], FULL.EXPECTED_CHECKPOINT)
        self.assertEqual(config["fit_calibration_split_sha256"], FULL.EXPECTED_SPLIT)

    def test_bootstrap_is_deterministic(self):
        rows = [
            {"source_video_id": "a", "robust_error": 0, "uniform_error": 1, "ref_len": 10},
            {"source_video_id": "b", "robust_error": 1, "uniform_error": 1, "ref_len": 10},
        ]
        self.assertEqual(FULL.source_bootstrap(rows, reps=100), FULL.source_bootstrap(rows, reps=100))


if __name__ == "__main__":
    unittest.main()
