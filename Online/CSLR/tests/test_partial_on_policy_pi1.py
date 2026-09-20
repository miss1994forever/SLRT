import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "analyze_phoenix_partial_on_policy_pi1",
    ROOT / "tools/analyze_phoenix_partial_on_policy_pi1.py",
)
PI1 = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = PI1
SPEC.loader.exec_module(PI1)


class TestOnPolicyPI1(unittest.TestCase):
    def test_preregistered_primary_is_zero_and_one_iteration(self):
        value = PI1.config()
        self.assertEqual(value["policy_iterations"], 1)
        self.assertEqual(value["primary_threshold"], "predicted regression expected advantage > 0")
        self.assertFalse(value["calibration_threshold_tuning"])
        self.assertFalse(value["old_eager_labels_mixed"])

    def test_feature_value_has_only_causal_state(self):
        row = {
            "bookkeeping": {"candidate_absolute_index": 7,
              "candidate_mod_skeleton_period": 3, "coverage_gap": 3,
              "frames_since_last_execute": 3, "bonus_token_balance": 1.0,
              "past_selected_count": 2},
            "prefix": {"decoder_prefix_length": 1,"decoder_raw_path_length": 2,
              "decoder_mean_entropy": .2,"decoder_last_entropy": .3,
              "decoder_entropy_delta": .1,"decoder_last_is_blank": False,
              "decoder_last_repeats_previous": False,"decoder_last_repeat_run": 1,
              "decoder_token_hash32": [0.0] * 16},
        }
        value = PI1.feature_value(row)
        self.assertEqual(set(value), {"bookkeeping", "prefix", "visual"})
        self.assertEqual(value["visual"].shape, (31, 17))
        self.assertTrue(np.all(value["visual"] == 0))

    def test_preregister_is_immutable(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            PI1.preregister(root)
            self.assertEqual(json.loads((root / "resolved_config_preregistered.json").read_text()), PI1.config())
            changed = PI1.config(); changed["epochs"] = 6
            (root / "resolved_config_preregistered.json").write_text(json.dumps(changed))
            with self.assertRaises(ValueError): PI1.preregister(root)

    def test_sample_rows_uses_same_pi0_continuation_and_excludes_forced(self):
        fake_result = {"gls_ref": "A", "adaptive_stride_metadata": [{"start": i} for i in range(8)]}
        logits = np.zeros((8, 3), dtype=np.float32)
        fake_model = mock.Mock(); fake_stats = {"bookkeeping": (np.zeros(9), np.ones(9)),
          "prefix": (np.zeros(24), np.ones(24)), "visual": (np.zeros(17), np.ones(17))}
        with mock.patch.object(PI1.BUILDER.ORACLE, "softmax_rows", return_value=logits), \
             mock.patch.object(PI1.BUILDER.REPLAY, "clean_phoenix_2014_trans", return_value=["A"]), \
             mock.patch.object(PI1.OLD, "online_feature_row", return_value={
                 "bookkeeping": {"candidate_absolute_index": 1,"candidate_mod_skeleton_period": 1,
                   "coverage_gap": 1,"frames_since_last_execute": 1,"bonus_token_balance": 1.,"past_selected_count": 1},
                 "prefix": {"decoder_prefix_length": 0,"decoder_raw_path_length": 0,
                   "decoder_mean_entropy": 0.,"decoder_last_entropy": 0.,"decoder_entropy_delta": 0.,
                   "decoder_last_is_blank": True,"decoder_last_repeats_previous": False,
                   "decoder_last_repeat_run": 0,"decoder_token_hash32": [0.] * 16}}), \
             mock.patch.object(PI1.OLD, "online_score", return_value=-1.), \
             mock.patch.object(PI1.AUDIT, "on_policy_advantage", return_value=(1, 2, 1)) as advantage:
            rows, _ = PI1.sample_rows("x-1", fake_result, logits, ["<blank>", "A"], 0,
                                      fake_model, fake_stats, 0., mock.Mock())
        self.assertTrue(rows)
        self.assertTrue(all(not row["provenance"]["forced_state"] for row in rows))
        self.assertTrue(all(row["provenance"]["continuation_same_frozen_pi0"] for row in rows))
        self.assertTrue(advantage.called)


if __name__ == "__main__":
    unittest.main()
