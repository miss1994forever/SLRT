import importlib.util
import unittest
from pathlib import Path

import numpy as np


CSLR_ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = CSLR_ROOT / "tools" / "analyze_phoenix_schedule_replay.py"
SPEC = importlib.util.spec_from_file_location("schedule_replay", MODULE_PATH)
REPLAY = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(REPLAY)


class ScheduleReplayTests(unittest.TestCase):
    def test_dense_start_index_requires_stride_one_coverage(self):
        results = {
            "sample": {
                "adaptive_stride_metadata": [{"start": 0}, {"start": 2}],
            }
        }
        logits = {"sample": np.zeros((2, 3), dtype=np.float32)}
        with self.assertRaisesRegex(ValueError, "not dense stride-1"):
            REPLAY.dense_start_index(results, logits)

    def test_extract_schedule_logits_uses_exact_starts(self):
        dense_logits = {
            "sample": np.asarray([[0, 1], [2, 3], [4, 5]], dtype=np.float32)
        }
        indices = {"sample": {0: 0, 1: 1, 2: 2}}
        selected = REPLAY.extract_schedule_logits("sample", dense_logits, indices, [0, 2])
        np.testing.assert_array_equal(selected, np.asarray([[0, 1], [4, 5]], dtype=np.float32))

    def test_extract_schedule_logits_rejects_missing_start(self):
        with self.assertRaisesRegex(ValueError, "absent from dense B0"):
            REPLAY.extract_schedule_logits(
                "sample", {"sample": np.zeros((1, 2), dtype=np.float32)}, {"sample": {0: 0}}, [1]
            )

    def test_alignment_corrects_schedule_dependent_left_padding(self):
        aligned, dense_pad, target_pad = REPLAY.align_starts_to_dense_inputs(
            list(range(10)), [0, 2, 4, 6, 8]
        )
        self.assertEqual((dense_pad, target_pad), (7, 7))
        self.assertEqual(aligned, [0, 2, 4, 6, 8])

        aligned, dense_pad, target_pad = REPLAY.align_starts_to_dense_inputs(
            list(range(10)), [0, 3, 6]
        )
        self.assertEqual((dense_pad, target_pad), (7, 6))
        self.assertEqual(aligned, [1, 4, 7])

    def test_test_paths_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "test paths are forbidden"):
            REPLAY.reject_test_path(Path("/tmp/test/dev_results.pkl"))


if __name__ == "__main__":
    unittest.main()
