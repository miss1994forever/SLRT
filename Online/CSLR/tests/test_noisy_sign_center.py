import importlib.util
import unittest
from pathlib import Path

import numpy as np


CSLR_ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = CSLR_ROOT / "tools/analyze_phoenix_noisy_sign_center.py"
SPEC = importlib.util.spec_from_file_location("noisy_sign_center", MODULE_PATH)
NOISY = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(NOISY)


class NoisySignCenterTests(unittest.TestCase):
    def test_offset_clips_to_physical_input_range(self):
        self.assertEqual(NOISY.offset_centers([0.5, 8.0], -3, 10), [0.0, 5.0])
        self.assertEqual(NOISY.offset_centers([0.5, 8.0], 4, 10), [4.5, 9.0])

    def test_jitter_is_deterministic_and_bounded(self):
        centers = [3.5, 9.0, 15.5]
        first = NOISY.jitter_centers(centers, 4, 20, 1829, "sample")
        second = NOISY.jitter_centers(centers, 4, 20, 1829, "sample")
        self.assertEqual(first, second)
        self.assertTrue(all(abs(a - b) <= 4 for a, b in zip(first, centers)))

    def test_dropout_rounding_ratio_and_determinism(self):
        centers = list(range(10))
        first = NOISY.dropout_centers(centers, 0.75, 1829, "sample")
        second = NOISY.dropout_centers(centers, 0.75, 1829, "sample")
        self.assertEqual(first, second)
        self.assertEqual(len(first), 8)
        with self.assertRaises(ValueError):
            NOISY.dropout_centers(centers, 1.1, 1829, "sample")

    def test_false_positive_count_exclusion_and_determinism(self):
        true = [2.0, 4.5, 7.0]
        first = NOISY.false_centers(true, 1.0, 10, 1829, "sample")
        second = NOISY.false_centers(true, 1.0, 10, 1829, "sample")
        self.assertEqual(first, second)
        self.assertEqual(len(first), 3)
        self.assertTrue(all(all(abs(value - center) > 0.25 for center in true) for value in first))

    def test_uniform_fallback_is_exact_unique_and_endpoint_covering(self):
        skeleton = NOISY.STRUCTURED.uniform_positions(
            31, NOISY.STRUCTURED.skeleton_count(31, 17)
        )
        result = NOISY.uniform_fallback_fill(31, 17, skeleton)
        self.assertEqual(len(result), 17)
        self.assertEqual(result, sorted(set(result)))
        self.assertEqual((result[0], result[-1]), (0, 30))
        self.assertTrue(set(skeleton).issubset(result))

    def test_dropout_schedule_keeps_perfect_at_full_recall(self):
        perfect = NOISY.schedule_from_centers(31, 17, [6.0, 15.5, 25.0])
        observed = NOISY.dropout_schedule(31, 17, [6.0, 15.5, 25.0], 1.0, 1829, "sample")
        self.assertEqual(observed, perfect)

    def test_noisy_schedule_exact_unique_deterministic_endpoints(self):
        centers = NOISY.jitter_centers([6.0, 15.5, 25.0], 4, 31, 1829, "sample")
        first = NOISY.schedule_from_centers(31, 17, centers)
        second = NOISY.schedule_from_centers(31, 17, centers)
        self.assertEqual(first, second)
        self.assertEqual(len(first), 17)
        self.assertEqual(first, sorted(set(first)))
        self.assertEqual((first[0], first[-1]), (0, 30))

    def test_vectorized_decoder_matches_registered_decoder(self):
        rng = np.random.default_rng(7)
        logits = rng.normal(size=(20, 5)).astype(np.float32)
        starts = list(range(20))
        names = ["sample"]
        schedules = {"sample": [0, 2, 5, 6, 10, 14, 19]}
        dense_logits = {"sample": logits}
        start_indices = {"sample": starts}
        vocab = ["<blank>", "a", "b", "c", "d"]
        expected = NOISY.CENTER.decode_schedules(
            names, schedules, dense_logits, start_indices, vocab, 0, "span15"
        )
        actual = NOISY.decode_schedules_fast(
            names, schedules, dense_logits, start_indices, vocab, 0
        )
        self.assertEqual(actual, expected)

    def test_fixed_coordinate_mapping_and_test_path_rejection(self):
        aligned, dense_pad, target_pad = NOISY.REPLAY.align_starts_to_dense_inputs(
            list(range(100)), [0, 3, 6, 97]
        )
        self.assertEqual((dense_pad, target_pad), (7, 6))
        self.assertEqual(aligned, [1, 4, 7, 98])
        with self.assertRaisesRegex(ValueError, "test paths are forbidden"):
            NOISY.REPLAY.reject_test_path(Path("/tmp/phoenix/test/results.pkl"))


if __name__ == "__main__":
    unittest.main()
