import importlib.util
import unittest
from pathlib import Path

import numpy as np


CSLR_ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = CSLR_ROOT / "tools/analyze_phoenix_sign_center_oracles.py"
SPEC = importlib.util.spec_from_file_location("sign_center_oracles", MODULE_PATH)
CENTER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(CENTER)


class SignCenterOracleTests(unittest.TestCase):
    def test_segment_midpoint_odd_and_even_lengths(self):
        self.assertEqual(CENTER.segment_midpoint(4, 9), 6.0)
        self.assertEqual(CENTER.segment_midpoint(4, 10), 6.5)

    def test_center_band_is_central_closed_half(self):
        self.assertEqual(CENTER.segment_center_band(4, 9), (5.0, 7.0))
        self.assertEqual(CENTER.segment_center_band(4, 10), (5.25, 7.75))

    def test_budget_definitions(self):
        self.assertEqual(CENTER.budget_for(101, 68, "a0_exact"), 68)
        self.assertEqual(CENTER.budget_for(101, 68, "dense_50pct"), 51)
        self.assertEqual(CENTER.budget_for(100, 68, "dense_33pct"), 33)

    def test_center_schedule_exact_unique_deterministic_endpoints_and_skeleton(self):
        first = CENTER.center_schedule(30, 19, [(6, 12), (20, 25)], "midpoint")
        second = CENTER.center_schedule(30, 19, [(6, 12), (20, 25)], "midpoint")
        self.assertEqual(first, second)
        self.assertEqual(len(first), 19)
        self.assertEqual(first, sorted(set(first)))
        self.assertEqual((first[0], first[-1]), (0, 29))
        skeleton = CENTER.STRUCTURED.uniform_positions(
            30, CENTER.STRUCTURED.skeleton_count(30, 19)
        )
        self.assertTrue(set(skeleton).issubset(first))

    def test_three_decoder_definitions(self):
        vocab = ["<blank>", "a", "b"]
        logits = np.asarray([
            [0.0, 4.0, 0.0],
            [0.0, 4.0, 0.0],
            [4.0, 0.0, 0.0],
            [0.0, 0.0, 4.0],
        ], dtype=np.float32)
        starts = [0, 1, 4, 8]
        direct = CENTER.decode(logits, starts, vocab, 0, "direct_span1")
        span7 = CENTER.decode(logits, starts, vocab, 0, "span7")
        span15 = CENTER.decode(logits, starts, vocab, 0, "span15")
        self.assertEqual(direct, "A B")
        self.assertIsInstance(span7, str)
        self.assertIsInstance(span15, str)

    def test_fixed_coordinate_mapping(self):
        aligned, dense_pad, target_pad = CENTER.REPLAY.align_starts_to_dense_inputs(
            list(range(100)), [0, 3, 6, 97]
        )
        self.assertEqual((dense_pad, target_pad), (7, 6))
        self.assertEqual(aligned, [1, 4, 7, 98])

    def test_test_paths_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "test paths are forbidden"):
            CENTER.load_segment_proxies(Path("/tmp/test/meta"), Path("/tmp/dev/labels"), [])


if __name__ == "__main__":
    unittest.main()
