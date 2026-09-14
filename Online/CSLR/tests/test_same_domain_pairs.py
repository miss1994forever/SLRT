import importlib.util
import math
import unittest
from pathlib import Path

import numpy as np


CSLR_ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = CSLR_ROOT / "tools/analyze_phoenix_same_domain_pairs.py"
SPEC = importlib.util.spec_from_file_location("same_domain_pairs", MODULE_PATH)
PAIRS = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(PAIRS)


class SameDomainPairTests(unittest.TestCase):
    def test_exhaustive_and_greedy_use_the_same_full_domain(self):
        rng = np.random.default_rng(3)
        probabilities = rng.random((12, 4), dtype=np.float32)
        probabilities /= probabilities.sum(axis=1, keepdims=True)
        skeleton = [0, 6, 11]
        result = PAIRS.same_domain_pair_diagnostic(
            probabilities,
            "A B",
            skeleton,
            ["<blank>", "A", "B", "C"],
            blank_id=0,
        )
        domain_size = len(probabilities) - len(skeleton)
        self.assertEqual(result["candidate_domain_size"], domain_size)
        self.assertEqual(result["pair_count"], math.comb(domain_size, 2))
        self.assertTrue(set(result["greedy_pair"]).isdisjoint(skeleton))
        self.assertTrue(set(result["best_pair"]).isdisjoint(skeleton))
        self.assertGreaterEqual(result["greedy_regret"], 0)
        self.assertEqual(
            result["best_minus_greedy_error"], -result["greedy_regret"]
        )

    def test_zero_plus_zero_can_have_positive_joint_utility(self):
        probabilities = np.asarray(
            [
                [0.011962535, 0.5286516, 0.45938587],
                [0.045485035, 0.7682851, 0.18622994],
                [0.32996628, 0.23849066, 0.43154308],
                [0.13168553, 0.26036966, 0.60794485],
                [0.24744624, 0.50616354, 0.24639024],
                [0.14671676, 0.4184211, 0.43486214],
                [0.48507026, 0.5008196, 0.014110106],
                [0.4999211, 0.26134583, 0.23873301],
            ],
            dtype=np.float32,
        )
        result = PAIRS.same_domain_pair_diagnostic(
            probabilities,
            "A B",
            [0, 7],
            ["<blank>", "A", "B"],
            blank_id=0,
        )
        self.assertTrue(result["all_single_utilities_zero"])
        self.assertEqual(result["zero_zero_positive_joint_pair_count"], 1)
        self.assertEqual(result["zero_zero_best_joint_utility"], 1)

    def test_exact_pair_can_beat_sequential_greedy(self):
        probabilities = np.asarray(
            [
                [0.000038081, 0.27507183, 0.35442457, 0.37046546],
                [0.6091475, 0.24648875, 0.008876978, 0.13548678],
                [0.346126, 0.24234417, 0.08201265, 0.32951725],
                [0.27394003, 0.41276932, 0.08984814, 0.22344257],
                [0.1474415, 0.07116956, 0.08067345, 0.7007155],
                [0.055931848, 0.3856223, 0.29390666, 0.26453918],
                [0.2597558, 0.017866034, 0.29892445, 0.42345375],
                [0.092918046, 0.116556324, 0.36808613, 0.42243952],
                [0.37157473, 0.2511648, 0.03503544, 0.34222507],
            ],
            dtype=np.float32,
        )
        result = PAIRS.same_domain_pair_diagnostic(
            probabilities,
            "A C",
            [0, 8],
            ["<blank>", "A", "B", "C"],
            blank_id=0,
        )
        self.assertEqual(result["greedy_regret"], 1)
        self.assertEqual(result["best_minus_greedy_error"], -1)


if __name__ == "__main__":
    unittest.main()
