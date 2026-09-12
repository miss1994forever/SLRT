import importlib.util
import unittest
from pathlib import Path


CSLR_ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = CSLR_ROOT / "tools/analyze_phoenix_boundary_proxy_oracles.py"
SPEC = importlib.util.spec_from_file_location("boundary_proxy_oracles", MODULE_PATH)
BOUNDARY = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(BOUNDARY)


class BoundaryProxyOracleTests(unittest.TestCase):
    def test_first_record_per_bag_retains_only_first(self):
        records = [
            {"bag": 3, "start": 4},
            {"bag": 3, "start": 5},
            {"bag": 4, "start": 8},
        ]
        result = BOUNDARY.first_record_per_bag(records)
        self.assertEqual(result[3]["start"], 4)
        self.assertEqual(result[4]["start"], 8)

    def test_nonzero_run_boundaries_and_zero_exclusion(self):
        starts, ends = BOUNDARY.nonzero_runs([0, 0, 7, 7, 8, 8, 0, 3])
        self.assertEqual(starts, [2, 4, 7])
        self.assertEqual(ends, [3, 5, 7])

    def test_boundary_variants(self):
        record = {"starts": [2, 8], "ends": [5, 9]}
        self.assertEqual(BOUNDARY.boundary_list(record, "start_only"), [2, 8])
        self.assertEqual(BOUNDARY.boundary_list(record, "end_only"), [5, 9])
        self.assertEqual(BOUNDARY.boundary_list(record, "start_and_end"), [2, 5, 8, 9])

    def test_boundary_schedule_exact_unique_deterministic_and_endpoints(self):
        first = BOUNDARY.boundary_schedule(30, 19, [7, 8, 20])
        second = BOUNDARY.boundary_schedule(30, 19, [7, 8, 20])
        self.assertEqual(first, second)
        self.assertEqual(len(first), 19)
        self.assertEqual(first, sorted(set(first)))
        self.assertEqual((first[0], first[-1]), (0, 29))

    def test_uniform_skeleton_is_preserved(self):
        total_frames, budget = 30, 19
        schedule = BOUNDARY.boundary_schedule(total_frames, budget, [7, 20])
        skeleton = BOUNDARY.STRUCTURED.uniform_positions(
            total_frames, BOUNDARY.STRUCTURED.skeleton_count(total_frames, budget)
        )
        self.assertTrue(set(skeleton).issubset(schedule))

    def test_fixed_coordinate_mapping_evidence(self):
        aligned, dense_pad, target_pad = BOUNDARY.REPLAY.align_starts_to_dense_inputs(
            list(range(100)), [0, 3, 6, 97]
        )
        self.assertEqual((dense_pad, target_pad), (7, 6))
        self.assertEqual(aligned, [1, 4, 7, 98])

    def test_pairwise_agreement_reports_directional_tolerances(self):
        proxies = {
            name: {"dev/a": {"starts": [2], "ends": [8]}}
            for name in BOUNDARY.PROXIES
        }
        proxies["pami0"]["dev/a"] = {"starts": [3], "ends": [9]}
        result = BOUNDARY.pairwise_agreement(proxies, ["dev/a"])
        comparison = result["center_label__pami0"]
        self.assertEqual(comparison["tolerance_0"]["symmetric_mean"], 0.0)
        self.assertEqual(comparison["tolerance_2"]["symmetric_mean"], 1.0)

    def test_test_paths_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "test paths are forbidden"):
            BOUNDARY.load_proxies(Path("/tmp/test/meta"), Path("/tmp/dev/labels"))


if __name__ == "__main__":
    unittest.main()
