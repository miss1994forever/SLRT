import importlib.util
import unittest
from pathlib import Path


CSLR_ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = CSLR_ROOT / "tools/analyze_phoenix_random_schedule.py"
SPEC = importlib.util.spec_from_file_location("random_schedule_replay", MODULE_PATH)
RANDOM = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(RANDOM)


class RandomScheduleReplayTests(unittest.TestCase):
    def test_per_sample_schedule_has_exact_budget_unique_centers_and_endpoints(self):
        starts = RANDOM.random_schedule_per_sample(20, 11, 1729, "dev/sample")
        self.assertEqual(len(starts), 11)
        self.assertEqual(len(set(starts)), 11)
        self.assertEqual(starts, sorted(starts))
        self.assertEqual((starts[0], starts[-1]), (0, 19))

    def test_per_sample_schedule_is_deterministic_and_name_stable(self):
        first = RANDOM.random_schedule_per_sample(40, 20, 1729, "dev/a")
        second = RANDOM.random_schedule_per_sample(40, 20, 1729, "dev/a")
        different_name = RANDOM.random_schedule_per_sample(40, 20, 1729, "dev/b")
        self.assertEqual(first, second)
        self.assertNotEqual(first, different_name)

    def test_per_sample_rejects_budget_without_two_endpoint_slots(self):
        with self.assertRaisesRegex(ValueError, "two distinct endpoints"):
            RANDOM.random_schedule_per_sample(5, 1, 1729, "dev/sample")

    def test_global_schedule_has_exact_total_budget_and_endpoint_coverage(self):
        lengths = {"dev/a": 10, "dev/b": 7, "dev/c": 1}
        schedules = RANDOM.random_schedules_global(lengths, 12, 1729)
        self.assertEqual(sum(map(len, schedules.values())), 12)
        for name, starts in schedules.items():
            self.assertEqual(starts, sorted(set(starts)))
            self.assertEqual(starts[0], 0)
            self.assertEqual(starts[-1], lengths[name] - 1)

    def test_global_schedule_is_deterministic(self):
        lengths = {"dev/a": 10, "dev/b": 7}
        self.assertEqual(
            RANDOM.random_schedules_global(lengths, 10, 1729),
            RANDOM.random_schedules_global(lengths, 10, 1729),
        )

    def test_test_paths_are_rejected_by_input_loader(self):
        with self.assertRaisesRegex(ValueError, "test paths are forbidden"):
            RANDOM.load_inputs(Path("/tmp/test"), Path("/tmp/dev_vocab.json"))

    def test_physical_coordinate_mapping_and_fixed_coordinate_differ(self):
        aligned, dense_pad, target_pad = RANDOM.REPLAY.align_starts_to_dense_inputs(
            list(range(100)), [0, 3, 6, 97]
        )
        self.assertEqual((dense_pad, target_pad), (7, 6))
        self.assertEqual(aligned, [1, 4, 7, 98])
        self.assertNotEqual(aligned, [0, 3, 6, 97])


if __name__ == "__main__":
    unittest.main()
