import unittest

import torch

from utils.window_sampling import generate_window_starts, resolve_sampling_config


class WindowSamplingTests(unittest.TestCase):
    def test_fixed_stride(self):
        starts, metadata = generate_window_starts(
            "fixed", 10, keypoints=None, config={"fixed_stride": 3}
        )
        self.assertEqual(starts, [0, 3, 6, 9])
        self.assertTrue(all(item["sampling_mode"] == "fixed" for item in metadata))
        self.assertTrue(all(item["stride"] == 3 for item in metadata))

    def test_uniform_rate_is_deterministic_and_increasing(self):
        starts, metadata = generate_window_starts(
            "uniform_rate", 10, keypoints=None, config={"uniform_mean_stride": 1.5}
        )
        self.assertEqual(starts, [0, 2, 3, 5, 6, 8, 9])
        self.assertEqual(starts, sorted(set(starts)))
        self.assertEqual([item["stride"] for item in metadata[:-1]], [2, 1, 2, 1, 2, 1])

    def test_uniform_rate_rejects_subunit_stride(self):
        with self.assertRaises(ValueError):
            generate_window_starts(
                "uniform_rate", 10, keypoints=None, config={"uniform_mean_stride": 0.9}
            )

    def test_legacy_adaptive_config_resolves_mode(self):
        resolved = resolve_sampling_config(
            {}, fixed_stride=1, adaptive_config={"enabled": True, "max_stride": 3}
        )
        self.assertEqual(resolved["mode"], "adaptive_motion")
        self.assertEqual(resolved["max_stride"], 3)

    def test_adaptive_mode_adds_common_metadata(self):
        keypoints = torch.zeros(12, 8, 3)
        keypoints[..., 2] = 1.0
        starts, metadata = generate_window_starts(
            "adaptive_motion",
            12,
            keypoints,
            {"min_stride": 1, "max_stride": 3, "warmup_frames": 2},
        )
        self.assertEqual(starts, sorted(set(starts)))
        self.assertTrue(all(item["sampling_mode"] == "adaptive_motion" for item in metadata))


if __name__ == "__main__":
    unittest.main()
