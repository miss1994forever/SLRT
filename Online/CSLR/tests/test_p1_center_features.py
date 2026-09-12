import importlib.util
from pathlib import Path
import unittest

import numpy as np


PATH = Path(__file__).resolve().parents[1] / "tools/build_phoenix_p1_center_features.py"
SPEC = importlib.util.spec_from_file_location("p1_center_features", PATH)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def keypoints(length=8):
    array = np.zeros((length, 133, 3), dtype=np.float32)
    array[..., 2] = 1.0
    array[..., 0] = np.arange(length)[:, None]
    return array


class P1CenterFeaturesTest(unittest.TestCase):
    def test_rejects_test_path(self):
        with self.assertRaisesRegex(ValueError, "test paths"):
            MODULE.reject_test_path("/data/test/file.pkl")

    def test_requires_train_only_keypoint_mapping(self):
        MODULE.assert_train_only_keys({"train/a": np.zeros(1)})
        with self.assertRaisesRegex(ValueError, "not train-only"):
            MODULE.assert_train_only_keys({"train/a": np.zeros(1), "dev/b": np.zeros(1)})

    def test_midpoint_target_is_nonblank_radius_one(self):
        target, centers = MODULE.midpoint_targets(
            10,
            [
                {"label": "WORD", "start": 2, "end": 7},
                {"label": "<blank>", "start": 7, "end": 10},
            ],
        )
        self.assertEqual(centers.tolist(), [4])
        self.assertEqual(np.flatnonzero(target).tolist(), [3, 4, 5])

    def test_nonfinite_is_explicitly_masked_and_features_remain_finite(self):
        array = keypoints()
        array[3, 91, 0] = np.nan
        xy, valid = MODULE.sanitize_keypoints(array)
        selected_position = list(MODULE.SELECTED).index(91)
        self.assertFalse(valid[3, selected_position])
        self.assertEqual(xy[3, selected_position].tolist(), [0.0, 0.0])
        self.assertTrue(np.isfinite(MODULE.causal_features(array)).all())

    def test_feature_at_t_does_not_read_future_frames(self):
        original = keypoints(12)
        changed = original.copy()
        changed[7:, :, :2] = 10000.0
        first = MODULE.causal_features(original)
        second = MODULE.causal_features(changed)
        np.testing.assert_allclose(first[:7], second[:7], rtol=0, atol=0)

    def test_group_split_is_deterministic_and_disjoint(self):
        names = [f"train/video-{index}" for index in range(100)]
        first = MODULE.deterministic_group_split(names)
        second = MODULE.deterministic_group_split(reversed(names))
        self.assertEqual(first, second)
        self.assertTrue(set(first[0]).isdisjoint(first[1]))
        self.assertEqual(set(first[0]) | set(first[1]), set(names))

    def test_reference_inconsistent_video_is_preregistered_for_exclusion(self):
        self.assertEqual(
            MODULE.EXCLUDED_TARGET_VIDEOS,
            {"train/25August_2009_Tuesday_heute-3301"},
        )


if __name__ == "__main__":
    unittest.main()
