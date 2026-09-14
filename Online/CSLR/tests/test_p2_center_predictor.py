import importlib.util
from pathlib import Path
import unittest

import numpy as np
import torch


TOOLS = Path(__file__).resolve().parents[1] / "tools"


def load(name):
    spec = importlib.util.spec_from_file_location(name, TOOLS / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


FEATURES = load("build_phoenix_p2_center_sequences")
TRAIN = load("train_phoenix_p2_center_predictor")


def keypoints(length=12):
    array = np.zeros((length, 133, 3), dtype=np.float32)
    array[..., 2] = 1.0
    array[..., 0] = np.arange(133)[None, :] + np.arange(length)[:, None]
    array[..., 1] = np.arange(133)[None, :] * 0.5
    return array


class P2CenterPredictorTest(unittest.TestCase):
    def test_feature_identity_and_finiteness(self):
        value = FEATURES.frame_features(keypoints())
        self.assertEqual(value.shape, (12, len(FEATURES.FEATURE_NAMES)))
        self.assertTrue(np.isfinite(value).all())

    def test_nonfinite_landmark_is_masked(self):
        value = keypoints()
        value[3, 91, 0] = np.nan
        features = FEATURES.frame_features(value)
        self.assertTrue(np.isfinite(features).all())
        left_valid_start = 22 + 11 + 11 + 42
        self.assertEqual(features[3, left_valid_start], 0.0)

    def test_frame_features_do_not_read_future(self):
        original = keypoints()
        changed = original.copy()
        changed[7:, :, :2] += 10000
        np.testing.assert_allclose(
            FEATURES.frame_features(original)[:7], FEATURES.frame_features(changed)[:7], rtol=0, atol=0
        )

    def test_tcn_does_not_read_future(self):
        torch.manual_seed(3)
        model = TRAIN.CenterTCN(5, channels=4, dropout=0.0).eval()
        original = torch.randn(1, 5, 20)
        changed = original.clone()
        changed[:, :, 11:] += 1000
        with torch.no_grad():
            first = model(original)
            second = model(changed)
        torch.testing.assert_close(first[:, :11], second[:, :11], rtol=0, atol=0)

    def test_receptive_field_is_31(self):
        self.assertEqual(1 + 2 * sum(TRAIN.DILATIONS), 31)


if __name__ == "__main__":
    unittest.main()
