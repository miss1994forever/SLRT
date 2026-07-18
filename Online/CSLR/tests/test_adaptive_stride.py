import unittest

import torch

from utils.adaptive_stride import adaptive_window_starts, span_weighted_predictions


def make_keypoints(positions, confidence=1.0, count=8):
    frames = []
    offsets = torch.linspace(0.0, 1.0, count).unsqueeze(-1).repeat(1, 2)
    for position in positions:
        xy = offsets + torch.tensor([float(position), 0.0])
        conf = torch.full((count, 1), confidence)
        frames.append(torch.cat([xy, conf], dim=-1))
    return torch.stack(frames)


class AdaptiveStrideTests(unittest.TestCase):
    def test_disabled_matches_stride_one(self):
        keypoints = make_keypoints(range(10))
        starts, _ = adaptive_window_starts(10, keypoints, {"enabled": False})
        self.assertEqual(starts, list(range(10)))

    def test_static_motion_uses_larger_stride_after_warmup(self):
        keypoints = make_keypoints([0] * 30)
        starts, metadata = adaptive_window_starts(
            30,
            keypoints,
            {"enabled": True, "min_stride": 1, "max_stride": 3, "warmup_frames": 4},
        )
        self.assertTrue(any(item["stride"] == 3 for item in metadata[4:]))
        self.assertEqual(starts, sorted(set(starts)))

    def test_low_confidence_falls_back_to_min_stride(self):
        keypoints = make_keypoints(range(20), confidence=0.0)
        _, metadata = adaptive_window_starts(
            20,
            keypoints,
            {"enabled": True, "min_stride": 1, "max_stride": 3, "warmup_frames": 2},
        )
        self.assertTrue(all(item["stride"] == 1 for item in metadata))

    def test_span_voting_uses_temporal_distance(self):
        logits = torch.tensor([[6.0, 0.0], [0.0, 6.0], [6.0, 0.0]])
        predictions = span_weighted_predictions(logits, centers=[0.0, 1.0, 20.0], span=5.0)
        self.assertEqual(predictions.tolist(), [0, 1, 0])

    def test_invalid_quantiles_are_rejected(self):
        with self.assertRaises(ValueError):
            adaptive_window_starts(3, make_keypoints(range(3)), {"enabled": True, "quantile_low": 0.8, "quantile_high": 0.2})


if __name__ == "__main__":
    unittest.main()
