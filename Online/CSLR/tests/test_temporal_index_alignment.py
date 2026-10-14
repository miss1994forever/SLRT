import random
import unittest
from unittest.mock import patch

import numpy as np

from dataset import VideoLoader


def _fake_frames(selected_index):
    selected_index = np.asarray(selected_index)
    return selected_index[:, None, None, None].astype(np.uint8)


class TemporalIndexAlignmentTest(unittest.TestCase):
    @patch.object(
        VideoLoader,
        "read_jpg",
        side_effect=lambda zip_file, dataset_name, selected_index, vlen, ori_vfile, strict_frame_loading=False:
            _fake_frames(selected_index),
    )
    def test_phoenix_even_rgb_and_keypoint_indices_share_absolute_coordinates(self, _):
        video, keypoint_indices, pad = VideoLoader.load_video(
            "unused.zip", "word_source_[10:15]", 5, 30, 16, "phoenix_iso", False,
            ["consecutive", "pad", "central", "even"], ori_vfile="source",
        )
        self.assertIsNone(pad)
        self.assertEqual(keypoint_indices.tolist(), video[:, 0, 0, 0].tolist())
        self.assertEqual(min(keypoint_indices), 10)
        self.assertEqual(max(keypoint_indices), 14)

    @patch.object(
        VideoLoader,
        "read_jpg",
        side_effect=lambda zip_file, dataset_name, selected_index, vlen, ori_vfile, strict_frame_loading=False:
            _fake_frames(selected_index),
    )
    def test_temp_scale_shortening_does_not_use_replication_indices(self, _):
        np.random.seed(4)
        random.seed(4)
        video, keypoint_indices, _ = VideoLoader.load_video(
            "unused.zip", "word_source_[10:30]", 20, 40, 16, "phoenix_iso", True,
            ["temp_scale", "temp_scale", "central", "pad"], temp_scale=[0.5, 0.5], ori_vfile="source",
        )
        self.assertEqual(len(video), 16)
        self.assertEqual(keypoint_indices.tolist(), video[:, 0, 0, 0].tolist())
        self.assertGreaterEqual(min(keypoint_indices), 10)
        self.assertLess(max(keypoint_indices), 30)


if __name__ == "__main__":
    unittest.main()
