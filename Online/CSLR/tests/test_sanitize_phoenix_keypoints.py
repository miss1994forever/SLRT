import importlib.util
from pathlib import Path
import unittest

import numpy as np


PATH = Path(__file__).resolve().parents[1] / "tools/sanitize_phoenix_keypoints.py"
SPEC = importlib.util.spec_from_file_location("sanitize_phoenix_keypoints", PATH)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class SanitizePhoenixKeypointsTest(unittest.TestCase):
    def test_bad_xy_zeroes_complete_joint_only(self):
        value = np.asarray([[[1, 2, .8], [3, np.inf, .9]], [[4, 5, .7], [6, 7, .6]]], dtype=np.float32)
        source = {"train/a": value}
        output, repairs = MODULE.sanitize_mapping(source)
        MODULE.assert_valid_repair(source, output, repairs)
        self.assertEqual(len(repairs), 1)
        np.testing.assert_array_equal(output["train/a"][0, 1], np.zeros(3, dtype=np.float32))
        np.testing.assert_array_equal(output["train/a"][1], value[1])
        np.testing.assert_array_equal(source["train/a"], value)

    def test_rejects_wrong_shape(self):
        with self.assertRaisesRegex(ValueError, "expected"):
            MODULE.sanitize_mapping({"train/a": np.zeros((2, 3), dtype=np.float32)})

    def test_validator_detects_unrelated_change(self):
        source = {"train/a": np.ones((1, 2, 3), dtype=np.float32)}
        output = {"train/a": source["train/a"].copy()}
        output["train/a"][0, 0, 0] = 2
        with self.assertRaisesRegex(ValueError, "unexpected change"):
            MODULE.assert_valid_repair(source, output, [])


if __name__ == "__main__":
    unittest.main()
