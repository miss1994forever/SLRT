import importlib.util
import tempfile
import unittest
from pathlib import Path

import numpy as np


PATH = Path(__file__).resolve().parents[1] / "tools/build_phoenix_full_causal_feature_archive.py"
SPEC = importlib.util.spec_from_file_location("full_causal_feature_archive", PATH)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class FullCausalFeatureArchiveTest(unittest.TestCase):
    def test_name_hash_depends_on_order(self):
        self.assertNotEqual(MODULE.names_sha256(["train/a", "train/b"]),
                            MODULE.names_sha256(["train/b", "train/a"]))

    def test_inspector_accepts_feature_only_p2_schema(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "features.npz"
            np.savez_compressed(
                path,
                features=np.zeros((3, len(MODULE.P2.FEATURE_NAMES)), dtype=np.float16),
                video_names=np.asarray(["train/a", "train/b"]),
                video_offsets=np.asarray([0, 1, 3], dtype=np.int64),
                feature_names=np.asarray(MODULE.P2.FEATURE_NAMES),
            )
            info = MODULE.inspect_feature_npz(path)
            self.assertEqual(info["samples"], 2)
            self.assertEqual(info["frames"], 3)
            self.assertEqual(info["lengths"], [1, 2])

    def test_inspector_rejects_feature_definition_change(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "features.npz"
            np.savez_compressed(
                path,
                features=np.zeros((1, 1), dtype=np.float16),
                video_names=np.asarray(["train/a"]),
                video_offsets=np.asarray([0, 1], dtype=np.int64),
                feature_names=np.asarray(["changed"]),
            )
            with self.assertRaisesRegex(ValueError, "feature definition mismatch"):
                MODULE.inspect_feature_npz(path)


if __name__ == "__main__":
    unittest.main()
