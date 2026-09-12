import importlib.util
import json
from pathlib import Path
import tempfile
import unittest


PATH = Path(__file__).resolve().parents[1] / "tools/evaluate_phoenix_p1_center_predictor.py"
SPEC = importlib.util.spec_from_file_location("p1_center_predictor_eval", PATH)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class P1CenterPredictorEvalTest(unittest.TestCase):
    def test_rejects_test_path(self):
        with self.assertRaises(ValueError):
            MODULE.reject_test_path("/data/test/keypoints.pkl")

    def test_freeze_validation_detects_output_mutation(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output = root / "model.json"
            output.write_text("original")
            freeze = {
                "status": "predev_frozen", "test_opened_or_run": False,
                "outputs": {str(output): {"bytes": output.stat().st_size,
                                           "sha256": MODULE.TRAIN.sha256_file(output)}},
            }
            (root / "predev_freeze_manifest.json").write_text(json.dumps(freeze))
            MODULE.validate_freeze(root)
            output.write_text("changed")
            with self.assertRaisesRegex(ValueError, "hash mismatch"):
                MODULE.validate_freeze(root)


if __name__ == "__main__":
    unittest.main()
