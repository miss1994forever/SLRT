import importlib.util
import inspect
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PATH = ROOT / "tools/audit_phoenix_new_observable_preview.py"
SPEC = importlib.util.spec_from_file_location("preview_audit_tested", PATH)
M = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = M
SPEC.loader.exec_module(M)


class NewObservablePreviewAuditTests(unittest.TestCase):
    def test_benchmark_is_train_fit_only_and_bounded(self):
        ids = M.hashed_fit_ids()
        self.assertEqual(len(ids), 32)
        self.assertTrue(all(x.startswith("train/") for x in ids))
        self.assertEqual(M.SAMPLES, 32)

    def test_hand_feature_is_appearance_not_logits(self):
        src = inspect.getsource(M.hand_hog)
        self.assertIn("cvtColor", src)
        self.assertNotIn("logit", src.lower())

    def test_partial_scale_uses_completed_train_metadata_only(self):
        src = inspect.getsource(M.completed_partial32_scale)
        self.assertIn("train_metadata.pkl.gz", src)
        self.assertIn("complete.json", src)

    def test_no_gpu(self):
        text = PATH.read_text()
        self.assertIn('os.environ["CUDA_VISIBLE_DEVICES"] = ""', text)
        self.assertNotIn("cuda()", text)


if __name__ == "__main__":
    unittest.main()
