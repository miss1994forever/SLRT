import importlib.util
import sys
import unittest
from pathlib import Path

import numpy as np
import torch

PATH = Path(__file__).resolve().parents[1] / "tools/analyze_phoenix_partial_dense_semantic_distillation.py"
SPEC = importlib.util.spec_from_file_location("dense_semantic_distillation_tested", PATH)
MOD = importlib.util.module_from_spec(SPEC); sys.modules[SPEC.name] = MOD; SPEC.loader.exec_module(MOD)


class DenseSemanticDistillationTest(unittest.TestCase):
    def test_frozen_protocol(self):
        cfg = MOD.config()
        self.assertEqual(cfg["teacher"]["PCA"]["dimensions"], 32)
        self.assertIn("inside each source-fold", cfg["teacher"]["leakage_control"])
        self.assertEqual(cfg["OOF"]["coverages"], [0.005, 0.01, 0.02, 0.05, 0.1])
        self.assertIn("calibration/evaluation outcomes", cfg["forbidden"])

    def test_fold_is_stable(self):
        self.assertEqual(MOD.source_fold("source-a"), MOD.source_fold("source-a"))
        self.assertTrue(0 <= MOD.source_fold("source-b") < 5)

    def test_model_shapes(self):
        model = MOD.SemanticResidual(189)
        semantic, score = model(torch.zeros(4, 3, 189))
        self.assertEqual(tuple(semantic.shape), (4, 3, 32))
        self.assertEqual(tuple(score.shape), (4, 3))

    def test_pca_target_shape(self):
        logits = np.arange(2*3*5, dtype=np.float32).reshape(2, 3, 5)
        pca = {"mean": np.zeros(5, np.float32), "components": np.eye(5, 2, dtype=np.float32), "scales": np.ones(2, np.float32)}
        self.assertEqual(MOD.pca_targets(logits, pca).shape, (2, 3, 2))

    def test_feature_normalization_is_train_only(self):
        value = np.zeros((2, 3, 2), np.float32); value[1] = 100
        mean, _ = MOD.feature_stats(value, np.asarray([True, False]))
        np.testing.assert_array_equal(mean, np.zeros(2))

    def test_cpu_guard_and_no_later_partitions(self):
        text = PATH.read_text()
        self.assertIn('os.environ["CUDA_VISIBLE_DEVICES"] = ""', text)
        self.assertNotIn("calibration_started", text)
        self.assertNotIn("evaluation_started", text)


if __name__ == "__main__":
    unittest.main()
