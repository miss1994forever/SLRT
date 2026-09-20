import importlib.util
import sys
import unittest
from pathlib import Path

import numpy as np
import torch

PATH = Path(__file__).resolve().parents[1] / "tools/analyze_phoenix_partial_contextual_meta_gate.py"
SPEC = importlib.util.spec_from_file_location("contextual_meta_gate_tested", PATH)
MOD = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MOD
SPEC.loader.exec_module(MOD)


class ContextualMetaGateTests(unittest.TestCase):
    def test_protocol_is_fixed(self):
        cfg = MOD.config()
        self.assertEqual(cfg["meta_model"]["architecture"], "multinomial logistic: Linear(feature_width,3)")
        self.assertEqual(cfg["meta_model"]["class_weights_left_center_right"], [4.0, 1.0, 4.0])
        self.assertFalse(cfg["pair_oracle"]["deployable"])
        self.assertIn("terminal reward", cfg["meta_features"]["forbidden"])

    def test_logistic_shape(self):
        self.assertEqual(tuple(MOD.MetaLogistic(7)(torch.zeros(3, 7)).shape), (3, 3))

    def test_candidate_label_center_tie_then_component_order(self):
        errors = np.asarray([[0, 0, 2], [0, 3, 0], [3, 2, 0]], np.float32)
        labels = MOD.candidate_label(errors, np.asarray([0, 0, 0]), np.asarray([2, 2, 2]))
        np.testing.assert_array_equal(labels, [1, 0, 2])

    def test_choose_available_respects_candidate_set(self):
        probability = np.asarray([[.1, .2, .7], [.7, .2, .1], [.5, .5, .0]], np.float32)
        choice = MOD.choose_available(probability, np.asarray([0, 0, 0]), np.asarray([1, 2, 2]))
        np.testing.assert_array_equal(choice, [1, 0, 1])

    def test_fold_stable(self):
        self.assertEqual(MOD.source_fold("abc"), MOD.source_fold("abc"))

    def test_no_later_partition_or_gpu(self):
        text = PATH.read_text()
        self.assertIn('os.environ["CUDA_VISIBLE_DEVICES"] = ""', text)
        self.assertNotIn("calibration_started", text)
        self.assertNotIn("evaluation_started", text)


if __name__ == "__main__":
    unittest.main()
