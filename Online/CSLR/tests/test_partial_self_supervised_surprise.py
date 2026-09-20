import importlib.util
import sys
import unittest
from pathlib import Path

import numpy as np
import torch

PATH=Path(__file__).resolve().parents[1]/"tools/analyze_phoenix_partial_self_supervised_surprise.py"
SPEC=importlib.util.spec_from_file_location("self_supervised_surprise_tested",PATH);MOD=importlib.util.module_from_spec(SPEC);sys.modules[SPEC.name]=MOD;SPEC.loader.exec_module(MOD)


class SelfSupervisedSurpriseTest(unittest.TestCase):
    def test_protocol(self):
        cfg=MOD.config();self.assertEqual(cfg["cheap_sequence"]["per_candidate_width"],89);self.assertEqual(cfg["policy"]["bounded_lookahead_candidate_arrivals"],2);self.assertFalse(cfg["self_supervision"]["terminal_labels_used_for_training"]);self.assertEqual(cfg["self_supervision"]["uncertainty"],"none")

    def test_model_shape(self):
        model=MOD.Predictor();self.assertEqual(tuple(model(torch.zeros(2,7,89)).shape),(2,7,89))

    def test_center_wins_tie(self):
        scores=np.asarray([[1,1,1],[2,1,2],[0,1,2]],np.float32);np.testing.assert_array_equal(MOD.choose(scores),[1,0,2])

    def test_average_precision(self):
        self.assertAlmostEqual(MOD.average_precision([1,0,1],[3,2,1]),(1+2/3)/2)

    def test_fold_stable(self):
        self.assertEqual(MOD.source_fold("abc"),MOD.source_fold("abc"));self.assertTrue(0<=MOD.source_fold("def")<5)

    def test_no_later_partition_or_gpu(self):
        text=PATH.read_text();self.assertIn('os.environ["CUDA_VISIBLE_DEVICES"]=""',text);self.assertNotIn("calibration_started",text);self.assertNotIn("evaluation_started",text)


if __name__=="__main__":unittest.main()
