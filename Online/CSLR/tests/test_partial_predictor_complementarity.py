import importlib.util
import sys
import unittest
from pathlib import Path

import numpy as np
import torch

PATH=Path(__file__).resolve().parents[1]/"tools/analyze_phoenix_partial_predictor_complementarity.py"
SPEC=importlib.util.spec_from_file_location("predictor_complementarity_tested",PATH);MOD=importlib.util.module_from_spec(SPEC);sys.modules[SPEC.name]=MOD;SPEC.loader.exec_module(MOD)


class ComplementarityTest(unittest.TestCase):
    def test_rules_frozen(self):
        cfg=MOD.config();self.assertIn("gate opens iff",cfg["combination_A"]["rule"]);self.assertIn("same side",cfg["combination_B"]["rule"]);self.assertFalse(cfg["pair_oracle"]["deployable"])

    def test_hazard_shape(self):
        model=MOD.HazardMLP();self.assertEqual(tuple(model(torch.zeros(4,3,98)).shape),(4,3))

    def test_history_choice(self):
        score=np.asarray([[1,-1],[-1,2],[-1,-2]],np.float32);np.testing.assert_array_equal(MOD.history_choice(score),[0,2,1])

    def test_pair_oracle(self):
        data={"errors":np.asarray([[2,1,0],[0,1,2]],np.float32)};folds=np.asarray([0,0]);choice,report=MOD.pair_oracle(data,np.asarray([0,0]),np.asarray([2,2]),folds);np.testing.assert_array_equal(choice,[2,0]);self.assertEqual(report["cumulative_terminal_reward"],2)

    def test_fold_stable(self):self.assertEqual(MOD.source_fold("abc"),MOD.source_fold("abc"))

    def test_no_later_partition_or_gpu(self):
        text=PATH.read_text();self.assertIn('os.environ["CUDA_VISIBLE_DEVICES"]=""',text);self.assertNotIn("calibration_started",text);self.assertNotIn("evaluation_started",text)


if __name__=="__main__":unittest.main()
