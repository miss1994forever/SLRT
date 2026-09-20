import importlib.util
import sys
import unittest
from pathlib import Path

import numpy as np
import torch

PATH=Path(__file__).resolve().parents[1]/"tools/analyze_phoenix_partial_full_decoder_history.py"
SPEC=importlib.util.spec_from_file_location("full_decoder_history_tested",PATH);MOD=importlib.util.module_from_spec(SPEC);sys.modules[SPEC.name]=MOD;SPEC.loader.exec_module(MOD)


class FullDecoderHistoryTest(unittest.TestCase):
    def test_frozen_protocol(self):
        cfg=MOD.config();self.assertEqual(cfg["behavior_history"]["history_length"],16);self.assertEqual(cfg["PCA"]["dimensions"],16);self.assertEqual(cfg["OOF"]["coverages"],[.005,.01,.02,.05,.1]);self.assertIn("offset1",cfg["behavior_history"]["forbidden_current_logits"])

    def test_old_state_audit(self):
        cfg=MOD.config();self.assertIn("no ordered",cfg["old_state_audit"]["B2"]);self.assertIn("ordered last 16",cfg["old_state_audit"]["new_information"])

    def test_models_and_parameter_match(self):
        base=torch.zeros(4,3,49);history=torch.zeros(4,16,30);lengths=torch.ones(4,dtype=torch.long)
        a,b=MOD.SummaryMLP(),MOD.HistoryGRU();self.assertEqual(tuple(a(base,history,lengths).shape),(4,3));self.assertEqual(tuple(b(base,history,lengths).shape),(4,3));self.assertLess(abs(MOD.parameter_count(a)-MOD.parameter_count(b))/MOD.parameter_count(a),.05)

    def test_collapse_ctc(self):
        self.assertEqual(MOD.collapse([1,1,0,1,2,2],0),(1,1,2))

    def test_token_hash_stable(self):
        self.assertEqual(MOD.token_hash4(42),MOD.token_hash4(42));self.assertEqual(len(MOD.token_hash4(42)),4)

    def test_cpu_and_no_later_partition(self):
        text=PATH.read_text();self.assertIn('os.environ["CUDA_VISIBLE_DEVICES"]=""',text);self.assertNotIn("calibration_started",text);self.assertNotIn("evaluation_started",text)


if __name__=="__main__":unittest.main()
