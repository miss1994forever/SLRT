import importlib.util,inspect,sys,unittest
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];PATH=ROOT/"tools/analyze_phoenix_partial_fresh_terminal_listwise.py";S=importlib.util.spec_from_file_location("fresh_terminal_listwise_tested",PATH);M=importlib.util.module_from_spec(S);sys.modules[S.name]=M;S.loader.exec_module(M)
class FreshTerminalListwiseTests(unittest.TestCase):
 def test_split_scale_is_frozen(self):
  c=M.config();self.assertEqual(c["split"]["partitions"]["fit"]["samples"],2541);self.assertEqual(c["split"]["partitions"]["calibration"]["samples"],459);self.assertEqual(c["split"]["partitions"]["evaluation"]["samples"],432)
 def test_loss_uses_only_informative(self):
  src=inspect.getsource(M.train_one);self.assertIn('sample_mask & d["informative"]',src);self.assertIn("REGRET_WEIGHT*expected",src)
 def test_t0_is_bookkeeping_only(self):
  src=inspect.getsource(M.raw_arrays);self.assertIn('x["bookkeeping"]+(x["prefix"] if full else [])',src)
 def test_gate_is_conjunctive(self):self.assertIn("all(checks.values())",inspect.getsource(M.train_models))
 def test_eval_is_blocked_on_failed_gate(self):self.assertIn('raise RuntimeError("calibration gate failed; evaluation forbidden")',inspect.getsource(M.load_full_model))
 def test_protocol(self):
  p=M.config()["protocol"];self.assertEqual(p["bounded_lookahead_candidate_arrivals"],2);self.assertTrue(p["unknown_EOS"]);self.assertEqual(p["max_gap"],3)
 def test_no_cuda(self):self.assertNotIn("cuda()",PATH.read_text())
if __name__=="__main__":unittest.main()
