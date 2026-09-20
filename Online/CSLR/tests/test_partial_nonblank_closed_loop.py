import importlib.util,inspect,sys,unittest
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];PATH=ROOT/"tools/analyze_phoenix_partial_nonblank_closed_loop.py";S=importlib.util.spec_from_file_location("nonblank_tested",PATH);M=importlib.util.module_from_spec(S);sys.modules[S.name]=M;S.loader.exec_module(M)
class NonblankClosedLoopTests(unittest.TestCase):
 def test_only_frozen_target(self):self.assertEqual(M.config()["target"],"full_nonblank_confidence")
 def test_post_selection_disclosed(self):self.assertIn("not confirmatory",M.config()["selection_bias"])
 def test_policy_uses_preview_not_full_logits(self):
  src=inspect.getsource(M.online_features);self.assertNotIn("logits",src);self.assertIn("sequences.history",src)
 def test_exact_one_bonus(self):self.assertIn("selected.append(candidates[j])",inspect.getsource(M.run_sample))
 def test_bounded_delay(self):self.assertEqual(M.config()["decision"]["bounded_lookahead_video_frames"],2)
 def test_no_cuda(self):self.assertNotIn("cuda()",PATH.read_text())
 def test_stop_rule_frozen(self):self.assertIn("not lower",M.config()["stop_rule"])
if __name__=="__main__":unittest.main()
