import importlib.util,inspect,sys,unittest
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];PATH=ROOT/"tools/analyze_phoenix_partial_dense_upgrade_targets.py";S=importlib.util.spec_from_file_location("upgrade_tested",PATH);M=importlib.util.module_from_spec(S);sys.modules[S.name]=M;S.loader.exec_module(M)
class UpgradeTargetTests(unittest.TestCase):
 def test_deployment_feature_has_no_full_logits(self):
  src=inspect.getsource(M.preview_feature);self.assertNotIn("logits",src);self.assertIn("sequences.history",src)
 def test_teacher_targets_are_explicitly_label_only(self):self.assertIn("candidate full ISLR logits",M.config()["teacher_only"])
 def test_unknown_eos_and_bounded_lookahead(self):
  x=M.config()["decision_availability"];self.assertTrue(x["unknown_EOS"]);self.assertEqual(x["bounded_lookahead_video_frames"],2)
 def test_unavailable_targets_are_omitted(self):
  x=M.config()["omitted_targets"];self.assertIn("sequence_loss",x);self.assertIn("cheap_vs_full_disagreement",x)
 def test_two_gates_are_conjunctive(self):
  src=inspect.getsource(M.analyze);self.assertIn('"gate_A"',src);self.assertIn('"gate_B"',src);self.assertIn("all(checksA.values()) and all(checksB.values())",src)
 def test_target_schedule_keeps_one_of_three(self):self.assertIn("chosen.append(cand[best])",inspect.getsource(M.enrich))
 def test_no_cuda(self):self.assertNotIn("cuda()",PATH.read_text())
if __name__=="__main__":unittest.main()
