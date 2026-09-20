import importlib.util,inspect,sys,unittest
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];PATH=ROOT/"tools/analyze_phoenix_partial_nonblank_fresh_replication.py";S=importlib.util.spec_from_file_location("fresh_tested",PATH);M=importlib.util.module_from_spec(S);sys.modules[S.name]=M;S.loader.exec_module(M)
class FreshReplicationTests(unittest.TestCase):
 @classmethod
 def setUpClass(cls):cls.s=M.split_spec()
 def test_source_disjoint(self):self.assertTrue(self.s["source_video_disjoint"])
 def test_old_evaluation_excluded(self):
  a=self.s["contamination_audit"];self.assertEqual(a["old_eval_in_fresh_eval"],0);self.assertEqual(a["old_fit_in_fresh_eval"],0)
 def test_eval_outcome_blind_split(self):self.assertFalse(self.s["contamination_audit"]["evaluation_outcomes_read_to_construct_split"])
 def test_eval_sealed_until_model(self):self.assertIn("loadmodel(root)",inspect.getsource(M.evalrun));self.assertIn("evaluation_started.marker",inspect.getsource(M.evalrun))
 def test_frozen_target_and_model(self):self.assertEqual(M.config()["frozen"]["target"],"full_nonblank_confidence")
 def test_strong_go_frozen(self):self.assertIn("delta WER <= -0.5pp",M.config()["strong_go"])
 def test_no_cuda(self):self.assertNotIn("cuda()",PATH.read_text())
if __name__=="__main__":unittest.main()
