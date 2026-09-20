import importlib.util,inspect,sys,unittest
from unittest import mock
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];PATH=ROOT/"tools/analyze_phoenix_partial_structured_block_preview.py";S=importlib.util.spec_from_file_location("preview_tested",PATH);M=importlib.util.module_from_spec(S);sys.modules[S.name]=M;S.loader.exec_module(M)
class PreviewTests(unittest.TestCase):
 def config_without_external_archive(self):
  with mock.patch.object(M,"sha256_file",return_value="fixture-sha256"):
   return M.config()
 def test_preview_is_audited_causal_history(self):
  c=self.config_without_external_archive()["preview"];self.assertEqual(c["history_frames"],31);self.assertEqual(c["maximum_additional_wait_video_frames"],2);self.assertEqual(c["lookahead_class"],"bounded-lookahead-2")
 def test_runtime_cost_is_not_called_free(self):self.assertIn("not free",self.config_without_external_archive()["preview"]["real_cost"])
 def test_gate_is_conjunctive(self):self.assertIn("all(checks.values())",inspect.getsource(M.analyze))
 def test_predictor_input_has_no_isrl_logits(self):
  src=inspect.getsource(M.raw_arrays);self.assertNotIn("logits",src);self.assertIn('row["features"]',src)
 def test_informative_weight_is_frozen(self):self.assertEqual(M.INFORMATIVE_WEIGHT,50.0)
 def test_no_cuda(self):self.assertNotIn("cuda()",PATH.read_text())
 def test_preview_history_uses_each_candidate_start(self):self.assertIn('sequences.history(row["sample_id"],s)',inspect.getsource(M.raw_arrays))
if __name__=="__main__":unittest.main()
