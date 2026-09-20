import importlib.util,inspect,sys,unittest
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];PATH=ROOT/"tools/analyze_phoenix_partial_deterministic_coverage_family.py";S=importlib.util.spec_from_file_location("family_tested",PATH);M=importlib.util.module_from_spec(S);sys.modules[S.name]=M;S.loader.exec_module(M)
class CoverageFamilyTests(unittest.TestCase):
 def test_family_exact(self):self.assertEqual(set(M.FAMILY),{"fixed_1","fixed_2","fixed_3","cycle_123","cycle_321","alternate_13","alternate_12","alternate_23"})
 def test_schedule_hard_skeleton_one_bonus(self):self.assertEqual(M.schedule(10,(1,2,3)),[0,1,4,6,8])
 def test_tail_no_topup(self):self.assertEqual(M.schedule(3,(3,)),[0])
 def test_selected_offset_counts(self):self.assertEqual(M.selected_offset_counts([{"selected":[0,3,4,7,8]}]),{"0":3,"1":0,"2":0,"3":2})
 def test_zero_lookahead(self):self.assertEqual(M.config()["causality"]["algorithm_lookahead_frames"],0)
 def test_tie_prefers_center(self):self.assertIn('0 if name=="fixed_2" else 1',inspect.getsource(M.select))
 def test_evaluation_only_winner_and_center(self):self.assertIn('policies={"fixed_2":FAMILY["fixed_2"]}',inspect.getsource(M.evaluate))
 def test_no_cuda(self):self.assertNotIn("cuda()",PATH.read_text())
if __name__=="__main__":unittest.main()
