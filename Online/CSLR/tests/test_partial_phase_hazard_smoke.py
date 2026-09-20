import importlib.util,inspect,sys,unittest
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];PATH=ROOT/"tools/analyze_phoenix_partial_phase_hazard_smoke.py";SPEC=importlib.util.spec_from_file_location("phase_hazard_tested",PATH);M=importlib.util.module_from_spec(SPEC);sys.modules[SPEC.name]=M;SPEC.loader.exec_module(M)
class PhaseHazardTests(unittest.TestCase):
 def test_targets_frozen(self):self.assertEqual(M.TARGETS,("broad_interior","interior_entry_hazard4","interior_entry_hazard8"))
 def test_hazard_is_not_boundary(self):self.assertEqual(M.config()["stage_A"]["distinction"],"interior-entry hazard, not gloss-boundary hazard")
 def test_phase_band(self):self.assertIn("[0.30,0.70]",M.config()["stage_A"]["targets"]["broad_interior"])
 def test_stage_b_frozen(self):
  b=M.config()["stage_B_frozen_before_stage_A"];self.assertEqual(sorted(map(int,b["lookaheads"])),[0,4,8]);self.assertIn("Linear(189,48)",b["architecture"])
 def test_stage_a_gate(self):
  src=inspect.getsource(M.stage_a);self.assertIn("all_fit_fold_closed_rewards_ge_0",src);self.assertIn("cal_closed_reward_gt_0",src)
 def test_no_eval_or_gpu(self):
  text=PATH.read_text();self.assertNotIn("cuda()",text);self.assertNotIn("evaluation_started",text);self.assertIn('os.environ["CUDA_VISIBLE_DEVICES"]=""',text)
if __name__=="__main__":unittest.main()
