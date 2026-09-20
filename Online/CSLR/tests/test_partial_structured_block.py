import importlib.util, inspect, sys, unittest
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]; PATH=ROOT/"tools/analyze_phoenix_partial_structured_block.py"
S=importlib.util.spec_from_file_location("structured_block_tested",PATH); M=importlib.util.module_from_spec(S); sys.modules[S.name]=M; S.loader.exec_module(M)

class StructuredBlockTests(unittest.TestCase):
    def test_uniform_has_skeleton_and_one_bonus_per_complete_block(self):
        self.assertEqual(M.block_uniform(10),[0,2,4,6,8])
        self.assertEqual(M.block_uniform(12),[0,2,4,6,8,10])
    def test_tail_is_not_topped_up(self):
        self.assertEqual(M.block_uniform(3),[0]); self.assertEqual(M.block_uniform(7),[0,2,4])
    def test_center_continuation_starts_after_current_block(self):
        self.assertEqual(M.center_continuation(11,4),[4,6,8])
    def test_candidate_features_do_not_receive_candidate_logits(self):
        src=inspect.getsource(M.candidate_features)
        self.assertIn("OLD.online_feature_row(probabilities, selected, start",src)
        self.assertIn("OLD.prefix_features(row)",src)
    def test_gate_is_conjunctive(self):
        src=inspect.getsource(M.analyze); self.assertIn("all(gate_checks.values())",src)
    def test_predictor_always_selects_one_of_three(self):
        self.assertIn("int(np.argmax(score))",inspect.getsource(M.choose))
    def test_no_cuda_or_visual_predictor(self):
        src=PATH.read_text(); self.assertNotIn("cuda()",src); self.assertNotIn("SequenceFeatures",src)
    def test_oracle_uses_fixed_center_continuation(self):
        src=inspect.getsource(M.oracle_sample); self.assertIn("center_continuation(len(p), s + 4)",src)
    def test_decision_waits_for_offset_three(self):
        cfg=M.config(); self.assertEqual(cfg["protocol"]["bounded_lookahead_candidate_arrivals"],2)
        self.assertEqual(cfg["protocol"]["decision_time"],"when offset3 arrives")

if __name__=="__main__": unittest.main()
