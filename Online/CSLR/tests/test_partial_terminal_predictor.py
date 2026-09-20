import hashlib, importlib.util, inspect, json, sys, tempfile, unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PATH = ROOT / "tools/analyze_phoenix_partial_terminal_predictor.py"
SPEC = importlib.util.spec_from_file_location("terminal_predictor_tested", PATH)
M = importlib.util.module_from_spec(SPEC); sys.modules[SPEC.name] = M; SPEC.loader.exec_module(M)


class TerminalPredictorProtocolTests(unittest.TestCase):
    def test_hash_subset_ranking_is_stable(self):
        names = ["z", "a", "b"]
        once = sorted(names, key=lambda x: (M.stable_rank(x), x))
        twice = sorted(reversed(names), key=lambda x: (M.stable_rank(x), x))
        self.assertEqual(once, twice)

    def test_causal_feature_ignores_future_label_fields(self):
        row = {"bookkeeping": {"x": 1}, "prefix": {"y": 2},
               "reference": "future", "remaining_windows": 99,
               "terminal_advantage": 2, "future_logits": [1]}
        self.assertEqual(M.causal_feature(row), {"bookkeeping": {"x": 1}, "prefix": {"y": 2}})

    def test_prefix_contract_is_past_selected(self):
        source = inspect.getsource(M.sample_rows)
        self.assertIn("OLD.online_feature_row(probabilities, selected, start", source)
        self.assertLess(source.index("OLD.online_feature_row"), source.index("AUDIT.terminal_advantage"))

    def test_terminal_branch_uses_same_frozen_policy(self):
        source = inspect.getsource(M.sample_rows)
        self.assertIn("AUDIT.terminal_advantage(\n                base", source)
        self.assertIn('"continuation_same_frozen_B2_old": True', source)

    def test_forced_states_are_not_serialized(self):
        source = inspect.getsource(M.sample_rows)
        forced = source.index("if forced:")
        autonomous = source.index("else:", forced)
        append = source.index("rows.append", autonomous)
        self.assertGreater(append, autonomous)

    def test_unknown_endpoint_not_in_feature_names(self):
        row = {"bookkeeping": {"candidate_absolute_index": 3}, "prefix": {"entropy": .2}}
        self.assertNotIn("T", M.causal_feature(row)); self.assertNotIn("EOS", M.causal_feature(row))
        self.assertNotIn("remaining", M.causal_feature(row))

    def test_gate_requires_every_preregistered_condition(self):
        base = {"pr_auc_regression_score": .1, "positive_recall_at_top_10pct": .2,
                "ndcg_at_top_10pct": .3, "score_decile_spearman": .1,
                "highest_minus_lowest_decile_mean_advantage": .1}
        better = dict(base); better.update(pr_auc_regression_score=.2,
            positive_recall_at_top_10pct=.3, ndcg_at_top_10pct=.4)
        self.assertTrue(M.gate({"T0": base, "T1": better})["passed"])
        better["ndcg_at_top_10pct"] = .2
        self.assertFalse(M.gate({"T0": base, "T1": better})["passed"])

    def test_threshold_is_fit_only_and_calibration_untuned(self):
        source = inspect.getsource(M.analyze)
        self.assertIn("np.quantile(fit_score, .90)", source)
        self.assertNotIn("np.quantile(score", source)
        self.assertIn("models[label] = (kind, model, stats, 0.0)", source)

    def test_atomic_hash_helper(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "x"; path.write_bytes(b"abc")
            self.assertEqual(M.sha256_file(path), hashlib.sha256(b"abc").hexdigest())

    def test_source_disjoint_guard_exists(self):
        self.assertIn('sources["fit"] & sources["calibration"]', inspect.getsource(M.load_data))

    def test_no_visual_model_or_cuda_execution(self):
        source = PATH.read_text()
        self.assertNotIn("cuda()", source); self.assertNotIn(".to(\"cuda\")", source)
        self.assertEqual(M.KIND0, "B0_bookkeeping"); self.assertEqual(M.KIND1, "B2_bookkeeping_prefix")


if __name__ == "__main__": unittest.main()
