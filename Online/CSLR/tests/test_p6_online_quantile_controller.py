"""Frozen-rule acceptance checks without training or reading held-out data."""
import importlib.util
from pathlib import Path
from statistics import median
import unittest

MODULE = Path(__file__).resolve().parents[1] / "utils/online_quantile_controller.py"
spec = importlib.util.spec_from_file_location("online_quantile_controller", MODULE)
mod = importlib.util.module_from_spec(spec)
import sys
sys.modules[spec.name] = mod
spec.loader.exec_module(mod)


def frozen_reference(scores, total):
    """Small independent transcription of frozen schedules(), using known length."""
    selected, past, balance = [], [], 0.0
    for start in range(0, total, 4):
        if start + 3 >= total:
            selected.append(start)
            continue
        value = scores[start // 4]
        balance = min(2.0, balance + 0.5)
        execute = balance >= 2.0 or (balance >= 1.0 and past and value >= median(past))
        selected.append(start)
        if execute:
            selected.append(start + 2)
            balance -= 1.0
        past.append(value)
    return sorted(selected), balance


class FrozenControllerTests(unittest.TestCase):
    def test_streaming_matches_frozen_for_lengths_and_scores(self):
        for total in range(0, 61):
            for scores in ([0.0] * 16, list(range(16)), list(range(16, 0, -1))):
                seen = []
                def score(start, selected):
                    self.assertEqual(selected, seen)
                    self.assertTrue(all(window < start for window in selected))
                    return scores[start // 4]
                policy = mod.OnlineQuantileController(score)
                for start in range(0, total - total % 4, 4):
                    policy.start_block(start)
                    policy.complete_block(start)
                    seen = policy.selected.copy()
                tail = total - total % 4 if total % 4 else None
                if tail is not None:
                    policy.start_block(tail)
                self.assertEqual((policy.finish(), policy.balance), frozen_reference(scores, total))
                self.assertEqual(policy.forced + policy.autonomous,
                                 sum(window % 4 == 2 for window in policy.selected))

    def test_future_and_tail_cannot_change_completed_decisions(self):
        def run(total, future):
            policy = mod.OnlineQuantileController(lambda start, selected: future[start // 4])
            for start in range(0, total - total % 4, 4):
                policy.start_block(start)
                policy.complete_block(start)
            before = policy.selected.copy(), policy.balance, policy.past_scores.copy()
            if total % 4:
                policy.start_block(total - total % 4)
            policy.finish()
            return before, policy
        prefix = [0.1, 0.2, 0.3]
        short = run(12, prefix)[0]
        long = run(20, prefix + [999, -999])[0]
        self.assertEqual(short[0], [window for window in long[0] if window < 12])
        self.assertEqual(short[2], long[2][:3])
        _, policy = run(15, prefix)
        self.assertEqual(policy.selected[-1], 12)
        self.assertEqual(policy.balance, run(12, prefix)[1].balance)

    def test_history_uses_only_paid_features_and_current(self):
        features = {0: [1.0] * 36, 2: [999.0] * 36, 4: [3.0] * 36}
        value = mod.k_history_feature(features, [0], 4, [])
        self.assertEqual(len(value), 153)
        self.assertEqual(value[:36], [3.0] * 36)
        self.assertEqual(value[36:72], [2.0] * 36)
        self.assertEqual(value[72:108], [1.0] * 36)
        self.assertEqual(value[108:144], [2.0] * 36)
        self.assertEqual(value[-4:], [0.0] * 4)
        class Bundle:
            def score(self, feature, fold):
                self.assert_feature = feature
                self.fold = fold
                return -1.0
        bundle = Bundle()
        callback = mod.ridge_score_callback(bundle, 3, features)
        self.assertEqual(callback(4, [0]), -1.0)
        self.assertEqual(bundle.assert_feature, value)
        self.assertEqual(bundle.fold, 3)

    def test_invalid_order_or_score_rejected(self):
        policy = mod.OnlineQuantileController(lambda *_: float("nan"))
        policy.start_block(0)
        with self.assertRaises(ValueError): policy.complete_block(0)
        policy = mod.OnlineQuantileController(lambda *_: 0.0)
        with self.assertRaises(ValueError): policy.complete_block(4)
        policy.start_block(0)
        policy.complete_block(0)
        with self.assertRaises(ValueError): policy.start_block(0)
        policy.finish()
        with self.assertRaises(ValueError): policy.complete_block(4)


if __name__ == "__main__":
    unittest.main()
