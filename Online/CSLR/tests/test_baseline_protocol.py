import importlib.util
import unittest
from pathlib import Path

import yaml


CSLR_ROOT = Path(__file__).resolve().parents[1]


def load_tool(name):
    path = CSLR_ROOT / "tools" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


runner = load_tool("run_adaptive_baseline_matrix")


class BaselineProtocolTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        with runner.DEFAULT_PROTOCOL.open(encoding="utf-8") as handle:
            cls.protocol = yaml.safe_load(handle)

    def test_registered_variants(self):
        self.assertEqual(
            list(self.protocol["variants"]),
            [
                "B0_fixed1_window7",
                "B1_fixed1_span15",
                "B2_uniform_rate_span15",
                "B3_fixed2_span15",
                "B4_fixed3_span15",
                "A0_adaptive_span15",
            ],
        )

    def test_fixed_and_adaptive_resolved_configs(self):
        fixed = runner.resolved_model_config(
            self.protocol, self.protocol["variants"]["B3_fixed2_span15"]
        )
        adaptive = runner.resolved_model_config(
            self.protocol, self.protocol["variants"]["A0_adaptive_span15"]
        )
        self.assertEqual(fixed["data"]["sampling"], {"mode": "fixed", "fixed_stride": 2})
        self.assertFalse(fixed["data"]["adaptive_stride"]["enabled"])
        self.assertEqual(adaptive["data"]["sampling"]["mode"], "adaptive_motion")
        self.assertTrue(adaptive["data"]["adaptive_stride"]["enabled"])
        self.assertEqual(adaptive["postprocess"]["span_weighted_voting"]["vote_span_frames"], 15)
        self.assertTrue(fixed["runtime_profile"]["enabled"])
        self.assertEqual(
            Path(fixed["training"]["model_dir"]) / "ckpts" / "best.ckpt",
            runner.resolve_path(self.protocol["checkpoint"]),
        )

    def test_faulty_gpu_is_rejected(self):
        for uuid in runner.FAULTY_GPU_UUIDS:
            with self.assertRaises(ValueError):
                runner.validate_gpu_uuid(uuid)

    def test_budget_stride_matches_frozen_dev_counts(self):
        value = self.protocol["variants"]["B2_uniform_rate_span15"]["sampling"][
            "uniform_mean_stride"
        ]
        self.assertAlmostEqual(value, 55775 / 37615, places=12)


if __name__ == "__main__":
    unittest.main()
