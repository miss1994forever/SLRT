import importlib.util
import json
import pickle
import tempfile
import unittest
from pathlib import Path

import yaml


CSLR_ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = CSLR_ROOT / "tools" / "evaluate_adaptive_baseline_matrix.py"
SPEC = importlib.util.spec_from_file_location("baseline_evaluator", MODULE_PATH)
EVALUATOR = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(EVALUATOR)


def variant(variant_id, errors, clips, wer):
    names = [f"sample_{index}" for index in range(len(errors))]
    return {
        "variant_id": variant_id,
        "references": {name: "A B" for name in names},
        "sentence_errors": dict(zip(names, errors)),
        "sentence_ref_lengths": {name: 2 for name in names},
        "metric": {"wer": wer},
        "clip_count": clips,
        "runtime": None,
    }


class BaselineEvaluatorTests(unittest.TestCase):
    def test_load_variant_reads_registered_decoder_and_metadata(self):
        protocol = {
            "checkpoint": "results/model/ckpts/best.ckpt",
            "common_inference": {
                "span_frames": 15,
                "span_kernel": "triangular",
                "span_min_weight": 0.05,
                "prediction_source": "ensemble",
                "blank_threshold": 0.5,
            },
            "variants": {
                "B0": {
                    "report_decoder": "window_greedy_7",
                    "sampling": {"mode": "fixed", "fixed_stride": 1},
                    "span_weighted_voting": True,
                }
            }
        }
        with tempfile.TemporaryDirectory() as directory:
            run_dir = Path(directory) / "dev" / "B0" / "run_01"
            run_dir.mkdir(parents=True)
            (run_dir / "complete.json").write_text("{}\n", encoding="utf-8")
            (run_dir / "resolved_config.yaml").write_text(
                yaml.safe_dump({
                    "data": {"sampling": {"mode": "fixed", "fixed_stride": 1}},
                    "postprocess": {"span_weighted_voting": {
                        "enabled": True,
                        "vote_span_frames": 15,
                        "kernel": "triangular",
                        "min_weight": 0.05,
                    }},
                }),
                encoding="utf-8",
            )
            (run_dir / "command.json").write_text(
                json.dumps([
                    "python", "prediction_slide.py", "--split", "dev",
                    "--pred_src", "ensemble", "--blank_thr", "0.5",
                    "--ckpt_name", "best.ckpt",
                ]),
                encoding="utf-8",
            )
            (run_dir / "runtime.json").write_text(
                json.dumps({"returncode": 0, "wall_time_seconds": 2.5}), encoding="utf-8"
            )
            results = {
                "sample": {
                    "gls_ref": "A B",
                    "window_greedy_7_gls_hyp": "A",
                    "adaptive_stride_metadata": [
                        {"stride": 1, "sampling_mode": "fixed"},
                        {"stride": 1, "sampling_mode": "fixed"},
                    ],
                }
            }
            evaluation = {
                "wer_window_greedy_7": {
                    "wer": 50.0,
                    "del": 50.0,
                    "ins": 0.0,
                    "sub": 0.0,
                    "ref_len": 2,
                    "error": 1,
                }
            }
            with (run_dir / "dev_results.pkl").open("wb") as handle:
                pickle.dump(results, handle)
            with (run_dir / "dev_evaluation_results.pkl").open("wb") as handle:
                pickle.dump(evaluation, handle)

            loaded = EVALUATOR.load_variant(protocol, Path(directory), "dev", "B0")
            self.assertEqual(loaded["metric"]["wer"], 50.0)
            self.assertEqual(loaded["clip_count"], 2)
            self.assertEqual(loaded["sampling_mode_counts"], {"fixed": 2})
            self.assertEqual(loaded["runtime"]["median_seconds"], 2.5)

    def test_paired_bootstrap_is_deterministic(self):
        baseline = variant("B0", [0, 1, 0, 1], 100, 25.0)
        candidate = variant("A0", [0, 0, 0, 1], 70, 12.5)
        first = EVALUATOR.paired_bootstrap(candidate, baseline, samples=100, seed=7)
        second = EVALUATOR.paired_bootstrap(candidate, baseline, samples=100, seed=7)
        self.assertEqual(first, second)
        self.assertEqual(first["wer_difference_pp"], -12.5)

    def test_budget_constraint_passes_within_tolerance(self):
        protocol = {
            "variants": {
                "B2": {
                    "budget_reference": "A0",
                    "maximum_budget_difference_percent": 2.0,
                },
                "A0": {},
            }
        }
        checks = EVALUATOR.validate_budget_constraints(
            protocol,
            [variant("B2", [0], 101, 0.0), variant("A0", [0], 100, 0.0)],
        )
        self.assertTrue(checks["B2"]["passed"])

    def test_budget_constraint_rejects_mismatch(self):
        protocol = {
            "variants": {
                "B2": {
                    "budget_reference": "A0",
                    "maximum_budget_difference_percent": 2.0,
                },
                "A0": {},
            }
        }
        with self.assertRaisesRegex(ValueError, "budget differs"):
            EVALUATOR.validate_budget_constraints(
                protocol,
                [variant("B2", [0], 110, 0.0), variant("A0", [0], 100, 0.0)],
            )


if __name__ == "__main__":
    unittest.main()
