#!/usr/bin/env python3
"""Read-only integrity and arithmetic audit of frozen train-fit P6 outputs."""
import argparse
from collections import defaultdict
import gzip
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "results/phoenix-2014t_ISLR"
OUT = BASE / "p6_k_history_online_quantile_oof_v1_49faacc3"
CONFIG = ROOT / "configs/p6_k_history_online_quantile_oof_v1_preregistered.json"
SCRIPT = ROOT / "tools/analyze_phoenix_p6_k_online_quantile_oof.py"
ASSETS = {
    "factorial_manifest_sha256": BASE / "p6_k_history_controller_state_factorial_audit_v1_49faacc3/manifest.json",
    "factorial_metrics_sha256": BASE / "p6_k_history_controller_state_factorial_audit_v1_49faacc3/metrics.json",
    "fit_p6_dataset_sha256": BASE / "p6_local_js_predictor_oof_v1_49faacc3/dataset.npz",
    "fit_calibration_split_sha256": BASE / "train_dense_stride1_v1_49faacc3/fit_calibration_split.json",
}


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def audit():
    config = json.loads(CONFIG.read_text())
    identity = json.loads((OUT / "preregistered_identity.json").read_text())
    manifest = json.loads((OUT / "manifest.json").read_text())
    metrics = json.loads((OUT / "metrics.json").read_text())
    checks = {}
    checks["frozen_config"] = sha(CONFIG) == identity["config_sha256"] == manifest["config_sha256"]
    checks["frozen_simulator"] = sha(SCRIPT) == identity["script_sha256_at_prepare"]
    checks["upstream_assets"] = all(sha(path) == config["fixed_assets"][name] for name, path in ASSETS.items())
    checks["outcome_artifacts"] = all(sha(path) == manifest[key] for key, path in {
        "identity_sha256": OUT / "preregistered_identity.json",
        "metrics_sha256": OUT / "metrics.json",
        "per_sample_sha256": OUT / "per_sample.jsonl.gz",
    }.items())
    keys = ("quantile", "online_uniform", "matched_uniform", "full_center")
    errors = defaultdict(int)
    windows = defaultdict(int)
    fold_by_source = {}
    samples = sources = reference = forced = autonomous = 0
    with gzip.open(OUT / "per_sample.jsonl.gz", "rt", encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            samples += 1
            source, fold = row["source"], row["fold"]
            if source in fold_by_source and fold_by_source[source] != fold:
                raise AssertionError("source crosses folds")
            fold_by_source[source] = fold
            expected_fold = int.from_bytes(hashlib.sha256(f"robust-oof-v1\0{source}".encode()).digest()[:8], "big") % 5
            if fold != expected_fold:
                raise AssertionError("source fold identity changed")
            if row["quantile_windows"] != row["matched_uniform_windows"]:
                raise AssertionError("matched budget differs from quantile")
            if row["quantile_windows"] > row["full_center_windows"]:
                raise AssertionError("quantile exceeds full center")
            if row["forced"] + row["autonomous"] > row["quantile_windows"]:
                raise AssertionError("invalid bonus count")
            coverage = (row["full_center_windows"] + 1) // 2
            if row["quantile_windows"] != coverage + row["forced"] + row["autonomous"]:
                raise AssertionError("coverage or bonus accounting changed")
            for key in keys:
                errors[key] += row[key]
                windows[key] += row[f"{key}_windows"]
            reference += row["ref_len"]
            forced += row["forced"]
            autonomous += row["autonomous"]
    sources = len(fold_by_source)
    checks["source_disjoint_folds"] = sources == 578
    checks["sample_totals"] = (samples == metrics["samples"] == 6378 and
        reference == metrics["reference_glosses"] and
        dict(errors) == metrics["errors"] and dict(windows) == metrics["selected_windows"] and
        forced == metrics["controller"]["forced"] and autonomous == metrics["controller"]["autonomous"])
    checks["budget_rules"] = (windows["quantile"] == windows["matched_uniform"] and
        windows["quantile"] <= windows["online_uniform"])
    checks["frozen_gates"] = metrics["gate_passed"] and all(metrics["checks"].values())
    checks["scope"] = (identity["train_fit_only"] and not identity["calibration_dev_test_heldout_read"] and
        metrics["train_fit_only"] and not metrics["calibration_dev_test_heldout_read"] and not metrics["gpu_used"])
    return {"passed": all(checks.values()), "checks": checks, "samples": samples,
            "sources": sources, "errors": dict(errors), "windows": dict(windows),
            "config_sha256": sha(CONFIG), "frozen_script_sha256": sha(SCRIPT),
            "fit_dataset_sha256": sha(ASSETS["fit_p6_dataset_sha256"])}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output")
    args = parser.parse_args()
    result = audit()
    if args.output:
        path = Path(args.output)
        if path.exists():
            raise FileExistsError(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    if not result["passed"]:
        raise SystemExit(1)
