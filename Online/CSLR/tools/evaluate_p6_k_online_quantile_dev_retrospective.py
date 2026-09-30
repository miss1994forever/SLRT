#!/usr/bin/env python3
"""One-shot CPU replay of frozen P6 controller on authorized repaired dev."""
import argparse
import gzip
import hashlib
import importlib.util
import json
import os
import pickle
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "results/phoenix-2014t_ISLR"
CONFIG = ROOT / "configs/p6_k_online_quantile_dev_retrospective_v1_preregistered.json"
FROZEN_SCRIPT = ROOT / "tools/analyze_phoenix_p6_k_online_quantile_oof.py"
FROZEN_CONFIG = ROOT / "configs/p6_k_history_online_quantile_oof_v1_preregistered.json"
FIT = BASE / "p6_local_js_predictor_oof_v1_49faacc3/dataset.npz"
DEV_BASE = BASE / "baseline_matrix_v1_repaired_49faacc3"
DEV = DEV_BASE / "dev/B0_fixed1_window7/run_01"
OUTPUT = BASE / "p6_k_online_quantile_dev_retrospective_v1_49faacc3"
ASSETS = {
    "fit_p6_dataset_sha256": FIT,
    "fit_controller_config_sha256": FROZEN_CONFIG,
    "fit_controller_script_sha256": FROZEN_SCRIPT,
    "dev_baseline_manifest_sha256": DEV_BASE / "protocol_manifest.json",
    "dev_results_sha256": DEV / "dev_results.pkl",
    "dev_logits_sha256": DEV / "dev_logits.pkl",
    "recognizer_checkpoint_sha256": BASE / "ckpts/best.ckpt",
    "vocab_sha256": ROOT.parents[1] / "data/phoenix_2014t/phoenix_iso_with_blank.vocab",
}


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def module_at(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def now():
    return datetime.now(timezone.utc).isoformat()


def check_assets():
    config = json.loads(CONFIG.read_text())
    for key, path in ASSETS.items():
        actual = sha(path)
        if actual != config["fixed_assets"][key]:
            raise RuntimeError(f"frozen asset mismatch: {key}: {actual}")
    return config


def prepare():
    check_assets()
    if OUTPUT.exists() and any(OUTPUT.iterdir()):
        raise FileExistsError(f"retrospective identity already exists: {OUTPUT}")
    OUTPUT.mkdir(parents=True, exist_ok=True)
    identity = {
        "created_utc": now(),
        "config_sha256": sha(CONFIG),
        "script_sha256": sha(Path(__file__)),
        "dev_policy_outcomes_read": False,
        "test_read": False,
        "evaluation_once": True,
    }
    (OUTPUT / "preregistered_identity.json").write_text(json.dumps(identity, indent=2) + "\n")
    return identity


def fit_full_k():
    with np.load(FIT, allow_pickle=False) as packed:
        x = np.asarray(packed["K"], np.float32)
        y = np.asarray(packed["target"], np.float32)
        names = packed["sample_names"]
    if x.shape != (183401, 153) or len(names) != 6378 or len(y) != len(x):
        raise RuntimeError("fit-only K dataset scope changed")
    mean = x.mean(axis=0, dtype=np.float64)
    scale = x.std(axis=0, dtype=np.float64)
    scale[scale < 1e-6] = 1.0
    standardized = (np.asarray(x, np.float64) - mean) / scale
    intercept = float(y.mean())
    gram = standardized.T @ standardized
    gram.flat[::len(gram) + 1] += 100.0
    weights = np.linalg.solve(gram, standardized.T @ (np.asarray(y, np.float64) - intercept))
    return mean, scale, weights, intercept


def run():
    config = check_assets()
    identity_path = OUTPUT / "preregistered_identity.json"
    if not identity_path.is_file():
        raise RuntimeError("prepare identity before outcome access")
    identity = json.loads(identity_path.read_text())
    if identity["config_sha256"] != sha(CONFIG) or identity["script_sha256"] != sha(Path(__file__)):
        raise RuntimeError("script or config changed after freeze")
    if any((OUTPUT / name).exists() for name in ("metrics.json", "per_sample.jsonl.gz", "manifest.json")):
        raise FileExistsError("one-shot dev outcome already exists")

    frozen = module_at(FROZEN_SCRIPT, "frozen_p6_quantile_dev_replay")
    model = fit_full_k()
    with (DEV / "dev_results.pkl").open("rb") as handle:
        results = pickle.load(handle)
    with (DEV / "dev_logits.pkl").open("rb") as handle:
        logits = pickle.load(handle)
    if len(results) != 519 or set(results) != set(logits) or any(not name.startswith("dev/") for name in results):
        raise RuntimeError("repaired dev scope changed")
    vocab = json.loads(frozen.CHRON.DEFAULT_VOCAB.read_text())
    blank = vocab.index("<blank>")
    rng = np.random.default_rng(261130)
    buckets = rng.integers(0, 32, size=len(vocab))
    signs = rng.choice(np.asarray([-1., 1.], np.float32), size=len(vocab))
    keys = ("quantile", "matched_uniform", "online_uniform", "full_center")
    rows = []
    for name in sorted(results):
        probabilities = frozen.BUILDER.ORACLE.softmax_rows(np.asarray(logits[name]))
        frame_features = frozen.RANK.dense_frame_features(probabilities, blank, buckets, signs)
        policies, trace = frozen.schedules(probabilities, frame_features, model)
        if len(policies["quantile"]) != len(policies["matched_uniform"]):
            raise RuntimeError("per-video matched budget failed")
        reference = frozen.BUILDER.REPLAY.clean_phoenix_2014_trans(results[name]["gls_ref"])
        counts = {key: frozen.BLOCK.decode_counts(probabilities, policies[key], reference, vocab, blank)
                  for key in keys}
        row = {"sample": name, "source": frozen.BUILDER.source_video(name),
               "ref_len": int(counts["quantile"]["ref_len"]), **trace}
        for key in keys:
            row[key] = int(counts[key]["error"])
            row[f"{key}_windows"] = len(policies[key])
        rows.append(row)

    if len(rows) != 519 or len({row["sample"] for row in rows}) != 519:
        raise RuntimeError("dev output incomplete")
    totals = {key: sum(row[key] for row in rows) for key in keys}
    windows = {key: sum(row[f"{key}_windows"] for row in rows) for key in keys}
    ref = sum(row["ref_len"] for row in rows)
    comparisons = {}
    for right in ("matched_uniform", "online_uniform", "full_center"):
        comparisons[f"quantile_vs_{right}"] = {
            "delta_errors": totals["quantile"] - totals[right],
            "delta_wer_pp": 100.0 * (totals["quantile"] - totals[right]) / ref,
            "source_cluster_bootstrap_ci95": frozen.bootstrap(rows, "quantile", right),
            "delta_windows": windows["quantile"] - windows[right],
        }
    if windows["quantile"] != windows["matched_uniform"] or not all(
        row["quantile_windows"] == row["matched_uniform_windows"] for row in rows
    ):
        raise RuntimeError("matched uniform window budget failed")
    ci = comparisons["quantile_vs_matched_uniform"]["source_cluster_bootstrap_ci95"]
    evidence = ("directional retrospective evidence" if totals["quantile"] < totals["matched_uniform"] and ci[1] < 0
                else "contrary direction" if totals["quantile"] > totals["matched_uniform"] and ci[0] > 0
                else "insufficient directional evidence")
    metrics = {"status": "complete", "created_utc": now(), "scope": config["scope"], "gpu_used": False,
               "samples": len(rows), "sources": len({row["source"] for row in rows}), "reference_glosses": ref,
               "errors": totals, "wer": {key: 100.0 * value / ref for key, value in totals.items()},
               "selected_windows": windows, "comparisons": comparisons, "main_result": evidence,
               "test_read": False, "model": config["model"], "decoder": config["decoder"]}
    with gzip.open(OUTPUT / "per_sample.jsonl.gz", "wt", encoding="utf-8", compresslevel=6) as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")
    (OUTPUT / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    manifest = {"status": "complete", "config_sha256": sha(CONFIG), "script_sha256": sha(Path(__file__)),
                "identity_sha256": sha(identity_path), "metrics_sha256": sha(OUTPUT / "metrics.json"),
                "per_sample_sha256": sha(OUTPUT / "per_sample.jsonl.gz"), "test_read": False}
    (OUTPUT / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    commands = parser.add_mutually_exclusive_group(required=True)
    commands.add_argument("--prepare", action="store_true")
    commands.add_argument("--run", action="store_true")
    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    print(json.dumps(prepare() if args.prepare else run(), indent=2))
