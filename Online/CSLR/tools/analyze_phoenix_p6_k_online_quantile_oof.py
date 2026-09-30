#!/usr/bin/env python3
"""CPU-only source-OOF unknown-EOS K controller with causal within-video median."""
import argparse
import gzip
import hashlib
import importlib.util
import json
import os
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "results/phoenix-2014t_ISLR"
DENSE = BASE / "train_dense_stride1_v1_49faacc3"
FIT = BASE / "p6_local_js_predictor_oof_v1_49faacc3"
FACTORIAL = BASE / "p6_k_history_controller_state_factorial_audit_v1_49faacc3"
CONFIG = ROOT / "configs/p6_k_history_online_quantile_oof_v1_preregistered.json"
OUTPUT = BASE / "p6_k_history_online_quantile_oof_v1_49faacc3"
IDENTITY = OUTPUT / "preregistered_identity.json"
METRICS = OUTPUT / "metrics.json"
PER_SAMPLE = OUTPUT / "per_sample.jsonl.gz"
BOOT_SEED = 261227
EPS = 1e-9


def imp(name):
    path = ROOT / f"tools/{name}.py"; spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec); sys.modules[spec.name] = module; spec.loader.exec_module(module); return module


PRED = imp("analyze_phoenix_p6_local_js_predictor_oof")
ORACLE, RANK = PRED.ORACLE, PRED.RANK
CHRON, BUILDER, BLOCK = PRED.CHRON, PRED.BUILDER, PRED.BLOCK


def now(): return datetime.now(timezone.utc).isoformat()


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""): digest.update(block)
    return digest.hexdigest()


def write_atomic(path, value):
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp"); temporary.write_text(json.dumps(value, indent=2) + "\n"); os.replace(temporary, path)


def verify():
    config = json.loads(CONFIG.read_text()); expected = config["fixed_assets"]
    actual = {"factorial_manifest_sha256": sha(FACTORIAL / "manifest.json"),
              "factorial_metrics_sha256": sha(FACTORIAL / "metrics.json"),
              "fit_p6_dataset_sha256": sha(FIT / "dataset.npz"),
              "fit_calibration_split_sha256": sha(DENSE / "fit_calibration_split.json")}
    if actual != expected: raise RuntimeError(f"asset mismatch: {actual}")
    if json.loads((FACTORIAL / "metrics.json").read_text())["decomposition"]["diagnosis"] != "controller/pacing-dominant":
        raise RuntimeError("factorial audit did not support controller redesign")
    return config


def prepare():
    verify()
    if OUTPUT.exists() and any(OUTPUT.iterdir()):
        if not IDENTITY.is_file(): raise FileExistsError(f"nonempty output without identity: {OUTPUT}")
        identity = json.loads(IDENTITY.read_text())
        if identity["config_sha256"] != sha(CONFIG): raise RuntimeError("configuration changed after prepare")
        return identity
    OUTPUT.mkdir(parents=True, exist_ok=True)
    identity = {"created_utc": now(), "config_sha256": sha(CONFIG), "script_sha256_at_prepare": sha(Path(__file__)),
                "created_before_policy_outcomes": True, "train_fit_only": True,
                "calibration_dev_test_heldout_read": False}
    IDENTITY.write_text(json.dumps(identity, indent=2) + "\n"); return identity


def fit_models():
    with np.load(FIT / "dataset.npz", allow_pickle=False) as packed:
        x = np.asarray(packed["K"], np.float32); y = np.asarray(packed["target"], np.float32); folds = np.asarray(packed["folds"], np.int8)
    models = []
    for fold in range(5):
        keep = folds != fold; tx = x[keep]; ty = y[keep]
        mean = tx.mean(axis=0, dtype=np.float64); scale = tx.std(axis=0, dtype=np.float64); scale[scale < 1e-6] = 1.0
        standardized = (np.asarray(tx, np.float64) - mean) / scale; intercept = float(ty.mean())
        gram = standardized.T @ standardized; gram.flat[::len(gram) + 1] += 100.0
        weights = np.linalg.solve(gram, standardized.T @ (np.asarray(ty, np.float64) - intercept))
        models.append((mean, scale, weights, intercept)); print(json.dumps({"fit_fold": fold}), flush=True)
    return models


def score(feature, model):
    mean, scale, weights, intercept = model
    return float(((np.asarray(feature, np.float64) - mean) / scale) @ weights + intercept)


def matched_indices(block_count, count):
    if count == 0: return []
    result = [int(np.floor((index + .5) * block_count / count)) for index in range(count)]
    if len(set(result)) != count: raise RuntimeError("matched uniform duplicate")
    return result


def schedules(probabilities, frame_features, model):
    blocks = ORACLE.complete_blocks(len(probabilities)); skeleton = ORACLE.base_skeleton(len(probabilities))
    selected, offsets, past_scores = [], [], []
    balance = 0.0; forced = autonomous = 0
    for start in blocks:
        value = score(RANK.k_history(frame_features, selected, start, offsets), model)
        balance = min(2.0, balance + .5); execute = False
        if balance + EPS >= 1.0:
            if balance + EPS >= 2.0:
                execute = True; forced += 1
            elif past_scores and value >= float(np.median(np.asarray(past_scores, np.float64))):
                execute = True; autonomous += 1
        selected.append(start)
        if execute: selected.append(start + 2); offsets.append(2); balance -= 1.0
        past_scores.append(value)
    complete = set(blocks); selected.extend(start for start in skeleton if start not in complete); selected = sorted(set(selected))
    online, ubalance = [], 0.0
    for start in blocks:
        online.append(start); ubalance = min(2.0, ubalance + .5)
        if ubalance + EPS >= 1.0: online.append(start + 2); ubalance -= 1.0
    online.extend(start for start in skeleton if start not in complete)
    bonus = sum(value % 4 == 2 for value in selected)
    matched = sorted(skeleton + [blocks[index] + 2 for index in matched_indices(len(blocks), bonus)])
    full = sorted(skeleton + [start + 2 for start in blocks])
    return {"quantile": selected, "online_uniform": sorted(online), "matched_uniform": matched, "full_center": full}, {
        "forced": forced, "autonomous": autonomous, "unspent": balance}


def bootstrap(rows, left, right, repetitions=10000):
    grouped = defaultdict(lambda: [0, 0])
    for row in rows: grouped[row["source"]][0] += row[left] - row[right]; grouped[row["source"]][1] += row["ref_len"]
    values = np.asarray(list(grouped.values()), np.float64); rng = np.random.default_rng(BOOT_SEED); result = np.empty(repetitions)
    for index in range(repetitions):
        chosen = rng.integers(0, len(values), len(values)); sample = values[chosen]; result[index] = 100.0 * sample[:, 0].sum() / sample[:, 1].sum()
    return [float(np.quantile(result, .025)), float(np.quantile(result, .975))]


def run():
    config = verify(); prepare()
    if METRICS.exists() or PER_SAMPLE.exists(): raise FileExistsError("policy outputs exist")
    models = fit_models()
    with np.load(FIT / "dataset.npz", allow_pickle=False) as packed:
        names = [str(value) for value in packed["sample_names"].tolist()]
        sample_indices = np.asarray(packed["sample_indices"], np.int64); folds = np.asarray(packed["folds"], np.int8)
    sample_fold = {}
    for index, name in enumerate(names):
        rows = np.flatnonzero(sample_indices == index); unique = np.unique(folds[rows])
        if len(unique) != 1: raise RuntimeError(f"sample fold mismatch: {name}")
        sample_fold[name] = int(unique[0])
    wanted = set(names); vocab = json.loads(CHRON.DEFAULT_VOCAB.read_text()); blank = vocab.index("<blank>")
    rng = np.random.default_rng(261130); buckets = rng.integers(0, 32, size=len(vocab)); signs = rng.choice(np.asarray([-1., 1.], np.float32), size=len(vocab))
    indices, shard_count = CHRON.completed_shard_indices(DENSE); output_rows = []
    with gzip.open(PER_SAMPLE, "wt", encoding="utf-8", compresslevel=6) as handle:
        for shard in indices:
            results, logits, _ = BUILDER.BUILDER.validate_dense_shard(DENSE, shard, shard_count, verify_hashes=False)
            for name in results:
                if name not in wanted: continue
                probabilities = BUILDER.ORACLE.softmax_rows(np.asarray(logits[name])); frame_features = RANK.dense_frame_features(probabilities, blank, buckets, signs)
                policies, trace = schedules(probabilities, frame_features, models[sample_fold[name]])
                reference = BUILDER.REPLAY.clean_phoenix_2014_trans(results[name]["gls_ref"])
                decoded = {key: BLOCK.decode_counts(probabilities, windows, reference, vocab, blank) for key, windows in policies.items()}
                row = {"sample": name, "source": BUILDER.source_video(name), "fold": sample_fold[name], "ref_len": int(decoded["quantile"]["ref_len"]), **trace}
                for key in policies: row[key] = int(decoded[key]["error"]); row[f"{key}_windows"] = len(policies[key])
                output_rows.append(row); handle.write(json.dumps(row) + "\n")
            print(json.dumps({"shard": shard, "samples": len(output_rows)}), flush=True)
    keys = ("quantile", "online_uniform", "matched_uniform", "full_center")
    totals = {key: sum(row[key] for row in output_rows) for key in keys}; windows = {key: sum(row[f"{key}_windows"] for row in output_rows) for key in keys}; ref = sum(row["ref_len"] for row in output_rows)
    comparisons = {}
    for right in ("online_uniform", "matched_uniform", "full_center"):
        comparisons[f"quantile_vs_{right}"] = {"delta_errors": totals["quantile"] - totals[right], "delta_wer_pp": 100.0 * (totals["quantile"] - totals[right]) / ref,
            "source_cluster_bootstrap_ci95": bootstrap(output_rows, "quantile", right), "delta_windows": windows["quantile"] - windows[right]}
    gates = config["primary_gates"]
    checks = {"quantile_windows": windows["quantile"] <= windows["online_uniform"],
        "online_errors": totals["online_uniform"] - totals["quantile"] >= int(gates["error_reduction_vs_online_uniform_at_least"]),
        "online_ci": comparisons["quantile_vs_online_uniform"]["source_cluster_bootstrap_ci95"][1] < 0,
        "matched_errors": totals["matched_uniform"] - totals["quantile"] >= int(gates["error_reduction_vs_matched_count_uniform_at_least"]),
        "matched_ci": comparisons["quantile_vs_matched_uniform"]["source_cluster_bootstrap_ci95"][1] < 0}
    passed = all(checks.values())
    metrics = {"status": "complete", "created_utc": now(), "gpu_used": False, "train_fit_only": True,
        "source_disjoint_oof": True, "unknown_eos": True, "tail_topup": False, "self_history": True,
        "calibration_dev_test_heldout_read": False, "samples": len(output_rows), "sources": 578,
        "reference_glosses": ref, "errors": totals, "wer": {key: 100.0 * value / ref for key, value in totals.items()},
        "selected_windows": windows, "comparisons": comparisons,
        "controller": {"forced": sum(row["forced"] for row in output_rows), "autonomous": sum(row["autonomous"] for row in output_rows), "unspent_tokens": sum(row["unspent"] for row in output_rows)},
        "checks": checks, "gate_passed": passed,
        "decision": ("request explicit authorization for separately frozen final held-out planning with multiplicity caveat" if passed else "stop causal quantile controller identity")}
    write_atomic(METRICS, metrics); manifest = {"status": "complete", "config_sha256": sha(CONFIG), "identity_sha256": sha(IDENTITY), "metrics_sha256": sha(METRICS), "per_sample_sha256": sha(PER_SAMPLE), "gate_passed": passed, "calibration_dev_test_heldout_read": False}; write_atomic(OUTPUT / "manifest.json", manifest); return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(); parser.add_argument("--prepare", action="store_true"); parser.add_argument("--run", action="store_true"); args = parser.parse_args()
    if args.prepare == args.run: parser.error("choose exactly one of --prepare or --run")
    os.environ["CUDA_VISIBLE_DEVICES"] = ""; print(json.dumps(prepare() if args.prepare else run(), indent=2))
