#!/usr/bin/env python3
"""CPU-only unknown-EOS self-history simulation for the independently confirmed P6 K model."""
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
CONFIRM = BASE / "p6_k_history_independent_calibration_v1_49faacc3"
CONFIG = ROOT / "configs/p6_k_history_unknown_eos_self_history_v1_preregistered.json"
OUTPUT = BASE / "p6_k_history_unknown_eos_self_history_v1_49faacc3"
IDENTITY = OUTPUT / "preregistered_identity.json"
METRICS = OUTPUT / "metrics.json"
PER_SAMPLE = OUTPUT / "per_sample.jsonl.gz"
BOOT_SEED = 261225
EPS = 1e-9


def imp(name):
    path = ROOT / f"tools/{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


PRED = imp("analyze_phoenix_p6_local_js_predictor_oof")
ORACLE, RANK = PRED.ORACLE, PRED.RANK
CHRON, BUILDER, BLOCK = PRED.CHRON, PRED.BUILDER, PRED.BLOCK


def now():
    return datetime.now(timezone.utc).isoformat()


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_atomic(path, value):
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    os.replace(temporary, path)


def verify():
    config = json.loads(CONFIG.read_text())
    expected = config["fixed_assets"]
    actual = {
        "confirmation_manifest_sha256": sha(CONFIRM / "manifest.json"),
        "confirmation_metrics_sha256": sha(CONFIRM / "metrics.json"),
        "calibration_dataset_sha256": sha(CONFIRM / "calibration_dataset.npz"),
        "fit_oof_predictions_sha256": sha(FIT / "oof_predictions.npz"),
        "fit_p6_dataset_sha256": sha(FIT / "dataset.npz"),
    }
    if actual != expected:
        raise RuntimeError(f"asset mismatch: {actual}")
    confirmation = json.loads((CONFIRM / "metrics.json").read_text())
    if not confirmation["gate_passed"]:
        raise RuntimeError("independent confirmation did not unlock simulation")
    return config


def prepare():
    config = verify()
    if OUTPUT.exists() and any(OUTPUT.iterdir()):
        if not IDENTITY.is_file():
            raise FileExistsError(f"nonempty output without identity: {OUTPUT}")
        identity = json.loads(IDENTITY.read_text())
        if identity["config_sha256"] != sha(CONFIG):
            raise RuntimeError("configuration changed after prepare")
        return identity
    OUTPUT.mkdir(parents=True, exist_ok=True)
    identity = {"created_utc": now(), "config_sha256": sha(CONFIG),
                "script_sha256_at_prepare": sha(Path(__file__)),
                "created_before_closed_loop_outcomes": True, "unknown_eos": True,
                "self_history": True, "calibration_tuning": False,
                "dev_test_heldout_read": False}
    IDENTITY.write_text(json.dumps(identity, indent=2) + "\n")
    return identity


def fit_model():
    with np.load(FIT / "dataset.npz", allow_pickle=False) as packed:
        train_x = np.asarray(packed["K"], np.float32)
        train_y = np.asarray(packed["target"], np.float32)
    mean = train_x.mean(axis=0, dtype=np.float64)
    scale = train_x.std(axis=0, dtype=np.float64); scale[scale < 1e-6] = 1.0
    x = (np.asarray(train_x, np.float64) - mean) / scale
    ymean = float(train_y.mean()); yc = np.asarray(train_y, np.float64) - ymean
    gram = x.T @ x; gram.flat[::len(gram) + 1] += 100.0
    weights = np.linalg.solve(gram, x.T @ yc)
    return mean, scale, weights, ymean


def score(feature, model):
    mean, scale, weights, intercept = model
    return float(((np.asarray(feature, np.float64) - mean) / scale) @ weights + intercept)


def matched_uniform_indices(block_count, count):
    if count == 0:
        return []
    chosen = [int(np.floor((index + 0.5) * block_count / count)) for index in range(count)]
    if len(set(chosen)) != count:
        raise RuntimeError("matched uniform duplicate")
    return chosen


def schedules(probabilities, frame_features, model, threshold):
    blocks = ORACLE.complete_blocks(len(probabilities))
    skeleton = ORACLE.base_skeleton(len(probabilities))
    selected, previous_offsets = [], []
    balance = 0.0; forced = 0; autonomous = 0; scores = []
    for start in blocks:
        feature = RANK.k_history(frame_features, selected, start, previous_offsets)
        value = score(feature, model); scores.append(value)
        balance = min(2.0, balance + 0.5)
        execute = False
        if balance + EPS >= 1.0:
            if balance + EPS >= 2.0:
                execute = True; forced += 1
            elif value >= threshold:
                execute = True; autonomous += 1
        selected.append(start)
        if execute:
            selected.append(start + 2); previous_offsets.append(2); balance -= 1.0
    complete_skeleton = set(blocks)
    selected.extend(start for start in skeleton if start not in complete_skeleton)
    selected = sorted(set(selected))
    online_uniform, uniform_balance = [], 0.0
    for start in blocks:
        online_uniform.append(start); uniform_balance = min(2.0, uniform_balance + 0.5)
        if uniform_balance + EPS >= 1.0:
            online_uniform.append(start + 2); uniform_balance -= 1.0
    online_uniform.extend(start for start in skeleton if start not in complete_skeleton)
    bonus_count = sum(1 for value in selected if value % 4 == 2)
    matched = sorted(skeleton + [blocks[index] + 2 for index in matched_uniform_indices(len(blocks), bonus_count)])
    full_center = sorted(skeleton + [start + 2 for start in blocks])
    return {"K": selected, "online_uniform": sorted(online_uniform),
            "matched_uniform": matched, "full_center": full_center}, {
                "unspent_tokens": balance, "uniform_unspent_tokens": uniform_balance,
                "forced_actions": forced, "autonomous_actions": autonomous,
                "score_min": min(scores) if scores else None, "score_max": max(scores) if scores else None,
            }


def bootstrap(rows, left, right, repetitions=10000):
    grouped = defaultdict(lambda: [0, 0])
    for row in rows:
        grouped[row["source"]][0] += row[left] - row[right]
        grouped[row["source"]][1] += row["ref_len"]
    values = np.asarray(list(grouped.values()), np.float64)
    rng = np.random.default_rng(BOOT_SEED); result = np.empty(repetitions, np.float64)
    for index in range(repetitions):
        chosen = rng.integers(0, len(values), len(values)); sample = values[chosen]
        result[index] = 100.0 * sample[:, 0].sum() / sample[:, 1].sum()
    return [float(np.quantile(result, 0.025)), float(np.quantile(result, 0.975))]


def run():
    config = verify(); prepare()
    if METRICS.exists() or PER_SAMPLE.exists():
        raise FileExistsError("closed-loop outputs already exist")
    threshold = float(config["threshold"]["value"]); model = fit_model()
    with np.load(CONFIRM / "calibration_dataset.npz", allow_pickle=False) as packed:
        wanted = {str(value) for value in packed["sample_names"].tolist()}
    vocab = json.loads(CHRON.DEFAULT_VOCAB.read_text()); blank = vocab.index("<blank>")
    rng = np.random.default_rng(261130); token_bucket = rng.integers(0, 32, size=len(vocab))
    token_sign = rng.choice(np.asarray([-1.0, 1.0], np.float32), size=len(vocab))
    indices, shard_count = CHRON.completed_shard_indices(DENSE); output_rows = []
    with gzip.open(PER_SAMPLE, "wt", encoding="utf-8", compresslevel=6) as handle:
        for shard in indices:
            results, logits, _ = BUILDER.BUILDER.validate_dense_shard(DENSE, shard, shard_count, verify_hashes=False)
            for name in results:
                if name not in wanted:
                    continue
                probabilities = BUILDER.ORACLE.softmax_rows(np.asarray(logits[name]))
                frame_features = RANK.dense_frame_features(probabilities, blank, token_bucket, token_sign)
                policies, trace = schedules(probabilities, frame_features, model, threshold)
                reference = BUILDER.REPLAY.clean_phoenix_2014_trans(results[name]["gls_ref"])
                decoded = {key: BLOCK.decode_counts(probabilities, windows, reference, vocab, blank)
                           for key, windows in policies.items()}
                row = {"sample": name, "source": BUILDER.source_video(name),
                       "ref_len": int(decoded["K"]["ref_len"]), **trace}
                for key in policies:
                    row[key] = int(decoded[key]["error"]); row[f"{key}_windows"] = len(policies[key])
                output_rows.append(row); handle.write(json.dumps(row) + "\n")
            print(json.dumps({"shard": shard, "samples": len(output_rows)}), flush=True)
    keys = ("K", "online_uniform", "matched_uniform", "full_center")
    totals = {key: sum(row[key] for row in output_rows) for key in keys}
    windows = {key: sum(row[f"{key}_windows"] for row in output_rows) for key in keys}
    ref_total = sum(row["ref_len"] for row in output_rows)
    comparisons = {}
    for right in ("online_uniform", "matched_uniform", "full_center"):
        comparisons[f"K_vs_{right}"] = {
            "delta_errors": totals["K"] - totals[right],
            "delta_wer_pp": 100.0 * (totals["K"] - totals[right]) / ref_total,
            "source_cluster_bootstrap_ci95": bootstrap(output_rows, "K", right),
            "delta_windows": windows["K"] - windows[right],
        }
    gates = config["primary_gates"]
    checks = {
        "K_windows_at_most_online_uniform": windows["K"] <= windows["online_uniform"],
        "K_online_uniform_errors": totals["online_uniform"] - totals["K"] >= int(gates["K_error_reduction_vs_online_uniform_at_least"]),
        "K_online_uniform_ci": comparisons["K_vs_online_uniform"]["source_cluster_bootstrap_ci95"][1] < 0,
        "K_matched_uniform_errors": totals["matched_uniform"] - totals["K"] >= int(gates["K_error_reduction_vs_matched_count_uniform_at_least"]),
        "K_matched_uniform_ci": comparisons["K_vs_matched_uniform"]["source_cluster_bootstrap_ci95"][1] < 0,
    }
    passed = all(checks.values())
    metrics = {
        "status": "complete", "created_utc": now(), "gpu_used": False,
        "partition": "calibration", "samples": len(output_rows), "sources": 65,
        "unknown_eos": True, "tail_topup": False, "self_history": True,
        "teacher_forced_history": False, "calibration_threshold_tuning": False,
        "dev_test_heldout_read": False, "threshold": threshold,
        "reference_glosses": ref_total, "errors": totals,
        "wer": {key: 100.0 * value / ref_total for key, value in totals.items()},
        "selected_windows": windows, "comparisons": comparisons,
        "controller": {"forced_actions": sum(row["forced_actions"] for row in output_rows),
                       "autonomous_actions": sum(row["autonomous_actions"] for row in output_rows),
                       "unspent_tokens": sum(row["unspent_tokens"] for row in output_rows)},
        "checks": checks, "gate_passed": passed,
        "decision": ("retain K/history deployable-controller candidate; permit separately frozen final held-out planning"
                     if passed else "stop K/history before held-out"),
    }
    write_atomic(METRICS, metrics)
    manifest = {"status": "complete", "config_sha256": sha(CONFIG), "identity_sha256": sha(IDENTITY),
                "metrics_sha256": sha(METRICS), "per_sample_sha256": sha(PER_SAMPLE),
                "gate_passed": passed, "dev_test_heldout_read": False}
    write_atomic(OUTPUT / "manifest.json", manifest)
    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(); parser.add_argument("--prepare", action="store_true"); parser.add_argument("--run", action="store_true")
    args = parser.parse_args()
    if args.prepare == args.run: parser.error("choose exactly one of --prepare or --run")
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    print(json.dumps(prepare() if args.prepare else run(), indent=2))
