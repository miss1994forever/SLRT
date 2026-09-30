#!/usr/bin/env python3
"""CPU-only audit separating P6 K controller shift from self-history feedback."""
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
TOPK = BASE / "p6_k_history_independent_calibration_v1_49faacc3"
SELF = BASE / "p6_k_history_unknown_eos_self_history_v1_49faacc3"
CONFIG = ROOT / "configs/p6_k_history_controller_state_factorial_audit_v1_preregistered.json"
OUTPUT = BASE / "p6_k_history_controller_state_factorial_audit_v1_49faacc3"
IDENTITY = OUTPUT / "preregistered_identity.json"
METRICS = OUTPUT / "metrics.json"
PER_SAMPLE = OUTPUT / "per_sample.jsonl.gz"
THRESHOLD = -3.8027050495147705
EPS = 1e-9
BOOT_SEED = 261226


def imp(name):
    path = ROOT / f"tools/{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


PRED = imp("analyze_phoenix_p6_local_js_predictor_oof")
SIM = imp("simulate_phoenix_p6_k_history_unknown_eos")
ORACLE, RANK = PRED.ORACLE, PRED.RANK
CHRON, BUILDER, BLOCK = PRED.CHRON, PRED.BUILDER, PRED.BLOCK


def now(): return datetime.now(timezone.utc).isoformat()


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
    config = json.loads(CONFIG.read_text()); expected = config["fixed_assets"]
    actual = {
        "known_topk_manifest_sha256": sha(TOPK / "manifest.json"),
        "known_topk_per_sample_sha256": sha(TOPK / "per_sample.jsonl.gz"),
        "self_history_manifest_sha256": sha(SELF / "manifest.json"),
        "self_history_per_sample_sha256": sha(SELF / "per_sample.jsonl.gz"),
        "fit_p6_dataset_sha256": sha(FIT / "dataset.npz"),
    }
    if actual != expected: raise RuntimeError(f"asset mismatch: {actual}")
    return config


def prepare():
    verify()
    if OUTPUT.exists() and any(OUTPUT.iterdir()):
        if not IDENTITY.is_file(): raise FileExistsError(f"nonempty output without identity: {OUTPUT}")
        identity = json.loads(IDENTITY.read_text())
        if identity["config_sha256"] != sha(CONFIG): raise RuntimeError("configuration changed after prepare")
        return identity
    OUTPUT.mkdir(parents=True, exist_ok=True)
    identity = {"created_utc": now(), "config_sha256": sha(CONFIG),
                "script_sha256_at_prepare": sha(Path(__file__)),
                "created_before_audit_outputs": True, "diagnostic_only": True,
                "dev_test_heldout_read": False}
    IDENTITY.write_text(json.dumps(identity, indent=2) + "\n")
    return identity


def load_rows(path):
    result = {}
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line); result[row["sample"]] = row
    return result


def token_actions(scores):
    balance = 0.0; actions = []; forced = autonomous = 0
    for value in scores:
        balance = min(2.0, balance + 0.5); execute = False
        if balance + EPS >= 1.0:
            if balance + EPS >= 2.0:
                execute = True; forced += 1
            elif value >= THRESHOLD:
                execute = True; autonomous += 1
        actions.append(execute)
        if execute: balance -= 1.0
    return actions, balance, forced, autonomous


def teacher_scores(frame_features, blocks, model):
    selected, offsets, scores = [], [], []
    for start in blocks:
        scores.append(SIM.score(RANK.k_history(frame_features, selected, start, offsets), model))
        selected.extend([start, start + 2]); offsets.append(2)
    return np.asarray(scores, np.float32)


def fixed_schedule_self_scores(frame_features, blocks, actions, model):
    selected, offsets, scores = [], [], []
    for start, execute in zip(blocks, actions):
        scores.append(SIM.score(RANK.k_history(frame_features, selected, start, offsets), model))
        selected.append(start)
        if execute: selected.append(start + 2); offsets.append(2)
    return np.asarray(scores, np.float32)


def windows_from_actions(total, blocks, actions):
    skeleton = ORACLE.base_skeleton(total)
    return sorted(skeleton + [start + 2 for start, execute in zip(blocks, actions) if execute])


def bootstrap(rows, left, right, repetitions=10000):
    grouped = defaultdict(lambda: [0, 0])
    for row in rows:
        grouped[row["source"]][0] += row[left] - row[right]
        grouped[row["source"]][1] += row["ref_len"]
    values = np.asarray(list(grouped.values()), np.float64); rng = np.random.default_rng(BOOT_SEED)
    result = np.empty(repetitions, np.float64)
    for index in range(repetitions):
        chosen = rng.integers(0, len(values), len(values)); sample = values[chosen]
        result[index] = 100.0 * sample[:, 0].sum() / sample[:, 1].sum()
    return [float(np.quantile(result, .025)), float(np.quantile(result, .975))]


def run():
    prepare()
    if METRICS.exists() or PER_SAMPLE.exists(): raise FileExistsError("audit output exists")
    topk_rows = load_rows(TOPK / "per_sample.jsonl.gz"); frozen_self = load_rows(SELF / "per_sample.jsonl.gz")
    wanted = set(topk_rows)
    if wanted != set(frozen_self) or len(wanted) != 718: raise RuntimeError("sample identity mismatch")
    model = SIM.fit_model(); vocab = json.loads(CHRON.DEFAULT_VOCAB.read_text()); blank = vocab.index("<blank>")
    rng = np.random.default_rng(261130); buckets = rng.integers(0, 32, size=len(vocab))
    signs = rng.choice(np.asarray([-1., 1.], np.float32), size=len(vocab))
    all_tf_scores, all_fixed_self_scores = [], []
    action_agree = action_intersection = action_union = action_total = 0
    indices, shard_count = CHRON.completed_shard_indices(DENSE); output_rows = []
    with gzip.open(PER_SAMPLE, "wt", encoding="utf-8", compresslevel=6) as handle:
        for shard in indices:
            results, logits, _ = BUILDER.BUILDER.validate_dense_shard(DENSE, shard, shard_count, verify_hashes=False)
            for name in results:
                if name not in wanted: continue
                probabilities = BUILDER.ORACLE.softmax_rows(np.asarray(logits[name]))
                frame_features = RANK.dense_frame_features(probabilities, blank, buckets, signs)
                blocks = ORACLE.complete_blocks(len(probabilities))
                tf_scores = teacher_scores(frame_features, blocks, model)
                tf_actions, balance, forced, autonomous = token_actions(tf_scores)
                fixed_self_scores = fixed_schedule_self_scores(frame_features, blocks, tf_actions, model)
                tf_windows = windows_from_actions(len(probabilities), blocks, tf_actions)
                self_policies, _ = SIM.schedules(probabilities, frame_features, model, THRESHOLD)
                self_windows = self_policies["K"]; self_window_set = set(self_windows)
                self_actions = [(start + 2) in self_window_set for start in blocks]
                reference = BUILDER.REPLAY.clean_phoenix_2014_trans(results[name]["gls_ref"])
                tf_count = BLOCK.decode_counts(probabilities, tf_windows, reference, vocab, blank)
                self_count = BLOCK.decode_counts(probabilities, self_windows, reference, vocab, blank)
                if int(self_count["error"]) != int(frozen_self[name]["K"]) or len(self_windows) != int(frozen_self[name]["K_windows"]):
                    raise RuntimeError(f"self-history reproduction mismatch: {name}")
                row = {"sample": name, "source": BUILDER.source_video(name), "ref_len": int(tf_count["ref_len"]),
                       "TF_topK": int(topk_rows[name]["K"]), "TF_token": int(tf_count["error"]),
                       "SH_token": int(self_count["error"]), "TF_token_windows": len(tf_windows),
                       "SH_token_windows": len(self_windows), "TF_forced": forced,
                       "TF_autonomous": autonomous, "TF_unspent": balance}
                output_rows.append(row); handle.write(json.dumps(row) + "\n")
                all_tf_scores.append(tf_scores); all_fixed_self_scores.append(fixed_self_scores)
                for left, right in zip(tf_actions, self_actions):
                    action_agree += int(left == right); action_intersection += int(left and right)
                    action_union += int(left or right); action_total += 1
            print(json.dumps({"shard": shard, "samples": len(output_rows)}), flush=True)
    tf_scores = np.concatenate(all_tf_scores); fixed_scores = np.concatenate(all_fixed_self_scores)
    keys = ("TF_topK", "TF_token", "SH_token")
    totals = {key: sum(row[key] for row in output_rows) for key in keys}; ref = sum(row["ref_len"] for row in output_rows)
    windows = {"TF_token": sum(row["TF_token_windows"] for row in output_rows),
               "SH_token": sum(row["SH_token_windows"] for row in output_rows)}
    controller_shift = totals["TF_token"] - totals["TF_topK"]
    state_shift = totals["SH_token"] - totals["TF_token"]
    if controller_shift >= state_shift + 10: diagnosis = "controller/pacing-dominant"
    elif state_shift >= controller_shift + 10: diagnosis = "self-history-feedback-dominant"
    else: diagnosis = "mixed_or_not_separable"
    comparisons = {}
    for left, right in (("TF_token", "TF_topK"), ("SH_token", "TF_token"), ("SH_token", "TF_topK")):
        comparisons[f"{left}_vs_{right}"] = {"delta_errors": totals[left] - totals[right],
            "delta_wer_pp": 100.0 * (totals[left] - totals[right]) / ref,
            "source_cluster_bootstrap_ci95": bootstrap(output_rows, left, right)}
    metrics = {"status": "complete", "created_utc": now(), "gpu_used": False,
        "diagnostic_only": True, "partition": "consumed_calibration", "samples": len(output_rows), "sources": 65,
        "dev_test_heldout_read": False, "errors": totals,
        "wer": {key: 100.0 * value / ref for key, value in totals.items()}, "selected_windows": windows,
        "decomposition": {"controller_shift_errors": controller_shift, "state_feedback_shift_errors": state_shift,
                          "diagnosis": diagnosis, "nonadditive_feedback_warning": True},
        "score_shift_along_fixed_TF_schedule": {"pearson": float(np.corrcoef(tf_scores, fixed_scores)[0, 1]),
            "mean_absolute_change": float(np.mean(np.abs(tf_scores - fixed_scores))),
            "threshold_side_disagreement": float(np.mean((tf_scores >= THRESHOLD) != (fixed_scores >= THRESHOLD)))},
        "TF_vs_SH_actions": {"agreement": action_agree / action_total,
            "jaccard": action_intersection / action_union if action_union else 1.0,
            "intersection": action_intersection, "union": action_union, "blocks": action_total},
        "comparisons": comparisons,
        "controller_counts": {"TF_forced": sum(row["TF_forced"] for row in output_rows),
                              "TF_autonomous": sum(row["TF_autonomous"] for row in output_rows),
                              "TF_unspent_tokens": sum(row["TF_unspent"] for row in output_rows)},
        "decision": "diagnostic only; use mechanism result to preregister new fit-only research, not to tune calibration or read held-out"}
    write_atomic(METRICS, metrics)
    manifest = {"status": "complete", "config_sha256": sha(CONFIG), "identity_sha256": sha(IDENTITY),
                "metrics_sha256": sha(METRICS), "per_sample_sha256": sha(PER_SAMPLE),
                "diagnostic_only": True, "dev_test_heldout_read": False}
    write_atomic(OUTPUT / "manifest.json", manifest)
    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(); parser.add_argument("--prepare", action="store_true"); parser.add_argument("--run", action="store_true")
    args = parser.parse_args()
    if args.prepare == args.run: parser.error("choose exactly one of --prepare or --run")
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    print(json.dumps(prepare() if args.prepare else run(), indent=2))
