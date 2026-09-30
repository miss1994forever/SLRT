#!/usr/bin/env python3
"""CPU-only complete-fit relevance oracle for local posterior JS information gain."""
import argparse
import concurrent.futures
import gzip
import hashlib
import importlib.util
import json
import os
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "results/phoenix-2014t_ISLR"
DENSE = BASE / "train_dense_stride1_v1_49faacc3"
OUTPUT = BASE / "p6_local_js_information_gain_oracle_v1_49faacc3"
CONFIG = ROOT / "configs/p6_local_js_information_gain_oracle_v1_preregistered.json"
BOOT_SEED = 261210


def imp(name):
    path = ROOT / f"tools/{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


P5 = imp("analyze_phoenix_p5_h4_robust_joint_schedule_oracle")
CHRON, BUILDER, BLOCK = P5.CHRON, P5.BUILDER, P5.BLOCK


def utcnow():
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_config():
    config = json.loads(CONFIG.read_text())
    if not config.get("created_before_oracle_outcomes"):
        raise RuntimeError("configuration is not preregistered")
    expected = config["fixed_assets"]
    if sha256_file(DENSE / "protocol_manifest.json") != expected["dense_protocol_sha256"]:
        raise RuntimeError("dense protocol mismatch")
    if sha256_file(DENSE / "fit_calibration_split.json") != expected["fit_calibration_split_sha256"]:
        raise RuntimeError("fit split mismatch")
    return config


def prepare():
    config = load_config()
    identity_path = OUTPUT / "preregistered_identity.json"
    if OUTPUT.exists() and any(OUTPUT.iterdir()):
        if not identity_path.is_file():
            raise FileExistsError(f"nonempty output lacks identity: {OUTPUT}")
        identity = json.loads(identity_path.read_text())
        if identity["config_sha256"] != sha256_file(CONFIG):
            raise RuntimeError("config changed after prepare")
        return identity
    (OUTPUT / "shards").mkdir(parents=True, exist_ok=True)
    fit_ids = sorted(P5.FULL.fit_assignments())
    sources = sorted({BUILDER.source_video(name) for name in fit_ids})
    if len(fit_ids) != config["scope"]["samples"] or len(sources) != config["scope"]["sources"]:
        raise RuntimeError("fit scope mismatch")
    identity = {"created_utc": utcnow(), "config_sha256": sha256_file(CONFIG),
                "script_sha256_at_prepare": sha256_file(Path(__file__)),
                "samples": len(fit_ids), "sources": len(sources), "outcomes_produced": False}
    identity_path.write_text(json.dumps(identity, indent=2) + "\n")
    return identity


def shard_paths(index, shard_count):
    stem = f"js-{index:05d}-of-{shard_count:05d}"
    return OUTPUT / "shards" / f"{stem}.jsonl.gz", OUTPUT / "shards" / f"{stem}.complete.json"


def valid_shard(index, shard_count):
    data, marker = shard_paths(index, shard_count)
    if not data.is_file() or not marker.is_file():
        return None
    value = json.loads(marker.read_text())
    if value.get("status") != "complete" or value.get("sha256") != sha256_file(data):
        return None
    return value


def js_score(left, right):
    middle = 0.5 * (left + right)
    return float(0.5 * np.sum(left * (np.log(np.maximum(left, 1e-12)) - np.log(np.maximum(middle, 1e-12)))) +
                 0.5 * np.sum(right * (np.log(np.maximum(right, 1e-12)) - np.log(np.maximum(middle, 1e-12)))))


def complete_blocks(total):
    return [start for start in range(0, total, 4) if start + 3 < total]


def base_skeleton(total):
    return list(range(0, total, 4))


def uniform_bonus_indices(blocks, count):
    if count == 0:
        return []
    size = len(blocks)
    chosen = [int(np.floor((index + 0.5) * size / count)) for index in range(count)]
    if len(set(chosen)) != count or min(chosen) < 0 or max(chosen) >= size:
        raise RuntimeError(f"uniform placement failure: B={size}, K={count}, chosen={chosen}")
    return chosen


def schedules(probabilities):
    total = len(probabilities)
    blocks = complete_blocks(total)
    count = (len(blocks) + 1) // 2
    skeleton = base_skeleton(total)
    full = sorted(skeleton + [start + 2 for start in blocks])
    uniform_indices = uniform_bonus_indices(blocks, count)
    uniform = sorted(skeleton + [blocks[index] + 2 for index in uniform_indices])
    scores = np.asarray([js_score(probabilities[start], probabilities[start + 2]) for start in blocks], np.float64)
    ranking = sorted(range(len(blocks)), key=lambda index: (-scores[index], index))
    chosen = ranking[:count]
    js = sorted(skeleton + [blocks[index] + 2 for index in chosen])
    return full, uniform, js, scores, chosen


def process_shard(index):
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    indices, shard_count = CHRON.completed_shard_indices(DENSE)
    if index not in indices:
        raise RuntimeError(f"invalid dense shard {index}")
    completed = valid_shard(index, shard_count)
    if completed is not None:
        return completed | {"resumed": True}
    data_path, marker_path = shard_paths(index, shard_count)
    if data_path.exists() or marker_path.exists():
        raise FileExistsError(f"incomplete output requires audit: {data_path}")
    wanted = P5.FULL.fit_assignments()
    vocab = json.loads(CHRON.DEFAULT_VOCAB.read_text())
    blank = vocab.index("<blank>")
    results, logits, _ = BUILDER.BUILDER.validate_dense_shard(DENSE, index, shard_count, verify_hashes=False)
    temporary = data_path.with_name(f".{data_path.name}.incomplete-{os.getpid()}")
    counts = Counter(); started = time.perf_counter()
    with gzip.open(temporary, "wt", encoding="utf-8", compresslevel=5) as handle:
        for name in results:
            if name not in wanted:
                continue
            probabilities = BUILDER.ORACLE.softmax_rows(np.asarray(logits[name]))
            reference = BUILDER.REPLAY.clean_phoenix_2014_trans(results[name]["gls_ref"])
            full, uniform, js, scores, chosen = schedules(probabilities)
            if len(uniform) != len(js):
                raise RuntimeError(f"budget mismatch: {name}")
            full_counts = BLOCK.decode_counts(probabilities, full, reference, vocab, blank)
            uniform_counts = BLOCK.decode_counts(probabilities, uniform, reference, vocab, blank)
            js_counts = BLOCK.decode_counts(probabilities, js, reference, vocab, blank)
            row = {"sample_id": name, "source_video_id": BUILDER.source_video(name),
                   "ref_len": int(full_counts["ref_len"]), "dense_windows": len(probabilities),
                   "complete_blocks": len(scores), "retained_bonuses": len(chosen),
                   "selected_windows": len(js), "full_center_error": int(full_counts["error"]),
                   "uniform_sparse_error": int(uniform_counts["error"]),
                   "js_oracle_error": int(js_counts["error"]),
                   "js_mean": float(scores.mean()) if len(scores) else 0.0,
                   "js_max": float(scores.max()) if len(scores) else 0.0}
            handle.write(json.dumps(row, separators=(",", ":")) + "\n")
            counts.update({"samples": 1, "ref_len": row["ref_len"],
                           "dense_windows": row["dense_windows"], "complete_blocks": row["complete_blocks"],
                           "retained_bonuses": row["retained_bonuses"], "selected_windows": row["selected_windows"],
                           "full_center_error": row["full_center_error"],
                           "uniform_sparse_error": row["uniform_sparse_error"],
                           "js_oracle_error": row["js_oracle_error"]})
    os.replace(temporary, data_path)
    marker = {"status": "complete", "shard": index, "shards": shard_count,
              "counts": dict(counts), "sha256": sha256_file(data_path),
              "bytes": data_path.stat().st_size, "seconds": time.perf_counter() - started}
    marker_path.write_text(json.dumps(marker, indent=2) + "\n")
    return marker


def bootstrap(rows, candidate, baseline, reps=10000):
    grouped = defaultdict(lambda: [0, 0])
    for row in rows:
        grouped[row["source_video_id"]][0] += row[candidate] - row[baseline]
        grouped[row["source_video_id"]][1] += row["ref_len"]
    keys = sorted(grouped)
    errors = np.asarray([grouped[key][0] for key in keys], np.int64)
    refs = np.asarray([grouped[key][1] for key in keys], np.int64)
    rng = np.random.default_rng(BOOT_SEED); values = np.empty(reps, np.float64)
    for index in range(reps):
        chosen = rng.integers(0, len(keys), size=len(keys))
        values[index] = 100.0 * errors[chosen].sum() / refs[chosen].sum()
    return [float(np.quantile(values, 0.025)), float(np.quantile(values, 0.975))]


def finalize(config):
    indices, shard_count = CHRON.completed_shard_indices(DENSE)
    aggregate = Counter(); rows = []; shards = []
    for index in indices:
        marker = valid_shard(index, shard_count)
        if marker is None:
            raise RuntimeError(f"missing valid shard {index}")
        shards.append(marker); data_path, _ = shard_paths(index, shard_count)
        with gzip.open(data_path, "rt", encoding="utf-8") as handle:
            for line in handle:
                row = json.loads(line); rows.append(row)
                aggregate.update({key: row[key] for key in (
                    "ref_len", "dense_windows", "complete_blocks", "retained_bonuses", "selected_windows",
                    "full_center_error", "uniform_sparse_error", "js_oracle_error")})
    sources = len({row["source_video_id"] for row in rows})
    integrity = {"samples": len(rows) == config["scope"]["samples"],
                 "sources": sources == config["scope"]["sources"],
                 "reference_length": aggregate["ref_len"] == config["scope"]["reference_length_expected"],
                 "full_center_reproduction": aggregate["full_center_error"] == config["fixed_assets"]["full_center_errors_expected"],
                 "per_sample_equal_budget": all(row["selected_windows"] == row["retained_bonuses"] +
                                                ((row["dense_windows"] + 3) // 4) for row in rows)}
    if not all(integrity.values()):
        raise RuntimeError(f"integrity failure: {integrity}")
    delta = aggregate["js_oracle_error"] - aggregate["uniform_sparse_error"]
    ci = bootstrap(rows, "js_oracle_error", "uniform_sparse_error")
    gates = config["gates"]
    checks = {"integrity": True,
              "error_reduction": -delta >= gates["js_error_reduction_vs_uniform_sparse_at_least"],
              "bootstrap": ci[1] < 0}
    ref = aggregate["ref_len"]
    metrics = {"status": "complete", "created_utc": utcnow(), "gpu_used": False,
               "train_fit_only": True, "calibration_dev_test_or_heldout_read": False,
               "scope": {"samples": len(rows), "sources": sources, "reference_length": ref},
               "budget": {"dense_windows": aggregate["dense_windows"],
                          "complete_blocks": aggregate["complete_blocks"],
                          "retained_bonuses": aggregate["retained_bonuses"],
                          "selected_windows": aggregate["selected_windows"]},
               "policies": {
                   "full_center": {"errors": aggregate["full_center_error"], "wer": 100.0 * aggregate["full_center_error"] / ref},
                   "uniform_sparse": {"errors": aggregate["uniform_sparse_error"], "wer": 100.0 * aggregate["uniform_sparse_error"] / ref},
                   "js_oracle": {"errors": aggregate["js_oracle_error"], "wer": 100.0 * aggregate["js_oracle_error"] / ref}},
               "js_vs_uniform_sparse": {"delta_errors": delta, "delta_wer_pp": 100.0 * delta / ref,
                                        "source_cluster_bootstrap_95pct_delta_wer_pp": ci},
               "integrity": integrity, "checks": checks, "gate_passed": all(checks.values()),
               "decision": ("unlock preregistered causal local-JS predictor" if all(checks.values()) else
                            "stop local-JS target before predictor, closed loop, or held-out")}
    (OUTPUT / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    with gzip.open(OUTPUT / "per_sample.jsonl.gz", "wt", encoding="utf-8") as handle:
        for row in rows: handle.write(json.dumps(row, separators=(",", ":")) + "\n")
    manifest = {"status": "complete", "config_sha256": sha256_file(CONFIG),
                "identity_sha256": sha256_file(OUTPUT / "preregistered_identity.json"),
                "metrics_sha256": sha256_file(OUTPUT / "metrics.json"),
                "per_sample_sha256": sha256_file(OUTPUT / "per_sample.jsonl.gz"),
                "gate_passed": metrics["gate_passed"], "shards": shards}
    (OUTPUT / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return metrics


def run(workers):
    config = load_config(); prepare()
    indices, shard_count = CHRON.completed_shard_indices(DENSE)
    pending = [index for index in indices if valid_shard(index, shard_count) is None]
    if pending:
        with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as pool:
            futures = {pool.submit(process_shard, index): index for index in pending}
            for future in concurrent.futures.as_completed(futures):
                value = future.result()
                print(json.dumps({"completed_shard": value["shard"],
                                  "samples": value["counts"].get("samples", 0),
                                  "seconds": round(value["seconds"], 1)}), flush=True)
    return finalize(config)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--prepare", action="store_true")
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    if args.prepare == args.run:
        parser.error("choose exactly one of --prepare or --run")
    if not 1 <= args.workers <= 16:
        parser.error("workers must be 1..16")
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    result = prepare() if args.prepare else run(args.workers)
    print(json.dumps(result, indent=2))
