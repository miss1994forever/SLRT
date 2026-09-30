#!/usr/bin/env python3
"""Train-fit equal-budget WER relevance ceiling for the frozen CSC-L8 winner."""
import argparse
import concurrent.futures
import gzip
import hashlib
import json
import os
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

import analyze_phoenix_full_robust_continuation_oracle as full
import analyze_phoenix_p4_csc_l8_identifiability_probe as csc


ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "results/phoenix-2014t_ISLR"
DENSE = BASE / "train_dense_stride1_v1_49faacc3"
TARGETS = BASE / "p4_csc_l8_target_audit_v1_49faacc3/target_scores.npz"
CONFIG = ROOT / "configs/p4_csc_l8_oracle_relevance_v1_preregistered.json"
OUTPUT = BASE / "p4_csc_l8_oracle_relevance_v1_49faacc3"
CHRON, BUILDER, BLOCK = full.CHRON, full.BUILDER, full.BLOCK


def utcnow():
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def fit_names():
    with np.load(csc.INDEX, allow_pickle=False) as packed:
        return [str(value) for value in packed["video_names"].tolist()]


def frozen_choice_map():
    names = fit_names()
    with np.load(TARGETS, allow_pickle=False) as packed:
        sample_indices = np.asarray(packed["sample_indices"], np.int64)
        starts = np.asarray(packed["candidate_starts"], np.int64)
        winners = np.asarray(packed["winners"], np.int8)
    if np.any(starts[:, 1] - starts[:, 0] != 2):
        raise RuntimeError("CSC candidate geometry mismatch")
    mapping = defaultdict(dict)
    for sample, candidates, winner in zip(sample_indices, starts, winners):
        block_start = int(candidates[0]) - 1
        if block_start in mapping[names[int(sample)]]:
            raise RuntimeError("duplicate CSC block target")
        mapping[names[int(sample)]][block_start] = int(winner)
    return dict(mapping)


def prepare():
    if OUTPUT.exists():
        raise FileExistsError(f"non-overwriting oracle relevance identity exists: {OUTPUT}")
    config = json.loads(CONFIG.read_text())
    checks = {
        "fit_split": sha256_file(DENSE / "fit_calibration_split.json") == config["inputs"]["fit_calibration_split_sha256"],
        "dense_protocol": sha256_file(DENSE / "protocol_manifest.json") == config["inputs"]["dense_protocol_sha256"],
        "csc_manifest": sha256_file(BASE / "p4_csc_l8_target_audit_v1_49faacc3/manifest.json") == config["inputs"]["csc_target_manifest_sha256"],
        "csc_targets": sha256_file(TARGETS) == config["inputs"]["csc_target_scores_sha256"],
    }
    if not all(checks.values()):
        raise RuntimeError({"identity_checks": checks})
    names = fit_names()
    split = json.loads((DENSE / "fit_calibration_split.json").read_text())
    expected = sorted(name for name, partition in split["assignments"].items() if partition == "fit")
    if set(names) != set(expected) or len(names) != len(set(names)) or len(names) != int(config["scope"]["samples"]):
        raise RuntimeError("fit sample identity or coverage mismatch")
    choice_map = frozen_choice_map()
    blocks = sum(len(value) for value in choice_map.values())
    if blocks != int(config["sanity_checks"]["valid_csc_blocks_expected"]):
        raise RuntimeError("valid CSC block count mismatch")
    (OUTPUT / "shards").mkdir(parents=True)
    identity = {
        "created_utc": utcnow(), "created_before_csc_policy_or_wer_outcomes": True,
        "config_sha256": sha256_file(CONFIG), "identity_checks": checks,
        "samples": len(names), "valid_csc_blocks": blocks,
        "sample_ids_sha256": hashlib.sha256("\n".join(sorted(names)).encode()).hexdigest(),
    }
    (OUTPUT / "preregistered_identity.json").write_text(json.dumps(identity, indent=2) + "\n")
    print(json.dumps(identity, indent=2))


def shard_paths(index, count):
    stem = f"oracle-{index:05d}-of-{count:05d}"
    return OUTPUT / "shards" / f"{stem}.jsonl.gz", OUTPUT / "shards" / f"{stem}.complete.json"


def valid_shard(index, count):
    data, marker = shard_paths(index, count)
    if not data.is_file() or not marker.is_file():
        return None
    value = json.loads(marker.read_text())
    if value.get("status") != "complete" or value.get("sha256") != sha256_file(data):
        return None
    return value


def process_shard(index):
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    indices, count = CHRON.completed_shard_indices(DENSE)
    if index not in indices:
        raise RuntimeError(f"dense shard is not complete: {index}")
    completed = valid_shard(index, count)
    if completed is not None:
        return completed | {"resumed": True}
    data_path, marker_path = shard_paths(index, count)
    if data_path.exists() or marker_path.exists():
        raise FileExistsError(f"incomplete prior shard requires audit: {data_path}")
    wanted = set(fit_names())
    choices = frozen_choice_map()
    vocab = json.loads(CHRON.DEFAULT_VOCAB.read_text())
    blank = vocab.index("<blank>")
    results, logits, _ = BUILDER.BUILDER.validate_dense_shard(DENSE, index, count, verify_hashes=False)
    temporary = data_path.with_name(f".{data_path.name}.incomplete-{os.getpid()}")
    aggregate = Counter()
    started = time.perf_counter()
    with gzip.open(temporary, "wt", encoding="utf-8", compresslevel=5) as handle:
        for name in results:
            if name not in wanted:
                continue
            probability = BUILDER.ORACLE.softmax_rows(np.asarray(logits[name]))
            reference = BUILDER.REPLAY.clean_phoenix_2014_trans(results[name]["gls_ref"])
            uniform = BLOCK.block_uniform(len(probability))
            selected = []
            local = Counter()
            sample_choices = choices.get(name, {})
            for start in range(0, len(probability), 4):
                selected.append(start)
                if start + 3 >= len(probability):
                    continue
                winner = int(sample_choices.get(start, 2))
                selected.append(start + winner)
                local["blocks"] += 1
                local["valid_csc_blocks"] += int(start in sample_choices)
                local["fallback_blocks"] += int(start not in sample_choices)
                local[f"offset{winner}_choices"] += 1
            if len(selected) != len(uniform):
                raise RuntimeError(f"budget mismatch: {name}")
            uniform_counts = BLOCK.decode_counts(probability, uniform, reference, vocab, blank)
            csc_counts = BLOCK.decode_counts(probability, selected, reference, vocab, blank)
            coverage = CHRON.coverage_metrics(selected, len(probability))
            if coverage["max_gap_including_endpoints"] > 4:
                raise RuntimeError(f"coverage invariant failure: {name}")
            row = {
                "sample_id": name, "source_video_id": BUILDER.source_video(name),
                "uniform_error": int(uniform_counts["error"]), "csc_error": int(csc_counts["error"]),
                "ref_len": int(uniform_counts["ref_len"]), "selected_windows": len(selected),
                "dense_windows": len(probability), **dict(local),
            }
            handle.write(json.dumps(row, separators=(",", ":")) + "\n")
            aggregate.update({"samples": 1, "uniform_error": row["uniform_error"],
                              "csc_error": row["csc_error"], "ref_len": row["ref_len"],
                              "selected_windows": row["selected_windows"],
                              "dense_windows": row["dense_windows"], **dict(local)})
    os.replace(temporary, data_path)
    marker = {
        "status": "complete", "shard": index, "shards": count, "counts": dict(aggregate),
        "sha256": sha256_file(data_path), "bytes": data_path.stat().st_size,
        "seconds": time.perf_counter() - started,
    }
    marker_path.write_text(json.dumps(marker, indent=2) + "\n")
    return marker


def bootstrap(rows, seed, repetitions=10000):
    grouped = defaultdict(lambda: [0, 0])
    for row in rows:
        grouped[row["source_video_id"]][0] += row["csc_error"] - row["uniform_error"]
        grouped[row["source_video_id"]][1] += row["ref_len"]
    sources = sorted(grouped)
    error = np.asarray([grouped[source][0] for source in sources], np.int64)
    ref = np.asarray([grouped[source][1] for source in sources], np.int64)
    rng = np.random.default_rng(seed)
    values = np.empty(repetitions, np.float64)
    for index in range(repetitions):
        chosen = rng.integers(0, len(sources), size=len(sources))
        values[index] = 100.0 * error[chosen].sum() / max(1, ref[chosen].sum())
    return [float(np.quantile(values, .025)), float(np.quantile(values, .975))]


def finalize():
    config = json.loads(CONFIG.read_text())
    indices, count = CHRON.completed_shard_indices(DENSE)
    rows = []
    aggregate = Counter()
    shard_markers = []
    for index in indices:
        marker = valid_shard(index, count)
        if marker is None:
            raise RuntimeError(f"missing valid relevance shard: {index}")
        shard_markers.append(marker)
        data, _ = shard_paths(index, count)
        with gzip.open(data, "rt", encoding="utf-8") as handle:
            for line in handle:
                row = json.loads(line)
                rows.append(row)
                aggregate.update({key: row.get(key, 0) for key in (
                    "uniform_error", "csc_error", "ref_len", "selected_windows", "dense_windows",
                    "blocks", "valid_csc_blocks", "fallback_blocks", "offset1_choices",
                    "offset2_choices", "offset3_choices")})
    names = [row["sample_id"] for row in rows]
    sources = {row["source_video_id"] for row in rows}
    if len(names) != int(config["scope"]["samples"]) or len(set(names)) != len(names):
        raise RuntimeError("sample coverage failure")
    if len(sources) != int(config["scope"]["sources"]):
        raise RuntimeError("source coverage failure")
    if aggregate["ref_len"] != int(config["scope"]["reference_length_expected"]):
        raise RuntimeError("reference length failure")
    if aggregate["uniform_error"] != int(config["sanity_checks"]["uniform_errors_expected"]):
        raise RuntimeError("uniform error parity failure")
    if aggregate["blocks"] != int(config["sanity_checks"]["all_complete_blocks_expected"]):
        raise RuntimeError("complete block count failure")
    if aggregate["valid_csc_blocks"] != int(config["sanity_checks"]["valid_csc_blocks_expected"]):
        raise RuntimeError("valid CSC block coverage failure")
    if any(row["selected_windows"] != len(BLOCK.block_uniform(row["dense_windows"])) for row in rows):
        raise RuntimeError("per-sample equal budget failure")
    uniform_wer = 100.0 * aggregate["uniform_error"] / aggregate["ref_len"]
    csc_wer = 100.0 * aggregate["csc_error"] / aggregate["ref_len"]
    delta_errors = aggregate["csc_error"] - aggregate["uniform_error"]
    delta_wer = csc_wer - uniform_wer
    ci = bootstrap(rows, int(config["primary_metrics"]["bootstrap_seed"]))
    gate = config["primary_gate"]
    checks = {
        "minimum_error_reduction": -delta_errors >= int(gate["minimum_error_reduction"]),
        "bootstrap_upper_lt_zero": ci[1] < float(gate["bootstrap_95pct_delta_wer_upper_strictly_less_than"]),
    }
    metrics = {
        "status": "complete", "created_utc": utcnow(), "train_fit_only": True,
        "reference_read": True, "calibration_dev_test_or_heldout_read": False,
        "scope": {"samples": len(rows), "sources": len(sources), "reference_length": aggregate["ref_len"]},
        "uniform_offset2": {"errors": aggregate["uniform_error"], "wer": uniform_wer},
        "perfect_csc_winner_oracle": {
            "errors": aggregate["csc_error"], "wer": csc_wer, "delta_errors": delta_errors,
            "delta_wer_pp": delta_wer, "source_cluster_bootstrap_95pct_delta_wer_pp": ci,
        },
        "schedule": {
            "complete_blocks": aggregate["blocks"], "valid_csc_blocks": aggregate["valid_csc_blocks"],
            "fallback_blocks": aggregate["fallback_blocks"],
            "winner_counts_including_fallback": {str(i): aggregate[f"offset{i}_choices"] for i in (1, 2, 3)},
            "selected_windows": aggregate["selected_windows"], "dense_windows": aggregate["dense_windows"],
            "window_rate": aggregate["selected_windows"] / aggregate["dense_windows"],
            "per_sample_equal_budget": True,
        },
        "checks": checks, "gate_passed": bool(all(checks.values())),
        "decision": ("CSC target has sufficient train-fit oracle relevance; predictor research may be separately preregistered"
                     if all(checks.values()) else
                     "stop CSC target: even perfect CSC winner lacks the preregistered equal-budget WER relevance"),
        "shards": shard_markers,
    }
    metrics_path = OUTPUT / "metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2) + "\n")
    per_sample_path = OUTPUT / "per_sample.jsonl.gz"
    with gzip.open(per_sample_path, "wt", encoding="utf-8", compresslevel=5) as handle:
        for row in rows:
            handle.write(json.dumps(row, separators=(",", ":")) + "\n")
    manifest = {
        "status": "complete", "created_utc": utcnow(), "config_sha256": sha256_file(CONFIG),
        "identity_sha256": sha256_file(OUTPUT / "preregistered_identity.json"),
        "metrics_sha256": sha256_file(metrics_path), "per_sample_sha256": sha256_file(per_sample_path),
        "gate_passed": metrics["gate_passed"],
    }
    (OUTPUT / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return metrics


def run(workers):
    indices, count = CHRON.completed_shard_indices(DENSE)
    pending = [index for index in indices if valid_shard(index, count) is None]
    if pending:
        with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as pool:
            futures = {pool.submit(process_shard, index): index for index in pending}
            for future in concurrent.futures.as_completed(futures):
                value = future.result()
                print(json.dumps({"completed_shard": value["shard"],
                                  "samples": value["counts"].get("samples", 0),
                                  "seconds": round(value["seconds"], 2)}), flush=True)
    result = finalize()
    print(json.dumps(result, indent=2))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prepare", action="store_true")
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    if args.prepare:
        prepare()
    elif args.run:
        if not (OUTPUT / "preregistered_identity.json").is_file():
            raise RuntimeError("run --prepare before --run")
        if not 1 <= args.workers <= 16:
            raise ValueError("workers must be 1..16")
        run(args.workers)
    else:
        parser.error("choose --prepare or --run")


if __name__ == "__main__":
    main()
