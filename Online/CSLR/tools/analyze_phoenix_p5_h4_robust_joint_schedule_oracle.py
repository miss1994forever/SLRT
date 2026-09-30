#!/usr/bin/env python3
"""Complete-fit CPU-only H4 robust joint-schedule oracle."""
import argparse
import concurrent.futures
import gzip
import hashlib
import importlib.util
import itertools
import json
import os
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parents[1]
BASE = ROOT / "results/phoenix-2014t_ISLR"
DENSE = BASE / "train_dense_stride1_v1_49faacc3"
OLD = BASE / "p3_fulltrain_robust_continuation_oracle_fit6378_v1_49faacc3"
OUTPUT = BASE / "p5_h4_robust_joint_schedule_oracle_v1_49faacc3"
CONFIG = ROOT / "configs/p5_h4_robust_joint_schedule_oracle_v1_preregistered.json"
SEED = 261123


def imp(name):
    path = ROOT / f"tools/{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


FULL = imp("analyze_phoenix_full_robust_continuation_oracle")
ROBUST, CHRON, BUILDER, BLOCK = FULL.ROBUST, FULL.CHRON, FULL.BUILDER, FULL.BLOCK


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_config():
    config = json.loads(CONFIG.read_text())
    if not config.get("created_before_oracle_outcomes"):
        raise ValueError("configuration is not preregistered")
    if sha256_file(DENSE / "fit_calibration_split.json") != config["fixed_assets"]["fit_calibration_split_sha256"]:
        raise ValueError("fit/calibration split hash mismatch")
    old = json.loads((OLD / "metrics.json").read_text())
    if old["uniform"]["errors"] != config["fixed_assets"]["uniform_errors_expected"]:
        raise ValueError("old uniform identity mismatch")
    if old["robust_oracle"]["errors"] != config["fixed_assets"]["existing_blockwise_oracle_errors_expected"]:
        raise ValueError("old blockwise oracle identity mismatch")
    return config


def prepare():
    config = load_config()
    if OUTPUT.exists() and any(OUTPUT.iterdir()):
        identity = OUTPUT / "preregistered_identity.json"
        if not identity.is_file():
            raise FileExistsError(f"nonempty output lacks preregistered identity: {OUTPUT}")
        return json.loads(identity.read_text())
    (OUTPUT / "shards").mkdir(parents=True, exist_ok=True)
    fit_ids = sorted(FULL.fit_assignments())
    sources = sorted({BUILDER.source_video(name) for name in fit_ids})
    if len(fit_ids) != config["scope"]["samples"] or len(sources) != config["scope"]["sources"]:
        raise RuntimeError("unexpected fit scope")
    identity = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "config_sha256": sha256_file(CONFIG),
        "script_sha256_at_prepare": sha256_file(Path(__file__)),
        "sample_ids_sha256": hashlib.sha256("\n".join(fit_ids).encode()).hexdigest(),
        "samples": len(fit_ids),
        "sources": len(sources),
        "outcomes_produced": False,
    }
    (OUTPUT / "preregistered_identity.json").write_text(json.dumps(identity, indent=2) + "\n")
    return identity


def shard_paths(index, shard_count):
    stem = f"h4-{index:05d}-of-{shard_count:05d}"
    return OUTPUT / "shards" / f"{stem}.jsonl.gz", OUTPUT / "shards" / f"{stem}.complete.json"


def valid_completed_shard(index, shard_count):
    data, marker = shard_paths(index, shard_count)
    if not data.is_file() or not marker.is_file():
        return None
    value = json.loads(marker.read_text())
    if value.get("status") != "complete" or value.get("sha256") != sha256_file(data):
        return None
    return value


def blockwise_schedule(probabilities, reference, vocab, blank):
    selected = []
    for start in range(0, len(probabilities), 4):
        selected.append(start)
        if start + 3 >= len(probabilities):
            continue
        candidates = [start + 1, start + 2, start + 3]
        futures = [ROBUST.continuation(len(probabilities), start + 4, 2),
                   ROBUST.continuation(len(probabilities), start + 4, 3)]
        choice, _, _ = ROBUST.robust_choice(
            probabilities, selected, candidates, futures, reference, vocab, blank
        )
        selected.append(candidates[choice])
    return selected


def horizon_selected(block_starts, offsets):
    values = []
    for start, offset in zip(block_starts, offsets):
        values.extend([start, start + offset])
    return values


def joint_h4_schedule(probabilities, reference, vocab, blank):
    total = len(probabilities)
    complete = [start for start in range(0, total, 4) if start + 3 < total]
    selected = []
    local = Counter()
    cursor = 0
    while cursor < len(complete):
        blocks = complete[cursor:cursor + 4]
        horizon = len(blocks)
        next_start = blocks[-1] + 4
        futures = [ROBUST.continuation(total, next_start, 2),
                   ROBUST.continuation(total, next_start, 3)]
        center = (2,) * horizon
        center_selected = horizon_selected(blocks, center)
        center_errors = [ROBUST.decode_error(
            probabilities, selected + center_selected + future, reference, vocab, blank
        ) for future in futures]
        best_offsets = center
        best_key = None
        best_rewards = (0, 0)
        for offsets in itertools.product((1, 2, 3), repeat=horizon):
            if offsets == center:
                continue
            candidate = horizon_selected(blocks, offsets)
            errors = [ROBUST.decode_error(
                probabilities, selected + candidate + future, reference, vocab, blank
            ) for future in futures]
            rewards = tuple(int(base - value) for base, value in zip(center_errors, errors))
            robust_value = min(rewards)
            if robust_value <= 0:
                continue
            noncenter = sum(value != 2 for value in offsets)
            key = (robust_value, sum(rewards), -noncenter, tuple(-value for value in offsets))
            if best_key is None or key > best_key:
                best_key = key
                best_offsets = offsets
                best_rewards = rewards
        selected.extend(horizon_selected(blocks, best_offsets))
        local["horizons"] += 1
        local[f"horizon_size_{horizon}"] += 1
        local["noncenter_bonuses"] += sum(value != 2 for value in best_offsets)
        local["changed_horizons"] += best_offsets != center
        local["summed_robust_horizon_reward"] += min(best_rewards)
        cursor += horizon
    if complete:
        next_incomplete = complete[-1] + 4
    else:
        next_incomplete = 0
    for start in range(next_incomplete, total, 4):
        if start + 3 >= total:
            selected.append(start)
    return selected, local


def process_shard(index):
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    indices, shard_count = CHRON.completed_shard_indices(DENSE)
    if index not in indices:
        raise ValueError(f"dense shard {index} not complete")
    completed = valid_completed_shard(index, shard_count)
    if completed is not None:
        return completed | {"resumed": True}
    data_path, marker_path = shard_paths(index, shard_count)
    if data_path.exists() or marker_path.exists():
        raise FileExistsError(f"incomplete output requires audit: {data_path}")
    wanted = FULL.fit_assignments()
    vocab = json.loads(CHRON.DEFAULT_VOCAB.read_text())
    blank = vocab.index("<blank>")
    results, logits, _ = BUILDER.BUILDER.validate_dense_shard(DENSE, index, shard_count, verify_hashes=False)
    temp = data_path.with_name(f".{data_path.name}.incomplete-{os.getpid()}")
    counts = Counter()
    started = time.perf_counter()
    with gzip.open(temp, "wt", encoding="utf-8", compresslevel=5) as handle:
        for name in results:
            if name not in wanted:
                continue
            probabilities = BUILDER.ORACLE.softmax_rows(np.asarray(logits[name]))
            reference = BUILDER.REPLAY.clean_phoenix_2014_trans(results[name]["gls_ref"])
            uniform = BLOCK.block_uniform(len(probabilities))
            blockwise = blockwise_schedule(probabilities, reference, vocab, blank)
            joint, local = joint_h4_schedule(probabilities, reference, vocab, blank)
            if not len(uniform) == len(blockwise) == len(joint):
                raise RuntimeError(f"per-sample budget mismatch: {name}")
            uniform_counts = BLOCK.decode_counts(probabilities, uniform, reference, vocab, blank)
            blockwise_counts = BLOCK.decode_counts(probabilities, blockwise, reference, vocab, blank)
            joint_counts = BLOCK.decode_counts(probabilities, joint, reference, vocab, blank)
            row = {
                "sample_id": name,
                "source_video_id": BUILDER.source_video(name),
                "uniform_error": int(uniform_counts["error"]),
                "blockwise_error": int(blockwise_counts["error"]),
                "joint_h4_error": int(joint_counts["error"]),
                "ref_len": int(uniform_counts["ref_len"]),
                "selected_windows": len(joint),
                "dense_windows": len(probabilities),
                **dict(local),
            }
            handle.write(json.dumps(row, separators=(",", ":")) + "\n")
            counts.update({"samples": 1, "uniform_error": row["uniform_error"],
                           "blockwise_error": row["blockwise_error"],
                           "joint_h4_error": row["joint_h4_error"], "ref_len": row["ref_len"],
                           "selected_windows": row["selected_windows"],
                           "dense_windows": row["dense_windows"], **dict(local)})
    temp.replace(data_path)
    value = {"status": "complete", "shard": index, "shards": shard_count,
             "counts": dict(counts), "sha256": sha256_file(data_path),
             "bytes": data_path.stat().st_size, "seconds": time.perf_counter() - started}
    marker_path.write_text(json.dumps(value, indent=2) + "\n")
    return value


def source_bootstrap(rows, candidate_key, baseline_key, reps=10000):
    grouped = defaultdict(lambda: [0, 0])
    for row in rows:
        grouped[row["source_video_id"]][0] += row[candidate_key] - row[baseline_key]
        grouped[row["source_video_id"]][1] += row["ref_len"]
    sources = sorted(grouped)
    errors = np.asarray([grouped[source][0] for source in sources], np.int64)
    refs = np.asarray([grouped[source][1] for source in sources], np.int64)
    rng = np.random.default_rng(SEED)
    values = np.empty(reps, np.float64)
    for index in range(reps):
        chosen = rng.integers(0, len(sources), size=len(sources))
        values[index] = 100.0 * errors[chosen].sum() / max(1, refs[chosen].sum())
    return [float(np.quantile(values, 0.025)), float(np.quantile(values, 0.975))]


def finalize(config):
    indices, shard_count = CHRON.completed_shard_indices(DENSE)
    rows, shards = [], []
    aggregate = Counter()
    for index in indices:
        marker = valid_completed_shard(index, shard_count)
        if marker is None:
            raise RuntimeError(f"missing valid shard {index}")
        shards.append(marker)
        data, _ = shard_paths(index, shard_count)
        with gzip.open(data, "rt", encoding="utf-8") as handle:
            for line in handle:
                row = json.loads(line)
                rows.append(row)
                aggregate.update({key: row.get(key, 0) for key in (
                    "uniform_error", "blockwise_error", "joint_h4_error", "ref_len",
                    "selected_windows", "dense_windows", "horizons", "horizon_size_1",
                    "horizon_size_2", "horizon_size_3", "horizon_size_4", "changed_horizons",
                    "noncenter_bonuses", "summed_robust_horizon_reward")})
    samples = len(rows)
    sources = len({row["source_video_id"] for row in rows})
    expected = config["fixed_assets"]
    integrity = {
        "samples": samples == config["scope"]["samples"],
        "sources": sources == config["scope"]["sources"],
        "reference_length": aggregate["ref_len"] == config["scope"]["reference_length_expected"],
        "uniform_reproduction": aggregate["uniform_error"] == expected["uniform_errors_expected"],
        "blockwise_reproduction": aggregate["blockwise_error"] == expected["existing_blockwise_oracle_errors_expected"],
        "per_sample_equal_budget": True,
    }
    if not all(integrity.values()):
        raise RuntimeError(f"integrity failure: {integrity}")
    ref = aggregate["ref_len"]
    result = {}
    for name, key in (("uniform", "uniform_error"), ("blockwise_robust", "blockwise_error"),
                      ("joint_h4", "joint_h4_error")):
        result[name] = {"errors": aggregate[key], "wer": 100.0 * aggregate[key] / ref}
    joint_uniform_delta = aggregate["joint_h4_error"] - aggregate["uniform_error"]
    joint_block_delta = aggregate["joint_h4_error"] - aggregate["blockwise_error"]
    ci_uniform = source_bootstrap(rows, "joint_h4_error", "uniform_error")
    ci_block = source_bootstrap(rows, "joint_h4_error", "blockwise_error")
    gates = config["gates"]
    checks = {
        "integrity": all(integrity.values()),
        "joint_total_error_reduction": -joint_uniform_delta >= gates["joint_total_error_reduction_vs_uniform_at_least"],
        "joint_incremental_error_reduction": -joint_block_delta >= gates["joint_incremental_error_reduction_vs_blockwise_at_least"],
        "joint_vs_uniform_bootstrap": ci_uniform[1] < 0,
        "joint_vs_blockwise_bootstrap": ci_block[1] < 0,
    }
    metrics = {
        "status": "complete", "created_utc": datetime.now(timezone.utc).isoformat(),
        "train_fit_only": True, "calibration_dev_test_or_heldout_read": False,
        "scope": {"samples": samples, "sources": sources, "reference_length": ref},
        "policies": result,
        "joint_h4_vs_uniform": {"delta_errors": joint_uniform_delta,
            "delta_wer_pp": 100.0 * joint_uniform_delta / ref,
            "source_cluster_bootstrap_95pct_delta_wer_pp": ci_uniform},
        "joint_h4_vs_blockwise_robust": {"delta_errors": joint_block_delta,
            "delta_wer_pp": 100.0 * joint_block_delta / ref,
            "source_cluster_bootstrap_95pct_delta_wer_pp": ci_block},
        "schedule": {key: aggregate[key] for key in (
            "horizons", "horizon_size_1", "horizon_size_2", "horizon_size_3", "horizon_size_4",
            "changed_horizons", "noncenter_bonuses", "summed_robust_horizon_reward",
            "selected_windows", "dense_windows")},
        "integrity": integrity, "checks": checks, "gate_passed": all(checks.values()),
        "decision": ("unlock preregistered H4 schedule predictor and ordered heatmap-U comparison"
                     if all(checks.values()) else
                     "stop H4 schedule-predictor branch; joint oracle lacks frozen incremental headroom"),
        "shards": shards,
    }
    (OUTPUT / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    with gzip.open(OUTPUT / "per_sample.jsonl.gz", "wt", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, separators=(",", ":")) + "\n")
    manifest = {
        "status": "complete", "config_sha256": sha256_file(CONFIG),
        "identity_sha256": sha256_file(OUTPUT / "preregistered_identity.json"),
        "metrics_sha256": sha256_file(OUTPUT / "metrics.json"),
        "per_sample_sha256": sha256_file(OUTPUT / "per_sample.jsonl.gz"),
        "gate_passed": metrics["gate_passed"],
    }
    (OUTPUT / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return metrics


def run(workers):
    config = load_config()
    identity = prepare()
    if identity["config_sha256"] != sha256_file(CONFIG):
        raise RuntimeError("prepared config identity mismatch")
    indices, shard_count = CHRON.completed_shard_indices(DENSE)
    pending = [index for index in indices if valid_completed_shard(index, shard_count) is None]
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
        parser.error("workers must be between 1 and 16")
    value = prepare() if args.prepare else run(args.workers)
    print(json.dumps(value, indent=2))
