#!/usr/bin/env python3
"""Resumable CPU-only materialization of the passed P5 H4 oracle labels."""
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
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "results/phoenix-2014t_ISLR"
DENSE = BASE / "train_dense_stride1_v1_49faacc3"
P5_OUTPUT = BASE / "p5_h4_robust_joint_schedule_oracle_v1_49faacc3"
OUTPUT = BASE / "p5_h4_schedule_label_dataset_v1_49faacc3"
CONFIG = ROOT / "configs/p5_h4_schedule_label_dataset_v1_preregistered.json"


def imp(name):
    path = ROOT / f"tools/{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


P5 = imp("analyze_phoenix_p5_h4_robust_joint_schedule_oracle")
ROBUST, CHRON, BUILDER, BLOCK = P5.ROBUST, P5.CHRON, P5.BUILDER, P5.BLOCK


def utcnow():
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def source_fold(source):
    value = hashlib.sha256(f"robust-oof-v1\0{source}".encode()).digest()
    return int.from_bytes(value[:8], "big") % 5


def load_config():
    config = json.loads(CONFIG.read_text())
    if not config.get("created_before_label_dataset_outputs"):
        raise RuntimeError("configuration is not preregistered")
    expected = config["fixed_assets"]
    checks = {
        "p5_config_sha256": ROOT / "configs/p5_h4_robust_joint_schedule_oracle_v1_preregistered.json",
        "p5_metrics_sha256": P5_OUTPUT / "metrics.json",
        "p5_manifest_sha256": P5_OUTPUT / "manifest.json",
        "p5_per_sample_sha256": P5_OUTPUT / "per_sample.jsonl.gz",
        "fit_calibration_split_sha256": DENSE / "fit_calibration_split.json",
    }
    for key, path in checks.items():
        if sha256_file(path) != expected[key]:
            raise RuntimeError(f"frozen asset mismatch: {key}")
    metrics = json.loads((P5_OUTPUT / "metrics.json").read_text())
    if not metrics["gate_passed"]:
        raise RuntimeError("P5 oracle gate did not pass")
    return config


def prepare():
    config = load_config()
    identity_path = OUTPUT / "preregistered_identity.json"
    if OUTPUT.exists() and any(OUTPUT.iterdir()):
        if not identity_path.is_file():
            raise FileExistsError(f"nonempty output lacks identity: {OUTPUT}")
        identity = json.loads(identity_path.read_text())
        if identity["config_sha256"] != sha256_file(CONFIG):
            raise RuntimeError("prepared config identity mismatch")
        return identity
    (OUTPUT / "shards").mkdir(parents=True, exist_ok=True)
    fit_ids = sorted(P5.FULL.fit_assignments())
    sources = sorted({BUILDER.source_video(name) for name in fit_ids})
    if len(fit_ids) != config["scope"]["samples"] or len(sources) != config["scope"]["sources"]:
        raise RuntimeError("fit scope mismatch")
    identity = {
        "created_utc": utcnow(),
        "config_sha256": sha256_file(CONFIG),
        "script_sha256_at_prepare": sha256_file(Path(__file__)),
        "sample_ids_sha256": hashlib.sha256("\n".join(fit_ids).encode()).hexdigest(),
        "samples": len(fit_ids), "sources": len(sources), "outcomes_produced": False,
    }
    identity_path.write_text(json.dumps(identity, indent=2) + "\n")
    return identity


def shard_paths(index, shard_count):
    stem = f"labels-{index:05d}-of-{shard_count:05d}"
    return OUTPUT / "shards" / f"{stem}.jsonl.gz", OUTPUT / "shards" / f"{stem}.complete.json"


def valid_completed_shard(index, shard_count):
    data, marker = shard_paths(index, shard_count)
    if not data.is_file() or not marker.is_file():
        return None
    value = json.loads(marker.read_text())
    if value.get("status") != "complete" or value.get("sha256") != sha256_file(data):
        return None
    return value


def horizon_labels(probabilities, reference, vocab, blank):
    total = len(probabilities)
    complete = [start for start in range(0, total, 4) if start + 3 < total]
    selected = []
    records = []
    cursor = 0
    while cursor < len(complete):
        blocks = complete[cursor:cursor + 4]
        horizon = len(blocks)
        next_start = blocks[-1] + 4
        futures = [ROBUST.continuation(total, next_start, 2),
                   ROBUST.continuation(total, next_start, 3)]
        center = (2,) * horizon
        center_selected = P5.horizon_selected(blocks, center)
        center_errors = [ROBUST.decode_error(
            probabilities, selected + center_selected + future, reference, vocab, blank
        ) for future in futures]
        schedules = list(itertools.product((1, 2, 3), repeat=horizon))
        rewards = []
        best_offsets = center
        best_key = None
        best_rewards = (0, 0)
        for offsets in schedules:
            if offsets == center:
                pair = (0, 0)
            else:
                candidate = P5.horizon_selected(blocks, offsets)
                errors = [ROBUST.decode_error(
                    probabilities, selected + candidate + future, reference, vocab, blank
                ) for future in futures]
                pair = tuple(int(base - value) for base, value in zip(center_errors, errors))
                robust_value = min(pair)
                if robust_value > 0:
                    noncenter = sum(value != 2 for value in offsets)
                    key = (robust_value, sum(pair), -noncenter, tuple(-value for value in offsets))
                    if best_key is None or key > best_key:
                        best_key, best_offsets, best_rewards = key, offsets, pair
            rewards.append(pair)
        chosen_index = schedules.index(best_offsets)
        records.append({
            "horizon_index": len(records), "block_starts": blocks, "horizon_size": horizon,
            "candidate_offsets": [list(value) for value in schedules],
            "reward_center_future": [value[0] for value in rewards],
            "reward_late_future": [value[1] for value in rewards],
            "robust_values": [min(value) for value in rewards],
            "chosen_index": chosen_index, "chosen_offsets": list(best_offsets),
            "chosen_rewards": list(best_rewards), "center_errors": center_errors,
            "past_selected_windows": len(selected),
        })
        selected.extend(P5.horizon_selected(blocks, best_offsets))
        cursor += horizon
    next_incomplete = complete[-1] + 4 if complete else 0
    for start in range(next_incomplete, total, 4):
        if start + 3 >= total:
            selected.append(start)
    return selected, records


def process_shard(index):
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    indices, shard_count = CHRON.completed_shard_indices(DENSE)
    if index not in indices:
        raise ValueError(f"dense shard {index} not complete")
    complete = valid_completed_shard(index, shard_count)
    if complete is not None:
        return complete | {"resumed": True}
    data_path, marker_path = shard_paths(index, shard_count)
    if data_path.exists() or marker_path.exists():
        raise FileExistsError(f"incomplete output requires audit: {data_path}")
    wanted = P5.FULL.fit_assignments()
    vocab = json.loads(CHRON.DEFAULT_VOCAB.read_text())
    blank = vocab.index("<blank>")
    results, logits, _ = BUILDER.BUILDER.validate_dense_shard(
        DENSE, index, shard_count, verify_hashes=False)
    temporary = data_path.with_name(f".{data_path.name}.incomplete-{os.getpid()}")
    counts = Counter()
    started = time.perf_counter()
    with gzip.open(temporary, "wt", encoding="utf-8", compresslevel=5) as handle:
        for name in results:
            if name not in wanted:
                continue
            probabilities = BUILDER.ORACLE.softmax_rows(np.asarray(logits[name]))
            reference = BUILDER.REPLAY.clean_phoenix_2014_trans(results[name]["gls_ref"])
            selected, horizons = horizon_labels(probabilities, reference, vocab, blank)
            uniform = BLOCK.block_uniform(len(probabilities))
            if len(selected) != len(uniform):
                raise RuntimeError(f"budget mismatch: {name}")
            uniform_counts = BLOCK.decode_counts(
                probabilities, uniform, reference, vocab, blank)
            uniform_error = int(uniform_counts["error"])
            joint_error = int(BLOCK.decode_counts(
                probabilities, selected, reference, vocab, blank)["error"])
            source = BUILDER.source_video(name)
            row = {
                "sample_id": name, "source_video_id": source, "fold": source_fold(source),
                "dense_windows": len(probabilities), "selected_windows": len(selected),
                "ref_len": int(uniform_counts["ref_len"]), "uniform_error": uniform_error,
                "joint_h4_error": joint_error, "horizons": horizons,
            }
            handle.write(json.dumps(row, separators=(",", ":")) + "\n")
            counts.update({
                "samples": 1, "ref_len": int(uniform_counts["ref_len"]), "uniform_error": uniform_error,
                "joint_h4_error": joint_error, "dense_windows": len(probabilities),
                "selected_windows": len(selected), "horizons": len(horizons),
            })
            for horizon in horizons:
                size = horizon["horizon_size"]
                counts[f"horizon_size_{size}"] += 1
                changed = any(value != 2 for value in horizon["chosen_offsets"])
                counts["changed_horizons"] += changed
                counts["noncenter_bonuses"] += sum(value != 2 for value in horizon["chosen_offsets"])
                counts["summed_robust_horizon_reward"] += min(horizon["chosen_rewards"])
                counts["candidate_schedules"] += 3 ** size
    os.replace(temporary, data_path)
    marker = {
        "status": "complete", "shard": index, "shards": shard_count,
        "counts": dict(counts), "sha256": sha256_file(data_path),
        "bytes": data_path.stat().st_size, "seconds": time.perf_counter() - started,
    }
    marker_path.write_text(json.dumps(marker, indent=2) + "\n")
    return marker


def frozen_p5_rows():
    rows = {}
    with gzip.open(P5_OUTPUT / "per_sample.jsonl.gz", "rt", encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            rows[row["sample_id"]] = row
    return rows


def repair_ref_lengths():
    """Repair the pre-finalization character-count bug from frozen P5 rows."""
    load_config()
    p5_rows = frozen_p5_rows()
    indices, shard_count = CHRON.completed_shard_indices(DENSE)
    repaired_samples = 0
    for index in indices:
        marker = valid_completed_shard(index, shard_count)
        if marker is None:
            raise RuntimeError(f"missing valid shard {index}")
        data_path, marker_path = shard_paths(index, shard_count)
        temporary = data_path.with_name(f".{data_path.name}.repair-{os.getpid()}")
        rows = []
        with gzip.open(data_path, "rt", encoding="utf-8") as handle:
            for line in handle:
                row = json.loads(line)
                expected = int(p5_rows[row["sample_id"]]["ref_len"])
                if int(row["ref_len"]) != expected:
                    repaired_samples += 1
                    row["ref_len"] = expected
                rows.append(row)
        with gzip.open(temporary, "wt", encoding="utf-8", compresslevel=5) as handle:
            for row in rows:
                handle.write(json.dumps(row, separators=(",", ":")) + "\n")
        os.replace(temporary, data_path)
        marker["counts"]["ref_len"] = sum(int(row["ref_len"]) for row in rows)
        marker["sha256"] = sha256_file(data_path)
        marker["bytes"] = data_path.stat().st_size
        marker["ref_len_repair"] = {
            "method": "frozen P5 per-sample ref_len",
            "p5_per_sample_sha256": sha256_file(P5_OUTPUT / "per_sample.jsonl.gz"),
        }
        marker_path.write_text(json.dumps(marker, indent=2) + "\n")
    audit = {
        "status": "complete", "created_utc": utcnow(),
        "reason": "initial builder serialized len(cleaned reference string) instead of decoder token ref_len",
        "repaired_samples": repaired_samples, "shards": len(indices),
        "source": "frozen P5 per_sample.jsonl.gz",
        "source_sha256": sha256_file(P5_OUTPUT / "per_sample.jsonl.gz"),
        "labels_rewards_schedules_or_errors_changed": False,
    }
    (OUTPUT / "ref_len_repair_audit.json").write_text(json.dumps(audit, indent=2) + "\n")
    return audit


def finalize(config):
    indices, shard_count = CHRON.completed_shard_indices(DENSE)
    p5_rows = frozen_p5_rows()
    aggregate = Counter()
    sample_count = 0
    sources, folds = set(), set()
    shard_records = []
    per_sample_parity = True
    for index in indices:
        marker = valid_completed_shard(index, shard_count)
        if marker is None:
            raise RuntimeError(f"missing valid shard {index}")
        shard_records.append(marker)
        data_path, _ = shard_paths(index, shard_count)
        with gzip.open(data_path, "rt", encoding="utf-8") as handle:
            for line in handle:
                row = json.loads(line)
                sample_count += 1
                sources.add(row["source_video_id"]); folds.add(row["fold"])
                local = Counter({
                    "samples": 1, "ref_len": row["ref_len"],
                    "uniform_error": row["uniform_error"], "joint_h4_error": row["joint_h4_error"],
                    "dense_windows": row["dense_windows"], "selected_windows": row["selected_windows"],
                    "horizons": len(row["horizons"]),
                })
                for horizon in row["horizons"]:
                    size = horizon["horizon_size"]
                    local[f"horizon_size_{size}"] += 1
                    local["candidate_schedules"] += 3 ** size
                    local["changed_horizons"] += any(value != 2 for value in horizon["chosen_offsets"])
                    local["noncenter_bonuses"] += sum(value != 2 for value in horizon["chosen_offsets"])
                    local["summed_robust_horizon_reward"] += min(horizon["chosen_rewards"])
                aggregate.update(local)
                frozen = p5_rows.get(row["sample_id"])
                parity_keys = ("uniform_error", "joint_h4_error", "ref_len", "dense_windows",
                               "selected_windows", "horizons", "horizon_size_1", "horizon_size_2",
                               "horizon_size_3", "horizon_size_4", "changed_horizons",
                               "noncenter_bonuses", "summed_robust_horizon_reward")
                if frozen is None or any(int(frozen.get(key, 0)) != int(local.get(key, 0))
                                         for key in parity_keys):
                    per_sample_parity = False
    expected_schedule = json.loads((P5_OUTPUT / "metrics.json").read_text())["schedule"]
    checks = {
        "samples": sample_count == config["scope"]["samples"],
        "sources": len(sources) == config["scope"]["sources"],
        "folds": folds == set(range(5)),
        "horizons": aggregate["horizons"] == config["scope"]["horizons_expected"],
        "horizon_size_distribution": all(
            aggregate[f"horizon_size_{size}"] == expected_schedule[f"horizon_size_{size}"]
            for size in range(1, 5)),
        "changed_horizons": aggregate["changed_horizons"] == expected_schedule["changed_horizons"],
        "noncenter_bonuses": aggregate["noncenter_bonuses"] == expected_schedule["noncenter_bonuses"],
        "summed_robust_reward": aggregate["summed_robust_horizon_reward"] == expected_schedule["summed_robust_horizon_reward"],
        "uniform_errors": aggregate["uniform_error"] == config["fixed_assets"]["uniform_errors_expected"],
        "joint_h4_errors": aggregate["joint_h4_error"] == config["fixed_assets"]["joint_h4_errors_expected"],
        "per_sample_p5_parity": per_sample_parity,
    }
    if not all(checks.values()):
        raise RuntimeError(f"label dataset integrity failure: {checks}")
    metrics = {
        "status": "complete", "created_utc": utcnow(), "gpu_used": False,
        "train_fit_only": True, "calibration_dev_test_heldout_or_closed_loop_read": False,
        "scope": {"samples": sample_count, "sources": len(sources), "folds": len(folds)},
        "counts": dict(aggregate), "integrity": checks, "gate_passed": True,
        "decision": "freeze a separate source-disjoint H4 schedule-ranker preregistration",
    }
    (OUTPUT / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    manifest = {
        "status": "complete", "config_sha256": sha256_file(CONFIG),
        "identity_sha256": sha256_file(OUTPUT / "preregistered_identity.json"),
        "metrics_sha256": sha256_file(OUTPUT / "metrics.json"),
        "shards": shard_records, "gate_passed": True,
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
    parser.add_argument("--repair-ref-len", action="store_true")
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    if sum((args.prepare, args.run, args.repair_ref_len)) != 1:
        parser.error("choose exactly one of --prepare, --run, or --repair-ref-len")
    if not 1 <= args.workers <= 16:
        parser.error("workers must be between 1 and 16")
    result = (prepare() if args.prepare else
              repair_ref_lengths() if args.repair_ref_len else run(args.workers))
    print(json.dumps(result, indent=2))
