#!/usr/bin/env python3
"""Resumable full-fit robust-continuation oracle on complete train replay."""
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
OUTPUT = BASE / "p3_fulltrain_robust_continuation_oracle_fit6378_v1_49faacc3"
EXPECTED_CHECKPOINT = "b3390f0dc4b6a826b53c88d3309b1d98fb75e5b58a5ec3f5779628cf3d51767b"
EXPECTED_SPLIT = "38763cd9c1ccde2e13bbd9f344b92cdc1b7873afe39d38316ddc877986fb6784"
SEED = 261027


def imp(name):
    path = ROOT / f"tools/{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


ROBUST = imp("analyze_phoenix_robust_continuation_oracle")
CHRON, BUILDER, BLOCK = ROBUST.CHRON, ROBUST.BUILDER, ROBUST.BLOCK


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def frozen_config():
    protocol = json.loads((DENSE / "protocol_manifest.json").read_text())
    split_path = DENSE / "fit_calibration_split.json"
    if sha256_file(split_path) != EXPECTED_SPLIT:
        raise ValueError("fit/calibration split hash mismatch")
    checkpoint = next(value for path, value in protocol["inputs"].items()
                      if path.endswith("/ckpts/best.ckpt"))
    if checkpoint["sha256"] != EXPECTED_CHECKPOINT:
        raise ValueError("checkpoint hash mismatch")
    split = json.loads(split_path.read_text())
    fit_ids = sorted(name for name, partition in split["assignments"].items() if partition == "fit")
    sources = sorted({BUILDER.source_video(name) for name in fit_ids})
    if len(fit_ids) != 6378 or len(sources) != 578:
        raise ValueError("unexpected frozen fit scope")
    return {
        "experiment": "complete-train fit robust-continuation structured oracle",
        "created_before_oracle_outcomes": True,
        "scope": {"samples": len(fit_ids), "sources": len(sources),
                  "sample_ids_sha256": hashlib.sha256("\n".join(fit_ids).encode()).hexdigest()},
        "dense_protocol_sha256": sha256_file(DENSE / "protocol_manifest.json"),
        "fit_calibration_split_sha256": EXPECTED_SPLIT,
        "checkpoint_sha256": EXPECTED_CHECKPOINT,
        "protocol": {"hard_skeleton_offset": 0, "one_bonus_offsets": [1, 2, 3],
                     "uniform_bonus_offset": 2, "fallback": "offset2 center",
                     "target_rate": "approximately 50%", "max_gap_including_endpoints": 4,
                     "decoder_span": 15, "unknown_EOS": True},
        "oracle": {
            "future_continuations": ["fixed offset2 center", "fixed offset3 late"],
            "eligible_side": "strict terminal edit-error improvement over center under both continuations",
            "selection": "maximize minimum improvement; ties left then right; otherwise center",
            "future_reference_EOS": "oracle label only, never deployment input",
        },
        "go": "delta WER versus fixed-center <= -0.5 pp and source-cluster bootstrap upper < 0",
        "forbidden": ["GPU", "calibration outcomes", "dev", "test", "predictor training", "git commit"],
        "classification": "confirmatory full-fit target-headroom audit after frozen partial32 protocol",
    }


def preregister():
    config = frozen_config()
    path = OUTPUT / "resolved_config_preregistered.json"
    if OUTPUT.exists():
        if not path.is_file() or json.loads(path.read_text()) != config:
            raise FileExistsError(f"non-resumable or mismatched output: {OUTPUT}")
    else:
        (OUTPUT / "shards").mkdir(parents=True)
        path.write_text(json.dumps(config, indent=2) + "\n")
    return config


def fit_assignments():
    split = json.loads((DENSE / "fit_calibration_split.json").read_text())
    return {name for name, partition in split["assignments"].items() if partition == "fit"}


def shard_paths(index, shard_count):
    stem = f"oracle-{index:05d}-of-{shard_count:05d}"
    return OUTPUT / "shards" / f"{stem}.jsonl.gz", OUTPUT / "shards" / f"{stem}.complete.json"


def valid_completed_shard(index, shard_count):
    data, marker = shard_paths(index, shard_count)
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
    indices, shard_count = CHRON.completed_shard_indices(DENSE)
    if index not in indices:
        raise ValueError(f"dense shard {index} not complete")
    completed = valid_completed_shard(index, shard_count)
    if completed is not None:
        return completed | {"resumed": True}
    data_path, marker_path = shard_paths(index, shard_count)
    if data_path.exists() or marker_path.exists():
        raise FileExistsError(f"incomplete prior output requires audit: {data_path}")
    wanted = fit_assignments()
    vocab = json.loads(CHRON.DEFAULT_VOCAB.read_text())
    blank = vocab.index("<blank>")
    results, logits, _ = BUILDER.BUILDER.validate_dense_shard(
        DENSE, index, shard_count, verify_hashes=False
    )
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
            selected = []
            local = Counter()
            for start in range(0, len(probabilities), 4):
                selected.append(start)
                if start + 3 >= len(probabilities):
                    continue
                candidates = [start + 1, start + 2, start + 3]
                futures = [ROBUST.continuation(len(probabilities), start + 4, 2),
                           ROBUST.continuation(len(probabilities), start + 4, 3)]
                chosen, _, rewards = ROBUST.robust_choice(
                    probabilities, selected, candidates, futures, reference, vocab, blank
                )
                selected.append(candidates[chosen])
                local["blocks"] += 1
                local["side_choices"] += chosen != 1
                local["left_choices"] += chosen == 0
                local["right_choices"] += chosen == 2
                local["worst_case_reward"] += int(rewards[chosen])
                local["positive_side_labels"] += int(rewards[0] > 0) + int(rewards[2] > 0)
                local["harmful_side_labels"] += int(rewards[0] < 0) + int(rewards[2] < 0)
            uniform_counts = BLOCK.decode_counts(probabilities, uniform, reference, vocab, blank)
            robust_counts = BLOCK.decode_counts(probabilities, selected, reference, vocab, blank)
            coverage = CHRON.coverage_metrics(selected, len(probabilities))
            if coverage["max_gap_including_endpoints"] > 4:
                raise RuntimeError(f"coverage invariant violated for {name}")
            row = {"sample_id": name, "source_video_id": BUILDER.source_video(name),
                   "uniform_error": int(uniform_counts["error"]),
                   "robust_error": int(robust_counts["error"]),
                   "ref_len": int(uniform_counts["ref_len"]),
                   "selected_windows": len(selected), "dense_windows": len(probabilities),
                   **dict(local)}
            handle.write(json.dumps(row, separators=(",", ":")) + "\n")
            counts.update({"samples": 1, "uniform_error": row["uniform_error"],
                           "robust_error": row["robust_error"], "ref_len": row["ref_len"],
                           "selected_windows": row["selected_windows"],
                           "dense_windows": row["dense_windows"], **dict(local)})
    temp.replace(data_path)
    value = {"status": "complete", "shard": index, "shards": shard_count,
             "counts": dict(counts), "sha256": sha256_file(data_path),
             "bytes": data_path.stat().st_size, "seconds": time.perf_counter() - started}
    marker_path.write_text(json.dumps(value, indent=2) + "\n")
    return value


def source_bootstrap(rows, reps=10000):
    grouped = defaultdict(lambda: [0, 0])
    for row in rows:
        value = grouped[row["source_video_id"]]
        value[0] += row["robust_error"] - row["uniform_error"]
        value[1] += row["ref_len"]
    sources = sorted(grouped)
    errors = np.asarray([grouped[source][0] for source in sources], dtype=np.int64)
    refs = np.asarray([grouped[source][1] for source in sources], dtype=np.int64)
    rng = np.random.default_rng(SEED)
    estimates = np.empty(reps, dtype=np.float64)
    for index in range(reps):
        chosen = rng.integers(0, len(sources), size=len(sources))
        estimates[index] = 100.0 * errors[chosen].sum() / max(1, refs[chosen].sum())
    return [float(np.quantile(estimates, 0.025)), float(np.quantile(estimates, 0.975))]


def finalize(config):
    indices, shard_count = CHRON.completed_shard_indices(DENSE)
    rows = []
    aggregate = Counter()
    shard_info = []
    for index in indices:
        marker = valid_completed_shard(index, shard_count)
        if marker is None:
            raise RuntimeError(f"missing valid oracle shard {index}")
        shard_info.append(marker)
        data, _ = shard_paths(index, shard_count)
        with gzip.open(data, "rt", encoding="utf-8") as handle:
            for line in handle:
                row = json.loads(line)
                rows.append(row)
                aggregate.update({"samples": 1, "uniform_error": row["uniform_error"],
                                  "robust_error": row["robust_error"], "ref_len": row["ref_len"],
                                  "selected_windows": row["selected_windows"],
                                  "dense_windows": row["dense_windows"],
                                  **{key: row.get(key, 0) for key in (
                                      "blocks", "side_choices", "left_choices", "right_choices",
                                      "worst_case_reward", "positive_side_labels", "harmful_side_labels")}})
    names = [row["sample_id"] for row in rows]
    sources = {row["source_video_id"] for row in rows}
    if len(names) != config["scope"]["samples"] or len(set(names)) != len(names):
        raise RuntimeError("fit sample coverage or uniqueness mismatch")
    if hashlib.sha256("\n".join(sorted(names)).encode()).hexdigest() != config["scope"]["sample_ids_sha256"]:
        raise RuntimeError("fit sample identity mismatch")
    if len(sources) != config["scope"]["sources"]:
        raise RuntimeError("fit source coverage mismatch")
    uniform_wer = 100.0 * aggregate["uniform_error"] / aggregate["ref_len"]
    robust_wer = 100.0 * aggregate["robust_error"] / aggregate["ref_len"]
    delta = robust_wer - uniform_wer
    ci = source_bootstrap(rows)
    value = {
        "status": "complete",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "scope": config["scope"],
        "checkpoint_sha256": config["checkpoint_sha256"],
        "uniform": {"errors": aggregate["uniform_error"], "wer": uniform_wer},
        "robust_oracle": {"errors": aggregate["robust_error"], "wer": robust_wer,
                          "delta_errors": aggregate["robust_error"] - aggregate["uniform_error"],
                          "delta_wer_pp": delta,
                          "source_cluster_bootstrap_95pct_delta_wer_pp": ci},
        "schedule": {"blocks": aggregate["blocks"], "side_choices": aggregate["side_choices"],
                     "left_choices": aggregate["left_choices"], "right_choices": aggregate["right_choices"],
                     "side_choice_rate": aggregate["side_choices"] / max(1, aggregate["blocks"]),
                     "positive_side_labels": aggregate["positive_side_labels"],
                     "harmful_side_labels": aggregate["harmful_side_labels"],
                     "selected_windows": aggregate["selected_windows"],
                     "dense_windows": aggregate["dense_windows"],
                     "window_rate": aggregate["selected_windows"] / aggregate["dense_windows"],
                     "summed_worst_case_local_reward": aggregate["worst_case_reward"]},
        "gate": {"passed": delta <= -0.5 and ci[1] < 0,
                 "checks": {"delta_wer_le_minus_0_5pp": delta <= -0.5,
                            "bootstrap_upper_lt_0": ci[1] < 0}},
        "shards": shard_info,
        "limitations": ["train-fit confirmatory target-headroom audit",
                        "reference/future-aware non-deployable oracle",
                        "calibration/dev/test outcomes not read"],
    }
    (OUTPUT / "metrics.json").write_text(json.dumps(value, indent=2) + "\n")
    return value


def run(workers):
    config = preregister()
    indices, _ = CHRON.completed_shard_indices(DENSE)
    pending = [index for index in indices if valid_completed_shard(index, len(indices)) is None]
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
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    if not 1 <= args.workers <= 16:
        raise ValueError("workers must be between 1 and 16")
    print(json.dumps(run(args.workers), indent=2))
