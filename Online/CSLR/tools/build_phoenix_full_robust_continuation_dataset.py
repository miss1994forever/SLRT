#!/usr/bin/env python3
"""Build resumable full-fit robust-continuation counterfactual labels."""
import argparse
import concurrent.futures
import gzip
import hashlib
import importlib.util
import json
import os
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "results/phoenix-2014t_ISLR"
OUTPUT = BASE / "p3_fulltrain_robust_continuation_fit6378_dataset_v2_49faacc3"


def imp(name):
    path = ROOT / f"tools/{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


FULL = imp("analyze_phoenix_full_robust_continuation_oracle")
ROBUST, CHRON, BUILDER, BLOCK = FULL.ROBUST, FULL.CHRON, FULL.BUILDER, FULL.BLOCK
ORACLE_RESULT = FULL.OUTPUT
DENSE = FULL.DENSE


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def frozen_config():
    oracle = json.loads((ORACLE_RESULT / "metrics.json").read_text())
    if not oracle["gate"]["passed"]:
        raise ValueError("full robust oracle did not pass its frozen gate")
    return {
        "experiment": "complete-train fit robust-continuation counterfactual block-label dataset",
        "created_before_dataset_outcomes": True,
        "scope": oracle["scope"],
        "checkpoint_sha256": oracle["checkpoint_sha256"],
        "teacher": {
            "past": "closed-loop robust-oracle selected past",
            "future_continuations": ["fixed offset2 center", "fixed offset3 late"],
            "side_label": "strictly positive minimum terminal edit-error improvement versus center",
            "selection": "maximum minimum improvement, tie left then right, otherwise center",
            "uses_reference_future_and_EOS": True,
        },
        "predictor_inputs": {
            "stored": ["bookkeeping", "past-paid-window decoder-prefix summary"],
            "prefix_implementation": "span-local banded decoder-state parity implementation",
            "external_visual_archive": "must be supplied by separately audited full-train causal feature build",
            "candidate_expensive_logits": False,
            "reference_future_total_length_or_EOS": False,
            "decision_arrival": "offset3, so all three candidate positions have arrived",
        },
        "intended_use": "source-disjoint fit-only OOF; calibration/dev/test forbidden for model selection",
        "oracle_metrics_sha256": sha256_file(ORACLE_RESULT / "metrics.json"),
        "forbidden": ["GPU", "calibration outcomes", "dev", "test", "predictor training", "git commit"],
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


def banded_decoder_state(probabilities, starts, span=15.0, min_weight=0.05):
    """Exact span-local form of the frozen dense weight-matrix decoder state."""
    starts = np.asarray(sorted(int(value) for value in starts), dtype=np.int64)
    if len(starts) == 0:
        return starts, np.empty((0, probabilities.shape[1]), dtype=np.float32), np.empty(0, dtype=np.int64)
    radius = np.float32(span / 2.0)
    scores = np.empty((len(starts), probabilities.shape[1]), dtype=np.float32)
    for index, center in enumerate(starts):
        left = int(np.searchsorted(starts, center - radius, side="left"))
        right = int(np.searchsorted(starts, center + radius, side="right"))
        distance = np.abs(starts[left:right] - center).astype(np.float32)
        weights = np.maximum(np.float32(1.0) - distance / radius, np.float32(min_weight))
        scores[index] = (weights[:, None] * probabilities[starts[left:right]]).sum(axis=0)
    return starts, scores, scores.argmax(axis=1)


def banded_prefix_features(probabilities, selected, current, blank):
    past = sorted(int(value) for value in selected if int(value) < int(current))
    starts, scores, token_ids = banded_decoder_state(probabilities, past)
    if len(scores):
        normalized = scores / np.maximum(scores.sum(axis=1, keepdims=True), 1e-12)
        entropy = -np.sum(normalized * np.log(np.maximum(normalized, 1e-12)), axis=1)
        collapsed = [int(value) for value in token_ids if int(value) != int(blank)]
        last_entropy = float(entropy[-1])
        mean_entropy = float(entropy.mean())
        delta_entropy = float(entropy[-1] - entropy[-2]) if len(entropy) > 1 else 0.0
        last_token = int(token_ids[-1])
        previous_token = int(token_ids[-2]) if len(token_ids) > 1 else int(blank)
        repeat_run = 1
        for value in token_ids[-2::-1]:
            if int(value) != last_token:
                break
            repeat_run += 1
    else:
        collapsed, last_entropy, mean_entropy, delta_entropy = [], 0.0, 0.0, 0.0
        last_token = previous_token = int(blank)
        repeat_run = 0
    buckets = [0.0] * 32
    for offset, token in enumerate(collapsed[-4:]):
        buckets[(int(token) * 131 + offset * 17) % len(buckets)] += 1.0
    raw = {
        "decoder_prefix_length": len(collapsed),
        "decoder_raw_path_length": len(token_ids),
        "decoder_mean_entropy": mean_entropy,
        "decoder_last_entropy": last_entropy,
        "decoder_entropy_delta": delta_entropy,
        "decoder_last_is_blank": float(last_token == blank),
        "decoder_last_repeats_previous": float(last_token == previous_token and len(token_ids) > 1),
        "decoder_last_repeat_run": repeat_run,
        "decoder_token_hash32": buckets,
        "prefix_selected_starts_sha256": hashlib.sha256(
            ",".join(str(int(value)) for value in starts).encode("ascii")
        ).hexdigest(),
    }
    return BLOCK.OLD.prefix_features({"prefix": raw}).tolist()


def shared_prefix_candidate_features(probabilities, selected, candidates, blank):
    """Equivalent to BLOCK.candidate_features when candidates share paid past."""
    if not selected:
        raise ValueError("structured block candidates must have a skeleton in paid past")
    if any(int(value) >= min(candidates) for value in selected):
        raise ValueError("candidate feature optimization requires strictly past selected windows")
    prefix = banded_prefix_features(probabilities, selected, candidates[0], blank)
    last = selected[-1]
    values = []
    for start in candidates:
        raw = {"bookkeeping": {
            "candidate_absolute_index": int(start),
            "candidate_mod_skeleton_period": int(start) % 4,
            "coverage_gap": int(start - last),
            "frames_since_last_execute": int(start - last),
            "bonus_token_balance": 1.0,
            "past_selected_count": len(selected),
        }}
        values.append({"bookkeeping": BLOCK.OLD.bookkeeping_features(raw).tolist(),
                       "prefix": prefix})
    return values


def process_shard(index):
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    torch.set_num_threads(1)
    indices, shard_count = CHRON.completed_shard_indices(DENSE)
    if index not in indices:
        raise ValueError(f"dense shard {index} not complete")
    completed = valid_completed_shard(index, shard_count)
    if completed is not None:
        return completed | {"resumed": True}
    data_path, marker_path = shard_paths(index, shard_count)
    if data_path.exists() or marker_path.exists():
        raise FileExistsError(f"incomplete prior output requires audit: {data_path}")
    wanted = FULL.fit_assignments()
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
            selected = []
            local = Counter()
            for start in range(0, len(probabilities), 4):
                selected.append(start)
                if start + 3 >= len(probabilities):
                    continue
                candidates = [start + 1, start + 2, start + 3]
                futures = [ROBUST.continuation(len(probabilities), start + 4, 2),
                           ROBUST.continuation(len(probabilities), start + 4, 3)]
                choice, errors, rewards = ROBUST.robust_choice(
                    probabilities, selected, candidates, futures, reference, vocab, blank
                )
                row = {
                    "type": "block", "sample_id": name,
                    "source_video_id": BUILDER.source_video(name),
                    "block_start": start, "decision_arrival": start + 3,
                    "candidate_starts": candidates,
                    "predictor_inputs": shared_prefix_candidate_features(
                        probabilities, selected, candidates, blank
                    ),
                    "label": {
                        "terminal_errors_center_future": errors[0],
                        "terminal_errors_late_future": errors[1],
                        "robust_reward_by_candidate": rewards,
                        "strictly_beneficial_side_mask": [rewards[0] > 0, False, rewards[2] > 0],
                        "teacher_choice": choice,
                    },
                    "provenance": {
                        "predictor_prefix_past_selected_only": True,
                        "candidate_expensive_logits_absent": True,
                        "reference_future_EOS_label_only": True,
                        "unknown_EOS_deployment_compatible_inputs": True,
                    },
                }
                handle.write(json.dumps(row, separators=(",", ":")) + "\n")
                selected.append(candidates[choice])
                local["blocks"] += 1
                local["side_choices"] += choice != 1
                local["positive_side_labels"] += int(rewards[0] > 0) + int(rewards[2] > 0)
                local["harmful_side_labels"] += int(rewards[0] < 0) + int(rewards[2] < 0)
            uniform_counts = BLOCK.decode_counts(
                probabilities, BLOCK.block_uniform(len(probabilities)), reference, vocab, blank
            )
            robust_counts = BLOCK.decode_counts(probabilities, selected, reference, vocab, blank)
            sample = {
                "type": "sample", "sample_id": name,
                "source_video_id": BUILDER.source_video(name),
                "dense_windows": len(probabilities), "selected_windows": len(selected),
                "ref_len": int(uniform_counts["ref_len"]),
                "uniform_error": int(uniform_counts["error"]),
                "robust_error": int(robust_counts["error"]), **dict(local),
            }
            handle.write(json.dumps(sample, separators=(",", ":")) + "\n")
            counts.update({"samples": 1, "uniform_error": sample["uniform_error"],
                           "robust_error": sample["robust_error"], "ref_len": sample["ref_len"],
                           "selected_windows": sample["selected_windows"],
                           "dense_windows": sample["dense_windows"], **dict(local)})
    oracle_marker = FULL.valid_completed_shard(index, shard_count)
    expected_keys = ("samples", "uniform_error", "robust_error", "ref_len", "blocks",
                     "side_choices", "positive_side_labels", "harmful_side_labels")
    observed = {key: counts[key] for key in expected_keys}
    expected = {key: oracle_marker["counts"].get(key, 0) for key in expected_keys}
    if observed != expected:
        raise RuntimeError(f"oracle shard reproduction mismatch: {observed} != {expected}")
    temp.replace(data_path)
    value = {"status": "complete", "shard": index, "shards": shard_count,
             "counts": dict(counts), "sha256": sha256_file(data_path),
             "bytes": data_path.stat().st_size, "seconds": time.perf_counter() - started}
    marker_path.write_text(json.dumps(value, indent=2) + "\n")
    return value


def finalize(config):
    indices, shard_count = CHRON.completed_shard_indices(DENSE)
    aggregate = Counter()
    shard_info = []
    sample_ids = []
    for index in indices:
        marker = valid_completed_shard(index, shard_count)
        if marker is None:
            raise RuntimeError(f"missing valid label shard {index}")
        shard_info.append(marker)
        aggregate.update(marker["counts"])
        data, _ = shard_paths(index, shard_count)
        with gzip.open(data, "rt", encoding="utf-8") as handle:
            for line in handle:
                row = json.loads(line)
                if row["type"] == "sample":
                    sample_ids.append(row["sample_id"])
    if len(sample_ids) != config["scope"]["samples"] or len(set(sample_ids)) != len(sample_ids):
        raise RuntimeError("sample coverage or uniqueness mismatch")
    if hashlib.sha256("\n".join(sorted(sample_ids)).encode()).hexdigest() != config["scope"]["sample_ids_sha256"]:
        raise RuntimeError("sample identity mismatch")
    oracle = json.loads((ORACLE_RESULT / "metrics.json").read_text())
    observed = {"uniform_error": aggregate["uniform_error"],
                "robust_error": aggregate["robust_error"], "blocks": aggregate["blocks"],
                "side_choices": aggregate["side_choices"]}
    expected = {"uniform_error": oracle["uniform"]["errors"],
                "robust_error": oracle["robust_oracle"]["errors"],
                "blocks": oracle["schedule"]["blocks"],
                "side_choices": oracle["schedule"]["side_choices"]}
    if observed != expected:
        raise RuntimeError(f"global oracle reproduction mismatch: {observed} != {expected}")
    value = {
        "status": "complete", "created_utc": datetime.now(timezone.utc).isoformat(),
        "scope": config["scope"], "counts": dict(aggregate),
        "oracle_reproduction": observed,
        "config_sha256": sha256_file(OUTPUT / "resolved_config_preregistered.json"),
        "shards": shard_info,
    }
    (OUTPUT / "dataset_manifest.json").write_text(json.dumps(value, indent=2) + "\n")
    return value


def run(workers):
    config = preregister()
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
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    if not 1 <= args.workers <= 16:
        raise ValueError("workers must be between 1 and 16")
    print(json.dumps(run(args.workers), indent=2))
