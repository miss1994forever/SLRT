#!/usr/bin/env python3
"""Build resumable chronological rollout-16 teacher data on partial32 train.

Future dense logits and references are used only to form the signed teacher
advantage.  Serialized predictor inputs are reconstructible from the current
arrival, the token/coverage state, and past executed windows.
"""
import argparse
import gzip
import hashlib
import importlib.util
import json
import math
import os
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def import_tool(name):
    path = ROOT / f"tools/{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


CHRON = import_tool("analyze_phoenix_partial_chronological_oracle")
BUILDER, ORACLE, REPLAY = CHRON.BUILDER, CHRON.ORACLE, CHRON.REPLAY
DEFAULT_DENSE_ROOT = CHRON.DEFAULT_DENSE_ROOT
DEFAULT_VOCAB = CHRON.DEFAULT_VOCAB
DEFAULT_OUTPUT_ROOT = ROOT / "results/phoenix-2014t_ISLR/p3_partial32_rollout_predictor_dataset_smoke_v1_49faacc3"


def sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def source_video(name):
    stem, suffix = name.rsplit("-", 1)
    if not suffix.isdigit():
        raise ValueError(f"unexpected sample id: {name}")
    return stem


def prefix_features(probabilities, selected, current, blank_id):
    """Fixed-width state from past selected logits only."""
    past = sorted(int(x) for x in selected if int(x) < int(current))
    starts, scores, token_ids = ORACLE.decoder_state(probabilities, past)
    if len(scores):
        normalized = scores / np.maximum(scores.sum(axis=1, keepdims=True), 1e-12)
        entropy = -np.sum(normalized * np.log(np.maximum(normalized, 1e-12)), axis=1)
        collapsed = [int(x) for x in token_ids if int(x) != int(blank_id)]
        last_entropy = float(entropy[-1])
        mean_entropy = float(entropy.mean())
        delta_entropy = float(entropy[-1] - entropy[-2]) if len(entropy) > 1 else 0.0
        last_token = int(token_ids[-1])
        prev_token = int(token_ids[-2]) if len(token_ids) > 1 else int(blank_id)
        repeat_run = 1
        for value in token_ids[-2::-1]:
            if int(value) != last_token:
                break
            repeat_run += 1
    else:
        collapsed, last_entropy, mean_entropy, delta_entropy = [], 0.0, 0.0, 0.0
        last_token = prev_token = int(blank_id)
        repeat_run = 0
    # Token identities are categorical; stable hash buckets avoid imposing an
    # arbitrary numeric vocabulary geometry while keeping this smoke compact.
    token_buckets = [0.0] * 32
    for offset, token in enumerate(collapsed[-4:]):
        token_buckets[(int(token) * 131 + offset * 17) % len(token_buckets)] += 1.0
    return {
        "decoder_prefix_length": len(collapsed),
        "decoder_raw_path_length": len(token_ids),
        "decoder_mean_entropy": mean_entropy,
        "decoder_last_entropy": last_entropy,
        "decoder_entropy_delta": delta_entropy,
        "decoder_last_is_blank": float(last_token == blank_id),
        "decoder_last_repeats_previous": float(last_token == prev_token and len(token_ids) > 1),
        "decoder_last_repeat_run": repeat_run,
        "decoder_token_hash32": token_buckets,
        "prefix_selected_starts_sha256": hashlib.sha256(
            ",".join(str(x) for x in starts).encode("ascii")
        ).hexdigest(),
    }


def sample_records(name, result, logits, vocab, blank_id, partition):
    probabilities = ORACLE.softmax_rows(np.asarray(logits))
    reference = REPLAY.clean_phoenix_2014_trans(result["gls_ref"])
    starts = [int(x["start"]) for x in result["adaptive_stride_metadata"]]
    if starts != list(range(len(probabilities))):
        raise ValueError(f"non-dense coordinates: {name}")
    selected, balance, records = [], 0.0, []
    counts = Counter()
    for start in range(len(probabilities)):
        if CHRON.is_skeleton(start):
            selected.append(start)
            counts["skeleton"] += 1
            continue
        balance = CHRON.accrue_token(balance)
        if not CHRON.token_eligible(balance):
            counts["ineligible"] += 1
            continue
        forced = CHRON.token_forced(balance)
        advantage, skip_error, execute_error = CHRON.rollout_advantage(
            probabilities, selected, balance, start, reference, vocab, blank_id,
            horizon=CHRON.ROLLOUT_HORIZON,
        )
        prefix = prefix_features(probabilities, selected, start, blank_id)
        last = selected[-1] if selected else None
        record = {
            "sample_id": name,
            "source_video_id": source_video(name),
            "partition": partition,
            "candidate_window_start": start,
            "visible_original_frame_upper": min(len(probabilities) - 1, start + CHRON.LOOKAHEAD_FRAMES),
            "requires_eos_flush": bool(start + CHRON.LOOKAHEAD_FRAMES >= len(probabilities)),
            "forced_by_token_capacity": forced,
            "autonomous_training_row": not forced,
            "bookkeeping": {
                "candidate_absolute_index": start,
                "candidate_mod_skeleton_period": start % CHRON.SKELETON_PERIOD,
                "coverage_gap": start - last if last is not None else start + 1,
                "frames_since_last_execute": start - last if last is not None else start + 1,
                "bonus_token_balance": float(balance),
                "past_selected_count": len(selected),
            },
            "prefix": prefix,
            "label": {
                "rollout16_signed_advantage": int(advantage),
                "rollout16_positive": bool(advantage > 0),
                "skip_error": int(skip_error),
                "execute_error": int(execute_error),
            },
            "teacher_action_execute": bool(forced or advantage > 0),
            "provenance": {
                "future_dense_logits_and_reference_used_only_for_label": True,
                "current_candidate_expensive_logits_absent_from_inputs": True,
                "prefix_uses_only_past_selected": True,
                "total_length_absent_from_inputs": True,
            },
        }
        records.append(record)
        counts["forced" if forced else "autonomous"] += 1
        counts[f"advantage_{advantage}"] += 1
        if forced or advantage > 0:
            selected.append(start)
            balance -= 1.0
    return records, counts


def build_shard(dense_root, output_root, shard_index, shard_count, assignments, vocab, blank_id):
    final = output_root / f"rollout-predictor-{shard_index:05d}-of-{shard_count:05d}.jsonl.gz"
    marker = output_root / f"rollout-predictor-{shard_index:05d}-of-{shard_count:05d}.complete.json"
    if marker.is_file() and final.is_file():
        payload = json.loads(marker.read_text())
        if payload["sha256"] == sha256_file(final):
            return {**payload, "resumed": True}
        raise ValueError(f"hash mismatch for completed shard {shard_index}")
    results, logits, _ = BUILDER.validate_dense_shard(
        dense_root, shard_index, shard_count, verify_hashes=False
    )
    temporary = final.with_name(f".{final.name}.incomplete-{os.getpid()}")
    totals = Counter()
    samples = 0
    started = time.perf_counter()
    with gzip.open(temporary, "wt", encoding="utf-8", compresslevel=5) as handle:
        for name in results:
            partition = assignments[name]
            records, counts = sample_records(name, results[name], logits[name], vocab, blank_id, partition)
            for record in records:
                handle.write(json.dumps(record, separators=(",", ":")) + "\n")
            totals.update(counts)
            totals[f"samples_{partition}"] += 1
            totals[f"rows_{partition}"] += len(records)
            samples += 1
    temporary.replace(final)
    payload = {
        "shard": shard_index, "samples": samples, "counts": dict(totals),
        "sha256": sha256_file(final), "bytes": final.stat().st_size,
        "elapsed_seconds": time.perf_counter() - started,
        "created_utc": datetime.now(timezone.utc).isoformat(),
    }
    marker.write_text(json.dumps(payload, indent=2) + "\n")
    return {**payload, "resumed": False}


def run(dense_root, vocab_path, output_root, worker_id=None, workers=1, finalize_only=False):
    BUILDER.reject_test_path(dense_root); BUILDER.reject_test_path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    indices, shard_count = CHRON.completed_shard_indices(dense_root)
    if len(indices) != 32:
        raise ValueError(f"expected frozen partial32, got {len(indices)}")
    split = json.loads((dense_root / "fit_calibration_split.json").read_text())
    assignments = split["assignments"]
    fit_sources = {source_video(k) for k, v in assignments.items() if v == "fit"}
    cal_sources = {source_video(k) for k, v in assignments.items() if v == "calibration"}
    if fit_sources & cal_sources:
        raise ValueError("fit/calibration source overlap")
    vocab = json.loads(vocab_path.read_text()); blank_id = vocab.index("<blank>")
    config = {
        "schema_version": 1, "status": "running", "created_utc": datetime.now(timezone.utc).isoformat(),
        "dense_shards": indices, "source_disjoint": True,
        "protocol": {"skeleton_period": 4, "accrual": 1/3, "initial": 0, "capacity": 2,
                     "eligible": 1, "forced": 2, "eos_topup": False, "rollout_horizon": 16,
                     "lookahead_frames": 8, "teacher_future_policy": "fixed eager"},
        "inputs_exclude": ["T", "EOS", "reference", "future_logits", "current_candidate_logits"],
    }
    (output_root / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    if worker_id is not None and not 0 <= worker_id < workers:
        raise ValueError("worker-id must be in [0, workers)")
    work_indices = [] if finalize_only else [
        index for position, index in enumerate(indices)
        if worker_id is None or position % workers == worker_id
    ]
    summaries = []
    for position, index in enumerate(work_indices, 1):
        value = build_shard(dense_root, output_root, index, shard_count, assignments, vocab, blank_id)
        summaries.append(value)
        print(json.dumps({"worker_id": worker_id, "completed": position, "total": len(work_indices), **value}), flush=True)
    if worker_id is not None:
        worker_manifest = {"worker_id": worker_id, "workers": workers, "shards": summaries}
        (output_root / f"worker-{worker_id:02d}.json").write_text(json.dumps(worker_manifest, indent=2) + "\n")
        return worker_manifest
    # A coordinator pass validates and aggregates every atomic marker.  It is
    # cheap after parallel workers finish and never trusts an unmarked shard.
    summaries = []
    for index in indices:
        final = output_root / f"rollout-predictor-{index:05d}-of-{shard_count:05d}.jsonl.gz"
        marker = output_root / f"rollout-predictor-{index:05d}-of-{shard_count:05d}.complete.json"
        if not final.is_file() or not marker.is_file():
            raise RuntimeError(f"missing completed shard {index}")
        value = json.loads(marker.read_text())
        if value["sha256"] != sha256_file(final):
            raise ValueError(f"hash mismatch for shard {index}")
        summaries.append(value)
    counts = Counter()
    for row in summaries: counts.update(row["counts"])
    manifest = {**config, "status": "complete", "completed_utc": datetime.now(timezone.utc).isoformat(),
                "counts": dict(counts), "shards": summaries}
    (output_root / "dataset_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dense-root", type=Path, default=DEFAULT_DENSE_ROOT)
    parser.add_argument("--vocab", type=Path, default=DEFAULT_VOCAB)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--worker-id", type=int)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--finalize-only", action="store_true")
    args = parser.parse_args()
    run(args.dense_root, args.vocab, args.output_root, args.worker_id, args.workers, args.finalize_only)


if __name__ == "__main__":
    main()
