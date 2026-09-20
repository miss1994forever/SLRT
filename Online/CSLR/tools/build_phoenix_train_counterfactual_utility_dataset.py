#!/usr/bin/env python3
"""Build resumable train counterfactual-utility shards from frozen dense logits.

This is offline reference-aware supervision.  It preserves the v1 state-record
schema used by the frozen dev dataset and keeps the train fit/calibration
assignment in a separate, source-video-disjoint manifest.
"""

import argparse
import gzip
import hashlib
import importlib.util
import json
import math
import multiprocessing as mp
import os
import pickle
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path


CSLR_ROOT = Path(__file__).resolve().parents[1]
BASE_PATH = CSLR_ROOT / "tools/build_phoenix_counterfactual_utility_dataset.py"
SPEC = importlib.util.spec_from_file_location("counterfactual_utility_base", BASE_PATH)
BASE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = BASE
SPEC.loader.exec_module(BASE)
REPLAY = BASE.REPLAY
ORACLE = BASE.ORACLE

DEFAULT_DENSE_ROOT = (
    CSLR_ROOT / "results/phoenix-2014t_ISLR/train_dense_stride1_v1_49faacc3"
)
DEFAULT_OUTPUT_ROOT = (
    CSLR_ROOT
    / "results/phoenix-2014t_ISLR/p3_train_counterfactual_utility_dataset_v1_49faacc3"
)
DEFAULT_VOCAB = REPLAY.DEFAULT_VOCAB
DATASET_VARIANT = "train_source_video_split_v1"
EXPECTED_DENSE_GENERATOR_COMMIT = "d513bd21e8135e03b9eee201a05bf88654d4f098"


def reject_test_path(path):
    BASE.reject_test_path(path)


def sha256_file(path, chunk_size=8 * 1024 * 1024):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_write_json(path, value):
    path = Path(path)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def dense_shard_paths(dense_root, shard_index, shard_count):
    root = dense_root / "shards" / f"shard-{shard_index:05d}-of-{shard_count:05d}"
    return root, root / "train_results.pkl", root / "train_logits.pkl", root / "complete.json"


def validate_dense_shard(dense_root, shard_index, shard_count, verify_hashes=True):
    shard_root, results_path, logits_path, complete_path = dense_shard_paths(
        dense_root, shard_index, shard_count
    )
    if not complete_path.is_file():
        raise FileNotFoundError(f"dense shard is incomplete: {shard_root}")
    completion = json.loads(complete_path.read_text(encoding="utf-8"))
    if completion["shard_index"] != shard_index:
        raise ValueError("dense shard index mismatch")
    for path in (results_path, logits_path):
        expected = completion["artifacts"][path.name]
        if path.stat().st_size != expected["bytes"]:
            raise ValueError(f"dense artifact size mismatch: {path}")
        if verify_hashes and sha256_file(path) != expected["sha256"]:
            raise ValueError(f"dense artifact hash mismatch: {path}")
    with results_path.open("rb") as handle:
        results = pickle.load(handle)
    with logits_path.open("rb") as handle:
        logits = pickle.load(handle)
    if list(results) != list(logits):
        raise ValueError("dense result/logit sample order differs")
    observed_windows = 0
    for name in results:
        if not name.startswith("train/"):
            raise ValueError("non-train sample in dense shard")
        starts = [int(row["start"]) for row in results[name]["adaptive_stride_metadata"]]
        if starts != list(range(len(logits[name]))):
            raise ValueError(f"non-dense candidate coordinates: {name}")
        observed_windows += len(logits[name])
    if len(results) != completion["samples"] or observed_windows != completion["windows"]:
        raise ValueError("dense completion counts differ from artifacts")
    return results, logits, completion


def validate_train_partition(split_manifest, names):
    assignments = split_manifest["assignments"]
    if set(assignments) != set(names):
        raise ValueError("fit/calibration assignments do not exactly cover train samples")
    if any(value not in {"fit", "calibration"} for value in assignments.values()):
        raise ValueError("unknown train partition")
    source_partition = {}
    for name, partition in assignments.items():
        source = name.rpartition("-")[0]
        previous = source_partition.setdefault(source, partition)
        if previous != partition:
            raise ValueError(f"source video crosses fit/calibration: {source}")
    return Counter(assignments.values())


def output_paths(output_root, shard_index, shard_count):
    filename = f"states-{shard_index:05d}-of-{shard_count:05d}.jsonl.gz"
    shard_path = output_root / "shards" / filename
    complete_path = output_root / "shards" / f"states-{shard_index:05d}-of-{shard_count:05d}.complete.json"
    return shard_path, complete_path


def validate_output_shard(output_root, shard_index, shard_count):
    shard_path, complete_path = output_paths(output_root, shard_index, shard_count)
    if not shard_path.is_file() or not complete_path.is_file():
        raise FileNotFoundError("utility shard is incomplete")
    completion = json.loads(complete_path.read_text(encoding="utf-8"))
    if completion["shard_index"] != shard_index:
        raise ValueError("utility shard index mismatch")
    if shard_path.stat().st_size != completion["bytes"]:
        raise ValueError("utility shard size mismatch")
    if sha256_file(shard_path) != completion["sha256"]:
        raise ValueError("utility shard hash mismatch")
    states = candidate_records = 0
    with gzip.open(shard_path, "rt", encoding="utf-8") as handle:
        for line in handle:
            record = json.loads(line)
            if record["schema_version"] != BASE.SCHEMA_VERSION:
                raise ValueError("state schema mismatch")
            if not record["sample_id"].startswith("train/"):
                raise ValueError("non-train state in utility shard")
            reconstructed = sorted(
                record["oracle_state"]["initial_skeleton"]
                + record["oracle_state"]["selected_bonus_before"]
            )
            if BASE.selected_set_hash(reconstructed) != record["oracle_state"]["selected_set_sha256"]:
                raise ValueError("state reconstruction hash mismatch")
            for candidate in record["candidates"]:
                BASE.assert_predictor_inputs_clean(candidate["predictor_inputs"])
            states += 1
            candidate_records += record["candidate_count"]
    if states != completion["states"] or candidate_records != completion["candidate_records"]:
        raise ValueError("utility shard count mismatch")
    return completion


def process_dense_shard(results, logits, vocab, workers):
    names = list(results)
    references = {
        name: REPLAY.clean_phoenix_2014_trans(results[name]["gls_ref"]) for name in names
    }
    BASE._LOGITS = logits
    BASE._REFERENCES = references
    BASE._VOCAB = vocab
    BASE._BLANK_ID = vocab.index("<blank>")
    BASE._PAIR_AUXILIARY_NAMES = set()
    statistics = {
        "samples": 0,
        "states": 0,
        "candidate_records": 0,
        "all_zero_states": 0,
        "positive_utility_candidates": 0,
        "negative_utility_candidates": 0,
        "utility_counts": Counter(),
    }
    with mp.get_context("fork").Pool(processes=workers) as pool:
        samples = list(pool.imap(BASE.build_sample, names, chunksize=1))
    for sample in samples:
        BASE.update_statistics(statistics, sample)
    statistics["utility_counts"] = dict(
        sorted(statistics["utility_counts"].items(), key=lambda row: int(row[0]))
    )
    expected_states, expected_candidates = BASE.estimate_size(logits)
    if statistics["states"] != expected_states:
        raise AssertionError("state count differs from analytic estimate")
    if statistics["candidate_records"] != expected_candidates:
        raise AssertionError("candidate count differs from analytic estimate")
    return samples, statistics


def write_utility_shard(path, samples):
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    with gzip.open(temporary, "wt", encoding="utf-8", compresslevel=6) as handle:
        for sample in samples:
            for state in sample["states"]:
                handle.write(json.dumps(state, ensure_ascii=False, separators=(",", ":")) + "\n")
    temporary.replace(path)


def combine_statistics(completions):
    combined = {
        "samples": 0,
        "states": 0,
        "candidate_records": 0,
        "all_zero_states": 0,
        "positive_utility_candidates": 0,
        "negative_utility_candidates": 0,
        "utility_counts": Counter(),
    }
    for completion in completions:
        stats = completion["statistics"]
        for key in combined:
            if key == "utility_counts":
                combined[key].update(stats[key])
            else:
                combined[key] += stats[key]
    combined["utility_counts"] = dict(
        sorted(combined["utility_counts"].items(), key=lambda row: int(row[0]))
    )
    return combined


def build_dataset(dense_root, output_root, vocab_path, workers=4, start_shard=0,
                  stop_shard=None, shard_indices=None, resume=False,
                  verify_dense_hashes=True):
    for path in (dense_root, output_root, vocab_path):
        reject_test_path(path)
    protocol_path = dense_root / "protocol_manifest.json"
    split_path = dense_root / "fit_calibration_split.json"
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    split_manifest = json.loads(split_path.read_text(encoding="utf-8"))
    if protocol["split"] != "train" or protocol["git_commit"] != EXPECTED_DENSE_GENERATOR_COMMIT:
        raise ValueError("dense train protocol is not the frozen source revision")
    shard_samples = int(protocol["shard_samples"])
    shard_count = int(math.ceil(protocol["samples"] / shard_samples))
    stop = shard_count if stop_shard is None else min(stop_shard, shard_count)
    if not 0 <= start_shard < stop:
        raise ValueError("empty shard range")
    selected_shards = (
        list(range(start_shard, stop))
        if shard_indices is None
        else sorted(set(int(value) for value in shard_indices))
    )
    if not selected_shards or selected_shards[0] < 0 or selected_shards[-1] >= shard_count:
        raise ValueError("invalid selected shard indices")
    with vocab_path.open("r", encoding="utf-8") as handle:
        vocab = json.load(handle)
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "shards").mkdir(exist_ok=True)

    dataset_protocol = {
        "schema_version": BASE.SCHEMA_VERSION,
        "dataset_variant": DATASET_VARIANT,
        "split": "train",
        "frozen_git_commit": BASE.FROZEN_COMMIT,
        "model_forward_run": False,
        "dense_protocol": {"path": str(protocol_path.resolve()), "sha256": sha256_file(protocol_path)},
        "fit_calibration_split": {"path": str(split_path.resolve()), "sha256": sha256_file(split_path)},
        "vocab": {"path": str(vocab_path.resolve()), "sha256": sha256_file(vocab_path)},
        "builder_inputs": {
            str(Path(__file__).resolve()): sha256_file(Path(__file__).resolve()),
            str(BASE_PATH.resolve()): sha256_file(BASE_PATH),
        },
        "total_budget_rate": 0.5,
        "uniform_skeleton_share": 0.5,
        "primary_label": "single-step decoder edit-distance marginal utility",
        "two_step_auxiliary": "eligibility marker only when every single-step utility is zero",
        "predictor_input_contract": "reference/future/candidate logits/exact EOS budget forbidden",
        "selected_dense_shards": selected_shards,
    }
    dataset_protocol_path = output_root / "protocol_manifest.json"
    if dataset_protocol_path.exists():
        if json.loads(dataset_protocol_path.read_text(encoding="utf-8")) != dataset_protocol:
            raise ValueError("existing train utility protocol differs")
    else:
        atomic_write_json(dataset_protocol_path, dataset_protocol)
    split_copy = output_root / "fit_calibration_split.json"
    if split_copy.exists() and split_copy.read_text(encoding="utf-8") != split_path.read_text(encoding="utf-8"):
        raise ValueError("existing split copy differs")
    if not split_copy.exists():
        temporary = split_copy.with_name(f".{split_copy.name}.tmp-{os.getpid()}")
        temporary.write_text(split_path.read_text(encoding="utf-8"), encoding="utf-8")
        temporary.replace(split_copy)

    completions = []
    seen_names = []
    started = time.perf_counter()
    for shard_index in selected_shards:
        shard_path, complete_path = output_paths(output_root, shard_index, shard_count)
        if resume and shard_path.is_file() and complete_path.is_file():
            completion = validate_output_shard(output_root, shard_index, shard_count)
            completions.append(completion)
            seen_names.extend(completion["sample_ids"])
            print(f"skip validated utility shard {shard_index + 1}/{shard_count}", flush=True)
            continue
        if shard_path.exists() or complete_path.exists():
            raise FileExistsError(f"refusing incomplete utility shard {shard_index}")
        results, logits, dense_completion = validate_dense_shard(
            dense_root, shard_index, shard_count, verify_hashes=verify_dense_hashes
        )
        shard_started = time.perf_counter()
        samples, statistics = process_dense_shard(results, logits, vocab, workers)
        write_utility_shard(shard_path, samples)
        completion = {
            "shard_index": shard_index,
            "source_dense_complete_sha256": sha256_file(
                dense_shard_paths(dense_root, shard_index, shard_count)[3]
            ),
            "sample_ids": list(results),
            "dense_windows": dense_completion["windows"],
            "statistics": statistics,
            "states": statistics["states"],
            "candidate_records": statistics["candidate_records"],
            "bytes": shard_path.stat().st_size,
            "sha256": sha256_file(shard_path),
            "wall_seconds": time.perf_counter() - shard_started,
        }
        atomic_write_json(complete_path, completion)
        completion = validate_output_shard(output_root, shard_index, shard_count)
        completions.append(completion)
        seen_names.extend(completion["sample_ids"])
        print(f"completed utility shard {shard_index + 1}/{shard_count}", flush=True)

    complete_dataset = selected_shards == list(range(shard_count))
    if complete_dataset:
        partition_counts = validate_train_partition(split_manifest, seen_names)
    else:
        partition_counts = Counter(split_manifest["assignments"][name] for name in seen_names)
    summary = {
        "schema_version": BASE.SCHEMA_VERSION,
        "dataset_variant": DATASET_VARIANT,
        "split": "train",
        "selected_shard_indices": selected_shards,
        "complete_dataset": complete_dataset,
        "shard_count": shard_count,
        "completed_shards": len(completions),
        "statistics": combine_statistics(completions),
        "partition_samples": dict(sorted(partition_counts.items())),
        "runtime": {"workers": workers, "wall_seconds_this_invocation": time.perf_counter() - started},
        "created_utc": datetime.now(timezone.utc).isoformat(),
    }
    atomic_write_json(output_root / "dataset_summary.json", summary)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dense-root", type=Path, default=DEFAULT_DENSE_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--vocab", type=Path, default=DEFAULT_VOCAB)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--start-shard", type=int, default=0)
    parser.add_argument("--stop-shard", type=int)
    parser.add_argument(
        "--shard-indices",
        help="comma-separated non-contiguous dense shards; intended for partial smoke datasets",
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--skip-dense-hash-verification", action="store_true")
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("workers must be positive")
    shard_indices = None
    if args.shard_indices:
        if args.start_shard != 0 or args.stop_shard is not None:
            parser.error("--shard-indices cannot be combined with explicit shard bounds")
        shard_indices = [int(value) for value in args.shard_indices.split(",") if value]
    summary = build_dataset(
        args.dense_root.resolve(), args.output_root.resolve(), args.vocab.resolve(),
        workers=args.workers, start_shard=args.start_shard, stop_shard=args.stop_shard,
        shard_indices=shard_indices, resume=args.resume,
        verify_dense_hashes=not args.skip_dense_hash_verification,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
