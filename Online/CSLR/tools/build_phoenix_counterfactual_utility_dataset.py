#!/usr/bin/env python3
"""Build sharded all-candidate decoder-utility labels from frozen dev logits.

The greedy set-building trajectory and its labels are offline oracle objects.
Only fields nested under ``predictor_inputs`` are eligible to become inputs to
a later causal predictor; this builder intentionally excludes references,
future selections, candidate logits, and exact-EOS budget facts from them.
"""

import argparse
import gzip
import hashlib
import importlib.util
import json
import math
import multiprocessing as mp
import os
import time
from collections import Counter
from datetime import datetime, timezone
from itertools import groupby
from pathlib import Path

import numpy as np


CSLR_ROOT = Path(__file__).resolve().parents[1]
ORACLE_PATH = CSLR_ROOT / "tools/analyze_phoenix_decoder_utility_oracle.py"
SPEC = importlib.util.spec_from_file_location("decoder_utility_oracle", ORACLE_PATH)
ORACLE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(ORACLE)
REPLAY = ORACLE.REPLAY
RANDOM = ORACLE.RANDOM

SCHEMA_VERSION = "phoenix_decoder_utility_states_v1"
FROZEN_COMMIT = "3540ac7cbf0c62fbdd1c6b5eb8f25f9fd0c97d26"
DEFAULT_OUTPUT_ROOT = (
    CSLR_ROOT
    / "results/phoenix-2014t_ISLR/p3_counterfactual_utility_dataset_v1_49faacc3"
)
DEFAULT_PAIR_DIAGNOSTIC = (
    CSLR_ROOT
    / "results/phoenix-2014t_ISLR/p3_same_domain_pair_diagnostic_v1_49faacc3"
    / "per_sample_results.jsonl"
)
FORBIDDEN_PREDICTOR_KEYS = {
    "reference",
    "gloss_reference",
    "utility",
    "error",
    "error_after",
    "logits",
    "probabilities",
    "future",
    "eos",
    "total_windows",
    "exact_budget",
    "decoder_hypothesis",
}

_LOGITS = None
_REFERENCES = None
_VOCAB = None
_BLANK_ID = None
_PAIR_AUXILIARY_NAMES = None


def reject_test_path(path):
    REPLAY.reject_test_path(path)


def selected_set_hash(starts):
    text = ",".join(str(int(value)) for value in starts)
    return hashlib.sha256(text.encode("ascii")).hexdigest()


def coverage_summary(selected, total_windows):
    selected = sorted(selected)
    if not selected:
        return {
            "selected_count": 0,
            "max_internal_gap": None,
            "mean_internal_gap": None,
            "left_endpoint_selected": False,
            "right_endpoint_selected": False,
        }
    gaps = np.diff(np.asarray(selected, dtype=np.int64))
    return {
        "selected_count": len(selected),
        "max_internal_gap": int(gaps.max()) if len(gaps) else 0,
        "mean_internal_gap": float(gaps.mean()) if len(gaps) else 0.0,
        "left_endpoint_selected": selected[0] == 0,
        "right_endpoint_selected": selected[-1] == total_windows - 1,
    }


def candidate_predictor_inputs(candidate, selected):
    """Features derivable from selections no later than this candidate time."""
    selected_past = [value for value in selected if value < candidate]
    last_selected = selected_past[-1] if selected_past else None
    return {
        "candidate_window_start": int(candidate),
        "selected_past_count": len(selected_past),
        "frames_since_last_selected_past": (
            int(candidate - last_selected) if last_selected is not None else int(candidate + 1)
        ),
        "has_selected_past": last_selected is not None,
    }


def assert_predictor_inputs_clean(value, path="predictor_inputs"):
    if isinstance(value, dict):
        for key, child in value.items():
            lowered = key.lower()
            if lowered in FORBIDDEN_PREDICTOR_KEYS or any(
                token in lowered for token in ("reference", "logit", "future", "utility", "error")
            ):
                raise ValueError(f"oracle information leaked into {path}.{key}")
            assert_predictor_inputs_clean(child, f"{path}.{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            assert_predictor_inputs_clean(child, f"{path}[{index}]")


def utility_dense_rank(utilities):
    unique = sorted(set(utilities), reverse=True)
    return {value: index + 1 for index, value in enumerate(unique)}


def state_record(
    name,
    probabilities,
    reference,
    initial_skeleton,
    selected_bonus,
    decision_step,
    total_budget,
    vocab,
    blank_id,
    pair_auxiliary_available,
):
    selected = sorted([*initial_skeleton, *selected_bonus])
    candidates = [index for index in range(len(probabilities)) if index not in set(selected)]
    current_error, ranked = ORACLE.rank_one_step_candidates(
        probabilities, reference, selected, candidates, vocab, blank_id
    )
    current_hypothesis = ORACLE.decode_probabilities(
        probabilities, selected, vocab, blank_id
    )
    current_counts = RANDOM.sentence_counts(reference, current_hypothesis)
    utilities = [row["utility"] for row in ranked]
    ranks = utility_dense_rank(utilities)
    tie_sizes = Counter(utilities)
    candidate_records = []
    for selection_order, row in enumerate(ranked, start=1):
        candidate = row["window"]
        tie_distance, tie_earlier = ORACLE.coverage_tie_key(
            candidate, selected, len(probabilities)
        )
        predictor_inputs = candidate_predictor_inputs(candidate, selected)
        assert_predictor_inputs_clean(predictor_inputs)
        candidate_records.append(
            {
                "candidate_window_start": int(candidate),
                "predictor_inputs": predictor_inputs,
                "label": {
                    "utility": int(row["utility"]),
                    "error_after": int(row["error_after"]),
                    "utility_dense_rank": int(ranks[row["utility"]]),
                    "utility_tie_size": int(tie_sizes[row["utility"]]),
                    "greedy_selection_order": selection_order,
                    "coverage_tiebreak_distance": int(tie_distance),
                    "earlier_index_tiebreak": int(tie_earlier),
                    "selected_by_greedy": selection_order == 1,
                },
            }
        )
    all_zero = all(value == 0 for value in utilities)
    state_id = f"{hashlib.sha256(name.encode('utf-8')).hexdigest()[:16]}:{decision_step:04d}"
    record = {
        "schema_version": SCHEMA_VERSION,
        "state_id": state_id,
        "sample_id": name,
        "decision_step": decision_step,
        "oracle_state": {
            "initial_skeleton": initial_skeleton,
            "selected_bonus_before": selected_bonus,
            "selected_set_sha256": selected_set_hash(selected),
            "selected_count": len(selected),
            "exact_total_budget_known_only_offline": total_budget,
            "exact_bonus_budget_known_only_offline": total_budget - len(initial_skeleton),
            "exact_bonus_slots_remaining_known_only_offline": total_budget - len(selected),
            "token_bucket_balance": None,
            "token_bucket_note": "not fabricated: exact per-sample oracle budget requires known EOS",
            "coverage": coverage_summary(selected, len(probabilities)),
        },
        "oracle_decoder_context": {
            "reference": reference,
            "current_hypothesis": current_hypothesis,
            "current_error": current_error,
            "current_error_counts": current_counts,
        },
        "candidate_count": len(candidate_records),
        "candidates": candidate_records,
        "auxiliary_two_step": {
            "eligible_all_single_utilities_zero": all_zero,
            "primary_label_remains_single_step": True,
            "existing_same_domain_pair_record": (
                {
                    "artifact": str(DEFAULT_PAIR_DIAGNOSTIC.resolve()),
                    "sample_id": name,
                }
                if decision_step == 0 and all_zero and pair_auxiliary_available
                else None
            ),
        },
        "provenance": {
            "offline_reference_aware": True,
            "candidate_expensive_logits_used_only_to_create_labels": True,
            "candidate_logits_serialized": False,
            "future_frames_allowed_as_predictor_inputs": False,
            "oracle_set_building_step_is_not_a_streaming_timestamp": True,
        },
    }
    return record, ranked[0]["window"]


def build_sample(name):
    probabilities = ORACLE.softmax_rows(_LOGITS[name])
    total_windows = len(probabilities)
    total_budget = ORACLE.dense_half_budget(total_windows)
    skeleton_count = ORACLE.skeleton_size(total_budget, ORACLE.PRIMARY_RATIO)
    initial_skeleton = ORACLE.uniform_positions(total_windows, skeleton_count)
    selected_bonus = []
    states = []
    pair_available = name in _PAIR_AUXILIARY_NAMES
    while len(initial_skeleton) + len(selected_bonus) < total_budget:
        record, choice = state_record(
            name,
            probabilities,
            _REFERENCES[name],
            initial_skeleton,
            list(selected_bonus),
            len(selected_bonus),
            total_budget,
            _VOCAB,
            _BLANK_ID,
            pair_available,
        )
        states.append(record)
        selected_bonus.append(int(choice))
    return {
        "sample_id": name,
        "dense_windows": total_windows,
        "total_budget": total_budget,
        "initial_skeleton": initial_skeleton,
        "final_selected_bonus": selected_bonus,
        "states": states,
    }


def load_pair_auxiliary_names(path):
    if not path.is_file():
        return set()
    reject_test_path(path)
    names = set()
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            names.add(json.loads(line)["name"])
    return names


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def estimate_size(logits_by_name):
    states = candidates = 0
    for logits in logits_by_name.values():
        total = len(logits)
        budget = ORACLE.dense_half_budget(total)
        skeleton = ORACLE.skeleton_size(budget, ORACLE.PRIMARY_RATIO)
        bonus = budget - skeleton
        states += bonus
        candidates += sum(total - skeleton - step for step in range(bonus))
    return states, candidates


def update_statistics(statistics, sample):
    statistics["samples"] += 1
    statistics["states"] += len(sample["states"])
    for state in sample["states"]:
        statistics["candidate_records"] += state["candidate_count"]
        statistics["all_zero_states"] += int(
            state["auxiliary_two_step"]["eligible_all_single_utilities_zero"]
        )
        for candidate in state["candidates"]:
            utility = candidate["label"]["utility"]
            statistics["utility_counts"][str(utility)] += 1
            statistics["positive_utility_candidates"] += int(utility > 0)
            statistics["negative_utility_candidates"] += int(utility < 0)


def write_shard(path, samples):
    with gzip.open(path, "wt", encoding="utf-8", compresslevel=6) as handle:
        for sample in samples:
            for state in sample["states"]:
                handle.write(json.dumps(state, ensure_ascii=False, separators=(",", ":")) + "\n")


def build_dataset(
    matrix_root,
    vocab_path,
    output_root,
    pair_diagnostic_path,
    max_samples=None,
    shard_samples=32,
    workers=4,
):
    reject_test_path(matrix_root)
    reject_test_path(vocab_path)
    reject_test_path(output_root)
    (
        all_names,
        b0_results,
        b0_logits,
        start_indices,
        _a0_results,
        _b2_results,
        vocab,
        input_paths,
    ) = RANDOM.load_inputs(matrix_root, vocab_path, max_samples=max_samples)
    names = list(all_names)
    references = {
        name: REPLAY.clean_phoenix_2014_trans(b0_results[name]["gls_ref"]) for name in names
    }
    for name in names:
        if list(start_indices[name]) != list(range(len(b0_logits[name]))):
            raise ValueError(f"non-dense candidate coordinates: {name}")
    expected_states, expected_candidates = estimate_size(b0_logits)
    pair_names = load_pair_auxiliary_names(pair_diagnostic_path)

    global _LOGITS, _REFERENCES, _VOCAB, _BLANK_ID, _PAIR_AUXILIARY_NAMES
    _LOGITS = b0_logits
    _REFERENCES = references
    _VOCAB = vocab
    _BLANK_ID = vocab.index("<blank>")
    _PAIR_AUXILIARY_NAMES = pair_names

    if output_root.exists():
        raise FileExistsError(f"refusing to overwrite existing output: {output_root}")
    temporary_root = output_root.with_name(f".{output_root.name}.incomplete-{os.getpid()}")
    if temporary_root.exists():
        raise FileExistsError(f"temporary output already exists: {temporary_root}")
    shards_root = temporary_root / "shards"
    shards_root.mkdir(parents=True)
    shard_count = int(math.ceil(len(names) / shard_samples))
    statistics = {
        "samples": 0,
        "states": 0,
        "candidate_records": 0,
        "all_zero_states": 0,
        "positive_utility_candidates": 0,
        "negative_utility_candidates": 0,
        "utility_counts": Counter(),
    }
    shard_records = []
    started = time.perf_counter()
    with mp.get_context("fork").Pool(processes=workers) as pool:
        iterator = pool.imap(build_sample, names, chunksize=1)
        pending = []
        for index, sample in enumerate(iterator, start=1):
            pending.append(sample)
            update_statistics(statistics, sample)
            if len(pending) == shard_samples or index == len(names):
                shard_index = len(shard_records)
                filename = f"states-{shard_index:05d}-of-{shard_count:05d}.jsonl.gz"
                path = shards_root / filename
                write_shard(path, pending)
                shard_records.append(
                    {
                        "path": f"shards/{filename}",
                        "samples": len(pending),
                        "states": sum(len(item["states"]) for item in pending),
                        "bytes": path.stat().st_size,
                        "sha256": sha256_file(path),
                    }
                )
                pending = []
            print(f"built {index}/{len(names)}: {sample['sample_id']}", flush=True)
    elapsed = time.perf_counter() - started
    statistics["utility_counts"] = dict(
        sorted(statistics["utility_counts"].items(), key=lambda item: int(item[0]))
    )
    if statistics["states"] != expected_states or statistics["candidate_records"] != expected_candidates:
        raise AssertionError("written dataset size differs from analytic estimate")

    schema = {
        "schema_version": SCHEMA_VERSION,
        "record_unit": "one oracle set-building decision state",
        "primary_label": "candidates[].label.utility = current edit errors - errors after adding candidate",
        "predictor_input_contract": {
            "allowed_location": "candidates[].predictor_inputs only",
            "excluded": sorted(FORBIDDEN_PREDICTOR_KEYS),
            "warning": "oracle trajectory is not chronological streaming; causal pose/prefix features must be joined later under an audited time index",
        },
        "state_reconstruction": "initial_skeleton union selected_bonus_before; verify selected_set_sha256",
    }
    (temporary_root / "schema.json").write_text(
        json.dumps(schema, indent=2) + "\n", encoding="utf-8"
    )
    summary = {
        "schema_version": SCHEMA_VERSION,
        "frozen_git_commit": FROZEN_COMMIT,
        "split": "dev",
        "oracle_only": True,
        "model_forward_run": False,
        "samples": len(names),
        "expected_states": expected_states,
        "expected_candidate_records": expected_candidates,
        "statistics": statistics,
        "runtime": {
            "workers": workers,
            "wall_seconds": elapsed,
            "compressed_shard_bytes": sum(row["bytes"] for row in shard_records),
        },
        "shards": shard_records,
    }
    (temporary_root / "dataset_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    config = {
        "matrix_root": str(matrix_root.resolve()),
        "vocab": str(vocab_path.resolve()),
        "output_root": str(output_root.resolve()),
        "pair_diagnostic": str(pair_diagnostic_path.resolve()),
        "max_samples": max_samples,
        "shard_samples": shard_samples,
        "workers": workers,
        "total_budget_rate": 0.5,
        "uniform_skeleton_share": 0.5,
        "decoder": "span15",
    }
    (temporary_root / "resolved_config.json").write_text(
        json.dumps(config, indent=2) + "\n", encoding="utf-8"
    )
    manifest_inputs = sorted(
        set([Path(__file__).resolve(), ORACLE_PATH, *input_paths]
            + ([pair_diagnostic_path] if pair_diagnostic_path.is_file() else []))
    )
    manifest = {
        "manifest_version": 1,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "frozen_git_commit": FROZEN_COMMIT,
        "inputs": {
            str(path.resolve()): {"bytes": path.stat().st_size, "sha256": sha256_file(path)}
            for path in manifest_inputs
        },
        "outputs": {
            "schema.json": {"sha256": sha256_file(temporary_root / "schema.json")},
            "dataset_summary.json": {
                "sha256": sha256_file(temporary_root / "dataset_summary.json")
            },
            "resolved_config.json": {
                "sha256": sha256_file(temporary_root / "resolved_config.json")
            },
            "shards": shard_records,
        },
    }
    (temporary_root / "dataset_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    temporary_root.rename(output_root)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matrix-root", type=Path, default=REPLAY.DEFAULT_MATRIX_ROOT)
    parser.add_argument("--vocab", type=Path, default=REPLAY.DEFAULT_VOCAB)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--pair-diagnostic", type=Path, default=DEFAULT_PAIR_DIAGNOSTIC)
    parser.add_argument("--max-samples", type=int)
    parser.add_argument("--shard-samples", type=int, default=32)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    if args.max_samples is not None and args.max_samples < 1:
        parser.error("max-samples must be positive")
    if args.shard_samples < 1 or args.workers < 1:
        parser.error("shard-samples and workers must be positive")
    summary = build_dataset(
        args.matrix_root.resolve(),
        args.vocab.resolve(),
        args.output_root.resolve(),
        args.pair_diagnostic.resolve(),
        max_samples=args.max_samples,
        shard_samples=args.shard_samples,
        workers=args.workers,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
