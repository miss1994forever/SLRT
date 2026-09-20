#!/usr/bin/env python3
"""CPU-only chronological decoder-utility oracle on partial Phoenix train.

This is an offline, reference-aware smoke diagnostic.  The scheduler advances
in window-start order and its state is causal, but the oracle labels may inspect
the reference and (for rollout advantage) future replay logits.  Consequently
this tool neither trains a predictor nor reports a deployable online result.
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

import numpy as np


ROOT = Path(__file__).resolve().parents[1]


def import_tool(name):
    path = ROOT / f"tools/{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


BUILDER = import_tool("build_phoenix_train_counterfactual_utility_dataset")
ORACLE = BUILDER.ORACLE
REPLAY = BUILDER.REPLAY
RANDOM = ORACLE.RANDOM

DEFAULT_DENSE_ROOT = (
    ROOT / "results/phoenix-2014t_ISLR/train_dense_stride1_v1_49faacc3"
)
DEFAULT_VOCAB = BUILDER.DEFAULT_VOCAB
DEFAULT_OUTPUT_ROOT = (
    ROOT / "results/phoenix-2014t_ISLR/"
    "p3_partial32_chronological_oracle_smoke_v1_49faacc3"
)

WINDOW_FRAMES = 16
PAD_LEFT = 7
LOOKAHEAD_FRAMES = 8
SKELETON_PERIOD = 4
BONUS_ACCRUAL_NON_SKELETON = 1.0 / 3.0
TOKEN_CAPACITY = 2.0
ROLLOUT_HORIZON = 16
EPSILON = 1e-9
BOOTSTRAP_SEED = 260923

_RESULTS = None
_LOGITS = None
_VOCAB = None
_BLANK_ID = None


def reject_test_path(path):
    BUILDER.reject_test_path(path)


def completed_shard_indices(dense_root):
    protocol = json.loads((dense_root / "protocol_manifest.json").read_text(encoding="utf-8"))
    shard_count = int(math.ceil(protocol["samples"] / protocol["shard_samples"]))
    values = []
    for index in range(shard_count):
        _, _, _, complete = BUILDER.dense_shard_paths(dense_root, index, shard_count)
        if complete.is_file():
            values.append(index)
    return values, shard_count


def window_input_coordinates(start, total_frames):
    """Return original-frame coordinates represented by frozen dense start.

    Dense generation always has starts 0..T-1, hence total padding is 15 and
    is split as seven repeated frames on the left and eight on the right.
    """
    start = int(start)
    total_frames = int(total_frames)
    if not 0 <= start < total_frames:
        raise ValueError("window start outside dense coordinates")
    raw_left = start - PAD_LEFT
    raw_right = raw_left + WINDOW_FRAMES - 1
    return {
        "unclamped_original_range": [raw_left, raw_right],
        "clamped_original_range": [max(0, raw_left), min(total_frames - 1, raw_right)],
        "left_repeats": max(0, -raw_left),
        "right_repeats": max(0, raw_right - (total_frames - 1)),
        "normal_available_frame": raw_right if raw_right < total_frames else None,
        "requires_eos_flush": raw_right >= total_frames,
    }


def is_skeleton(start):
    return int(start) % SKELETON_PERIOD == 0


def accrue_token(balance):
    return min(TOKEN_CAPACITY, float(balance) + BONUS_ACCRUAL_NON_SKELETON)


def token_eligible(balance):
    return float(balance) + EPSILON >= 1.0


def token_forced(balance):
    return float(balance) + EPSILON >= TOKEN_CAPACITY


def prefix_state(probabilities, selected, candidate):
    """Deployment-eligible state; candidate/future logits are never read."""
    selected = sorted(int(value) for value in selected if int(value) < int(candidate))
    starts, raw_scores, token_ids = ORACLE.decoder_state(probabilities, selected)
    if len(raw_scores):
        normalized = raw_scores / np.maximum(raw_scores.sum(axis=1, keepdims=True), 1e-12)
        entropy = -np.sum(normalized * np.log(np.maximum(normalized, 1e-12)), axis=1)
        mean_entropy = float(entropy.mean())
        last_entropy = float(entropy[-1])
    else:
        mean_entropy = last_entropy = None
    last = selected[-1] if selected else None
    return {
        "candidate_window_start": int(candidate),
        "selected_past_count": len(selected),
        "frames_since_last_selected_past": (
            int(candidate - last) if last is not None else int(candidate + 1)
        ),
        "has_selected_past": last is not None,
        "coverage_gap_before_current": (
            int(candidate - last) if last is not None else int(candidate + 1)
        ),
        "decoder_prefix_token_ids": [int(value) for value in token_ids],
        "decoder_prefix_length": len(token_ids),
        "decoder_prefix_mean_entropy": mean_entropy,
        "decoder_prefix_last_entropy": last_entropy,
        "prefix_selected_starts_sha256": hashlib.sha256(
            ",".join(str(value) for value in starts).encode("ascii")
        ).hexdigest(),
    }


def decode_counts(probabilities, selected, reference, vocab, blank_id):
    hypothesis = ORACLE.decode_probabilities(probabilities, selected, vocab, blank_id)
    counts = RANDOM.sentence_counts(reference, hypothesis)
    return hypothesis, {key: int(counts[key]) for key in ("error", "del", "ins", "sub", "ref_len")}


def immediate_utility(probabilities, selected, candidate, reference, vocab, blank_id):
    starts, raw_scores, token_ids = ORACLE.decoder_state(probabilities, selected)
    before = ORACLE.error_count(reference, ORACLE.hypothesis_from_token_ids(token_ids, vocab, blank_id))
    after_hypothesis = ORACLE.candidate_hypothesis(
        probabilities, starts, raw_scores, token_ids, candidate, vocab, blank_id
    )
    after = ORACLE.error_count(reference, after_hypothesis)
    return int(before - after), int(before), int(after)


def eager_future(selected, balance, current, total_windows, execute_current, horizon):
    """Apply one branch action, then the same fixed eager online policy."""
    chosen = list(selected)
    value = float(balance)
    if execute_current:
        chosen.append(int(current))
        value -= 1.0
    stop = min(total_windows, int(current) + 1 + int(horizon))
    for start in range(int(current) + 1, stop):
        if is_skeleton(start):
            chosen.append(start)
            continue
        value = accrue_token(value)
        if token_eligible(value):
            chosen.append(start)
            value -= 1.0
    return sorted(chosen), value


def rollout_advantage(probabilities, selected, balance, current, reference, vocab, blank_id,
                      horizon=ROLLOUT_HORIZON):
    execute, _ = eager_future(
        selected, balance, current, len(probabilities), True, horizon
    )
    skip, _ = eager_future(
        selected, balance, current, len(probabilities), False, horizon
    )
    execute_hypothesis = ORACLE.decode_probabilities(
        probabilities, execute, vocab, blank_id
    )
    skip_hypothesis = ORACLE.decode_probabilities(probabilities, skip, vocab, blank_id)
    execute_error = ORACLE.error_count(reference, execute_hypothesis)
    skip_error = ORACLE.error_count(reference, skip_hypothesis)
    return int(skip_error - execute_error), int(skip_error), int(execute_error)


def coverage_metrics(selected, total_windows):
    selected = sorted(int(value) for value in selected)
    if not selected:
        return {"max_gap_including_endpoints": total_windows, "max_internal_gap": None}
    internal = np.diff(np.asarray(selected, dtype=np.int64)).tolist()
    endpoint_gaps = [selected[0] + 1, total_windows - selected[-1]]
    return {
        "max_gap_including_endpoints": int(max(endpoint_gaps + internal)),
        "max_internal_gap": int(max(internal)) if internal else 0,
        "left_endpoint_selected": selected[0] == 0,
        "right_endpoint_selected": selected[-1] == total_windows - 1,
    }


def online_uniform_schedule(total_windows):
    selected = []
    balance = 0.0
    for start in range(total_windows):
        if is_skeleton(start):
            selected.append(start)
        else:
            balance = accrue_token(balance)
            if token_eligible(balance):
                selected.append(start)
                balance -= 1.0
    return selected, balance


def run_policy(probabilities, reference, vocab, blank_id, policy):
    if policy not in {"myopic", "rollout16"}:
        raise ValueError(policy)
    selected = []
    balance = 0.0
    trace = []
    utility_counts = Counter()
    rollout_counts = Counter()
    for start in range(len(probabilities)):
        if is_skeleton(start):
            selected.append(start)
            continue
        balance = accrue_token(balance)
        if not token_eligible(balance):
            continue
        state = prefix_state(probabilities, selected, start)
        state["bonus_token_balance"] = float(balance)
        state["bonus_token_capacity"] = TOKEN_CAPACITY
        state["bonus_forced_by_capacity"] = token_forced(balance)
        immediate, before_error, immediate_error = immediate_utility(
            probabilities, selected, start, reference, vocab, blank_id
        )
        utility_counts[str(immediate)] += 1
        rollout = rollout_skip_error = rollout_execute_error = None
        if policy == "rollout16":
            rollout, rollout_skip_error, rollout_execute_error = rollout_advantage(
                probabilities, selected, balance, start, reference, vocab, blank_id
            )
            rollout_counts[str(rollout)] += 1
            score = rollout
        else:
            score = immediate
        forced = token_forced(balance)
        execute = bool(forced or score > 0)
        trace.append({
            "candidate_window_start": start,
            "predictor_inputs": state,
            "offline_reference_aware_labels": {
                "immediate_edit_distance_utility": immediate,
                "current_error": before_error,
                "execute_current_error": immediate_error,
                "rollout16_advantage": rollout,
                "rollout16_skip_error": rollout_skip_error,
                "rollout16_execute_error": rollout_execute_error,
            },
            "oracle_action_execute": execute,
            "forced_by_token_capacity": forced,
            "provenance": {
                "state_uses_only_past_executed_logits": True,
                "reference_used_only_for_label_and_oracle_action": True,
                "future_logits_used_by_rollout_label_only": policy == "rollout16",
            },
        })
        if execute:
            selected.append(start)
            balance -= 1.0
    selected.sort()
    hypothesis, counts = decode_counts(probabilities, selected, reference, vocab, blank_id)
    return {
        "selected": selected,
        "unspent_bonus_tokens": float(balance),
        "trace": trace,
        "immediate_utility_counts": dict(sorted(utility_counts.items(), key=lambda row: int(row[0]))),
        "rollout16_advantage_counts": dict(sorted(rollout_counts.items(), key=lambda row: int(row[0]))),
        "hypothesis": hypothesis,
        "counts": counts,
        "coverage": coverage_metrics(selected, len(probabilities)),
    }


def matched_count_uniform(total_windows, count):
    # Explicitly offline: matching K and endpoint coverage requires EOS/T.
    return ORACLE.uniform_positions(total_windows, count)


def process_sample(name):
    result = _RESULTS[name]
    logits = np.asarray(_LOGITS[name])
    probabilities = ORACLE.softmax_rows(logits)
    reference = REPLAY.clean_phoenix_2014_trans(result["gls_ref"])
    starts = [int(row["start"]) for row in result["adaptive_stride_metadata"]]
    if starts != list(range(len(logits))):
        raise ValueError(f"non-dense coordinates: {name}")
    uniform_selected, uniform_balance = online_uniform_schedule(len(logits))
    uniform_hypothesis, uniform_counts = decode_counts(
        probabilities, uniform_selected, reference, _VOCAB, _BLANK_ID
    )
    policies = {}
    for policy in ("myopic", "rollout16"):
        output = run_policy(probabilities, reference, _VOCAB, _BLANK_ID, policy)
        matched = matched_count_uniform(len(logits), len(output["selected"]))
        matched_hypothesis, matched_counts = decode_counts(
            probabilities, matched, reference, _VOCAB, _BLANK_ID
        )
        output["matched_actual_count_uniform"] = {
            "selected": matched,
            "known_eos_diagnostic_only": True,
            "hypothesis": matched_hypothesis,
            "counts": matched_counts,
            "coverage": coverage_metrics(matched, len(logits)),
        }
        policies[policy] = output
    coordinate_audit = {
        "first": window_input_coordinates(0, len(logits)),
        "last_normal": window_input_coordinates(max(0, len(logits) - 9), len(logits)),
        "first_eos_flush": window_input_coordinates(max(0, len(logits) - 8), len(logits)),
        "last": window_input_coordinates(len(logits) - 1, len(logits)),
    }
    return {
        "name": name,
        "reference": reference,
        "dense_windows": len(logits),
        "coordinate_audit": coordinate_audit,
        "online_uniform": {
            "selected": uniform_selected,
            "unspent_bonus_tokens": float(uniform_balance),
            "hypothesis": uniform_hypothesis,
            "counts": uniform_counts,
            "coverage": coverage_metrics(uniform_selected, len(logits)),
        },
        "policies": policies,
    }


def aggregate_counts(rows):
    total = Counter()
    for row in rows:
        total.update(row)
    return {
        "wer": 100.0 * total["error"] / total["ref_len"],
        **{key: int(total[key]) for key in ("error", "del", "ins", "sub", "ref_len")},
    }


def paired_bootstrap(candidate, baseline, repetitions=2000, seed=BOOTSTRAP_SEED):
    candidate_error = np.asarray([row["error"] for row in candidate], dtype=np.int64)
    baseline_error = np.asarray([row["error"] for row in baseline], dtype=np.int64)
    ref_len = np.asarray([row["ref_len"] for row in candidate], dtype=np.int64)
    if not np.array_equal(ref_len, np.asarray([row["ref_len"] for row in baseline])):
        raise ValueError("paired reference lengths differ")
    rng = np.random.default_rng(seed)
    values = np.empty(repetitions, dtype=np.float64)
    for index in range(repetitions):
        draw = rng.integers(0, len(ref_len), len(ref_len))
        values[index] = 100.0 * (
            candidate_error[draw].sum() - baseline_error[draw].sum()
        ) / ref_len[draw].sum()
    return {
        "unit": "WER percentage points (candidate - baseline)",
        "seed": seed,
        "repetitions": repetitions,
        "ci95": [float(value) for value in np.quantile(values, [0.025, 0.975])],
        "mean": float(values.mean()),
    }


def add_distribution(target, values):
    for key, value in values.items():
        target[str(key)] += int(value)


def signed_distribution(values):
    return {
        "negative": int(sum(count for utility, count in values.items() if int(utility) < 0)),
        "zero": int(values.get("0", 0)),
        "positive": int(sum(count for utility, count in values.items() if int(utility) > 0)),
    }


def trace_action_counts(trace_path):
    decision = Counter()
    forced = Counter()
    with gzip.open(trace_path, "rt", encoding="utf-8") as handle:
        for line in handle:
            record = json.loads(line)
            policy = record["policy_trajectory"]
            decision[policy] += 1
            forced[policy] += int(record["forced_by_token_capacity"])
    return {"decision": decision, "forced": forced}


def summarize(samples, shard_indices, shard_count, elapsed, trace_counts=None):
    uniform_rows = [row["online_uniform"]["counts"] for row in samples]
    summary = {
        "status": "complete",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "scope": {
            "split": "train_partial32_frozen_calibration_nonrandom",
            "samples": len(samples),
            "dense_shards": shard_indices,
            "completed_dense_shards": len(shard_indices),
            "full_dense_shards": shard_count,
            "gpu_used": False,
            "dev_used": False,
            "test_used": False,
        },
        "frozen_protocol": {
            "window_frames": WINDOW_FRAMES,
            "dense_coordinate_semantics": "start=s maps to repeated-edge original frames [s-7,s+8]",
            "lookahead_frames": LOOKAHEAD_FRAMES,
            "tail_windows": "last 8 become available only at EOS flush; EOS is not a state feature and no token top-up occurs",
            "skeleton": "execute every start divisible by 4 (25% causal coverage target)",
            "bonus_bucket": {
                "initial_tokens": 0.0,
                "accrual": "1/3 token at each non-skeleton arrival",
                "capacity": TOKEN_CAPACITY,
                "eligible_at": 1.0,
                "forced_execute_at": TOKEN_CAPACITY,
                "eos_flush_unspent_tokens": False,
            },
            "target_long_run_total_rate": 0.5,
            "exact_half_per_sample_guaranteed": False,
            "exact_half_note": "unknown EOS makes exact 0.5T impossible without tail correction",
            "rollout_horizon_candidates": ROLLOUT_HORIZON,
            "rollout_future_policy": "same fixed eager token-bucket policy in execute and skip branches",
        },
        "online_uniform": {
            "metrics": aggregate_counts(uniform_rows),
            "executed_windows": int(sum(len(row["online_uniform"]["selected"]) for row in samples)),
            "dense_windows": int(sum(row["dense_windows"] for row in samples)),
            "per_sample_rate": {
                "mean": float(np.mean([len(row["online_uniform"]["selected"]) / row["dense_windows"] for row in samples])),
                "min": float(min(len(row["online_uniform"]["selected"]) / row["dense_windows"] for row in samples)),
                "max": float(max(len(row["online_uniform"]["selected"]) / row["dense_windows"] for row in samples)),
            },
            "max_gap": int(max(row["online_uniform"]["coverage"]["max_gap_including_endpoints"] for row in samples)),
            "unspent_tokens_total": float(sum(row["online_uniform"]["unspent_bonus_tokens"] for row in samples)),
        },
        "policies": {},
        "elapsed_seconds": float(elapsed),
        "limitations": [
            "partial32 calibration is source-video-disjoint from fit but remains a non-random incomplete train subset",
            "reference is used to form oracle labels/actions",
            "known future dense replay is used only by rollout labels/actions, never causal state",
            "matched-count uniform knows final T and is diagnostic only",
            "no predictor is trained",
            "no dev/test, formal confidence, real wall-time, or submission-latency conclusion",
        ],
    }
    for policy in ("myopic", "rollout16"):
        outputs = [row["policies"][policy] for row in samples]
        rows = [value["counts"] for value in outputs]
        matched_rows = [value["matched_actual_count_uniform"]["counts"] for value in outputs]
        immediate_distribution = Counter()
        rollout_distribution = Counter()
        for value in outputs:
            add_distribution(immediate_distribution, value["immediate_utility_counts"])
            add_distribution(rollout_distribution, value["rollout16_advantage_counts"])
        metrics = aggregate_counts(rows)
        matched_metrics = aggregate_counts(matched_rows)
        summary["policies"][policy] = {
            "decision_states": int(sum(
                sum(int(count) for count in value["immediate_utility_counts"].values())
                for value in outputs
            )),
            "immediate_utility_counts": dict(sorted(immediate_distribution.items(), key=lambda row: int(row[0]))),
            "immediate_utility_signs": signed_distribution(immediate_distribution),
            "rollout16_advantage_counts": dict(sorted(rollout_distribution.items(), key=lambda row: int(row[0]))),
            "rollout16_advantage_signs": signed_distribution(rollout_distribution),
            "metrics": metrics,
            "delta_wer_pp_vs_online_uniform": metrics["wer"] - summary["online_uniform"]["metrics"]["wer"],
            "error_delta_vs_online_uniform": metrics["error"] - summary["online_uniform"]["metrics"]["error"],
            "bootstrap_vs_online_uniform": paired_bootstrap(rows, uniform_rows),
            "matched_actual_count_uniform": {
                "metrics": matched_metrics,
                "delta_wer_pp": metrics["wer"] - matched_metrics["wer"],
                "error_delta": metrics["error"] - matched_metrics["error"],
                "bootstrap": paired_bootstrap(rows, matched_rows),
                "known_eos_diagnostic_only": True,
            },
            "executed_windows": int(sum(len(value["selected"]) for value in outputs)),
            "dense_windows": summary["online_uniform"]["dense_windows"],
            "per_sample_rate": {
                "mean": float(np.mean([len(value["selected"]) / sample["dense_windows"] for value, sample in zip(outputs, samples)])),
                "min": float(min(len(value["selected"]) / sample["dense_windows"] for value, sample in zip(outputs, samples))),
                "max": float(max(len(value["selected"]) / sample["dense_windows"] for value, sample in zip(outputs, samples))),
            },
            "max_gap": int(max(value["coverage"]["max_gap_including_endpoints"] for value in outputs)),
            "unspent_tokens_total": float(sum(value["unspent_bonus_tokens"] for value in outputs)),
            "forced_actions": (
                int(trace_counts["forced"][policy]) if trace_counts is not None else None
            ),
        }
        if trace_counts is not None and (
            trace_counts["decision"][policy]
            != summary["policies"][policy]["decision_states"]
        ):
            raise AssertionError(f"trace/state count mismatch for {policy}")
    return summary


def decision_from_summary(summary):
    def clear(policy):
        result = summary["policies"][policy]
        return (
            result["delta_wer_pp_vs_online_uniform"] <= -0.5
            and result["bootstrap_vs_online_uniform"]["ci95"][1] < 0.0
        )

    if clear("myopic"):
        return (
            "myopic chronological oracle clears the predeclared smoke threshold "
            "(at least 0.5 pp and descriptive CI below zero); a new chronological predictor is warranted"
        )
    if clear("rollout16"):
        return (
            "only the future-aware rollout ceiling clears the smoke threshold; this diagnoses long-horizon "
            "credit assignment but does not establish that a causal predictor can recover the advantage"
        )
    return (
        "no chronological oracle clears the predeclared smoke threshold; stop this utility/budget/decoder "
        "definition unless a separately declared longer-horizon or block ceiling is justified"
    )


def run(dense_root, vocab_path, output_root, workers=4, bootstrap_repetitions=2000):
    for path in (dense_root, vocab_path, output_root):
        reject_test_path(path)
    if output_root.exists():
        raise FileExistsError(f"refusing to overwrite {output_root}")
    shard_indices, shard_count = completed_shard_indices(dense_root)
    if len(shard_indices) != 32:
        raise ValueError(f"expected frozen partial32, found {len(shard_indices)} completed shards")
    vocab = json.loads(vocab_path.read_text(encoding="utf-8"))
    blank_id = vocab.index("<blank>")
    split_manifest = json.loads(
        (dense_root / "fit_calibration_split.json").read_text(encoding="utf-8")
    )
    assignments = split_manifest["assignments"]
    temporary = output_root.with_name(f".{output_root.name}.incomplete-{os.getpid()}")
    temporary.mkdir(parents=True)
    trace_path = temporary / "chronological_traces.jsonl.gz"
    per_sample_path = temporary / "per_sample_results.jsonl"
    started = time.perf_counter()
    all_samples = []
    expected_calibration = 0
    with gzip.open(trace_path, "wt", encoding="utf-8", compresslevel=6) as trace_handle, per_sample_path.open("w", encoding="utf-8") as sample_handle:
        processed = 0
        for shard_index in shard_indices:
            results, logits, _ = BUILDER.validate_dense_shard(
                dense_root, shard_index, shard_count, verify_hashes=False
            )
            names = [name for name in results if assignments[name] == "calibration"]
            expected_calibration += len(names)
            results = {name: results[name] for name in names}
            logits = {name: logits[name] for name in names}
            global _RESULTS, _LOGITS, _VOCAB, _BLANK_ID
            _RESULTS, _LOGITS, _VOCAB, _BLANK_ID = results, logits, vocab, blank_id
            if workers == 1:
                iterator = map(process_sample, names)
                pool = None
            else:
                pool = mp.get_context("fork").Pool(processes=workers)
                iterator = pool.imap(process_sample, names, chunksize=1)
            try:
                for sample in iterator:
                    compact = {
                        key: value for key, value in sample.items()
                        if key not in {"reference"}
                    }
                    for policy in compact["policies"].values():
                        trace = policy.pop("trace")
                        for state in trace:
                            trace_handle.write(json.dumps({
                                "sample_id": sample["name"],
                                "policy_trajectory": "myopic" if policy is compact["policies"]["myopic"] else "rollout16",
                                **state,
                            }, ensure_ascii=False, separators=(",", ":")) + "\n")
                    sample_handle.write(json.dumps(compact, ensure_ascii=False, separators=(",", ":")) + "\n")
                    all_samples.append(compact)
                    processed += 1
            finally:
                if pool is not None:
                    pool.close()
                    pool.join()
            print(json.dumps({"completed_shard": shard_index, "processed_samples": processed}), flush=True)
    elapsed = time.perf_counter() - started
    # The repetition argument is exposed only for provenance; summary helper uses
    # the frozen smoke default to avoid accidental per-arm differences.
    if bootstrap_repetitions != 2000:
        raise ValueError("smoke bootstrap repetitions are frozen at 2000")
    summary = summarize(
        all_samples, shard_indices, shard_count, elapsed,
        trace_counts=trace_action_counts(trace_path),
    )
    if len(all_samples) != expected_calibration:
        raise AssertionError("partial32 calibration coverage mismatch")
    summary["scope"]["frozen_partition"] = "calibration"
    summary["scope"]["fit_samples_used"] = 0
    summary["decision"] = decision_from_summary(summary)
    summary["artifacts"] = {
        "per_sample_results": str((output_root / per_sample_path.name).resolve()),
        "chronological_traces": str((output_root / trace_path.name).resolve()),
    }
    (temporary / "metrics.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    protocol = {
        "schema_version": 1,
        "tool": str(Path(__file__).resolve()),
        "inputs": {
            "dense_root": str(dense_root.resolve()),
            "vocab": str(vocab_path.resolve()),
        },
        "output_sha256": {
            path.name: BUILDER.sha256_file(path)
            for path in (temporary / "metrics.json", per_sample_path, trace_path)
        },
    }
    (temporary / "protocol_manifest.json").write_text(
        json.dumps(protocol, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    temporary.replace(output_root)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dense-root", type=Path, default=DEFAULT_DENSE_ROOT)
    parser.add_argument("--vocab", type=Path, default=DEFAULT_VOCAB)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--bootstrap-repetitions", type=int, default=2000)
    args = parser.parse_args()
    result = run(
        args.dense_root, args.vocab, args.output_root,
        workers=args.workers, bootstrap_repetitions=args.bootstrap_repetitions,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
