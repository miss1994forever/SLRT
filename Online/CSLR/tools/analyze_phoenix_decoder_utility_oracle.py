#!/usr/bin/env python3
"""Decoder-marginal-utility oracle on frozen Phoenix dev dense logits.

This is an offline upper-bound diagnostic, not a deployable scheduler.  It
uses the reference transcript while selecting windows and therefore must never
be reported as causal or as a wall-time result.
"""

import argparse
import hashlib
import importlib.util
import json
import math
import multiprocessing as mp
from datetime import datetime, timezone
from itertools import combinations, groupby
from pathlib import Path

import numpy as np


CSLR_ROOT = Path(__file__).resolve().parents[1]
RANDOM_PATH = CSLR_ROOT / "tools/analyze_phoenix_random_schedule.py"
SPEC = importlib.util.spec_from_file_location("random_schedule", RANDOM_PATH)
RANDOM = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(RANDOM)
REPLAY = RANDOM.REPLAY

DEFAULT_OUTPUT_ROOT = (
    CSLR_ROOT
    / "results/phoenix-2014t_ISLR/p3_decoder_utility_oracle_v1_49faacc3"
)
SKELETON_RATIOS = (0.0, 0.25, 0.5, 0.75, 1.0)
PRIMARY_RATIO = 0.5
_WORKER_LOGITS = None
_WORKER_REFERENCES = None
_WORKER_VOCAB = None
_WORKER_BLANK_ID = None


def dense_half_budget(total_windows):
    """Nearest-integer 50% budget, matching prior dense_50pct convention."""
    if total_windows < 1:
        raise ValueError("sample has no dense windows")
    return min(
        total_windows,
        max(2 if total_windows > 1 else 1, int(math.floor(0.5 * total_windows + 0.5))),
    )


def uniform_positions(total_windows, count):
    if count == 0:
        return []
    if not 1 <= count <= total_windows:
        raise ValueError("uniform count must be in [1, total_windows]")
    if count == 1:
        return [0]
    values = [
        int(math.floor(index * (total_windows - 1) / (count - 1) + 0.5))
        for index in range(count)
    ]
    if len(set(values)) != count or values[0] != 0 or values[-1] != total_windows - 1:
        raise AssertionError("uniform positions are not exact and endpoint-covering")
    return values


def skeleton_size(budget, ratio):
    if not 0.0 <= ratio <= 1.0:
        raise ValueError("skeleton ratio must be in [0, 1]")
    count = int(math.floor(budget * ratio + 0.5))
    if ratio > 0.0 and budget > 1:
        count = max(2, count)
    return min(budget, count)


def softmax_rows(logits):
    values = np.asarray(logits, dtype=np.float32)
    shifted = values - values.max(axis=1, keepdims=True)
    exponent = np.exp(shifted)
    return exponent / exponent.sum(axis=1, keepdims=True)


def decode_probabilities(probabilities, starts, vocab, blank_id, span=15.0, min_weight=0.05):
    """Numpy parity implementation of the frozen span-15 decoder."""
    starts = list(sorted(int(value) for value in starts))
    if not starts:
        return ""
    centers = np.asarray(starts, dtype=np.float32) + np.float32(7.5)
    radius = np.float32(span / 2.0)
    token_ids = []
    for center in centers:
        distance = np.abs(centers - center)
        selected = distance <= radius
        weights = np.maximum(
            np.float32(1.0) - distance[selected] / radius, np.float32(min_weight)
        )
        weights = weights / max(weights.sum(), np.float32(1e-12))
        score = (probabilities[np.asarray(starts)[selected]] * weights[:, None]).sum(axis=0)
        token_ids.append(int(score.argmax()))
    collapsed = [token for token, _ in groupby(token_ids) if token != blank_id]
    hypothesis = " ".join(REPLAY.map_phoenix_gloss(vocab[token]) for token in collapsed)
    return REPLAY.clean_phoenix_2014_trans(hypothesis)


def hypothesis_from_token_ids(token_ids, vocab, blank_id):
    collapsed = [int(token) for token, _ in groupby(token_ids) if int(token) != blank_id]
    hypothesis = " ".join(REPLAY.map_phoenix_gloss(vocab[token]) for token in collapsed)
    return REPLAY.clean_phoenix_2014_trans(hypothesis)


def decoder_state(probabilities, starts, span=15.0, min_weight=0.05):
    """Return raw class scores and token ids for an existing selected set."""
    starts = np.asarray(sorted(int(value) for value in starts), dtype=np.int64)
    if len(starts) == 0:
        return starts, np.empty((0, probabilities.shape[1]), dtype=np.float32), np.empty(0, dtype=np.int64)
    distance = np.abs(starts[:, None] - starts[None, :]).astype(np.float32)
    selected = distance <= np.float32(span / 2.0)
    weights = np.where(
        selected,
        np.maximum(np.float32(1.0) - distance / np.float32(span / 2.0), np.float32(min_weight)),
        np.float32(0.0),
    )
    raw_scores = weights @ probabilities[starts]
    return starts, raw_scores, raw_scores.argmax(axis=1)


def candidate_hypothesis(
    probabilities,
    selected_starts,
    raw_scores,
    token_ids,
    candidate,
    vocab,
    blank_id,
    span=15.0,
    min_weight=0.05,
):
    """Decode S union {candidate} by updating only its span-local scores."""
    candidate = int(candidate)
    radius = np.float32(span / 2.0)
    distances = np.abs(selected_starts - candidate).astype(np.float32)
    affected = distances <= radius
    updated_tokens = np.asarray(token_ids, dtype=np.int64).copy()
    if np.any(affected):
        weights = np.maximum(
            np.float32(1.0) - distances[affected] / radius, np.float32(min_weight)
        )
        changed_scores = raw_scores[affected] + weights[:, None] * probabilities[candidate]
        updated_tokens[affected] = changed_scores.argmax(axis=1)
        candidate_score = (
            weights[:, None] * probabilities[selected_starts[affected]]
        ).sum(axis=0) + probabilities[candidate]
    else:
        candidate_score = probabilities[candidate]
    insert_at = int(np.searchsorted(selected_starts, candidate))
    all_tokens = np.insert(updated_tokens, insert_at, int(candidate_score.argmax()))
    return hypothesis_from_token_ids(all_tokens, vocab, blank_id)


def error_count(reference, hypothesis):
    return int(REPLAY.wer_single(reference, hypothesis)["num_err"])


def coverage_tie_key(candidate, selected, total_windows):
    """Deterministic farthest-point tie-break; it is not part of utility."""
    if selected:
        distance = min(abs(candidate - other) for other in selected)
    else:
        distance = min(candidate, total_windows - 1 - candidate)
    return distance, -candidate


def rank_one_step_candidates(probabilities, reference, selected, candidates, vocab, blank_id):
    selected_starts, raw_scores, token_ids = decoder_state(probabilities, selected)
    current_hypothesis = hypothesis_from_token_ids(token_ids, vocab, blank_id)
    current_error = error_count(reference, current_hypothesis)
    ranked = []
    for candidate in candidates:
        hypothesis = candidate_hypothesis(
            probabilities,
            selected_starts,
            raw_scores,
            token_ids,
            candidate,
            vocab,
            blank_id,
        )
        candidate_error = error_count(reference, hypothesis)
        ranked.append(
            {
                "window": int(candidate),
                "utility": int(current_error - candidate_error),
                "error_after": int(candidate_error),
                "hypothesis_after": hypothesis,
            }
        )
    ranked.sort(
        key=lambda row: (
            row["utility"],
            *coverage_tie_key(row["window"], selected, len(probabilities)),
        ),
        reverse=True,
    )
    return current_error, ranked


def greedy_schedule(probabilities, reference, budget, skeleton, vocab, blank_id):
    selected = list(sorted(skeleton))
    trace = []
    while len(selected) < budget:
        candidates = [index for index in range(len(probabilities)) if index not in set(selected)]
        error_before, ranked = rank_one_step_candidates(
            probabilities, reference, selected, candidates, vocab, blank_id
        )
        choice = ranked[0]
        selected.append(choice["window"])
        selected.sort()
        trace.append(
            {
                "step": len(trace) + 1,
                "window": choice["window"],
                "error_before": error_before,
                "error_after": choice["error_after"],
                "utility": choice["utility"],
            }
        )
    return selected, trace


def local_pair_diagnostic(probabilities, reference, skeleton, vocab, blank_id, pool_size=5):
    """Compare two-step greedy with all pairs among top one-step candidates."""
    candidates = [index for index in range(len(probabilities)) if index not in set(skeleton)]
    if len(candidates) < 2:
        return None
    _, ranked = rank_one_step_candidates(
        probabilities, reference, list(skeleton), candidates, vocab, blank_id
    )
    pool = [row["window"] for row in ranked[:pool_size]]
    greedy_first = ranked[0]["window"]
    remaining = [value for value in candidates if value != greedy_first]
    _, second_ranked = rank_one_step_candidates(
        probabilities, reference, sorted([*skeleton, greedy_first]), remaining, vocab, blank_id
    )
    greedy_pair = sorted([greedy_first, second_ranked[0]["window"]])
    greedy_error = error_count(
        reference,
        decode_probabilities(probabilities, sorted([*skeleton, *greedy_pair]), vocab, blank_id),
    )
    exact_records = []
    for left, right in combinations(pool, 2):
        hypothesis = decode_probabilities(
            probabilities, sorted([*skeleton, left, right]), vocab, blank_id
        )
        exact_records.append((error_count(reference, hypothesis), left, right))
    best_error, left, right = min(exact_records, key=lambda row: (row[0], row[1], row[2]))
    return {
        "candidate_pool_size": len(pool),
        "pair_count": len(exact_records),
        "greedy_pair": greedy_pair,
        "greedy_error": int(greedy_error),
        "best_pair": [int(left), int(right)],
        "best_pair_error": int(best_error),
        "exact_pair_better": bool(best_error < greedy_error),
        "error_improvement": int(greedy_error - best_error),
    }


def paired_bootstrap(candidate_counts, baseline_counts, seed=260914, repetitions=10000):
    candidate_error = np.asarray([row["error"] for row in candidate_counts], dtype=np.int64)
    baseline_error = np.asarray([row["error"] for row in baseline_counts], dtype=np.int64)
    ref_len = np.asarray([row["ref_len"] for row in candidate_counts], dtype=np.int64)
    if not all(row["ref_len"] == base["ref_len"] for row, base in zip(candidate_counts, baseline_counts)):
        raise ValueError("paired reference lengths differ")
    rng = np.random.default_rng(seed)
    values = np.empty(repetitions, dtype=np.float64)
    for iteration in range(repetitions):
        draw = rng.integers(0, len(ref_len), len(ref_len))
        values[iteration] = 100.0 * (
            candidate_error[draw].sum() - baseline_error[draw].sum()
        ) / ref_len[draw].sum()
    return {
        "unit": "WER percentage points (candidate - uniform)",
        "resampling_unit": "dev sentence, paired with replacement",
        "seed": int(seed),
        "repetitions": int(repetitions),
        "ci95": [float(value) for value in np.quantile(values, [0.025, 0.975])],
        "mean": float(values.mean()),
        "probability_less_than_zero": float(np.mean(values < 0.0)),
    }


def process_sample(name):
    logits = np.asarray(_WORKER_LOGITS[name])
    reference = _WORKER_REFERENCES[name]
    probabilities = softmax_rows(logits)
    budget = dense_half_budget(len(logits))
    outputs = {}
    pair_diagnostic = None
    for ratio in SKELETON_RATIOS:
        ratio_key = f"{ratio:.2f}"
        skeleton = uniform_positions(len(logits), skeleton_size(budget, ratio))
        selected, trace = greedy_schedule(
            probabilities, reference, budget, skeleton, _WORKER_VOCAB, _WORKER_BLANK_ID
        )
        if len(selected) != budget or len(set(selected)) != budget:
            raise AssertionError("greedy schedule violated exact per-sample budget")
        hypothesis = decode_probabilities(
            probabilities, selected, _WORKER_VOCAB, _WORKER_BLANK_ID
        )
        outputs[ratio_key] = {
            "selected": selected,
            "trace": trace,
            "hypothesis": hypothesis,
            "counts": RANDOM.sentence_counts(reference, hypothesis),
        }
        if ratio == PRIMARY_RATIO:
            pair_diagnostic = local_pair_diagnostic(
                probabilities,
                reference,
                skeleton,
                _WORKER_VOCAB,
                _WORKER_BLANK_ID,
            )
    return name, len(logits), budget, outputs, pair_diagnostic


def run_analysis(
    matrix_root,
    vocab_path,
    max_samples=None,
    bootstrap_repetitions=10000,
    workers=1,
):
    (
        names,
        b0_results,
        b0_logits,
        start_indices,
        _a0_results,
        _b2_results,
        vocab,
        input_paths,
    ) = RANDOM.load_inputs(matrix_root, vocab_path, max_samples)
    blank_id = vocab.index("<blank>")
    references = {
        name: REPLAY.clean_phoenix_2014_trans(b0_results[name]["gls_ref"]) for name in names
    }
    ratios = {f"{ratio:.2f}": ratio for ratio in SKELETON_RATIOS}
    pair_diagnostics = {}
    hypotheses = {key: [] for key in ratios}
    counts = {key: [] for key in ratios}
    rows = []
    for name in names:
        if list(start_indices[name]) != list(range(len(b0_logits[name]))):
            raise ValueError(f"dense coordinates are not 0..T-1: {name}")

    global _WORKER_LOGITS, _WORKER_REFERENCES, _WORKER_VOCAB, _WORKER_BLANK_ID
    _WORKER_LOGITS = b0_logits
    _WORKER_REFERENCES = references
    _WORKER_VOCAB = vocab
    _WORKER_BLANK_ID = blank_id
    pool = None
    if workers == 1:
        iterator = map(process_sample, names)
    else:
        pool = mp.get_context("fork").Pool(processes=workers)
        iterator = pool.imap(process_sample, names, chunksize=1)
    try:
        for sample_index, result in enumerate(iterator, start=1):
            name, dense_windows, budget, outputs, pair_diagnostic = result
            reference = references[name]
            pair_diagnostics[name] = pair_diagnostic
            for ratio_key, ratio in ratios.items():
                output = outputs[ratio_key]
                hypotheses[ratio_key].append(output["hypothesis"])
                counts[ratio_key].append(output["counts"])
                rows.append(
                    {
                        "name": name,
                        "skeleton_ratio": float(ratio),
                        "dense_windows": dense_windows,
                        "budget": budget,
                        "skeleton_count": skeleton_size(budget, ratio),
                        "selected_starts": output["selected"],
                        "greedy_trace": output["trace"],
                        "reference": reference,
                        "hypothesis": output["hypothesis"],
                        **output["counts"],
                        "local_pair_diagnostic": (
                            pair_diagnostic if ratio == PRIMARY_RATIO else None
                        ),
                    }
                )
            print(f"selected {sample_index}/{len(names)}: {name}", flush=True)
    finally:
        if pool is not None:
            pool.close()
            pool.join()

    uniform_key = "1.00"
    metrics = {
        key: REPLAY.compact_metrics(
            [references[name] for name in names], hypotheses[key]
        )
        for key in ratios
    }
    cells = {}
    for key, ratio in ratios.items():
        delta = float(metrics[key]["wer"] - metrics[uniform_key]["wer"])
        bootstrap = None
        if key != uniform_key:
            bootstrap = paired_bootstrap(
                counts[key], counts[uniform_key], repetitions=bootstrap_repetitions
            )
        cells[key] = {
            "skeleton_ratio": ratio,
            "role": (
                "preregistered_primary" if ratio == PRIMARY_RATIO else
                "exact_equal_budget_uniform_baseline" if ratio == 1.0 else
                "exploratory_ablation"
            ),
            "metrics": metrics[key],
            "delta_vs_uniform": {
                "wer_pp": delta,
                "error_count": int(metrics[key]["error"] - metrics[uniform_key]["error"]),
            },
            "paired_bootstrap_vs_uniform": bootstrap,
            "meets_numeric_threshold": bool(
                key != uniform_key
                and delta <= -0.5
                and bootstrap["ci95"][1] < 0.0
            ),
            "strong_go": bool(
                ratio == PRIMARY_RATIO
                and delta <= -0.5
                and bootstrap["ci95"][1] < 0.0
            ),
        }

    pair_values = [value for value in pair_diagnostics.values() if value is not None]
    summary = {
        "sample_count": len(names),
        "full_dev": max_samples is None,
        "oracle_scope": "offline reference-aware dense-logit replay; no runtime claim",
        "budget": "nearest-integer 50% of dense windows independently per sample",
        "decoder": "frozen triangular span-15 voting, blank removal, repeat collapse, Phoenix cleanup",
        "utility": "decrease in frozen decoder sentence error count after adding one window",
        "tie_break": "maximum distance to current selected set, then earlier index",
        "primary_ratio": PRIMARY_RATIO,
        "strong_go_rule": "primary WER delta <= -0.5 pp and paired bootstrap CI95 upper < 0",
        "cells": cells,
        "local_pair_diagnostic": {
            "definition": "at the primary skeleton, exhaust all pairs among top five one-step candidates",
            "samples": len(pair_values),
            "samples_exact_pair_better": sum(value["exact_pair_better"] for value in pair_values),
            "total_error_improvement": sum(value["error_improvement"] for value in pair_values),
        },
    }
    return summary, rows, sorted(set([Path(__file__).resolve(), *input_paths]))


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_outputs(output_root, config, summary, rows, input_paths):
    if output_root.exists() and any(output_root.iterdir()):
        raise FileExistsError(f"refusing to overwrite non-empty output directory: {output_root}")
    aggregate_dir = output_root / "aggregate"
    aggregate_dir.mkdir(parents=True, exist_ok=True)
    (output_root / "resolved_config.json").write_text(
        json.dumps(config, indent=2) + "\n", encoding="utf-8"
    )
    (aggregate_dir / "dev_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    with (output_root / "per_sample_results.jsonl").open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    manifest = {
        "manifest_version": 1,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "policy": "Phoenix repaired dev only; CPU dense-logit replay; reference-aware offline oracle",
        "inputs": {
            str(Path(path).resolve()): {
                "bytes": Path(path).stat().st_size,
                "sha256": sha256_file(path),
            }
            for path in input_paths
        },
    }
    (output_root / "protocol_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matrix-root", type=Path, default=RANDOM.REPLAY.DEFAULT_MATRIX_ROOT)
    parser.add_argument("--vocab", type=Path, default=RANDOM.REPLAY.DEFAULT_VOCAB)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--max-samples", type=int)
    parser.add_argument("--bootstrap-repetitions", type=int, default=10000)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.max_samples is not None and args.max_samples < 1:
        parser.error("--max-samples must be positive")
    if args.bootstrap_repetitions < 1:
        parser.error("--bootstrap-repetitions must be positive")
    if args.workers < 1:
        parser.error("--workers must be positive")
    config = {
        "matrix_root": str(args.matrix_root.resolve()),
        "vocab": str(args.vocab.resolve()),
        "output_root": str(args.output_root.resolve()),
        "max_samples": args.max_samples,
        "bootstrap_repetitions": args.bootstrap_repetitions,
        "workers": args.workers,
        "device": "cpu",
        "skeleton_ratios": list(SKELETON_RATIOS),
        "primary_ratio": PRIMARY_RATIO,
    }
    summary, rows, input_paths = run_analysis(
        args.matrix_root.resolve(),
        args.vocab.resolve(),
        max_samples=args.max_samples,
        bootstrap_repetitions=args.bootstrap_repetitions,
        workers=args.workers,
    )
    print(json.dumps(summary, indent=2))
    if not args.dry_run:
        write_outputs(args.output_root.resolve(), config, summary, rows, input_paths)


if __name__ == "__main__":
    main()
