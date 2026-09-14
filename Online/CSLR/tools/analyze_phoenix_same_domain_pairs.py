#!/usr/bin/env python3
"""Same-domain two-window complementarity diagnostic on frozen Phoenix dev."""

import argparse
import hashlib
import importlib.util
import json
import math
import multiprocessing as mp
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


CSLR_ROOT = Path(__file__).resolve().parents[1]
ORACLE_PATH = CSLR_ROOT / "tools/analyze_phoenix_decoder_utility_oracle.py"
SPEC = importlib.util.spec_from_file_location("decoder_utility_oracle", ORACLE_PATH)
ORACLE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(ORACLE)
REPLAY = ORACLE.REPLAY
RANDOM = ORACLE.RANDOM

DEFAULT_OUTPUT_ROOT = (
    CSLR_ROOT
    / "results/phoenix-2014t_ISLR/p3_same_domain_pair_diagnostic_v1_49faacc3"
)
DEFAULT_TARGET_SAMPLES = 100
DEFAULT_SEED = 260915

_LOGITS = None
_REFERENCES = None
_VOCAB = None
_BLANK_ID = None


def initial_primary_skeleton(total_windows):
    budget = ORACLE.dense_half_budget(total_windows)
    count = ORACLE.skeleton_size(budget, ORACLE.PRIMARY_RATIO)
    return ORACLE.uniform_positions(total_windows, count)


def initial_single_utilities(probabilities, reference, skeleton, vocab, blank_id):
    candidates = [index for index in range(len(probabilities)) if index not in set(skeleton)]
    current_error, ranked = ORACLE.rank_one_step_candidates(
        probabilities, reference, list(skeleton), candidates, vocab, blank_id
    )
    utilities = {row["window"]: row["utility"] for row in ranked}
    return current_error, candidates, utilities, ranked


def evaluate_pair(probabilities, reference, selected, left, right, vocab, blank_id):
    selected_with_left = sorted([*selected, left])
    starts, raw_scores, token_ids = ORACLE.decoder_state(probabilities, selected_with_left)
    hypothesis = ORACLE.candidate_hypothesis(
        probabilities,
        starts,
        raw_scores,
        token_ids,
        right,
        vocab,
        blank_id,
    )
    return ORACLE.error_count(reference, hypothesis)


def same_domain_pair_diagnostic(probabilities, reference, skeleton, vocab, blank_id):
    """Compare exhaustive pairs and sequential greedy over one candidate domain."""
    current_error, candidates, single_utilities, ranked = initial_single_utilities(
        probabilities, reference, skeleton, vocab, blank_id
    )
    if len(candidates) < 2:
        return None

    greedy_first = ranked[0]["window"]
    remaining = [candidate for candidate in candidates if candidate != greedy_first]
    _, second_ranked = ORACLE.rank_one_step_candidates(
        probabilities,
        reference,
        sorted([*skeleton, greedy_first]),
        remaining,
        vocab,
        blank_id,
    )
    greedy_second = second_ranked[0]["window"]
    greedy_pair = sorted([greedy_first, greedy_second])
    greedy_error = evaluate_pair(
        probabilities, reference, skeleton, greedy_pair[0], greedy_pair[1], vocab, blank_id
    )

    best_error = None
    best_pair = None
    zero_zero_positive_count = 0
    zero_zero_best_joint_utility = 0
    pair_count = 0
    for left_offset, left in enumerate(candidates[:-1]):
        selected_with_left = sorted([*skeleton, left])
        starts, raw_scores, token_ids = ORACLE.decoder_state(probabilities, selected_with_left)
        for right in candidates[left_offset + 1 :]:
            hypothesis = ORACLE.candidate_hypothesis(
                probabilities,
                starts,
                raw_scores,
                token_ids,
                right,
                vocab,
                blank_id,
            )
            pair_error = ORACLE.error_count(reference, hypothesis)
            pair_count += 1
            if best_error is None or (pair_error, left, right) < (best_error, *best_pair):
                best_error = pair_error
                best_pair = (left, right)
            joint_utility = current_error - pair_error
            if (
                single_utilities[left] == 0
                and single_utilities[right] == 0
                and joint_utility > 0
            ):
                zero_zero_positive_count += 1
                zero_zero_best_joint_utility = max(zero_zero_best_joint_utility, joint_utility)

    regret = int(greedy_error - best_error)
    return {
        "candidate_domain_size": len(candidates),
        "pair_count": pair_count,
        "initial_error": int(current_error),
        "single_utility_counts": {
            str(key): int(value) for key, value in sorted(Counter(single_utilities.values()).items())
        },
        "all_single_utilities_zero": all(value == 0 for value in single_utilities.values()),
        "greedy_pair": greedy_pair,
        "greedy_pair_error": int(greedy_error),
        "best_pair": [int(value) for value in best_pair],
        "best_pair_error": int(best_error),
        "best_minus_greedy_error": int(best_error - greedy_error),
        "greedy_regret": regret,
        "zero_zero_positive_joint_pair_count": int(zero_zero_positive_count),
        "zero_zero_best_joint_utility": int(zero_zero_best_joint_utility),
    }


def audit_high_risk(name):
    probabilities = ORACLE.softmax_rows(_LOGITS[name])
    skeleton = initial_primary_skeleton(len(probabilities))
    _, _, utilities, _ = initial_single_utilities(
        probabilities, _REFERENCES[name], skeleton, _VOCAB, _BLANK_ID
    )
    return {
        "name": name,
        "dense_windows": len(probabilities),
        "reference_length": len(_REFERENCES[name].split()),
        "all_single_utilities_zero": all(value == 0 for value in utilities.values()),
        "best_single_utility": int(max(utilities.values())),
        "worst_single_utility": int(min(utilities.values())),
    }


def quantile_bin(values, value, bins=3):
    edges = np.quantile(np.asarray(values, dtype=np.float64), np.arange(1, bins) / bins)
    return int(np.searchsorted(edges, value, side="right")), [float(edge) for edge in edges]


def deterministic_stratified_sample(records, target, seed=DEFAULT_SEED):
    """Proportional 3x3 allocation with largest-remainder rounding."""
    if target >= len(records):
        selected = list(records)
    else:
        video_values = [row["dense_windows"] for row in records]
        reference_values = [row["reference_length"] for row in records]
        cells = defaultdict(list)
        for row in records:
            video_bin, video_edges = quantile_bin(video_values, row["dense_windows"])
            reference_bin, reference_edges = quantile_bin(
                reference_values, row["reference_length"]
            )
            enriched = dict(row, video_length_bin=video_bin, reference_length_bin=reference_bin)
            cells[(video_bin, reference_bin)].append(enriched)
        ideals = {cell: target * len(items) / len(records) for cell, items in cells.items()}
        allocation = {cell: int(math.floor(value)) for cell, value in ideals.items()}
        remaining = target - sum(allocation.values())
        order = sorted(cells, key=lambda cell: (-(ideals[cell] - allocation[cell]), cell))
        for cell in order[:remaining]:
            allocation[cell] += 1
        selected = []
        for cell in sorted(cells):
            items = sorted(
                cells[cell],
                key=lambda row: hashlib.sha256(
                    f"{seed}:{row['name']}".encode("utf-8")
                ).hexdigest(),
            )
            selected.extend(items[: allocation[cell]])
        selected.sort(key=lambda row: row["name"])
        return selected, {
            "method": "proportional 3x3 empirical-tertile cells; largest-remainder allocation; SHA-256 seeded order within cell",
            "seed": seed,
            "video_length_tertile_edges": video_edges,
            "reference_length_tertile_edges": reference_edges,
            "cell_population": {f"{a},{b}": len(items) for (a, b), items in sorted(cells.items())},
            "cell_selected": {f"{a},{b}": allocation[(a, b)] for a, b in sorted(cells)},
        }
    video_values = [row["dense_windows"] for row in records]
    reference_values = [row["reference_length"] for row in records]
    for row in selected:
        row["video_length_bin"], video_edges = quantile_bin(video_values, row["dense_windows"])
        row["reference_length_bin"], reference_edges = quantile_bin(
            reference_values, row["reference_length"]
        )
    return selected, {
        "method": "all eligible samples because population did not exceed target",
        "seed": seed,
        "video_length_tertile_edges": video_edges,
        "reference_length_tertile_edges": reference_edges,
    }


def evaluate_selected(row):
    name = row["name"]
    started = time.perf_counter()
    probabilities = ORACLE.softmax_rows(_LOGITS[name])
    skeleton = initial_primary_skeleton(len(probabilities))
    diagnostic = same_domain_pair_diagnostic(
        probabilities, _REFERENCES[name], skeleton, _VOCAB, _BLANK_ID
    )
    return {
        **row,
        "initial_skeleton": skeleton,
        "elapsed_seconds": float(time.perf_counter() - started),
        **diagnostic,
    }


def stratum_summary(rows, first_key, second_key):
    groups = defaultdict(list)
    for row in rows:
        groups[(row[first_key], row[second_key])].append(row)
    output = {}
    for key, values in sorted(groups.items()):
        output[f"{key[0]},{key[1]}"] = {
            "samples": len(values),
            "greedy_positive_regret_samples": sum(row["greedy_regret"] > 0 for row in values),
            "greedy_regret_sum": sum(row["greedy_regret"] for row in values),
            "zero_zero_positive_joint_pairs": sum(
                row["zero_zero_positive_joint_pair_count"] for row in values
            ),
        }
    return output


def run_analysis(matrix_root, vocab_path, target_samples=100, workers=4):
    (
        names,
        b0_results,
        b0_logits,
        start_indices,
        _a0_results,
        _b2_results,
        vocab,
        input_paths,
    ) = RANDOM.load_inputs(matrix_root, vocab_path)
    references = {
        name: REPLAY.clean_phoenix_2014_trans(b0_results[name]["gls_ref"]) for name in names
    }
    for name in names:
        if list(start_indices[name]) != list(range(len(b0_logits[name]))):
            raise ValueError(f"non-dense candidate coordinates: {name}")
    global _LOGITS, _REFERENCES, _VOCAB, _BLANK_ID
    _LOGITS = b0_logits
    _REFERENCES = references
    _VOCAB = vocab
    _BLANK_ID = vocab.index("<blank>")

    started = time.perf_counter()
    with mp.get_context("fork").Pool(processes=workers) as pool:
        audit = list(pool.imap(audit_high_risk, names, chunksize=1))
    eligible = [row for row in audit if row["all_single_utilities_zero"]]
    selected, sampling = deterministic_stratified_sample(
        eligible, min(target_samples, len(eligible))
    )
    with mp.get_context("fork").Pool(processes=workers) as pool:
        rows = list(pool.imap(evaluate_selected, selected, chunksize=1))
    elapsed = time.perf_counter() - started

    regret_values = [row["greedy_regret"] for row in rows]
    summary = {
        "full_dev_audit_samples": len(audit),
        "eligible_all_single_utilities_zero": len(eligible),
        "selected_samples": len(rows),
        "selection": sampling,
        "state": "initial 50% skeleton within a per-sample 50%-of-dense total budget",
        "candidate_domain": "all dense windows not in the initial skeleton, identical for greedy and exhaustive pair search",
        "decoder": "frozen triangular span-15 replay decoder",
        "runtime": {
            "workers": workers,
            "wall_seconds": float(elapsed),
            "sum_per_sample_pair_seconds": float(sum(row["elapsed_seconds"] for row in rows)),
            "total_pairs_evaluated": int(sum(row["pair_count"] for row in rows)),
        },
        "greedy_vs_exhaustive": {
            "best_minus_greedy_error_distribution": {
                str(key): int(value)
                for key, value in sorted(Counter(-value for value in regret_values).items())
            },
            "greedy_positive_regret_samples": int(sum(value > 0 for value in regret_values)),
            "greedy_positive_regret_fraction": float(np.mean(np.asarray(regret_values) > 0)),
            "greedy_regret_sum": int(sum(regret_values)),
            "greedy_regret_max": int(max(regret_values, default=0)),
        },
        "zero_plus_zero_complementarity": {
            "positive_joint_pair_count": int(
                sum(row["zero_zero_positive_joint_pair_count"] for row in rows)
            ),
            "samples_with_positive_joint_pair": int(
                sum(row["zero_zero_positive_joint_pair_count"] > 0 for row in rows)
            ),
            "maximum_joint_utility": int(
                max((row["zero_zero_best_joint_utility"] for row in rows), default=0)
            ),
        },
        "strata_video_by_reference_tertile": stratum_summary(
            rows, "video_length_bin", "reference_length_bin"
        ),
    }
    return summary, rows, sorted(set([Path(__file__).resolve(), ORACLE_PATH, *input_paths]))


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_outputs(output_root, config, summary, rows, input_paths):
    if output_root.exists() and any(output_root.iterdir()):
        raise FileExistsError(f"refusing to overwrite non-empty output directory: {output_root}")
    aggregate = output_root / "aggregate"
    aggregate.mkdir(parents=True, exist_ok=True)
    (output_root / "resolved_config.json").write_text(
        json.dumps(config, indent=2) + "\n", encoding="utf-8"
    )
    (aggregate / "dev_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    with (output_root / "per_sample_results.jsonl").open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    manifest = {
        "manifest_version": 1,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "policy": "Phoenix repaired dev; cached dense logits only; reference-aware diagnostic",
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
    parser.add_argument("--matrix-root", type=Path, default=REPLAY.DEFAULT_MATRIX_ROOT)
    parser.add_argument("--vocab", type=Path, default=REPLAY.DEFAULT_VOCAB)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--target-samples", type=int, default=DEFAULT_TARGET_SAMPLES)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.target_samples < 1 or args.workers < 1:
        parser.error("target-samples and workers must be positive")
    config = {
        "matrix_root": str(args.matrix_root.resolve()),
        "vocab": str(args.vocab.resolve()),
        "output_root": str(args.output_root.resolve()),
        "target_samples": args.target_samples,
        "workers": args.workers,
        "seed": DEFAULT_SEED,
        "device": "cpu",
    }
    summary, rows, inputs = run_analysis(
        args.matrix_root.resolve(),
        args.vocab.resolve(),
        target_samples=args.target_samples,
        workers=args.workers,
    )
    print(json.dumps(summary, indent=2))
    if not args.dry_run:
        write_outputs(args.output_root.resolve(), config, summary, rows, inputs)


if __name__ == "__main__":
    main()
