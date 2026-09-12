#!/usr/bin/env python3
"""Audit fixed coordinates and replay random Phoenix dev schedules on CPU.

Only repaired Phoenix-2014T ``dev`` artifacts are accepted.  Random schedules
index the frozen dense B0 logits; they never run the recognition model.
"""

import argparse
import hashlib
import importlib.util
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


CSLR_ROOT = Path(__file__).resolve().parents[1]
REPLAY_PATH = CSLR_ROOT / "tools/analyze_phoenix_schedule_replay.py"
SPEC = importlib.util.spec_from_file_location("schedule_replay", REPLAY_PATH)
REPLAY = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(REPLAY)

DEFAULT_OUTPUT_ROOT = (
    CSLR_ROOT / "results/phoenix-2014t_ISLR/p0_random_schedule_v1_49faacc3"
)
DEFAULT_SEEDS = tuple(range(1729, 1759))


def sample_seed(seed, name):
    digest = hashlib.sha256(name.encode("utf-8")).digest()
    return np.random.SeedSequence(
        [int(seed), int.from_bytes(digest[:4], "little"), int.from_bytes(digest[4:8], "little")]
    )


def random_schedule_per_sample(total_frames, budget, seed, name):
    """Choose exactly ``budget`` unique centers, always covering both ends."""
    if total_frames < 1 or budget < 1 or budget > total_frames:
        raise ValueError("budget must satisfy 1 <= budget <= total_frames")
    if total_frames == 1:
        return [0]
    if budget == 1:
        raise ValueError("two distinct endpoints require budget >= 2")
    if budget == 2:
        return [0, total_frames - 1]
    rng = np.random.default_rng(sample_seed(seed, name))
    interior = rng.choice(
        np.arange(1, total_frames - 1, dtype=np.int64),
        size=budget - 2,
        replace=False,
    )
    return [0, *sorted(int(value) for value in interior), total_frames - 1]


def random_schedules_global(total_frames_by_name, total_budget, seed):
    """Choose a global exact budget after reserving two endpoints per sample."""
    schedules = {}
    candidates = []
    reserved = 0
    for name, total_frames in total_frames_by_name.items():
        if total_frames < 1:
            raise ValueError(f"sample has no dense centers: {name}")
        endpoints = [0] if total_frames == 1 else [0, total_frames - 1]
        schedules[name] = endpoints
        reserved += len(endpoints)
        candidates.extend((name, start) for start in range(1, total_frames - 1))
    remaining = total_budget - reserved
    if remaining < 0 or remaining > len(candidates):
        raise ValueError("global budget is incompatible with endpoint coverage")
    rng = np.random.default_rng(int(seed))
    selected = rng.choice(len(candidates), size=remaining, replace=False)
    for index in selected:
        name, start = candidates[int(index)]
        schedules[name].append(start)
    for name in schedules:
        schedules[name].sort()
    return schedules


def sentence_counts(reference, hypothesis):
    result = REPLAY.wer_single(reference, hypothesis)
    return {
        "error": int(result["num_err"]),
        "del": int(result["num_del"]),
        "ins": int(result["num_ins"]),
        "sub": int(result["num_sub"]),
        "ref_len": int(result["num_ref"]),
    }


def decode_schedules(names, schedules, dense_logits, start_indices, vocab, blank_id):
    hypotheses = []
    for name in names:
        starts = schedules[name]
        selected = REPLAY.extract_schedule_logits(name, dense_logits, start_indices, starts)
        hypotheses.append(
            REPLAY.clean_phoenix_2014_trans(
                REPLAY.decode_span15(selected, starts, vocab, blank_id)
            )
        )
    return hypotheses


def metric_delta(metrics, baseline):
    return {
        "wer_pp": float(metrics["wer"] - baseline["wer"]),
        "error_count": int(metrics["error"] - baseline["error"]),
        "del_count": int(round(metrics["del"] * metrics["ref_len"] / 100))
        - int(round(baseline["del"] * baseline["ref_len"] / 100)),
        "ins_count": int(round(metrics["ins"] * metrics["ref_len"] / 100))
        - int(round(baseline["ins"] * baseline["ref_len"] / 100)),
        "sub_count": int(round(metrics["sub"] * metrics["ref_len"] / 100))
        - int(round(baseline["sub"] * baseline["ref_len"] / 100)),
    }


def distribution_summary(seed_records, a0_wer, b2_wer):
    keys = ("wer", "del", "ins", "sub", "error")
    summary = {}
    for key in keys:
        values = np.asarray([record["metrics"][key] for record in seed_records], dtype=float)
        summary[key] = {
            "mean": float(values.mean()),
            "sample_std": float(values.std(ddof=1)) if len(values) > 1 else 0.0,
            "min": float(values.min()),
            "median": float(np.median(values)),
            "max": float(values.max()),
        }
    wers = np.asarray([record["metrics"]["wer"] for record in seed_records])
    summary["a0_empirical_percentile"] = {
        "definition": "100 * fraction(random WER <= A0 WER); lower WER is better",
        "value": float(100.0 * np.mean(wers <= a0_wer)),
    }
    summary["b2_empirical_percentile"] = {
        "definition": "100 * fraction(random WER <= B2 WER); lower WER is better",
        "value": float(100.0 * np.mean(wers <= b2_wer)),
    }
    return summary


def paired_bootstrap(candidate_counts, baseline_counts, seed=260910, repetitions=10000):
    """Sentence-paired bootstrap for one seed fixed before observing results."""
    candidate_error = np.asarray([row["error"] for row in candidate_counts], dtype=np.int64)
    baseline_error = np.asarray([row["error"] for row in baseline_counts], dtype=np.int64)
    ref_len = np.asarray([row["ref_len"] for row in candidate_counts], dtype=np.int64)
    if not all(row["ref_len"] == base["ref_len"] for row, base in zip(candidate_counts, baseline_counts)):
        raise ValueError("paired bootstrap reference lengths differ")
    rng = np.random.default_rng(seed)
    values = np.empty(repetitions, dtype=np.float64)
    for index in range(repetitions):
        draw = rng.integers(0, len(ref_len), size=len(ref_len))
        values[index] = 100.0 * (
            candidate_error[draw].sum() - baseline_error[draw].sum()
        ) / ref_len[draw].sum()
    return {
        "unit": "WER percentage points (candidate - baseline)",
        "resampling_unit": "dev sentence, paired with replacement",
        "seed": int(seed),
        "repetitions": int(repetitions),
        "ci95": [float(value) for value in np.quantile(values, [0.025, 0.975])],
        "mean": float(values.mean()),
        "probability_less_than_zero": float(np.mean(values < 0.0)),
    }


def load_inputs(matrix_root, vocab_path, max_samples=None):
    REPLAY.reject_test_path(matrix_root)
    REPLAY.reject_test_path(vocab_path)
    manifest_path = matrix_root / "protocol_manifest.json"
    if REPLAY.sha256_file(manifest_path) != REPLAY.EXPECTED_MATRIX_MANIFEST_SHA256:
        raise ValueError("baseline matrix is not frozen repaired dev")
    b0_results, b0_logits, b0_results_path, b0_logits_path = REPLAY.load_run(
        matrix_root / "dev/B0_fixed1_window7/run_01"
    )
    all_names = list(b0_results)
    names = all_names if max_samples is None else all_names[:max_samples]
    b0_results = {name: b0_results[name] for name in names}
    b0_logits = {name: b0_logits[name] for name in names}
    start_indices = REPLAY.dense_start_index(b0_results, b0_logits)
    with vocab_path.open("r", encoding="utf-8") as handle:
        vocab = json.load(handle)
    a0_all, _, a0_results_path, a0_logits_path = REPLAY.load_run(
        matrix_root / "dev/A0_adaptive_span15/run_01"
    )
    b2_all, _, b2_results_path, b2_logits_path = REPLAY.load_run(
        matrix_root / "dev/B2_uniform_rate_span15/run_01"
    )
    if list(a0_all) != all_names or list(b2_all) != all_names:
        raise ValueError("schedule sample order differs from B0")
    a0_results = {name: a0_all[name] for name in names}
    b2_results = {name: b2_all[name] for name in names}
    inputs = [
        Path(__file__).resolve(),
        REPLAY_PATH,
        manifest_path,
        b0_results_path,
        b0_logits_path,
        a0_results_path,
        a0_logits_path,
        b2_results_path,
        b2_logits_path,
        vocab_path,
    ]
    return names, b0_results, b0_logits, start_indices, a0_results, b2_results, vocab, inputs


def run_analysis(matrix_root, vocab_path, seeds=DEFAULT_SEEDS, max_samples=None, include_global=True):
    if not seeds or len(set(seeds)) != len(seeds):
        raise ValueError("seeds must be non-empty and unique")
    (
        names,
        b0_results,
        b0_logits,
        start_indices,
        a0_results,
        b2_results,
        vocab,
        input_paths,
    ) = load_inputs(matrix_root, vocab_path, max_samples)
    blank_id = vocab.index("<blank>")
    references = [REPLAY.clean_phoenix_2014_trans(b0_results[name]["gls_ref"]) for name in names]
    total_frames = {name: len(start_indices[name]) for name in names}
    a0_saved = REPLAY.starts_from_results(a0_results, names)
    b2_saved = REPLAY.starts_from_results(b2_results, names)

    # Two interpretations: preserve the actual frozen physical inputs, or treat
    # every saved numeric start in B0's fixed-pad coordinate system.
    physical_schedules = {}
    fixed_coordinate_schedules = {}
    offset_counts = {}
    for label, schedules in (("B1", {name: list(range(total_frames[name])) for name in names}),
                             ("B2", b2_saved), ("A0", a0_saved)):
        physical_schedules[label] = {}
        fixed_coordinate_schedules[label] = {}
        offsets = {}
        for name in names:
            aligned, dense_pad, target_pad = REPLAY.align_starts_to_dense_inputs(
                list(range(total_frames[name])), schedules[name]
            )
            physical_schedules[label][name] = aligned
            fixed_coordinate_schedules[label][name] = list(schedules[name])
            offset = dense_pad - target_pad
            offsets[str(offset)] = offsets.get(str(offset), 0) + 1
        offset_counts[label] = dict(sorted(offsets.items()))

    coordinate_audit = {
        "definitions": {
            "frozen_physical_input": "map each schedule through its own centered left padding into B0 dense input windows",
            "fixed_b0_coordinate": "interpret saved numeric starts directly under B0's fixed left padding of 7",
        },
        "dense_b0_evidence": {
            "coverage": "one saved GPU logit row for every padded-array start 0..T-1",
            "fixed_left_padding_frames": 7,
            "derivation": "floor(((T-1)+16-T)/2) = 7 for every sample",
        },
        "limitations": [
            "Both interpretations retain centered 16-frame windows and offline symmetric span-15 decoding.",
            "Dense B0 logits cannot exactly represent zero-lookahead trailing windows at stream onset.",
        ],
        "strict_streaming_minimum_experiment": {
            "not_claimed_here": "zero-lookahead trailing-window WER or end-to-end streaming latency",
            "missing_dense_coverage": (
                "A trailing window ending at decision frame t maps to B0 start t-8; "
                "for t=0..7 that start is negative and absent from B0 logits."
            ),
            "minimum_new_forward": (
                "On repaired dev only, run the frozen checkpoint for the missing trailing windows "
                "at stream onset (at most 8 per sample), reuse B0 logits for later matching inputs, "
                "then replay a past-only decoder. A separate sequential run is still required for latency."
            ),
        },
        "variants": {},
    }
    baseline_hypotheses = {}
    baseline_counts = {}
    baseline_metrics = {}
    for label in ("B1", "B2", "A0"):
        frozen_hyp = decode_schedules(
            names, physical_schedules[label], b0_logits, start_indices, vocab, blank_id
        )
        fixed_hyp = decode_schedules(
            names, fixed_coordinate_schedules[label], b0_logits, start_indices, vocab, blank_id
        )
        frozen_metrics = REPLAY.compact_metrics(references, frozen_hyp)
        fixed_metrics = REPLAY.compact_metrics(references, fixed_hyp)
        baseline_hypotheses[label] = frozen_hyp
        baseline_counts[label] = [sentence_counts(r, h) for r, h in zip(references, frozen_hyp)]
        baseline_metrics[label] = frozen_metrics
        coordinate_audit["variants"][label] = {
            "padding_offset_sample_counts": offset_counts[label],
            "frozen_physical_input_metrics": frozen_metrics,
            "fixed_b0_coordinate_metrics": fixed_metrics,
            "fixed_minus_frozen_wer_pp": float(fixed_metrics["wer"] - frozen_metrics["wer"]),
            "hypotheses_changed": int(sum(left != right for left, right in zip(frozen_hyp, fixed_hyp))),
        }

    a0_budgets = {name: len(a0_saved[name]) for name in names}
    total_budget = sum(a0_budgets.values())
    protocols = ["per_sample_exact"] + (["global_exact"] if include_global else [])
    random_results = {}
    representative_rows = []
    for protocol in protocols:
        seed_records = []
        representative_counts = None
        for seed in seeds:
            if protocol == "per_sample_exact":
                schedules = {
                    name: random_schedule_per_sample(total_frames[name], a0_budgets[name], seed, name)
                    for name in names
                }
                if any(len(schedules[name]) != a0_budgets[name] for name in names):
                    raise AssertionError("random schedule violated a per-sample A0 budget")
            else:
                schedules = random_schedules_global(total_frames, total_budget, seed)
            if sum(map(len, schedules.values())) != total_budget:
                raise AssertionError("random schedule violated total budget")
            hypotheses = decode_schedules(names, schedules, b0_logits, start_indices, vocab, blank_id)
            metrics = REPLAY.compact_metrics(references, hypotheses)
            counts = [sentence_counts(r, h) for r, h in zip(references, hypotheses)]
            seed_records.append({
                "seed": int(seed),
                "clip_count": int(sum(map(len, schedules.values()))),
                "metrics": metrics,
                "delta_vs_B2": metric_delta(metrics, baseline_metrics["B2"]),
                "delta_vs_A0": metric_delta(metrics, baseline_metrics["A0"]),
            })
            if seed == seeds[0]:
                representative_counts = counts
                for index, name in enumerate(names):
                    representative_rows.append({
                        "protocol": protocol,
                        "seed": int(seed),
                        "name": name,
                        "clips": len(schedules[name]),
                        "reference": references[index],
                        "hypothesis": hypotheses[index],
                        **counts[index],
                    })
        random_results[protocol] = {
            "budget_rule": (
                "exact A0 clip count independently for every sample"
                if protocol == "per_sample_exact"
                else "exact A0 total dev clip count; two endpoints reserved per sample"
            ),
            "endpoint_coverage": "centers 0 and T-1 (or only 0 when T=1)",
            "seed_records": seed_records,
            "distribution": distribution_summary(
                seed_records, baseline_metrics["A0"]["wer"], baseline_metrics["B2"]["wer"]
            ),
            "representative_seed": int(seeds[0]),
            "representative_seed_selection": "first seed fixed before observing results",
            "representative_paired_bootstrap": {
                "vs_B2": paired_bootstrap(representative_counts, baseline_counts["B2"]),
                "vs_A0": paired_bootstrap(representative_counts, baseline_counts["A0"]),
            },
        }

    summary = {
        "sample_count": len(names),
        "full_dev": max_samples is None,
        "seeds": [int(seed) for seed in seeds],
        "random_seed_uncertainty": "empirical distribution over predeclared scheduler seeds",
        "sentence_uncertainty": "paired sentence bootstrap only for the predeclared representative first seed; not pooled with seed variance",
        "window_frames": 16,
        "decoder": "triangular span-weighted-15, minimum weight 0.05",
        "total_a0_budget": total_budget,
        "coordinate_audit": coordinate_audit,
        "frozen_baselines": baseline_metrics,
        "random_protocols": random_results,
    }
    if max_samples is None:
        for label, replay_id in (
            ("B1", "B0_fixed1_span15_replay"),
            ("B2", "B2_uniform_rate_span15_from_B0"),
            ("A0", "A0_adaptive_span15_from_B0"),
        ):
            expected = REPLAY.EXPECTED_FULL_DEV[replay_id]
            observed = baseline_metrics[label]
            if observed["error"] != expected["error"] or abs(observed["wer"] - expected["wer"]) > 1e-12:
                raise AssertionError(f"{label} no longer reproduces frozen repaired dev")
    return summary, representative_rows, sorted(set(input_paths))


def write_outputs(output_root, config, summary, representative_rows, input_paths):
    if output_root.exists() and any(output_root.iterdir()):
        raise FileExistsError(f"refusing to overwrite non-empty output directory: {output_root}")
    aggregate = output_root / "aggregate"
    aggregate.mkdir(parents=True, exist_ok=True)
    (output_root / "resolved_config.json").write_text(json.dumps(config, indent=2) + "\n")
    (aggregate / "dev_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    with (output_root / "representative_per_sample.jsonl").open("w", encoding="utf-8") as handle:
        for row in representative_rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    manifest = {
        "manifest_version": 1,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "policy": "repaired Phoenix-2014T dev only; CPU dense-logit replay; no model forward",
        "inputs": {
            str(path.resolve()): {"bytes": path.stat().st_size, "sha256": REPLAY.sha256_file(path)}
            for path in input_paths
        },
    }
    (output_root / "protocol_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


def parse_seeds(value):
    seeds = tuple(int(item) for item in value.split(",") if item.strip())
    if not seeds:
        raise argparse.ArgumentTypeError("at least one seed is required")
    return seeds


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matrix-root", type=Path, default=REPLAY.DEFAULT_MATRIX_ROOT)
    parser.add_argument("--vocab", type=Path, default=REPLAY.DEFAULT_VOCAB)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--seeds", type=parse_seeds, default=DEFAULT_SEEDS)
    parser.add_argument("--max-samples", type=int)
    parser.add_argument("--skip-global", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.max_samples is not None and args.max_samples < 1:
        parser.error("--max-samples must be positive")
    config = {
        "matrix_root": str(args.matrix_root.resolve()),
        "vocab": str(args.vocab.resolve()),
        "output_root": str(args.output_root.resolve()),
        "seeds": list(args.seeds),
        "max_samples": args.max_samples,
        "include_global_budget_protocol": not args.skip_global,
        "device": "cpu",
    }
    summary, rows, inputs = run_analysis(
        args.matrix_root.resolve(), args.vocab.resolve(), args.seeds,
        args.max_samples, not args.skip_global,
    )
    print(json.dumps(summary, indent=2))
    if not args.dry_run:
        write_outputs(args.output_root.resolve(), config, summary, rows, inputs)


if __name__ == "__main__":
    main()
