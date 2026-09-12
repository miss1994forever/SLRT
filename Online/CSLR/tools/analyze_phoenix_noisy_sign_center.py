#!/usr/bin/env python3
"""Robustness bridge for the fixed repaired-dev sign-center proxy oracle.

This remains a label-derived offline diagnostic. Noise tests localization
tolerance only; it does not make segment centers observable online.
"""

import argparse
import importlib.util
import json
import math
from datetime import datetime, timezone
from itertools import groupby
from pathlib import Path

import numpy as np


CSLR_ROOT = Path(__file__).resolve().parents[1]
CENTER_PATH = CSLR_ROOT / "tools/analyze_phoenix_sign_center_oracles.py"
SPEC = importlib.util.spec_from_file_location("sign_center", CENTER_PATH)
CENTER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(CENTER)
BOUNDARY = CENTER.BOUNDARY
STRUCTURED = CENTER.STRUCTURED
RANDOM = CENTER.RANDOM
REPLAY = CENTER.REPLAY

DEFAULT_OUTPUT_ROOT = (
    CSLR_ROOT / "results/phoenix-2014t_ISLR/p0_noisy_sign_center_v1_49faacc3"
)
OFFSETS = (-12, -8, -4, -2, -1, 0, 1, 2, 4, 8, 12)
JITTER_AMPLITUDES = (1, 2, 4, 8)
KEEP_RATIOS = (1.0, 0.9, 0.75, 0.5, 0.25)
FALSE_RATIOS = (0.0, 0.25, 0.5, 1.0)
SEEDS = tuple(range(1829, 1859))
BOOTSTRAP_COMPARISONS = 25


def clip_centers(centers, total_frames):
    return [min(float(total_frames - 1), max(0.0, float(center))) for center in centers]


def offset_centers(centers, offset, total_frames):
    return clip_centers([center + offset for center in centers], total_frames)


def jitter_centers(centers, amplitude, total_frames, seed, name):
    rng = np.random.default_rng(RANDOM.sample_seed(seed, f"jitter:{amplitude}:{name}"))
    noise = rng.integers(-amplitude, amplitude + 1, size=len(centers))
    return clip_centers([center + int(delta) for center, delta in zip(centers, noise)], total_frames)


def dropout_centers(centers, keep_ratio, seed, name):
    return [centers[index] for index in dropout_indices(len(centers), keep_ratio, seed, name)]


def dropout_indices(count, keep_ratio, seed, name):
    if not 0.0 <= keep_ratio <= 1.0:
        raise ValueError("keep ratio must be in [0, 1]")
    keep = int(math.floor(count * keep_ratio + 0.5))
    if keep == count:
        return list(range(count))
    if keep == 0:
        return []
    rng = np.random.default_rng(RANDOM.sample_seed(seed, f"drop:{keep_ratio}:{name}"))
    return sorted(int(index) for index in rng.choice(count, size=keep, replace=False))


def false_centers(true_centers, false_ratio, total_frames, seed, name):
    count = int(math.floor(len(true_centers) * false_ratio + 0.5))
    if count == 0:
        return []
    candidates = [position / 2.0 for position in range(2 * total_frames - 1)]
    candidates = [
        position for position in candidates
        if all(abs(position - center) > 0.25 for center in true_centers)
    ]
    if count > len(candidates):
        raise ValueError(f"insufficient false-center candidates for {name}: {count}>{len(candidates)}")
    rng = np.random.default_rng(RANDOM.sample_seed(seed, f"false:{false_ratio}:{name}"))
    return sorted(float(candidates[int(index)]) for index in rng.choice(len(candidates), size=count, replace=False))


def schedule_from_centers(total_frames, budget, centers):
    if not centers:
        # Explicit deterministic uniform fallback when all proxy bonus targets
        # are missing. Partial misses retain the registered distance ranking
        # around observed centers and still exhaust the exact bonus budget.
        return STRUCTURED.uniform_positions(total_frames, budget)
    frame_centers = np.arange(total_frames, dtype=np.float64)[:, None]
    distance = np.abs(frame_centers - np.asarray(centers, dtype=np.float64)[None, :]).min(axis=1)
    return STRUCTURED.offline_event_bonus_schedule(-distance, budget)


def decode_schedules_fast(names, schedules, dense_logits, start_indices, vocab, blank_id, cache=None):
    """Vectorized CPU equivalent of the registered triangular span-15 decoder."""
    hypotheses = []
    for name in names:
        starts = schedules[name]
        cache_key = (name, tuple(starts))
        if cache is not None and cache_key in cache:
            hypotheses.append(cache[cache_key])
            continue
        logits = REPLAY.extract_schedule_logits(name, dense_logits, start_indices, starts)
        logits = np.asarray(logits, dtype=np.float32)
        shifted = logits - logits.max(axis=1, keepdims=True)
        probabilities = np.exp(shifted)
        probabilities /= probabilities.sum(axis=1, keepdims=True)
        centers = np.asarray(starts, dtype=np.float32) + np.float32(7.5)
        distance = np.abs(centers[:, None] - centers[None, :])
        selected = distance <= np.float32(7.5)
        weights = np.where(
            selected,
            np.maximum(np.float32(1.0) - distance / np.float32(7.5), np.float32(0.05)),
            np.float32(0.0),
        )
        weights /= weights.sum(axis=1, keepdims=True)
        token_ids = (weights @ probabilities).argmax(axis=1).tolist()
        collapsed = [token for token, _ in groupby(token_ids) if token != blank_id]
        hypothesis = " ".join(REPLAY.map_phoenix_gloss(vocab[token]) for token in collapsed)
        hypothesis = REPLAY.clean_phoenix_2014_trans(hypothesis)
        hypotheses.append(hypothesis)
        if cache is not None:
            cache[cache_key] = hypothesis
    return hypotheses


def uniform_fallback_fill(total_frames, budget, selected):
    """Fill missing proxy-owned bonus slots by deterministic coverage."""
    selected = set(int(value) for value in selected)
    while len(selected) < budget:
        candidates = [index for index in range(total_frames) if index not in selected]
        choice = max(
            candidates,
            key=lambda index: (min(abs(index - other) for other in selected), -index),
        )
        selected.add(choice)
    return sorted(selected)


def dropout_schedule(total_frames, budget, centers, keep_ratio, seed, name):
    """Drop events and uniformly replace the bonus slots they owned."""
    keep = set(dropout_indices(len(centers), keep_ratio, seed, name))
    skeleton = STRUCTURED.uniform_positions(
        total_frames, STRUCTURED.skeleton_count(total_frames, budget)
    )
    skeleton_set = set(skeleton)
    bonus = [value for value in schedule_from_centers(total_frames, budget, centers)
             if value not in skeleton_set]
    retained_bonus = []
    for value in bonus:
        owner = min(range(len(centers)), key=lambda index: (abs(value - centers[index]), index))
        if owner in keep:
            retained_bonus.append(value)
    return uniform_fallback_fill(total_frames, budget, skeleton + retained_bonus)


def sentence_bootstrap(candidate_counts, baseline_counts, seed=260914, repetitions=10000):
    candidate = np.asarray([row["error"] for row in candidate_counts], dtype=np.float64)
    baseline = np.asarray([row["error"] for row in baseline_counts], dtype=np.float64)
    ref_len = np.asarray([row["ref_len"] for row in baseline_counts], dtype=np.int64)
    rng = np.random.default_rng(seed)
    values = np.empty(repetitions, dtype=np.float64)
    delta = candidate - baseline
    # Batched vectorization preserves the predeclared RNG stream while
    # avoiding 10k Python iterations for every condition.
    for begin in range(0, repetitions, 500):
        end = min(repetitions, begin + 500)
        draw = rng.integers(0, len(ref_len), size=(end - begin, len(ref_len)))
        values[begin:end] = 100.0 * delta[draw].sum(axis=1) / ref_len[draw].sum(axis=1)
    alpha = 0.05 / BOOTSTRAP_COMPARISONS
    return {
        "unit": "WER percentage points (candidate - baseline)",
        "sentence_bootstrap_seed": seed,
        "repetitions": repetitions,
        "ci95_unadjusted": [float(value) for value in np.quantile(values, [0.025, 0.975])],
        "ci_bonferroni_25_conditions": [
            float(value) for value in np.quantile(values, [alpha / 2, 1 - alpha / 2])
        ],
        "probability_less_than_zero": float(np.mean(values < 0.0)),
    }


def distribution(records, metric):
    values = np.asarray([
        record["metrics"][metric] if metric in record["metrics"] else record[metric]
        for record in records
    ], dtype=np.float64)
    return {
        "mean": float(values.mean()),
        "sample_std": float(values.std(ddof=1)),
        "min": float(values.min()),
        "median": float(np.median(values)),
        "max": float(values.max()),
    }


def summarize_random_condition(seed_records, representative_counts, mean_counts, uniform_metrics,
                               perfect_metrics, uniform_counts, perfect_counts):
    representative = seed_records[0]
    return {
        "seed_records": seed_records,
        "distribution": {
            metric: distribution(seed_records, metric)
            for metric in ("wer", "del", "ins", "sub", "error", "hypotheses_changed_vs_uniform")
        },
        "mean_delta_vs_uniform_wer_pp": float(distribution(seed_records, "wer")["mean"] - uniform_metrics["wer"]),
        "mean_delta_vs_perfect_wer_pp": float(distribution(seed_records, "wer")["mean"] - perfect_metrics["wer"]),
        "representative_seed": int(SEEDS[0]),
        "representative_selection": "first predeclared seed; not selected by result",
        "representative": representative,
        "representative_paired_bootstrap": {
            "vs_uniform": sentence_bootstrap(representative_counts, uniform_counts),
            "vs_perfect": sentence_bootstrap(representative_counts, perfect_counts),
        },
        "seed_mean_sentence_bootstrap": {
            "definition": "paired sentence bootstrap of each sentence's mean error over 30 seeds; separate from empirical seed variance",
            "vs_uniform": sentence_bootstrap(mean_counts, uniform_counts),
            "vs_perfect": sentence_bootstrap(mean_counts, perfect_counts),
        },
    }


def run_analysis(matrix_root, vocab_path, meta_path, center_label_path, max_samples=None):
    (
        names,
        b0_results,
        b0_logits,
        start_indices,
        a0_results,
        _,
        vocab,
        input_paths,
    ) = RANDOM.load_inputs(matrix_root, vocab_path, max_samples)
    references = [REPLAY.clean_phoenix_2014_trans(b0_results[name]["gls_ref"]) for name in names]
    blank_id = vocab.index("<blank>")
    a0_starts = REPLAY.starts_from_results(a0_results, names)
    segments = CENTER.load_segment_proxies(meta_path, center_label_path, names)["center_label"]
    true_centers = {
        name: [CENTER.segment_midpoint(start, end) for start, end in segments[name]] for name in names
    }
    budgets = {
        name: CENTER.budget_for(len(start_indices[name]), len(a0_starts[name]), "dense_50pct")
        for name in names
    }
    uniform_schedules = {
        name: STRUCTURED.uniform_positions(len(start_indices[name]), budgets[name]) for name in names
    }
    perfect_schedules = {
        name: schedule_from_centers(len(start_indices[name]), budgets[name], true_centers[name]) for name in names
    }
    decoder_cache = {}
    uniform_hyp = decode_schedules_fast(
        names, uniform_schedules, b0_logits, start_indices, vocab, blank_id, decoder_cache
    )
    perfect_hyp = decode_schedules_fast(
        names, perfect_schedules, b0_logits, start_indices, vocab, blank_id, decoder_cache
    )
    uniform_metrics = REPLAY.compact_metrics(references, uniform_hyp)
    perfect_metrics = REPLAY.compact_metrics(references, perfect_hyp)
    uniform_counts = [RANDOM.sentence_counts(r, h) for r, h in zip(references, uniform_hyp)]
    perfect_counts = [RANDOM.sentence_counts(r, h) for r, h in zip(references, perfect_hyp)]
    fixed_results = {}
    representative_rows = []

    for offset in OFFSETS:
        condition = f"offset_{offset:+d}"
        schedules = {
            name: schedule_from_centers(
                len(start_indices[name]), budgets[name],
                offset_centers(true_centers[name], offset, len(start_indices[name])),
            ) for name in names
        }
        hypotheses = decode_schedules_fast(
            names, schedules, b0_logits, start_indices, vocab, blank_id, decoder_cache
        )
        metrics = REPLAY.compact_metrics(references, hypotheses)
        counts = [RANDOM.sentence_counts(r, h) for r, h in zip(references, hypotheses)]
        fixed_results[condition] = {
            "offset_frames": offset,
            "interpretation": "negative is anticipatory; positive is delayed; neither is strict causal detection",
            "metrics": metrics,
            "delta_vs_uniform": RANDOM.metric_delta(metrics, uniform_metrics),
            "delta_vs_perfect": RANDOM.metric_delta(metrics, perfect_metrics),
            "hypotheses_changed_vs_uniform": int(sum(a != b for a, b in zip(hypotheses, uniform_hyp))),
            "hypotheses_changed_vs_perfect": int(sum(a != b for a, b in zip(hypotheses, perfect_hyp))),
            "paired_bootstrap": {
                "vs_uniform": sentence_bootstrap(counts, uniform_counts),
                "vs_perfect": sentence_bootstrap(counts, perfect_counts),
            },
        }
        for index, name in enumerate(names):
            representative_rows.append({
                "family": "fixed_offset", "condition": condition, "name": name,
                "clips": len(schedules[name]), "reference": references[index],
                "hypothesis": hypotheses[index], **counts[index],
            })

    def evaluate_random(family, condition, center_builder=None, schedule_builder=None):
        seed_records = []
        per_seed_counts = []
        first_counts = None
        first_hypotheses = None
        first_schedules = None
        for seed in SEEDS:
            if schedule_builder is not None:
                schedules = {name: schedule_builder(name, seed) for name in names}
            else:
                noisy = {name: center_builder(name, seed) for name in names}
                schedules = {
                    name: schedule_from_centers(len(start_indices[name]), budgets[name], noisy[name])
                    for name in names
                }
            if any(len(schedules[name]) != budgets[name] or len(set(schedules[name])) != budgets[name]
                   or schedules[name][0] != 0 or schedules[name][-1] != len(start_indices[name]) - 1
                   for name in names):
                raise AssertionError("noisy schedule violated exact budget/endpoints/uniqueness")
            hypotheses = decode_schedules_fast(
                names, schedules, b0_logits, start_indices, vocab, blank_id, decoder_cache
            )
            metrics = REPLAY.compact_metrics(references, hypotheses)
            counts = [RANDOM.sentence_counts(r, h) for r, h in zip(references, hypotheses)]
            per_seed_counts.append(counts)
            seed_records.append({
                "seed": seed,
                "metrics": metrics,
                "delta_vs_uniform": RANDOM.metric_delta(metrics, uniform_metrics),
                "delta_vs_perfect": RANDOM.metric_delta(metrics, perfect_metrics),
                "hypotheses_changed_vs_uniform": int(sum(a != b for a, b in zip(hypotheses, uniform_hyp))),
                "hypotheses_changed_vs_perfect": int(sum(a != b for a, b in zip(hypotheses, perfect_hyp))),
            })
            if seed == SEEDS[0]:
                first_counts, first_hypotheses, first_schedules = counts, hypotheses, schedules
        mean_counts = []
        for index in range(len(names)):
            mean_counts.append({
                "error": float(np.mean([counts[index]["error"] for counts in per_seed_counts])),
                "ref_len": uniform_counts[index]["ref_len"],
            })
        for index, name in enumerate(names):
            representative_rows.append({
                "family": family, "condition": condition, "seed": SEEDS[0], "name": name,
                "clips": len(first_schedules[name]), "reference": references[index],
                "hypothesis": first_hypotheses[index], **first_counts[index],
            })
        return summarize_random_condition(
            seed_records, first_counts, mean_counts, uniform_metrics, perfect_metrics,
            uniform_counts, perfect_counts,
        )

    random_results = {"jitter": {}, "dropout": {}, "false_positive": {}, "combined": {}}
    for amplitude in JITTER_AMPLITUDES:
        random_results["jitter"][f"amplitude_{amplitude}"] = evaluate_random(
            "jitter", f"amplitude_{amplitude}",
            lambda name, seed, amplitude=amplitude: jitter_centers(
                true_centers[name], amplitude, len(start_indices[name]), seed, name
            ),
        )
    for ratio in KEEP_RATIOS:
        random_results["dropout"][f"keep_{ratio:g}"] = evaluate_random(
            "dropout", f"keep_{ratio:g}",
            schedule_builder=lambda name, seed, ratio=ratio: dropout_schedule(
                len(start_indices[name]), budgets[name], true_centers[name], ratio, seed, name
            ),
        )
    for ratio in FALSE_RATIOS:
        random_results["false_positive"][f"ratio_{ratio:g}"] = evaluate_random(
            "false_positive", f"ratio_{ratio:g}",
            lambda name, seed, ratio=ratio: true_centers[name] + false_centers(
                true_centers[name], ratio, len(start_indices[name]), seed, name
            ),
        )

    def combined(name, seed):
        retained = dropout_centers(true_centers[name], 0.75, seed, f"combined:{name}")
        jittered = jitter_centers(retained, 4, len(start_indices[name]), seed, f"combined:{name}")
        false = false_centers(true_centers[name], 0.5, len(start_indices[name]), seed, f"combined:{name}")
        return jittered + false

    random_results["combined"]["jitter4_keep0.75_fp0.5"] = evaluate_random(
        "combined", "jitter4_keep0.75_fp0.5", combined
    )

    fixed_strong = [
        offset for offset in OFFSETS
        if -fixed_results[f"offset_{offset:+d}"]["delta_vs_uniform"]["wer_pp"] >= 0.5
        and fixed_results[f"offset_{offset:+d}"]["paired_bootstrap"]["vs_uniform"]["ci_bonferroni_25_conditions"][1] < 0
    ]

    def random_strong(record):
        return (
            -record["mean_delta_vs_uniform_wer_pp"] >= 0.5
            and record["seed_mean_sentence_bootstrap"]["vs_uniform"]["ci_bonferroni_25_conditions"][1] < 0
        )

    robust = {
        "fixed_offsets_meeting_strong_rule": fixed_strong,
        "maximum_absolute_fixed_offset_meeting_rule": max((abs(value) for value in fixed_strong), default=None),
        "jitter_amplitudes_meeting_rule": [
            amplitude for amplitude in JITTER_AMPLITUDES
            if random_strong(random_results["jitter"][f"amplitude_{amplitude}"])
        ],
        "keep_ratios_meeting_rule": [
            ratio for ratio in KEEP_RATIOS
            if random_strong(random_results["dropout"][f"keep_{ratio:g}"])
        ],
        "false_ratios_meeting_rule": [
            ratio for ratio in FALSE_RATIOS
            if random_strong(random_results["false_positive"][f"ratio_{ratio:g}"])
        ],
        "combined_meets_rule": random_strong(random_results["combined"]["jitter4_keep0.75_fp0.5"]),
        "rule": "seed-mean gain vs uniform >=.5 pp and Bonferroni-25 sentence CI excludes zero; seed distribution reported separately",
    }
    summary = {
        "sample_count": len(names),
        "full_dev": max_samples is None,
        "identity_warning": "Dev-label-derived offline midpoint noise diagnostic; not ground truth, deployable scheduling, or online observability evidence.",
        "fixed_protocol": {
            "proxy": "center-label nonblank midpoint",
            "budget": "round-half-up 50% dense independently per sample",
            "total_clips": sum(budgets.values()),
            "coordinate": "fixed B0 left pad 7",
            "schedule": "half-ceiling uniform skeleton plus center-distance bonus; dropout removes missed-center-owned bonus windows and fills them by deterministic farthest-point uniform coverage",
            "decoder": "centered16 plus triangular span15/min weight .05",
        },
        "references": {"uniform": uniform_metrics, "perfect_proxy": perfect_metrics},
        "fixed_offsets": fixed_results,
        "random_conditions": random_results,
        "robustness_thresholds": robust,
        "deployability_limit": (
            "Jitter/dropout/false-positive tests measure localization robustness after centers are supplied. They do not answer when an online system knows a center. "
            "Positive offsets approximate detection delay and negative offsets anticipatory prediction, but neither is strict causal evaluation."
        ),
        "predictor_target_bridge": (
            "Perfect midpoint requires segment end. A train-only predictor should instead use causal signness plus event-end probability and a bounded center-offset/phase target; "
            "evaluate calibration, recall, false positives, and latency without fitting dev labels."
        ),
    }
    if max_samples is None:
        if uniform_metrics["error"] != 864 or perfect_metrics["error"] != 807:
            raise AssertionError("registered uniform/perfect reference reproduction failed")
    input_paths.extend([
        meta_path, center_label_path, Path(__file__).resolve(), CENTER_PATH,
        CENTER.BOUNDARY_PATH, BOUNDARY.STRUCTURED_PATH, STRUCTURED.RANDOM_PATH, RANDOM.REPLAY_PATH,
    ])
    return summary, representative_rows, sorted(set(Path(path) for path in input_paths))


def write_outputs(output_root, config, summary, rows, input_paths):
    if output_root.exists() and any(output_root.iterdir()):
        raise FileExistsError(f"refusing to overwrite non-empty output directory: {output_root}")
    aggregate = output_root / "aggregate"
    aggregate.mkdir(parents=True, exist_ok=True)
    (output_root / "resolved_config.json").write_text(json.dumps(config, indent=2) + "\n")
    (aggregate / "dev_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    with (output_root / "representative_per_sample.jsonl").open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    manifest = {
        "manifest_version": 1,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "identity_warning": summary["identity_warning"],
        "policy": "repaired Phoenix dev only; CPU dense-logit noisy-center diagnostic",
        "inputs": {
            str(path.resolve()): {"bytes": path.stat().st_size, "sha256": REPLAY.sha256_file(path)}
            for path in input_paths if path.is_file()
        },
    }
    (output_root / "protocol_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matrix-root", type=Path, default=REPLAY.DEFAULT_MATRIX_ROOT)
    parser.add_argument("--vocab", type=Path, default=REPLAY.DEFAULT_VOCAB)
    parser.add_argument("--meta", type=Path, default=STRUCTURED.DEFAULT_META)
    parser.add_argument("--center-label", type=Path, default=STRUCTURED.DEFAULT_CENTER_LABEL)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--max-samples", type=int)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.max_samples is not None and args.max_samples < 1:
        parser.error("--max-samples must be positive")
    for path in (args.matrix_root, args.vocab, args.meta, args.center_label, args.output_root):
        REPLAY.reject_test_path(path)
    config = {
        "matrix_root": str(args.matrix_root.resolve()), "vocab": str(args.vocab.resolve()),
        "meta": str(args.meta.resolve()), "center_label": str(args.center_label.resolve()),
        "output_root": str(args.output_root.resolve()), "max_samples": args.max_samples,
        "device": "cpu", "offsets": list(OFFSETS), "jitter_amplitudes": list(JITTER_AMPLITUDES),
        "keep_ratios": list(KEEP_RATIOS), "false_ratios": list(FALSE_RATIOS),
        "seeds": list(SEEDS), "combined": "jitter4_keep0.75_fp0.5",
    }
    summary, rows, inputs = run_analysis(
        args.matrix_root.resolve(), args.vocab.resolve(), args.meta.resolve(),
        args.center_label.resolve(), args.max_samples,
    )
    print(json.dumps(summary, indent=2))
    if not args.dry_run:
        write_outputs(args.output_root.resolve(), config, summary, rows, inputs)


if __name__ == "__main__":
    main()
