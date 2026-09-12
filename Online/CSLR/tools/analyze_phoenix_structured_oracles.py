#!/usr/bin/env python3
"""Structured prediction-change schedule diagnostics on repaired Phoenix dev.

The registered main coordinate is B0's fixed padded-array coordinate (left
padding 7). All strategies use the same per-sample clip budget as frozen A0.
"""

import argparse
import gzip
import hashlib
import importlib.util
import json
import math
import pickle
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


CSLR_ROOT = Path(__file__).resolve().parents[1]
RANDOM_PATH = CSLR_ROOT / "tools/analyze_phoenix_random_schedule.py"
SPEC = importlib.util.spec_from_file_location("random_schedule", RANDOM_PATH)
RANDOM = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(RANDOM)
REPLAY = RANDOM.REPLAY

DEFAULT_OUTPUT_ROOT = (
    CSLR_ROOT / "results/phoenix-2014t_ISLR/p0_structured_oracles_v1_49faacc3"
)
DEFAULT_META = CSLR_ROOT.parents[1] / "data/phoenix_2014t/phoenix14t.dev"
DEFAULT_CENTER_LABEL = (
    CSLR_ROOT.parents[1] / "data/phoenix_2014t/phoenix_iso_center_label.dev"
)
DEFAULT_RELEASE_README = (
    CSLR_ROOT.parents[1] / "data/phoenix_2014t/PHOENIX-2014-T-release-v3/README"
)
SIGNALS = (
    "top1_change",
    "blank_transition",
    "js_divergence",
    "entropy_increase",
    "margin_drop",
)


def probabilities(logits):
    values = np.asarray(logits, dtype=np.float64)
    values = values - values.max(axis=1, keepdims=True)
    exponent = np.exp(values)
    return exponent / exponent.sum(axis=1, keepdims=True)


def compute_past_only_signals(logits, blank_id):
    """Signals at i depend only on logits i and i-1; index 0 is always zero."""
    prob = probabilities(logits)
    count = len(prob)
    top1 = prob.argmax(axis=1)
    top_two = np.partition(prob, -2, axis=1)[:, -2:]
    margin = top_two.max(axis=1) - top_two.min(axis=1)
    entropy = -(prob * np.log(np.clip(prob, 1e-300, None))).sum(axis=1)
    result = {name: np.zeros(count, dtype=np.float64) for name in SIGNALS}
    if count < 2:
        return result
    result["top1_change"][1:] = top1[1:] != top1[:-1]
    result["blank_transition"][1:] = (top1[1:] == blank_id) != (top1[:-1] == blank_id)
    midpoint = 0.5 * (prob[1:] + prob[:-1])
    result["js_divergence"][1:] = 0.5 * (
        (prob[1:] * (np.log(np.clip(prob[1:], 1e-300, None)) - np.log(midpoint))).sum(axis=1)
        + (prob[:-1] * (np.log(np.clip(prob[:-1], 1e-300, None)) - np.log(midpoint))).sum(axis=1)
    )
    result["entropy_increase"][1:] = np.maximum(entropy[1:] - entropy[:-1], 0.0)
    result["margin_drop"][1:] = np.maximum(margin[:-1] - margin[1:], 0.0)
    return result


def uniform_positions(total_frames, count):
    if not 1 <= count <= total_frames:
        raise ValueError("uniform count must satisfy 1 <= count <= total_frames")
    if count == 1:
        return [0]
    values = [int(math.floor(index * (total_frames - 1) / (count - 1) + 0.5)) for index in range(count)]
    if len(set(values)) != count or values[0] != 0 or values[-1] != total_frames - 1:
        raise AssertionError("uniform skeleton is not exact and endpoint-covering")
    return values


def skeleton_count(total_frames, budget):
    if total_frames == 1:
        return 1
    return min(budget, max(2, int(math.ceil(budget / 2))))


def offline_event_bonus_schedule(scores, budget):
    """Full-sequence ranking upper bound: half uniform, half event bonus."""
    total_frames = len(scores)
    skeleton = uniform_positions(total_frames, skeleton_count(total_frames, budget))
    selected = set(skeleton)
    while len(selected) < budget:
        candidates = [index for index in range(total_frames) if index not in selected]
        best_score = max(float(scores[index]) for index in candidates)
        ties = [index for index in candidates if float(scores[index]) == best_score]
        # Exact score ties use deterministic farthest-point coverage, not an
        # early-frame positional bias.
        choice = max(
            ties,
            key=lambda index: (min(abs(index - other) for other in selected), -index),
        )
        selected.add(choice)
    return sorted(selected)


def causal_paced_schedule(scores, budget, quantile=0.75, min_history=8):
    """Known-length paced policy whose event decisions read no future score."""
    total_frames = len(scores)
    skeleton = set(uniform_positions(total_frames, skeleton_count(total_frames, budget)))
    candidates = [index for index in range(total_frames) if index not in skeleton]
    bonus_budget = budget - len(skeleton)
    chosen = set(skeleton)
    history = []
    bonus_used = 0
    for seen, index in enumerate(candidates, start=1):
        due = math.floor(bonus_budget * seen / max(len(candidates), 1))
        ceiling = math.ceil(bonus_budget * seen / max(len(candidates), 1))
        event = False
        if len(history) >= min_history and float(scores[index]) > 0.0:
            event = float(scores[index]) > float(np.quantile(history, quantile))
        if bonus_used < due or (event and bonus_used < ceiling):
            chosen.add(index)
            bonus_used += 1
        history.append(float(scores[index]))
    if len(chosen) != budget:
        raise AssertionError("causal paced selection violated exact budget")
    return sorted(chosen)


def event_positions(scores, signal):
    values = np.asarray(scores)
    if signal in ("top1_change", "blank_transition"):
        return np.flatnonzero(values > 0.0).tolist(), "all positive binary transitions"
    positive_domain = values[1:]
    if not np.any(positive_domain > 0.0):
        return [], "no positive scores"
    threshold = float(np.quantile(positive_domain, 0.90))
    return np.flatnonzero((values >= threshold) & (values > 0.0)).tolist(), "per-sample top decile"


def aggregate_event_metrics(event_lists, schedules):
    event_count = exact = within_two = selected_on_event = selected_count = 0
    distances = []
    for events, selected in zip(event_lists, schedules):
        selected_set = set(selected)
        event_set = set(events)
        event_count += len(events)
        selected_count += len(selected)
        selected_on_event += len(selected_set & event_set)
        for event in events:
            distance = min(abs(event - center) for center in selected)
            distances.append(distance)
            exact += int(distance == 0)
            within_two += int(distance <= 2)
    return {
        "event_count": event_count,
        "exact_event_recall": float(exact / event_count) if event_count else None,
        "within_2_frames_event_recall": float(within_two / event_count) if event_count else None,
        "mean_event_to_selected_distance": float(np.mean(distances)) if distances else None,
        "selected_center_event_precision": float(selected_on_event / selected_count) if selected_count else None,
    }


def bootstrap_comparison(candidate_counts, baseline_counts, seed=260911, repetitions=10000, comparisons=10):
    candidate_error = np.asarray([row["error"] for row in candidate_counts], dtype=np.int64)
    baseline_error = np.asarray([row["error"] for row in baseline_counts], dtype=np.int64)
    ref_len = np.asarray([row["ref_len"] for row in candidate_counts], dtype=np.int64)
    rng = np.random.default_rng(seed)
    values = np.empty(repetitions, dtype=np.float64)
    for iteration in range(repetitions):
        draw = rng.integers(0, len(ref_len), len(ref_len))
        values[iteration] = 100.0 * (
            candidate_error[draw].sum() - baseline_error[draw].sum()
        ) / ref_len[draw].sum()
    alpha = 0.05 / comparisons
    return {
        "unit": "WER percentage points (candidate - baseline)",
        "sentence_bootstrap_seed": seed,
        "repetitions": repetitions,
        "ci95_unadjusted": [float(x) for x in np.quantile(values, [0.025, 0.975])],
        "ci_bonferroni_10_comparisons": [
            float(x) for x in np.quantile(values, [alpha / 2, 1 - alpha / 2])
        ],
        "probability_less_than_zero": float(np.mean(values < 0.0)),
    }


def alignment_audit(meta_path, center_label_path):
    REPLAY.reject_test_path(meta_path)
    REPLAY.reject_test_path(center_label_path)
    with gzip.open(meta_path, "rb") as handle:
        metadata = pickle.load(handle)
    with center_label_path.open("rb") as handle:
        center_records = pickle.load(handle)
    first_by_bag = {}
    for record in center_records:
        first_by_bag.setdefault(int(record["bag"]), record)
    by_video = defaultdict(list)
    for record in first_by_bag.values():
        by_video[record["video_file"]].append(record)
    pami_keys = sorted(set.intersection(*(set(item["alignments"]) for item in metadata)))
    pami_full_length = {
        key: sum(len(item["alignments"][key].split()) == item["num_frames"] for item in metadata)
        for key in pami_keys
    }
    exact_sequence = 0
    invalid_ranges = 0
    overlapping_nonblank_videos = 0
    for item in metadata:
        records = by_video[item["name"]]
        nonblank = [record for record in records if record["label"] != "<blank>"]
        labels = [record["label"].lower() for record in nonblank]
        exact_sequence += int(labels == [token.lower() for token in item["gloss"].split()])
        invalid_ranges += sum(
            not (0 <= record["start"] < record["end"] <= item["num_frames"])
            for record in records
        )
        overlapping_nonblank_videos += int(
            any(left["end"] > right["start"] for left, right in zip(nonblank, nonblank[1:]))
        )
    nonblank_count = sum(record["label"] != "<blank>" for record in first_by_bag.values())
    return {
        "decision": "no independently verified manual gloss temporal ground truth found",
        "candidates": {
            str(meta_path.resolve()): {
                "sample_coverage": f"{len(metadata)}/{len(metadata)}",
                "fields": ["num_frames", "gloss", f"alignments: {', '.join(pami_keys)}"],
                "alignment_lengths_equal_num_frames": pami_full_length,
                "semantics": "three full-length integer alignment sequences; provenance/model vocabulary not documented locally",
                "classification": "undocumented alignment-derived candidate; not verified manual ground truth",
            },
            str(center_label_path.resolve()): {
                "record_count": len(center_records),
                "unique_bags": len(first_by_bag),
                "video_coverage": f"{len(by_video)}/{len(metadata)}",
                "nonblank_base_segments": nonblank_count,
                "blank_base_segments": len(first_by_bag) - nonblank_count,
                "reference_token_sequences_equal_ignoring_case": f"{exact_sequence}/{len(metadata)}",
                "invalid_base_ranges": invalid_ranges,
                "videos_with_overlapping_adjacent_nonblank_segments": overlapping_nonblank_videos,
                "field_semantics": "start inclusive, end exclusive; sign_augment.py expands each upstream base segment into centered 16-frame items",
                "provenance_limit": "upstream phoenix_iso.dev is absent; local files do not establish independent human timing labels",
                "classification": "complete label-aware boundary proxy only; not ground truth",
            },
            "release_README": {
                "semantics": "corpus segment boundaries are mixed: mostly matched from Phoenix 2014, otherwise estimated by a training alignment; README warns some are incorrect",
                "classification": "does not establish independent per-gloss manual boundaries for this analysis",
            },
        },
        "minimum_unambiguous_boundary_proxy": (
            "Use only the first record per bag from phoenix_iso_center_label.dev; treat [start,end) of nonblank segments as label-aware proxy boundaries, "
            "publish overlap/range audit, and label all resulting selection as offline proxy oracle."
        ),
    }


def run_analysis(matrix_root, vocab_path, meta_path, center_label_path, max_samples=None):
    (
        names,
        b0_results,
        b0_logits,
        start_indices,
        a0_results,
        b2_results,
        vocab,
        input_paths,
    ) = RANDOM.load_inputs(matrix_root, vocab_path, max_samples)
    blank_id = vocab.index("<blank>")
    references = [REPLAY.clean_phoenix_2014_trans(b0_results[name]["gls_ref"]) for name in names]
    a0_saved = REPLAY.starts_from_results(a0_results, names)
    b2_schedules = REPLAY.starts_from_results(b2_results, names)
    a0_physical = {}
    for name in names:
        a0_physical[name], _, _ = REPLAY.align_starts_to_dense_inputs(
            list(range(len(start_indices[name]))), a0_saved[name]
        )
    matched_uniform = {
        name: uniform_positions(len(start_indices[name]), len(a0_saved[name])) for name in names
    }
    baseline_schedules = {"B2": b2_schedules, "A0": a0_physical, "U_A0_budget": matched_uniform}
    baseline_hypotheses = {
        label: RANDOM.decode_schedules(names, schedules, b0_logits, start_indices, vocab, blank_id)
        for label, schedules in baseline_schedules.items()
    }
    baseline_metrics = {
        label: REPLAY.compact_metrics(references, hypotheses)
        for label, hypotheses in baseline_hypotheses.items()
    }
    baseline_counts = {
        label: [RANDOM.sentence_counts(r, h) for r, h in zip(references, hypotheses)]
        for label, hypotheses in baseline_hypotheses.items()
    }
    all_signals = {name: compute_past_only_signals(b0_logits[name], blank_id) for name in names}
    strategies = {}
    per_sample_rows = []
    random_summary_path = RANDOM.DEFAULT_OUTPUT_ROOT / "aggregate/dev_summary.json"
    random_distribution = None
    if max_samples is None and random_summary_path.is_file():
        random_distribution = json.loads(random_summary_path.read_text())["random_protocols"]["per_sample_exact"]["distribution"]
        input_paths.append(random_summary_path)

    for policy in ("offline_full_sequence_rank", "past_only_causal_paced_known_length"):
        for signal in SIGNALS:
            strategy_id = f"{policy}__{signal}"
            schedules = {}
            event_lists = []
            event_definitions = set()
            for name in names:
                scores = all_signals[name][signal]
                budget = len(a0_saved[name])
                if policy == "offline_full_sequence_rank":
                    schedule = offline_event_bonus_schedule(scores, budget)
                else:
                    schedule = causal_paced_schedule(scores, budget)
                if len(schedule) != budget or len(set(schedule)) != budget:
                    raise AssertionError("structured schedule violated exact per-sample budget")
                if schedule[0] != 0 or schedule[-1] != len(scores) - 1:
                    raise AssertionError("structured schedule violated endpoint coverage")
                schedules[name] = schedule
                events, definition = event_positions(scores, signal)
                event_lists.append(events)
                event_definitions.add(definition)
            hypotheses = RANDOM.decode_schedules(
                names, schedules, b0_logits, start_indices, vocab, blank_id
            )
            metrics = REPLAY.compact_metrics(references, hypotheses)
            counts = [RANDOM.sentence_counts(r, h) for r, h in zip(references, hypotheses)]
            comparison = {}
            for baseline in ("B2", "A0", "U_A0_budget"):
                comparison[baseline] = {
                    "delta": RANDOM.metric_delta(metrics, baseline_metrics[baseline]),
                    "hypotheses_changed": int(sum(
                        left != right for left, right in zip(hypotheses, baseline_hypotheses[baseline])
                    )),
                    "paired_bootstrap": bootstrap_comparison(counts, baseline_counts[baseline]),
                }
            if random_distribution is not None:
                seed_wers = np.asarray([
                    record["metrics"]["wer"]
                    for record in json.loads(random_summary_path.read_text())["random_protocols"]["per_sample_exact"]["seed_records"]
                ])
                random_comparison = {
                    "fraction_random_seed_wer_at_or_below_strategy": float(np.mean(seed_wers <= metrics["wer"])),
                    "random_mean_minus_strategy_wer_pp": float(random_distribution["wer"]["mean"] - metrics["wer"]),
                    "note": "descriptive 30-seed scheduler distribution; not pooled with sentence bootstrap",
                }
            else:
                random_comparison = None
            strategies[strategy_id] = {
                "policy_classification": (
                    "offline upper-bound diagnostic; full sequence scores are ranked before selection"
                    if policy == "offline_full_sequence_rank"
                    else "past-only score decisions with known T/K quota pacing; not unknown-EOS streaming"
                ),
                "signal_read_range": "current and immediately previous dense logit only",
                "budget": "exact per-sample A0 clip count; half-ceiling uniform skeleton",
                "metrics": metrics,
                "comparisons": comparison,
                "events": {
                    "definitions": sorted(event_definitions),
                    **aggregate_event_metrics(event_lists, [schedules[name] for name in names]),
                },
                "random_distribution_comparison": random_comparison,
            }
            for index, name in enumerate(names):
                per_sample_rows.append({
                    "strategy": strategy_id,
                    "name": name,
                    "clips": len(schedules[name]),
                    "reference": references[index],
                    "hypothesis": hypotheses[index],
                    **counts[index],
                })

    best_id = min(strategies, key=lambda key: strategies[key]["metrics"]["wer"])
    best = strategies[best_id]
    gain_vs_b2 = -best["comparisons"]["B2"]["delta"]["wer_pp"]
    adjusted_upper = best["comparisons"]["B2"]["paired_bootstrap"]["ci_bonferroni_10_comparisons"][1]
    if gain_vs_b2 >= 0.5 and adjusted_upper < 0:
        decision = "strong_go"
    elif gain_vs_b2 >= 0.2:
        decision = "weak_go_exploratory"
    else:
        decision = "no_go_for_prediction_change_scheduler"
    summary = {
        "sample_count": len(names),
        "full_dev": max_samples is None,
        "preregistered_main_coordinate": "fixed B0 padded-array starts; constant left padding 7",
        "coordinate_reason": "independent of final schedule start and exactly covered by dense B0 logits",
        "window_decoder": "centered 16-frame inputs; triangular span-weighted-15; min weight 0.05",
        "exploration_notice": "10 predeclared strategy combinations; best_id is post-hoc descriptive and Bonferroni 10-comparison CI is reported",
        "alignment_audit": alignment_audit(meta_path, center_label_path),
        "references": {
            "frozen_physical_baselines": {
                "B2": baseline_metrics["B2"],
                "A0": baseline_metrics["A0"],
            },
            "fixed_coordinate_uniform_exact_A0_per_sample_budget": baseline_metrics["U_A0_budget"],
        },
        "strategies": strategies,
        "posthoc_best": {
            "strategy": best_id,
            "metrics": best["metrics"],
            "gain_vs_B2_wer_pp": gain_vs_b2,
            "gain_vs_exact_budget_uniform_wer_pp": -best["comparisons"]["U_A0_budget"]["delta"]["wer_pp"],
            "decision": decision,
        },
        "conclusion": {
            "prediction_change_upper_bound": "not observed under the registered half-uniform/half-event design",
            "strong_go_rule": "gain versus B2 >= 0.5 pp and Bonferroni-adjusted CI excludes zero",
            "observed_decision": decision,
            "recommended_next": (
                "Review decoder/schedule coupling first. If boundary work continues, manually audit a fixed subset before running the explicitly label-aware "
                "center-label proxy oracle. Do not train a causal scheduler or spend GPU on these prediction-change signals yet."
            ),
        },
        "streaming_limit": "Signals are one-step past-only, but centered inputs require lookahead; causal-paced policy also assumes known T and K.",
    }
    if max_samples is None:
        for label, replay_id in (
            ("B2", "B2_uniform_rate_span15_from_B0"),
            ("A0", "A0_adaptive_span15_from_B0"),
        ):
            expected = REPLAY.EXPECTED_FULL_DEV[replay_id]
            observed = baseline_metrics[label]
            if observed["error"] != expected["error"] or abs(observed["wer"] - expected["wer"]) > 1e-12:
                raise AssertionError(f"{label} no longer reproduces frozen repaired dev")
    input_paths.extend([
        meta_path,
        center_label_path,
        DEFAULT_RELEASE_README,
        Path(__file__).resolve(),
        RANDOM_PATH,
        RANDOM.REPLAY_PATH,
    ])
    return summary, per_sample_rows, sorted(set(Path(path) for path in input_paths))


def write_outputs(output_root, config, summary, rows, input_paths):
    if output_root.exists() and any(output_root.iterdir()):
        raise FileExistsError(f"refusing to overwrite non-empty output directory: {output_root}")
    aggregate = output_root / "aggregate"
    aggregate.mkdir(parents=True, exist_ok=True)
    (output_root / "resolved_config.json").write_text(json.dumps(config, indent=2) + "\n")
    (aggregate / "dev_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    with (output_root / "per_sample_results.jsonl").open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    manifest = {
        "manifest_version": 1,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "policy": "repaired Phoenix-2014T dev only; CPU dense-logit structured diagnostics",
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
    parser.add_argument("--meta", type=Path, default=DEFAULT_META)
    parser.add_argument("--center-label", type=Path, default=DEFAULT_CENTER_LABEL)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--max-samples", type=int)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.max_samples is not None and args.max_samples < 1:
        parser.error("--max-samples must be positive")
    for path in (args.matrix_root, args.vocab, args.meta, args.center_label, args.output_root):
        REPLAY.reject_test_path(path)
    config = {
        "matrix_root": str(args.matrix_root.resolve()),
        "vocab": str(args.vocab.resolve()),
        "meta": str(args.meta.resolve()),
        "center_label": str(args.center_label.resolve()),
        "output_root": str(args.output_root.resolve()),
        "max_samples": args.max_samples,
        "device": "cpu",
        "signals": list(SIGNALS),
        "policies": ["offline_full_sequence_rank", "past_only_causal_paced_known_length"],
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
