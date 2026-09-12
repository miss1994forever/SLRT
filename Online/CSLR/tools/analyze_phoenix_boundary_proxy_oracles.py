#!/usr/bin/env python3
"""Label-aware boundary-proxy schedule diagnostics on repaired Phoenix dev.

These are alignment-derived offline proxies, not manual ground truth and not
deployable boundary detectors. The registered coordinate is fixed B0 padding.
"""

import argparse
import gzip
import importlib.util
import json
import math
import pickle
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


CSLR_ROOT = Path(__file__).resolve().parents[1]
STRUCTURED_PATH = CSLR_ROOT / "tools/analyze_phoenix_structured_oracles.py"
SPEC = importlib.util.spec_from_file_location("structured_oracles", STRUCTURED_PATH)
STRUCTURED = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(STRUCTURED)
RANDOM = STRUCTURED.RANDOM
REPLAY = STRUCTURED.REPLAY

DEFAULT_OUTPUT_ROOT = (
    CSLR_ROOT / "results/phoenix-2014t_ISLR/p0_boundary_proxy_oracles_v1_49faacc3"
)
PROXIES = ("center_label", "pami0", "pami1", "pami2")
VARIANTS = ("start_only", "end_only", "start_and_end")


def first_record_per_bag(records):
    first = {}
    for record in records:
        first.setdefault(int(record["bag"]), record)
    return first


def nonzero_runs(values):
    """Return inclusive frame-center starts/ends for constant nonzero runs."""
    starts, ends = [], []
    run_start = 0
    for index in range(1, len(values) + 1):
        if index == len(values) or values[index] != values[run_start]:
            if values[run_start] != 0:
                starts.append(run_start)
                ends.append(index - 1)
            run_start = index
    return starts, ends


def load_proxies(meta_path, center_label_path, selected_names=None):
    REPLAY.reject_test_path(meta_path)
    REPLAY.reject_test_path(center_label_path)
    with gzip.open(meta_path, "rb") as handle:
        metadata = pickle.load(handle)
    with center_label_path.open("rb") as handle:
        center_records = pickle.load(handle)
    if selected_names is not None:
        selected = set(selected_names)
        metadata = [item for item in metadata if item["name"] in selected]
        if [item["name"] for item in metadata] != list(selected_names):
            raise ValueError("metadata sample order differs from dense B0")
    meta_by_name = {item["name"]: item for item in metadata}
    first_by_bag = first_record_per_bag(center_records)
    base_by_video = defaultdict(list)
    for bag, record in first_by_bag.items():
        if record["video_file"] in meta_by_name:
            base_by_video[record["video_file"]].append((bag, record))

    proxies = {proxy: {} for proxy in PROXIES}
    invalid_ranges = empty_segments = duplicate_segments = missing_base_fields = 0
    overlapping_nonblank_videos = overlapping_pairs = exact_reference_sequences = 0
    seen_segments = set()
    nonblank_segments = blank_segments = 0
    for item in metadata:
        name, length = item["name"], int(item["num_frames"])
        records = [record for _, record in sorted(base_by_video[name])]
        nonblank = []
        for record in records:
            missing_base_fields += int("base_start" not in record or "base_end" not in record)
            start = int(record.get("base_start", record["start"]))
            end = int(record.get("base_end", record["end"]))
            empty_segments += int(start >= end)
            invalid_ranges += int(not (0 <= start < end <= length))
            key = (name, record["label"], start, end)
            duplicate_segments += int(key in seen_segments)
            seen_segments.add(key)
            if record["label"] == "<blank>":
                blank_segments += 1
            else:
                nonblank_segments += 1
                nonblank.append((record["label"], start, end))
        exact_reference_sequences += int(
            [label.lower() for label, _, _ in nonblank]
            == [token.lower() for token in item["gloss"].split()]
        )
        overlaps = sum(left[2] > right[1] for left, right in zip(nonblank, nonblank[1:]))
        overlapping_pairs += overlaps
        overlapping_nonblank_videos += int(overlaps > 0)
        proxies["center_label"][name] = {
            "starts": sorted(set(start for _, start, _ in nonblank)),
            # Convert [start,end) to the last included frame-center so all
            # proxy coordinates remain in the selectable domain 0..T-1.
            "ends": sorted(set(end - 1 for _, _, end in nonblank)),
        }
        for key in ("pami0", "pami1", "pami2"):
            values = [int(value) for value in item["alignments"][key].split()]
            if len(values) != length:
                raise ValueError(f"{key} length differs from num_frames: {name}")
            starts, ends = nonzero_runs(values)
            proxies[key][name] = {"starts": starts, "ends": ends}

    audit = {
        "identity_warning": (
            "All four sources are label-aware/alignment-derived offline proxies, not independently verified manual ground truth, "
            "not deployable methods, and not evidence that a causal boundary detector works."
        ),
        "center_label": {
            "rule": "first stored record per bag; nonblank [base_start,base_end), falling back to stored [start,end) because base fields are absent; end proxy uses end-1 frame-center",
            "sample_coverage": f"{len(base_by_video)}/{len(metadata)}",
            "nonblank_segments": nonblank_segments,
            "blank_segments_excluded": blank_segments,
            "reference_sequences_equal_ignoring_case": f"{exact_reference_sequences}/{len(metadata)}",
            "invalid_ranges": invalid_ranges,
            "empty_segments": empty_segments,
            "duplicate_base_segments": duplicate_segments,
            "records_missing_base_start_or_base_end": missing_base_fields,
            "videos_with_adjacent_nonblank_overlap": overlapping_nonblank_videos,
            "overlapping_adjacent_nonblank_pairs": overlapping_pairs,
            "provenance_limitation": (
                "sign_augment.py shows first records originate from upstream segments, but phoenix_iso.dev is absent; "
                "human-versus-forced-alignment provenance cannot be resolved locally."
            ),
        },
        "pami": {
            "rule": (
                "parse each full-length integer sequence; provisionally treat integer 0 as blank; each maximal constant nonzero run is a segment; "
                "start is its first frame-center and end is its last frame-center; every nonzero integer transition closes and opens runs"
            ),
            "field_semantics_uncertainty": (
                "pami0/pami1/pami2 vocabulary and generator are undocumented locally; integer changes may be HMM/state changes rather than gloss changes"
            ),
            "sample_coverage": f"{len(metadata)}/{len(metadata)} for each pami field",
        },
    }
    return proxies, audit


def boundary_list(proxy_record, variant):
    if variant == "start_only":
        return proxy_record["starts"]
    if variant == "end_only":
        return proxy_record["ends"]
    if variant == "start_and_end":
        return sorted(set(proxy_record["starts"]) | set(proxy_record["ends"]))
    raise ValueError(f"unknown boundary variant: {variant}")


def boundary_distance_scores(total_frames, boundaries):
    if not boundaries:
        return np.zeros(total_frames, dtype=np.float64)
    centers = np.arange(total_frames, dtype=np.float64)[:, None]
    distance = np.abs(centers - np.asarray(boundaries, dtype=np.float64)[None, :]).min(axis=1)
    return -distance


def boundary_schedule(total_frames, budget, boundaries):
    scores = boundary_distance_scores(total_frames, boundaries)
    return STRUCTURED.offline_event_bonus_schedule(scores, budget)


def boundary_coverage(boundary_lists, schedules):
    count = exact = within_two = within_four = selected_near_two = selected_count = 0
    distances = []
    density_two, density_four = [], []
    for boundaries, selected in zip(boundary_lists, schedules):
        selected_set = set(selected)
        selected_count += len(selected)
        for boundary in boundaries:
            distance = min(abs(boundary - center) for center in selected)
            distances.append(distance)
            count += 1
            exact += int(distance == 0)
            within_two += int(distance <= 2)
            within_four += int(distance <= 4)
            density_two.append(sum(abs(boundary - center) <= 2 for center in selected))
            density_four.append(sum(abs(boundary - center) <= 4 for center in selected))
        selected_near_two += sum(
            any(abs(center - boundary) <= 2 for boundary in boundaries) for center in selected_set
        )
    return {
        "boundary_count": count,
        "exact_recall": float(exact / count) if count else None,
        "within_2_recall": float(within_two / count) if count else None,
        "within_4_recall": float(within_four / count) if count else None,
        "mean_boundary_to_selected_distance": float(np.mean(distances)) if distances else None,
        "mean_selected_centers_within_2_per_boundary": float(np.mean(density_two)) if density_two else None,
        "mean_selected_centers_within_4_per_boundary": float(np.mean(density_four)) if density_four else None,
        "selected_center_within_2_of_boundary_fraction": float(selected_near_two / selected_count) if selected_count else None,
    }


def directed_tolerance(left_by_name, right_by_name, names, tolerance):
    matched = total = 0
    for name in names:
        right = right_by_name[name]
        for boundary in left_by_name[name]:
            total += 1
            matched += int(bool(right) and min(abs(boundary - other) for other in right) <= tolerance)
    return float(matched / total) if total else None


def pairwise_agreement(proxies, names):
    unions = {
        proxy: {
            name: boundary_list(proxies[proxy][name], "start_and_end") for name in names
        }
        for proxy in PROXIES
    }
    result = {}
    for left_index, left in enumerate(PROXIES):
        for right in PROXIES[left_index + 1:]:
            record = {
                "left_boundary_count": sum(len(unions[left][name]) for name in names),
                "right_boundary_count": sum(len(unions[right][name]) for name in names),
            }
            for tolerance in (0, 2, 4):
                left_to_right = directed_tolerance(unions[left], unions[right], names, tolerance)
                right_to_left = directed_tolerance(unions[right], unions[left], names, tolerance)
                record[f"tolerance_{tolerance}"] = {
                    "left_to_right_recall": left_to_right,
                    "right_to_left_recall": right_to_left,
                    "symmetric_mean": float((left_to_right + right_to_left) / 2),
                }
            result[f"{left}__{right}"] = record
    return result


def paired_bootstrap(candidate_counts, baseline_counts, seed=260912, repetitions=10000, comparisons=12):
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
        "ci95_unadjusted": [float(value) for value in np.quantile(values, [0.025, 0.975])],
        "ci_bonferroni_12_strategies": [
            float(value) for value in np.quantile(values, [alpha / 2, 1 - alpha / 2])
        ],
        "probability_less_than_zero": float(np.mean(values < 0.0)),
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
    uniform_exact = {}
    for name in names:
        total_frames = len(start_indices[name])
        a0_physical[name], _, _ = REPLAY.align_starts_to_dense_inputs(
            list(range(total_frames)), a0_saved[name]
        )
        uniform_exact[name] = STRUCTURED.uniform_positions(total_frames, len(a0_saved[name]))
    baseline_schedules = {"B2": b2_schedules, "A0": a0_physical, "U_A0_budget": uniform_exact}
    baseline_hypotheses = {
        label: RANDOM.decode_schedules(names, schedules, b0_logits, start_indices, vocab, blank_id)
        for label, schedules in baseline_schedules.items()
    }
    baseline_metrics = {
        label: REPLAY.compact_metrics(references, hypotheses)
        for label, hypotheses in baseline_hypotheses.items()
    }
    baseline_counts = {
        label: [RANDOM.sentence_counts(reference, hypothesis) for reference, hypothesis in zip(references, hypotheses)]
        for label, hypotheses in baseline_hypotheses.items()
    }
    proxies, proxy_audit = load_proxies(meta_path, center_label_path, names)
    proxy_counts = {
        proxy: {
            variant: sum(len(boundary_list(proxies[proxy][name], variant)) for name in names)
            for variant in VARIANTS
        }
        for proxy in PROXIES
    }
    random_summary_path = RANDOM.DEFAULT_OUTPUT_ROOT / "aggregate/dev_summary.json"
    random_seed_wers = None
    random_mean = None
    if max_samples is None and random_summary_path.is_file():
        random_summary = json.loads(random_summary_path.read_text())
        random_records = random_summary["random_protocols"]["per_sample_exact"]["seed_records"]
        random_seed_wers = np.asarray([record["metrics"]["wer"] for record in random_records])
        random_mean = random_summary["random_protocols"]["per_sample_exact"]["distribution"]["wer"]["mean"]
        input_paths.append(random_summary_path)

    strategies = {}
    rows = []
    for proxy in PROXIES:
        for variant in VARIANTS:
            strategy_id = f"{proxy}__{variant}"
            schedules = {}
            boundaries_by_sample = []
            for name in names:
                boundaries = boundary_list(proxies[proxy][name], variant)
                schedule = boundary_schedule(
                    len(start_indices[name]), len(a0_saved[name]), boundaries
                )
                if len(schedule) != len(a0_saved[name]) or len(set(schedule)) != len(schedule):
                    raise AssertionError("boundary schedule violated exact per-sample budget")
                if schedule[0] != 0 or schedule[-1] != len(start_indices[name]) - 1:
                    raise AssertionError("boundary schedule violated endpoint coverage")
                schedules[name] = schedule
                boundaries_by_sample.append(boundaries)
            hypotheses = RANDOM.decode_schedules(
                names, schedules, b0_logits, start_indices, vocab, blank_id
            )
            metrics = REPLAY.compact_metrics(references, hypotheses)
            counts = [RANDOM.sentence_counts(reference, hypothesis) for reference, hypothesis in zip(references, hypotheses)]
            comparisons = {}
            for baseline in ("U_A0_budget", "B2", "A0"):
                comparisons[baseline] = {
                    "delta": RANDOM.metric_delta(metrics, baseline_metrics[baseline]),
                    "hypotheses_changed": int(sum(
                        left != right for left, right in zip(hypotheses, baseline_hypotheses[baseline])
                    )),
                    "paired_bootstrap": paired_bootstrap(counts, baseline_counts[baseline]),
                }
            if random_seed_wers is None:
                random_comparison = None
            else:
                random_comparison = {
                    "fraction_random_seed_wer_at_or_below_strategy": float(np.mean(random_seed_wers <= metrics["wer"])),
                    "random_mean_minus_strategy_wer_pp": float(random_mean - metrics["wer"]),
                    "note": "descriptive 30-seed distribution; not pooled with sentence bootstrap",
                }
            strategies[strategy_id] = {
                "identity": "label-aware/alignment-derived offline proxy oracle; not ground truth or deployable",
                "metrics": metrics,
                "comparisons": comparisons,
                "boundary_coverage": boundary_coverage(
                    boundaries_by_sample, [schedules[name] for name in names]
                ),
                "random_distribution_comparison": random_comparison,
            }
            for index, name in enumerate(names):
                rows.append({
                    "strategy": strategy_id,
                    "name": name,
                    "clips": len(schedules[name]),
                    "boundary_count": len(boundaries_by_sample[index]),
                    "reference": references[index],
                    "hypothesis": hypotheses[index],
                    **counts[index],
                })

    main_ids = [f"center_label__{variant}" for variant in VARIANTS]
    best_main_id = min(main_ids, key=lambda key: strategies[key]["metrics"]["wer"])
    best_all_id = min(strategies, key=lambda key: strategies[key]["metrics"]["wer"])
    main_comparison = strategies[best_main_id]["comparisons"]["U_A0_budget"]
    main_gain = -main_comparison["delta"]["wer_pp"]
    adjusted_upper = main_comparison["paired_bootstrap"]["ci_bonferroni_12_strategies"][1]
    main_variant = best_main_id.split("__", 1)[1]
    consistent_proxy_count = sum(
        strategies[f"{proxy}__{main_variant}"]["comparisons"]["U_A0_budget"]["delta"]["wer_pp"] < 0
        for proxy in PROXIES
    )
    if main_gain >= 0.5 and adjusted_upper < 0 and consistent_proxy_count >= 2:
        decision = "strong_go"
    elif main_gain >= 0.2 and consistent_proxy_count >= 2:
        decision = "weak_go_exploratory"
    else:
        decision = "no_go"
    summary = {
        "sample_count": len(names),
        "full_dev": max_samples is None,
        "identity_warning": proxy_audit["identity_warning"],
        "preregistered_main_proxy": "center_label first-record-per-bag nonblank segments",
        "preregistered_main_coordinate": "fixed B0 padded-array starts; constant left padding 7",
        "schedule": "half-ceiling uniform skeleton plus nearest-boundary bonus; exact per-sample A0 budget; endpoints 0/T-1; unique centers",
        "window_decoder": "centered 16-frame; triangular span-weighted-15; min weight 0.05; no tuning",
        "exploration_notice": "12 strategies were predeclared and all are reported; best identifiers are post-hoc descriptions",
        "proxy_audit": proxy_audit,
        "proxy_boundary_counts": proxy_counts,
        "proxy_pairwise_tolerance_agreement": pairwise_agreement(proxies, names),
        "references": {
            "fixed_coordinate_uniform_exact_A0_per_sample_budget": baseline_metrics["U_A0_budget"],
            "frozen_B2": baseline_metrics["B2"],
            "frozen_A0": baseline_metrics["A0"],
        },
        "strategies": strategies,
        "decision": {
            "strong_go_rule": "main proxy gain vs U >= 0.5 pp, Bonferroni-adjusted CI excludes zero, and same variant improves for at least two proxies",
            "weak_go_rule": "main proxy gain vs U >= 0.2 pp with at least two proxies improving in the same variant",
            "posthoc_best_main": best_main_id,
            "posthoc_best_all": best_all_id,
            "best_main_gain_vs_U_wer_pp": main_gain,
            "same_variant_improving_proxy_count": consistent_proxy_count,
            "outcome": decision,
        },
        "limitations": [
            "Proxy boundaries consume dev labels/alignments and cannot be deployed.",
            "PAMI integer semantics and center-label human/forced provenance are unresolved.",
            "Centered inputs and symmetric span decoding are not strict zero-lookahead streaming.",
            "A proxy oracle result cannot establish that a causal learned boundary detector will reproduce it.",
        ],
    }
    if max_samples is None:
        for label, replay_id in (("B2", "B2_uniform_rate_span15_from_B0"), ("A0", "A0_adaptive_span15_from_B0")):
            expected = REPLAY.EXPECTED_FULL_DEV[replay_id]
            if baseline_metrics[label]["error"] != expected["error"]:
                raise AssertionError(f"{label} failed frozen reproduction")
    input_paths.extend([
        meta_path,
        center_label_path,
        Path(__file__).resolve(),
        STRUCTURED_PATH,
        RANDOM.RANDOM_PATH if hasattr(RANDOM, "RANDOM_PATH") else STRUCTURED.RANDOM_PATH,
        RANDOM.REPLAY_PATH,
        STRUCTURED.DEFAULT_RELEASE_README,
    ])
    return summary, rows, sorted(set(Path(path) for path in input_paths))


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
        "identity_warning": summary["identity_warning"],
        "policy": "repaired Phoenix-2014T dev only; CPU dense-logit proxy oracle",
        "provenance_limitation": summary["proxy_audit"],
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
        "matrix_root": str(args.matrix_root.resolve()),
        "vocab": str(args.vocab.resolve()),
        "meta": str(args.meta.resolve()),
        "center_label": str(args.center_label.resolve()),
        "output_root": str(args.output_root.resolve()),
        "max_samples": args.max_samples,
        "device": "cpu",
        "proxies": list(PROXIES),
        "variants": list(VARIANTS),
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
