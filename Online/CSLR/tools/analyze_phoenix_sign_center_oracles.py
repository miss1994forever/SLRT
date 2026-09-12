#!/usr/bin/env python3
"""Offline label/alignment-derived sign-center proxy diagnostics on dev only.

The proxies are not manual ground truth and the schedules are not deployable.
All logits come from the repaired-dev dense B0 run in fixed-pad coordinates.
"""

import argparse
import gzip
import importlib.util
import json
import math
import pickle
from collections import defaultdict
from datetime import datetime, timezone
from itertools import groupby
from pathlib import Path

import numpy as np
import torch


CSLR_ROOT = Path(__file__).resolve().parents[1]
BOUNDARY_PATH = CSLR_ROOT / "tools/analyze_phoenix_boundary_proxy_oracles.py"
SPEC = importlib.util.spec_from_file_location("boundary_proxy", BOUNDARY_PATH)
BOUNDARY = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(BOUNDARY)
STRUCTURED = BOUNDARY.STRUCTURED
RANDOM = BOUNDARY.RANDOM
REPLAY = BOUNDARY.REPLAY

DEFAULT_OUTPUT_ROOT = (
    CSLR_ROOT / "results/phoenix-2014t_ISLR/p0_sign_center_oracles_v1_49faacc3"
)
PROXIES = BOUNDARY.PROXIES
CENTER_VARIANTS = ("midpoint", "center_band_50")
BUDGETS = ("a0_exact", "dense_50pct", "dense_33pct")
DECODERS = ("span15", "span7", "direct_span1")


def segment_midpoint(start, end_exclusive):
    if end_exclusive <= start:
        raise ValueError("segment must be non-empty")
    return (float(start) + float(end_exclusive) - 1.0) / 2.0


def segment_center_band(start, end_exclusive):
    """Closed central-50% interval over included frame-center endpoints."""
    if end_exclusive <= start:
        raise ValueError("segment must be non-empty")
    last = float(end_exclusive - 1)
    span = last - float(start)
    return float(start) + 0.25 * span, float(start) + 0.75 * span


def load_segment_proxies(meta_path, center_label_path, selected_names):
    REPLAY.reject_test_path(meta_path)
    REPLAY.reject_test_path(center_label_path)
    with gzip.open(meta_path, "rb") as handle:
        all_meta = pickle.load(handle)
    meta = [item for item in all_meta if item["name"] in set(selected_names)]
    if [item["name"] for item in meta] != list(selected_names):
        raise ValueError("metadata sample order differs from dense B0")
    with center_label_path.open("rb") as handle:
        records = pickle.load(handle)
    first = BOUNDARY.first_record_per_bag(records)
    by_video = defaultdict(list)
    for bag, record in first.items():
        if record["video_file"] in set(selected_names) and record["label"] != "<blank>":
            by_video[record["video_file"]].append((bag, record))
    proxies = {proxy: {} for proxy in PROXIES}
    for item in meta:
        name = item["name"]
        segments = []
        for _, record in sorted(by_video[name]):
            start = int(record.get("base_start", record["start"]))
            end = int(record.get("base_end", record["end"]))
            if not 0 <= start < end <= item["num_frames"]:
                raise ValueError(f"invalid center-label segment: {name}")
            segments.append((start, end))
        proxies["center_label"][name] = segments
        for key in ("pami0", "pami1", "pami2"):
            values = [int(value) for value in item["alignments"][key].split()]
            if len(values) != item["num_frames"]:
                raise ValueError(f"{key} length differs from num_frames: {name}")
            starts, ends = BOUNDARY.nonzero_runs(values)
            proxies[key][name] = [(start, end + 1) for start, end in zip(starts, ends)]
    return proxies


def target_scores(total_frames, segments, variant):
    centers = np.arange(total_frames, dtype=np.float64)
    if not segments:
        return np.zeros(total_frames, dtype=np.float64)
    if variant == "midpoint":
        targets = np.asarray([segment_midpoint(start, end) for start, end in segments])
        distance = np.abs(centers[:, None] - targets[None, :]).min(axis=1)
    elif variant == "center_band_50":
        bands = [segment_center_band(start, end) for start, end in segments]
        distance = np.full(total_frames, np.inf)
        for left, right in bands:
            distance = np.minimum(distance, np.maximum(np.maximum(left - centers, centers - right), 0.0))
    else:
        raise ValueError(f"unknown center variant: {variant}")
    return -distance


def center_schedule(total_frames, budget, segments, variant):
    return STRUCTURED.offline_event_bonus_schedule(
        target_scores(total_frames, segments, variant), budget
    )


def budget_for(total_frames, a0_budget, budget_id):
    if budget_id == "a0_exact":
        return a0_budget
    rate = 0.5 if budget_id == "dense_50pct" else (1.0 / 3.0)
    return min(total_frames, max(2 if total_frames > 1 else 1, int(math.floor(total_frames * rate + 0.5))))


def decode(logits, starts, vocab, blank_id, decoder):
    if decoder == "direct_span1":
        token_ids = np.asarray(logits).argmax(axis=1).tolist()
        collapsed = [token for token, _ in groupby(token_ids) if token != blank_id]
        hypothesis = " ".join(REPLAY.map_phoenix_gloss(vocab[token]) for token in collapsed)
    else:
        span = 15.0 if decoder == "span15" else 7.0
        centers = [float(start) + 7.5 for start in starts]
        token_ids = STRUCTURED.REPLAY.span_weighted_predictions(
            torch.from_numpy(np.asarray(logits)), centers, span=span, min_weight=0.05
        ).tolist()
        collapsed = [token for token, _ in groupby(token_ids) if token != blank_id]
        hypothesis = " ".join(REPLAY.map_phoenix_gloss(vocab[token]) for token in collapsed)
    return REPLAY.clean_phoenix_2014_trans(hypothesis)


def decode_schedules(names, schedules, dense_logits, start_indices, vocab, blank_id, decoder):
    hypotheses = []
    for name in names:
        starts = schedules[name]
        logits = REPLAY.extract_schedule_logits(name, dense_logits, start_indices, starts)
        hypotheses.append(decode(logits, starts, vocab, blank_id, decoder))
    return hypotheses


def center_coverage(segments_by_sample, schedules):
    distances = []
    density_two, density_four = [], []
    inside_band = selected_count = 0
    for segments, selected in zip(segments_by_sample, schedules):
        bands = [segment_center_band(start, end) for start, end in segments]
        for start, end in segments:
            midpoint = segment_midpoint(start, end)
            distances.append(min(abs(midpoint - center) for center in selected))
            density_two.append(sum(abs(midpoint - center) <= 2 for center in selected))
            density_four.append(sum(abs(midpoint - center) <= 4 for center in selected))
        for center in selected:
            selected_count += 1
            inside_band += int(any(left <= center <= right for left, right in bands))
    values = np.asarray(distances, dtype=np.float64)
    return {
        "segment_midpoint_count": len(values),
        "midpoint_hit_within_0_5": float(np.mean(values <= 0.5)) if len(values) else None,
        "midpoint_hit_within_1": float(np.mean(values <= 1.0)) if len(values) else None,
        "midpoint_hit_within_2": float(np.mean(values <= 2.0)) if len(values) else None,
        "midpoint_hit_within_4": float(np.mean(values <= 4.0)) if len(values) else None,
        "midpoint_nearest_distance": {
            "mean": float(values.mean()) if len(values) else None,
            "median": float(np.median(values)) if len(values) else None,
            "p90": float(np.quantile(values, 0.9)) if len(values) else None,
            "max": float(values.max()) if len(values) else None,
        },
        "mean_selected_centers_within_2_per_midpoint": float(np.mean(density_two)) if density_two else None,
        "mean_selected_centers_within_4_per_midpoint": float(np.mean(density_four)) if density_four else None,
        "selected_center_inside_any_central50_band_fraction": float(inside_band / selected_count) if selected_count else None,
    }


def paired_bootstrap(candidate_counts, baseline_counts, seed=260913, repetitions=10000, comparisons=9):
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
        "unit": "WER percentage points (center - same-cell uniform)",
        "sentence_bootstrap_seed": seed,
        "repetitions": repetitions,
        "ci95_unadjusted": [float(value) for value in np.quantile(values, [0.025, 0.975])],
        "ci_bonferroni_9_primary_cells": [
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
    references = [REPLAY.clean_phoenix_2014_trans(b0_results[name]["gls_ref"]) for name in names]
    blank_id = vocab.index("<blank>")
    a0_starts = REPLAY.starts_from_results(a0_results, names)
    proxies = load_segment_proxies(meta_path, center_label_path, names)
    schedules = {budget: {"uniform": {}} for budget in BUDGETS}
    budget_totals = {budget: 0 for budget in BUDGETS}
    for budget in BUDGETS:
        for proxy in PROXIES:
            for variant in CENTER_VARIANTS:
                schedules[budget][f"{proxy}__{variant}"] = {}
        for name in names:
            total_frames = len(start_indices[name])
            count = budget_for(total_frames, len(a0_starts[name]), budget)
            budget_totals[budget] += count
            schedules[budget]["uniform"][name] = STRUCTURED.uniform_positions(total_frames, count)
            for proxy in PROXIES:
                for variant in CENTER_VARIANTS:
                    values = center_schedule(total_frames, count, proxies[proxy][name], variant)
                    if len(values) != count or len(set(values)) != count:
                        raise AssertionError("center schedule violated exact per-sample budget")
                    if values[0] != 0 or values[-1] != total_frames - 1:
                        raise AssertionError("center schedule violated endpoints")
                    schedules[budget][f"{proxy}__{variant}"][name] = values

    uniform_cells = {}
    strategy_cells = {}
    rows = []
    for budget in BUDGETS:
        uniform_cells[budget] = {}
        for decoder in DECODERS:
            uniform_hyp = decode_schedules(
                names, schedules[budget]["uniform"], b0_logits, start_indices, vocab, blank_id, decoder
            )
            uniform_metric = REPLAY.compact_metrics(references, uniform_hyp)
            uniform_count = [RANDOM.sentence_counts(r, h) for r, h in zip(references, uniform_hyp)]
            uniform_cells[budget][decoder] = {
                "metrics": uniform_metric,
                "hypotheses": uniform_hyp,
                "counts": uniform_count,
            }
            for proxy in PROXIES:
                for variant in CENTER_VARIANTS:
                    key = f"{proxy}__{variant}__{budget}__{decoder}"
                    current_schedules = schedules[budget][f"{proxy}__{variant}"]
                    hypotheses = decode_schedules(
                        names, current_schedules, b0_logits, start_indices, vocab, blank_id, decoder
                    )
                    metrics = REPLAY.compact_metrics(references, hypotheses)
                    counts = [RANDOM.sentence_counts(r, h) for r, h in zip(references, hypotheses)]
                    cell = {
                        "identity": "label/alignment-derived offline sign-center proxy; not ground truth or deployable",
                        "metrics": metrics,
                        "delta_vs_same_cell_uniform": RANDOM.metric_delta(metrics, uniform_metric),
                        "hypotheses_changed_vs_uniform": int(sum(
                            left != right for left, right in zip(hypotheses, uniform_hyp)
                        )),
                        "center_coverage": center_coverage(
                            [proxies[proxy][name] for name in names],
                            [current_schedules[name] for name in names],
                        ),
                    }
                    if proxy == "center_label" and variant == "midpoint":
                        cell["paired_bootstrap_primary"] = paired_bootstrap(counts, uniform_count)
                    strategy_cells[key] = cell
                    for index, name in enumerate(names):
                        rows.append({
                            "strategy": key,
                            "name": name,
                            "clips": len(current_schedules[name]),
                            "reference": references[index],
                            "hypothesis": hypotheses[index],
                            **counts[index],
                        })

    compact_uniform = {
        budget: {decoder: {"metrics": value["metrics"]} for decoder, value in cells.items()}
        for budget, cells in uniform_cells.items()
    }
    primary_keys = [f"center_label__midpoint__{budget}__{decoder}" for budget in BUDGETS for decoder in DECODERS]
    best_primary = min(primary_keys, key=lambda key: strategy_cells[key]["metrics"]["wer"] - uniform_cells[key.split("__")[-2]][key.split("__")[-1]]["metrics"]["wer"])
    best_all = min(strategy_cells, key=lambda key: strategy_cells[key]["delta_vs_same_cell_uniform"]["wer_pp"])
    _, _, best_budget, best_decoder = best_primary.split("__")
    best_delta = strategy_cells[best_primary]["delta_vs_same_cell_uniform"]["wer_pp"]
    best_gain = -best_delta
    adjusted_upper = strategy_cells[best_primary]["paired_bootstrap_primary"]["ci_bonferroni_9_primary_cells"][1]
    pami_same_direction = sum(
        strategy_cells[f"{proxy}__midpoint__{best_budget}__{best_decoder}"]["delta_vs_same_cell_uniform"]["wer_pp"] < 0
        for proxy in ("pami0", "pami1", "pami2")
    )
    strong_qualifying = []
    for key in primary_keys:
        _, _, budget_id, decoder_id = key.split("__")
        cell = strategy_cells[key]
        gain = -cell["delta_vs_same_cell_uniform"]["wer_pp"]
        upper = cell["paired_bootstrap_primary"]["ci_bonferroni_9_primary_cells"][1]
        agreeing = sum(
            strategy_cells[f"{proxy}__midpoint__{budget_id}__{decoder_id}"]["delta_vs_same_cell_uniform"]["wer_pp"] < 0
            for proxy in ("pami0", "pami1", "pami2")
        )
        if gain >= 0.5 and upper < 0 and agreeing >= 1:
            strong_qualifying.append({
                "cell": key,
                "gain_wer_pp": gain,
                "bonferroni_upper": upper,
                "agreeing_pami_proxies": agreeing,
            })
    if best_gain >= 0.5 and adjusted_upper < 0 and pami_same_direction >= 1:
        outcome = "strong_go"
    elif best_gain >= 0.2:
        outcome = "weak_go"
    else:
        outcome = "no_go"

    # External frozen references are intentionally restricted to the comparable
    # A0-budget/span15 main cell.
    b2_hyp = RANDOM.decode_schedules(
        names, REPLAY.starts_from_results(b2_results, names), b0_logits, start_indices, vocab, blank_id
    )
    a0_physical = {}
    for name in names:
        a0_physical[name], _, _ = REPLAY.align_starts_to_dense_inputs(
            list(range(len(start_indices[name]))), a0_starts[name]
        )
    a0_hyp = RANDOM.decode_schedules(names, a0_physical, b0_logits, start_indices, vocab, blank_id)
    summary = {
        "sample_count": len(names),
        "full_dev": max_samples is None,
        "identity_warning": "All centers are label/alignment-derived offline proxies, not verified manual ground truth or deployable schedulers.",
        "main_proxy": "center-label first-record-per-bag nonblank segment midpoint (start+end-1)/2",
        "secondary_variant": "closed central 50% interval over included frame-center endpoints; predeclared before results",
        "coordinate": "fixed B0 padded-array coordinate, constant left pad 7",
        "schedule": "half-ceiling uniform skeleton plus center-distance bonus; exact per-sample budget; unique 0/T-1 endpoints",
        "budget_definitions": {
            "a0_exact": "each sample exactly matches its frozen A0 clip count",
            "dense_50pct": "round-half-up 0.5*T, at least 2 for T>1",
            "dense_33pct": "round-half-up T/3, at least 2 for T>1",
        },
        "budget_total_clips": budget_totals,
        "decoder_definitions": {
            "span15": "triangular probability voting over selected centers within 15 frames; min weight .05",
            "span7": "same rule with 7-frame span; min weight .05",
            "direct_span1": "per-selected-window argmax, collapse consecutive repeated ids, then remove blank; no local voting",
        },
        "proxy_provenance": {
            "center_label": "same audited first-record-per-bag proxy; stored start/end fallback because base fields are absent; upstream provenance unresolved",
            "pami0_pami1_pami2": "midpoints of maximal constant nonzero integer runs with provisional 0=blank; integers may be states rather than gloss labels",
        },
        "uniform_cells": compact_uniform,
        "strategy_cells": strategy_cells,
        "external_frozen_reference_only_for_a0_exact_span15": {
            "B2": REPLAY.compact_metrics(references, b2_hyp),
            "A0": REPLAY.compact_metrics(references, a0_hyp),
        },
        "decision": {
            "strong_go_rule": "primary center-label midpoint gains >=.5 pp in a cell, Bonferroni-9 CI excludes zero, and >=1 PAMI midpoint agrees in direction",
            "weak_go_rule": "best primary gain >=.2 pp but strong rule not met",
            "posthoc_best_primary": best_primary,
            "posthoc_best_all": best_all,
            "posthoc_selection_criterion": "most negative WER delta versus its own same-budget/same-decoder uniform cell",
            "best_primary_gain_wer_pp": best_gain,
            "pami_proxies_same_direction_at_best_primary_cell": pami_same_direction,
            "strong_qualifying_primary_cells": strong_qualifying,
            "outcome": outcome,
        },
        "limitations": [
            "Every proxy consumes dev label/alignment information and is offline.",
            "Center-label provenance and PAMI integer semantics remain unresolved.",
            "The matrix changes only schedule, budget, and decoder over fixed dense logits; it does not test a learned center detector.",
            "Centered 16-frame inputs and span15/span7 include lookahead; direct span1 removes voting but not centered-input lookahead.",
        ],
    }
    if max_samples is None:
        if summary["external_frozen_reference_only_for_a0_exact_span15"]["B2"]["error"] != 845:
            raise AssertionError("B2 frozen reproduction failed")
        if summary["external_frozen_reference_only_for_a0_exact_span15"]["A0"]["error"] != 840:
            raise AssertionError("A0 frozen reproduction failed")
    input_paths.extend([
        meta_path,
        center_label_path,
        Path(__file__).resolve(),
        BOUNDARY_PATH,
        STRUCTURED.STRUCTURED_PATH if hasattr(STRUCTURED, "STRUCTURED_PATH") else BOUNDARY.STRUCTURED_PATH,
        STRUCTURED.RANDOM_PATH,
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
        "policy": "repaired Phoenix dev only; CPU dense-logit sign-center proxy matrix",
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
        "center_variants": list(CENTER_VARIANTS),
        "budgets": list(BUDGETS),
        "decoders": list(DECODERS),
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
