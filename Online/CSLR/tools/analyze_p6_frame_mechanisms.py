#!/usr/bin/env python3
"""Frame-level mechanism audit for the frozen P6 online-quantile result.

This is a read-only diagnostic over the existing stride-1 logits and the
alignment-derived sign segments.  It never opens test data or runs the visual
recognizer.  For every reference gloss it asks whether the correct class is
visible in any dense sign-interior window and whether P6 retained such a
window.  For every insertion it localizes the strongest selected evidence to
a sign interior, an inter-sign gap, or outer background.

The segment timings are label/alignment-derived proxies, not independently
verified manual frame boundaries.  All causal labels in the output therefore
remain evidence categories rather than mutually exclusive ground truth causes.
"""

import argparse
import gzip
import hashlib
import json
import os
import pickle
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "results/phoenix-2014t_ISLR"
PREVIOUS = BASE / "p6_online_quantile_wer_error_analysis_v1"
OUTPUT = BASE / "p6_frame_mechanism_analysis_v1"
TRAIN_BAGS = ROOT.parents[1] / "data/phoenix_2014t/phoenix_iso_center_label_bag2items.train"
DEV_SEGMENTS = ROOT.parents[1] / "data/phoenix_2014t/phoenix_iso_center_label.dev"
WINDOW_FRAMES = 16
BOOTSTRAP_SEED = 260930
CHECKPOINT = BASE / "ckpts/best.ckpt"
TRAIN_CONFIG = ROOT / "configs/phoenix-2014t_ISLR.yaml"
SIGN_AUGMENT = ROOT / "sign_augment.py"
VIDEO_LOADER = ROOT / "dataset/VideoLoader.py"


def load_module(path, name):
    import importlib.util

    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


WER = load_module(ROOT / "tools/analyze_p6_online_quantile_wer_errors.py", "p6_mechanism_wer")
SIGN = load_module(ROOT / "tools/analyze_phoenix_sign_center_oracles.py", "p6_mechanism_sign")
P6 = WER.P6


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path, value):
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def load_previous_rows():
    path = PREVIOUS / "per_sample_predictions.jsonl.gz"
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        return {row["sample"]: row for row in map(json.loads, handle)}


def load_segments():
    def collapse_repeats(segments, tokens):
        collapsed_segments, collapsed_tokens = [], []
        for (start, end), token in zip(segments, tokens):
            token = token.upper()
            if collapsed_tokens and token == collapsed_tokens[-1]:
                old_start, old_end = collapsed_segments[-1]
                collapsed_segments[-1] = (min(old_start, start), max(old_end, end))
            else:
                collapsed_segments.append((int(start), int(end)))
                collapsed_tokens.append(token)
        return collapsed_segments, collapsed_tokens

    metadata = WER.metadata()
    dev_names = list(metadata["dev"])
    dev_proxy = SIGN.load_segment_proxies(WER.META["dev"], DEV_SEGMENTS, dev_names)["center_label"]

    with TRAIN_BAGS.open("rb") as handle:
        bags = pickle.load(handle)
    train = defaultdict(list)
    for bag, records in bags.items():
        base = [item for item in records if int(item.get("aug", -1)) == 0]
        if len(base) != 1:
            raise RuntimeError(f"train bag {bag} does not have exactly one base record")
        record = base[0]
        if record["label"] != "<blank>":
            train[record["video_file"]].append((int(bag), record))
    packed = {"train_fit": {}, "dev": {}}
    labels = {"train_fit": {}, "dev": {}}
    for name, records in train.items():
        ordered = [record for _, record in sorted(records)]
        raw_segments = [
            (int(record.get("base_start", record["start"])), int(record.get("base_end", record["end"])))
            for record in ordered
        ]
        raw_labels = [record["label"].upper() for record in ordered]
        packed["train_fit"][name], labels["train_fit"][name] = collapse_repeats(raw_segments, raw_labels)

    with DEV_SEGMENTS.open("rb") as handle:
        records = pickle.load(handle)
    first = SIGN.BOUNDARY.first_record_per_bag(records)
    by_video = defaultdict(list)
    for bag, record in first.items():
        if record["label"] != "<blank>":
            by_video[record["video_file"]].append((int(bag), record))
    for name in dev_names:
        ordered = [record for _, record in sorted(by_video[name])]
        raw_segments = [tuple(value) for value in dev_proxy[name]]
        raw_labels = [record["label"].upper() for record in ordered]
        packed["dev"][name], labels["dev"][name] = collapse_repeats(raw_segments, raw_labels)
    return metadata, packed, labels


def reference_ops(alignment):
    result = []
    for op in alignment["align_lst"]:
        if op in ("C", "D", "S"):
            result.append(op)
    return result


def window_target_fraction(center, start, end):
    # Dense B0 uses centered 16-frame input with seven frames of left padding.
    left, right = center - 7, center + 9
    return max(0, min(right, end) - max(left, start)) / float(WINDOW_FRAMES)


def interval_positions(start, end, length):
    start, end = max(0, int(start)), min(int(length), int(end))
    inside = np.arange(start, end, dtype=np.int64)
    last = end - 1
    midpoint = (start + last) // 2
    span = last - start
    band_left = int(np.ceil(start + 0.25 * span))
    band_right = int(np.floor(start + 0.75 * span))
    band = np.arange(band_left, band_right + 1, dtype=np.int64)
    return inside, midpoint, band


def evidence(probabilities, positions, label_id):
    positions = np.asarray(positions, dtype=np.int64)
    if len(positions) == 0:
        return {
            "positions": 0,
            "any_top1_correct": False,
            "any_top5_correct": False,
            "max_correct_probability": None,
            "best_correct_rank": None,
            "best_position": None,
            "best_top1": None,
        }
    subset = probabilities[positions]
    top1 = subset.argmax(axis=1)
    best_local = int(np.argmax(subset[:, label_id]))
    best_position = int(positions[best_local])
    row = subset[best_local]
    rank = int(1 + np.sum(row > row[label_id]))
    return {
        "positions": int(len(positions)),
        "any_top1_correct": bool(np.any(top1 == label_id)),
        "any_top5_correct": bool(rank <= 5),
        "max_correct_probability": float(row[label_id]),
        "best_correct_rank": rank,
        "best_position": best_position,
        "best_top1": int(row.argmax()),
    }


def location(position, segments, labels, token=None):
    hits = [index for index, (start, end) in enumerate(segments) if start <= position < end]
    if hits:
        hit_labels = [labels[index] for index in hits]
        if token is not None and token in hit_labels:
            category = "inside_same_gloss"
        else:
            category = "inside_other_gloss"
        return category, hits, hit_labels
    if segments and segments[0][0] <= position < segments[-1][1]:
        left = max((index for index, (_, end) in enumerate(segments) if end <= position), default=None)
        right = min((index for index, (start, _) in enumerate(segments) if start > position), default=None)
        neighbors = []
        if left is not None:
            neighbors.append(labels[left])
        if right is not None:
            neighbors.append(labels[right])
        return "between_glosses", [], neighbors
    return "outer_background", [], []


def cause_category(op, dense, selected):
    if op not in ("D", "S"):
        return "correct"
    if not dense["any_top5_correct"]:
        return "recognizer_no_dense_top5"
    if not dense["any_top1_correct"]:
        return "recognizer_ambiguous_top5"
    if not selected["any_top1_correct"]:
        return "selection_missed_dense_top1"
    return "decoder_missed_selected_top1"


def analyze_sample(split, name, probabilities, schedule, previous, segments, segment_labels, vocab):
    reference = previous["reference"].split()
    alignment = previous["policies"]["quantile"]["alignment"]
    operations = reference_ops(alignment)
    if not (len(reference) == len(segments) == len(segment_labels) == len(operations)):
        return None, {
            "split": split,
            "sample": name,
            "reason": "reference_segment_length_mismatch",
            "reference": reference,
            "segment_labels": segment_labels,
        }
    if [token.upper() for token in reference] != [token.upper() for token in segment_labels]:
        return None, {
            "split": split,
            "sample": name,
            "reason": "reference_segment_label_mismatch",
            "reference": reference,
            "segment_labels": segment_labels,
        }
    vocab_index = {token.upper(): index for index, token in enumerate(vocab)}
    selected = np.asarray(schedule, dtype=np.int64)
    if len(probabilities) != previous["num_frames"]:
        raise RuntimeError(f"dense coordinate mismatch: {name}")

    segment_rows = []
    for index, (token, op, (start, end)) in enumerate(zip(reference, operations, segments)):
        label_id = vocab_index[token.upper()]
        inside, midpoint, band = interval_positions(start, end, len(probabilities))
        selected_inside = selected[(selected >= start) & (selected < end)]
        dense_ev = evidence(probabilities, inside, label_id)
        selected_ev = evidence(probabilities, selected_inside, label_id)
        midpoint_ev = evidence(probabilities, [midpoint], label_id)
        start_ev = evidence(probabilities, [start], label_id)
        end_ev = evidence(probabilities, [end - 1], label_id)
        band_ev = evidence(probabilities, band, label_id)
        row = {
            "split": split,
            "sample": name,
            "source": previous["source"],
            "ref_position": index,
            "gloss": token,
            "operation": op,
            "segment_start": int(start),
            "segment_end": int(end),
            "duration": int(end - start),
            "midpoint": midpoint,
            "midpoint_target_fraction": window_target_fraction(midpoint, start, end),
            "selected_inside_count": int(len(selected_inside)),
            "dense": dense_ev,
            "selected": selected_ev,
            "midpoint_evidence": midpoint_ev,
            "central50_evidence": band_ev,
            "start_evidence": start_ev,
            "end_evidence": end_ev,
        }
        row["mechanism"] = cause_category(op, dense_ev, selected_ev)
        segment_rows.append(row)

    insertion_rows = []
    for event_index, event in enumerate(previous["policies"]["quantile"]["events"]):
        if event["op"] != "I":
            continue
        token = event["hyp"]
        label_id = vocab_index[token.upper()]
        selected_best = int(selected[np.argmax(probabilities[selected, label_id])])
        dense_best = int(np.argmax(probabilities[:, label_id]))
        selected_category, selected_hits, selected_neighbors = location(
            selected_best, segments, segment_labels, token.upper()
        )
        dense_category, dense_hits, dense_neighbors = location(
            dense_best, segments, segment_labels, token.upper()
        )
        insertion_rows.append({
            "split": split,
            "sample": name,
            "source": previous["source"],
            "event_index": event_index,
            "token": token,
            "insertion_boundary": event["insertion_boundary"],
            "selected_peak_position": selected_best,
            "selected_peak_probability": float(probabilities[selected_best, label_id]),
            "selected_peak_category": selected_category,
            "selected_peak_segment_indices": selected_hits,
            "selected_peak_neighbor_labels": selected_neighbors,
            "dense_peak_position": dense_best,
            "dense_peak_probability": float(probabilities[dense_best, label_id]),
            "dense_peak_category": dense_category,
            "dense_peak_segment_indices": dense_hits,
            "dense_peak_neighbor_labels": dense_neighbors,
        })

    location_counts = Counter()
    for position in selected:
        category, _, _ = location(int(position), segments, segment_labels)
        location_counts[category] += 1
    sample = {
        "split": split,
        "sample": name,
        "source": previous["source"],
        "reference": previous["reference"],
        "hypothesis": previous["policies"]["quantile"]["hypothesis"],
        "counts": previous["policies"]["quantile"]["counts"],
        "num_frames": previous["num_frames"],
        "selected_windows": int(len(selected)),
        "selected_location_counts": dict(location_counts),
    }
    return {"sample": sample, "segments": segment_rows, "insertions": insertion_rows}, None


def duration_bin(value):
    if value <= 4:
        return "1-4"
    if value <= 6:
        return "5-6"
    if value <= 9:
        return "7-9"
    return "10+"


def occupancy_bin(value):
    if value <= 0.25:
        return "<=0.25"
    if value <= 0.50:
        return "0.25-0.50"
    if value <= 0.75:
        return "0.50-0.75"
    return ">0.75"


def rate_table(rows, key):
    grouped = defaultdict(list)
    for row in rows:
        grouped[str(row[key])].append(row)
    result = {}
    for group, values in sorted(grouped.items()):
        result[group] = summarize_segments(values)
    return result


def summarize_segments(rows):
    operations = Counter(row["operation"] for row in rows)
    mechanisms = Counter(row["mechanism"] for row in rows if row["operation"] in ("D", "S"))
    count = len(rows)
    errors = operations["D"] + operations["S"]
    return {
        "segments": count,
        "correct_alignment": operations["C"],
        "deletions": operations["D"],
        "substitutions": operations["S"],
        "reference_error_rate": errors / count if count else None,
        "midpoint_top1_accuracy": sum(row["midpoint_evidence"]["any_top1_correct"] for row in rows) / count if count else None,
        "start_top1_accuracy": sum(row["start_evidence"]["any_top1_correct"] for row in rows) / count if count else None,
        "end_top1_accuracy": sum(row["end_evidence"]["any_top1_correct"] for row in rows) / count if count else None,
        "dense_any_interior_top1_recall": sum(row["dense"]["any_top1_correct"] for row in rows) / count if count else None,
        "dense_any_interior_top5_recall": sum(row["dense"]["any_top5_correct"] for row in rows) / count if count else None,
        "selected_any_interior_top1_recall": sum(row["selected"]["any_top1_correct"] for row in rows) / count if count else None,
        "selected_center_coverage": sum(row["selected_inside_count"] > 0 for row in rows) / count if count else None,
        "error_mechanisms": dict(mechanisms),
        "error_mechanism_fraction": {
            key: value / errors for key, value in sorted(mechanisms.items())
        } if errors else {},
    }


def cluster_bootstrap(segment_rows, insertion_rows, repetitions=10000):
    by_source_segments = defaultdict(list)
    by_source_insertions = defaultdict(list)
    for row in segment_rows:
        by_source_segments[row["source"]].append(row)
    for row in insertion_rows:
        by_source_insertions[row["source"]].append(row)
    sources = sorted(by_source_segments)
    metrics = (
        "midpoint_top1_accuracy", "dense_any_interior_top1_recall",
        "selected_any_interior_top1_recall", "error_fraction__recognizer_no_dense_top5",
        "error_fraction__recognizer_no_dense_top1",
        "error_fraction__recognizer_ambiguous_top5", "error_fraction__selection_missed_dense_top1",
        "error_fraction__decoder_missed_selected_top1", "insertion_fraction__between_glosses",
    )
    numerators = {key: [] for key in metrics}
    denominators = {key: [] for key in metrics}
    for source in sources:
        seg = by_source_segments[source]
        errors = [row for row in seg if row["operation"] in ("D", "S")]
        ins = by_source_insertions.get(source, [])
        values = {
            "midpoint_top1_accuracy": sum(row["midpoint_evidence"]["any_top1_correct"] for row in seg),
            "dense_any_interior_top1_recall": sum(row["dense"]["any_top1_correct"] for row in seg),
            "selected_any_interior_top1_recall": sum(row["selected"]["any_top1_correct"] for row in seg),
            "error_fraction__recognizer_no_dense_top5": sum(row["mechanism"] == "recognizer_no_dense_top5" for row in errors),
            "error_fraction__recognizer_no_dense_top1": sum(row["mechanism"].startswith("recognizer_") for row in errors),
            "error_fraction__recognizer_ambiguous_top5": sum(row["mechanism"] == "recognizer_ambiguous_top5" for row in errors),
            "error_fraction__selection_missed_dense_top1": sum(row["mechanism"] == "selection_missed_dense_top1" for row in errors),
            "error_fraction__decoder_missed_selected_top1": sum(row["mechanism"] == "decoder_missed_selected_top1" for row in errors),
            "insertion_fraction__between_glosses": sum(row["selected_peak_category"] == "between_glosses" for row in ins),
        }
        for key in metrics:
            numerators[key].append(values[key])
            denominators[key].append(len(ins) if key.startswith("insertion_") else len(errors) if key.startswith("error_") else len(seg))
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    sampled = rng.integers(0, len(sources), size=(repetitions, len(sources)))
    values = {}
    for key in metrics:
        numerator = np.asarray(numerators[key], dtype=np.float64)[sampled].sum(axis=1)
        denominator = np.asarray(denominators[key], dtype=np.float64)[sampled].sum(axis=1)
        values[key] = np.divide(numerator, denominator, out=np.zeros_like(numerator), where=denominator > 0)
    return {
        key: [float(np.quantile(value, 0.025)), float(np.quantile(value, 0.975))]
        for key, value in sorted(values.items())
    }


def summarize(split, samples, segment_rows, insertion_rows, mismatches):
    selected_locations = Counter()
    selected_total = 0
    for row in samples:
        for key, value in row["selected_location_counts"].items():
            selected_locations["inside_gloss" if key == "inside_other_gloss" else key] += value
        selected_total += row["selected_windows"]
    insertion_categories = Counter(row["selected_peak_category"] for row in insertion_rows)
    insertion_dense_categories = Counter(row["dense_peak_category"] for row in insertion_rows)
    result = {
        "split": split,
        "samples": len(samples),
        "sources": len(set(row["source"] for row in samples)),
        "mismatches": mismatches,
        "overall": summarize_segments(segment_rows),
        "selected_window_locations": {
            "count": dict(selected_locations),
            "fraction": {key: value / selected_total for key, value in sorted(selected_locations.items())},
            "total": selected_total,
        },
        "insertions": {
            "count": len(insertion_rows),
            "selected_peak_category_count": dict(insertion_categories),
            "selected_peak_category_fraction": {
                key: value / len(insertion_rows) for key, value in sorted(insertion_categories.items())
            } if insertion_rows else {},
            "dense_peak_category_count": dict(insertion_dense_categories),
        },
        "by_duration": rate_table(
            [dict(row, duration_bin=duration_bin(row["duration"])) for row in segment_rows], "duration_bin"
        ),
        "by_midpoint_target_fraction": rate_table(
            [dict(row, occupancy_bin=occupancy_bin(row["midpoint_target_fraction"])) for row in segment_rows], "occupancy_bin"
        ),
        "sequence_context": sequence_context(samples, segment_rows),
        "bootstrap_source_cluster_ci95": cluster_bootstrap(segment_rows, insertion_rows),
    }
    return result


def short_gloss_control(train_rows, dev_rows):
    counts = {split: defaultdict(lambda: [0, 0]) for split in ("train_fit", "dev")}
    for split, rows in (("train_fit", train_rows), ("dev", dev_rows)):
        for row in rows:
            if row["duration"] <= 6:
                counts[split][row["gloss"]][0] += 1
                counts[split][row["gloss"]][1] += int(row["operation"] in ("D", "S"))
    common = sorted(
        gloss for gloss in counts["dev"]
        if counts["dev"][gloss][0] >= 3 and counts["train_fit"][gloss][0] >= 10
    )
    dev_reference = sum(counts["dev"][gloss][0] for gloss in common)
    dev_errors = sum(counts["dev"][gloss][1] for gloss in common)
    train_reference = sum(counts["train_fit"][gloss][0] for gloss in common)
    train_errors = sum(counts["train_fit"][gloss][1] for gloss in common)
    expected = sum(
        counts["dev"][gloss][0]
        * counts["train_fit"][gloss][1]
        / counts["train_fit"][gloss][0]
        for gloss in common
    )
    return {
        "definition": "duration<=6; same gloss must have >=3 dev and >=10 train-fit reference instances",
        "common_glosses": len(common),
        "dev_reference": dev_reference,
        "dev_deletion_or_substitution": dev_errors,
        "dev_error_rate": dev_errors / dev_reference,
        "train_fit_reference": train_reference,
        "train_fit_deletion_or_substitution": train_errors,
        "train_fit_error_rate": train_errors / train_reference,
        "expected_dev_errors_at_train_within_gloss_rates": expected,
        "observed_to_expected_ratio": dev_errors / expected,
        "interpretation": "controls short-segment duration and gloss identity descriptively; recognizer train videos remain in-sample",
    }


def sequence_context(samples, segment_rows):
    by_sample = defaultdict(list)
    for row in segment_rows:
        by_sample[row["sample"]].append(row)
    for rows in by_sample.values():
        rows.sort(key=lambda row: row["ref_position"])

    ngrams = {2: defaultdict(Counter), 3: defaultdict(Counter)}
    multi_runs = []
    for rows in by_sample.values():
        for width in (2, 3):
            for start in range(len(rows) - width + 1):
                window = rows[start : start + width]
                short_count = sum(row["duration"] <= 6 for row in window)
                key = f"{short_count}_short_of_{width}"
                ngrams[width][key]["total"] += 1
                ngrams[width][key]["any_reference_error"] += int(
                    any(row["operation"] in ("D", "S") for row in window)
                )
                ngrams[width][key]["all_reference_error"] += int(
                    all(row["operation"] in ("D", "S") for row in window)
                )
        current = []
        for row in rows + [None]:
            if row is not None and row["operation"] in ("D", "S"):
                current.append(row)
            else:
                if len(current) >= 2:
                    multi_runs.append(current)
                current = []

    ngram_summary = {}
    for width in (2, 3):
        ngram_summary[str(width)] = {}
        for key, counter in sorted(ngrams[width].items()):
            total = counter["total"]
            ngram_summary[str(width)][key] = {
                **dict(counter),
                "any_reference_error_rate": counter["any_reference_error"] / total,
                "all_reference_error_rate": counter["all_reference_error"] / total,
            }

    sentence_bins = defaultdict(Counter)
    sample_lookup = {row["sample"]: row for row in samples}
    for name, rows in by_sample.items():
        fraction = sum(row["duration"] <= 6 for row in rows) / len(rows)
        if fraction == 0:
            key = "0"
        elif fraction <= 1 / 3:
            key = "(0,1/3]"
        elif fraction <= 2 / 3:
            key = "(1/3,2/3]"
        else:
            key = "(2/3,1]"
        counts = sample_lookup[name]["counts"]
        sentence_bins[key]["sentences"] += 1
        sentence_bins[key]["reference"] += int(counts["ref_len"])
        sentence_bins[key]["errors"] += int(counts["error"])
        sentence_bins[key]["exact"] += int(counts["error"] == 0)
    sentence_summary = {}
    for key, counter in sentence_bins.items():
        sentence_summary[key] = {
            **dict(counter),
            "wer": counter["errors"] / counter["reference"],
            "exact_sentence_rate": counter["exact"] / counter["sentences"],
        }

    run_tokens = sum(len(run) for run in multi_runs)
    return {
        "adjacent_reference_ngrams_by_short_count": ngram_summary,
        "contiguous_reference_error_runs_length_at_least_2": {
            "runs": len(multi_runs),
            "error_tokens": run_tokens,
            "all_short_runs": sum(all(row["duration"] <= 6 for row in run) for run in multi_runs),
            "any_short_runs": sum(any(row["duration"] <= 6 for row in run) for run in multi_runs),
            "all_short_error_tokens": sum(
                len(run) for run in multi_runs if all(row["duration"] <= 6 for row in run)
            ),
        },
        "sentences_by_short_reference_fraction": sentence_summary,
    }


def run(output):
    previous = load_previous_rows()
    metadata, all_segments, all_labels = load_segments()
    train_frozen = WER.frozen_rows(WER.TRAIN_FROZEN)
    dev_frozen = WER.frozen_rows(WER.DEV_FROZEN)
    folds = WER.train_sample_folds()
    models = WER.portable_fold_models()
    vocab = json.loads(P6.CHRON.DEFAULT_VOCAB.read_text(encoding="utf-8"))
    blank = vocab.index("<blank>")
    rng = np.random.default_rng(261130)
    buckets = rng.integers(0, 32, size=len(vocab))
    signs = rng.choice(np.asarray([-1.0, 1.0], np.float32), size=len(vocab))

    output.mkdir(parents=True, exist_ok=False)
    rows_path = output / "per_event.jsonl.gz"
    collected = {split: {"samples": [], "segments": [], "insertions": [], "mismatches": []} for split in ("train_fit", "dev")}

    def consume(split, name, result, logits, model, frozen):
        probabilities = P6.BUILDER.ORACLE.softmax_rows(np.asarray(logits))
        features = P6.RANK.dense_frame_features(probabilities, blank, buckets, signs)
        policies, _ = P6.schedules(probabilities, features, model)
        if len(policies["quantile"]) != int(frozen[name]["quantile_windows"]):
            raise RuntimeError(f"frozen schedule mismatch: {name}")
        decoded = WER.decode_one(
            split, name, result, logits, model, frozen, metadata[split][name],
            vocab, blank, buckets, signs,
        )
        if decoded["policies"]["quantile"]["hypothesis"] != previous[name]["policies"]["quantile"]["hypothesis"]:
            raise RuntimeError(f"previous reconstruction mismatch: {name}")
        analyzed, mismatch = analyze_sample(
            split, name, probabilities, policies["quantile"], previous[name],
            all_segments[split].get(name, []), all_labels[split].get(name, []), vocab,
        )
        if mismatch:
            collected[split]["mismatches"].append(mismatch)
            return
        for kind in ("sample", "segments", "insertions"):
            value = analyzed[kind]
            if kind == "sample":
                collected[split]["samples"].append(value)
            else:
                collected[split][kind].extend(value)

    shard_indices, shard_count = P6.CHRON.completed_shard_indices(WER.TRAIN_DENSE)
    for shard in shard_indices:
        results, logits, _ = P6.BUILDER.BUILDER.validate_dense_shard(
            WER.TRAIN_DENSE, shard, shard_count, verify_hashes=False
        )
        for name in results:
            if name in folds:
                consume("train_fit", name, results[name], logits[name], models[folds[name]], train_frozen)
        print(json.dumps({"split": "train_fit", "shard": shard}), flush=True)

    model = WER.DEV.fit_full_k()
    with (WER.DEV_INPUT / "dev_results.pkl").open("rb") as handle:
        results = pickle.load(handle)
    with (WER.DEV_INPUT / "dev_logits.pkl").open("rb") as handle:
        logits = pickle.load(handle)
    for name in sorted(results):
        consume("dev", name, results[name], logits[name], model, dev_frozen)

    with gzip.open(rows_path, "wt", encoding="utf-8", compresslevel=6) as handle:
        for split in ("train_fit", "dev"):
            for kind in ("samples", "segments", "insertions"):
                for row in collected[split][kind]:
                    handle.write(json.dumps({"kind": kind[:-1] if kind.endswith("s") else kind, **row}, ensure_ascii=False, separators=(",", ":")) + "\n")

    summary = {
        "schema_version": 1,
        "identity": "frozen P6 frame-level mechanism audit",
        "scope": "train-fit controller OOF and authorized retrospective dev; test not opened",
        "segment_warning": "alignment-derived proxy segments; not independently verified manual frame boundaries",
        "coordinate": "dense stride-1 logit index equals centered-16 original-frame center under seven-frame left padding",
        "cause_warning": "mechanism categories are diagnostic evidence and may overlap with unobserved causes",
        "splits": {},
    }
    for split in ("train_fit", "dev"):
        data = collected[split]
        summary["splits"][split] = summarize(
            split, data["samples"], data["segments"], data["insertions"], data["mismatches"]
        )
    summary["short_gloss_train_dev_control"] = short_gloss_control(
        collected["train_fit"]["segments"], collected["dev"]["segments"]
    )
    checkpoint = torch.load(CHECKPOINT, map_location="cpu")
    summary["recognizer_training_design"] = {
        "input_frames": WINDOW_FRAMES,
        "bag_size": 6,
        "augmentation": "one base crop plus one centered 16-frame crop for every frame in the proxy segment",
        "short_base_padding": "when a base crop is shorter than 16, training pads to 16 by repeating endpoint frames with random left/right allocation",
        "five_frame_bag": "one padded base plus five continuous-video centered crops; all six are sampled because bag_size is six",
        "continuous_five_frame_crop_target_fraction_maximum": 5 / 16,
        "checkpoint_best_isolated_dev_per_instance_accuracy": float(checkpoint["best_score"]),
        "checkpoint_score_warning": "training checkpoint score covers the flat augmented isolated dev set, including blank and augmented records; it is not base-only nonblank accuracy",
    }
    dev = summary["splits"]["dev"]
    mechanisms = dev["overall"]["error_mechanisms"]
    insertion_categories = dev["insertions"]["selected_peak_category_count"]
    total_dev_errors = (
        dev["overall"]["deletions"] + dev["overall"]["substitutions"] + dev["insertions"]["count"]
    )
    evidence_counts = {
        "no_dense_interior_top1_for_reference_gloss": (
            mechanisms.get("recognizer_no_dense_top5", 0) + mechanisms.get("recognizer_ambiguous_top5", 0)
        ),
        "dense_top1_exists_but_p6_did_not_select_it": mechanisms.get("selection_missed_dense_top1", 0),
        "selected_top1_exists_but_sequence_alignment_still_wrong": mechanisms.get("decoder_missed_selected_top1", 0),
        "insertion_peak_between_gloss_proxies": insertion_categories.get("between_glosses", 0),
        "insertion_peak_inside_other_gloss_proxy": insertion_categories.get("inside_other_gloss", 0),
        "insertion_peak_inside_same_gloss_proxy": insertion_categories.get("inside_same_gloss", 0),
        "insertion_peak_outer_background": insertion_categories.get("outer_background", 0),
    }
    summary["dev_all_wer_error_evidence_partition"] = {
        "total_errors": total_dev_errors,
        "counts": evidence_counts,
        "fractions": {key: value / total_dev_errors for key, value in evidence_counts.items()},
        "warning": "diagnostic evidence partition based on proxy segments and posterior peaks; only the edit types themselves are exact causal labels",
    }
    atomic_json(output / "summary.json", summary)

    inputs = [
        Path(__file__).resolve(), PREVIOUS / "per_sample_predictions.jsonl.gz",
        PREVIOUS / "manifest.json", WER.FOLD_MODELS, WER.META["train_fit"],
        WER.META["dev"], TRAIN_BAGS, DEV_SEGMENTS,
        WER.TRAIN_DENSE / "protocol_manifest.json", WER.DEV_INPUT.parents[2] / "protocol_manifest.json",
        CHECKPOINT, TRAIN_CONFIG, SIGN_AUGMENT, VIDEO_LOADER,
    ]
    manifest = {
        "schema_version": 1,
        "inputs": {str(path.resolve()): sha256(path) for path in inputs},
        "outputs": {
            "per_event.jsonl.gz": sha256(rows_path),
            "summary.json": sha256(output / "summary.json"),
        },
    }
    atomic_json(output / "manifest.json", manifest)
    return summary


def verify(output):
    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    for path, expected in manifest["inputs"].items():
        observed = sha256(path)
        if observed != expected:
            raise RuntimeError(f"input hash mismatch: {path}")
    for name, expected in manifest["outputs"].items():
        observed = sha256(output / name)
        if observed != expected:
            raise RuntimeError(f"output hash mismatch: {name}")
    summary = json.loads((output / "summary.json").read_text(encoding="utf-8"))
    if summary["splits"]["dev"]["samples"] != 519:
        raise RuntimeError("dev coverage mismatch")
    print(json.dumps({"verified": True, "manifest": str((output / "manifest.json").resolve())}))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args()
    if "test" in "/".join(part.lower() for part in args.output.resolve().parts):
        raise ValueError("test paths are forbidden")
    if args.verify:
        verify(args.output)
        return
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite existing output: {args.output}")
    summary = run(args.output)
    print(json.dumps(summary["splits"], indent=2))


if __name__ == "__main__":
    main()
