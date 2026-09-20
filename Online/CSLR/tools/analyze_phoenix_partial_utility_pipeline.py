#!/usr/bin/env python3
"""CPU-only partial-train smoke of utility prediction and schedule replay.

This intentionally diagnoses plumbing only.  It uses non-chronological oracle
states for supervision and known per-sample length for the equal-budget replay;
it is not a deployable unknown-EOS scheduler and cannot freeze hyperparameters.
"""

import argparse
import gzip
import hashlib
import importlib.util
import json
import math
import os
import pickle
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
BUILDER_PATH = ROOT / "tools/build_phoenix_train_counterfactual_utility_dataset.py"
SPEC = importlib.util.spec_from_file_location("train_utility_builder", BUILDER_PATH)
BUILDER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(BUILDER)
BASE = BUILDER.BASE
ORACLE = BASE.ORACLE
REPLAY = BASE.REPLAY

DEFAULT_UTILITY_ROOT = (
    ROOT
    / "results/phoenix-2014t_ISLR/p3_train_counterfactual_utility_partial32_smoke_v1_49faacc3"
)
DEFAULT_OUTPUT_ROOT = (
    ROOT / "results/phoenix-2014t_ISLR/p3_partial32_predictor_pipeline_smoke_v1_49faacc3"
)


def reject_test_path(path):
    BASE.reject_test_path(path)


def sha256_file(path, chunk_size=8 * 1024 * 1024):
    return BUILDER.sha256_file(path, chunk_size=chunk_size)


def atomic_json(path, value):
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def raw_features(predictor_inputs):
    """Only deployment-eligible causal bookkeeping fields are consumed."""
    BASE.assert_predictor_inputs_clean(predictor_inputs)
    return np.asarray(
        [
            math.log1p(int(predictor_inputs["candidate_window_start"])),
            math.log1p(int(predictor_inputs["selected_past_count"])),
            math.log1p(int(predictor_inputs["frames_since_last_selected_past"])),
            float(bool(predictor_inputs["has_selected_past"])),
        ],
        dtype=np.float64,
    )


def state_candidate_inputs(candidate, selected):
    return BASE.candidate_predictor_inputs(int(candidate), sorted(int(value) for value in selected))


def deterministic_keep_nonpositive(state_id, candidate_start, percent=2):
    digest = hashlib.sha256(f"{state_id}:{candidate_start}".encode("utf-8")).digest()
    return int.from_bytes(digest[:4], "big") % 100 < percent


def iter_records(utility_root, shard_indices):
    shard_count = json.loads((utility_root / "dataset_summary.json").read_text())["shard_count"]
    for shard_index in shard_indices:
        shard_path, complete_path = BUILDER.output_paths(utility_root, shard_index, shard_count)
        completion = BUILDER.validate_output_shard(utility_root, shard_index, shard_count)
        if completion["sha256"] != sha256_file(shard_path):
            raise ValueError("utility shard changed after validation")
        with gzip.open(shard_path, "rt", encoding="utf-8") as handle:
            for line in handle:
                yield json.loads(line)


def collect_fit_rows(utility_root, shard_indices, assignments):
    features, targets, utilities = [], [], []
    seen_states = 0
    for state in iter_records(utility_root, shard_indices):
        if assignments[state["sample_id"]] != "fit":
            continue
        seen_states += 1
        for candidate in state["candidates"]:
            utility = int(candidate["label"]["utility"])
            if utility != 0 or deterministic_keep_nonpositive(
                state["state_id"], candidate["candidate_window_start"]
            ):
                features.append(raw_features(candidate["predictor_inputs"]))
                targets.append(float(utility > 0))
                utilities.append(utility)
    matrix = np.asarray(features, dtype=np.float64)
    target = np.asarray(targets, dtype=np.float64)
    if not len(matrix) or target.sum() == 0 or target.sum() == len(target):
        raise ValueError("sampled fit labels lack both classes")
    return matrix, target, np.asarray(utilities, dtype=np.int16), seen_states


def partial_partition_summary(utility_root, shard_indices, assignments, shard_count):
    names = []
    for shard_index in shard_indices:
        _, complete_path = BUILDER.output_paths(utility_root, shard_index, shard_count)
        names.extend(json.loads(complete_path.read_text())["sample_ids"])
    groups = {"fit": set(), "calibration": set()}
    counts = Counter()
    for name in names:
        partition = assignments[name]
        counts[partition] += 1
        groups[partition].add(name.rpartition("-")[0])
    overlap = groups["fit"] & groups["calibration"]
    if overlap:
        raise ValueError("partial source-video groups cross fit/calibration")
    return {
        "samples": dict(sorted(counts.items())),
        "source_videos": {key: len(value) for key, value in groups.items()},
        "source_video_overlap": 0,
    }


def train_logistic(matrix, target, iterations=30, l2=1e-3):
    mean = matrix.mean(axis=0)
    scale = matrix.std(axis=0)
    scale[scale < 1e-6] = 1.0
    design = np.column_stack([np.ones(len(matrix)), (matrix - mean) / scale])
    positive = target.sum()
    class_weight = np.where(target > 0, (len(target) - positive) / positive, 1.0)
    coefficient = np.zeros(design.shape[1], dtype=np.float64)
    history = []
    for iteration in range(iterations):
        score = np.clip(design @ coefficient, -30.0, 30.0)
        probability = 1.0 / (1.0 + np.exp(-score))
        gradient = design.T @ (class_weight * (probability - target))
        curvature = class_weight * probability * (1.0 - probability)
        hessian = design.T @ (design * curvature[:, None])
        regularizer = np.eye(len(coefficient)) * l2 * len(matrix)
        regularizer[0, 0] = 0.0
        step = np.linalg.solve(hessian + regularizer, gradient + regularizer @ coefficient)
        coefficient -= step
        loss = -np.average(
            target * np.log(np.maximum(probability, 1e-12))
            + (1.0 - target) * np.log(np.maximum(1.0 - probability, 1e-12)),
            weights=class_weight,
        )
        history.append(float(loss))
        if np.linalg.norm(step) < 1e-7:
            break
    return {"coefficient": coefficient, "mean": mean, "scale": scale, "loss": history}


def predict(model, matrix):
    values = np.asarray(matrix, dtype=np.float64)
    if values.ndim == 1:
        values = values[None, :]
    design = np.column_stack([np.ones(len(values)), (values - model["mean"]) / model["scale"]])
    score = np.clip(design @ model["coefficient"], -30.0, 30.0)
    return 1.0 / (1.0 + np.exp(-score))


def average_precision(labels, scores):
    labels = np.asarray(labels, dtype=np.int8)
    if labels.sum() == 0:
        return None
    order = np.argsort(-np.asarray(scores), kind="stable")
    ranked = labels[order]
    precision = np.cumsum(ranked) / np.arange(1, len(ranked) + 1)
    return float((precision * ranked).sum() / ranked.sum())


def ranking_metrics(utility_root, shard_indices, assignments, model):
    all_labels, all_scores = [], []
    recall_numerator = recall_denominator = 0
    ndcg_values, regrets = [], []
    states = positive_states = 0
    for state in iter_records(utility_root, shard_indices):
        if assignments[state["sample_id"]] != "calibration":
            continue
        states += 1
        utilities = np.asarray([int(row["label"]["utility"]) for row in state["candidates"]])
        features = np.vstack([raw_features(row["predictor_inputs"]) for row in state["candidates"]])
        scores = predict(model, features)
        labels = utilities > 0
        all_labels.extend(labels.tolist())
        all_scores.extend(scores.tolist())
        k = max(1, int(math.ceil(0.1 * len(scores))))
        chosen = np.argsort(-scores, kind="stable")[:k]
        recall_numerator += int(labels[chosen].sum())
        recall_denominator += int(labels.sum())
        gains = np.maximum(utilities, 0).astype(np.float64)
        if gains.sum() > 0:
            positive_states += 1
            discount = 1.0 / np.log2(np.arange(2, k + 2))
            dcg = float((gains[chosen] * discount).sum())
            ideal = np.sort(gains)[::-1][:k]
            ndcg_values.append(dcg / float((ideal * discount).sum()))
        regrets.append(int(utilities.max() - utilities[int(np.argmax(scores))]))
    return {
        "calibration_states": states,
        "positive_utility_states": positive_states,
        "candidate_records": len(all_labels),
        "positive_candidates": int(sum(all_labels)),
        "pr_auc_average_precision": average_precision(all_labels, all_scores),
        "positive_recall_at_top_10pct": (
            float(recall_numerator / recall_denominator) if recall_denominator else None
        ),
        "mean_ndcg_at_top_10pct_positive_states": (
            float(np.mean(ndcg_values)) if ndcg_values else None
        ),
        "top1_utility_regret": {
            "mean": float(np.mean(regrets)),
            "positive_fraction": float(np.mean(np.asarray(regrets) > 0)),
        },
    }


def learned_schedule(total_windows, model):
    total_budget = ORACLE.dense_half_budget(total_windows)
    skeleton_count = ORACLE.skeleton_size(total_budget, ORACLE.PRIMARY_RATIO)
    selected = ORACLE.uniform_positions(total_windows, skeleton_count)
    while len(selected) < total_budget:
        selected_set = set(selected)
        candidates = [value for value in range(total_windows) if value not in selected_set]
        matrix = np.vstack([raw_features(state_candidate_inputs(value, selected)) for value in candidates])
        scores = predict(model, matrix)
        best = min(
            range(len(candidates)),
            key=lambda index: (-float(scores[index]), candidates[index]),
        )
        selected.append(candidates[best])
        selected.sort()
    return selected


def sentence_row(reference, hypothesis):
    counts = BASE.RANDOM.sentence_counts(reference, hypothesis)
    return {key: int(counts[key]) for key in ("error", "del", "ins", "sub", "ref_len")}


def aggregate_wer(rows):
    totals = Counter()
    for row in rows:
        totals.update(row)
    return {
        "wer": 100.0 * totals["error"] / totals["ref_len"],
        **{key: int(totals[key]) for key in ("error", "del", "ins", "sub", "ref_len")},
    }


def schedule_replay(dense_root, shard_indices, assignments, vocab, model):
    blank_id = vocab.index("<blank>")
    learned_rows, uniform_rows, records = [], [], []
    dense_protocol = json.loads((dense_root / "protocol_manifest.json").read_text())
    shard_count = int(math.ceil(dense_protocol["samples"] / dense_protocol["shard_samples"]))
    for shard_index in shard_indices:
        results, logits, _ = BUILDER.validate_dense_shard(
            dense_root, shard_index, shard_count, verify_hashes=True
        )
        for name in results:
            if assignments[name] != "calibration":
                continue
            probabilities = ORACLE.softmax_rows(logits[name])
            budget = ORACLE.dense_half_budget(len(probabilities))
            uniform = ORACLE.uniform_positions(len(probabilities), budget)
            learned = learned_schedule(len(probabilities), model)
            if len(learned) != len(uniform) or len(set(learned)) != budget:
                raise AssertionError("learned schedule violates exact equal budget")
            reference = REPLAY.clean_phoenix_2014_trans(results[name]["gls_ref"])
            learned_hypothesis = ORACLE.decode_probabilities(
                probabilities, learned, vocab, blank_id
            )
            uniform_hypothesis = ORACLE.decode_probabilities(
                probabilities, uniform, vocab, blank_id
            )
            learned_row = sentence_row(reference, learned_hypothesis)
            uniform_row = sentence_row(reference, uniform_hypothesis)
            learned_rows.append(learned_row)
            uniform_rows.append(uniform_row)
            records.append(
                {
                    "sample_id": name,
                    "dense_windows": len(probabilities),
                    "budget": budget,
                    "learned_selected": learned,
                    "uniform_selected": uniform,
                    "learned_counts": learned_row,
                    "uniform_counts": uniform_row,
                }
            )
    learned = aggregate_wer(learned_rows)
    uniform = aggregate_wer(uniform_rows)
    return records, {
        "samples": len(records),
        "learned": learned,
        "equal_budget_uniform": uniform,
        "delta_learned_minus_uniform": {
            "wer_pp": learned["wer"] - uniform["wer"],
            "error_count": learned["error"] - uniform["error"],
        },
    }


def run(utility_root, dense_root, output_root, vocab_path):
    for path in (utility_root, dense_root, output_root, vocab_path):
        reject_test_path(path)
    if output_root.exists():
        raise FileExistsError(f"refusing to overwrite {output_root}")
    started = time.perf_counter()
    utility_summary = json.loads((utility_root / "dataset_summary.json").read_text())
    shard_indices = utility_summary["selected_shard_indices"]
    split_path = utility_root / "fit_calibration_split.json"
    assignments = json.loads(split_path.read_text())["assignments"]
    partition = partial_partition_summary(
        utility_root, shard_indices, assignments, utility_summary["shard_count"]
    )
    fit_matrix, fit_target, fit_utility, fit_states = collect_fit_rows(
        utility_root, shard_indices, assignments
    )
    model = train_logistic(fit_matrix, fit_target)
    ranking = ranking_metrics(utility_root, shard_indices, assignments, model)
    with vocab_path.open("r", encoding="utf-8") as handle:
        vocab = json.load(handle)
    schedule_records, schedule_metrics = schedule_replay(
        dense_root, shard_indices, assignments, vocab, model
    )
    output_root.mkdir(parents=True)
    schedules_path = output_root / "calibration_schedules.jsonl.gz"
    with gzip.open(schedules_path, "wt", encoding="utf-8", compresslevel=6) as handle:
        for record in schedule_records:
            handle.write(json.dumps(record, separators=(",", ":")) + "\n")
    model_path = output_root / "linear_predictor.npz"
    np.savez(
        model_path,
        coefficient=model["coefficient"],
        mean_fit_only=model["mean"],
        scale_fit_only=model["scale"],
    )
    metrics = {
        "scope": "partial-train pipeline smoke only; not a research result",
        "limitations": [
            "only 32/56 non-random available train dense shards",
            "oracle training states are non-chronological",
            "equal-budget schedule replay uses known per-sample length/EOS",
            "features are bookkeeping-only; no causal pose, phase/hazard, or decoder prefix",
            "no hyperparameter selection, paired bootstrap, wall time, or latency claim",
        ],
        "gpu_used": False,
        "dev_used": False,
        "test_used": False,
        "partial_utility_dataset": {
            "selected_shards": len(shard_indices),
            "samples": utility_summary["statistics"]["samples"],
            "states": utility_summary["statistics"]["states"],
            "candidate_records": utility_summary["statistics"]["candidate_records"],
            "utility_counts": utility_summary["statistics"]["utility_counts"],
            "all_zero_states": utility_summary["statistics"]["all_zero_states"],
        },
        "train_partition": partition,
        "fit": {
            "states": fit_states,
            "sampled_candidates": len(fit_target),
            "sampled_positive_candidates": int(fit_target.sum()),
            "sampled_utility_counts": dict(
                sorted(Counter(int(value) for value in fit_utility).items())
            ),
            "negative_zero_sampling": "all nonzero plus deterministic 2% of zero labels",
            "training_loss": model["loss"],
        },
        "ranking": ranking,
        "schedule_replay": schedule_metrics,
        "runtime_seconds": time.perf_counter() - started,
    }
    config = {
        "feature_names": [
            "log1p(candidate_window_start)",
            "log1p(selected_past_count)",
            "log1p(frames_since_last_selected_past)",
            "has_selected_past",
        ],
        "predictor": "fit-only class-balanced linear logistic regression",
        "ranking_target": "utility > 0",
        "schedule": "50% total known-EOS budget; 50% uniform skeleton plus learned bonus",
        "decoder": "frozen span15",
        "selected_dense_shards": shard_indices,
    }
    atomic_json(output_root / "metrics.json", metrics)
    atomic_json(output_root / "resolved_config.json", config)
    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "pipeline_smoke_not_frozen",
        "inputs": {
            str(path.resolve()): {"bytes": path.stat().st_size, "sha256": sha256_file(path)}
            for path in (
                Path(__file__), BUILDER_PATH, utility_root / "protocol_manifest.json",
                utility_root / "dataset_summary.json", split_path,
                dense_root / "protocol_manifest.json", vocab_path,
            )
        },
        "outputs": {
            path.name: {"bytes": path.stat().st_size, "sha256": sha256_file(path)}
            for path in (
                output_root / "metrics.json", output_root / "resolved_config.json",
                schedules_path, model_path,
            )
        },
    }
    atomic_json(output_root / "manifest.json", manifest)
    return metrics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--utility-root", type=Path, default=DEFAULT_UTILITY_ROOT)
    parser.add_argument("--dense-root", type=Path, default=BUILDER.DEFAULT_DENSE_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--vocab", type=Path, default=BUILDER.DEFAULT_VOCAB)
    args = parser.parse_args()
    metrics = run(
        args.utility_root.resolve(), args.dense_root.resolve(),
        args.output_root.resolve(), args.vocab.resolve(),
    )
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
