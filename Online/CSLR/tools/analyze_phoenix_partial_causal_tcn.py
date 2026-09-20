#!/usr/bin/env python3
"""CPU-only ordered-history TCN smoke on partial Phoenix train replay.

This is an engineering diagnostic, not an online CSLR result.  The utility
states are oracle-generated/non-chronological and schedule replay knows EOS.
The temporal predictor itself is strictly causal at a candidate: it sees only
31 cheap-feature frames ending when that 16-frame ISLR window is available.
"""

import argparse
import gzip
import importlib.util
import json
import math
import os
import random
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from torch import nn


ROOT = Path(__file__).resolve().parents[1]


def import_tool(name):
    path = ROOT / f"tools/{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


RICH = import_tool("analyze_phoenix_partial_rich_predictors")
BASELINE = RICH.BASELINE
BUILDER = RICH.BUILDER
ORACLE = RICH.ORACLE
REPLAY = RICH.REPLAY

DEFAULT_UTILITY_ROOT = RICH.DEFAULT_UTILITY_ROOT
DEFAULT_DENSE_ROOT = RICH.DEFAULT_DENSE_ROOT
DEFAULT_SEQUENCES = RICH.DEFAULT_SEQUENCES
DEFAULT_PREVIOUS = RICH.DEFAULT_OUTPUT_ROOT / "metrics.json"
DEFAULT_OUTPUT_ROOT = (
    ROOT / "results/phoenix-2014t_ISLR/"
    "p3_partial32_causal_tcn_smoke_v1_49faacc3"
)
HISTORY_FRAMES = 31
SEED = 260921
EPOCHS = 4
BATCH_SIZE = 1024
CHANNELS = 12

# Pre-registered stable, cheap motion/geometry subset.  This avoids feeding the
# 211-dimensional raw keypoint vector into a CPU smoke while retaining order.
TEMPORAL_FEATURE_NAMES = [
    "left_valid_fraction", "right_valid_fraction",
    "left_mean_confidence", "right_mean_confidence",
    "left_centroid_relative_shoulder_x", "left_centroid_relative_shoulder_y",
    "right_centroid_relative_shoulder_x", "right_centroid_relative_shoulder_y",
    "interhand_distance", "pose_motion", "left_global_motion",
    "right_global_motion", "left_shape_change", "right_shape_change",
    "interhand_distance_change", "left_confidence_change",
    "right_confidence_change",
]


def set_determinism(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(min(8, os.cpu_count() or 1))
    torch.use_deterministic_algorithms(True)


class TemporalSequenceFeatures(RICH.SequenceFeatures):
    """Return ordered causal histories from the audited P2 feature archive."""

    def __init__(self, path):
        super().__init__(path)
        missing = sorted(set(TEMPORAL_FEATURE_NAMES) - set(self.feature_names))
        if missing:
            raise ValueError(f"missing temporal features: {missing}")
        self.temporal_indices = np.asarray(
            [self.feature_names.index(name) for name in TEMPORAL_FEATURE_NAMES], dtype=int
        )
        self._history_cache_name = None
        self._history_cache = None

    @property
    def temporal_width(self):
        return len(self.temporal_indices)

    def histories(self, name):
        """Return [candidate, oldest..newest, feature], left-edge replicated."""
        if name == self._history_cache_name:
            return self._history_cache
        if name not in self.slices:
            raise KeyError(name)
        frames = self.features[self.slices[name]][:, self.temporal_indices].astype(np.float32)
        length = len(frames)
        starts = np.arange(length, dtype=np.int64)
        visible = np.minimum(starts + RICH.WINDOW_VISIBLE_OFFSET, length - 1)
        offsets = np.arange(-HISTORY_FRAMES + 1, 1, dtype=np.int64)
        indices = np.maximum(visible[:, None] + offsets[None, :], 0)
        result = frames[indices]
        self._history_cache_name, self._history_cache = name, result
        return result

    def history(self, name, start):
        return self.histories(name)[int(start)]


class TinyCausalTCN(nn.Module):
    """Valid temporal convolutions; the final activation depends only on history."""

    def __init__(self, visual_width, channels=CHANNELS):
        super().__init__()
        self.temporal = nn.Sequential(
            nn.Conv1d(visual_width, channels, 3, dilation=1), nn.GELU(),
            nn.Conv1d(channels, channels, 3, dilation=2), nn.GELU(),
            nn.Conv1d(channels, channels, 3, dilation=4), nn.GELU(),
        )
        self.fusion = nn.Sequential(
            nn.Linear(channels + 4, 16), nn.GELU(), nn.Linear(16, 1)
        )

    def encode(self, history):
        return self.temporal(history.transpose(1, 2))[:, :, -1]

    def fuse(self, bookkeeping, embedding):
        return self.fusion(torch.cat([bookkeeping, embedding], dim=1)).squeeze(-1)

    def forward(self, bookkeeping, history):
        return self.fuse(bookkeeping, self.encode(history))


def fixed_time_permutation(seed=SEED + 17):
    return np.random.default_rng(seed).permutation(HISTORY_FRAMES).astype(np.int64)


def collect_fit_rows(utility_root, shard_indices, assignments, sequences):
    bookkeeping, targets, utilities, names, starts = [], [], [], [], []
    states = 0
    for state in BASELINE.iter_records(utility_root, shard_indices):
        name = state["sample_id"]
        if assignments[name] != "fit" or name not in sequences.slices:
            continue
        states += 1
        for candidate in state["candidates"]:
            utility = int(candidate["label"]["utility"])
            start = int(candidate["candidate_window_start"])
            if utility != 0 or BASELINE.deterministic_keep_nonpositive(state["state_id"], start):
                bookkeeping.append(BASELINE.raw_features(candidate["predictor_inputs"]))
                targets.append(float(utility > 0))
                utilities.append(utility)
                names.append(name)
                starts.append(start)
    bookkeeping = np.asarray(bookkeeping, dtype=np.float32)
    history = np.empty(
        (len(starts), HISTORY_FRAMES, sequences.temporal_width), dtype=np.float16
    )
    by_name = {}
    for index, name in enumerate(names):
        by_name.setdefault(name, []).append(index)
    for name, row_indices in by_name.items():
        row_indices = np.asarray(row_indices, dtype=np.int64)
        sample_starts = np.asarray([starts[index] for index in row_indices], dtype=np.int64)
        history[row_indices] = sequences.histories(name)[sample_starts].astype(np.float16)
    return (
        bookkeeping, history, np.asarray(targets, dtype=np.float32),
        np.asarray(utilities, dtype=np.int16), states,
    )


def normalization(bookkeeping, histories):
    book_mean = bookkeeping.mean(axis=0, dtype=np.float64).astype(np.float32)
    book_scale = bookkeeping.std(axis=0, dtype=np.float64).astype(np.float32)
    book_scale[book_scale < 1e-6] = 1.0
    total = np.zeros(histories.shape[2], dtype=np.float64)
    total2 = np.zeros_like(total)
    count = 0
    for left in range(0, len(histories), 8192):
        values = histories[left:left + 8192].astype(np.float32).reshape(-1, histories.shape[2])
        total += values.sum(axis=0, dtype=np.float64)
        total2 += np.square(values, dtype=np.float32).sum(axis=0, dtype=np.float64)
        count += len(values)
    visual_mean = (total / count).astype(np.float32)
    visual_scale = np.sqrt(np.maximum(total2 / count - np.square(total / count), 0)).astype(np.float32)
    visual_scale[visual_scale < 1e-6] = 1.0
    return book_mean, book_scale, visual_mean, visual_scale


def train_tcn(bookkeeping, histories, target, permutation=None):
    book_mean, book_scale, visual_mean, visual_scale = normalization(bookkeeping, histories)
    set_determinism(SEED)  # identical initialization/order for ordered and shuffled arms
    network = TinyCausalTCN(histories.shape[2])
    optimizer = torch.optim.AdamW(network.parameters(), lr=2e-3, weight_decay=1e-4)
    positive = float(target.sum())
    criterion = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor((len(target) - positive) / positive)
    )
    generator = torch.Generator().manual_seed(SEED)
    permutation = None if permutation is None else np.asarray(permutation, dtype=np.int64)
    history_log = []
    started = time.perf_counter()
    for epoch in range(EPOCHS):
        network.train()
        order = torch.randperm(len(target), generator=generator).numpy()
        total = 0.0
        for left in range(0, len(order), BATCH_SIZE):
            indices = order[left:left + BATCH_SIZE]
            bx = (bookkeeping[indices] - book_mean) / book_scale
            bh = histories[indices].astype(np.float32)
            if permutation is not None:
                bh = bh[:, permutation, :]
            bh = (bh - visual_mean[None, None, :]) / visual_scale[None, None, :]
            by = target[indices]
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(network(torch.from_numpy(bx), torch.from_numpy(bh)), torch.from_numpy(by))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(network.parameters(), 1.0)
            optimizer.step()
            total += float(loss.detach()) * len(indices)
        history_log.append({"epoch": epoch + 1, "weighted_bce": total / len(target)})
        print(json.dumps({"permuted": permutation is not None, **history_log[-1]}), flush=True)
    network.eval()
    return {
        "network": network, "book_mean": book_mean, "book_scale": book_scale,
        "visual_mean": visual_mean, "visual_scale": visual_scale,
        "permutation": permutation, "history": history_log,
        "training_seconds": time.perf_counter() - started,
    }


def encode_histories(model, histories, batch_size=4096):
    values = np.asarray(histories, dtype=np.float32)
    if model["permutation"] is not None:
        values = values[:, model["permutation"], :]
    result = []
    with torch.no_grad():
        for left in range(0, len(values), batch_size):
            batch = (values[left:left + batch_size] - model["visual_mean"]) / model["visual_scale"]
            result.append(model["network"].encode(torch.from_numpy(batch)).numpy())
    return np.concatenate(result)


def fuse_scores(model, bookkeeping, embedding, batch_size=32768):
    bookkeeping = np.asarray(bookkeeping, dtype=np.float32)
    embedding = np.asarray(embedding, dtype=np.float32)
    result = []
    with torch.no_grad():
        for left in range(0, len(bookkeeping), batch_size):
            bx = (bookkeeping[left:left + batch_size] - model["book_mean"]) / model["book_scale"]
            result.append(torch.sigmoid(model["network"].fuse(
                torch.from_numpy(bx), torch.from_numpy(embedding[left:left + batch_size])
            )).numpy())
    return np.concatenate(result)


def cached_embeddings(models, sequences, name):
    histories = sequences.histories(name)
    return {kind: encode_histories(model, histories) for kind, model in models.items()}


def ranking_metrics(utility_root, shard_indices, assignments, sequences, models):
    accumulators = {
        kind: {"labels": [], "scores": [], "recall_num": 0, "recall_den": 0,
               "ndcg": [], "regret": []} for kind in models
    }
    states = positive_states = candidate_records = positive_candidates = 0
    cache_name = None
    cache = None
    started = time.perf_counter()
    for state in BASELINE.iter_records(utility_root, shard_indices):
        name = state["sample_id"]
        if assignments[name] != "calibration" or name not in sequences.slices:
            continue
        if cache_name != name:
            cache_name, cache = name, cached_embeddings(models, sequences, name)
        utilities = np.asarray([int(row["label"]["utility"]) for row in state["candidates"]])
        labels = utilities > 0
        starts = np.asarray([int(row["candidate_window_start"]) for row in state["candidates"]])
        bookkeeping = np.vstack([
            BASELINE.raw_features(row["predictor_inputs"]) for row in state["candidates"]
        ]).astype(np.float32)
        k = max(1, int(math.ceil(0.1 * len(starts))))
        states += 1
        candidate_records += len(starts)
        positive_candidates += int(labels.sum())
        positive_states += int(labels.any())
        for kind, model in models.items():
            score = fuse_scores(model, bookkeeping, cache[kind][starts])
            chosen = np.argsort(-score, kind="stable")[:k]
            acc = accumulators[kind]
            acc["labels"].extend(labels.tolist())
            acc["scores"].extend(score.tolist())
            acc["recall_num"] += int(labels[chosen].sum())
            acc["recall_den"] += int(labels.sum())
            gains = np.maximum(utilities, 0).astype(np.float64)
            if gains.sum() > 0:
                discount = 1.0 / np.log2(np.arange(2, k + 2))
                ideal = np.sort(gains)[::-1][:k]
                acc["ndcg"].append(float((gains[chosen] * discount).sum()) /
                                   float((ideal * discount).sum()))
            acc["regret"].append(int(utilities.max() - utilities[int(np.argmax(score))]))
    elapsed = time.perf_counter() - started
    result = {}
    for kind, acc in accumulators.items():
        regrets = np.asarray(acc["regret"])
        result[kind] = {
            "calibration_states": states, "positive_utility_states": positive_states,
            "candidate_records": candidate_records, "positive_candidates": positive_candidates,
            "pr_auc_average_precision": BASELINE.average_precision(acc["labels"], acc["scores"]),
            "positive_recall_at_top_10pct": acc["recall_num"] / acc["recall_den"],
            "mean_ndcg_at_top_10pct_positive_states": float(np.mean(acc["ndcg"])),
            "top1_utility_regret": {"mean": float(regrets.mean()),
                                    "positive_fraction": float((regrets > 0).mean())},
            "evaluation_seconds_shared_models": elapsed,
        }
    return result


def learned_schedule(total_windows, model, embeddings):
    total_budget = ORACLE.dense_half_budget(total_windows)
    skeleton_count = ORACLE.skeleton_size(total_budget, ORACLE.PRIMARY_RATIO)
    selected = ORACLE.uniform_positions(total_windows, skeleton_count)
    while len(selected) < total_budget:
        selected_set = set(selected)
        candidates = [value for value in range(total_windows) if value not in selected_set]
        bookkeeping = np.vstack([
            BASELINE.raw_features(BASELINE.state_candidate_inputs(value, selected))
            for value in candidates
        ]).astype(np.float32)
        scores = fuse_scores(model, bookkeeping, embeddings[np.asarray(candidates)])
        best = min(range(len(candidates)), key=lambda i: (-float(scores[i]), candidates[i]))
        selected.append(candidates[best])
        selected.sort()
    return selected


def schedule_replay(dense_root, shard_indices, assignments, vocab, sequences, models):
    blank_id = vocab.index("<blank>")
    rows = {kind: [] for kind in models}
    uniform_rows, records = [], []
    schedule_seconds = Counter()
    decode_seconds = Counter()
    started = time.perf_counter()
    protocol = json.loads((dense_root / "protocol_manifest.json").read_text())
    shard_count = int(math.ceil(protocol["samples"] / protocol["shard_samples"]))
    for shard_index in shard_indices:
        results, logits, _ = BUILDER.validate_dense_shard(
            dense_root, shard_index, shard_count, verify_hashes=True
        )
        for name in results:
            if assignments[name] != "calibration" or name not in sequences.slices:
                continue
            probabilities = ORACLE.softmax_rows(logits[name])
            budget = ORACLE.dense_half_budget(len(probabilities))
            uniform = ORACLE.uniform_positions(len(probabilities), budget)
            reference = REPLAY.clean_phoenix_2014_trans(results[name]["gls_ref"])
            tick = time.perf_counter()
            uniform_hypothesis = ORACLE.decode_probabilities(probabilities, uniform, vocab, blank_id)
            decode_seconds["uniform"] += time.perf_counter() - tick
            uniform_row = BASELINE.sentence_row(reference, uniform_hypothesis)
            uniform_rows.append(uniform_row)
            record = {"sample_id": name, "dense_windows": len(probabilities), "budget": budget,
                      "uniform_selected": uniform, "uniform_counts": uniform_row}
            embeddings = cached_embeddings(models, sequences, name)
            for kind, model in models.items():
                tick = time.perf_counter()
                selected = learned_schedule(len(probabilities), model, embeddings[kind])
                schedule_seconds[kind] += time.perf_counter() - tick
                if len(selected) != budget or len(set(selected)) != budget:
                    raise AssertionError("learned schedule violates exact equal budget")
                tick = time.perf_counter()
                hypothesis = ORACLE.decode_probabilities(probabilities, selected, vocab, blank_id)
                decode_seconds[kind] += time.perf_counter() - tick
                row = BASELINE.sentence_row(reference, hypothesis)
                rows[kind].append(row)
                record[f"{kind}_selected"] = selected
                record[f"{kind}_counts"] = row
            records.append(record)
    uniform = BASELINE.aggregate_wer(uniform_rows)
    result = {"samples": len(records), "equal_budget_uniform": uniform,
              "replay_seconds_all_models": time.perf_counter() - started,
              "controller_schedule_seconds": dict(schedule_seconds),
              "decoder_replay_seconds": dict(decode_seconds)}
    for kind in models:
        aggregate = BASELINE.aggregate_wer(rows[kind])
        result[kind] = aggregate
        result[f"{kind}_minus_uniform"] = {
            "wer_pp": aggregate["wer"] - uniform["wer"],
            "error_count": aggregate["error"] - uniform["error"],
        }
    return records, result


def save_model(path, model):
    torch.save({
        "state_dict": model["network"].state_dict(),
        "book_mean_fit_only": model["book_mean"], "book_scale_fit_only": model["book_scale"],
        "visual_mean_fit_only": model["visual_mean"],
        "visual_scale_fit_only": model["visual_scale"],
        "temporal_feature_names": TEMPORAL_FEATURE_NAMES,
        "time_permutation": model["permutation"], "training_history": model["history"],
    }, path)


def audit_decoder_prefix(utility_root):
    """Document why a causal prefix cannot be attached to these oracle states."""
    return {
        "included": False,
        "reason": (
            "initial_skeleton is a complete known-EOS set and contains windows after a "
            "candidate decision time; selected_bonus_before follows offline oracle utility "
            "order rather than chronological arrival. Decoding their union would leak future."
        ),
        "required_reconstruction": (
            "rebuild states in frame-arrival order under an unknown-EOS token bucket; at each "
            "decision persist only logits of expensive windows already executed by that time, "
            "then derive prefix tokens/entropy/stability from that causal history"
        ),
        "utility_root": str(Path(utility_root).resolve()),
    }


def run(utility_root, dense_root, sequences_path, previous_path, output_root, vocab_path):
    for path in (utility_root, dense_root, sequences_path, previous_path, output_root, vocab_path):
        BASELINE.reject_test_path(path)
    if output_root.exists():
        raise FileExistsError(f"refusing to overwrite {output_root}")
    set_determinism()
    overall_started = time.perf_counter()
    summary = json.loads((utility_root / "dataset_summary.json").read_text())
    shard_indices = summary["selected_shard_indices"]
    split_path = utility_root / "fit_calibration_split.json"
    assignments = json.loads(split_path.read_text())["assignments"]
    previous = json.loads(previous_path.read_text())
    sequences = TemporalSequenceFeatures(sequences_path)
    selected_samples = []
    for shard_index in shard_indices:
        _, complete = BUILDER.output_paths(utility_root, shard_index, summary["shard_count"])
        selected_samples.extend(json.loads(complete.read_text())["sample_ids"])
    excluded = sorted(set(selected_samples) - set(sequences.names))
    if excluded != previous["shared_data"]["excluded_samples"]:
        raise ValueError("sample exclusion differs from previous common comparison")
    bookkeeping, histories, target, utilities, fit_states = collect_fit_rows(
        utility_root, shard_indices, assignments, sequences
    )
    if len(target) != previous["shared_data"]["sampled_candidates"]:
        raise ValueError("fit label sample differs from previous comparison")
    permutation = fixed_time_permutation()
    models = {
        "causal_ordered_tcn": train_tcn(bookkeeping, histories, target),
        "fixed_time_shuffled_tcn": train_tcn(bookkeeping, histories, target, permutation),
    }
    ranking = ranking_metrics(
        utility_root, shard_indices, assignments, sequences, models
    )
    with vocab_path.open("r", encoding="utf-8") as handle:
        vocab = json.load(handle)
    records, replay = schedule_replay(
        dense_root, shard_indices, assignments, vocab, sequences, models
    )
    if replay["samples"] != previous["schedule_replay"]["samples"]:
        raise ValueError("calibration sample count differs from previous comparison")
    if replay["equal_budget_uniform"] != previous["schedule_replay"]["equal_budget_uniform"]:
        raise ValueError("uniform replay differs from previous comparison")
    output_root.mkdir(parents=True)
    schedules_path = output_root / "calibration_schedules.jsonl.gz"
    with gzip.open(schedules_path, "wt", encoding="utf-8", compresslevel=6) as handle:
        for record in records:
            handle.write(json.dumps(record, separators=(",", ":")) + "\n")
    model_paths = []
    for kind, model in models.items():
        path = output_root / f"{kind}.pt"
        save_model(path, model)
        model_paths.append(path)
    previous_comparators = {
        kind: {
            "ranking": previous["ranking"][kind],
            "schedule_replay": previous["schedule_replay"][kind],
            "minus_uniform": previous["schedule_replay"][f"{kind}_minus_uniform"],
        }
        for kind in ("bookkeeping_linear", "bookkeeping_mlp", "causal_pose_handshape_mlp")
    }
    metrics = {
        "scope": "partial-train ordered-history TCN smoke only; not a research result",
        "gpu_used": False, "dev_used": False, "test_used": False,
        "limitations": [
            "only 32/56 non-random available train dense shards",
            "oracle utility states are non-chronological",
            "known-EOS replay chooses a complete schedule offline",
            "no formal paired CI, real system wall-time, controller-cost, or submission-latency claim",
            "one P2-excluded calibration sample omitted consistently; common set is 526 samples",
        ],
        "causality_contract": {
            "candidate_coordinate": "padded 16-frame ISLR window start",
            "latest_original_frame_visible": "min(T-1, candidate_start+8)",
            "history": "31 ordered cheap-feature frames, oldest to newest, ending at latest visible",
            "left_edge": "replicate original frame zero; never use a later frame",
            "forbidden": ["future frames", "reference", "candidate expensive logits",
                          "exact EOS", "exact total or remaining budget"],
        },
        "decoder_prefix_audit": audit_decoder_prefix(utility_root),
        "shared_data": {
            "fit_states": fit_states, "sampled_candidates": len(target),
            "sampled_positive_candidates": int(target.sum()),
            "sampled_utility_counts": dict(sorted(Counter(map(int, utilities)).items())),
            "excluded_samples": excluded,
        },
        "models": {
            kind: {"model": "three-layer valid-convolution tiny causal TCN",
                   "channels": CHANNELS, "epochs": EPOCHS, "batch_size": BATCH_SIZE,
                   "temporal_features": TEMPORAL_FEATURE_NAMES,
                   "time_permutation": (None if model["permutation"] is None
                                        else model["permutation"].tolist()),
                   "training_history": model["history"],
                   "training_seconds": model["training_seconds"]}
            for kind, model in models.items()
        },
        "ranking": ranking, "schedule_replay": replay,
        "previous_common_comparators": previous_comparators,
        "runtime_seconds": time.perf_counter() - overall_started,
    }
    config = {
        "seed": SEED, "epochs": EPOCHS, "batch_size": BATCH_SIZE, "channels": CHANNELS,
        "temporal_feature_names": TEMPORAL_FEATURE_NAMES,
        "ablation": "same model/init/order with one fixed permutation of 31 time positions",
        "schedule": "known-EOS 50% total budget; 50% uniform skeleton plus learned bonus",
        "decoder": "frozen span15", "selected_dense_shards": shard_indices,
        "no_hyperparameter_selection_on_calibration": True,
    }
    BASELINE.atomic_json(output_root / "metrics.json", metrics)
    BASELINE.atomic_json(output_root / "resolved_config.json", config)
    inputs = [Path(__file__), sequences_path, previous_path,
              utility_root / "dataset_summary.json", utility_root / "protocol_manifest.json",
              split_path, dense_root / "protocol_manifest.json", vocab_path]
    outputs = [output_root / "metrics.json", output_root / "resolved_config.json",
               schedules_path, *model_paths]
    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "partial32_ordered_tcn_smoke_not_frozen",
        "inputs": {str(path.resolve()): {"bytes": path.stat().st_size,
                    "sha256": BASELINE.sha256_file(path)} for path in inputs},
        "outputs": {path.name: {"bytes": path.stat().st_size,
                     "sha256": BASELINE.sha256_file(path)} for path in outputs},
    }
    BASELINE.atomic_json(output_root / "manifest.json", manifest)
    return metrics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--utility-root", type=Path, default=DEFAULT_UTILITY_ROOT)
    parser.add_argument("--dense-root", type=Path, default=DEFAULT_DENSE_ROOT)
    parser.add_argument("--sequences", type=Path, default=DEFAULT_SEQUENCES)
    parser.add_argument("--previous", type=Path, default=DEFAULT_PREVIOUS)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--vocab", type=Path, default=BUILDER.DEFAULT_VOCAB)
    args = parser.parse_args()
    paths = (args.utility_root, args.dense_root, args.sequences, args.previous,
             args.output_root, args.vocab)
    metrics = run(*(path.resolve() for path in paths))
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
