#!/usr/bin/env python3
"""CPU-only rich-predictor smoke on the incomplete Phoenix train replay.

This is deliberately not an online-system result.  The utility states are
non-chronological and replay has known EOS.  Predictor features, however, are
causal at each candidate: bookkeeping plus an optional 31-frame keypoint
history ending when the centered 16-frame ISLR window becomes available.
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


BASELINE = import_tool("analyze_phoenix_partial_utility_pipeline")
BUILDER = BASELINE.BUILDER
ORACLE = BASELINE.ORACLE
REPLAY = BASELINE.REPLAY

DEFAULT_UTILITY_ROOT = BASELINE.DEFAULT_UTILITY_ROOT
DEFAULT_DENSE_ROOT = BUILDER.DEFAULT_DENSE_ROOT
DEFAULT_SEQUENCES = Path("/tmp/phoenix_p2_train_center_sequences_v1.npz")
DEFAULT_OUTPUT_ROOT = (
    ROOT / "results/phoenix-2014t_ISLR/p3_partial32_rich_predictors_smoke_v1_49faacc3"
)
WINDOW_SIZE = 16
PAD_LEFT = 7
WINDOW_VISIBLE_OFFSET = WINDOW_SIZE - 1 - PAD_LEFT  # start s sees original through s+8
HISTORY_FRAMES = 31
SEED = 260917


def set_determinism(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(min(8, os.cpu_count() or 1))
    torch.use_deterministic_algorithms(True)


class SequenceFeatures:
    """Access precomputed causal P2 frame features without target labels."""

    def __init__(self, path):
        path = Path(path)
        BASELINE.reject_test_path(path)
        data = np.load(path)
        required = {"features", "video_names", "video_offsets", "feature_names"}
        if not required.issubset(data.files):
            raise ValueError("causal sequence archive lacks required fields")
        self.path = path
        self.features = data["features"]
        self.names = [str(value) for value in data["video_names"].tolist()]
        self.offsets = data["video_offsets"].astype(np.int64)
        self.feature_names = [str(value) for value in data["feature_names"].tolist()]
        if len(self.offsets) != len(self.names) + 1 or self.features.shape[1] != len(self.feature_names):
            raise ValueError("invalid causal sequence archive dimensions")
        self.slices = {
            name: slice(int(self.offsets[index]), int(self.offsets[index + 1]))
            for index, name in enumerate(self.names)
        }
        self._cache_name = None
        self._cache = None

    @property
    def visual_width(self):
        return 2 * self.features.shape[1]

    def descriptors(self, name):
        """Return [current, causal-31-frame-mean] at every candidate coordinate."""
        if name == self._cache_name:
            return self._cache
        if name not in self.slices:
            raise KeyError(name)
        frames = self.features[self.slices[name]].astype(np.float32)
        length = len(frames)
        cumulative = np.vstack(
            [np.zeros((1, frames.shape[1]), dtype=np.float64), np.cumsum(frames, axis=0)]
        )
        candidate = np.arange(length, dtype=np.int64)
        visible = np.minimum(candidate + WINDOW_VISIBLE_OFFSET, length - 1)
        left = np.maximum(visible - HISTORY_FRAMES + 1, 0)
        means = (cumulative[visible + 1] - cumulative[left]) / (visible - left + 1)[:, None]
        result = np.concatenate([frames[visible], means.astype(np.float32)], axis=1)
        self._cache_name, self._cache = name, result
        return result

    def candidate(self, name, start):
        return self.descriptors(name)[int(start)]


class MLP(nn.Module):
    def __init__(self, width, hidden=32):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(width, hidden), nn.GELU(), nn.Linear(hidden, hidden), nn.GELU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, value):
        return self.layers(value).squeeze(-1)


def collect_rows(utility_root, shard_indices, assignments, sequences):
    bookkeeping, visual, target, utility = [], [], [], []
    states = 0
    for state in BASELINE.iter_records(utility_root, shard_indices):
        name = state["sample_id"]
        if assignments[name] != "fit" or name not in sequences.slices:
            continue
        states += 1
        descriptors = sequences.descriptors(name)
        for candidate in state["candidates"]:
            value = int(candidate["label"]["utility"])
            start = int(candidate["candidate_window_start"])
            if value != 0 or BASELINE.deterministic_keep_nonpositive(state["state_id"], start):
                bookkeeping.append(BASELINE.raw_features(candidate["predictor_inputs"]))
                visual.append(descriptors[start])
                target.append(float(value > 0))
                utility.append(value)
    return (
        np.asarray(bookkeeping, dtype=np.float32), np.asarray(visual, dtype=np.float32),
        np.asarray(target, dtype=np.float32), np.asarray(utility, dtype=np.int16), states,
    )


def fit_normalization(matrix):
    mean = matrix.mean(axis=0, dtype=np.float64).astype(np.float32)
    scale = matrix.std(axis=0, dtype=np.float64).astype(np.float32)
    scale[scale < 1e-6] = 1.0
    return mean, scale


def train_mlp(matrix, target, epochs=4, batch_size=4096, seed=SEED):
    mean, scale = fit_normalization(matrix)
    x = np.asarray((matrix - mean) / scale, dtype=np.float32)
    y = np.asarray(target, dtype=np.float32)
    positive = float(y.sum())
    if positive <= 0 or positive >= len(y):
        raise ValueError("training sample lacks both classes")
    model = MLP(x.shape[1])
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor((len(y) - positive) / positive))
    generator = torch.Generator().manual_seed(seed)
    history = []
    started = time.perf_counter()
    for epoch in range(epochs):
        model.train()
        order = torch.randperm(len(y), generator=generator)
        total = 0.0
        for left in range(0, len(y), batch_size):
            indices = order[left:left + batch_size].numpy()
            bx = torch.from_numpy(x[indices])
            by = torch.from_numpy(y[indices])
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(bx), by)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total += float(loss.detach()) * len(indices)
        history.append({"epoch": epoch + 1, "weighted_bce": total / len(y)})
        print(json.dumps({"training": matrix.shape[1], **history[-1]}), flush=True)
    model.eval()
    return {"network": model, "mean": mean, "scale": scale, "history": history,
            "training_seconds": time.perf_counter() - started}


def predict_mlp(model, matrix, batch_size=32768):
    values = np.asarray(matrix, dtype=np.float32)
    if values.ndim == 1:
        values = values[None, :]
    values = (values - model["mean"]) / model["scale"]
    result = []
    with torch.no_grad():
        for left in range(0, len(values), batch_size):
            logits = model["network"](torch.from_numpy(values[left:left + batch_size]))
            result.append(torch.sigmoid(logits).numpy())
    return np.concatenate(result)


def predict_model(kind, model, matrix):
    if kind == "bookkeeping_linear":
        return BASELINE.predict(model, matrix)
    return predict_mlp(model, matrix)


def feature_matrix(kind, bookkeeping, visual):
    if kind in ("bookkeeping_linear", "bookkeeping_mlp"):
        return bookkeeping
    if kind == "causal_pose_handshape_mlp":
        return np.concatenate([bookkeeping, visual], axis=1)
    raise ValueError(kind)


def candidate_matrix(kind, state, sequences, name):
    bookkeeping = np.vstack(
        [BASELINE.raw_features(row["predictor_inputs"]) for row in state["candidates"]]
    ).astype(np.float32)
    if kind in ("bookkeeping_linear", "bookkeeping_mlp"):
        return bookkeeping
    starts = np.asarray([row["candidate_window_start"] for row in state["candidates"]], dtype=int)
    return np.concatenate([bookkeeping, sequences.descriptors(name)[starts]], axis=1)


def ranking_metrics(utility_root, shard_indices, assignments, sequences, kind, model):
    all_labels, all_scores = [], []
    recall_numerator = recall_denominator = 0
    ndcg_values, regrets = [], []
    states = positive_states = 0
    started = time.perf_counter()
    for state in BASELINE.iter_records(utility_root, shard_indices):
        name = state["sample_id"]
        if assignments[name] != "calibration" or name not in sequences.slices:
            continue
        states += 1
        utilities = np.asarray([int(row["label"]["utility"]) for row in state["candidates"]])
        scores = predict_model(kind, model, candidate_matrix(kind, state, sequences, name))
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
        "calibration_states": states, "positive_utility_states": positive_states,
        "candidate_records": len(all_labels), "positive_candidates": int(sum(all_labels)),
        "pr_auc_average_precision": BASELINE.average_precision(all_labels, all_scores),
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
        "evaluation_seconds": time.perf_counter() - started,
    }


def score_schedule_candidates(kind, model, sequences, name, candidates, selected):
    bookkeeping = np.vstack([
        BASELINE.raw_features(BASELINE.state_candidate_inputs(value, selected))
        for value in candidates
    ]).astype(np.float32)
    if kind == "causal_pose_handshape_mlp":
        visual = sequences.descriptors(name)[np.asarray(candidates, dtype=int)]
        bookkeeping = np.concatenate([bookkeeping, visual], axis=1)
    return predict_model(kind, model, bookkeeping)


def learned_schedule(total_windows, kind, model, sequences, name):
    total_budget = ORACLE.dense_half_budget(total_windows)
    skeleton_count = ORACLE.skeleton_size(total_budget, ORACLE.PRIMARY_RATIO)
    selected = ORACLE.uniform_positions(total_windows, skeleton_count)
    while len(selected) < total_budget:
        selected_set = set(selected)
        candidates = [value for value in range(total_windows) if value not in selected_set]
        scores = score_schedule_candidates(kind, model, sequences, name, candidates, selected)
        best = min(range(len(candidates)), key=lambda index: (-float(scores[index]), candidates[index]))
        selected.append(candidates[best])
        selected.sort()
    return selected


def schedule_replay(dense_root, shard_indices, assignments, vocab, sequences, models):
    blank_id = vocab.index("<blank>")
    rows = {kind: [] for kind in models}
    uniform_rows, records = [], []
    started = time.perf_counter()
    dense_protocol = json.loads((dense_root / "protocol_manifest.json").read_text())
    shard_count = int(math.ceil(dense_protocol["samples"] / dense_protocol["shard_samples"]))
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
            uniform_hypothesis = ORACLE.decode_probabilities(probabilities, uniform, vocab, blank_id)
            uniform_row = BASELINE.sentence_row(reference, uniform_hypothesis)
            uniform_rows.append(uniform_row)
            record = {"sample_id": name, "dense_windows": len(probabilities), "budget": budget,
                      "uniform_selected": uniform, "uniform_counts": uniform_row}
            for kind, model in models.items():
                selected = learned_schedule(len(probabilities), kind, model, sequences, name)
                if len(selected) != budget or len(set(selected)) != budget:
                    raise AssertionError("learned schedule violates exact equal budget")
                hypothesis = ORACLE.decode_probabilities(probabilities, selected, vocab, blank_id)
                row = BASELINE.sentence_row(reference, hypothesis)
                rows[kind].append(row)
                record[f"{kind}_selected"] = selected
                record[f"{kind}_counts"] = row
            records.append(record)
    uniform = BASELINE.aggregate_wer(uniform_rows)
    metrics = {"samples": len(records), "equal_budget_uniform": uniform,
               "replay_seconds_all_models": time.perf_counter() - started}
    for kind in models:
        aggregate = BASELINE.aggregate_wer(rows[kind])
        metrics[kind] = aggregate
        metrics[f"{kind}_minus_uniform"] = {
            "wer_pp": aggregate["wer"] - uniform["wer"],
            "error_count": aggregate["error"] - uniform["error"],
        }
    return records, metrics


def save_model(path, model, feature_names):
    torch.save({"state_dict": model["network"].state_dict(), "mean_fit_only": model["mean"],
                "scale_fit_only": model["scale"], "feature_names": feature_names,
                "training_history": model["history"]}, path)


def save_linear_model(path, model, feature_names):
    np.savez(path, coefficient=model["coefficient"], mean_fit_only=model["mean"],
             scale_fit_only=model["scale"], feature_names=np.asarray(feature_names),
             training_loss=np.asarray(model["loss"]))


def run(utility_root, dense_root, sequences_path, output_root, vocab_path):
    for path in (utility_root, dense_root, sequences_path, output_root, vocab_path):
        BASELINE.reject_test_path(path)
    if output_root.exists():
        raise FileExistsError(f"refusing to overwrite {output_root}")
    set_determinism()
    overall_started = time.perf_counter()
    summary = json.loads((utility_root / "dataset_summary.json").read_text())
    shard_indices = summary["selected_shard_indices"]
    split_path = utility_root / "fit_calibration_split.json"
    assignments = json.loads(split_path.read_text())["assignments"]
    sequences = SequenceFeatures(sequences_path)
    available = set(sequences.names)
    selected_samples = []
    for shard_index in shard_indices:
        _, complete = BUILDER.output_paths(utility_root, shard_index, summary["shard_count"])
        selected_samples.extend(json.loads(complete.read_text())["sample_ids"])
    excluded = sorted(set(selected_samples) - available)
    if excluded != ["train/25August_2009_Tuesday_heute-3301"]:
        raise ValueError(f"unexpected sequence coverage difference: {excluded}")
    bookkeeping, visual, target, utilities, fit_states = collect_rows(
        utility_root, shard_indices, assignments, sequences
    )
    models = {}
    linear_started = time.perf_counter()
    models["bookkeeping_linear"] = BASELINE.train_logistic(
        bookkeeping.astype(np.float64), target.astype(np.float64)
    )
    models["bookkeeping_linear"]["training_seconds"] = time.perf_counter() - linear_started
    models["bookkeeping_mlp"] = train_mlp(bookkeeping, target)
    models["causal_pose_handshape_mlp"] = train_mlp(
        feature_matrix("causal_pose_handshape_mlp", bookkeeping, visual), target
    )
    ranking = {
        kind: ranking_metrics(utility_root, shard_indices, assignments, sequences, kind, model)
        for kind, model in models.items()
    }
    with vocab_path.open("r", encoding="utf-8") as handle:
        vocab = json.load(handle)
    records, replay = schedule_replay(
        dense_root, shard_indices, assignments, vocab, sequences, models
    )
    output_root.mkdir(parents=True)
    schedules_path = output_root / "calibration_schedules.jsonl.gz"
    with gzip.open(schedules_path, "wt", encoding="utf-8", compresslevel=6) as handle:
        for record in records:
            handle.write(json.dumps(record, separators=(",", ":")) + "\n")
    bookkeeping_names = [
        "log1p(candidate_window_start)", "log1p(selected_past_count)",
        "log1p(frames_since_last_selected_past)", "has_selected_past",
    ]
    visual_names = [f"current_at_window_availability:{name}" for name in sequences.feature_names]
    visual_names += [f"causal_mean_31:{name}" for name in sequences.feature_names]
    model_paths = {}
    for kind, model in models.items():
        names = (bookkeeping_names if kind in ("bookkeeping_linear", "bookkeeping_mlp")
                 else bookkeeping_names + visual_names)
        if kind == "bookkeeping_linear":
            path = output_root / f"{kind}.npz"
            save_linear_model(path, model, names)
        else:
            path = output_root / f"{kind}.pt"
            save_model(path, model, names)
        model_paths[kind] = path
    metrics = {
        "scope": "partial-train rich-predictor smoke only; not a research result",
        "gpu_used": False, "dev_used": False, "test_used": False,
        "limitations": [
            "only 32/56 non-random available train dense shards",
            "one P2-excluded calibration sample omitted consistently from all comparisons",
            "oracle training states are non-chronological",
            "known-EOS replay chooses a complete schedule offline",
            "no paired bootstrap CI, real wall-time, controller-cost, or latency claim",
            "fixed smoke hyperparameters were not selected or frozen for a formal experiment",
        ],
        "causality_contract": {
            "window_size": WINDOW_SIZE, "padding_left": PAD_LEFT,
            "candidate_coordinate": "padded 16-frame ISLR window start",
            "latest_original_frame_visible": "min(T-1, candidate_start+8)",
            "visual_history": "31 frames ending at latest_original_frame_visible",
            "forbidden": ["future frames", "reference", "candidate expensive logits",
                          "exact EOS", "exact total or remaining budget"],
        },
        "shared_data": {
            "fit_states": fit_states, "sampled_candidates": len(target),
            "sampled_positive_candidates": int(target.sum()),
            "sampled_utility_counts": dict(sorted(Counter(map(int, utilities)).items())),
            "negative_zero_sampling": "all nonzero plus deterministic 2% of zero labels",
            "excluded_samples": excluded,
        },
        "models": {
            kind: ({"input_width": len(model["mean"]), "model": "linear_logistic",
                    "training_history": model["loss"],
                    "training_seconds": model["training_seconds"]}
                   if kind == "bookkeeping_linear" else
                   {"input_width": len(model["mean"]), "model": "two_hidden_layer_mlp",
                    "hidden_width": 32, "training_history": model["history"],
                    "training_seconds": model["training_seconds"]})
            for kind, model in models.items()
        },
        "ranking": ranking, "schedule_replay": replay,
        "runtime_seconds": time.perf_counter() - overall_started,
    }
    config = {
        "seed": SEED, "epochs": 4, "batch_size": 4096,
        "predictors": ["bookkeeping_linear", "bookkeeping_mlp", "causal_pose_handshape_mlp"],
        "visual_descriptor": "current plus mean over causal history up to 31 frames",
        "schedule": "known-EOS 50% total budget; 50% uniform skeleton plus learned bonus",
        "decoder": "frozen span15", "selected_dense_shards": shard_indices,
    }
    BASELINE.atomic_json(output_root / "metrics.json", metrics)
    BASELINE.atomic_json(output_root / "resolved_config.json", config)
    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "rich_predictor_pipeline_smoke_not_frozen",
        "inputs": {
            str(path.resolve()): {"bytes": path.stat().st_size,
                                  "sha256": BASELINE.sha256_file(path)}
            for path in (Path(__file__), sequences_path, utility_root / "dataset_summary.json",
                         utility_root / "protocol_manifest.json", split_path,
                         dense_root / "protocol_manifest.json", vocab_path)
        },
        "outputs": {
            path.name: {"bytes": path.stat().st_size, "sha256": BASELINE.sha256_file(path)}
            for path in [output_root / "metrics.json", output_root / "resolved_config.json",
                         schedules_path, *model_paths.values()]
        },
    }
    BASELINE.atomic_json(output_root / "manifest.json", manifest)
    return metrics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--utility-root", type=Path, default=DEFAULT_UTILITY_ROOT)
    parser.add_argument("--dense-root", type=Path, default=DEFAULT_DENSE_ROOT)
    parser.add_argument("--sequences", type=Path, default=DEFAULT_SEQUENCES)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--vocab", type=Path, default=BUILDER.DEFAULT_VOCAB)
    args = parser.parse_args()
    metrics = run(*(path.resolve() for path in (
        args.utility_root, args.dense_root, args.sequences, args.output_root, args.vocab
    )))
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
