#!/usr/bin/env python3
"""CPU-only predictability smoke for chronological rollout-16 advantage.

The offline teacher may use a reference and future dense replay to construct a
label.  Every predictor input is available when the current window arrives:
bookkeeping, past-selected decoder state, and (optionally) a 31-frame causal
pose/handshape/motion history.  Calibration is evaluation-only.
"""
import argparse
import gzip
import hashlib
import importlib.util
import json
import math
import os
import random
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET_ROOT = ROOT / "results/phoenix-2014t_ISLR/p3_partial32_rollout_predictor_dataset_smoke_v1_49faacc3"
DEFAULT_SEQUENCES = Path("/tmp/phoenix_p2_train_center_sequences_v1.npz")
DEFAULT_OUTPUT_ROOT = ROOT / "results/phoenix-2014t_ISLR/p3_partial32_rollout_predictor_closed_loop_smoke_v2_49faacc3"
DEFAULT_DENSE_ROOT = DATASET_DENSE_PLACEHOLDER = ROOT / "results/phoenix-2014t_ISLR/train_dense_stride1_v1_49faacc3"
SEED = 260924
HISTORY = 31
VISIBLE_OFFSET = 8
# Frozen in the preceding causal-TCN smoke.  Using this audited inexpensive
# subset avoids materializing 31 x 211 float32 values for every decision.
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


def import_tool(name):
    path = ROOT / f"tools/{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


DATA_BUILDER = import_tool("build_phoenix_partial_rollout_predictor_dataset")


def set_determinism(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(min(8, os.cpu_count() or 1))
    torch.use_deterministic_algorithms(True)


def reject_test_path(path):
    if "test" in str(Path(path)).lower().replace("latest", ""):
        raise ValueError(f"test path forbidden: {path}")


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


class SequenceFeatures:
    """Read causal P2 inputs and expose left-padded histories at availability."""

    def __init__(self, path):
        reject_test_path(path)
        data = np.load(path)
        required = {"features", "video_names", "video_offsets", "feature_names"}
        if not required.issubset(data.files):
            raise ValueError("sequence archive lacks required fields")
        self.features = data["features"]
        self.names = [str(x) for x in data["video_names"].tolist()]
        self.offsets = np.asarray(data["video_offsets"], dtype=np.int64)
        self.feature_names = [str(x) for x in data["feature_names"].tolist()]
        if len(self.offsets) != len(self.names) + 1:
            raise ValueError("invalid sequence offsets")
        self.slices = {
            name: slice(int(self.offsets[i]), int(self.offsets[i + 1]))
            for i, name in enumerate(self.names)
        }
        missing = sorted(set(TEMPORAL_FEATURE_NAMES) - set(self.feature_names))
        if missing:
            raise ValueError(f"missing temporal features: {missing}")
        self.temporal_indices = np.asarray(
            [self.feature_names.index(name) for name in TEMPORAL_FEATURE_NAMES], dtype=np.int64
        )
        self._cache_name = None
        self._cache_frames = None

    @property
    def width(self):
        return len(self.temporal_indices)

    def history(self, name, start):
        if name != self._cache_name:
            self._cache_name = name
            self._cache_frames = np.asarray(
                self.features[self.slices[name]][:, self.temporal_indices], dtype=np.float16
            )
        frames = self._cache_frames
        # Tail windows arrive only after EOS has actually been observed; clamp
        # to the last real frame exactly as dense generation does.  The EOS
        # flag itself is never exposed to the predictor.
        visible = min(len(frames) - 1, int(start) + VISIBLE_OFFSET)
        left = max(0, visible - HISTORY + 1)
        values = frames[left:visible + 1]
        if len(values) < HISTORY:
            values = np.concatenate([
                np.repeat(values[:1], HISTORY - len(values), axis=0), values
            ], axis=0)
        return values


def bookkeeping_features(row):
    value = row["bookkeeping"]
    period = int(value["candidate_mod_skeleton_period"])
    one_hot = [float(period == i) for i in range(4)]
    return np.asarray([
        math.log1p(int(value["candidate_absolute_index"])),
        math.log1p(int(value["coverage_gap"])),
        math.log1p(int(value["frames_since_last_execute"])),
        float(value["bonus_token_balance"]),
        math.log1p(int(value["past_selected_count"])),
        *one_hot,
    ], dtype=np.float32)


def prefix_features(row):
    value = row["prefix"]
    hashed = [float(x) for x in value["decoder_token_hash32"]]
    return np.asarray([
        math.log1p(int(value["decoder_prefix_length"])),
        math.log1p(int(value["decoder_raw_path_length"])),
        float(value["decoder_mean_entropy"]),
        float(value["decoder_last_entropy"]),
        float(value["decoder_entropy_delta"]),
        float(value["decoder_last_is_blank"]),
        float(value["decoder_last_repeats_previous"]),
        math.log1p(int(value["decoder_last_repeat_run"])),
        *hashed,
    ], dtype=np.float32)


def completed_paths(dataset_root):
    manifest_path = dataset_root / "dataset_manifest.json"
    if not manifest_path.is_file():
        raise RuntimeError("rollout dataset is not atomically complete")
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("status") != "complete":
        raise RuntimeError("rollout dataset manifest is not complete")
    paths = []
    for shard in manifest["shards"]:
        index = int(shard["shard"])
        # The dense shard count is encoded in each filename and can differ from
        # the number of completed partial shards (32 versus 56).
        matches = sorted(dataset_root.glob(f"rollout-predictor-{index:05d}-of-*.jsonl.gz"))
        if len(matches) != 1 or sha256_file(matches[0]) != shard["sha256"]:
            raise ValueError(f"missing or corrupt rollout shard {index}")
        paths.append(matches[0])
    return manifest, paths


def iter_rows(paths):
    for path in paths:
        with gzip.open(path, "rt", encoding="utf-8") as handle:
            for line in handle:
                row = json.loads(line)
                if not row["autonomous_training_row"]:
                    continue
                provenance = row["provenance"]
                if not (provenance["current_candidate_expensive_logits_absent_from_inputs"]
                        and provenance["prefix_uses_only_past_selected"]
                        and provenance["total_length_absent_from_inputs"]):
                    raise ValueError("causality provenance failed")
                yield row


def collect(paths, sequences):
    values = {"fit": [], "calibration": []}
    missing = Counter()
    for row in iter_rows(paths):
        partition = row["partition"]
        if partition not in values:
            raise ValueError(f"unexpected partition {partition}")
        name = row["sample_id"]
        if name not in sequences.slices:
            missing[name] += 1
            continue
        values[partition].append({
            "bookkeeping": bookkeeping_features(row),
            "prefix": prefix_features(row),
            "visual": sequences.history(name, row["candidate_window_start"]).astype(np.float16),
            "advantage": float(row["label"]["rollout16_signed_advantage"]),
            "positive": float(row["label"]["rollout16_positive"]),
        })
    if not values["fit"] or not values["calibration"]:
        raise ValueError("both fit and calibration rows are required")
    return values, dict(missing)


def stack_rows(rows):
    result = {
        key: np.stack([row[key] for row in rows]).astype(np.float32)
        for key in ("bookkeeping", "prefix")
    } | {
        "advantage": np.asarray([row["advantage"] for row in rows], dtype=np.float32),
        "positive": np.asarray([row["positive"] for row in rows], dtype=np.float32),
    }
    result["visual"] = np.stack([row["visual"] for row in rows]).astype(np.float16)
    return result


class CausalVisualEncoder(nn.Module):
    def __init__(self, width, hidden=24):
        super().__init__()
        self.input = nn.Conv1d(width, hidden, 1)
        self.layers = nn.ModuleList([nn.Conv1d(hidden, hidden, 3, dilation=d) for d in (1, 2, 4)])

    def forward(self, x):
        x = self.input(x.transpose(1, 2))
        for layer, dilation in zip(self.layers, (1, 2, 4)):
            residual = x
            x = layer(F.pad(x, (2 * dilation, 0)))
            x = F.gelu(x) + residual
        return x[:, :, -1]


class RolloutPredictor(nn.Module):
    def __init__(self, bookkeeping_width, prefix_width, visual_width, use_prefix, use_visual):
        super().__init__()
        self.use_prefix = use_prefix
        self.use_visual = use_visual
        visual_out = 24 if use_visual else 0
        if use_visual:
            self.visual_encoder = CausalVisualEncoder(visual_width)
        width = bookkeeping_width + (prefix_width if use_prefix else 0) + visual_out
        self.trunk = nn.Sequential(nn.Linear(width, 48), nn.GELU(), nn.Linear(48, 32), nn.GELU())
        self.regression = nn.Linear(32, 1)
        self.positive = nn.Linear(32, 1)

    def forward(self, bookkeeping, prefix, visual):
        parts = [bookkeeping]
        if self.use_prefix:
            parts.append(prefix)
        if self.use_visual:
            parts.append(self.visual_encoder(visual))
        hidden = self.trunk(torch.cat(parts, dim=1))
        return self.regression(hidden).squeeze(1), self.positive(hidden).squeeze(1)


VARIANTS = {
    "B0_bookkeeping": (False, False),
    "B1_bookkeeping_visual_tcn": (False, True),
    "B2_bookkeeping_prefix": (True, False),
    "B3_bookkeeping_visual_tcn_prefix": (True, True),
}


def normalization(fit):
    result = {}
    for key in ("bookkeeping", "prefix", "visual"):
        axes = (0, 1) if key == "visual" else 0
        mean = fit[key].mean(axis=axes, dtype=np.float64).astype(np.float32)
        scale = fit[key].std(axis=axes, dtype=np.float64).astype(np.float32)
        scale[scale < 1e-6] = 1.0
        result[key] = (mean, scale)
    return result


def normalize(data, stats):
    output = {**data, **{
        key: ((data[key] - stats[key][0]) / stats[key][1]).astype(np.float32)
        for key in ("bookkeeping", "prefix")
    }}
    # Compute with float32 fit-only statistics, then retain compact storage.
    output["visual"] = ((data["visual"] - stats["visual"][0]) /
                        stats["visual"][1]).astype(np.float16)
    return output


def train_variant(kind, fit, epochs=5, batch_size=1024, seed=SEED):
    # Every ablation receives the same initialization RNG state and minibatch
    # ordering. Architectures differ, but no variant gets a favorable seed.
    torch.manual_seed(seed)
    use_prefix, use_visual = VARIANTS[kind]
    model = RolloutPredictor(fit["bookkeeping"].shape[1], fit["prefix"].shape[1],
                             fit["visual"].shape[2], use_prefix, use_visual)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    positives = float(fit["positive"].sum())
    pos_weight = min(20.0, (len(fit["positive"]) - positives) / max(positives, 1.0))
    generator = torch.Generator().manual_seed(seed)
    history = []
    for epoch in range(epochs):
        model.train()
        order = torch.randperm(len(fit["advantage"]), generator=generator)
        totals = np.zeros(3, dtype=np.float64)
        for left in range(0, len(order), batch_size):
            index = order[left:left + batch_size].numpy()
            args = [torch.from_numpy(fit[key][index]).float()
                    for key in ("bookkeeping", "prefix", "visual")]
            target = torch.from_numpy(fit["advantage"][index])
            positive = torch.from_numpy(fit["positive"][index])
            optimizer.zero_grad(set_to_none=True)
            prediction, positive_logit = model(*args)
            regression_loss = F.smooth_l1_loss(prediction, target)
            bce_loss = F.binary_cross_entropy_with_logits(
                positive_logit, positive, pos_weight=torch.tensor(pos_weight)
            )
            loss = regression_loss + 0.25 * bce_loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            totals += np.asarray([float(loss.detach()), float(regression_loss.detach()),
                                  float(bce_loss.detach())]) * len(index)
        history.append({"epoch": epoch + 1, "total": totals[0] / len(order),
                        "smooth_l1": totals[1] / len(order), "weighted_bce": totals[2] / len(order)})
        print(json.dumps({"variant": kind, **history[-1]}), flush=True)
    model.eval()
    return model, history, pos_weight


def predict(model, data, batch_size=4096):
    scores, positive_scores = [], []
    with torch.no_grad():
        for left in range(0, len(data["advantage"]), batch_size):
            right = left + batch_size
            args = [torch.from_numpy(data[key][left:right]).float()
                    for key in ("bookkeeping", "prefix", "visual")]
            value, positive = model(*args)
            scores.extend(value.numpy().tolist())
            positive_scores.extend(torch.sigmoid(positive).numpy().tolist())
    return np.asarray(scores), np.asarray(positive_scores)


def average_precision(labels, scores):
    labels = np.asarray(labels, dtype=bool)
    if not labels.any():
        return None
    order = np.argsort(-np.asarray(scores), kind="stable")
    ranked = labels[order]
    precision = np.cumsum(ranked) / np.arange(1, len(ranked) + 1)
    return float(precision[ranked].sum() / ranked.sum())


def rankdata(values):
    values = np.asarray(values)
    order = np.argsort(values, kind="stable")
    ranks = np.empty(len(values), dtype=np.float64)
    left = 0
    while left < len(values):
        right = left + 1
        while right < len(values) and values[order[right]] == values[order[left]]:
            right += 1
        ranks[order[left:right]] = (left + right - 1) / 2.0
        left = right
    return ranks


def spearman(x, y):
    rx, ry = rankdata(x), rankdata(y)
    if rx.std() == 0 or ry.std() == 0:
        return 0.0
    return float(np.corrcoef(rx, ry)[0, 1])


def evaluation_metrics(labels, advantages, scores, positive_scores):
    labels = np.asarray(labels, dtype=bool)
    advantages, scores = np.asarray(advantages), np.asarray(scores)
    k = max(1, int(math.ceil(0.10 * len(scores))))
    chosen = np.argsort(-scores, kind="stable")[:k]
    recall = float(labels[chosen].sum() / labels.sum()) if labels.any() else None
    gains = np.maximum(advantages, 0.0)
    discount = 1.0 / np.log2(np.arange(2, k + 2))
    ideal = np.sort(gains)[::-1][:k]
    denominator = float((ideal * discount).sum())
    ndcg = float((gains[chosen] * discount).sum() / denominator) if denominator else None
    # Equal-count score bins avoid threshold tuning on calibration. Ties are
    # deterministically resolved by row order and are disclosed in the output.
    order = np.argsort(scores, kind="stable")
    bins = np.array_split(order, 10)
    means = [float(advantages[index].mean()) for index in bins]
    oracle_choice = np.argsort(-advantages, kind="stable")[:k]
    action_regret = float(advantages[oracle_choice].sum() - advantages[chosen].sum())
    return {
        "rows": len(scores), "positive_rows": int(labels.sum()),
        "positive_prevalence": float(labels.mean()),
        "pr_auc_regression_score": average_precision(labels, scores),
        "pr_auc_auxiliary_positive_score": average_precision(labels, positive_scores),
        "positive_recall_at_top_10pct": recall,
        "ndcg_at_top_10pct": ndcg,
        "offline_calibration_matched_top10_action_regret": {
            "definition": "sum(true advantage of oracle top-K) - sum(true advantage of score top-K)",
            "K": k, "total": action_regret, "per_selected_action": action_regret / k,
            "warning": "offline calibration diagnostic only; no threshold was tuned for deployment",
        },
        "signed_advantage_mae": float(np.abs(scores - advantages).mean()),
        "signed_advantage_smooth_l1": float(np.where(
            np.abs(scores - advantages) < 1, 0.5 * (scores - advantages) ** 2,
            np.abs(scores - advantages) - 0.5).mean()),
        "score_decile_true_mean_advantage_low_to_high": means,
        "score_decile_spearman": spearman(np.arange(10), means),
        "highest_minus_lowest_decile_mean_advantage": means[-1] - means[0],
    }


def preregistered_gate(metrics):
    b0, b3 = metrics["B0_bookkeeping"], metrics["B3_bookkeeping_visual_tcn_prefix"]
    checks = {
        "B3_PR_AUC_gt_B0": b3["pr_auc_regression_score"] > b0["pr_auc_regression_score"],
        "B3_recall_at_10pct_gt_B0": b3["positive_recall_at_top_10pct"] > b0["positive_recall_at_top_10pct"],
        "B3_NDCG_at_10pct_gt_B0": b3["ndcg_at_top_10pct"] > b0["ndcg_at_top_10pct"],
        "B3_decile_spearman_gt_0": b3["score_decile_spearman"] > 0,
        "B3_highest_decile_gt_lowest": b3["highest_minus_lowest_decile_mean_advantage"] > 0,
    }
    return {"passed": all(checks.values()), "checks": checks,
            "action": "eligible_for_closed_loop" if all(checks.values()) else "no_go_stop_before_closed_loop"}


def online_feature_row(probabilities, selected, start, balance, blank_id):
    last = selected[-1] if selected else None
    return {
        "bookkeeping": {
            "candidate_absolute_index": int(start),
            "candidate_mod_skeleton_period": int(start) % 4,
            "coverage_gap": int(start - last) if last is not None else int(start + 1),
            "frames_since_last_execute": int(start - last) if last is not None else int(start + 1),
            "bonus_token_balance": float(balance),
            "past_selected_count": len(selected),
        },
        "prefix": DATA_BUILDER.prefix_features(probabilities, selected, start, blank_id),
    }


def online_score(kind, model, stats, sequences, name, row, start):
    book = (bookkeeping_features(row) - stats["bookkeeping"][0]) / stats["bookkeeping"][1]
    prefix = (prefix_features(row) - stats["prefix"][0]) / stats["prefix"][1]
    visual = sequences.history(name, start).astype(np.float32)
    visual = (visual - stats["visual"][0]) / stats["visual"][1]
    with torch.no_grad():
        score, _ = model(
            torch.from_numpy(book[None]).float(),
            torch.from_numpy(prefix[None]).float(),
            torch.from_numpy(visual[None]).float(),
        )
    return float(score.item())


def run_predictor_policy(kind, model, threshold, stats, sequences, name,
                         probabilities, reference, vocab, blank_id):
    chron = DATA_BUILDER.CHRON
    selected, balance, forced_actions, autonomous_actions = [], 0.0, 0, 0
    book_sum = np.zeros_like(stats["bookkeeping"][0], dtype=np.float64)
    prefix_sum = np.zeros_like(stats["prefix"][0], dtype=np.float64)
    score_sum = score_sq_sum = execute_score_count = 0.0
    for start in range(len(probabilities)):
        if chron.is_skeleton(start):
            selected.append(start)
            continue
        balance = chron.accrue_token(balance)
        if not chron.token_eligible(balance):
            continue
        forced = chron.token_forced(balance)
        execute = forced
        if forced:
            forced_actions += 1
        else:
            autonomous_actions += 1
            row = online_feature_row(probabilities, selected, start, balance, blank_id)
            book_sum += (bookkeeping_features(row) - stats["bookkeeping"][0]) / stats["bookkeeping"][1]
            prefix_sum += (prefix_features(row) - stats["prefix"][0]) / stats["prefix"][1]
            score = online_score(kind, model, stats, sequences, name, row, start)
            score_sum += score; score_sq_sum += score * score
            execute = score > threshold
            execute_score_count += int(execute)
        if execute:
            selected.append(start)
            balance -= 1.0
    hypothesis, counts = chron.decode_counts(probabilities, selected, reference, vocab, blank_id)
    return {"selected": selected, "counts": counts, "hypothesis": hypothesis,
            "unspent_bonus_tokens": float(balance), "forced_actions": forced_actions,
            "autonomous_decisions": autonomous_actions,
            "visited_state_audit": {"count": autonomous_actions,
                "standardized_bookkeeping_sum": book_sum.tolist(),
                "standardized_prefix_sum": prefix_sum.tolist(),
                "score_sum": score_sum, "score_sq_sum": score_sq_sum,
                "autonomous_execute_count": int(execute_score_count)},
            "coverage": chron.coverage_metrics(selected, len(probabilities))}


def summarize_policy(rows, uniform_rows, name, threshold=None):
    chron = DATA_BUILDER.CHRON
    counts = chron.aggregate_counts([row["counts"] for row in rows])
    uniform_counts = chron.aggregate_counts([row["counts"] for row in uniform_rows])
    result = {
        "metrics": counts,
        "delta_wer_pp_vs_online_uniform": counts["wer"] - uniform_counts["wer"],
        "error_delta_vs_online_uniform": counts["error"] - uniform_counts["error"],
        "executed_windows": int(sum(len(row["selected"]) for row in rows)),
        "dense_windows": int(sum(row["dense_windows"] for row in rows)),
        "actual_window_rate": float(sum(len(row["selected"]) for row in rows) /
                                    sum(row["dense_windows"] for row in rows)),
        "max_gap": int(max(row["coverage"]["max_gap_including_endpoints"] for row in rows)),
        "unspent_tokens_total": float(sum(row["unspent_bonus_tokens"] for row in rows)),
        "forced_actions": int(sum(row.get("forced_actions", 0) for row in rows)),
        "paired_bootstrap_vs_online_uniform": chron.paired_bootstrap(
            [row["counts"] for row in rows], [row["counts"] for row in uniform_rows]
        ),
    }
    if threshold is not None:
        result["fit_only_90th_percentile_threshold"] = float(threshold)
        audits = [row["visited_state_audit"] for row in rows]
        audit_count = sum(row["count"] for row in audits)
        result["visited_state_audit"] = {
            "autonomous_rows": audit_count,
            "autonomous_execute_rate": float(sum(row["autonomous_execute_count"] for row in audits) / audit_count),
            "mean_standardized_bookkeeping": (sum(
                (np.asarray(row["standardized_bookkeeping_sum"]) for row in audits),
                np.zeros(len(audits[0]["standardized_bookkeeping_sum"]))
            ) / audit_count).tolist(),
            "mean_standardized_prefix": (sum(
                (np.asarray(row["standardized_prefix_sum"]) for row in audits),
                np.zeros(len(audits[0]["standardized_prefix_sum"]))
            ) / audit_count).tolist(),
            "mean_score": float(sum(row["score_sum"] for row in audits) / audit_count),
        }
    return result


def closed_loop_evaluation(dense_root, sequences, models, stats, thresholds, teacher_reference):
    chron = DATA_BUILDER.CHRON
    indices, shard_count = chron.completed_shard_indices(dense_root)
    split = json.loads((dense_root / "fit_calibration_split.json").read_text())
    assignments = split["assignments"]
    vocab = json.loads(chron.DEFAULT_VOCAB.read_text()); blank_id = vocab.index("<blank>")
    outputs = {"online_uniform": [], "B0_bookkeeping": [],
               "B3_bookkeeping_visual_tcn_prefix": [], "myopic_oracle": [],
               "rollout16_oracle": [], "matched_B0": [], "matched_B3": []}
    for shard_index in indices:
        results, logits, _ = DATA_BUILDER.BUILDER.validate_dense_shard(
            dense_root, shard_index, shard_count, verify_hashes=False
        )
        for name in results:
            if assignments[name] != "calibration" or name not in sequences.slices:
                continue
            probabilities = DATA_BUILDER.ORACLE.softmax_rows(np.asarray(logits[name]))
            reference = DATA_BUILDER.REPLAY.clean_phoenix_2014_trans(results[name]["gls_ref"])
            total = len(probabilities)
            uniform_selected, uniform_balance = chron.online_uniform_schedule(total)
            _, uniform_counts = chron.decode_counts(probabilities, uniform_selected, reference, vocab, blank_id)
            outputs["online_uniform"].append({"sample_id": name, "selected": uniform_selected,
                "counts": uniform_counts, "dense_windows": total, "unspent_bonus_tokens": uniform_balance,
                "coverage": chron.coverage_metrics(uniform_selected, total)})
            for kind in ("B0_bookkeeping", "B3_bookkeeping_visual_tcn_prefix"):
                value = run_predictor_policy(kind, models[kind], thresholds[kind], stats,
                                             sequences, name, probabilities, reference, vocab, blank_id)
                value.update({"sample_id": name, "dense_windows": total})
                outputs[kind].append(value)
                matched_selected = DATA_BUILDER.ORACLE.uniform_positions(total, len(value["selected"]))
                _, matched_counts = chron.decode_counts(
                    probabilities, matched_selected, reference, vocab, blank_id
                )
                matched_key = "matched_B0" if kind == "B0_bookkeeping" else "matched_B3"
                outputs[matched_key].append({"sample_id": name, "selected": matched_selected,
                    "counts": matched_counts, "dense_windows": total, "unspent_bonus_tokens": 0.0,
                    "coverage": chron.coverage_metrics(matched_selected, total)})
            for source, target in (("myopic", "myopic_oracle"), ("rollout16", "rollout16_oracle")):
                value = chron.run_policy(probabilities, reference, vocab, blank_id, source)
                outputs[target].append({"sample_id": name, "selected": value["selected"],
                    "counts": value["counts"], "dense_windows": total,
                    "unspent_bonus_tokens": value["unspent_bonus_tokens"],
                    "forced_actions": sum(int(x["forced_by_token_capacity"]) for x in value["trace"]),
                    "coverage": value["coverage"]})
    uniform = outputs["online_uniform"]
    summary = {"samples": len(uniform), "policies": {}}
    for kind, rows in outputs.items():
        if kind.startswith("matched_"):
            continue
        summary["policies"][kind] = summarize_policy(
            rows, uniform, kind, thresholds.get(kind)
        )
    # Per-sample matched-count uniform is an explicitly known-EOS diagnostic.
    for kind in ("B0_bookkeeping", "B3_bookkeeping_visual_tcn_prefix"):
        matched_key = "matched_B0" if kind == "B0_bookkeeping" else "matched_B3"
        matched = outputs[matched_key]
        candidate_counts = [row["counts"] for row in outputs[kind]]
        matched_counts = [row["counts"] for row in matched]
        candidate_metrics = chron.aggregate_counts(candidate_counts)
        matched_metrics = chron.aggregate_counts(matched_counts)
        summary["policies"][kind]["matched_actual_count_uniform"] = {
            "known_eos_diagnostic_only": True,
            "metrics": matched_metrics,
            "delta_wer_pp": candidate_metrics["wer"] - matched_metrics["wer"],
            "error_delta": candidate_metrics["error"] - matched_metrics["error"],
            "paired_bootstrap": chron.paired_bootstrap(candidate_counts, matched_counts),
        }
    rollout_gain = (summary["policies"]["online_uniform"]["metrics"]["wer"] -
                    summary["policies"]["rollout16_oracle"]["metrics"]["wer"])
    for kind in ("B0_bookkeeping", "B3_bookkeeping_visual_tcn_prefix"):
        gain = (summary["policies"]["online_uniform"]["metrics"]["wer"] -
                summary["policies"][kind]["metrics"]["wer"])
        summary["policies"][kind]["fraction_of_rollout_oracle_wer_gap_recovered"] = (
            float(gain / rollout_gain) if rollout_gain > 0 else None
        )
        visited = summary["policies"][kind]["visited_state_audit"]
        book_delta = np.asarray(visited["mean_standardized_bookkeeping"]) - np.asarray(
            teacher_reference["mean_standardized_bookkeeping"])
        prefix_delta = np.asarray(visited["mean_standardized_prefix"]) - np.asarray(
            teacher_reference["mean_standardized_prefix"])
        visited["teacher_trajectory_reference"] = teacher_reference
        visited["mean_absolute_standardized_bookkeeping_shift"] = float(np.abs(book_delta).mean())
        visited["mean_absolute_standardized_prefix_shift"] = float(np.abs(prefix_delta).mean())
    summary["audit"] = {
        "unknown_eos_no_tail_topup": True,
        "predictor_prefix_recomputed_from_own_past_selected_trajectory": True,
        "threshold_selected_from_fit_scores_only": True,
        "covariate_shift_note": "teacher trajectories generate fit states; predictor closed loop visits self-induced states",
    }
    return summary


def run(dataset_root, sequences_path, output_root, dense_root=DEFAULT_DENSE_ROOT):
    for path in (dataset_root, sequences_path, output_root, dense_root):
        reject_test_path(path)
    if output_root.exists():
        raise FileExistsError(f"refusing to overwrite {output_root}")
    temporary = output_root.with_name(f".{output_root.name}.incomplete-{os.getpid()}")
    temporary.mkdir(parents=True)
    config = {"seed": SEED, "epochs": 5, "batch_size": 1024, "optimizer": "AdamW",
              "learning_rate": 1e-3, "weight_decay": 1e-4,
              "primary_loss": "SmoothL1(signed rollout16 advantage)",
              "auxiliary_loss": "0.25 * BCE(advantage>0), pos_weight capped at 20",
              "variants": VARIANTS, "history_frames": HISTORY, "visible_offset": VISIBLE_OFFSET,
              "temporal_feature_names": TEMPORAL_FEATURE_NAMES,
              "visual_storage_dtype": "float16; converted to float32 per training/eval batch",
              "gate": "B3 strictly beats B0 on PR-AUC, top10 recall, NDCG; B3 decile Spearman>0 and highest>lowest"}
    # Frozen before any calibration row is loaded or scored.
    (temporary / "resolved_config_preregistered.json").write_text(json.dumps(config, indent=2) + "\n")
    set_determinism()
    started = time.perf_counter()
    dataset_manifest, paths = completed_paths(dataset_root)
    sequences = SequenceFeatures(sequences_path)
    rows, missing = collect(paths, sequences)
    fit, calibration = stack_rows(rows["fit"]), stack_rows(rows["calibration"])
    stats = normalization(fit)
    fit, calibration = normalize(fit, stats), normalize(calibration, stats)
    teacher_reference = {
        "rows": len(calibration["advantage"]),
        "mean_standardized_bookkeeping": calibration["bookkeeping"].mean(axis=0).tolist(),
        "mean_standardized_prefix": calibration["prefix"].mean(axis=0).tolist(),
        "positive_prevalence": float(calibration["positive"].mean()),
    }
    models, model_summaries, metrics, fit_scores = {}, {}, {}, {}
    for kind in VARIANTS:
        model, history, pos_weight = train_variant(kind, fit, seed=SEED)
        score, positive_score = predict(model, calibration)
        fit_score, _ = predict(model, fit)
        fit_scores[kind] = fit_score
        metrics[kind] = evaluation_metrics(
            calibration["positive"], calibration["advantage"], score, positive_score
        )
        models[kind] = model
        model_summaries[kind] = {"history": history, "positive_class_weight": pos_weight,
                                 "parameters": sum(x.numel() for x in model.parameters())}
    gate = preregistered_gate(metrics)
    thresholds = {kind: float(np.quantile(fit_scores[kind], 0.90)) for kind in VARIANTS}
    closed_loop = None
    if gate["passed"]:
        closed_loop = closed_loop_evaluation(
            dense_root, sequences, models, stats, thresholds, teacher_reference
        )
    for kind, model in models.items():
        torch.save({"state_dict": model.state_dict(), "normalization_fit_only": stats,
                    "variant": kind}, temporary / f"{kind}.pt")
    result = {
        "scope": "partial32 train-only chronological rollout predictor smoke; not a research result",
        "gpu_used": False, "dev_used": False, "test_used": False,
        "data": {"fit_rows": len(rows["fit"]), "calibration_rows": len(rows["calibration"]),
                 "eos_flush_rows_retained_but_flag_hidden_from_predictor": True,
                 "missing_sequence_rows_by_sample": missing},
        "objective": {"primary": "signed rollout-16 advantage SmoothL1",
                      "auxiliary": "0.25 * capped-positive-weight BCE(advantage>0)"},
        "models": model_summaries, "held_out_calibration": metrics,
        "preregistered_gate": gate,
        "fit_only_thresholds": thresholds,
        "closed_loop": closed_loop if gate["passed"] else "not run because preregistered gate failed",
        "limitations": ["32/56 non-random train shards", "smoke hyperparameters",
                        "rollout teacher uses future dense replay and reference labels",
                        "tail windows enter only after observed EOS; no EOS flag is a predictor input"],
        "runtime_seconds": time.perf_counter() - started,
    }
    (temporary / "metrics.json").write_text(json.dumps(result, indent=2) + "\n")
    manifest = {"status": "complete", "created_utc": datetime.now(timezone.utc).isoformat(),
                "dataset_manifest_sha256": sha256_file(dataset_root / "dataset_manifest.json"),
                "metrics_sha256": sha256_file(temporary / "metrics.json"),
                "preregistered_config_sha256": sha256_file(temporary / "resolved_config_preregistered.json")}
    (temporary / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    temporary.replace(output_root)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--sequences", type=Path, default=DEFAULT_SEQUENCES)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--dense-root", type=Path, default=DEFAULT_DENSE_ROOT)
    args = parser.parse_args()
    result = run(args.dataset_root.resolve(), args.sequences.resolve(), args.output_root.resolve(),
                 args.dense_root.resolve())
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
