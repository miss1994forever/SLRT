#!/usr/bin/env python3
"""Source-disjoint OOF predictability smoke for robust side-window utility."""
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
from pathlib import Path

import numpy as np
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "results/phoenix-2014t_ISLR"
DATA = BASE / "p3_partial32_robust_continuation_fit512_dataset_v1_49faacc3"
OUTPUT = BASE / "p3_partial32_robust_predictability_oof_v2_49faacc3"
SEQUENCES = Path("/tmp/phoenix_p2_train_center_sequences_v1.npz")
SEED = 261026
FOLDS = 5
ENSEMBLE_SEEDS = [261026, 261027, 261028]


def imp(name):
    path = ROOT / f"tools/{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


OLD = imp("analyze_phoenix_partial_rollout_predictor")


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def source_fold(source):
    value = hashlib.sha256(f"robust-oof-v1\0{source}".encode()).digest()
    return int.from_bytes(value[:8], "big") % FOLDS


def visual_summary(history):
    value = np.asarray(history, dtype=np.float32)
    return np.concatenate([
        value[-1], value.mean(axis=0), value.std(axis=0), value[-1] - value[0]
    ]).astype(np.float32)


def preregister():
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    manifest = json.loads((DATA / "dataset_manifest.json").read_text())
    config = {
        "experiment": "source-disjoint OOF robust-side predictability smoke",
        "created_before_predictor_outcomes": True,
        "dataset_sha256": manifest["data_sha256"],
        "scope": {"samples": manifest["counts"]["samples"], "blocks": manifest["counts"]["blocks"]},
        "rows": "two side candidates per complete block; center is not a prediction row",
        "target": "robust_reward > 0 under both fixed-center and fixed-late future continuations",
        "folds": "5 deterministic source-video-disjoint folds; SHA256 robust-oof-v1 mapping",
        "models": {
            "B0_bookkeeping_prefix": ["candidate bookkeeping", "past-paid-window decoder prefix"],
            "B1_plus_causal_visual": ["B0", "31-step cheap pose/hand/motion history at offset3 decision arrival",
                                      "last, mean, std, last-minus-first summary"],
            "network": "MLP 32-16, weighted BCE, fixed 20 epochs, AdamW lr=0.002",
            "ensemble_seeds": ENSEMBLE_SEEDS,
        },
        "causality": {
            "decision": "after offset3 arrives; bounded candidate lookahead, unknown EOS",
            "candidate_expensive_logits": False,
            "reference_future_total_length_or_EOS": False,
            "labels_only_use_reference_and_future": True,
        },
        "primary_metrics": ["OOF PR-AUC", "positive recall@K", "precision@K", "top-K signed robust utility"],
        "K": "number of positive side labels, frozen from dataset count rather than predictor scores",
        "gate": ["B1 PR-AUC > B0", "B1 recall@K > B0", "B1 top-K signed utility > 0"],
        "interpretation": "intermediate predictability only; final success still requires closed-loop WER",
        "forbidden": ["dev", "test", "fold-wise early stopping", "threshold tuning", "GPU", "git commit"],
    }
    OUTPUT.mkdir(parents=True)
    (OUTPUT / "resolved_config_preregistered.json").write_text(json.dumps(config, indent=2) + "\n")
    return config


def load_rows():
    sequences = OLD.SequenceFeatures(SEQUENCES)
    values = []
    path = DATA / "robust-block-labels.jsonl.gz"
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            if row["type"] != "block":
                continue
            history = visual_summary(sequences.history(row["sample_id"], row["decision_arrival"]))
            for index in (0, 2):
                inputs = row["predictor_inputs"][index]
                # The dataset builder already serialized the audited numeric
                # feature vectors, rather than their original JSON mappings.
                base = np.concatenate([
                    np.asarray(inputs["bookkeeping"], dtype=np.float32),
                    np.asarray(inputs["prefix"], dtype=np.float32),
                ]).astype(np.float32)
                reward = int(row["label"]["robust_reward_by_candidate"][index])
                values.append({
                    "source": row["source_video_id"],
                    "sample": row["sample_id"],
                    "block_start": row["block_start"],
                    "side": "left" if index == 0 else "right",
                    "base": base,
                    "visual": history,
                    "reward": reward,
                    "positive": float(reward > 0),
                })
    return values


class MLP(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.layers = nn.Sequential(nn.Linear(width, 32), nn.GELU(), nn.Linear(32, 16),
                                    nn.GELU(), nn.Linear(16, 1))

    def forward(self, x):
        return self.layers(x).squeeze(1)


def train_predict(x_train, y_train, x_eval, seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    mean = x_train.mean(axis=0, dtype=np.float64).astype(np.float32)
    scale = x_train.std(axis=0, dtype=np.float64).astype(np.float32)
    scale[scale < 1e-6] = 1.0
    train = np.asarray((x_train - mean) / scale, dtype=np.float32)
    evaluate = np.asarray((x_eval - mean) / scale, dtype=np.float32)
    model = MLP(train.shape[1])
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)
    positive = float(y_train.sum())
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor((len(y_train) - positive) / positive))
    generator = torch.Generator().manual_seed(seed)
    for _ in range(20):
        order = torch.randperm(len(train), generator=generator)
        for left in range(0, len(train), 4096):
            indices = order[left:left + 4096].numpy()
            logits = model(torch.from_numpy(train[indices]))
            loss = criterion(logits, torch.from_numpy(y_train[indices]))
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
    model.eval()
    with torch.no_grad():
        return torch.sigmoid(model(torch.from_numpy(evaluate))).numpy()


def average_precision(labels, scores):
    order = np.argsort(-scores, kind="stable")
    sorted_labels = labels[order]
    positive = sorted_labels.sum()
    if positive == 0:
        return float("nan")
    precision = np.cumsum(sorted_labels) / np.arange(1, len(labels) + 1)
    return float((precision * sorted_labels).sum() / positive)


def metrics(labels, rewards, scores):
    k = int(labels.sum())
    order = np.argsort(-scores, kind="stable")
    chosen = order[:k]
    return {
        "pr_auc": average_precision(labels, scores),
        "positive_prevalence": float(labels.mean()),
        "K": k,
        "positive_recall_at_K": float(labels[chosen].sum() / labels.sum()),
        "precision_at_K": float(labels[chosen].mean()),
        "top_K_signed_robust_utility": int(rewards[chosen].sum()),
        "top_K_positive_utility_captured": int(np.maximum(rewards[chosen], 0).sum()),
        "top_K_harmful_actions": int((rewards[chosen] < 0).sum()),
        "all_positive_utility": int(np.maximum(rewards, 0).sum()),
    }


def run():
    preregister()
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    torch.set_num_threads(min(8, os.cpu_count() or 1))
    torch.use_deterministic_algorithms(True)
    rows = load_rows()
    labels = np.asarray([row["positive"] for row in rows], dtype=np.float32)
    rewards = np.asarray([row["reward"] for row in rows], dtype=np.int64)
    folds = np.asarray([source_fold(row["source"]) for row in rows], dtype=np.int64)
    base = np.stack([row["base"] for row in rows]).astype(np.float32)
    visual = np.stack([row["visual"] for row in rows]).astype(np.float32)
    matrices = {
        "B0_bookkeeping_prefix": base,
        "B1_plus_causal_visual": np.concatenate([base, visual], axis=1),
    }
    scores = {name: np.full(len(rows), np.nan, dtype=np.float32) for name in matrices}
    fold_summary = []
    started = time.perf_counter()
    for fold in range(FOLDS):
        evaluate = folds == fold
        train = ~evaluate
        if labels[train].sum() == 0 or labels[evaluate].sum() == 0:
            raise RuntimeError(f"fold {fold} lacks positive training or evaluation rows")
        item = {"fold": fold, "train_rows": int(train.sum()), "eval_rows": int(evaluate.sum()),
                "train_positive": int(labels[train].sum()), "eval_positive": int(labels[evaluate].sum()),
                "eval_sources": len({rows[i]["source"] for i in np.flatnonzero(evaluate)})}
        for name, matrix in matrices.items():
            predictions = [train_predict(matrix[train], labels[train], matrix[evaluate], seed)
                           for seed in ENSEMBLE_SEEDS]
            scores[name][evaluate] = np.mean(predictions, axis=0)
        fold_summary.append(item)
        print(json.dumps({**item, "elapsed_seconds": round(time.perf_counter() - started, 1)}), flush=True)
    if any(np.isnan(value).any() for value in scores.values()):
        raise RuntimeError("OOF score coverage failure")
    result_metrics = {name: metrics(labels, rewards, value) for name, value in scores.items()}
    b0 = result_metrics["B0_bookkeeping_prefix"]
    b1 = result_metrics["B1_plus_causal_visual"]
    checks = {
        "B1_pr_auc_gt_B0": b1["pr_auc"] > b0["pr_auc"],
        "B1_recall_at_K_gt_B0": b1["positive_recall_at_K"] > b0["positive_recall_at_K"],
        "B1_top_K_signed_utility_gt_0": b1["top_K_signed_robust_utility"] > 0,
    }
    value = {
        "rows": len(rows),
        "sources": len({row["source"] for row in rows}),
        "reward_histogram": dict(sorted(Counter(map(int, rewards)).items())),
        "folds": fold_summary,
        "metrics": result_metrics,
        "gate": {"passed": all(checks.values()), "checks": checks,
                 "action": "eligible_for_closed_loop_smoke" if all(checks.values()) else "stop_before_closed_loop"},
        "limitations": ["fit-only OOF intermediate metric", "partial32-derived replay",
                        "offset3 decision has bounded candidate lookahead", "no final WER claim"],
        "elapsed_seconds": time.perf_counter() - started,
    }
    (OUTPUT / "metrics.json").write_text(json.dumps(value, indent=2) + "\n")
    np.savez_compressed(OUTPUT / "oof_scores.npz", folds=folds, labels=labels, rewards=rewards,
                        B0=scores["B0_bookkeeping_prefix"], B1=scores["B1_plus_causal_visual"])
    return value


if __name__ == "__main__":
    print(json.dumps(run(), indent=2))
