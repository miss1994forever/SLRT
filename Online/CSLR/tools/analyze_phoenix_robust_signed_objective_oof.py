#!/usr/bin/env python3
"""OOF audit of a signed three-class objective for robust side utility."""
import importlib.util
import json
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
OUTPUT = BASE / "p3_partial32_robust_signed_objective_oof_v1_49faacc3"
BINARY_RESULT = BASE / "p3_partial32_robust_predictability_oof_v2_49faacc3"
SEEDS = [261029, 261030, 261031]


def imp(name):
    path = ROOT / f"tools/{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


OOF = imp("analyze_phoenix_robust_predictability_oof")


def signed_classes(rewards):
    rewards = np.asarray(rewards)
    return np.where(rewards < 0, 0, np.where(rewards > 0, 2, 1)).astype(np.int64)


class SignedMLP(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.layers = nn.Sequential(nn.Linear(width, 32), nn.GELU(), nn.Linear(32, 16),
                                    nn.GELU(), nn.Linear(16, 3))

    def forward(self, x):
        return self.layers(x)


def train_predict(x_train, target_train, x_eval, seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    mean = x_train.mean(axis=0, dtype=np.float64).astype(np.float32)
    scale = x_train.std(axis=0, dtype=np.float64).astype(np.float32)
    scale[scale < 1e-6] = 1.0
    train = np.asarray((x_train - mean) / scale, dtype=np.float32)
    evaluate = np.asarray((x_eval - mean) / scale, dtype=np.float32)
    counts = np.bincount(target_train, minlength=3).astype(np.float32)
    weights = len(target_train) / (3.0 * counts)
    model = SignedMLP(train.shape[1])
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss(weight=torch.from_numpy(weights))
    generator = torch.Generator().manual_seed(seed)
    for _ in range(20):
        order = torch.randperm(len(train), generator=generator)
        for left in range(0, len(train), 4096):
            indices = order[left:left + 4096].numpy()
            logits = model(torch.from_numpy(train[indices]))
            loss = criterion(logits, torch.from_numpy(target_train[indices]))
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
    model.eval()
    with torch.no_grad():
        probabilities = torch.softmax(model(torch.from_numpy(evaluate)), dim=1).numpy()
    # Expected signed class value: benefit probability minus harm probability.
    return probabilities[:, 2] - probabilities[:, 0]


def best_prefix(rewards, scores):
    order = np.argsort(-scores, kind="stable")
    cumulative = np.cumsum(rewards[order])
    index = int(np.argmax(cumulative))
    return {"signed_utility": int(cumulative[index]), "K": index + 1}


def preregister():
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    binary = json.loads((BINARY_RESULT / "metrics.json").read_text())
    config = {
        "experiment": "signed three-class robust-utility OOF objective audit",
        "created_before_signed_objective_outcomes": True,
        "motivation": "binary positive-vs-rest loss conflates harmful and neutral side actions",
        "same_as_binary_audit": ["fit512 rows", "source-disjoint folds", "B1 causal inputs",
                                 "network hidden widths", "20 epochs", "no threshold tuning"],
        "changed_only": {
            "target": ["harmful", "neutral", "beneficial"],
            "loss": "inverse-frequency weighted three-class cross entropy",
            "ranking_score": "P(beneficial)-P(harmful)",
            "ensemble_seeds": SEEDS,
        },
        "frozen_binary_baseline_top_K_signed_utility": binary["metrics"]["B1_plus_causal_visual"]["top_K_signed_robust_utility"],
        "gates": {
            "minimal": "top-K signed utility > binary baseline and > 0",
            "strong_headroom": "top-K signed utility >= 21 errors (ceil(0.5pp * 4041 reference tokens))",
        },
        "interpretation": "fit-only intermediate ranking; neither gate is final WER evidence",
        "forbidden": ["dev", "test", "threshold tuning", "GPU", "git commit"],
    }
    OUTPUT.mkdir(parents=True)
    (OUTPUT / "resolved_config_preregistered.json").write_text(json.dumps(config, indent=2) + "\n")
    return config


def run():
    config = preregister()
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    torch.set_num_threads(min(8, os.cpu_count() or 1))
    torch.use_deterministic_algorithms(True)
    rows = OOF.load_rows()
    rewards = np.asarray([row["reward"] for row in rows], dtype=np.int64)
    labels = (rewards > 0).astype(np.float32)
    target = signed_classes(rewards)
    folds = np.asarray([OOF.source_fold(row["source"]) for row in rows], dtype=np.int64)
    base = np.stack([row["base"] for row in rows]).astype(np.float32)
    visual = np.stack([row["visual"] for row in rows]).astype(np.float32)
    matrix = np.concatenate([base, visual], axis=1)
    scores = np.full(len(rows), np.nan, dtype=np.float32)
    fold_summary = []
    started = time.perf_counter()
    for fold in range(OOF.FOLDS):
        evaluate = folds == fold
        train = ~evaluate
        predictions = [train_predict(matrix[train], target[train], matrix[evaluate], seed) for seed in SEEDS]
        scores[evaluate] = np.mean(predictions, axis=0)
        item = {"fold": fold, "train_class_counts": np.bincount(target[train], minlength=3).tolist(),
                "eval_class_counts": np.bincount(target[evaluate], minlength=3).tolist()}
        fold_summary.append(item)
        print(json.dumps({**item, "elapsed_seconds": round(time.perf_counter() - started, 1)}), flush=True)
    if np.isnan(scores).any():
        raise RuntimeError("OOF score coverage failure")
    signed_metrics = OOF.metrics(labels, rewards, scores)
    binary_metrics = json.loads((BINARY_RESULT / "metrics.json").read_text())["metrics"]["B1_plus_causal_visual"]
    checks = {
        "signed_top_K_gt_binary": signed_metrics["top_K_signed_robust_utility"] > binary_metrics["top_K_signed_robust_utility"],
        "signed_top_K_gt_0": signed_metrics["top_K_signed_robust_utility"] > 0,
        "signed_top_K_ge_21": signed_metrics["top_K_signed_robust_utility"] >= 21,
    }
    value = {
        "rows": len(rows),
        "class_order": ["harmful", "neutral", "beneficial"],
        "class_counts": np.bincount(target, minlength=3).tolist(),
        "folds": fold_summary,
        "binary_B1_frozen": binary_metrics,
        "signed_three_class": signed_metrics,
        "posthoc_threshold_diagnostic_not_a_gate": best_prefix(rewards, scores),
        "gate": {
            "minimal_passed": checks["signed_top_K_gt_binary"] and checks["signed_top_K_gt_0"],
            "strong_headroom_passed": checks["signed_top_K_ge_21"],
            "checks": checks,
            "action": "closed_loop_only_if_strong_headroom" if checks["signed_top_K_ge_21"] else "stop_before_closed_loop",
        },
        "limitations": ["fit-only OOF", "partial32 replay", "bounded offset3 decision lookahead",
                        "top-K is an intermediate ranking diagnostic, not deployable unknown-EOS control"],
        "elapsed_seconds": time.perf_counter() - started,
    }
    (OUTPUT / "metrics.json").write_text(json.dumps(value, indent=2) + "\n")
    np.savez_compressed(OUTPUT / "oof_scores.npz", folds=folds, labels=labels,
                        rewards=rewards, signed_score=scores)
    return value


if __name__ == "__main__":
    print(json.dumps(run(), indent=2))
