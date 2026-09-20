#!/usr/bin/env python3
"""Full-fit source-disjoint OOF audit for robust causal predictors."""
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
from pathlib import Path

import numpy as np
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "results/phoenix-2014t_ISLR"
DATA = BASE / "p3_fulltrain_robust_continuation_fit6378_dataset_v2_49faacc3"
SEQUENCES = (BASE / "p3_fulltrain_causal_pose_hand_motion_v1_49faacc3"
             / "train_causal_pose_hand_motion_sequences.npz")
OUTPUT = BASE / "p3_fulltrain_robust_predictor_oof_v1_49faacc3"
FOLDS = 5
BINARY_SEEDS = [261026, 261027, 261028]
SIGNED_SEEDS = [261029, 261030, 261031]
EPOCHS = 20
BATCH_SIZE = 8192


def imp(name):
    path = ROOT / f"tools/{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


PARTIAL = imp("analyze_phoenix_robust_predictability_oof")


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def frozen_config():
    data_manifest = json.loads((DATA / "dataset_manifest.json").read_text())
    feature_manifest = json.loads((SEQUENCES.parent / "manifest.json").read_text())
    ref_len = int(data_manifest["counts"]["ref_len"])
    return {
        "experiment": "complete-fit source-disjoint robust predictor OOF",
        "created_before_predictor_outcomes": True,
        "scope": data_manifest["scope"],
        "rows": 2 * int(data_manifest["counts"]["blocks"]),
        "class_counts": {
            "beneficial": int(data_manifest["counts"]["positive_side_labels"]),
            "harmful": int(data_manifest["counts"]["harmful_side_labels"]),
            "neutral": (2 * int(data_manifest["counts"]["blocks"])
                        - int(data_manifest["counts"]["positive_side_labels"])
                        - int(data_manifest["counts"]["harmful_side_labels"])),
        },
        "inputs": {
            "counterfactual_dataset_manifest_sha256": sha256_file(DATA / "dataset_manifest.json"),
            "causal_feature_manifest_sha256": sha256_file(SEQUENCES.parent / "manifest.json"),
            "causal_feature_archive_sha256": feature_manifest["output"]["sha256"],
        },
        "folds": "5 deterministic source-disjoint folds using frozen robust-oof-v1 SHA256 mapping",
        "feature_sets": {
            "B0": "candidate bookkeeping plus past-paid-window decoder prefix",
            "K": "B0 plus frozen 31-step history of the same 17 P2 temporal features, summarized by last/mean/std/delta",
        },
        "objectives": {
            "binary_compatibility": "beneficial versus rest; weighted BCE; diagnostic only",
            "signed_primary": "harmful/neutral/beneficial; inverse-frequency weighted cross entropy",
            "network": "MLP 32-16; 20 fixed epochs; AdamW lr 0.002; three-seed ensemble",
            "binary_seeds": BINARY_SEEDS,
            "signed_seeds": SIGNED_SEEDS,
        },
        "primary_metrics": ["OOF PR-AUC", "recall@K", "precision@K",
                            "top-K signed robust utility", "top-K harmful actions"],
        "K": int(data_manifest["counts"]["positive_side_labels"]),
        "strong_headroom_errors": int(math.ceil(0.005 * ref_len)),
        "candidate_order": ["K", "B0"],
        "closed_loop_gate": (
            "first candidate in frozen order whose signed top-K utility is at least "
            f"ceil(0.5pp * {ref_len})={math.ceil(0.005 * ref_len)} errors"
        ),
        "causality": {
            "decision": "after offset3 arrives; same bounded candidate lookahead as partial32",
            "candidate_expensive_logits": False,
            "reference_future_total_length_or_EOS": False,
            "labels_only_use_reference_future_and_EOS": True,
        },
        "accelerator": "one stable GPU allowed; device changes runtime only, not protocol",
        "forbidden": ["calibration outcomes", "dev", "test", "threshold tuning",
                      "feature expansion", "architecture tuning", "git commit"],
    }


def preregister():
    config = frozen_config()
    path = OUTPUT / "resolved_config_preregistered.json"
    if OUTPUT.exists():
        if not path.is_file() or json.loads(path.read_text()) != config:
            raise FileExistsError(f"non-resumable or mismatched output: {OUTPUT}")
    else:
        OUTPUT.mkdir(parents=True)
        path.write_text(json.dumps(config, indent=2) + "\n")
    return config


def label_paths():
    return sorted((DATA / "shards").glob("labels-*.jsonl.gz"))


def load_rows(config):
    sequences = PARTIAL.OLD.SequenceFeatures(SEQUENCES)
    count = int(config["rows"])
    base_width = 9 + 40
    visual_width = 4 * sequences.width
    base = np.empty((count, base_width), dtype=np.float32)
    visual = np.empty((count, visual_width), dtype=np.float32)
    rewards = np.empty(count, dtype=np.int8)
    folds = np.empty(count, dtype=np.int8)
    source_indices = np.empty(count, dtype=np.int16)
    source_to_index = {}
    cursor = 0
    blocks = Counter()
    for path in label_paths():
        with gzip.open(path, "rt", encoding="utf-8") as handle:
            for line in handle:
                row = json.loads(line)
                if row["type"] != "block":
                    continue
                history = PARTIAL.visual_summary(
                    sequences.history(row["sample_id"], row["decision_arrival"])
                )
                source = row["source_video_id"]
                source_index = source_to_index.setdefault(source, len(source_to_index))
                fold = PARTIAL.source_fold(source)
                for candidate_index in (0, 2):
                    inputs = row["predictor_inputs"][candidate_index]
                    candidate_base = np.concatenate([
                        np.asarray(inputs["bookkeeping"], dtype=np.float32),
                        np.asarray(inputs["prefix"], dtype=np.float32),
                    ])
                    if candidate_base.shape != (base_width,):
                        raise ValueError("unexpected B0 feature width")
                    base[cursor] = candidate_base
                    visual[cursor] = history
                    reward = int(row["label"]["robust_reward_by_candidate"][candidate_index])
                    rewards[cursor] = reward
                    folds[cursor] = fold
                    source_indices[cursor] = source_index
                    cursor += 1
                    blocks["beneficial"] += reward > 0
                    blocks["harmful"] += reward < 0
                    blocks["neutral"] += reward == 0
    if cursor != count:
        raise RuntimeError(f"row coverage mismatch: {cursor} != {count}")
    expected = config["class_counts"]
    if any(blocks[key] != expected[key] for key in expected):
        raise RuntimeError(f"class-count mismatch: {dict(blocks)} != {expected}")
    if len(source_to_index) != config["scope"]["sources"]:
        raise RuntimeError("source coverage mismatch")
    return {"B0": base, "K": np.concatenate([base, visual], axis=1),
            "rewards": rewards.astype(np.int64), "folds": folds,
            "source_indices": source_indices, "sources": len(source_to_index)}


class BinaryMLP(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.layers = nn.Sequential(nn.Linear(width, 32), nn.GELU(), nn.Linear(32, 16),
                                    nn.GELU(), nn.Linear(16, 1))

    def forward(self, value):
        return self.layers(value).squeeze(1)


class SignedMLP(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.layers = nn.Sequential(nn.Linear(width, 32), nn.GELU(), nn.Linear(32, 16),
                                    nn.GELU(), nn.Linear(16, 3))

    def forward(self, value):
        return self.layers(value)


def seed_everything(seed, device):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)


def normalized_tensors(matrix, train_mask, eval_mask, device):
    train_values = matrix[train_mask]
    mean = train_values.mean(axis=0, dtype=np.float64).astype(np.float32)
    scale = train_values.std(axis=0, dtype=np.float64).astype(np.float32)
    scale[scale < 1e-6] = 1.0
    train = torch.from_numpy(np.asarray((train_values - mean) / scale, dtype=np.float32)).to(device)
    evaluate = torch.from_numpy(np.asarray((matrix[eval_mask] - mean) / scale, dtype=np.float32)).to(device)
    return train, evaluate


def train_binary(train_x, train_y, eval_x, seed, device):
    seed_everything(seed, device)
    model = BinaryMLP(train_x.shape[1]).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)
    positive = float(train_y.sum().item())
    criterion = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor((len(train_y) - positive) / positive, device=device)
    )
    generator = torch.Generator().manual_seed(seed)
    for _ in range(EPOCHS):
        order = torch.randperm(len(train_y), generator=generator)
        for left in range(0, len(order), BATCH_SIZE):
            index = order[left:left + BATCH_SIZE].to(device)
            loss = criterion(model(train_x[index]), train_y[index])
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
    model.eval()
    with torch.no_grad():
        output = torch.sigmoid(model(eval_x)).cpu().numpy()
    del model, optimizer
    return output


def train_signed(train_x, train_y, eval_x, seed, device):
    seed_everything(seed, device)
    model = SignedMLP(train_x.shape[1]).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)
    # The class counts are data constants; compute them on CPU because the
    # legacy CUDA bincount kernel has no deterministic implementation.
    counts = torch.bincount(train_y.cpu(), minlength=3).float().to(device)
    weights = len(train_y) / (3.0 * counts)
    criterion = nn.CrossEntropyLoss(weight=weights)
    generator = torch.Generator().manual_seed(seed)
    for _ in range(EPOCHS):
        order = torch.randperm(len(train_y), generator=generator)
        for left in range(0, len(order), BATCH_SIZE):
            index = order[left:left + BATCH_SIZE].to(device)
            loss = criterion(model(train_x[index]), train_y[index])
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
    model.eval()
    with torch.no_grad():
        probabilities = torch.softmax(model(eval_x), dim=1)
        output = (probabilities[:, 2] - probabilities[:, 0]).cpu().numpy()
    del model, optimizer
    return output


def fold_percentiles(scores, folds):
    output = np.empty_like(scores)
    for fold in range(FOLDS):
        indices = np.flatnonzero(folds == fold)
        order = np.argsort(scores[indices], kind="stable")
        output[indices[order]] = np.arange(len(indices), dtype=np.float32) / max(1, len(indices) - 1)
    return output


def selected_mask(scores, k):
    mask = np.zeros(len(scores), dtype=bool)
    mask[np.argsort(-scores, kind="stable")[:k]] = True
    return mask


def source_bootstrap_delta(rewards, source_indices, first_scores, second_scores, k, reps=10000):
    first = selected_mask(first_scores, k)
    second = selected_mask(second_scores, k)
    sources = int(source_indices.max()) + 1
    deltas = np.bincount(source_indices, weights=rewards * (first.astype(int) - second.astype(int)),
                         minlength=sources)
    rng = np.random.default_rng(261032)
    values = np.empty(reps, dtype=np.float64)
    for index in range(reps):
        chosen = rng.integers(0, sources, size=sources)
        values[index] = deltas[chosen].sum()
    return [float(np.quantile(values, 0.025)), float(np.quantile(values, 0.975))]


def run(device_name):
    config = preregister()
    device = torch.device(device_name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("requested CUDA but torch cannot access the selected stable GPU")
    torch.set_num_threads(min(8, os.cpu_count() or 1))
    torch.use_deterministic_algorithms(True)
    values = load_rows(config)
    rewards = values["rewards"]
    positive = (rewards > 0).astype(np.float32)
    signed = np.where(rewards < 0, 0, np.where(rewards > 0, 2, 1)).astype(np.int64)
    folds = values["folds"]
    scores = {objective: {name: np.full(len(rewards), np.nan, dtype=np.float32)
                          for name in ("B0", "K")}
              for objective in ("binary", "signed")}
    fold_summary = []
    started = time.perf_counter()
    for fold in range(FOLDS):
        eval_mask = folds == fold
        train_mask = ~eval_mask
        item = {"fold": fold, "train_rows": int(train_mask.sum()), "eval_rows": int(eval_mask.sum()),
                "train_beneficial": int(positive[train_mask].sum()),
                "eval_beneficial": int(positive[eval_mask].sum())}
        for name in ("B0", "K"):
            train_x, eval_x = normalized_tensors(values[name], train_mask, eval_mask, device)
            train_binary_y = torch.from_numpy(positive[train_mask]).to(device)
            train_signed_y = torch.from_numpy(signed[train_mask]).to(device)
            binary_predictions = [train_binary(train_x, train_binary_y, eval_x, seed, device)
                                  for seed in BINARY_SEEDS]
            signed_predictions = [train_signed(train_x, train_signed_y, eval_x, seed, device)
                                  for seed in SIGNED_SEEDS]
            scores["binary"][name][eval_mask] = np.mean(binary_predictions, axis=0)
            scores["signed"][name][eval_mask] = np.mean(signed_predictions, axis=0)
            del train_x, eval_x, train_binary_y, train_signed_y
            if device.type == "cuda":
                torch.cuda.empty_cache()
        fold_summary.append(item)
        print(json.dumps({**item, "elapsed_seconds": round(time.perf_counter() - started, 1)}), flush=True)
    if any(np.isnan(array).any() for objective in scores.values() for array in objective.values()):
        raise RuntimeError("OOF score coverage failure")
    metrics = {objective: {name: PARTIAL.metrics(positive, rewards, array)
                           for name, array in objective_scores.items()}
               for objective, objective_scores in scores.items()}
    sensitivity = {objective: {name: PARTIAL.metrics(
        positive, rewards, fold_percentiles(array, folds))
        for name, array in objective_scores.items()}
        for objective, objective_scores in scores.items()}
    k = int(config["K"])
    signed_ci = source_bootstrap_delta(
        rewards, values["source_indices"], scores["signed"]["K"], scores["signed"]["B0"], k
    )
    threshold = int(config["strong_headroom_errors"])
    selected_candidate = next((name for name in config["candidate_order"]
                               if metrics["signed"][name]["top_K_signed_robust_utility"] >= threshold), None)
    value = {
        "device": {"requested": device_name,
                   "torch_cuda_name": torch.cuda.get_device_name(device) if device.type == "cuda" else None},
        "rows": len(rewards), "sources": int(values["sources"]),
        "class_counts": dict(sorted(Counter(map(int, rewards)).items())),
        "folds": fold_summary,
        "metrics": metrics,
        "fold_percentile_sensitivity_not_primary": sensitivity,
        "signed_K_minus_B0_top_K_utility_source_bootstrap_95pct": signed_ci,
        "gate": {
            "strong_headroom_errors": threshold,
            "selected_closed_loop_candidate": selected_candidate,
            "passed": selected_candidate is not None,
            "action": "run_frozen_closed_loop" if selected_candidate else "stop_robust_predictor_branch",
        },
        "limitations": ["fit-only OOF intermediate ranking", "top-K is not unknown-EOS deployment control",
                        "offset3 decision has bounded candidate lookahead", "no final WER claim"],
        "elapsed_seconds": time.perf_counter() - started,
    }
    (OUTPUT / "metrics.json").write_text(json.dumps(value, indent=2) + "\n")
    np.savez_compressed(OUTPUT / "oof_scores.npz", folds=folds, rewards=rewards,
                        source_indices=values["source_indices"],
                        binary_B0=scores["binary"]["B0"], binary_K=scores["binary"]["K"],
                        signed_B0=scores["signed"]["B0"], signed_K=scores["signed"]["K"])
    return value


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    print(json.dumps(run(args.device), indent=2))
