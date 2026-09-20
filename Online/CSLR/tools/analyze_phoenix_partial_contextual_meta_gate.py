#!/usr/bin/env python3
"""Fit-only contextual stacked gate over frozen hazard4 and surprise OOF scores."""
import argparse
import hashlib
import importlib.util
import json
import os
import random
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "results/phoenix-2014t_ISLR/p3_partial32_contextual_meta_gate_fit512_oof_v2_49faacc3"
SEED = 261029
FOLD_SEED = 261022
FOLDS = 5
META_EPOCHS = 200
META_LR = 0.03
META_WEIGHT_DECAY = 1e-4
# Pre-registered, fixed throughout every outer fold: (left, center, right).
CLASS_WEIGHTS = (4.0, 1.0, 4.0)


def imp(name):
    path = ROOT / f"tools/{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


COMP = imp("analyze_phoenix_partial_predictor_complementarity")
SURPRISE = COMP.SURPRISE


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def source_fold(source):
    return int(hashlib.sha256(f"{FOLD_SEED}\0{source}".encode()).hexdigest()[:16], 16) % FOLDS


def config():
    ids = SURPRISE.frozen_ids()
    return {
        "experiment": "fit512 contextual meta-gate, source-group outer OOF",
        "created_before_contextual_meta_terminal_results": True,
        "cpu_only": True,
        "scope": {
            "samples": 512,
            "sources": 211,
            "sample_ids_sha256": hashlib.sha256("\n".join(ids).encode()).hexdigest(),
            "outer_folds": 5,
            "assignment": "sha256(261022+NUL+source_video_id) mod5",
            "calibration_evaluation": "never read",
        },
        "component_scores": {
            "hazard4": "reconstructed per-source OOF hazard probabilities using the frozen 98-d causal hazard model and source folds",
            "surprise": "reconstructed per-source OOF learned causal surprise scores using the frozen 89-d self-supervised GRU and source folds",
            "all_meta_inputs": "each source receives component predictions from models that exclude that source",
        },
        "actions": {
            "candidate_set": ["center", "hazard4_choice", "surprise_choice"],
            "label": "on each outer-training block choose the candidate-set action with lowest terminal error; ties prefer center, then hazard4 candidate order",
            "deploy": "choose the highest meta class probability only among the available candidate-set actions; ties prefer center then left then right",
        },
        "meta_features": {
            "allowed": [
                "hazard4 OOF candidate probabilities, max, and top-two margin",
                "surprise OOF candidate scores, max, and top-two margin",
                "hazard4/surprise choices and agreement category",
                "causal bookkeeping and current past-prefix summary at candidate 3",
            ],
            "forbidden": ["terminal reward", "terminal error", "reference", "future", "EOS", "full or unexecuted logits"],
        },
        "meta_model": {
            "architecture": "multinomial logistic: Linear(feature_width,3)",
            "optimizer": f"AdamW(lr={META_LR}, weight_decay={META_WEIGHT_DECAY})",
            "epochs": META_EPOCHS,
            "standardization": "mean/std from outer-training sources only",
            "class_weights_left_center_right": list(CLASS_WEIGHTS),
            "selection": "single pre-registered model and feature set; no threshold, model, feature, or weight search",
        },
        "baselines": ["fixed center", "hazard4 choice", "surprise choice", "frozen hazard4-gate AND surprise"],
        "pair_oracle": {
            "choices": ["center", "hazard4 choice", "surprise choice"],
            "deployable": False,
            "purpose": "upper bound and capture accounting only",
        },
        "pass": "aggregate reward>0, each outer-fold reward>=0, >=20 noncenter choices, and reward strictly exceeds hazard4 and fixed AND",
        "forbidden": ["GPU/CUDA", "dev", "test", "calibration/evaluation outcomes", "terminal threshold tuning", "git commit"],
    }


def preregister(root):
    if root.exists():
        raise FileExistsError(root)
    root.mkdir(parents=True)
    (root / "resolved_config_preregistered.json").write_text(json.dumps(config(), indent=2) + "\n")
    return {"status": "preregistered", "samples": 512, "sources": 211, "outer_folds": FOLDS}


def loadcfg(root):
    value = json.loads((root / "resolved_config_preregistered.json").read_text())
    if value != config():
        raise ValueError("preregistered configuration changed")


def choose(scores):
    """Choose left/center/right, with center and then left winning exact ties."""
    center, left, right = scores[:, 1], scores[:, 0], scores[:, 2]
    choice = np.ones(len(scores), np.int64)
    choice[left > center] = 0
    choice[right > np.maximum(left, center)] = 2
    return choice


def top_margin(scores):
    ordered = np.sort(scores, axis=1)
    return ordered[:, -1] - ordered[:, -2]


def candidate_label(errors, hazard_choice, surprise_choice):
    """Terminal label used only inside each outer training partition."""
    labels = np.empty(len(errors), np.int64)
    for i, (hazard, surprise) in enumerate(zip(hazard_choice, surprise_choice)):
        candidates = [1, int(hazard), int(surprise)]
        best = min(errors[i, action] for action in candidates)
        if errors[i, 1] == best:
            labels[i] = 1
        else:
            labels[i] = next(action for action in candidates if errors[i, action] == best)
    return labels


def meta_features(data, hazard_prob, surprise_scores, hazard_choice, surprise_choice):
    hazard_onehot = np.eye(3, dtype=np.float32)[hazard_choice]
    surprise_onehot = np.eye(3, dtype=np.float32)[surprise_choice]
    relation = np.zeros((len(hazard_choice), 3), np.float32)
    relation[hazard_choice == surprise_choice, 0] = 1.0
    relation[(hazard_choice != surprise_choice) & (hazard_choice != 1) & (surprise_choice != 1), 1] = 1.0
    relation[relation.sum(1) == 0, 2] = 1.0
    # The decision occurs after candidate 3 arrives; this is a causal prefix state.
    context = np.asarray([row["features"][2]["bookkeeping"] + row["features"][2]["prefix"] for row in data["rows"]], np.float32)
    derived = np.stack([hazard_prob.max(1), top_margin(hazard_prob), surprise_scores.max(1), top_margin(surprise_scores)], 1)
    return np.concatenate([hazard_prob, surprise_scores, derived, hazard_onehot, surprise_onehot, relation, context], 1).astype(np.float32)


class MetaLogistic(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.linear = nn.Linear(width, 3)

    def forward(self, value):
        return self.linear(value)


def train_meta(features, labels, train, fold):
    mean = features[train].mean(0, dtype=np.float64).astype(np.float32)
    scale = features[train].std(0, dtype=np.float64).astype(np.float32)
    scale[scale < 1e-6] = 1.0
    normalized = (features - mean) / scale
    torch.manual_seed(SEED + 100 + fold)
    model = MetaLogistic(features.shape[1])
    optimizer = torch.optim.AdamW(model.parameters(), lr=META_LR, weight_decay=META_WEIGHT_DECAY)
    value = torch.from_numpy(normalized[train]).float()
    target = torch.from_numpy(labels[train]).long()
    weights = torch.tensor(CLASS_WEIGHTS, dtype=torch.float32)
    losses = []
    for _ in range(META_EPOCHS):
        loss = F.cross_entropy(model(value), target, weight=weights)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append(float(loss))
    with torch.no_grad():
        probability = torch.softmax(model(torch.from_numpy(normalized).float()), dim=1).numpy()
    return probability, losses, {"mean": mean, "scale": scale}


def choose_available(probability, hazard_choice, surprise_choice):
    result = np.ones(len(probability), np.int64)
    for i in range(len(result)):
        available = sorted({1, int(hazard_choice[i]), int(surprise_choice[i])}, key=lambda action: (0 if action == 1 else 1, action))
        score = max(probability[i, action] for action in available)
        result[i] = next(action for action in available if probability[i, action] == score)
    return result


def fixed_and(hazard_prob, threshold, surprise_choice):
    opened = hazard_prob.max(1) >= threshold
    return np.where(opened & (surprise_choice != 1), surprise_choice, 1).astype(np.int64)


def choice_metrics(errors, choice, folds):
    picked = errors[np.arange(len(errors)), choice]
    center = errors[:, 1]
    minimum = errors.min(1)
    reward = center - picked
    side = choice != 1
    return {
        "choice_counts": {str(k): int(v) for k, v in zip(*np.unique(choice, return_counts=True))},
        "noncenter_choices": int(side.sum()),
        "positive_noncenter": int(np.sum(side & (reward > 0))),
        "harmful_noncenter": int(np.sum(side & (reward < 0))),
        "neutral_noncenter": int(np.sum(side & (reward == 0))),
        "cumulative_terminal_reward": int(reward.sum()),
        "total_terminal_regret": int(np.sum(picked - minimum)),
        "fixed_center_regret": int(np.sum(center - minimum)),
        "fold_reward": {str(fold): int(reward[folds == fold].sum()) for fold in range(FOLDS)},
        "reward_array": reward,
    }


def clean(report):
    return {key: value for key, value in report.items() if key != "reward_array"}


def pair_oracle(errors, hazard_choice, surprise_choice):
    return candidate_label(errors, hazard_choice, surprise_choice)


def oracle_capture(errors, oracle_choice, meta_choice):
    center = errors[:, 1]
    oracle_reward = center - errors[np.arange(len(errors)), oracle_choice]
    meta_reward = center - errors[np.arange(len(errors)), meta_choice]
    opportunity = oracle_reward > 0
    total = int(oracle_reward.sum())
    captured = int(meta_reward[opportunity].sum())
    return {
        "oracle_noncenter_opportunities": int(opportunity.sum()),
        "exact_oracle_action_on_opportunities": int(np.sum(opportunity & (oracle_choice == meta_choice))),
        "reward_captured_on_oracle_opportunities": captured,
        "oracle_reward_capture_fraction": float(captured / total) if total else None,
        "overall_meta_over_oracle_reward_fraction": float(meta_reward.sum() / total) if total else None,
    }


def reconstruct_components(data, folds, root):
    cache = cache_root(root)
    score_paths = [cache / f"surprise_fold{fold}.npz" for fold in range(FOLDS)] + [cache / f"hazard_fold{fold}.npz" for fold in range(FOLDS)]
    if not all(path.exists() for path in score_paths):
        raise RuntimeError("component cache incomplete; run the preregistered staged component commands before oof")
    surprise_scores = np.full((len(folds), 3), np.nan, np.float32)
    hazard_prob = np.full((len(folds), 3), np.nan, np.float32)
    thresholds = np.full(len(folds), np.nan, np.float32)
    reports = {}
    for fold in range(FOLDS):
        surprise = np.load(cache / f"surprise_fold{fold}.npz")
        hazard = np.load(cache / f"hazard_fold{fold}.npz")
        held = folds == fold
        surprise_scores[held] = surprise["scores"]
        hazard_prob[held] = hazard["probability"]
        thresholds[held] = float(hazard["threshold"])
        reports[str(fold)] = json.loads((cache / f"fold{fold}_report.json").read_text())
    if np.isnan(surprise_scores).any() or np.isnan(hazard_prob).any() or np.isnan(thresholds).any():
        raise RuntimeError("incomplete source-OOF component predictions")
    return surprise_scores, hazard_prob, thresholds, reports


def cache_root(root):
    # v2 is a clean deterministic restart after an interrupted pre-cache attempt.
    return root / "component_oof_cache_v2"


def cache_data(root):
    cache = cache_root(root)
    cache.mkdir(exist_ok=True)
    path = cache / "data.pt"
    if path.exists():
        raise FileExistsError(path)
    data = SURPRISE.build_data()
    torch.save(data, path)
    return {"status": "data_cached", "blocks": len(data["errors"]), "samples": len(data["slices"])}


def load_cached_data(root):
    path = cache_root(root) / "data.pt"
    if not path.exists():
        raise FileNotFoundError("cache-data must run before staged component reconstruction")
    return torch.load(path)


def surprise_init(root, fold):
    data = load_cached_data(root)
    state_path = cache_root(root) / f"surprise_state_fold{fold}.pt"
    if state_path.exists():
        raise FileExistsError(state_path)
    stats = SURPRISE.fold_stats(data, fold)
    torch.manual_seed(SURPRISE.SEED + fold)
    model = SURPRISE.Predictor()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    torch.save({"epoch": 0, "stats": stats, "model": model.state_dict(), "optimizer": optimizer.state_dict(), "losses": []}, state_path)
    return {"status": "surprise_initialized", "fold": fold}


def surprise_epoch(root, fold):
    data = load_cached_data(root)
    state_path = cache_root(root) / f"surprise_state_fold{fold}.pt"
    state = torch.load(state_path)
    epoch = int(state["epoch"])
    if epoch >= SURPRISE.EPOCHS:
        raise ValueError("all surprise epochs already complete")
    model = SURPRISE.Predictor()
    model.load_state_dict(state["model"])
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    optimizer.load_state_dict(state["optimizer"])
    names = [name for name in SURPRISE.frozen_ids() if SURPRISE.source_fold(data["source_by_sample"][name]) != fold]
    generator = np.random.default_rng(SURPRISE.SEED + 100 + fold)
    for _ in range(epoch):
        generator.permutation(len(names))
    order = generator.permutation(len(names))
    total = 0.0
    count = 0
    stats = state["stats"]
    for pos in order:
        sequence = (data["features"][data["slices"][names[int(pos)]]] - stats[0]) / stats[1]
        if len(sequence) < 2:
            continue
        value = torch.from_numpy(sequence[:-1][None]).float()
        target = torch.from_numpy(sequence[1:][None]).float()
        loss = F.smooth_l1_loss(model(value), target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total += float(loss) * (len(sequence) - 1)
        count += len(sequence) - 1
    state.update({"epoch": epoch + 1, "model": model.state_dict(), "optimizer": optimizer.state_dict(), "losses": state["losses"] + [total / count]})
    torch.save(state, state_path)
    return {"status": "surprise_epoch_complete", "fold": fold, "epoch": epoch + 1, "loss": total / count}


def surprise_score(root, fold):
    data = load_cached_data(root)
    cache = cache_root(root)
    state = torch.load(cache / f"surprise_state_fold{fold}.pt")
    if int(state["epoch"]) != SURPRISE.EPOCHS:
        raise ValueError("surprise score requested before all fixed epochs complete")
    model = SURPRISE.Predictor()
    model.load_state_dict(state["model"])
    model.eval()
    residual = SURPRISE.train_residual_stats(model, data, fold, state["stats"])
    mask, scores, cost = SURPRISE.held_scores(model, data, fold, state["stats"], residual)
    np.savez_compressed(cache / f"surprise_fold{fold}.npz", scores=scores["learned"])
    report_path = cache / f"fold{fold}_report.json"
    report = json.loads(report_path.read_text()) if report_path.exists() else {}
    report.update({"surprise_last_loss": state["losses"][-1], "surprise_controller_ms_per_candidate": cost})
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    return {"status": "surprise_scores_complete", "fold": fold, "residual_train_mean": residual[0]}


def hazard_fold(root, fold):
    data = load_cached_data(root)
    folds = np.asarray([source_fold(source) for source in data["source"]], np.int64)
    value, labels = COMP.hazard_arrays(data)
    probability, threshold, history, scale = COMP.train_hazard(value, labels, folds != fold, COMP.SEED + 200 + fold)
    np.savez_compressed(cache_root(root) / f"hazard_fold{fold}.npz", probability=probability[folds == fold], threshold=np.asarray(threshold, np.float32))
    report_path = cache_root(root) / f"fold{fold}_report.json"
    report = json.loads(report_path.read_text()) if report_path.exists() else {}
    report.update({"hazard_last_loss": history[-1], "hazard_threshold": threshold, "hazard_scale": scale})
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    return {"status": "hazard_scores_complete", "fold": fold, "threshold": threshold}


def _old_reconstruct_components(data, folds):
    hazard_x, hazard_y = COMP.hazard_arrays(data)
    surprise_scores = np.full((len(folds), 3), np.nan, np.float32)
    hazard_prob = np.full((len(folds), 3), np.nan, np.float32)
    thresholds = np.full(len(folds), np.nan, np.float32)
    reports = {}
    for fold in range(FOLDS):
        train, held = folds != fold, folds == fold
        mask, scores, surprise_history, surprise_cost = COMP.train_surprise_fold(data, fold)
        surprise_scores[mask] = scores
        probability, threshold, hazard_history, hazard_scale = COMP.train_hazard(hazard_x, hazard_y, train, SEED + 200 + fold)
        hazard_prob[held] = probability[held]
        thresholds[held] = threshold
        reports[str(fold)] = {
            "surprise_last_loss": surprise_history[-1],
            "surprise_controller_ms_per_candidate": surprise_cost,
            "hazard_last_loss": hazard_history[-1],
            "hazard_threshold": threshold,
            "hazard_scale": hazard_scale,
        }
    if np.isnan(surprise_scores).any() or np.isnan(hazard_prob).any() or np.isnan(thresholds).any():
        raise RuntimeError("incomplete source-OOF component predictions")
    return surprise_scores, hazard_prob, thresholds, reports


def run(root):
    loadcfg(root)
    (root / "oof_started.marker").write_text(datetime.now(timezone.utc).isoformat() + "\n")
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.set_num_threads(min(8, os.cpu_count() or 1))
    torch.use_deterministic_algorithms(True)
    started = time.perf_counter()
    data = SURPRISE.build_data()
    folds = np.asarray([source_fold(source) for source in data["source"]], np.int64)
    surprise_scores, hazard_prob, thresholds, component_folds = reconstruct_components(data, folds, root)
    hazard_choice = choose(hazard_prob)
    surprise_choice = choose(surprise_scores)
    features = meta_features(data, hazard_prob, surprise_scores, hazard_choice, surprise_choice)
    labels = candidate_label(data["errors"], hazard_choice, surprise_choice)
    probability = np.full((len(folds), 3), np.nan, np.float32)
    meta_fold_reports = {}
    for fold in range(FOLDS):
        train, held = folds != fold, folds == fold
        predicted, losses, stats = train_meta(features, labels, train, fold)
        probability[held] = predicted[held]
        meta_fold_reports[str(fold)] = {
            "train_blocks": int(train.sum()),
            "held_blocks": int(held.sum()),
            "train_label_counts": {str(k): int(v) for k, v in zip(*np.unique(labels[train], return_counts=True))},
            "last_loss": losses[-1],
            "feature_width": int(features.shape[1]),
            "standardization": "outer-train only",
        }
        print(json.dumps({"fold_complete": fold, "meta_last_loss": losses[-1], "seconds": time.perf_counter() - started}), flush=True)
    if np.isnan(probability).any():
        raise RuntimeError("incomplete outer-OOF meta predictions")
    meta_choice = choose_available(probability, hazard_choice, surprise_choice)
    and_choice = fixed_and(hazard_prob, thresholds, surprise_choice)
    center_choice = np.ones(len(folds), np.int64)
    oracle_choice = pair_oracle(data["errors"], hazard_choice, surprise_choice)
    reports = {
        "fixed_center": clean(choice_metrics(data["errors"], center_choice, folds)),
        "hazard4": clean(choice_metrics(data["errors"], hazard_choice, folds)),
        "surprise": clean(choice_metrics(data["errors"], surprise_choice, folds)),
        "fixed_AND": clean(choice_metrics(data["errors"], and_choice, folds)),
        "meta_gate": clean(choice_metrics(data["errors"], meta_choice, folds)),
        "pair_oracle_non_deployable": clean(choice_metrics(data["errors"], oracle_choice, folds)),
    }
    disagreement = hazard_choice != surprise_choice
    disagreement_accuracy = float(np.mean(meta_choice[disagreement] == labels[disagreement])) if disagreement.any() else None
    meta = reports["meta_gate"]
    checks = {
        "aggregate_reward_gt_0": meta["cumulative_terminal_reward"] > 0,
        "every_fold_reward_ge_0": all(value >= 0 for value in meta["fold_reward"].values()),
        "at_least_20_noncenter_choices": meta["noncenter_choices"] >= 20,
        "reward_strictly_gt_hazard4": meta["cumulative_terminal_reward"] > reports["hazard4"]["cumulative_terminal_reward"],
        "reward_strictly_gt_fixed_AND": meta["cumulative_terminal_reward"] > reports["fixed_AND"]["cumulative_terminal_reward"],
    }
    result = {
        "scope": "frozen512 fit-only contextual meta-gate, source-group nested OOF",
        "gpu_used": False,
        "dev_used": False,
        "test_used": False,
        "calibration_or_evaluation_outcomes_read": False,
        "data": {"samples": len(data["slices"]), "sources": len(set(data["source"])), "blocks": len(data["errors"]), "informative_blocks": int(data["informative"].sum()), "runtime_seconds": time.perf_counter() - started},
        "component_oof": {"hazard4_choice_counts": {str(k): int(v) for k, v in zip(*np.unique(hazard_choice, return_counts=True))}, "surprise_choice_counts": {str(k): int(v) for k, v in zip(*np.unique(surprise_choice, return_counts=True))}, "exact_choice_agreement": float(np.mean(hazard_choice == surprise_choice)), "disagreement_blocks": int(disagreement.sum()), "fold_reports": component_folds},
        "meta": {"architecture": "multinomial logistic", "feature_width": int(features.shape[1]), "class_weights_left_center_right": list(CLASS_WEIGHTS), "outer_fold_reports": meta_fold_reports, "label_counts_all_blocks": {str(k): int(v) for k, v in zip(*np.unique(labels, return_counts=True))}, "disagreement_subset_exact_action_accuracy": disagreement_accuracy, "selection_frequency": reports["meta_gate"]["choice_counts"]},
        "methods": reports,
        "pair_oracle_capture": oracle_capture(data["errors"], oracle_choice, meta_choice),
        "gate": {"passed": all(checks.values()), "checks": checks},
        "decision": "eligible for a separate calibration/on-policy audit" if all(checks.values()) else "stop contextual meta-gate; do not read calibration",
    }
    (root / "metrics.json").write_text(json.dumps(result, indent=2) + "\n")
    (root / "manifest.json").write_text(json.dumps({"status": "fit_oof_complete", "created_utc": datetime.now(timezone.utc).isoformat(), "gate_passed": all(checks.values()), "calibration_outcomes_read": False, "evaluation_outcomes_read": False, "metrics_sha256": sha(root / "metrics.json")}, indent=2) + "\n")
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("preregister", "cache-data", "surprise-init", "surprise-epoch", "surprise-score", "hazard-fold", "oof"))
    parser.add_argument("--output-root", type=Path, default=OUTPUT)
    parser.add_argument("--fold", type=int, choices=range(FOLDS))
    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    torch.set_num_threads(2)
    root = args.output_root.resolve()
    if args.mode == "preregister":
        result = preregister(root)
    elif args.mode == "cache-data":
        loadcfg(root); result = cache_data(root)
    elif args.mode == "surprise-init":
        if args.fold is None: parser.error("--fold is required for surprise-init")
        loadcfg(root); result = surprise_init(root, args.fold)
    elif args.mode == "surprise-epoch":
        if args.fold is None: parser.error("--fold is required for surprise-epoch")
        loadcfg(root); result = surprise_epoch(root, args.fold)
    elif args.mode == "surprise-score":
        if args.fold is None: parser.error("--fold is required for surprise-score")
        loadcfg(root); result = surprise_score(root, args.fold)
    elif args.mode == "hazard-fold":
        if args.fold is None: parser.error("--fold is required for hazard-fold")
        loadcfg(root); result = hazard_fold(root, args.fold)
    else:
        result = run(root)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
