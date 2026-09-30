#!/usr/bin/env python3
"""Post-direct-OOF H4 opportunity abstention plus conditional schedule ranking."""
import argparse
import hashlib
import importlib.util
import json
import os
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "results/phoenix-2014t_ISLR"
PARENT = BASE / "p5_h4_schedule_ranker_oof_v1_49faacc3"
OUTPUT = BASE / "p5_h4_abstention_ranker_oof_v1_49faacc3"
CONFIG = ROOT / "configs/p5_h4_abstention_ranker_oof_v1_preregistered.json"
DATASET = PARENT / "h4_ranker_dataset.npz"
CENTER = 40
ALPHA = 100.0
BOOT_SEED = 261143


def imp(name):
    path = ROOT / f"tools/{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


DIRECT = imp("analyze_phoenix_p5_h4_schedule_ranker_oof")


def utcnow():
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def prepare():
    config = json.loads(CONFIG.read_text())
    assets = {
        "ranker_dataset_sha256": DATASET,
        "parent_config_sha256": DIRECT.CONFIG,
        "parent_metrics_sha256": PARENT / "metrics.json",
    }
    for key, path in assets.items():
        if sha256_file(path) != config["fixed_assets"][key]:
            raise RuntimeError(f"frozen asset mismatch: {key}")
    identity_path = OUTPUT / "preregistered_identity.json"
    if OUTPUT.exists() and any(OUTPUT.iterdir()):
        if not identity_path.is_file():
            raise FileExistsError(f"nonempty output lacks identity: {OUTPUT}")
        identity = json.loads(identity_path.read_text())
        if identity["config_sha256"] != sha256_file(CONFIG):
            raise RuntimeError("config changed after prepare")
        return identity
    OUTPUT.mkdir(parents=True, exist_ok=True)
    identity = {
        "created_utc": utcnow(), "config_sha256": sha256_file(CONFIG),
        "script_sha256_at_prepare": sha256_file(Path(__file__)),
        "post_direct_oof_exploration": True, "outcomes_produced": False,
    }
    identity_path.write_text(json.dumps(identity, indent=2) + "\n")
    return identity


def standardize(train_x, eval_x):
    mean = train_x.mean(0, dtype=np.float64)
    scale = train_x.std(0, dtype=np.float64)
    scale[scale < 1e-6] = 1.0
    return ((np.asarray(train_x, np.float64) - mean) / scale,
            (np.asarray(eval_x, np.float64) - mean) / scale)


def weighted_detector(train_x, train_y, eval_x):
    x, z = standardize(train_x, eval_x)
    target = np.asarray(train_y, np.float64)
    positives = float(target.sum()); negatives = float(len(target) - positives)
    weights = np.where(target > 0, negatives / max(1.0, positives), 1.0)
    intercept = float(np.sum(weights * target) / np.sum(weights))
    centered = target - intercept
    root = np.sqrt(weights)
    wx = x * root[:, None]
    gram = wx.T @ wx
    gram.flat[::len(gram) + 1] += ALPHA
    coef = np.linalg.solve(gram, wx.T @ (centered * root))
    return (z @ coef + intercept).astype(np.float32)


def average_precision(labels, scores):
    labels = np.asarray(labels, np.int8)
    order = np.lexsort((np.arange(len(scores)), -np.asarray(scores)))
    ranked = labels[order]
    positives = int(ranked.sum())
    if positives == 0:
        return float("nan")
    precision = np.cumsum(ranked) / np.arange(1, len(ranked) + 1)
    return float(precision[ranked == 1].sum() / positives)


def bootstrap(values, sources, reps=10000):
    grouped = defaultdict(float)
    for value, source in zip(values, sources):
        grouped[int(source)] += float(value)
    totals = np.asarray([grouped[key] for key in sorted(grouped)], np.float64)
    rng = np.random.default_rng(BOOT_SEED)
    result = np.empty(reps, np.float64)
    for index in range(reps):
        result[index] = totals[rng.integers(0, len(totals), size=len(totals))].sum()
    return [float(np.quantile(result, 0.025)), float(np.quantile(result, 0.975))]


def evaluate(matrix, targets, opportunity, folds):
    rows = len(targets)
    detector_scores = np.empty(rows, np.float32)
    chosen = np.full(rows, CENTER, np.int16)
    selected = np.zeros(rows, bool)
    fold_records = []
    for fold in range(5):
        train = folds != fold
        evaluate_rows = np.flatnonzero(folds == fold)
        train_positive = train & opportunity
        scores = weighted_detector(matrix[train], opportunity[train], matrix[evaluate_rows])
        detector_scores[evaluate_rows] = scores
        prevalence = float(opportunity[train].mean())
        action_count = int(round(len(evaluate_rows) * prevalence))
        order = np.lexsort((evaluate_rows, -scores))
        acting = evaluate_rows[order[:action_count]]
        selected[acting] = True
        prediction = DIRECT.ridge_predict(
            matrix[train_positive], targets[train_positive], matrix[acting], ALPHA)
        prediction[:, CENTER] = 0.0
        chosen[acting] = prediction.argmax(1).astype(np.int16)
        fold_records.append({
            "fold": fold, "train_rows": int(train.sum()),
            "train_positive": int(train_positive.sum()), "eval_rows": len(evaluate_rows),
            "action_count": action_count, "eval_positive": int(opportunity[evaluate_rows].sum()),
            "opportunity_recalled": int(opportunity[acting].sum()),
        })
    reward = targets[np.arange(rows), chosen]
    ap = average_precision(opportunity, detector_scores)
    return chosen, reward, detector_scores, {
        "detector_average_precision": ap,
        "opportunity_prevalence": float(opportunity.mean()),
        "selected_horizons": int(selected.sum()),
        "changed_horizons": int(np.sum(chosen != CENTER)),
        "opportunity_recall_at_action_count": float(opportunity[selected].sum() / opportunity.sum()),
        "signed_robust_reward": float(reward.sum()),
        "positive_reward_horizons": int(np.sum(reward > 0)),
        "negative_reward_horizons": int(np.sum(reward < 0)),
        "zero_reward_horizons": int(np.sum(reward == 0)),
        "folds": fold_records,
    }


def run():
    prepare()
    if (OUTPUT / "metrics.json").exists():
        raise FileExistsError("non-overwriting abstention outcome exists")
    config = json.loads(CONFIG.read_text())
    with np.load(DATASET, allow_pickle=False) as data:
        k, s, o, x = (np.asarray(data[key], np.float32) for key in ("K", "S", "O", "X"))
        targets = np.asarray(data["targets"], np.float32)
        folds = np.asarray(data["folds"], np.int8)
        sources = np.asarray(data["source_indices"], np.int16)
    opportunity = targets.max(1) > 0
    if len(targets) != config["scope"]["rows"] or int(opportunity.sum()) != config["scope"]["positive_opportunity_rows_expected"]:
        raise RuntimeError("scope identity mismatch")
    matrices = {"K": k, "KS": np.concatenate([k, s], 1),
                "KO": np.concatenate([k, o], 1), "KX": np.concatenate([k, x], 1)}
    model_metrics, choices, rewards, scores = {}, {}, {}, {}
    for name, matrix in matrices.items():
        choices[name], rewards[name], scores[name], model_metrics[name] = evaluate(
            matrix, targets, opportunity, folds)
        model_metrics[name]["source_bootstrap_95pct_total_reward"] = bootstrap(rewards[name], sources)
        print(json.dumps({"model": name, **model_metrics[name]}), flush=True)
    delta_k = rewards["KO"] - rewards["K"]
    delta_x = rewards["KO"] - rewards["KX"]
    comparisons = {
        "KO_minus_K": {
            "average_precision": model_metrics["KO"]["detector_average_precision"] - model_metrics["K"]["detector_average_precision"],
            "reward": float(delta_k.sum()), "reward_bootstrap_95pct": bootstrap(delta_k, sources)},
        "KO_minus_KX": {
            "average_precision": model_metrics["KO"]["detector_average_precision"] - model_metrics["KX"]["detector_average_precision"],
            "reward": float(delta_x.sum()), "reward_bootstrap_95pct": bootstrap(delta_x, sources)},
    }
    gates = config["gates"]
    checks = {
        "KO_average_precision": model_metrics["KO"]["detector_average_precision"] >= gates["KO_average_precision_at_least"],
        "KO_minus_K_average_precision": comparisons["KO_minus_K"]["average_precision"] >= gates["KO_minus_K_average_precision_at_least"],
        "KO_minus_KX_average_precision": comparisons["KO_minus_KX"]["average_precision"] >= gates["KO_minus_KX_average_precision_at_least"],
        "KO_reward": model_metrics["KO"]["signed_robust_reward"] >= gates["KO_signed_robust_reward_at_least"],
        "KO_reward_bootstrap": model_metrics["KO"]["source_bootstrap_95pct_total_reward"][0] > gates["KO_reward_bootstrap_lower_gt"],
        "KO_minus_K_reward": comparisons["KO_minus_K"]["reward"] >= gates["KO_minus_K_reward_at_least"],
        "KO_minus_K_reward_bootstrap": comparisons["KO_minus_K"]["reward_bootstrap_95pct"][0] > gates["KO_minus_K_reward_bootstrap_lower_gt"],
        "KO_minus_KX_reward": comparisons["KO_minus_KX"]["reward"] >= gates["KO_minus_KX_reward_at_least"],
        "KO_minus_KX_reward_bootstrap": comparisons["KO_minus_KX"]["reward_bootstrap_95pct"][0] > gates["KO_minus_KX_reward_bootstrap_lower_gt"],
    }
    passed = all(checks.values())
    metrics = {
        "status": "complete", "created_utc": utcnow(), "gpu_used": False,
        "train_fit_reused_post_direct_oof": True,
        "calibration_dev_test_heldout_or_closed_loop_read": False,
        "rows": len(targets), "sources": len(np.unique(sources)), "positive_opportunities": int(opportunity.sum()),
        "models": model_metrics, "comparisons": comparisons, "checks": checks, "gate_passed": passed,
        "decision": ("unlock separately preregistered independent calibration confirmation only" if passed else
                     "stop this abstention identity; do not run calibration, closed loop, or held-out"),
    }
    (OUTPUT / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    np.savez_compressed(OUTPUT / "oof_abstention.npz", folds=folds, source_indices=sources,
                        opportunity=opportunity, **{f"chosen_{key}": value for key, value in choices.items()},
                        **{f"reward_{key}": value for key, value in rewards.items()},
                        **{f"detector_score_{key}": value for key, value in scores.items()})
    manifest = {
        "status": "complete", "config_sha256": sha256_file(CONFIG),
        "identity_sha256": sha256_file(OUTPUT / "preregistered_identity.json"),
        "metrics_sha256": sha256_file(OUTPUT / "metrics.json"),
        "oof_sha256": sha256_file(OUTPUT / "oof_abstention.npz"), "gate_passed": passed,
    }
    (OUTPUT / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--prepare", action="store_true")
    parser.add_argument("--run", action="store_true")
    args = parser.parse_args()
    if args.prepare == args.run:
        parser.error("choose exactly one of --prepare or --run")
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    result = prepare() if args.prepare else run()
    print(json.dumps(result, indent=2))
