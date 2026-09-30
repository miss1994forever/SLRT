#!/usr/bin/env python3
"""Post-OOF CPU-only split and fixed-model failure-localization audit."""
import argparse
import hashlib
import importlib.util
import json
import os
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "results/phoenix-2014t_ISLR"
PARENT = BASE / "p5_h4_schedule_ranker_oof_v1_49faacc3"
ABSTENTION = BASE / "p5_h4_abstention_ranker_oof_v1_49faacc3"
OUTPUT = BASE / "p5_h4_oof_design_audit_v1_49faacc3"
CONFIG = ROOT / "configs/p5_h4_oof_design_audit_v1_preregistered.json"
DATASET = PARENT / "h4_ranker_dataset.npz"
CENTER = 40


def imp(name):
    path = ROOT / f"tools/{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


DIRECT = imp("analyze_phoenix_p5_h4_schedule_ranker_oof")
ABSTAIN = imp("analyze_phoenix_p5_h4_abstention_ranker_oof")


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
        "dataset_sha256": DATASET,
        "direct_metrics_sha256": PARENT / "metrics.json",
        "abstention_metrics_sha256": ABSTENTION / "metrics.json",
    }
    for key, path in assets.items():
        if sha256_file(path) != config["fixed_assets"][key]:
            raise RuntimeError(f"asset mismatch: {key}")
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
        "diagnostic_only": True, "outcomes_produced": False,
    }
    identity_path.write_text(json.dumps(identity, indent=2) + "\n")
    return identity


def hash_fold(namespace, value):
    token = f"{namespace}\0{value}".encode()
    return int.from_bytes(hashlib.sha256(token).digest()[:8], "big") % 5


def detector_oof(matrix, opportunity, folds):
    scores = np.empty(len(opportunity), np.float32)
    records = []
    for fold in range(5):
        train, evaluate = folds != fold, folds == fold
        scores[evaluate] = ABSTAIN.weighted_detector(
            matrix[train], opportunity[train], matrix[evaluate])
        records.append({
            "fold": fold, "train_rows": int(train.sum()), "eval_rows": int(evaluate.sum()),
            "train_positive": int(opportunity[train].sum()),
            "eval_positive": int(opportunity[evaluate].sum()),
        })
    return scores, records


def detector_resub(matrix, opportunity):
    return ABSTAIN.weighted_detector(matrix, opportunity, matrix)


def conditional_rank(matrix, targets, opportunity, folds=None):
    positive_rows = np.flatnonzero(opportunity)
    prediction = np.empty((len(positive_rows), targets.shape[1]), np.float32)
    if folds is None:
        prediction[:] = DIRECT.ridge_predict(
            matrix[positive_rows], targets[positive_rows], matrix[positive_rows], 100.0)
    else:
        for fold in range(5):
            train = opportunity & (folds != fold)
            evaluate_local = np.flatnonzero(folds[positive_rows] == fold)
            evaluation = positive_rows[evaluate_local]
            prediction[evaluate_local] = DIRECT.ridge_predict(
                matrix[train], targets[train], matrix[evaluation], 100.0)
    prediction[:, CENTER] = 0.0
    chosen = prediction.argmax(1)
    reward = targets[positive_rows, chosen]
    oracle = targets[positive_rows].max(1)
    return {
        "positive_rows": len(positive_rows),
        "signed_reward": float(reward.sum()),
        "oracle_reward": float(oracle.sum()),
        "oracle_reward_capture": float(reward.sum() / oracle.sum()),
        "positive_selected": int(np.sum(reward > 0)),
        "negative_selected": int(np.sum(reward < 0)),
        "zero_selected": int(np.sum(reward == 0)),
        "center_selected": int(np.sum(chosen == CENTER)),
    }


def evaluate_regime(matrices, targets, opportunity, folds=None):
    result = {}
    for name, matrix in matrices.items():
        if folds is None:
            scores = detector_resub(matrix, opportunity)
            records = None
        else:
            scores, records = detector_oof(matrix, opportunity, folds)
        result[name] = {
            "opportunity_average_precision": ABSTAIN.average_precision(opportunity, scores),
            "conditional_ranker": conditional_rank(matrix, targets, opportunity, folds),
        }
        if records is not None:
            result[name]["folds"] = records
    return result


def run():
    prepare()
    if (OUTPUT / "metrics.json").exists():
        raise FileExistsError("non-overwriting audit outcome exists")
    config = json.loads(CONFIG.read_text())
    with np.load(DATASET, allow_pickle=False) as data:
        k, s, o, x = (np.asarray(data[key], np.float32) for key in ("K", "S", "O", "X"))
        targets = np.asarray(data["targets"], np.float32)
        source_folds = np.asarray(data["folds"], np.int8)
        sources = np.asarray(data["source_indices"], np.int16)
        sample_ids = [str(value) for value in data["sample_ids"].tolist()]
        horizon_ids = np.asarray(data["horizon_indices"], np.int16)
    opportunity = targets.max(1) > 0
    if len(targets) != config["scope"]["rows"] or int(opportunity.sum()) != config["scope"]["positive_opportunities"]:
        raise RuntimeError("scope identity mismatch")
    row_folds = np.asarray([hash_fold("p5-oof-audit-row-v1", f"{sample}\0{horizon}")
                            for sample, horizon in zip(sample_ids, horizon_ids)], np.int8)
    sample_fold_map = {sample: hash_fold("p5-oof-audit-sample-v1", sample) for sample in set(sample_ids)}
    sample_folds = np.asarray([sample_fold_map[sample] for sample in sample_ids], np.int8)
    matrices = {"K": k, "KS": np.concatenate([k, s], 1),
                "KO": np.concatenate([k, o], 1), "KX": np.concatenate([k, x], 1)}
    regimes = {
        "resubstitution": evaluate_regime(matrices, targets, opportunity, None),
        "row_random_5fold": evaluate_regime(matrices, targets, opportunity, row_folds),
        "sample_disjoint_5fold": evaluate_regime(matrices, targets, opportunity, sample_folds),
        "source_disjoint_5fold": evaluate_regime(matrices, targets, opportunity, source_folds),
    }
    rng = np.random.default_rng(261153)
    shuffled = opportunity[rng.permutation(len(opportunity))]
    shuffled_scores, shuffled_folds = detector_oof(matrices["KO"], shuffled, source_folds)
    negative_control = {
        "feature_set": "KO", "positive_count": int(shuffled.sum()),
        "average_precision": ABSTAIN.average_precision(shuffled, shuffled_scores),
        "prevalence": float(shuffled.mean()), "folds": shuffled_folds,
    }
    positive_by_source = Counter(int(source) for source in sources[opportunity])
    ordered_counts = sorted(positive_by_source.values(), reverse=True)
    concentration = {
        "sources_total": len(np.unique(sources)),
        "sources_with_positive": len(positive_by_source),
        "sources_without_positive": len(np.unique(sources)) - len(positive_by_source),
        "max_positive_in_one_source": max(ordered_counts),
        "top_10_source_fraction": float(sum(ordered_counts[:10]) / opportunity.sum()),
        "median_positive_among_positive_sources": float(np.median(ordered_counts)),
        "source_fold_positive_counts": [int(opportunity[source_folds == fold].sum()) for fold in range(5)],
    }
    gaps = {}
    for name in matrices:
        random_ap = regimes["row_random_5fold"][name]["opportunity_average_precision"]
        sample_ap = regimes["sample_disjoint_5fold"][name]["opportunity_average_precision"]
        source_ap = regimes["source_disjoint_5fold"][name]["opportunity_average_precision"]
        gaps[name] = {
            "row_random_minus_source_ap": random_ap - source_ap,
            "sample_disjoint_minus_source_ap": sample_ap - source_ap,
        }
    ko_row = regimes["row_random_5fold"]["KO"]["opportunity_average_precision"]
    ko_source = regimes["source_disjoint_5fold"]["KO"]["opportunity_average_precision"]
    ko_resub_capture = regimes["resubstitution"]["KO"]["conditional_ranker"]["oracle_reward_capture"]
    interpretation = {
        "large_KO_row_source_gap": ko_row - ko_source >= 0.02,
        "KO_all_oof_splits_low": ko_row < 0.05,
        "KO_conditional_resub_capture_below_25pct": ko_resub_capture < 0.25,
        "label_shuffle_near_prevalence": abs(negative_control["average_precision"] - negative_control["prevalence"]) < 0.01,
    }
    metrics = {
        "status": "complete", "created_utc": utcnow(), "gpu_used": False,
        "diagnostic_only": True, "calibration_dev_test_heldout_or_closed_loop_read": False,
        "rows": len(targets), "positive_opportunities": int(opportunity.sum()),
        "opportunity_prevalence": float(opportunity.mean()),
        "positive_source_concentration": concentration,
        "regimes": regimes, "split_gaps": gaps,
        "label_shuffled_negative_control": negative_control,
        "interpretation_checks": interpretation,
        "decision": "diagnose target/model/split failure only; no policy unlock",
    }
    (OUTPUT / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    manifest = {
        "status": "complete", "config_sha256": sha256_file(CONFIG),
        "identity_sha256": sha256_file(OUTPUT / "preregistered_identity.json"),
        "metrics_sha256": sha256_file(OUTPUT / "metrics.json"), "diagnostic_only": True,
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
