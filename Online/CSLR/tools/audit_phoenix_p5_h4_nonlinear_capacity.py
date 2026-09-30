#!/usr/bin/env python3
"""Fixed CPU-MLP audit of nonlinear H4 opportunity detectability."""
import argparse
import hashlib
import importlib.util
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "results/phoenix-2014t_ISLR"
PARENT = BASE / "p5_h4_schedule_ranker_oof_v1_49faacc3"
OOF_AUDIT = BASE / "p5_h4_oof_design_audit_v1_49faacc3"
STABILITY = BASE / "p5_h4_target_stability_audit_v1_49faacc3"
OUTPUT = BASE / "p5_h4_nonlinear_capacity_audit_v1_49faacc3"
CONFIG = ROOT / "configs/p5_h4_nonlinear_capacity_audit_v1_preregistered.json"
DATASET = PARENT / "h4_ranker_dataset.npz"


def imp(name):
    path = ROOT / f"tools/{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


OOF = imp("audit_phoenix_p5_h4_oof_design")
ABSTAIN = OOF.ABSTAIN


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
        "oof_audit_metrics_sha256": OOF_AUDIT / "metrics.json",
        "target_stability_metrics_sha256": STABILITY / "metrics.json",
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
    identity = {"created_utc": utcnow(), "config_sha256": sha256_file(CONFIG),
                "script_sha256_at_prepare": sha256_file(Path(__file__)),
                "diagnostic_only": True, "gpu_used": False}
    identity_path.write_text(json.dumps(identity, indent=2) + "\n")
    return identity


class OpportunityMLP(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.network = nn.Sequential(nn.Linear(width, 64), nn.ReLU(),
                                     nn.Linear(64, 16), nn.ReLU(), nn.Linear(16, 1))

    def forward(self, value):
        return self.network(value).squeeze(1)


def fit_predict(matrix, labels, train, evaluate, seed):
    mean = matrix[train].mean(0, dtype=np.float64)
    scale = matrix[train].std(0, dtype=np.float64)
    scale[scale < 1e-6] = 1.0
    train_x = np.asarray((matrix[train] - mean) / scale, np.float32)
    eval_x = np.asarray((matrix[evaluate] - mean) / scale, np.float32)
    train_y = np.asarray(labels[train], np.float32)
    torch.manual_seed(seed)
    model = OpportunityMLP(matrix.shape[1])
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.0001)
    positive = float(train_y.sum()); negative = float(len(train_y) - positive)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(negative / max(1.0, positive)))
    x_tensor = torch.from_numpy(train_x)
    y_tensor = torch.from_numpy(train_y)
    generator = torch.Generator().manual_seed(seed + 1000)
    model.train()
    losses = []
    for _ in range(10):
        order = torch.randperm(len(train_y), generator=generator)
        total = 0.0
        for left in range(0, len(order), 2048):
            chosen = order[left:left + 2048]
            optimizer.zero_grad(set_to_none=True)
            loss = loss_fn(model(x_tensor[chosen]), y_tensor[chosen])
            loss.backward(); optimizer.step()
            total += float(loss.detach()) * len(chosen)
        losses.append(total / len(train_y))
    model.eval()
    with torch.no_grad():
        scores = model(torch.from_numpy(eval_x)).numpy().astype(np.float32)
    return scores, losses


def resub(matrix, labels, seed):
    mask = np.ones(len(labels), bool)
    scores, losses = fit_predict(matrix, labels, mask, mask, seed)
    return {"average_precision": ABSTAIN.average_precision(labels, scores),
            "epoch_losses": losses}, scores


def oof(matrix, labels, folds, seeds):
    scores = np.empty(len(labels), np.float32)
    records = []
    for fold in range(5):
        train, evaluate = folds != fold, folds == fold
        scores[evaluate], losses = fit_predict(matrix, labels, train, evaluate, seeds[fold])
        records.append({"fold": fold, "train_rows": int(train.sum()),
                        "eval_rows": int(evaluate.sum()), "train_positive": int(labels[train].sum()),
                        "eval_positive": int(labels[evaluate].sum()), "epoch_losses": losses})
        print(json.dumps({"fold": fold, "eval_rows": int(evaluate.sum()),
                          "last_loss": losses[-1]}), flush=True)
    return {"average_precision": ABSTAIN.average_precision(labels, scores), "folds": records}, scores


def run():
    prepare()
    if (OUTPUT / "metrics.json").exists():
        raise FileExistsError("non-overwriting nonlinear audit exists")
    config = json.loads(CONFIG.read_text())
    torch.set_num_threads(8)
    torch.use_deterministic_algorithms(True)
    with np.load(DATASET, allow_pickle=False) as data:
        matrix = np.concatenate([np.asarray(data["K"], np.float32),
                                 np.asarray(data["O"], np.float32)], axis=1)
        targets = np.asarray(data["targets"], np.float32)
        source_folds = np.asarray(data["folds"], np.int8)
        sample_ids = [str(value) for value in data["sample_ids"].tolist()]
        horizon_ids = np.asarray(data["horizon_indices"], np.int16)
    labels = targets.max(1) > 0
    if matrix.shape != (43437, 409) or int(labels.sum()) != 824:
        raise RuntimeError("scope mismatch")
    row_folds = np.asarray([OOF.hash_fold("p5-oof-audit-row-v1", f"{sample}\0{horizon}")
                            for sample, horizon in zip(sample_ids, horizon_ids)], np.int8)
    sample_map = {sample: OOF.hash_fold("p5-oof-audit-sample-v1", sample) for sample in set(sample_ids)}
    sample_folds = np.asarray([sample_map[sample] for sample in sample_ids], np.int8)
    seeds = config["model"]["seeds"]
    regimes, score_archive = {}, {}
    regimes["resubstitution"], score_archive["resubstitution"] = resub(
        matrix, labels, seeds["resubstitution"])
    for name, folds, key in (
        ("row_random_5fold", row_folds, "row_random_folds"),
        ("sample_disjoint_5fold", sample_folds, "sample_disjoint_folds"),
        ("source_disjoint_5fold", source_folds, "source_disjoint_folds"),
    ):
        regimes[name], score_archive[name] = oof(matrix, labels, folds, seeds[key])
    rng = np.random.default_rng(261153)
    shuffled = labels[rng.permutation(len(labels))]
    negative, score_archive["shuffled_label_source_5fold"] = oof(
        matrix, shuffled, source_folds, seeds["shuffled_label_source_folds"])
    negative["prevalence"] = float(shuffled.mean())
    ridge = json.loads((OOF_AUDIT / "metrics.json").read_text())
    ridge_source = ridge["regimes"]["source_disjoint_5fold"]["KO"]["opportunity_average_precision"]
    checks = {
        "material_nonlinear_source_gain": regimes["source_disjoint_5fold"]["average_precision"] - ridge_source >= 0.02,
        "large_nonlinear_row_source_gap": regimes["row_random_5fold"]["average_precision"] - regimes["source_disjoint_5fold"]["average_precision"] >= 0.02,
        "strong_resub_only": (regimes["resubstitution"]["average_precision"] >= 0.10 and
                              max(regimes[key]["average_precision"] for key in
                                  ("row_random_5fold", "sample_disjoint_5fold", "source_disjoint_5fold")) < 0.05),
        "negative_control_near_prevalence": abs(negative["average_precision"] - negative["prevalence"]) < 0.01,
    }
    metrics = {
        "status": "complete", "created_utc": utcnow(), "gpu_used": False,
        "diagnostic_only": True, "calibration_dev_test_heldout_or_closed_loop_read": False,
        "rows": len(labels), "positives": int(labels.sum()), "prevalence": float(labels.mean()),
        "regimes": regimes, "shuffled_label_source_5fold": negative,
        "ridge_source_ap": ridge_source,
        "nonlinear_source_gain": regimes["source_disjoint_5fold"]["average_precision"] - ridge_source,
        "interpretation_checks": checks,
        "decision": "capacity diagnostic only; no policy unlock",
    }
    (OUTPUT / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    np.savez_compressed(OUTPUT / "scores.npz", labels=labels, row_folds=row_folds,
                        sample_folds=sample_folds, source_folds=source_folds, **score_archive)
    manifest = {"status": "complete", "config_sha256": sha256_file(CONFIG),
                "identity_sha256": sha256_file(OUTPUT / "preregistered_identity.json"),
                "metrics_sha256": sha256_file(OUTPUT / "metrics.json"),
                "scores_sha256": sha256_file(OUTPUT / "scores.npz"), "diagnostic_only": True}
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
