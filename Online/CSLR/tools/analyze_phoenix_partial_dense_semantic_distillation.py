#!/usr/bin/env python3
"""Fit-only CPU smoke for dense-ISLR semantic distillation into a cheap selector."""
import argparse
import copy
import hashlib
import importlib.util
import json
import math
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
FRESH = ROOT / "results/phoenix-2014t_ISLR/p3_partial32_fresh_terminal_listwise_exploratory_v1_49faacc3"
DATA = FRESH.with_name(FRESH.name + "_dataset")
HOG_ROOT = ROOT / "results/phoenix-2014t_ISLR/p3_partial32_hog_preview_fit512_oof_v1_49faacc3"
OUTPUT = ROOT / "results/phoenix-2014t_ISLR/p3_partial32_dense_semantic_distillation_fit512_oof_v1_49faacc3"
POSE_ARCHIVE = Path("/tmp/phoenix_p2_train_center_sequences_v1.npz")
SEED = 261025
FOLD_SEED = 261022
FOLDS = 5
PCA_DIM = 32
PCA_Q = 40
PCA_NITER = 5
DISTILL_EPOCHS = 10
RESIDUAL_EPOCHS = 20
BATCH = 128
FALSE_OVERRIDE_COST = 4.0
COVERAGES = (0.005, 0.01, 0.02, 0.05, 0.10)


def imp(name):
    path = ROOT / f"tools/{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


FRESHMOD = imp("analyze_phoenix_partial_fresh_terminal_listwise")
HOGMOD = imp("analyze_phoenix_partial_hog_preview_oof")
SELECTIVE = HOGMOD.SELECTIVE
PREVIEW, CHRON, BUILDER = FRESHMOD.PREVIEW, FRESHMOD.CHRON, FRESHMOD.BUILDER


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def frozen_ids():
    cfg = json.loads((FRESH / "resolved_config_preregistered.json").read_text())
    ids = cfg["fit_512_scaling_subset"]["sample_ids"]
    if len(ids) != 512:
        raise ValueError("frozen subset is not 512 samples")
    return ids


def source_fold(source):
    value = hashlib.sha256(f"{FOLD_SEED}\0{source}".encode()).hexdigest()
    return int(value[:16], 16) % FOLDS


def config():
    ids = frozen_ids()
    hog_manifest = json.loads((HOG_ROOT / "cache_manifest.json").read_text())
    return {
        "experiment": "fit-only dense semantic distillation source-group OOF smoke",
        "created_before_OOF_outcomes": True,
        "cpu_only": True,
        "scope": {
            "samples": 512, "sources": 211,
            "sample_ids_sha256": hashlib.sha256("\n".join(ids).encode()).hexdigest(),
            "terminal_rows": str(DATA), "dense_logits": str(CHRON.DEFAULT_DENSE_ROOT),
            "pose": str(POSE_ARCHIVE), "HOG": str(HOG_ROOT / "hog_preview_fit512_fp16.npz"),
            "HOG_sha256": hog_manifest["sha256"],
        },
        "teacher": {
            "value": "raw 1116-dimensional complete-ISLR candidate gloss logits; training teacher only",
            "PCA": {"dimensions": PCA_DIM, "method": "torch randomized low-rank PCA", "q": PCA_Q,
                    "niter": PCA_NITER, "centering": True, "whiten_regression_target": True},
            "leakage_control": "PCA mean/components/scales fit separately inside each source-fold training partition",
            "metrics": ["train explained-variance ratio", "held-out embedding R2", "held-out cosine",
                        "student reconstructed-top1 agreement", "PCA-ceiling reconstructed-top1 agreement"],
        },
        "student": {
            "input": "189-d candidate feature: 49 bookkeeping/past prefix + 68 pose summary + 72 HOG summary",
            "causality": "pose/HOG are the already-audited 16-frame candidate preview available at candidate execution; no full logits/reference/EOS at deployment",
            "shared_encoder": "Linear(189,64)-GELU-Linear(64,32)-GELU",
            "semantic_head": "Linear(32,32)",
            "residual_head": "Linear(32,16)-GELU-Linear(16,1); side score minus center score",
            "feature_normalization": "mean/std fit inside each source-fold training partition",
        },
        "training": {
            "comparison_A": "same randomly initialized encoder/head trained only for residual utility",
            "comparison_B": "identical initial state; encoder first distilled to PCA teacher, then encoder/residual head fine-tuned for utility",
            "distillation": {"epochs": DISTILL_EPOCHS, "loss": "MSE on whitened PCA coordinates", "all_training_candidates": True},
            "residual": {"epochs": RESIDUAL_EPOCHS, "loss": "SmoothL1 signed center-minus-side terminal advantage; false-positive override cost 4",
                         "training_blocks": "terminal-informative blocks only"},
            "optimizer": "AdamW(lr=1e-3, weight_decay=1e-4)", "batch": BATCH, "seed": SEED,
            "semantic_retention_during_residual": "none; report held-out semantic metrics before and after fine-tuning",
        },
        "OOF": {
            "folds": FOLDS, "assignment": "sha256(261022+NUL+source_video_id) mod5",
            "coverages": list(COVERAGES),
            "threshold": "global OOF max-side-score quantile for each method, additionally score>0",
            "reports": "positive/harmful/neutral overrides, precision, reward, regret, fold rewards",
            "pass": "at least one coverage where distilled reward>0, every fold reward>=0, overrides>=20, reward>random-init, and regret<random-init",
        },
        "after_gate": "pass only permits a separate calibration experiment; this run never reads calibration/evaluation",
        "forbidden": ["GPU/CUDA", "dev", "test", "calibration/evaluation outcomes", "deployment full logits/reference/EOS", "git commit"],
    }


def preregister(root):
    if root.exists():
        raise FileExistsError(root)
    root.mkdir(parents=True)
    value = config()
    (root / "resolved_config_preregistered.json").write_text(json.dumps(value, indent=2) + "\n")
    return {"status": "preregistered", "samples": 512, "sources": 211, "folds": FOLDS,
            "PCA_dimensions": PCA_DIM, "distill_epochs": DISTILL_EPOCHS, "residual_epochs": RESIDUAL_EPOCHS}


def loadcfg(root):
    value = json.loads((root / "resolved_config_preregistered.json").read_text())
    if value != config():
        raise ValueError("preregistered configuration changed")
    return value


class SemanticResidual(nn.Module):
    def __init__(self, width=189):
        super().__init__()
        self.encoder = nn.Sequential(nn.Linear(width, 64), nn.GELU(), nn.Linear(64, 32), nn.GELU())
        self.semantic_head = nn.Linear(32, PCA_DIM)
        self.residual_head = nn.Sequential(nn.Linear(32, 16), nn.GELU(), nn.Linear(16, 1))

    def forward(self, value):
        batch, choices, width = value.shape
        hidden = self.encoder(value.reshape(batch * choices, width))
        semantic = self.semantic_head(hidden).reshape(batch, choices, PCA_DIM)
        score = self.residual_head(hidden).reshape(batch, choices)
        return semantic, score


def load_data():
    ids = frozen_ids()
    rows = HOGMOD.load_fit_rows(ids)
    pose = PREVIEW.PreviewArchive(POSE_ARCHIVE)
    hog = HOGMOD.HogArchive(HOG_ROOT)
    base = HOGMOD.arrays(rows, pose, hog)
    features = np.concatenate([base["base_pose"], base["hog"]], axis=2).astype(np.float32)
    wanted = set(ids)
    logits_by_name = {}
    indices, workers = CHRON.completed_shard_indices(CHRON.DEFAULT_DENSE_ROOT)
    for shard in indices:
        results, logits, _ = BUILDER.BUILDER.validate_dense_shard(CHRON.DEFAULT_DENSE_ROOT, shard, workers, verify_hashes=False)
        for name in results:
            if name in wanted:
                logits_by_name[name] = np.asarray(logits[name], np.float32)
    if set(logits_by_name) != wanted:
        raise ValueError("dense logits do not cover frozen fit512")
    teacher = []
    identities = []
    samples = []
    for row in rows:
        matrix = logits_by_name[row["sample_id"]]
        starts = np.asarray(row["candidate_starts"], np.int64)
        if starts.min() < 0 or starts.max() >= len(matrix) or matrix.shape[1] != 1116:
            raise ValueError("unreliable dense candidate-logit schema")
        teacher.append(matrix[starts])
        identities.append(f"{row['sample_id']}:{row['block_start']}")
        samples.append(row["sample_id"])
    return {**base, "features": features, "teacher": np.asarray(teacher, np.float32),
            "identity": np.asarray(identities), "sample": np.asarray(samples)}


def feature_stats(features, train_mask):
    values = features[train_mask]
    mean = values.mean((0, 1), dtype=np.float64).astype(np.float32)
    scale = values.std((0, 1), dtype=np.float64).astype(np.float32)
    scale[scale < 1e-6] = 1
    return mean, scale


def fit_pca(logits, train_mask, seed):
    values = torch.from_numpy(logits[train_mask].reshape(-1, logits.shape[-1])).float()
    mean = values.mean(0)
    centered = values - mean
    torch.manual_seed(seed)
    _, singular, vectors = torch.pca_lowrank(centered, q=PCA_Q, center=False, niter=PCA_NITER)
    components = vectors[:, :PCA_DIM].contiguous()
    singular = singular[:PCA_DIM]
    variance = singular.square() / max(1, len(values) - 1)
    total_variance = centered.square().sum() / max(1, len(values) - 1)
    scales = variance.sqrt().clamp_min(1e-6)
    return {"mean": mean.numpy(), "components": components.numpy(), "scales": scales.numpy(),
            "explained_variance_ratio": float(variance.sum() / total_variance), "train_candidates": int(len(values))}


def pca_targets(logits, pca):
    return (((logits - pca["mean"]) @ pca["components"]) / pca["scales"]).astype(np.float32)


def normalized_batch(data, indices, stats):
    return torch.from_numpy((data["features"][indices] - stats[0]) / stats[1]).float()


def distill(model, data, train_mask, stats, pca, seed):
    idx = np.flatnonzero(train_mask)
    target = pca_targets(data["teacher"], pca)
    optimizer = torch.optim.AdamW(list(model.encoder.parameters()) + list(model.semantic_head.parameters()), lr=1e-3, weight_decay=1e-4)
    generator = torch.Generator().manual_seed(seed)
    history = []
    for _ in range(DISTILL_EPOCHS):
        order = idx[torch.randperm(len(idx), generator=generator).numpy()]
        total = 0.0
        for left in range(0, len(order), BATCH):
            batch = order[left:left+BATCH]
            pred, _ = model(normalized_batch(data, batch, stats))
            truth = torch.from_numpy(target[batch]).float()
            loss = F.mse_loss(pred, truth)
            optimizer.zero_grad(); loss.backward(); optimizer.step()
            total += float(loss) * len(batch)
        history.append(total / len(order))
    return history


def residual_train(model, data, train_mask, stats, seed):
    idx = np.flatnonzero(train_mask & data["informative"])
    optimizer = torch.optim.AdamW(list(model.encoder.parameters()) + list(model.residual_head.parameters()), lr=1e-3, weight_decay=1e-4)
    generator = torch.Generator().manual_seed(seed)
    history = []
    for _ in range(RESIDUAL_EPOCHS):
        order = idx[torch.randperm(len(idx), generator=generator).numpy()]
        total = 0.0
        for left in range(0, len(order), BATCH):
            batch = order[left:left+BATCH]
            _, score = model(normalized_batch(data, batch, stats))
            pred = score[:, [0, 2]] - score[:, 1:2]
            target = torch.from_numpy(data["advantage"][batch]).float()
            per = F.smooth_l1_loss(pred, target, reduction="none")
            weight = torch.where((target <= 0) & (pred > 0), torch.full_like(per, FALSE_OVERRIDE_COST), torch.ones_like(per))
            loss = (per * weight).mean()
            optimizer.zero_grad(); loss.backward(); optimizer.step()
            total += float(loss) * len(batch)
        history.append(total / len(order))
    model.eval()
    return history


def predict_scores(model, data, mask, stats):
    idx = np.flatnonzero(mask); scores = []; semantic = []
    with torch.no_grad():
        for left in range(0, len(idx), 1024):
            part = idx[left:left+1024]
            sem, score = model(normalized_batch(data, part, stats))
            scores.append((score[:, [0, 2]] - score[:, 1:2]).numpy())
            semantic.append(sem.numpy())
    return idx, np.concatenate(scores), np.concatenate(semantic)


def semantic_metrics(predicted, teacher_logits, pca):
    truth = pca_targets(teacher_logits, pca).reshape(-1, PCA_DIM)
    pred = predicted.reshape(-1, PCA_DIM)
    centered = truth - truth.mean(0, keepdims=True)
    r2 = 1.0 - float(np.square(pred-truth).sum() / max(1e-12, np.square(centered).sum()))
    cosine = np.sum(pred*truth, axis=1) / np.maximum(1e-12, np.linalg.norm(pred, axis=1)*np.linalg.norm(truth, axis=1))
    raw = teacher_logits.reshape(-1, teacher_logits.shape[-1])
    teacher_top = raw.argmax(1)
    reconstructed_student = (pred * pca["scales"]) @ pca["components"].T + pca["mean"]
    reconstructed_pca = (truth * pca["scales"]) @ pca["components"].T + pca["mean"]
    return {"candidates": int(len(pred)), "embedding_R2": r2, "mean_cosine": float(cosine.mean()),
            "student_reconstructed_top1_agreement": float(np.mean(reconstructed_student.argmax(1) == teacher_top)),
            "PCA_ceiling_reconstructed_top1_agreement": float(np.mean(reconstructed_pca.argmax(1) == teacher_top))}


def reports(data, scores, folds):
    result = {}
    for coverage in COVERAGES:
        threshold = SELECTIVE.candidate_threshold(scores, coverage)
        result[str(coverage)] = {"nominal_coverage": coverage, **SELECTIVE.policy_metrics(data, scores, threshold, folds)}
    return result


def run_oof(root):
    loadcfg(root)
    (root / "oof_started.marker").write_text(datetime.now(timezone.utc).isoformat() + "\n")
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
    torch.set_num_threads(min(8, os.cpu_count() or 1)); torch.use_deterministic_algorithms(True)
    tick = time.perf_counter(); data = load_data(); assembly_seconds = time.perf_counter() - tick
    if data["features"].shape[2] != 189 or data["teacher"].shape[1] != 3 or data["teacher"].shape[2] != 1116:
        raise ValueError("unexpected student/teacher schema")
    folds = np.asarray([source_fold(x) for x in data["source"]], np.int64)
    random_scores = np.full((len(folds), 2), np.nan, np.float32)
    distilled_scores = np.full_like(random_scores, np.nan)
    fold_reports = {}
    for fold in range(FOLDS):
        train_mask, held = folds != fold, folds == fold
        stats = feature_stats(data["features"], train_mask)
        pca = fit_pca(data["teacher"], train_mask, SEED + fold)
        torch.manual_seed(SEED + fold)
        initial = SemanticResidual(data["features"].shape[2])
        random_model = copy.deepcopy(initial); distilled_model = copy.deepcopy(initial)
        distill_history = distill(distilled_model, data, train_mask, stats, pca, SEED + 100 + fold)
        held_idx, _, semantic_before = predict_scores(distilled_model, data, held, stats)
        before = semantic_metrics(semantic_before, data["teacher"][held_idx], pca)
        random_history = residual_train(random_model, data, train_mask, stats, SEED + 200 + fold)
        distilled_history = residual_train(distilled_model, data, train_mask, stats, SEED + 200 + fold)
        ridx, rscore, _ = predict_scores(random_model, data, held, stats)
        didx, dscore, semantic_after = predict_scores(distilled_model, data, held, stats)
        if not np.array_equal(ridx, didx) or not np.array_equal(ridx, held_idx):
            raise RuntimeError("held-out prediction mismatch")
        random_scores[ridx] = rscore; distilled_scores[didx] = dscore
        after = semantic_metrics(semantic_after, data["teacher"][didx], pca)
        fold_reports[str(fold)] = {
            "train_sources": len(set(data["source"][train_mask])), "heldout_sources": len(set(data["source"][held])),
            "train_blocks": int(train_mask.sum()), "heldout_blocks": int(held.sum()),
            "informative_train_blocks": int((train_mask & data["informative"]).sum()),
            "teacher_PCA": {"dimensions": PCA_DIM, "train_candidates": pca["train_candidates"], "explained_variance_ratio": pca["explained_variance_ratio"]},
            "heldout_semantics_before_residual": before, "heldout_semantics_after_residual": after,
            "history": {"distill": distill_history, "random_residual": random_history, "distilled_residual": distilled_history},
        }
        print(json.dumps({"fold_complete": fold, "PCA_explained": pca["explained_variance_ratio"], "R2_before": before["embedding_R2"], "seconds": time.perf_counter()-tick}), flush=True)
    if np.isnan(random_scores).any() or np.isnan(distilled_scores).any():
        raise RuntimeError("incomplete OOF")
    random_report = reports(data, random_scores, folds)
    distilled_report = reports(data, distilled_scores, folds)
    comparison = []
    for coverage in COVERAGES:
        key = str(coverage); a, b = random_report[key], distilled_report[key]
        checks = {"aggregate_reward_gt_0": b["cumulative_terminal_reward"] > 0,
                  "every_fold_reward_ge_0": all(x >= 0 for x in b["fold_reward"].values()),
                  "at_least_20_overrides": b["overrides"] >= 20,
                  "reward_gt_random_init": b["cumulative_terminal_reward"] > a["cumulative_terminal_reward"],
                  "regret_lt_random_init": b["total_terminal_regret"] < a["total_terminal_regret"]}
        comparison.append({"nominal_coverage": coverage, "passed": all(checks.values()), "checks": checks,
                           "distilled_minus_random_reward": b["cumulative_terminal_reward"]-a["cumulative_terminal_reward"],
                           "distilled_minus_random_regret": b["total_terminal_regret"]-a["total_terminal_regret"]})
    passing = [x for x in comparison if x["passed"]]
    selected = sorted(passing, key=lambda x: (-distilled_report[str(x["nominal_coverage"])]["cumulative_terminal_reward"], x["nominal_coverage"]))[0] if passing else None
    result = {"scope": "frozen512 fit-only dense semantic distillation source-group OOF", "gpu_used": False,
              "dev_used": False, "test_used": False, "calibration_or_evaluation_outcomes_read": False,
              "data": {"samples": len(set(data["sample"])), "sources": len(set(data["source"])), "blocks": len(data["errors"]),
                       "candidates": int(np.prod(data["teacher"].shape[:2])), "teacher_width": data["teacher"].shape[2],
                       "informative_blocks": int(data["informative"].sum()), "assembly_seconds": assembly_seconds},
              "empty_policy": SELECTIVE.empty_metrics(data, folds), "random_init": random_report,
              "dense_semantic_distilled": distilled_report, "comparison": comparison,
              "fold_reports": fold_reports, "gate": {"passed": selected is not None, "selected": selected},
              "decision": "eligible for separate calibration" if selected else "stop dense semantic distillation; do not read calibration"}
    (root / "metrics.json").write_text(json.dumps(result, indent=2) + "\n")
    (root / "manifest.json").write_text(json.dumps({"status": "fit_oof_complete", "created_utc": datetime.now(timezone.utc).isoformat(),
        "gate_passed": selected is not None, "calibration_outcomes_read": False, "evaluation_outcomes_read": False,
        "metrics_sha256": sha(root / "metrics.json")}, indent=2) + "\n")
    return result


def main():
    parser = argparse.ArgumentParser(); parser.add_argument("mode", choices=("preregister", "oof")); parser.add_argument("--output-root", type=Path, default=OUTPUT)
    args = parser.parse_args(); os.environ["CUDA_VISIBLE_DEVICES"] = ""; root = args.output_root.resolve()
    result = preregister(root) if args.mode == "preregister" else run_oof(root)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
