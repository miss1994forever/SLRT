#!/usr/bin/env python3
"""Fit-only source-group OOF audit of a causal hand-appearance HOG preview."""
import argparse
import gzip
import hashlib
import importlib.util
import io
import json
import math
import os
import pickle
import random
import sys
import time
import zipfile
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parents[1]
FRESH = ROOT / "results/phoenix-2014t_ISLR/p3_partial32_fresh_terminal_listwise_exploratory_v1_49faacc3"
DATA = FRESH.with_name(FRESH.name + "_dataset")
OUTPUT = ROOT / "results/phoenix-2014t_ISLR/p3_partial32_hog_preview_fit512_oof_v1_49faacc3"
ZIP_PATH = REPO / "data/phoenix_2014t/PHOENIX2014T_videos.zip"
KEYPOINTS = REPO / "data/phoenix_2014t/keypoints_hrnet_dark_coco_wholebody_iso.train.pkl"
POSE_ARCHIVE = Path("/tmp/phoenix_p2_train_center_sequences_v1.npz")
SEED = 261023
FOLD_SEED = 261022
FOLDS = 5
EPOCHS = 20
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
PREVIEW = FRESHMOD.PREVIEW
SELECTIVE = imp("analyze_phoenix_partial_selective_residual")
AUDIT = imp("audit_phoenix_new_observable_preview")


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def frozen_ids():
    value = json.loads((FRESH / "resolved_config_preregistered.json").read_text())
    ids = value["fit_512_scaling_subset"]["sample_ids"]
    if len(ids) != 512:
        raise ValueError("frozen subset is not 512 samples")
    return ids


def ids_hash(ids):
    return hashlib.sha256("\n".join(ids).encode()).hexdigest()


def source_fold(source):
    value = hashlib.sha256(f"{FOLD_SEED}\0{source}".encode()).hexdigest()
    return int(value[:16], 16) % FOLDS


def config():
    ids = frozen_ids()
    return {
        "experiment": "fit-only HOG-preview selective residual source-group OOF",
        "created_before_cache_generation_or_oof_outcomes": True,
        "cpu_only": True,
        "subset": {"source": "fresh terminal listwise frozen hash512 fit subset", "samples": 512, "sources": 211, "sample_ids_sha256": ids_hash(ids), "sample_ids": ids},
        "cache": {
            "feature": "pose-guided two-hand appearance HOG",
            "per_frame_shape": [648],
            "dtype": "float16",
            "crop": "each hand current-frame HRNet points only; confidence>=0.2; square 1.5xbbox with minimum 24 pixels; resize 32x32; invalid is zeros",
            "hog": "OpenCV HOG win32 block16 stride8 cell8 bins9; 324 dimensions per hand",
            "causality": "no future fill, reference, labels, full logits, or EOS",
        },
        "candidate_representation": {
            "availability": "16-frame window raw indices clamp(start-7,...,start+8); available when the ISLR candidate window is available",
            "hog_compression": "reshape each hand 324 to 36x9 and mean cells -> 18/frame; concatenate temporal mean,std,last-first,max -> 72/candidate",
            "pose_compression": "existing causal 31x17 pose/hand preview; temporal mean,std,last-first,max -> 68/candidate",
            "base": "49-d bookkeeping plus past decoder prefix",
            "HOG_model_width": 189,
            "no_HOG_model_width": 117,
            "PCA": "none",
            "normalization": "feature mean/std fit inside each OOF training fold only",
        },
        "model": {
            "architecture": "shared candidate MLP Linear(D,48)-GELU-Linear(48,32)-GELU-Linear(32,1); side residual = side score-center score",
            "optimizer": "AdamW(lr=1e-3, weight_decay=1e-4)",
            "epochs": EPOCHS,
            "batch": BATCH,
            "training_blocks": "terminal-informative blocks only; signed center-error minus side-error targets",
            "loss": "SmoothL1; true advantage<=0 and predicted>0 receives 4x cost",
            "false_override_cost": FALSE_OVERRIDE_COST,
            "seed": SEED,
        },
        "oof": {
            "folds": FOLDS,
            "fold_assignment": "exact selective-residual folds: sha256(261022+NUL+source_video_id) mod5",
            "coverages": list(COVERAGES),
            "threshold": "OOF raw max-side score quantile, additionally require score>0",
            "no_HOG_baseline": "same rows, folds, labels, loss, MLP, base and pose summary; only HOG summary removed",
            "pass": "at least one coverage has HOG aggregate reward>0, every-fold reward>=0, >=20 overrides, HOG reward>no-HOG reward, and HOG regret<no-HOG regret",
            "selection": "maximum HOG reward then lower coverage",
        },
        "after_pass": "stop after fit OOF; independent later round may train full fit and read calibration",
        "forbidden": ["GPU/CUDA", "dev", "test", "calibration/evaluation outcomes", "full logits/reference/future/EOS as inputs", "git commit"],
    }


def preregister(root):
    if root.exists():
        raise FileExistsError(root)
    root.mkdir(parents=True)
    (root / "resolved_config_preregistered.json").write_text(json.dumps(config(), indent=2) + "\n")
    return {"status": "preregistered", "root": str(root), "samples": 512, "sources": 211}


def loadcfg(root):
    value = json.loads((root / "resolved_config_preregistered.json").read_text())
    if value != config():
        raise ValueError("preregistered configuration changed")
    return value


def member_name(name, frame):
    return f"images/{name.split('/', 1)[1]}/images{frame + 1:04d}.png"


def build_cache(root):
    cfg = loadcfg(root)
    if (root / "oof_started.marker").exists():
        raise RuntimeError("OOF already started")
    cache = root / "hog_preview_fit512_fp16.npz"
    manifest_path = root / "cache_manifest.json"
    if cache.exists() or manifest_path.exists():
        raise FileExistsError(cache)
    with KEYPOINTS.open("rb") as handle:
        keypoints = pickle.load(handle)
    if any(not str(name).startswith("train/") for name in keypoints):
        raise ValueError("keypoint cache is not train-only")
    ids = cfg["subset"]["sample_ids"]
    hog = cv2.HOGDescriptor((32, 32), (16, 16), (8, 8), (8, 8), 9)
    matrices, offsets = [], [0]
    invalid_hands = invalid_frames = 0
    decoded_bytes = 0
    zip_tick = time.perf_counter(); zf = zipfile.ZipFile(ZIP_PATH); zip_open = time.perf_counter() - zip_tick
    tick = time.perf_counter()
    for number, name in enumerate(ids, 1):
        points = keypoints[name]
        matrix = np.zeros((len(points), 648), np.float16)
        for frame_index, current_points in enumerate(points):
            with zf.open(member_name(name, frame_index)) as handle:
                raw = handle.read(); decoded_bytes += len(raw)
            frame = np.asarray(Image.open(io.BytesIO(raw)).convert("RGB"))
            pair = []
            frame_failures = 0
            for side in ("left", "right"):
                crop, valid = AUDIT.hand_crop(frame, current_points, side)
                if not valid:
                    invalid_hands += 1; frame_failures += 1
                pair.append(hog.compute(cv2.cvtColor(crop, cv2.COLOR_RGB2GRAY)).ravel())
            if frame_failures:
                invalid_frames += 1
            matrix[frame_index] = np.concatenate(pair).astype(np.float16)
        matrices.append(matrix); offsets.append(offsets[-1] + len(matrix))
        if number % 25 == 0 or number == len(ids):
            print(json.dumps({"cache_samples": number, "frames": offsets[-1], "seconds": time.perf_counter() - tick}), flush=True)
    zf.close()
    features = np.concatenate(matrices)
    temp = root / ".hog_preview_fit512_fp16.incomplete.npz"
    np.savez(temp, features=features, video_names=np.asarray(ids), video_offsets=np.asarray(offsets, np.int64))
    temp.replace(cache)
    elapsed = time.perf_counter() - tick
    manifest = {
        "status": "complete",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "samples": len(ids), "sources": len({x.rsplit('-', 1)[0] for x in ids}), "frames": int(len(features)),
        "shape": list(features.shape), "dtype": str(features.dtype), "bytes": cache.stat().st_size, "sha256": sha(cache),
        "zip_open_seconds": zip_open, "extraction_and_write_seconds": elapsed, "decoded_png_bytes": decoded_bytes,
        "invalid_hand_crops": invalid_hands, "total_hand_crops": int(2 * len(features)), "invalid_hand_crop_rate": invalid_hands / (2 * len(features)),
        "frames_with_any_invalid_hand": invalid_frames, "invalid_frame_rate": invalid_frames / len(features),
        "causality": "each row used only same-frame RGB and same-frame HRNet points; invalid hands zeroed without temporal fill",
        "sample_ids_sha256": ids_hash(ids),
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


class HogArchive:
    def __init__(self, root):
        manifest = json.loads((root / "cache_manifest.json").read_text())
        path = root / "hog_preview_fit512_fp16.npz"
        if sha(path) != manifest["sha256"]:
            raise ValueError("HOG cache hash mismatch")
        data = np.load(path)
        self.features = data["features"]
        names = [str(x) for x in data["video_names"]]
        offsets = np.asarray(data["video_offsets"], np.int64)
        self.slices = {name: slice(int(offsets[i]), int(offsets[i + 1])) for i, name in enumerate(names)}

    def summary(self, name, start):
        frames = self.features[self.slices[name]]
        indices = np.clip(np.arange(int(start) - 7, int(start) + 9), 0, len(frames) - 1)
        value = np.asarray(frames[indices], np.float32).reshape(16, 2, 36, 9).mean(2).reshape(16, 18)
        return temporal_summary(value)


def temporal_summary(value):
    value = np.asarray(value, np.float32)
    return np.concatenate([value.mean(0), value.std(0), value[-1] - value[0], value.max(0)]).astype(np.float32)


def load_fit_rows(sample_ids):
    manifest = json.loads((DATA / "dataset_manifest.json").read_text())
    rows = []
    wanted = set(sample_ids)
    for info in manifest["workers"]:
        path = DATA / "workers" / f"worker-{info['worker']:02d}-of-{info['workers']:02d}.jsonl.gz"
        if sha(path) != info["sha256"]:
            raise ValueError("terminal row hash mismatch")
        with gzip.open(path, "rt", encoding="utf-8") as handle:
            for line in handle:
                row = json.loads(line)
                if row["partition"] == "fit" and row["sample_id"] in wanted:
                    rows.append(row)
    if {row["sample_id"] for row in rows} != wanted:
        raise ValueError("fit512 row coverage mismatch")
    return rows


def arrays(rows, pose, hog):
    base_pose, hog_features, errors, sources = [], [], [], []
    for row in rows:
        bp, hp = [], []
        for feature, start in zip(row["features"], row["candidate_starts"]):
            base = np.asarray(feature["bookkeeping"] + feature["prefix"], np.float32)
            bp.append(np.concatenate([base, temporal_summary(pose.history(row["sample_id"], start))]))
            hp.append(hog.summary(row["sample_id"], start))
        base_pose.append(bp); hog_features.append(hp); errors.append(row["label"]["terminal_errors"]); sources.append(row["source_video_id"])
    errors = np.asarray(errors, np.float32)
    return {"base_pose": np.asarray(base_pose, np.float32), "hog": np.asarray(hog_features, np.float32), "errors": errors,
            "advantage": errors[:, 1:2] - errors[:, [0, 2]], "informative": np.any(errors != errors[:, 1:2], axis=1), "source": np.asarray(sources)}


class Scorer(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(width, 48), nn.GELU(), nn.Linear(48, 32), nn.GELU(), nn.Linear(32, 1))

    def forward(self, value):
        batch, choices, width = value.shape
        return self.net(value.reshape(batch * choices, width)).reshape(batch, choices)


def matrix(data, use_hog):
    return np.concatenate([data["base_pose"], data["hog"]], 2) if use_hog else data["base_pose"]


def fit_stats(value, mask):
    selected = value[mask]
    mean = selected.mean((0, 1), dtype=np.float64).astype(np.float32)
    scale = selected.std((0, 1), dtype=np.float64).astype(np.float32)
    scale[scale < 1e-6] = 1
    return mean, scale


def train_one(data, train_mask, use_hog, seed):
    value = matrix(data, use_hog)
    mean, scale = fit_stats(value, train_mask)
    idx = np.flatnonzero(train_mask & data["informative"])
    torch.manual_seed(seed)
    model = Scorer(value.shape[2])
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    generator = torch.Generator().manual_seed(seed)
    history = []
    for _ in range(EPOCHS):
        order = idx[torch.randperm(len(idx), generator=generator).numpy()]
        total = 0.0
        for left in range(0, len(order), BATCH):
            batch = order[left:left + BATCH]
            x = torch.from_numpy((value[batch] - mean) / scale).float()
            score = model(x); pred = score[:, [0, 2]] - score[:, 1:2]
            target = torch.from_numpy(data["advantage"][batch]).float()
            per = F.smooth_l1_loss(pred, target, reduction="none")
            weights = torch.where((target <= 0) & (pred > 0), torch.full_like(per, FALSE_OVERRIDE_COST), torch.ones_like(per))
            loss = (per * weights).mean(); optimizer.zero_grad(); loss.backward(); optimizer.step(); total += float(loss) * len(batch)
        history.append(total / len(order))
    model.eval()
    return model, (mean, scale), history


def predict(model, stats, data, mask, use_hog):
    value = matrix(data, use_hog); idx = np.flatnonzero(mask); out = []
    with torch.no_grad():
        for left in range(0, len(idx), 1024):
            part = idx[left:left + 1024]
            score = model(torch.from_numpy((value[part] - stats[0]) / stats[1]).float())
            out.append((score[:, [0, 2]] - score[:, 1:2]).numpy())
    return idx, np.concatenate(out)


def oof_predictions(data, use_hog):
    folds = np.asarray([source_fold(source) for source in data["source"]], np.int64)
    out = np.full((len(folds), 2), np.nan, np.float32); histories = {}; scales = {}
    for fold in range(FOLDS):
        train_mask, held = folds != fold, folds == fold
        model, stats, history = train_one(data, train_mask, use_hog, SEED + fold)
        idx, score = predict(model, stats, data, held, use_hog); out[idx] = score
        histories[str(fold)] = history
        scales[str(fold)] = {"train_sources": len(set(data["source"][train_mask])), "heldout_sources": len(set(data["source"][held])), "train_blocks": int(train_mask.sum()), "heldout_blocks": int(held.sum()), "informative_train_blocks": int((train_mask & data["informative"]).sum())}
    if np.isnan(out).any():
        raise RuntimeError("incomplete OOF")
    return out, folds, histories, scales


def reports(data, scores, folds):
    out = {}
    for coverage in COVERAGES:
        threshold = SELECTIVE.candidate_threshold(scores, coverage)
        out[str(coverage)] = {"nominal_coverage": coverage, **SELECTIVE.policy_metrics(data, scores, threshold, folds)}
    return out


def run_oof(root):
    cfg = loadcfg(root)
    cache_manifest = json.loads((root / "cache_manifest.json").read_text())
    (root / "oof_started.marker").write_text(datetime.now(timezone.utc).isoformat() + "\n")
    random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED); torch.set_num_threads(min(8, os.cpu_count() or 1)); torch.use_deterministic_algorithms(True)
    ids = cfg["subset"]["sample_ids"]
    rows = load_fit_rows(ids)
    pose = PREVIEW.PreviewArchive(POSE_ARCHIVE); hog = HogArchive(root)
    tick = time.perf_counter(); data = arrays(rows, pose, hog); feature_seconds = time.perf_counter() - tick
    baseline_scores, folds, baseline_history, baseline_scale = oof_predictions(data, False)
    hog_scores, hog_folds, hog_history, hog_scale = oof_predictions(data, True)
    if not np.array_equal(folds, hog_folds):
        raise RuntimeError("fold mismatch")
    baseline = reports(data, baseline_scores, folds); enhanced = reports(data, hog_scores, folds)
    comparison = []
    for coverage in COVERAGES:
        key = str(coverage); h, b = enhanced[key], baseline[key]
        checks = {"aggregate_reward_gt_0": h["cumulative_terminal_reward"] > 0,
                  "every_fold_reward_ge_0": all(v >= 0 for v in h["fold_reward"].values()),
                  "at_least_20_overrides": h["overrides"] >= 20,
                  "reward_gt_no_HOG": h["cumulative_terminal_reward"] > b["cumulative_terminal_reward"],
                  "regret_lt_no_HOG": h["total_terminal_regret"] < b["total_terminal_regret"]}
        comparison.append({"nominal_coverage": coverage, "passed": all(checks.values()), "checks": checks,
                           "HOG_reward_minus_no_HOG": h["cumulative_terminal_reward"] - b["cumulative_terminal_reward"],
                           "HOG_regret_minus_no_HOG": h["total_terminal_regret"] - b["total_terminal_regret"]})
    passing = [x for x in comparison if x["passed"]]
    selected = sorted(passing, key=lambda x: (-enhanced[str(x["nominal_coverage"])]["cumulative_terminal_reward"], x["nominal_coverage"]))[0] if passing else None
    result = {"scope": "frozen512 fit-only HOG-preview source-group OOF", "gpu_used": False, "dev_used": False, "test_used": False,
              "calibration_or_evaluation_outcomes_read": False, "cache": cache_manifest,
              "data": {"samples": len(ids), "sources": len(set(data["source"])), "blocks": len(data["errors"]), "informative_blocks": int(data["informative"].sum()), "feature_assembly_seconds": feature_seconds},
              "empty_policy": SELECTIVE.empty_metrics(data, folds), "no_HOG_baseline": baseline, "HOG_preview": enhanced,
              "comparison": comparison, "gate": {"passed": selected is not None, "selected": selected},
              "fold_scale": {"no_HOG": baseline_scale, "HOG": hog_scale}, "training_history": {"no_HOG": baseline_history, "HOG": hog_history},
              "decision": "eligible for a separate frozen full-fit/calibration round" if selected else "stop HOG-preview route; no calibration read"}
    (root / "metrics.json").write_text(json.dumps(result, indent=2) + "\n")
    (root / "manifest.json").write_text(json.dumps({"status": "fit_oof_complete", "created_utc": datetime.now(timezone.utc).isoformat(), "gate_passed": selected is not None, "calibration_outcomes_read": False, "evaluation_outcomes_read": False, "metrics_sha256": sha(root / "metrics.json"), "cache_sha256": cache_manifest["sha256"]}, indent=2) + "\n")
    return result


def main():
    parser = argparse.ArgumentParser(); parser.add_argument("mode", choices=("preregister", "cache", "oof")); parser.add_argument("--output-root", type=Path, default=OUTPUT); args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = ""; root = args.output_root.resolve()
    result = preregister(root) if args.mode == "preregister" else build_cache(root) if args.mode == "cache" else run_oof(root)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
