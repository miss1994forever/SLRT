#!/usr/bin/env python3
"""Preregistered CSC-L8-v2 heatmap-U source-disjoint identifiability probe."""
import argparse
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

import analyze_phoenix_p4_csc_l8_identifiability_probe as prior


ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "results/phoenix-2014t_ISLR"
CONFIG = ROOT / "configs/p4_csc_l8_heatmap_u_identifiability_v2_preregistered.json"
FULL = BASE / "p4_hrnet_positive_mass_fulltrain_v1_49faacc3"
OUTPUT = BASE / "p4_csc_l8_heatmap_u_identifiability_v2_49faacc3"
IDENTITY = OUTPUT / "preregistered_identity.json"
FRAME_FEATURES = OUTPUT / "u_frame_features_float32.npy"
FRAME_INDEX = OUTPUT / "u_frame_feature_index.npz"
FRAME_MANIFEST = OUTPUT / "u_frame_feature_manifest.json"
BUILD_PROGRESS = OUTPUT / "u_frame_build_progress.json"
BLOCK_FEATURES = OUTPUT / "u_block_projection_float32.npy"
OOF = OUTPUT / "oof_predictions.npz"
METRICS = OUTPUT / "metrics.json"
MANIFEST = OUTPUT / "manifest.json"
GROUPS = [("body", 0, 23), ("face", 23, 91),
          ("left_hand", 91, 112), ("right_hand", 112, 133)]
QUANTITIES = ["normalized_covariance_trace", "normalized_covariance_anisotropy",
              "entropy", "probability_peak", "log1p_positive_mass", "raw_peak"]
POOLS = ["mean", "standard_deviation", "minimum", "maximum"]


def utcnow():
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json_atomic(path, value):
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    os.replace(temporary, path)


def load_fit_index():
    with np.load(prior.INDEX, allow_pickle=False) as packed:
        names = [str(value) for value in packed["video_names"].tolist()]
        offsets = np.asarray(packed["video_offsets"], np.int64)
    return names, offsets


def full_paths():
    assignment = json.loads((FULL / "assignment.json").read_text())
    mapping = {}
    for worker, rows in enumerate(assignment["workers"]):
        for row in rows:
            name = row["sample_id"]
            token = hashlib.sha256(name.encode()).hexdigest()[:24]
            mapping[name] = FULL / "shards" / f"worker-{worker:02d}" / f"{token}.npz"
    return mapping


def prepare():
    if OUTPUT.exists():
        raise FileExistsError(f"non-overwriting probe identity exists: {OUTPUT}")
    full_manifest = FULL / "manifest.json"
    full_metrics = FULL / "metrics.json"
    if not json.loads(full_manifest.read_text())["gate_passed"]:
        raise RuntimeError("full heatmap archive did not pass completion gates")
    if json.loads(full_metrics.read_text())["outcomes_read"] is not False:
        raise RuntimeError("full heatmap archive leakage marker failure")
    names, offsets = load_fit_index()
    paths = full_paths()
    missing = [name for name in names if name not in paths or not paths[name].is_file()]
    if missing:
        raise RuntimeError(f"missing heatmap samples: {missing[:5]}")
    lengths = np.diff(offsets)
    for index, name in enumerate(names):
        with np.load(paths[name], allow_pickle=False) as packed:
            if int(packed["frame_count"].item()) != int(lengths[index]):
                raise RuntimeError(f"frame length mismatch: {name}")
    OUTPUT.mkdir(parents=True)
    identity = {
        "created_utc": utcnow(), "config_sha256": sha256_file(CONFIG),
        "fulltrain_manifest_sha256": sha256_file(full_manifest),
        "fulltrain_metrics_sha256": sha256_file(full_metrics),
        "csc_target_manifest_sha256": sha256_file(prior.A1 / "manifest.json"),
        "prior_K_probe_script_sha256": sha256_file(Path(prior.__file__)),
        "fit_videos": len(names), "fit_frames": int(offsets[-1]),
        "outcomes_read": False,
    }
    IDENTITY.write_text(json.dumps(identity, indent=2) + "\n")
    print(json.dumps(identity, indent=2))


def frame_descriptor(statistics):
    value = np.asarray(statistics, np.float32)
    if value.ndim != 3 or value.shape[1:] != (133, 11):
        raise RuntimeError(f"unexpected heatmap statistic shape: {value.shape}")
    varx, vary, cov = value[..., 2], value[..., 3], value[..., 4]
    trace = varx + vary
    anisotropy = np.sqrt(np.maximum((varx - vary) ** 2 + 4.0 * cov ** 2, 0.0)) / (trace + 1e-8)
    quantities = np.stack([trace, anisotropy, value[..., 7], value[..., 6],
                           np.log1p(value[..., 8]), value[..., 5]], axis=2)
    output = []
    for _, begin, end in GROUPS:
        group = np.asarray(quantities[:, begin:end], np.float32)
        output.extend([
            group.mean(axis=1, dtype=np.float64).astype(np.float32),
            group.std(axis=1, dtype=np.float64).astype(np.float32),
            group.min(axis=1), group.max(axis=1),
        ])
    descriptor = np.concatenate(output, axis=1)
    if descriptor.shape != (len(value), 96) or not np.isfinite(descriptor).all():
        raise RuntimeError("invalid U frame descriptor")
    return descriptor


def build_frames():
    if not IDENTITY.exists():
        raise RuntimeError("run --prepare first")
    names, offsets = load_fit_index()
    paths = full_paths()
    feature_names = [f"{group}.{quantity}.{pool}" for group, _, _ in GROUPS
                     for pool in POOLS for quantity in QUANTITIES]
    if len(feature_names) != 96:
        raise RuntimeError("U feature name dimension failure")
    if FRAME_MANIFEST.exists():
        print(FRAME_MANIFEST.read_text())
        return
    if FRAME_FEATURES.exists():
        features = np.lib.format.open_memmap(FRAME_FEATURES, mode="r+")
        progress = json.loads(BUILD_PROGRESS.read_text())
        start = int(progress["videos_complete"])
    else:
        features = np.lib.format.open_memmap(
            FRAME_FEATURES, mode="w+", dtype=np.float32, shape=(int(offsets[-1]), 96))
        start = 0
        write_json_atomic(BUILD_PROGRESS, {"status": "running", "videos_complete": 0,
                                           "videos_total": len(names), "frames_complete": 0})
    for index in range(start, len(names)):
        name = names[index]
        begin, end = int(offsets[index]), int(offsets[index + 1])
        with np.load(paths[name], allow_pickle=False) as packed:
            features[begin:end] = frame_descriptor(packed["statistics"])
        features.flush()
        write_json_atomic(BUILD_PROGRESS, {
            "status": "running", "updated_utc": utcnow(), "videos_complete": index + 1,
            "videos_total": len(names), "frames_complete": end, "frames_total": int(offsets[-1]),
            "current_sample": name,
        })
    del features
    np.savez_compressed(FRAME_INDEX, video_names=np.asarray(names), video_offsets=offsets,
                        feature_names=np.asarray(feature_names))
    manifest = {
        "status": "complete", "created_utc": utcnow(), "videos": len(names),
        "frames": int(offsets[-1]), "shape": [int(offsets[-1]), 96], "dtype": "float32",
        "features_sha256": sha256_file(FRAME_FEATURES), "index_sha256": sha256_file(FRAME_INDEX),
        "identity_sha256": sha256_file(IDENTITY), "outcomes_read": False,
    }
    FRAME_MANIFEST.write_text(json.dumps(manifest, indent=2) + "\n")
    write_json_atomic(BUILD_PROGRESS, {"status": "complete", "finished_utc": utcnow(),
                                       "videos_complete": len(names), "frames_complete": int(offsets[-1])})
    print(json.dumps(manifest, indent=2))


def projection_matrix(seed):
    rng = np.random.default_rng(seed)
    bucket = rng.integers(0, 64, size=384)
    sign = rng.choice(np.asarray([-1.0, 1.0], np.float32), size=384)
    matrix = np.zeros((384, 64), np.float32)
    matrix[np.arange(384), bucket] = sign
    occupancy = np.bincount(bucket, minlength=64).astype(np.float32)
    matrix /= np.sqrt(np.maximum(occupancy, 1.0))[None, :]
    return matrix


def build_block_u(block_indices, batch=4096):
    if BLOCK_FEATURES.exists():
        return np.load(BLOCK_FEATURES, mmap_mode="r")
    _, offsets = load_fit_index()
    with np.load(prior.HISTORY, allow_pickle=False) as packed:
        arrivals = np.asarray(packed["decision_arrivals"], np.int64)[block_indices]
        sample_indices = np.asarray(packed["sample_indices"], np.int64)[block_indices]
    lengths = np.diff(offsets)
    visible = np.minimum(lengths[sample_indices] - 1, arrivals + prior.VISIBLE_OFFSET)
    frames = np.load(FRAME_FEATURES, mmap_mode="r")
    output = np.lib.format.open_memmap(
        BLOCK_FEATURES, mode="w+", dtype=np.float32, shape=(len(block_indices), 64))
    projection = projection_matrix(261121)
    history_delta = np.arange(prior.HISTORY_LENGTH - 1, -1, -1, dtype=np.int64)
    for left in range(0, len(output), batch):
        right = min(left + batch, len(output))
        local = np.maximum(visible[left:right, None] - history_delta[None, :], 0)
        global_index = offsets[sample_indices[left:right], None] + local
        values = np.asarray(frames[global_index], np.float32)
        summary = np.concatenate([values[:, -1], values.mean(axis=1), values.std(axis=1),
                                  values[:, -1] - values[:, 0]], axis=1).astype(np.float32)
        output[left:right] = summary @ projection
    output.flush()
    del output
    return np.load(BLOCK_FEATURES, mmap_mode="r")


def evaluate(matrix, target, winner, folds, alpha):
    prediction = np.empty_like(target)
    fold_metrics = {}
    total_sse = 0.0
    total_sst = 0.0
    for fold in range(5):
        train, evaluate_rows = folds != fold, folds == fold
        predicted, baseline = prior.ridge_oof(matrix, target, train, evaluate_rows, alpha)
        prediction[evaluate_rows] = predicted
        residual = np.asarray(target[evaluate_rows] - predicted, np.float64)
        centered = np.asarray(target[evaluate_rows] - baseline, np.float64)
        sse = float(np.sum(residual * residual))
        sst = float(np.sum(centered * centered))
        total_sse += sse
        total_sst += sst
        fold_metrics[str(fold)] = {
            "blocks": int(evaluate_rows.sum()), "r2": float(1.0 - sse / sst),
            "winner_accuracy": float(np.mean(predicted.argmax(axis=1) + 1 == winner[evaluate_rows])),
        }
    return prediction, {
        "pooled_r2": float(1.0 - total_sse / total_sst),
        "winner_accuracy": float(np.mean(prediction.argmax(axis=1) + 1 == winner)),
        "positive_r2_folds": int(sum(row["r2"] > 0 for row in fold_metrics.values())),
        "folds": fold_metrics,
    }


def probe():
    if METRICS.exists() or OOF.exists():
        raise FileExistsError("non-overwriting probe result exists")
    config = json.loads(CONFIG.read_text())
    if not json.loads(FRAME_MANIFEST.read_text())["outcomes_read"] is False:
        raise RuntimeError("U feature leakage marker failure")
    with np.load(prior.TARGETS, allow_pickle=False) as packed:
        block_indices = np.asarray(packed["block_indices"], np.int64)
        folds = np.asarray(packed["folds"], np.int8)
        source_indices = np.asarray(packed["source_indices"], np.int16)
        target = np.asarray(packed["js_t1"], np.float32)
        winner = np.asarray(packed["winners"], np.int8)
    names, _ = load_fit_index()
    matrix_k = prior.build_k(block_indices, names)
    matrix_u = np.asarray(build_block_u(block_indices), np.float32)
    matrix_ku = np.concatenate([matrix_k, matrix_u], axis=1)
    alpha = float(config["model"]["alpha"])
    predictions = {}
    model_metrics = {}
    for name, matrix in (("K", matrix_k), ("U", matrix_u), ("KU", matrix_ku)):
        predictions[name], model_metrics[name] = evaluate(matrix, target, winner, folds, alpha)
    delta_r2 = model_metrics["KU"]["pooled_r2"] - model_metrics["K"]["pooled_r2"]
    delta_accuracy = model_metrics["KU"]["winner_accuracy"] - model_metrics["K"]["winner_accuracy"]
    gates = config["primary_gates"]
    checks = {
        "KU_pooled_r2": model_metrics["KU"]["pooled_r2"] >= float(gates["KU_pooled_r2_at_least"]),
        "KU_winner_accuracy": model_metrics["KU"]["winner_accuracy"] >= float(gates["KU_winner_accuracy_at_least"]),
        "KU_positive_r2_folds": model_metrics["KU"]["positive_r2_folds"] >= int(gates["KU_positive_r2_folds_at_least"]),
        "KU_minus_K_pooled_r2": delta_r2 >= float(gates["KU_minus_K_pooled_r2_at_least"]),
        "KU_minus_K_winner_accuracy": delta_accuracy >= float(gates["KU_minus_K_winner_accuracy_at_least"]),
    }
    metrics = {
        "experiment": config["experiment"], "created_utc": utcnow(),
        "phase": "outcome-blind new-observation source-disjoint identifiability",
        "gpu_used": False, "train_only": True, "robust_reward_read": False,
        "reference_or_wer_read": False, "future_continuation_read": False,
        "true_eos_read": False, "dev_or_test_read": False,
        "blocks": len(target), "sources": len(np.unique(source_indices)),
        "models": model_metrics,
        "KU_minus_K": {"pooled_r2": delta_r2, "winner_accuracy": delta_accuracy},
        "checks": checks, "gate_passed": bool(all(checks.values())),
        "next_step": ("preregister formal CSC-L8 heatmap-U predictor OOF without policy outcome"
                      if all(checks.values()) else
                      "stop heatmap-U CSC-L8 branch before policy outcome, closed loop, or held-out evaluation"),
    }
    np.savez_compressed(OOF, prediction_K=predictions["K"], prediction_U=predictions["U"],
                        prediction_KU=predictions["KU"], target=target, winner=winner,
                        folds=folds, source_indices=source_indices)
    METRICS.write_text(json.dumps(metrics, indent=2) + "\n")
    manifest = {
        "status": "complete", "created_utc": utcnow(), "config_sha256": sha256_file(CONFIG),
        "identity_sha256": sha256_file(IDENTITY), "frame_manifest_sha256": sha256_file(FRAME_MANIFEST),
        "block_features_sha256": sha256_file(BLOCK_FEATURES), "oof_sha256": sha256_file(OOF),
        "metrics_sha256": sha256_file(METRICS), "gate_passed": metrics["gate_passed"],
    }
    MANIFEST.write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(metrics, indent=2))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prepare", action="store_true")
    parser.add_argument("--build-u", action="store_true")
    parser.add_argument("--probe", action="store_true")
    args = parser.parse_args()
    if args.prepare:
        prepare()
    elif args.build_u:
        build_frames()
    elif args.probe:
        probe()
    else:
        parser.error("choose --prepare, --build-u, or --probe")


if __name__ == "__main__":
    main()
