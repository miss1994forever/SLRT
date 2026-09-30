#!/usr/bin/env python3
"""Preregistered source-disjoint ridge probe for CSC-L8 semantic curvature."""
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "results/phoenix-2014t_ISLR"
CONFIG = ROOT / "configs/p4_causal_semantic_coverage_l8_a2_probe_preregistered.json"
A1 = BASE / "p4_csc_l8_target_audit_v1_49faacc3"
TARGETS = A1 / "target_scores.npz"
HISTORY = (BASE / "p3_fulltrain_decoder_conditioned_value_oof_v1_49faacc3"
           / "paid_decoder_history_blocks.npz")
POSE = (BASE / "p3_fulltrain_causal_pose_hand_motion_v1_49faacc3"
        / "train_causal_pose_hand_motion_sequences.npz")
FINGER = BASE / "p3_fulltrain_finger_rgb_tinycnn_robust_predictor_oof_v1_49faacc3"
FACE = BASE / "p3_fulltrain_face_rgb_tinycnn_robust_predictor_oof_v1_49faacc3"
INDEX = FINGER / "ssl_crop_index.npz"
OUTPUT = BASE / "p4_csc_l8_identifiability_probe_v1_49faacc3"
OOF = OUTPUT / "oof_predictions.npz"
METRICS = OUTPUT / "metrics.json"
MANIFEST = OUTPUT / "manifest.json"
HISTORY_LENGTH = 31
VISIBLE_OFFSET = 8
TEMPORAL_FEATURE_NAMES = [
    "left_valid_fraction", "right_valid_fraction",
    "left_mean_confidence", "right_mean_confidence",
    "left_centroid_relative_shoulder_x", "left_centroid_relative_shoulder_y",
    "right_centroid_relative_shoulder_x", "right_centroid_relative_shoulder_y",
    "interhand_distance", "pose_motion", "left_global_motion",
    "right_global_motion", "left_shape_change", "right_shape_change",
    "interhand_distance_change", "left_confidence_change",
    "right_confidence_change",
]


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def summary(values):
    return np.concatenate([values[:, -1], values.mean(1), values.std(1),
                           values[:, -1] - values[:, 0]], axis=1).astype(np.float32)


def build_k(block_indices, fit_names):
    # Only identity and decision time are accessed; utility arrays in HISTORY remain unread.
    with np.load(HISTORY, allow_pickle=False) as packed:
        arrivals = np.asarray(packed["decision_arrivals"], np.int64)[block_indices]
        sample_indices = np.asarray(packed["sample_indices"], np.int64)[block_indices]
    with np.load(POSE, allow_pickle=False) as packed:
        all_features = np.asarray(packed["features"], np.float16)
        names = [str(value) for value in packed["video_names"].tolist()]
        offsets = np.asarray(packed["video_offsets"], np.int64)
        feature_names = [str(value) for value in packed["feature_names"].tolist()]
    temporal = np.asarray([feature_names.index(name) for name in TEMPORAL_FEATURE_NAMES], np.int64)
    slices = {name: (int(offsets[i]), int(offsets[i + 1])) for i, name in enumerate(names)}
    output = np.empty((len(block_indices), 4 * len(temporal)), np.float32)
    for sample_index, name in enumerate(fit_names):
        rows = np.flatnonzero(sample_indices == sample_index)
        if not len(rows):
            continue
        begin, end = slices[name]
        length = end - begin
        visible = np.minimum(length - 1, arrivals[rows] + VISIBLE_OFFSET)
        history = visible[:, None] - np.arange(HISTORY_LENGTH - 1, -1, -1)[None, :]
        history = np.maximum(history, 0) + begin
        values = np.asarray(all_features[history[..., None], temporal], np.float32)
        output[rows] = summary(values)
    if not np.isfinite(output).all():
        raise RuntimeError("nonfinite K probe feature")
    return output


def projection_matrix(seed):
    rng = np.random.default_rng(seed)
    bucket = rng.integers(0, 64, size=396)
    sign = rng.choice(np.asarray([-1.0, 1.0], np.float32), size=396)
    matrix = np.zeros((396, 64), np.float32)
    matrix[np.arange(396), bucket] = sign
    occupancy = np.bincount(bucket, minlength=64).astype(np.float32)
    matrix /= np.sqrt(np.maximum(occupancy, 1.0))[None, :]
    return matrix


def fold_matrix(fold, common_k, candidate_starts, sample_indices, offsets, projection, batch=2048):
    finger = np.load(FINGER / f"finger_frame_embeddings_fold{fold}_float16.npy", mmap_mode="r")
    face = np.load(FACE / f"face_frame_embeddings_fold{fold}_float16.npy", mmap_mode="r")
    output = np.empty((len(candidate_starts), 260), np.float32)
    output[:, :68] = common_k
    starts3 = np.stack([candidate_starts[:, 0], candidate_starts[:, 0] + 1,
                        candidate_starts[:, 1]], axis=1).astype(np.int64)
    lengths = np.diff(offsets)
    frame_delta = np.arange(-7, 9, dtype=np.int64)
    for left in range(0, len(output), batch):
        right = min(left + batch, len(output))
        sample = sample_indices[left:right]
        local = starts3[left:right, :, None] + frame_delta[None, None, :]
        local = np.maximum(local, 0)
        local = np.minimum(local, lengths[sample, None, None] - 1)
        global_index = offsets[sample, None, None] + local
        values = np.concatenate([
            np.asarray(finger[global_index], np.float32),
            np.asarray(face[global_index], np.float32),
        ], axis=-1)
        descriptor = values.reshape(len(sample), 3, 4, 4, 99).mean(axis=3).reshape(len(sample), 3, 396)
        projected = descriptor @ projection
        output[left:right, 68:] = projected.reshape(len(sample), 192)
    return output


def ridge_oof(matrix, target, train, evaluate, alpha):
    mean = matrix[train].mean(axis=0, dtype=np.float64).astype(np.float32)
    scale = matrix[train].std(axis=0, dtype=np.float64).astype(np.float32)
    scale[scale < 1e-6] = 1.0
    xtrain = np.asarray((matrix[train] - mean) / scale, np.float64)
    xeval = np.asarray((matrix[evaluate] - mean) / scale, np.float64)
    ymean = target[train].mean(axis=0, dtype=np.float64)
    ytrain = np.asarray(target[train] - ymean, np.float64)
    gram = xtrain.T @ xtrain
    gram.flat[::len(gram) + 1] += alpha
    weights = np.linalg.solve(gram, xtrain.T @ ytrain)
    return (xeval @ weights + ymean).astype(np.float32), ymean.astype(np.float32)


def main():
    config = json.loads(CONFIG.read_text())
    if sha256_file(A1 / "manifest.json") != config["a1_manifest_sha256"]:
        raise RuntimeError("A1 identity mismatch")
    a1_metrics = json.loads((A1 / "metrics.json").read_text())
    if not a1_metrics["gate_passed"]:
        raise RuntimeError("A1 gate did not unlock A2")
    if OUTPUT.exists():
        raise FileExistsError(f"non-overwriting probe output already exists: {OUTPUT}")
    with np.load(TARGETS, allow_pickle=False) as packed:
        block_indices = np.asarray(packed["block_indices"], np.int64)
        folds = np.asarray(packed["folds"], np.int8)
        source_indices = np.asarray(packed["source_indices"], np.int16)
        sample_indices = np.asarray(packed["sample_indices"], np.int64)
        candidate_starts = np.asarray(packed["candidate_starts"], np.int64)
        target = np.asarray(packed["js_t1"], np.float32)
        winner = np.asarray(packed["winners"], np.int8)
    with np.load(INDEX, allow_pickle=False) as packed:
        fit_names = [str(value) for value in packed["video_names"].tolist()]
        offsets = np.asarray(packed["video_offsets"], np.int64)
    if len(np.unique(source_indices)) != 578 or set(np.unique(folds)) != set(range(5)):
        raise RuntimeError("fold/source coverage mismatch")
    common_k = build_k(block_indices, fit_names)
    projection = projection_matrix(261110)
    prediction = np.empty_like(target)
    fold_metrics = {}
    total_sse = 0.0
    total_sst = 0.0
    for fold in range(5):
        matrix = fold_matrix(fold, common_k, candidate_starts, sample_indices, offsets, projection)
        train = folds != fold
        evaluate = folds == fold
        predicted, baseline = ridge_oof(matrix, target, train, evaluate,
                                        float(config["model"]["alpha"]))
        prediction[evaluate] = predicted
        residual = np.asarray(target[evaluate] - predicted, np.float64)
        centered = np.asarray(target[evaluate] - baseline, np.float64)
        sse = float(np.sum(residual * residual))
        sst = float(np.sum(centered * centered))
        total_sse += sse
        total_sst += sst
        fold_metrics[str(fold)] = {
            "blocks": int(evaluate.sum()),
            "r2": float(1.0 - sse / sst),
            "winner_accuracy": float(np.mean(predicted.argmax(axis=1) + 1 == winner[evaluate])),
        }
        del matrix
    pooled_r2 = float(1.0 - total_sse / total_sst)
    winner_accuracy = float(np.mean(prediction.argmax(axis=1) + 1 == winner))
    positive_folds = sum(value["r2"] > 0 for value in fold_metrics.values())
    gate = config["gate"]
    checks = {
        "pooled_r2": pooled_r2 >= float(gate["pooled_r2_min"]),
        "winner_accuracy": winner_accuracy >= float(gate["winner_accuracy_min"]),
        "positive_r2_folds": positive_folds >= int(gate["positive_r2_folds_min"]),
    }
    metrics = {
        "experiment": config["experiment"],
        "phase": "A2 outcome-blind source-disjoint identifiability",
        "gpu_used": False,
        "robust_reward_read": False,
        "reference_or_wer_read": False,
        "future_continuation_read": False,
        "true_eos_read": False,
        "dev_or_test_read": False,
        "blocks": len(target),
        "sources": len(np.unique(source_indices)),
        "pooled_r2": pooled_r2,
        "winner_accuracy": winner_accuracy,
        "positive_r2_folds": positive_folds,
        "folds": fold_metrics,
        "checks": checks,
        "gate_passed": bool(all(checks.values())),
        "next_step": ("preregister formal CSC-L8 predictor OOF without reading policy outcome" if all(checks.values())
                      else "stop CSC-L8-v1 without reading train policy, closed-loop, or held-out outcomes"),
    }
    OUTPUT.mkdir(parents=True)
    np.savez_compressed(OOF, prediction=prediction, target=target, winner=winner,
                        folds=folds, source_indices=source_indices)
    METRICS.write_text(json.dumps(metrics, indent=2) + "\n")
    manifest = {
        "status": "complete",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "config_sha256": sha256_file(CONFIG),
        "a1_manifest_sha256": sha256_file(A1 / "manifest.json"),
        "metrics_sha256": sha256_file(METRICS),
        "oof_sha256": sha256_file(OOF),
        "gate_passed": metrics["gate_passed"],
    }
    MANIFEST.write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
