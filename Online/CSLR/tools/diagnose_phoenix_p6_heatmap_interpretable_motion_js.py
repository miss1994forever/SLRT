#!/usr/bin/env python3
"""CPU-only, train-fit diagnostic of interpretable heatmap motion versus P6 local JS."""
import argparse
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "results/phoenix-2014t_ISLR"
CONFIG = ROOT / "configs/p6_heatmap_interpretable_motion_js_diagnostic_v1_preregistered.json"
P6 = BASE / "p6_local_js_predictor_oof_v1_49faacc3"
FULL = BASE / "p4_hrnet_positive_mass_fulltrain_v1_49faacc3"
OUTPUT = BASE / "p6_heatmap_interpretable_motion_js_diagnostic_v1_49faacc3"
IDENTITY = OUTPUT / "preregistered_identity.json"
DATA = OUTPUT / "interpretable_features.npz"
METRICS = OUTPUT / "metrics.json"
GROUPS = (("body", 0, 23), ("face", 23, 91),
          ("left_hand", 91, 112), ("right_hand", 112, 133))
MOTION = ("step_mean", "step_std", "step_p90", "step_max", "recent_step_mean",
          "net_displacement_mean", "path_length_mean", "acceleration_mean")
QUALITY = ("trace_current", "trace_history", "trace_delta", "entropy_current",
           "entropy_history", "entropy_delta", "probability_peak_current",
           "probability_peak_history", "probability_peak_delta")


def now():
    return datetime.now(timezone.utc).isoformat()


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_atomic(path, value):
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    os.replace(temporary, path)


def verify_assets():
    config = json.loads(CONFIG.read_text())
    expected = config["fixed_assets"]
    actual = {
        "p6_predictor_dataset_sha256": sha(P6 / "dataset.npz"),
        "heatmap_fulltrain_manifest_sha256": sha(FULL / "manifest.json"),
        "heatmap_assignment_sha256": sha(FULL / "assignment.json"),
    }
    if actual != expected:
        raise RuntimeError(f"asset mismatch: {actual}")
    return config


def prepare():
    config = verify_assets()
    if OUTPUT.exists() and any(OUTPUT.iterdir()):
        if not IDENTITY.is_file():
            raise FileExistsError(f"nonempty output without identity: {OUTPUT}")
        identity = json.loads(IDENTITY.read_text())
        if identity["config_sha256"] != sha(CONFIG):
            raise RuntimeError("configuration changed after prepare")
        return identity
    OUTPUT.mkdir(parents=True, exist_ok=True)
    identity = {
        "created_utc": now(), "config_sha256": sha(CONFIG),
        "script_sha256_at_prepare": sha(Path(__file__)),
        "created_before_diagnostic_outputs": True,
        "outcomes_allowed": ["frozen_train_fit_local_js"],
        "forbidden_outcomes_read": True,
        "scope": config["scope"],
    }
    IDENTITY.write_text(json.dumps(identity, indent=2) + "\n")
    return identity


def full_paths():
    assignment = json.loads((FULL / "assignment.json").read_text())
    result = {}
    for worker, rows in enumerate(assignment["workers"]):
        for row in rows:
            name = row["sample_id"]
            token = hashlib.sha256(name.encode()).hexdigest()[:24]
            result[name] = FULL / "shards" / f"worker-{worker:02d}" / f"{token}.npz"
    return result


def history_indices(start):
    return np.arange(max(0, start - 30), start + 1, dtype=np.int64)


def group_features(statistics, indices, begin, end):
    values = np.asarray(statistics[indices, begin:end], np.float32)
    xy = values[..., :2]
    if len(values) > 1:
        delta = np.diff(xy, axis=0)
        displacement = np.linalg.norm(delta, axis=2)
        step_flat = displacement.reshape(-1)
        recent = float(displacement[-1].mean())
        path = float(displacement.sum(axis=0).mean())
        acceleration = (float(np.linalg.norm(np.diff(delta, axis=0), axis=2).mean())
                        if len(delta) > 1 else 0.0)
    else:
        step_flat = np.zeros(1, np.float32)
        recent = path = acceleration = 0.0
    net = float(np.linalg.norm(xy[-1] - xy[0], axis=1).mean())
    motion = [float(step_flat.mean()), float(step_flat.std()),
              float(np.quantile(step_flat, 0.9)), float(step_flat.max()), recent,
              net, path, acceleration]
    trace = values[..., 2] + values[..., 3]
    entropy = values[..., 7]
    peak = values[..., 6]
    quality = []
    for quantity in (trace, entropy, peak):
        quality.extend([float(quantity[-1].mean()), float(quantity.mean()),
                        float((quantity[-1] - quantity[0]).mean())])
    return motion, quality


def shuffled_motion(statistics, indices, begin, end, name, block_index):
    if len(indices) < 2:
        shuffled = indices
    else:
        token = f"p6-interpretable-motion-shuffle-v1\0{name}\0{block_index}".encode()
        seed = int.from_bytes(hashlib.sha256(token).digest()[:8], "big")
        shuffled = indices[np.random.default_rng(seed).permutation(len(indices))]
    motion, _ = group_features(statistics, shuffled, begin, end)
    return motion


def build():
    prepare()
    if DATA.exists():
        raise FileExistsError(DATA)
    with np.load(P6 / "dataset.npz", allow_pickle=False) as packed:
        sample_names = [str(value) for value in packed["sample_names"].tolist()]
        sample_indices = np.asarray(packed["sample_indices"], np.int64)
        block_indices = np.asarray(packed["block_indices"], np.int64)
    paths = full_paths()
    motion = np.empty((len(sample_indices), 32), np.float32)
    quality = np.empty((len(sample_indices), 36), np.float32)
    shuffled = np.empty((len(sample_indices), 32), np.float32)
    rows_by_sample = [np.flatnonzero(sample_indices == index) for index in range(len(sample_names))]
    for sample_index, (name, rows) in enumerate(zip(sample_names, rows_by_sample)):
        path = paths.get(name)
        if path is None or not path.is_file():
            raise RuntimeError(f"missing heatmap archive: {name}")
        with np.load(path, allow_pickle=False) as packed:
            statistics = np.asarray(packed["statistics"], np.float32)
        for row in rows:
            block = int(block_indices[row]); start = 4 * block
            if start >= len(statistics):
                raise RuntimeError(f"block beyond heatmap sequence: {name} {block}")
            indices = history_indices(start)
            mrow, qrow, xrow = [], [], []
            for _, begin, end in GROUPS:
                m, q = group_features(statistics, indices, begin, end)
                mrow.extend(m); qrow.extend(q)
                xrow.extend(shuffled_motion(statistics, indices, begin, end, name, block))
            motion[row] = mrow; quality[row] = qrow; shuffled[row] = xrow
        if (sample_index + 1) % 250 == 0 or sample_index + 1 == len(sample_names):
            print(json.dumps({"videos": sample_index + 1, "rows": int(sum(len(x) for x in rows_by_sample[:sample_index + 1]))}), flush=True)
    if not (np.isfinite(motion).all() and np.isfinite(quality).all() and np.isfinite(shuffled).all()):
        raise RuntimeError("non-finite interpretable feature")
    motion_names = [f"{group}.{feature}" for group, _, _ in GROUPS for feature in MOTION]
    quality_names = [f"{group}.{feature}" for group, _, _ in GROUPS for feature in QUALITY]
    np.savez_compressed(DATA, motion=motion, quality=quality, shuffled_motion=shuffled,
                        motion_names=np.asarray(motion_names), quality_names=np.asarray(quality_names))
    manifest = {"status": "complete", "created_utc": now(), "rows": len(motion),
                "motion_shape": list(motion.shape), "quality_shape": list(quality.shape),
                "data_sha256": sha(DATA), "reference_or_wer_read": False}
    write_atomic(OUTPUT / "dataset_manifest.json", manifest)
    return manifest


def ridge_predict(train_x, train_y, test_x, alpha=100.0):
    mean = train_x.mean(axis=0, dtype=np.float64)
    scale = train_x.std(axis=0, dtype=np.float64)
    scale[scale < 1e-6] = 1.0
    x = (np.asarray(train_x, np.float64) - mean) / scale
    z = (np.asarray(test_x, np.float64) - mean) / scale
    ymean = float(np.mean(train_y))
    yc = np.asarray(train_y, np.float64) - ymean
    gram = x.T @ x
    gram.flat[::len(gram) + 1] += alpha
    weights = np.linalg.solve(gram, x.T @ yc)
    return (z @ weights + ymean).astype(np.float32)


def rankdata(values):
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), np.float64)
    sorted_values = values[order]
    left = 0
    while left < len(values):
        right = left + 1
        while right < len(values) and sorted_values[right] == sorted_values[left]:
            right += 1
        ranks[order[left:right]] = 0.5 * (left + right - 1)
        left = right
    return ranks


def spearman(left, right):
    a, b = rankdata(np.asarray(left)), rankdata(np.asarray(right))
    if a.std() == 0 or b.std() == 0:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])


def oof(matrix, target, folds):
    prediction = np.empty(len(target), np.float32)
    fold_r2 = {}
    total_sse = total_sst = 0.0
    for fold in range(5):
        train = folds != fold; test = ~train
        prediction[test] = ridge_predict(matrix[train], target[train], matrix[test])
        residual = np.asarray(target[test] - prediction[test], np.float64)
        centered = np.asarray(target[test] - target[train].mean(), np.float64)
        sse = float(residual @ residual); sst = float(centered @ centered)
        total_sse += sse; total_sst += sst
        fold_r2[str(fold)] = 1.0 - sse / sst
    return prediction, {"pooled_r2": 1.0 - total_sse / total_sst,
                        "fold_r2": fold_r2, "spearman": spearman(prediction, target)}


def deciles(feature, raw_js):
    order = np.argsort(feature, kind="mergesort")
    threshold = float(np.quantile(raw_js, 0.75))
    rows = []
    for index, selected in enumerate(np.array_split(order, 10)):
        rows.append({"decile": index + 1, "feature_min": float(feature[selected].min()),
                     "feature_max": float(feature[selected].max()),
                     "mean_raw_js": float(raw_js[selected].mean()),
                     "median_raw_js": float(np.median(raw_js[selected])),
                     "top_js_quartile_rate": float(np.mean(raw_js[selected] >= threshold))})
    means = np.asarray([row["mean_raw_js"] for row in rows])
    direction = np.sign(np.diff(means))
    direction = direction[direction != 0]
    turns = int(np.sum(direction[1:] != direction[:-1])) if len(direction) > 1 else 0
    return rows, turns


def run():
    prepare()
    if METRICS.exists():
        raise FileExistsError(METRICS)
    if not DATA.exists():
        raise RuntimeError("run --build first")
    with np.load(P6 / "dataset.npz", allow_pickle=False) as packed:
        k = np.asarray(packed["K"], np.float32)
        target = np.asarray(packed["target"], np.float32)
        raw_js = np.asarray(packed["raw_js"], np.float32)
        folds = np.asarray(packed["folds"], np.int8)
    with np.load(DATA, allow_pickle=False) as packed:
        motion = np.asarray(packed["motion"], np.float32)
        quality = np.asarray(packed["quality"], np.float32)
        shuffled = np.asarray(packed["shuffled_motion"], np.float32)
        motion_names = [str(x) for x in packed["motion_names"].tolist()]
        quality_names = [str(x) for x in packed["quality_names"].tolist()]
    matrices = {
        "M": motion, "Q": quality, "MQ": np.concatenate([motion, quality], axis=1),
        "K": k, "KM": np.concatenate([k, motion], axis=1),
        "KQ": np.concatenate([k, quality], axis=1),
        "KMQ": np.concatenate([k, motion, quality], axis=1),
        "KMM2": np.concatenate([k, motion, motion * motion], axis=1),
        "KX": np.concatenate([k, shuffled], axis=1),
    }
    model_metrics, predictions = {}, {}
    for name, matrix in matrices.items():
        predictions[name], model_metrics[name] = oof(matrix, target, folds)
        print(json.dumps({"model": name, **model_metrics[name]}), flush=True)
    descriptive = {}
    for index, name in enumerate(motion_names):
        bins, turns = deciles(motion[:, index], raw_js)
        descriptive[name] = {"spearman_raw_js": spearman(motion[:, index], raw_js),
                             "decile_mean_turns": turns, "deciles": bins}
    quality_correlations = {name: spearman(quality[:, index], raw_js)
                            for index, name in enumerate(quality_names)}
    base = model_metrics["K"]["pooled_r2"]
    deltas = {name: value["pooled_r2"] - base for name, value in model_metrics.items() if name != "K"}
    metrics = {
        "status": "complete", "created_utc": now(), "gpu_used": False,
        "train_fit_only": True, "reference_or_wer_read": False,
        "calibration_dev_test_heldout_read": False, "policy_replay_run": False,
        "rows": len(target), "models": model_metrics, "delta_r2_vs_K": deltas,
        "motion_descriptive": descriptive, "quality_spearman_raw_js": quality_correlations,
        "interpretation_status": "diagnostic_only_no_unlock",
    }
    write_atomic(METRICS, metrics)
    np.savez_compressed(OUTPUT / "oof_predictions.npz", folds=folds, target=target, **predictions)
    manifest = {"status": "complete", "config_sha256": sha(CONFIG),
                "identity_sha256": sha(IDENTITY), "dataset_sha256": sha(DATA),
                "metrics_sha256": sha(METRICS),
                "oof_predictions_sha256": sha(OUTPUT / "oof_predictions.npz"),
                "decision": "diagnostic only; no downstream experiment unlocked"}
    write_atomic(OUTPUT / "manifest.json", manifest)
    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--prepare", action="store_true")
    parser.add_argument("--build", action="store_true")
    parser.add_argument("--run", action="store_true")
    args = parser.parse_args()
    if sum((args.prepare, args.build, args.run)) != 1:
        parser.error("choose exactly one of --prepare, --build, --run")
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    result = prepare() if args.prepare else build() if args.build else run()
    print(json.dumps(result, indent=2))
