#!/usr/bin/env python3
"""CPU-only source-disjoint H4 schedule-ranker OOF."""
import argparse
import gzip
import hashlib
import importlib.util
import json
import os
import sys
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "results/phoenix-2014t_ISLR"
DENSE = BASE / "train_dense_stride1_v1_49faacc3"
LABELS = BASE / "p5_h4_schedule_label_dataset_v1_49faacc3"
U_ROOT = BASE / "p4_csc_l8_heatmap_u_identifiability_v2_49faacc3"
OUTPUT = BASE / "p5_h4_schedule_ranker_oof_v1_49faacc3"
CONFIG = ROOT / "configs/p5_h4_schedule_ranker_oof_v1_preregistered.json"
DATASET = OUTPUT / "h4_ranker_dataset.npz"
SEED_BOOT = 261133
CENTER = 40


def imp(name):
    path = ROOT / f"tools/{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


P5 = imp("analyze_phoenix_p5_h4_robust_joint_schedule_oracle")
CHRON, BUILDER = P5.CHRON, P5.BUILDER


def utcnow():
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def projection(rows, columns, seed):
    rng = np.random.default_rng(seed)
    bucket = rng.integers(0, columns, size=rows)
    sign = rng.choice(np.asarray([-1.0, 1.0], np.float32), size=rows)
    matrix = np.zeros((rows, columns), np.float32)
    matrix[np.arange(rows), bucket] = sign
    occupancy = np.bincount(bucket, minlength=columns).astype(np.float32)
    matrix /= np.sqrt(np.maximum(occupancy, 1.0))[None, :]
    return matrix


def check_assets(config):
    paths = {
        "label_metrics_sha256": LABELS / "metrics.json",
        "label_manifest_sha256": LABELS / "manifest.json",
        "heatmap_U_frames_sha256": U_ROOT / "u_frame_features_float32.npy",
        "heatmap_U_index_sha256": U_ROOT / "u_frame_feature_index.npz",
        "dense_protocol_sha256": DENSE / "protocol_manifest.json",
    }
    for key, path in paths.items():
        if sha256_file(path) != config["fixed_assets"][key]:
            raise RuntimeError(f"frozen asset mismatch: {key}")
    if not json.loads((LABELS / "metrics.json").read_text())["gate_passed"]:
        raise RuntimeError("label dataset gate did not pass")


def prepare():
    config = json.loads(CONFIG.read_text())
    check_assets(config)
    if OUTPUT.exists() and any(OUTPUT.iterdir()):
        path = OUTPUT / "preregistered_identity.json"
        if not path.is_file():
            raise FileExistsError(f"nonempty output lacks identity: {OUTPUT}")
        identity = json.loads(path.read_text())
        if identity["config_sha256"] != sha256_file(CONFIG):
            raise RuntimeError("config changed after prepare")
        return identity
    OUTPUT.mkdir(parents=True, exist_ok=True)
    identity = {
        "created_utc": utcnow(), "config_sha256": sha256_file(CONFIG),
        "script_sha256_at_prepare": sha256_file(Path(__file__)),
        "created_before_predictor_outcomes": True,
    }
    (OUTPUT / "preregistered_identity.json").write_text(json.dumps(identity, indent=2) + "\n")
    return identity


def label_path(index, shard_count):
    return LABELS / "shards" / f"labels-{index:05d}-of-{shard_count:05d}.jsonl.gz"


def dct_basis(length=31, coefficients=8):
    n = np.arange(length, dtype=np.float64)[None, :]
    k = np.arange(coefficients, dtype=np.float64)[:, None]
    basis = np.cos(np.pi * (n + 0.5) * k / length)
    basis[0] *= np.sqrt(1.0 / length)
    basis[1:] *= np.sqrt(2.0 / length)
    return basis.astype(np.float32)


def u_history(frames, start):
    local = np.maximum(start - np.arange(30, -1, -1, dtype=np.int64), 0)
    return np.asarray(frames[local], np.float32)


def shuffle_history(value, sample_id, horizon_index):
    token = f"p5-h4-u-shuffle-v1\0{sample_id}\0{horizon_index}".encode()
    seed = int.from_bytes(hashlib.sha256(token).digest()[:8], "big")
    return value[np.random.default_rng(seed).permutation(len(value))]


def dense_frame_features(probabilities, blank, token_bucket, token_sign):
    count = len(probabilities)
    output = np.zeros((count, 36), np.float32)
    top = np.argpartition(probabilities, -8, axis=1)[:, -8:]
    top_prob = np.take_along_axis(probabilities, top, axis=1)
    for row in range(count):
        output[row, :32] = np.bincount(
            token_bucket[top[row]], weights=top_prob[row] * token_sign[top[row]], minlength=32)
    ordered = np.sort(top_prob, axis=1)
    output[:, 32] = -np.sum(probabilities * np.log(np.maximum(probabilities, 1e-12)), axis=1)
    output[:, 33] = probabilities[:, blank]
    output[:, 34] = ordered[:, -1]
    output[:, 35] = ordered[:, -1] - ordered[:, -2]
    return output


def k_history(frame_features, selected, current, previous_offsets):
    visible = (selected + [current])[-16:]
    values = np.asarray(frame_features[visible], np.float32)
    pooled = np.concatenate([values[-1], values.mean(0), values.std(0), values[-1] - values[0]])
    counts = np.asarray([previous_offsets.count(value) for value in (1, 2, 3)], np.float32)
    counts /= max(1.0, float(len(previous_offsets)))
    last = np.zeros(4, np.float32)
    for index, value in enumerate(previous_offsets[-4:]):
        last[4 - len(previous_offsets[-4:]) + index] = value - 2
    bookkeeping = np.concatenate([
        np.asarray([np.log1p(current), np.log1p(len(selected))], np.float32), counts, last])
    return np.concatenate([pooled, bookkeeping]).astype(np.float32)


def build_dataset():
    prepare()
    if DATASET.exists():
        raise FileExistsError(DATASET)
    with np.load(U_ROOT / "u_frame_feature_index.npz", allow_pickle=False) as packed:
        u_names = [str(value) for value in packed["video_names"].tolist()]
        u_offsets = np.asarray(packed["video_offsets"], np.int64)
    u_index = {name: index for index, name in enumerate(u_names)}
    u_frames = np.load(U_ROOT / "u_frame_features_float32.npy", mmap_mode="r")
    summary_projection = projection(384, 64, 261131)
    ordered_projection = projection(768, 256, 261132)
    basis = dct_basis()
    vocab = json.loads(CHRON.DEFAULT_VOCAB.read_text())
    blank = vocab.index("<blank>")
    rng = np.random.default_rng(261130)
    token_bucket = rng.integers(0, 32, size=len(vocab))
    token_sign = rng.choice(np.asarray([-1.0, 1.0], np.float32), size=len(vocab))
    matrices = {"K": [], "S": [], "O": [], "X": []}
    targets, folds, sources, refs, sample_ids, horizon_ids = [], [], [], [], [], []
    source_map = {}
    indices, shard_count = CHRON.completed_shard_indices(DENSE)
    started = time.perf_counter()
    for shard in indices:
        label_rows = {}
        with gzip.open(label_path(shard, shard_count), "rt", encoding="utf-8") as handle:
            for line in handle:
                row = json.loads(line); label_rows[row["sample_id"]] = row
        results, logits, _ = BUILDER.BUILDER.validate_dense_shard(
            DENSE, shard, shard_count, verify_hashes=False)
        for name, row in label_rows.items():
            probabilities = BUILDER.ORACLE.softmax_rows(np.asarray(logits[name]))
            frame_k = dense_frame_features(probabilities, blank, token_bucket, token_sign)
            ui = u_index[name]; ub, ue = int(u_offsets[ui]), int(u_offsets[ui + 1])
            if ue - ub != len(probabilities):
                raise RuntimeError(f"U/dense length mismatch: {name}")
            local_u = u_frames[ub:ue]
            selected = []
            previous_offsets = []
            source_index = source_map.setdefault(row["source_video_id"], len(source_map))
            for horizon in row["horizons"]:
                blocks = horizon["block_starts"]
                if horizon["horizon_size"] == 4:
                    start = blocks[0]
                    uh = u_history(local_u, start)
                    summary = np.concatenate([uh[-1], uh.mean(0), uh.std(0), uh[-1] - uh[0]]) @ summary_projection
                    ordered = (basis @ uh).reshape(-1) @ ordered_projection
                    shuffled = (basis @ shuffle_history(uh, name, horizon["horizon_index"])).reshape(-1) @ ordered_projection
                    matrices["K"].append(k_history(frame_k, selected, start, previous_offsets))
                    matrices["S"].append(summary.astype(np.float32))
                    matrices["O"].append(ordered.astype(np.float32))
                    matrices["X"].append(shuffled.astype(np.float32))
                    targets.append(np.asarray(horizon["robust_values"], np.float32))
                    folds.append(row["fold"]); sources.append(source_index); refs.append(row["ref_len"])
                    sample_ids.append(name); horizon_ids.append(horizon["horizon_index"])
                selected.extend(P5.horizon_selected(blocks, tuple(horizon["chosen_offsets"])))
                previous_offsets.extend(horizon["chosen_offsets"])
        print(json.dumps({"built_shard": shard, "rows": len(targets),
                          "seconds": round(time.perf_counter() - started, 1)}), flush=True)
    arrays = {key: np.asarray(value, np.float32) for key, value in matrices.items()}
    target = np.asarray(targets, np.float32)
    if target.shape != (43437, 81) or len(source_map) != 578:
        raise RuntimeError(f"dataset identity mismatch: {target.shape}, sources={len(source_map)}")
    if not np.all(target[:, CENTER] == 0):
        raise RuntimeError("center target is not zero")
    np.savez_compressed(DATASET, **arrays, targets=target, folds=np.asarray(folds, np.int8),
                        source_indices=np.asarray(sources, np.int16), ref_len=np.asarray(refs, np.int16),
                        sample_ids=np.asarray(sample_ids), horizon_indices=np.asarray(horizon_ids, np.int16))
    manifest = {
        "status": "complete", "created_utc": utcnow(), "rows": len(target), "sources": len(source_map),
        "shapes": {key: list(value.shape) for key, value in arrays.items()} | {"targets": list(target.shape)},
        "dataset_sha256": sha256_file(DATASET), "outcomes_read": False,
    }
    (OUTPUT / "dataset_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def ridge_predict(train_x, train_y, eval_x, alpha):
    mean = train_x.mean(0, dtype=np.float64)
    scale = train_x.std(0, dtype=np.float64)
    scale[scale < 1e-6] = 1.0
    x = (np.asarray(train_x, np.float64) - mean) / scale
    z = (np.asarray(eval_x, np.float64) - mean) / scale
    y_mean = train_y.mean(0, dtype=np.float64)
    y = np.asarray(train_y, np.float64) - y_mean
    gram = x.T @ x
    gram.flat[::len(gram) + 1] += alpha
    weights = np.linalg.solve(gram, x.T @ y)
    return (z @ weights + y_mean).astype(np.float32)


def bootstrap(values, sources, reps=10000):
    grouped = defaultdict(float)
    for value, source in zip(values, sources):
        grouped[int(source)] += float(value)
    keys = sorted(grouped)
    totals = np.asarray([grouped[key] for key in keys], np.float64)
    rng = np.random.default_rng(SEED_BOOT)
    result = np.empty(reps, np.float64)
    for index in range(reps):
        result[index] = totals[rng.integers(0, len(totals), size=len(totals))].sum()
    return [float(np.quantile(result, 0.025)), float(np.quantile(result, 0.975))]


def evaluate_model(name, matrix, targets, folds):
    predictions = np.empty_like(targets)
    fold_rows = []
    for fold in range(5):
        train, evaluate = folds != fold, folds == fold
        prediction = ridge_predict(matrix[train], targets[train], matrix[evaluate], 100.0)
        prediction[:, CENTER] = 0.0
        predictions[evaluate] = prediction
        fold_rows.append({"fold": fold, "train_rows": int(train.sum()), "eval_rows": int(evaluate.sum())})
    chosen = predictions.argmax(1)
    reward = targets[np.arange(len(targets)), chosen]
    oracle = targets.max(1)
    return chosen, reward, {
        "signed_robust_reward": float(reward.sum()),
        "oracle_robust_reward": float(oracle.sum()),
        "oracle_reward_capture": float(reward.sum() / max(1.0, oracle.sum())),
        "winner_accuracy": float(np.mean(chosen == targets.argmax(1))),
        "changed_horizons": int(np.sum(chosen != CENTER)),
        "negative_reward_horizons": int(np.sum(reward < 0)),
        "positive_reward_horizons": int(np.sum(reward > 0)),
        "folds": fold_rows,
    }


def run():
    identity = prepare()
    if not DATASET.exists():
        raise RuntimeError("run --build before --run")
    if (OUTPUT / "metrics.json").exists():
        raise FileExistsError("non-overwriting predictor outcome exists")
    config = json.loads(CONFIG.read_text())
    with np.load(DATASET, allow_pickle=False) as data:
        k, s, o, x = (np.asarray(data[key], np.float32) for key in ("K", "S", "O", "X"))
        targets = np.asarray(data["targets"], np.float32)
        folds = np.asarray(data["folds"], np.int8)
        sources = np.asarray(data["source_indices"], np.int16)
    matrices = {"K": k, "KS": np.concatenate([k, s], 1),
                "KO": np.concatenate([k, o], 1), "KX": np.concatenate([k, x], 1)}
    model_metrics, chosen_all, rewards = {}, {}, {}
    for name, matrix in matrices.items():
        chosen_all[name], rewards[name], model_metrics[name] = evaluate_model(
            name, matrix, targets, folds)
        model_metrics[name]["source_bootstrap_95pct_total_reward"] = bootstrap(rewards[name], sources)
        print(json.dumps({"model": name, **model_metrics[name]}), flush=True)
    delta_k = rewards["KO"] - rewards["K"]
    delta_x = rewards["KO"] - rewards["KX"]
    comparisons = {
        "KO_minus_K": {"reward": float(delta_k.sum()),
                       "source_bootstrap_95pct_total_reward": bootstrap(delta_k, sources)},
        "KO_minus_KX": {"reward": float(delta_x.sum()),
                        "source_bootstrap_95pct_total_reward": bootstrap(delta_x, sources)},
    }
    gates = config["gates"]
    checks = {
        "KO_reward": model_metrics["KO"]["signed_robust_reward"] >= gates["KO_signed_robust_reward_at_least"],
        "KO_bootstrap": model_metrics["KO"]["source_bootstrap_95pct_total_reward"][0] > gates["KO_source_bootstrap_lower_gt"],
        "KO_minus_K_reward": comparisons["KO_minus_K"]["reward"] >= gates["KO_minus_K_reward_at_least"],
        "KO_minus_K_bootstrap": comparisons["KO_minus_K"]["source_bootstrap_95pct_total_reward"][0] > gates["KO_minus_K_source_bootstrap_lower_gt"],
        "KO_minus_KX_reward": comparisons["KO_minus_KX"]["reward"] >= gates["KO_minus_KX_reward_at_least"],
        "KO_minus_KX_bootstrap": comparisons["KO_minus_KX"]["source_bootstrap_95pct_total_reward"][0] > gates["KO_minus_KX_source_bootstrap_lower_gt"],
    }
    passed = all(checks.values())
    metrics = {
        "status": "complete", "created_utc": utcnow(), "gpu_used": False, "train_only": True,
        "calibration_dev_test_heldout_or_closed_loop_read": False,
        "rows": len(targets), "sources": len(np.unique(sources)), "models": model_metrics,
        "comparisons": comparisons, "checks": checks, "gate_passed": passed,
        "decision": ("unlock preregistered unknown-EOS equal-budget closed loop" if passed else
                     "stop this ordered heatmap-U H4 ranker identity; do not run closed loop or held-out"),
    }
    (OUTPUT / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    np.savez_compressed(OUTPUT / "oof_choices.npz", folds=folds, source_indices=sources,
                        targets=targets, **{f"chosen_{key}": value for key, value in chosen_all.items()},
                        **{f"reward_{key}": value for key, value in rewards.items()})
    manifest = {
        "status": "complete", "config_sha256": sha256_file(CONFIG),
        "identity_sha256": sha256_file(OUTPUT / "preregistered_identity.json"),
        "dataset_sha256": sha256_file(DATASET), "metrics_sha256": sha256_file(OUTPUT / "metrics.json"),
        "oof_choices_sha256": sha256_file(OUTPUT / "oof_choices.npz"), "gate_passed": passed,
    }
    (OUTPUT / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--prepare", action="store_true")
    parser.add_argument("--build", action="store_true")
    parser.add_argument("--run", action="store_true")
    args = parser.parse_args()
    if sum((args.prepare, args.build, args.run)) != 1:
        parser.error("choose exactly one of --prepare, --build, or --run")
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    result = prepare() if args.prepare else build_dataset() if args.build else run()
    print(json.dumps(result, indent=2))
