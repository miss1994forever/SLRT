#!/usr/bin/env python3
"""CPU-only independent calibration confirmation of frozen P6 K/history."""
import argparse
import gzip
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
DENSE = BASE / "train_dense_stride1_v1_49faacc3"
FIT = BASE / "p6_local_js_predictor_oof_v1_49faacc3"
CONFIG = ROOT / "configs/p6_k_history_independent_calibration_v1_preregistered.json"
OUTPUT = BASE / "p6_k_history_independent_calibration_v1_49faacc3"
IDENTITY = OUTPUT / "preregistered_identity.json"
DATA = OUTPUT / "calibration_dataset.npz"
METRICS = OUTPUT / "metrics.json"
PER_SAMPLE = OUTPUT / "per_sample.jsonl.gz"
BOOT_SEED = 261224


def imp(name):
    path = ROOT / f"tools/{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


PRED = imp("analyze_phoenix_p6_local_js_predictor_oof")
ORACLE, RANK = PRED.ORACLE, PRED.RANK
CHRON, BUILDER, BLOCK = PRED.CHRON, PRED.BUILDER, PRED.BLOCK


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


def verify():
    config = json.loads(CONFIG.read_text())
    expected = config["fixed_assets"]
    actual = {
        "dense_protocol_sha256": sha(DENSE / "protocol_manifest.json"),
        "fit_calibration_split_sha256": sha(DENSE / "fit_calibration_split.json"),
        "fit_p6_dataset_sha256": sha(FIT / "dataset.npz"),
        "fit_p6_metrics_sha256": sha(FIT / "metrics.json"),
    }
    if actual != expected:
        raise RuntimeError(f"asset mismatch: {actual}")
    return config


def calibration_names():
    split = json.loads((DENSE / "fit_calibration_split.json").read_text())
    names = {name for name, partition in split["assignments"].items() if partition == "calibration"}
    sources = {BUILDER.source_video(name) for name in names}
    if len(names) != 718 or len(sources) != 65:
        raise RuntimeError(f"calibration identity mismatch: {len(names)} {len(sources)}")
    return names


def prepare():
    config = verify()
    if OUTPUT.exists() and any(OUTPUT.iterdir()):
        if not IDENTITY.is_file():
            raise FileExistsError(f"nonempty output without identity: {OUTPUT}")
        identity = json.loads(IDENTITY.read_text())
        if identity["config_sha256"] != sha(CONFIG):
            raise RuntimeError("configuration changed after prepare")
        return identity
    OUTPUT.mkdir(parents=True, exist_ok=True)
    names = calibration_names()
    identity = {
        "created_utc": now(), "config_sha256": sha(CONFIG),
        "script_sha256_at_prepare": sha(Path(__file__)),
        "created_before_calibration_outcomes": True,
        "calibration_authorized": True, "calibration_samples": len(names),
        "dev_test_heldout_read": False,
    }
    IDENTITY.write_text(json.dumps(identity, indent=2) + "\n")
    return identity


def build():
    prepare()
    if DATA.exists():
        raise FileExistsError(DATA)
    wanted = calibration_names()
    vocab = json.loads(CHRON.DEFAULT_VOCAB.read_text())
    blank = vocab.index("<blank>")
    rng = np.random.default_rng(261130)
    token_bucket = rng.integers(0, 32, size=len(vocab))
    token_sign = rng.choice(np.asarray([-1.0, 1.0], np.float32), size=len(vocab))
    features, target, raw_js = [], [], []
    sample_indices, block_indices, sample_names = [], [], []
    indices, shard_count = CHRON.completed_shard_indices(DENSE)
    for shard in indices:
        results, logits, _ = BUILDER.BUILDER.validate_dense_shard(
            DENSE, shard, shard_count, verify_hashes=False)
        for name in results:
            if name not in wanted:
                continue
            sample_index = len(sample_names); sample_names.append(name)
            probabilities = BUILDER.ORACLE.softmax_rows(np.asarray(logits[name]))
            frame_features = RANK.dense_frame_features(
                probabilities, blank, token_bucket, token_sign)
            selected, previous_offsets = [], []
            for block_index, start in enumerate(ORACLE.complete_blocks(len(probabilities))):
                score = ORACLE.js_score(probabilities[start], probabilities[start + 2])
                features.append(RANK.k_history(frame_features, selected, start, previous_offsets))
                raw_js.append(score); target.append(np.log(score + 1e-8))
                sample_indices.append(sample_index); block_indices.append(block_index)
                selected.extend([start, start + 2]); previous_offsets.append(2)
        print(json.dumps({"shard": shard, "samples": len(sample_names), "rows": len(target)}), flush=True)
    if len(sample_names) != 718 or len({BUILDER.source_video(name) for name in sample_names}) != 65:
        raise RuntimeError("calibration scope mismatch")
    arrays = {
        "K": np.asarray(features, np.float32), "target": np.asarray(target, np.float32),
        "raw_js": np.asarray(raw_js, np.float32),
        "sample_indices": np.asarray(sample_indices, np.int16),
        "block_indices": np.asarray(block_indices, np.int16),
        "sample_names": np.asarray(sample_names),
    }
    np.savez_compressed(DATA, **arrays)
    manifest = {"status": "complete", "created_utc": now(), "samples": len(sample_names),
                "sources": 65, "rows": len(target), "shapes": {k: list(v.shape) for k, v in arrays.items()},
                "dataset_sha256": sha(DATA), "calibration_outcomes_materialized": True,
                "dev_test_heldout_read": False}
    write_atomic(OUTPUT / "dataset_manifest.json", manifest)
    return manifest


def ridge_fit_predict(train_x, train_y, test_x, alpha=100.0):
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
    return (z @ weights + ymean).astype(np.float32), ymean


def bootstrap(rows, repetitions=10000):
    grouped = defaultdict(lambda: [0, 0])
    for row in rows:
        grouped[row["source"]][0] += row["K"] - row["uniform"]
        grouped[row["source"]][1] += row["ref_len"]
    values = np.asarray(list(grouped.values()), np.float64)
    rng = np.random.default_rng(BOOT_SEED)
    result = np.empty(repetitions, np.float64)
    for index in range(repetitions):
        chosen = rng.integers(0, len(values), len(values))
        sample = values[chosen]
        result[index] = 100.0 * sample[:, 0].sum() / sample[:, 1].sum()
    return [float(np.quantile(result, 0.025)), float(np.quantile(result, 0.975))]


def run():
    prepare()
    if not DATA.exists():
        raise RuntimeError("run --build first")
    if METRICS.exists() or PER_SAMPLE.exists():
        raise FileExistsError("calibration outcome already exists")
    with np.load(FIT / "dataset.npz", allow_pickle=False) as packed:
        fit_x = np.asarray(packed["K"], np.float32)
        fit_y = np.asarray(packed["target"], np.float32)
    with np.load(DATA, allow_pickle=False) as packed:
        cal_x = np.asarray(packed["K"], np.float32)
        cal_y = np.asarray(packed["target"], np.float32)
        sample_indices = np.asarray(packed["sample_indices"], np.int64)
        sample_names = [str(value) for value in packed["sample_names"].tolist()]
    prediction, fit_mean = ridge_fit_predict(fit_x, fit_y, cal_x)
    sse = float(np.sum((np.asarray(cal_y, np.float64) - prediction) ** 2))
    sst = float(np.sum((np.asarray(cal_y, np.float64) - fit_mean) ** 2))
    r2 = 1.0 - sse / sst
    rows_by_sample = {index: np.flatnonzero(sample_indices == index) for index in range(len(sample_names))}
    selected = {}
    for sample_index, rows in rows_by_sample.items():
        count = (len(rows) + 1) // 2
        order = np.lexsort((rows, -prediction[rows]))[:count]
        selected[sample_index] = set(rows[order].tolist())
    wanted = set(sample_names); name_to_index = {name: index for index, name in enumerate(sample_names)}
    vocab = json.loads(CHRON.DEFAULT_VOCAB.read_text()); blank = vocab.index("<blank>")
    indices, shard_count = CHRON.completed_shard_indices(DENSE)
    output_rows = []
    with gzip.open(PER_SAMPLE, "wt", encoding="utf-8", compresslevel=6) as handle:
        for shard in indices:
            results, logits, _ = BUILDER.BUILDER.validate_dense_shard(
                DENSE, shard, shard_count, verify_hashes=False)
            for name in results:
                if name not in wanted:
                    continue
                probabilities = BUILDER.ORACLE.softmax_rows(np.asarray(logits[name]))
                reference = BUILDER.REPLAY.clean_phoenix_2014_trans(results[name]["gls_ref"])
                blocks = ORACLE.complete_blocks(len(probabilities)); skeleton = ORACLE.base_skeleton(len(probabilities))
                count = (len(blocks) + 1) // 2
                uniform_local = ORACLE.uniform_bonus_indices(blocks, count)
                uniform_windows = sorted(skeleton + [blocks[index] + 2 for index in uniform_local])
                full_windows = sorted(skeleton + [start + 2 for start in blocks])
                sample_index = name_to_index[name]; global_rows = rows_by_sample[sample_index]
                chosen_local = [local for local, global_row in enumerate(global_rows)
                                if int(global_row) in selected[sample_index]]
                k_windows = sorted(skeleton + [blocks[index] + 2 for index in chosen_local])
                if len(k_windows) != len(uniform_windows):
                    raise RuntimeError(f"budget mismatch: {name}")
                decoded = {key: BLOCK.decode_counts(probabilities, windows, reference, vocab, blank)
                           for key, windows in (("uniform", uniform_windows), ("K", k_windows),
                                                ("full_center", full_windows))}
                row = {"sample": name, "source": BUILDER.source_video(name),
                       "ref_len": int(decoded["uniform"]["ref_len"]),
                       "uniform": int(decoded["uniform"]["error"]),
                       "K": int(decoded["K"]["error"]),
                       "full_center": int(decoded["full_center"]["error"]),
                       "selected_windows": len(k_windows)}
                output_rows.append(row); handle.write(json.dumps(row) + "\n")
            print(json.dumps({"replay_shard": shard, "samples": len(output_rows)}), flush=True)
    totals = {key: sum(row[key] for row in output_rows) for key in ("uniform", "K", "full_center")}
    ref_total = sum(row["ref_len"] for row in output_rows)
    ci = bootstrap(output_rows)
    gates = json.loads(CONFIG.read_text())["primary_gates"]
    checks = {
        "calibration_r2": r2 >= float(gates["calibration_r2_against_fit_mean_at_least"]),
        "K_error_reduction_vs_uniform": totals["uniform"] - totals["K"] >= int(gates["K_error_reduction_vs_uniform_at_least"]),
        "K_vs_uniform_ci": ci[1] < 0,
    }
    passed = all(checks.values())
    metrics = {
        "status": "complete", "created_utc": now(), "gpu_used": False,
        "independent_calibration": True, "dev_test_heldout_read": False,
        "teacher_forced_history": True, "known_length_topk": True, "unknown_eos_closed_loop": False,
        "fit_rows": len(fit_y), "calibration_rows": len(cal_y), "samples": len(output_rows), "sources": 65,
        "target": {"r2_against_fit_mean": r2, "fit_target_mean": fit_mean},
        "reference_glosses": ref_total, "errors": totals,
        "wer": {key: 100.0 * value / ref_total for key, value in totals.items()},
        "K_vs_uniform": {"delta_errors": totals["K"] - totals["uniform"],
                         "delta_wer_pp": 100.0 * (totals["K"] - totals["uniform"]) / ref_total,
                         "source_cluster_bootstrap_ci95": ci},
        "checks": checks, "gate_passed": passed,
        "decision": ("unlock separately preregistered calibration unknown-EOS self-history simulation"
                     if passed else "stop K/history before closed loop and held-out"),
    }
    write_atomic(METRICS, metrics)
    np.savez_compressed(OUTPUT / "calibration_predictions.npz", target=cal_y, prediction=prediction,
                        sample_indices=sample_indices)
    manifest = {"status": "complete", "config_sha256": sha(CONFIG),
                "identity_sha256": sha(IDENTITY), "dataset_sha256": sha(DATA),
                "metrics_sha256": sha(METRICS), "per_sample_sha256": sha(PER_SAMPLE),
                "predictions_sha256": sha(OUTPUT / "calibration_predictions.npz"),
                "gate_passed": passed, "dev_test_heldout_read": False}
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
