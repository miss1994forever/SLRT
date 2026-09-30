#!/usr/bin/env python3
"""Independent train-fit posterior replay of frozen schedules and release controller."""
import argparse
import hashlib
import json
import struct
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from tools import analyze_phoenix_p6_k_online_quantile_oof as frozen
from utils.online_quantile_controller import OnlineQuantileController, FrozenRidgeBundle, k_history_feature

BASE = ROOT / "results/phoenix-2014t_ISLR"
DENSE = BASE / "train_dense_stride1_v1_49faacc3"
FIT = BASE / "p6_local_js_predictor_oof_v1_49faacc3"
MODEL = ROOT / "configs/frozen_models/p6_online_quantile_fold_models_v1.json"
OUT = Path(__file__).resolve().parent
MODEL_SHA = "116996aef99b4177fb6159086eba6a9ca90f03df6e61e8974a50a760922221e0"


def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def fit_identity():
    config = frozen.verify()
    split_path = DENSE / "fit_calibration_split.json"
    split = json.loads(split_path.read_text())
    allowed = {name for name, part in split["assignments"].items() if part == "fit"}
    with np.load(FIT / "dataset.npz", allow_pickle=False) as z:
        names = list(map(str, z["sample_names"].tolist()))
        ids = z["sample_indices"]
        folds = z["folds"]
        fold_by_name = {}
        for index, name in enumerate(names):
            unique = np.unique(folds[ids == index])
            if len(unique) != 1:
                raise ValueError("sample crosses folds")
            fold_by_name[name] = int(unique[0])
    if len(allowed) != 6378 or set(names) != allowed or any(not x.startswith("train/") for x in allowed):
        raise ValueError("train-fit whitelist mismatch")
    return config, allowed, fold_by_name


def selected_logits(path, allowed, vocab_size):
    """Scan pickle opcodes; seek across excluded array bytes without decoding them.

    The frozen dense logits pickle is a dict of train/name -> float32 ndarray.
    This deliberately supports only its observed protocol-4 opcode subset.
    """
    values = {}
    current = None
    with open(path, "rb") as f:
        while True:
            code = f.read(1)
            if not code:
                raise ValueError("truncated pickle")
            b = code[0]
            if b == 0x2e:  # STOP
                break
            if b == 0x8c:  # SHORT_BINUNICODE
                n = f.read(1)[0]
                raw = f.read(n)
                if raw.startswith(b"train/"):
                    current = raw.decode("utf-8")
            elif b == 0x42:  # BINBYTES: ndarray payload
                n = struct.unpack("<I", f.read(4))[0]
                if current is None or n % (4 * vocab_size):
                    raise ValueError("unexpected ndarray payload")
                if current in allowed:
                    if current in values:
                        raise ValueError("duplicate sample")
                    values[current] = np.frombuffer(f.read(n), dtype="<f4").reshape(-1, vocab_size)
                else:
                    f.seek(n, 1)
                current = None
            elif b == 0x95:  # FRAME
                f.seek(8, 1)
            elif b in (0x4b, 0x68, 0x80, 0x43):  # one-byte arg
                n = f.read(1)[0]
                if b == 0x43:
                    f.seek(n, 1)
            elif b == 0x4d:
                f.seek(2, 1)
            elif b == 0x4a:
                f.seek(4, 1)
            elif b in (0x7d, 0x94, 0x28, 0x93, 0x85, 0x86, 0x87, 0x52,
                       0x89, 0x88, 0x74, 0x62, 0x4e, 0x75):
                pass
            else:
                raise ValueError(f"unrecognized pickle opcode {b:#x} in {path}")
    return values


def model_check():
    if sha(MODEL) != MODEL_SHA:
        raise ValueError("release model hash differs from independently frozen audit identity")
    payload = json.loads(MODEL.read_text())
    with np.load(FIT / "dataset.npz", allow_pickle=False) as z:
        x = z["K"]
        folds = z["folds"]
    with np.load(FIT / "oof_predictions.npz", allow_pickle=False) as z:
        target = z["K"]
    predicted = np.empty(len(target), np.float64)
    for fold, model in enumerate(payload["folds"]):
        mask = folds == fold
        predicted[mask] = ((x[mask].astype(np.float64) - model["mean"]) / model["scale"]) @ model["weights"] + model["intercept"]
    diff = predicted - target.astype(np.float64)
    result = {"rows": len(diff), "max_abs_error": float(np.max(np.abs(diff))),
              "rmse": float(np.sqrt(np.mean(diff * diff))), "model_sha256": sha(MODEL),
              "fit_dataset_sha256": sha(FIT / "dataset.npz"),
              "oof_predictions_sha256": sha(FIT / "oof_predictions.npz")}
    result["passed"] = bool(len(diff) == 183401 and result["max_abs_error"] < 1e-4)
    (OUT / "model_validation.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def compare_one(name, raw_logits, fold, models, bundle, blank, buckets, signs, trace):
    probabilities = frozen.BUILDER.ORACLE.softmax_rows(np.asarray(raw_logits))
    features = frozen.RANK.dense_frame_features(probabilities, blank, buckets, signs)
    original_features = []
    original_scores = []
    real_k = frozen.RANK.k_history
    real_score = frozen.score

    def capture_k(*args):
        value = real_k(*args)
        original_features.append(value)
        return value

    def capture_score(*args):
        value = real_score(*args)
        original_scores.append(value)
        return value

    frozen.RANK.k_history = capture_k
    frozen.score = capture_score
    try:
        schedules, original_summary = frozen.schedules(probabilities, features, models[fold])
    finally:
        frozen.RANK.k_history = real_k
        frozen.score = real_score
    block_count = len(original_scores)
    paid = {}
    current_features = []
    current_scores = []

    def score(start, selected):
        offsets = [2 for index in selected if index % 4 == 2]
        feature = k_history_feature(paid, selected, start, offsets)
        value = bundle.score(feature, fold)
        current_features.append(feature)
        current_scores.append(value)
        return value

    controller = OnlineQuantileController(score)
    mismatch = 0
    max_score = max_feature = 0.0
    for block in range(block_count):
        start = 4 * block
        paid[start] = features[start]
        controller.start_block(start)
        prior_token = controller.balance
        threshold = float(np.median(np.asarray(original_scores[:block], np.float64))) if block else None
        action = controller.complete_block(start)
        original_action = start + 2 in schedules["quantile"]
        if action:
            paid[start + 2] = features[start + 2]
        score_error = abs(current_scores[-1] - original_scores[block])
        feature_error = float(np.max(np.abs(np.asarray(current_features[-1]) - original_features[block])))
        max_score = max(max_score, score_error)
        max_feature = max(max_feature, feature_error)
        mismatch += int(action != original_action)
        trace.write(json.dumps({"sample": name, "block": block, "start": start,
                                "original_score": original_scores[block], "release_score": current_scores[-1],
                                "threshold": threshold, "token_before": prior_token,
                                "token_after": controller.balance, "original_action": original_action,
                                "release_action": action, "feature_max_abs_error": feature_error}) + "\n")
    if len(probabilities) % 4:
        start = 4 * block_count
        paid[start] = features[start]
        controller.start_block(start)
    selected = controller.finish()
    if selected != schedules["quantile"]:
        mismatch += 1
    if (controller.forced != original_summary["forced"] or
        controller.autonomous != original_summary["autonomous"] or
        controller.balance != original_summary["unspent"]):
        mismatch += 1
    return {"blocks": block_count, "mismatches": mismatch, "max_score_error": max_score,
            "max_feature_error": max_feature}


def main(limit):
    _, allowed, fold_by_name = fit_identity()
    model = model_check()
    if not model["passed"]:
        raise ValueError("current model does not match original OOF predictions")
    bundle = FrozenRidgeBundle(MODEL, frozen.CONFIG)
    models = frozen.fit_models()
    vocab = json.loads(frozen.CHRON.DEFAULT_VOCAB.read_text())
    blank = vocab.index("<blank>")
    rng = np.random.default_rng(261130)
    buckets = rng.integers(0, 32, size=len(vocab))
    signs = rng.choice(np.asarray([-1., 1.], np.float32), size=len(vocab))
    selected = set(sorted(allowed)[:limit]) if limit else allowed
    totals = {"samples": 0, "blocks": 0, "mismatches": 0, "max_score_error": 0.0,
              "max_feature_error": 0.0}
    seen = set()
    indices, shard_count = frozen.CHRON.completed_shard_indices(DENSE)
    path = OUT / ("trace_smoke.jsonl" if limit else "trace_full.jsonl")
    with path.open("x") as trace:
        for shard in indices:
            shard_path = DENSE / "shards" / f"shard-{shard:05d}-of-{shard_count:05d}" / "train_logits.pkl"
            for name, logits in selected_logits(shard_path, selected, len(vocab)).items():
                if name in seen:
                    raise ValueError("duplicate sample across shards")
                seen.add(name)
                row = compare_one(name, logits, fold_by_name[name], models, bundle, blank, buckets, signs, trace)
                totals["samples"] += 1
                totals["blocks"] += row["blocks"]
                totals["mismatches"] += row["mismatches"]
                totals["max_score_error"] = max(totals["max_score_error"], row["max_score_error"])
                totals["max_feature_error"] = max(totals["max_feature_error"], row["max_feature_error"])
            if shard % 8 == 0:
                print(json.dumps({"shard": shard, "seen": len(seen), "mismatches": totals["mismatches"]}), flush=True)
    if seen != selected:
        raise ValueError(f"missing {len(selected - seen)} fit samples")
    result = {**totals, "complete_fit": not bool(limit), "whitelist_count": len(allowed),
              "trace_sha256": sha(path), "frozen_script_sha256": sha(Path(frozen.__file__)),
              "controller_sha256": sha(ROOT / "utils/online_quantile_controller.py"),
              "model": model, "passed": bool(totals["mismatches"] == 0 and totals["max_score_error"] < 1e-4)}
    report = OUT / ("smoke_report.json" if limit else "full_report.json")
    report.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    if not result["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--limit", type=int, default=0)
    main(p.parse_args().limit)
