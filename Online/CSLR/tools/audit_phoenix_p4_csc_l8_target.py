#!/usr/bin/env python3
"""Outcome-blind target stability audit for preregistered CSC-L8-v1."""
import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "results/phoenix-2014t_ISLR"
CONFIG = ROOT / "configs/p4_causal_semantic_coverage_l8_v1_preregistered.json"
HISTORY = (BASE / "p3_fulltrain_decoder_conditioned_value_oof_v1_49faacc3"
           / "paid_decoder_history_blocks.npz")
DENSE_ROOT = BASE / "p3_fulltrain_utility_information_ladder_oof_v1_49faacc3"
DENSE = DENSE_ROOT / "fit_dense_logits_float16.npy"
DENSE_INDEX = DENSE_ROOT / "dense_index.npz"
OUTPUT = BASE / "p4_csc_l8_target_audit_v1_49faacc3"
TARGETS = OUTPUT / "target_scores.npz"
METRICS = OUTPUT / "metrics.json"
MANIFEST = OUTPUT / "manifest.json"


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def softmax(values, temperature):
    values = values.astype(np.float32, copy=False) / np.float32(temperature)
    values = values - values.max(axis=-1, keepdims=True)
    values = np.exp(values)
    values /= values.sum(axis=-1, keepdims=True)
    return values


def residuals(probability):
    """Return JS and Hellinger residuals for offsets 1/2/3 between anchors 0/4."""
    candidate = probability[:, 1:4]
    alpha = np.asarray([0.25, 0.5, 0.75], np.float32)[None, :, None]
    interpolation = ((1.0 - alpha) * probability[:, :1]
                     + alpha * probability[:, 4:5])
    midpoint = 0.5 * (candidate + interpolation)
    tiny = np.float32(1e-30)
    js = 0.5 * np.sum(
        candidate * (np.log(np.maximum(candidate, tiny)) - np.log(np.maximum(midpoint, tiny)))
        + interpolation * (np.log(np.maximum(interpolation, tiny)) - np.log(np.maximum(midpoint, tiny))),
        axis=-1,
    )
    hellinger = 0.5 * np.sum(
        (np.sqrt(candidate) - np.sqrt(interpolation)) ** 2, axis=-1
    )
    return js.astype(np.float32), hellinger.astype(np.float32)


def cluster_bootstrap_lower(correct, sources, source_count, seed, repetitions):
    counts = np.bincount(sources, minlength=source_count).astype(np.float64)
    successes = np.bincount(sources, weights=correct.astype(np.float64),
                            minlength=source_count)
    if np.any(counts == 0):
        raise RuntimeError("source identity has an empty cluster")
    rng = np.random.default_rng(seed)
    values = np.empty(repetitions, np.float64)
    for index in range(repetitions):
        sampled = rng.integers(0, source_count, size=source_count)
        values[index] = successes[sampled].sum() / counts[sampled].sum()
    return {
        "repetitions": repetitions,
        "seed": seed,
        "ci95": [float(np.quantile(values, 0.025)), float(np.quantile(values, 0.975))],
        "mean": float(values.mean()),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-blocks", type=int, default=512)
    args = parser.parse_args()
    config = json.loads(CONFIG.read_text())
    if config["status"] != "phase_a_preregistered_before_target_audit":
        raise RuntimeError("unexpected preregistration status")
    if OUTPUT.exists():
        raise FileExistsError(f"non-overwriting audit output already exists: {OUTPUT}")

    # Whitelist only outcome-blind arrays. Do not access rewards or continuous utility targets.
    with np.load(HISTORY, allow_pickle=False) as packed:
        allowed = ("folds", "source_indices", "sample_indices", "candidate_starts", "sample_names")
        data = {key: np.asarray(packed[key]) for key in allowed}
    with np.load(DENSE_INDEX, allow_pickle=False) as packed:
        dense_names = [str(value) for value in packed["sample_names"].tolist()]
        offsets = np.asarray(packed["offsets"], np.int64)
    names = [str(value) for value in data["sample_names"].tolist()]
    if names != dense_names:
        raise RuntimeError("history/dense sample identity mismatch")
    blocks = len(data["folds"])
    if blocks != int(config["scope"]["blocks"]):
        raise RuntimeError("block count mismatch")
    if len(np.unique(data["source_indices"])) != int(config["scope"]["sources"]):
        raise RuntimeError("source count mismatch")
    starts = np.asarray(data["candidate_starts"], np.int64)
    if starts.shape != (blocks, 2) or not np.all(starts[:, 1] == starts[:, 0] + 2):
        raise RuntimeError("candidate geometry mismatch")
    skeleton = starts[:, 0] - 1
    sample_indices = np.asarray(data["sample_indices"], np.int64)
    lengths = np.diff(offsets)
    valid = (skeleton >= 0) & (skeleton + 4 < lengths[sample_indices])
    block_indices = np.flatnonzero(valid)
    local5 = skeleton[valid, None] + np.arange(5, dtype=np.int64)[None, :]
    global5 = offsets[sample_indices[valid], None] + local5

    dense = np.load(DENSE, mmap_mode="r")
    if dense.shape[0] != int(offsets[-1]):
        raise RuntimeError("dense logit coverage mismatch")
    scores = {"js_t1": np.empty((len(block_indices), 3), np.float32),
              "js_t08": np.empty((len(block_indices), 3), np.float32),
              "js_t125": np.empty((len(block_indices), 3), np.float32),
              "hellinger_t1": np.empty((len(block_indices), 3), np.float32)}
    for left in range(0, len(block_indices), args.batch_blocks):
        right = min(left + args.batch_blocks, len(block_indices))
        logits = np.asarray(dense[global5[left:right]], np.float32)
        for key, temperature in (("js_t08", 0.8), ("js_t1", 1.0), ("js_t125", 1.25)):
            js, hellinger = residuals(softmax(logits, temperature))
            scores[key][left:right] = js
            if temperature == 1.0:
                scores["hellinger_t1"][left:right] = hellinger

    winners = {key: value.argmax(axis=1).astype(np.int8) + 1
               for key, value in scores.items()}
    primary = winners["js_t1"]
    source_indices = np.asarray(data["source_indices"], np.int64)[valid]
    source_count = int(config["scope"]["sources"])
    gate = config["phase_a1_gates"]
    comparisons = {}
    for number, key in enumerate(("js_t08", "js_t125", "hellinger_t1")):
        correct = primary == winners[key]
        comparisons[key] = {
            "agreement": float(correct.mean()),
            "source_cluster_bootstrap": cluster_bootstrap_lower(
                correct, source_indices, source_count,
                int(gate["bootstrap_seed"]) + number,
                int(gate["bootstrap_repetitions"]),
            ),
        }
    winner_counts = {str(offset): int(np.sum(primary == offset)) for offset in (1, 2, 3)}
    winner_fractions = {key: value / len(primary) for key, value in winner_counts.items()}
    nonzero = np.max(scores["js_t1"], axis=1) > float(gate["nonzero_epsilon"])
    checks = {
        "coverage": float(valid.mean()) >= float(gate["coverage_min"]),
        "winner_balance": min(winner_fractions.values()) >= float(gate["winner_fraction_each_min"]),
        "temperature_08_agreement": comparisons["js_t08"]["agreement"] >= float(gate["temperature_winner_agreement_min"]),
        "temperature_125_agreement": comparisons["js_t125"]["agreement"] >= float(gate["temperature_winner_agreement_min"]),
        "temperature_08_ci_lower": comparisons["js_t08"]["source_cluster_bootstrap"]["ci95"][0] >= float(gate["temperature_source_bootstrap_lower_min"]),
        "temperature_125_ci_lower": comparisons["js_t125"]["source_cluster_bootstrap"]["ci95"][0] >= float(gate["temperature_source_bootstrap_lower_min"]),
        "hellinger_agreement": comparisons["hellinger_t1"]["agreement"] >= float(gate["hellinger_winner_agreement_min"]),
        "hellinger_ci_lower": comparisons["hellinger_t1"]["source_cluster_bootstrap"]["ci95"][0] >= float(gate["hellinger_source_bootstrap_lower_min"]),
        "nonzero_fraction": float(nonzero.mean()) >= float(gate["nonzero_block_fraction_min"]),
    }
    metrics = {
        "experiment": config["experiment"],
        "phase": "A1 outcome-blind target stability",
        "gpu_used": False,
        "robust_reward_read": False,
        "reference_or_wer_read": False,
        "future_continuation_read": False,
        "true_eos_read": False,
        "dev_or_test_read": False,
        "blocks_total": blocks,
        "blocks_valid": int(valid.sum()),
        "coverage": float(valid.mean()),
        "winner_counts": winner_counts,
        "winner_fractions": winner_fractions,
        "nonzero_fraction": float(nonzero.mean()),
        "comparisons": comparisons,
        "checks": checks,
        "gate_passed": bool(all(checks.values())),
        "next_step": ("run preregistered A2 source-disjoint ridge probe" if all(checks.values())
                      else "stop CSC-L8-v1 before identifiability probe and policy outcomes"),
    }
    OUTPUT.mkdir(parents=True)
    np.savez_compressed(
        TARGETS,
        block_indices=block_indices.astype(np.int32),
        folds=np.asarray(data["folds"], np.int8)[valid],
        source_indices=source_indices.astype(np.int16),
        sample_indices=sample_indices[valid].astype(np.int16),
        candidate_starts=starts[valid].astype(np.int32),
        js_t1=scores["js_t1"],
        winners=primary,
    )
    METRICS.write_text(json.dumps(metrics, indent=2) + "\n")
    manifest = {
        "status": "complete",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "config_sha256": sha256_file(CONFIG),
        "metrics_sha256": sha256_file(METRICS),
        "targets_sha256": sha256_file(TARGETS),
        "gate_passed": metrics["gate_passed"],
    }
    MANIFEST.write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
