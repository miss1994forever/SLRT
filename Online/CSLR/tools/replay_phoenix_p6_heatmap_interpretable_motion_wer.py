#!/usr/bin/env python3
"""CPU-only equal-budget WER replay for frozen interpretable heatmap OOF scores."""
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
CONFIG = ROOT / "configs/p6_heatmap_interpretable_motion_wer_replay_v1_preregistered.json"
DIAG = BASE / "p6_heatmap_interpretable_motion_js_diagnostic_v1_49faacc3"
P6DATA = BASE / "p6_local_js_predictor_oof_v1_49faacc3" / "dataset.npz"
ORACLE_ROOT = BASE / "p6_local_js_information_gain_oracle_v1_49faacc3"
OUTPUT = BASE / "p6_heatmap_interpretable_motion_wer_replay_v1_49faacc3"
IDENTITY = OUTPUT / "preregistered_identity.json"
METRICS = OUTPUT / "metrics.json"
PER_SAMPLE = OUTPUT / "per_sample.jsonl.gz"
MODELS = ("K", "KM", "KQ", "KMQ", "KMM2", "KX")
BOOT_SEED = 261223


def imp(name):
    path = ROOT / f"tools/{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


PRED = imp("analyze_phoenix_p6_local_js_predictor_oof")
ORACLE, CHRON, BUILDER, BLOCK = PRED.ORACLE, PRED.CHRON, PRED.BUILDER, PRED.BLOCK
DENSE = PRED.DENSE


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
        "diagnostic_manifest_sha256": sha(DIAG / "manifest.json"),
        "diagnostic_metrics_sha256": sha(DIAG / "metrics.json"),
        "diagnostic_oof_predictions_sha256": sha(DIAG / "oof_predictions.npz"),
        "p6_dataset_sha256": sha(P6DATA),
        "p6_oracle_manifest_sha256": sha(ORACLE_ROOT / "manifest.json"),
    }
    if actual != expected:
        raise RuntimeError(f"asset mismatch: {actual}")
    return config


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
    identity = {
        "created_utc": now(), "config_sha256": sha(CONFIG),
        "script_sha256_at_prepare": sha(Path(__file__)), "created_before_wer_outputs": True,
        "post_diagnostic": True, "independent_confirmation": False,
        "calibration_dev_test_heldout_read": False, "scope": config["scope"],
    }
    IDENTITY.write_text(json.dumps(identity, indent=2) + "\n")
    return identity


def bootstrap(rows, left, right, repetitions=10000):
    grouped = defaultdict(lambda: [0, 0])
    for row in rows:
        grouped[row["source"]][0] += row[left] - row[right]
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
    if METRICS.exists() or PER_SAMPLE.exists():
        raise FileExistsError("WER outputs already exist")
    with np.load(P6DATA, allow_pickle=False) as packed:
        sample_indices = np.asarray(packed["sample_indices"], np.int64)
        sample_names = [str(value) for value in packed["sample_names"].tolist()]
    with np.load(DIAG / "oof_predictions.npz", allow_pickle=False) as packed:
        scores = {name: np.asarray(packed[name], np.float32) for name in MODELS}
    rows_by_sample = {index: np.flatnonzero(sample_indices == index) for index in range(len(sample_names))}
    selected = {}
    for name, score in scores.items():
        selected[name] = {}
        for sample_index, rows in rows_by_sample.items():
            count = (len(rows) + 1) // 2
            order = np.lexsort((rows, -score[rows]))[:count]
            selected[name][sample_index] = set(rows[order].tolist())
    vocab = json.loads(CHRON.DEFAULT_VOCAB.read_text())
    blank = vocab.index("<blank>")
    wanted = ORACLE.P5.FULL.fit_assignments()
    name_to_index = {name: index for index, name in enumerate(sample_names)}
    indices, shard_count = CHRON.completed_shard_indices(DENSE)
    output_rows = []
    with gzip.open(PER_SAMPLE, "wt", encoding="utf-8", compresslevel=6) as handle:
        for shard in indices:
            results, logits, _ = BUILDER.BUILDER.validate_dense_shard(
                DENSE, shard, shard_count, verify_hashes=False)
            for sample_name in results:
                if sample_name not in wanted:
                    continue
                probabilities = BUILDER.ORACLE.softmax_rows(np.asarray(logits[sample_name]))
                reference = BUILDER.REPLAY.clean_phoenix_2014_trans(results[sample_name]["gls_ref"])
                blocks = ORACLE.complete_blocks(len(probabilities))
                skeleton = ORACLE.base_skeleton(len(probabilities))
                bonus_count = (len(blocks) + 1) // 2
                uniform_blocks = ORACLE.uniform_bonus_indices(blocks, bonus_count)
                uniform_windows = sorted(skeleton + [blocks[index] + 2 for index in uniform_blocks])
                uniform_counts = BLOCK.decode_counts(probabilities, uniform_windows, reference, vocab, blank)
                row = {"sample": sample_name, "source": BUILDER.source_video(sample_name),
                       "ref_len": int(uniform_counts["ref_len"]),
                       "uniform": int(uniform_counts["error"]), "selected_windows": len(uniform_windows)}
                sample_index = name_to_index[sample_name]
                global_rows = rows_by_sample[sample_index]
                for model in MODELS:
                    local_blocks = [local for local, global_row in enumerate(global_rows)
                                    if int(global_row) in selected[model][sample_index]]
                    windows = sorted(skeleton + [blocks[index] + 2 for index in local_blocks])
                    if len(windows) != len(uniform_windows):
                        raise RuntimeError(f"budget mismatch: {sample_name} {model}")
                    row[model] = int(BLOCK.decode_counts(
                        probabilities, windows, reference, vocab, blank)["error"])
                output_rows.append(row)
                handle.write(json.dumps(row) + "\n")
            print(json.dumps({"shard": shard, "samples": len(output_rows)}), flush=True)
    if len(output_rows) != 6378:
        raise RuntimeError(f"sample count mismatch: {len(output_rows)}")
    keys = ("uniform",) + MODELS
    totals = {key: sum(row[key] for row in output_rows) for key in keys}
    reference_total = sum(row["ref_len"] for row in output_rows)
    comparisons = {}
    pairs = [(model, "uniform") for model in MODELS]
    pairs += [(model, "K") for model in ("KM", "KQ", "KMQ", "KMM2", "KX")]
    pairs += [("KMQ", "KX")]
    for left, right in pairs:
        key = f"{left}_vs_{right}"
        comparisons[key] = {
            "delta_errors": totals[left] - totals[right],
            "delta_wer_pp": 100.0 * (totals[left] - totals[right]) / reference_total,
            "source_cluster_bootstrap_ci95": bootstrap(output_rows, left, right),
        }
    checks = {
        "KMQ_vs_uniform_errors": totals["uniform"] - totals["KMQ"] >= 125,
        "KMQ_vs_uniform_ci": comparisons["KMQ_vs_uniform"]["source_cluster_bootstrap_ci95"][1] < 0,
        "KMQ_vs_K_errors": totals["K"] - totals["KMQ"] >= 25,
        "KMQ_vs_K_ci": comparisons["KMQ_vs_K"]["source_cluster_bootstrap_ci95"][1] < 0,
        "KMQ_vs_KX_errors": totals["KX"] - totals["KMQ"] >= 25,
        "KMQ_vs_KX_ci": comparisons["KMQ_vs_KX"]["source_cluster_bootstrap_ci95"][1] < 0,
    }
    passed = all(checks.values())
    metrics = {
        "status": "complete", "created_utc": now(), "gpu_used": False,
        "train_fit_only": True, "post_diagnostic": True, "independent_confirmation": False,
        "calibration_dev_test_heldout_read": False, "unknown_eos_closed_loop": False,
        "samples": len(output_rows), "sources": len({row["source"] for row in output_rows}),
        "reference_glosses": reference_total, "errors": totals,
        "wer": {key: 100.0 * value / reference_total for key, value in totals.items()},
        "comparisons": comparisons, "checks": checks, "primary_pattern_passed": passed,
        "decision": ("request explicit authorization for independent calibration confirmation"
                     if passed else "stop interpretable heatmap scheduler branch"),
        "no_unlock": True,
    }
    write_atomic(METRICS, metrics)
    manifest = {"status": "complete", "config_sha256": sha(CONFIG),
                "identity_sha256": sha(IDENTITY), "metrics_sha256": sha(METRICS),
                "per_sample_sha256": sha(PER_SAMPLE), "primary_pattern_passed": passed,
                "independent_confirmation": False, "no_unlock": True}
    write_atomic(OUTPUT / "manifest.json", manifest)
    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--prepare", action="store_true")
    parser.add_argument("--run", action="store_true")
    args = parser.parse_args()
    if args.prepare == args.run:
        parser.error("choose exactly one of --prepare or --run")
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    print(json.dumps(prepare() if args.prepare else run(), indent=2))
