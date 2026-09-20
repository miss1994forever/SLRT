#!/usr/bin/env python3
"""Audit eager-continuation versus predictor-continuation rollout labels.

This is a train-calibration, partial32, CPU-only diagnostic.  A predictor's
current input is causal.  Reference transcripts and future dense logits are
used only to score the two counterfactual branches.
"""
import argparse
import gzip
import hashlib
import importlib.util
import json
import math
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
OLD_ROOT = ROOT / "results/phoenix-2014t_ISLR/p3_partial32_rollout_predictor_closed_loop_smoke_v2_49faacc3"
DA_ROOT = ROOT / "results/phoenix-2014t_ISLR/p3_partial32_rollout_dagger_da1_smoke_v1_49faacc3"
OUTPUT_ROOT = ROOT / "results/phoenix-2014t_ISLR/p3_partial32_on_policy_advantage_smoke_v1_49faacc3"
SEQUENCES = Path("/tmp/phoenix_p2_train_center_sequences_v1.npz")
HORIZON = 16


def import_tool(name):
    path = ROOT / f"tools/{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


OLD = import_tool("analyze_phoenix_partial_rollout_predictor")
SELF = import_tool("build_phoenix_partial_rollout_dagger_self_dataset")
DA = import_tool("analyze_phoenix_partial_rollout_dagger_da1")
CHRON, DATA_BUILDER = OLD.DATA_BUILDER.CHRON, OLD.DATA_BUILDER


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_da(kind, key, root=DA_ROOT):
    checkpoint = torch.load(root / f"{key}.pt", map_location="cpu")
    if checkpoint["variant"] != kind or checkpoint["DA_round"] != 1:
        raise ValueError("unexpected DA checkpoint")
    stats = checkpoint["normalization_fit_only_source_balanced"]
    use_prefix, use_visual = OLD.VARIANTS[kind]
    model = OLD.RolloutPredictor(
        len(stats["bookkeeping"][0]), len(stats["prefix"][0]),
        len(stats["visual"][0]), use_prefix, use_visual,
    )
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    metrics = json.loads((root / "metrics.json").read_text())
    threshold = float(metrics["training"][key]["threshold"])
    return model, stats, threshold


def load_policies(old_root, da_root):
    result = {}
    for prefix, kind in (
        ("B2", "B2_bookkeeping_prefix"),
        ("B3", "B3_bookkeeping_visual_tcn_prefix"),
    ):
        result[f"{prefix}_old"] = (kind, *SELF.load_old(kind, old_root))
        result[f"{prefix}_DA1"] = (kind, *load_da(kind, f"{prefix}_DA1", da_root))
    return result


def branch_future(kind, model, stats, threshold, sequences, name, probabilities,
                  blank_id, selected, balance, current, execute_current,
                  horizon=HORIZON):
    """Take one branch action, then follow the same frozen predictor."""
    chosen = list(selected)
    value = float(balance)
    if execute_current:
        chosen.append(int(current))
        value -= 1.0
    stop = min(len(probabilities), int(current) + 1 + int(horizon))
    for start in range(int(current) + 1, stop):
        if CHRON.is_skeleton(start):
            chosen.append(start)
            continue
        value = CHRON.accrue_token(value)
        if not CHRON.token_eligible(value):
            continue
        execute = CHRON.token_forced(value)
        if not execute:
            row = OLD.online_feature_row(probabilities, chosen, start, value, blank_id)
            score = OLD.online_score(kind, model, stats, sequences, name, row, start)
            execute = score > threshold
        if execute:
            chosen.append(start)
            value -= 1.0
    return sorted(chosen)


def on_policy_advantage(kind, model, stats, threshold, sequences, name,
                        probabilities, selected, balance, current, reference,
                        vocab, blank_id):
    execute = branch_future(
        kind, model, stats, threshold, sequences, name, probabilities, blank_id,
        selected, balance, current, True,
    )
    skip = branch_future(
        kind, model, stats, threshold, sequences, name, probabilities, blank_id,
        selected, balance, current, False,
    )
    execute_hypothesis = DATA_BUILDER.ORACLE.decode_probabilities(
        probabilities, execute, vocab, blank_id
    )
    skip_hypothesis = DATA_BUILDER.ORACLE.decode_probabilities(
        probabilities, skip, vocab, blank_id
    )
    execute_error = DATA_BUILDER.ORACLE.error_count(reference, execute_hypothesis)
    skip_error = DATA_BUILDER.ORACLE.error_count(reference, skip_hypothesis)
    return int(skip_error - execute_error), int(skip_error), int(execute_error)


def sign(value):
    return (int(value) > 0) - (int(value) < 0)


def audit_sample(name, result, logits, vocab, blank_id, policy, sequences):
    kind, model, stats, threshold = policy
    probabilities = DATA_BUILDER.ORACLE.softmax_rows(np.asarray(logits))
    reference = DATA_BUILDER.REPLAY.clean_phoenix_2014_trans(result["gls_ref"])
    selected, balance = [], 0.0
    rows = []
    for start in range(len(probabilities)):
        if CHRON.is_skeleton(start):
            selected.append(start)
            continue
        balance = CHRON.accrue_token(balance)
        if not CHRON.token_eligible(balance):
            continue
        forced = CHRON.token_forced(balance)
        row = OLD.online_feature_row(probabilities, selected, start, balance, blank_id)
        score = OLD.online_score(kind, model, stats, sequences, name, row, start)
        execute = bool(forced or score > threshold)
        if not forced:
            eager, _, _ = CHRON.rollout_advantage(
                probabilities, selected, balance, start, reference, vocab,
                blank_id, horizon=HORIZON,
            )
            own, skip_error, execute_error = on_policy_advantage(
                kind, model, stats, threshold, sequences, name, probabilities,
                selected, balance, start, reference, vocab, blank_id,
            )
            rows.append({
                "sample_id": name,
                "candidate_window_start": start,
                "score": float(score),
                "threshold": float(threshold),
                "action_execute": execute,
                "eager_advantage": int(eager),
                "on_policy_advantage": int(own),
                "on_policy_skip_error": skip_error,
                "on_policy_execute_error": execute_error,
                "eager_sign": sign(eager),
                "on_policy_sign": sign(own),
            })
        if execute:
            selected.append(start)
            balance -= 1.0
    return rows


def summarize(rows):
    eager = np.asarray([row["eager_advantage"] for row in rows], dtype=np.int16)
    own = np.asarray([row["on_policy_advantage"] for row in rows], dtype=np.int16)
    action = np.asarray([row["action_execute"] for row in rows], dtype=bool)
    eager_sign = np.sign(eager)
    own_sign = np.sign(own)
    union_nonzero = (eager_sign != 0) | (own_sign != 0)
    own_positive = own > 0
    true_positive = action & own_positive
    precision = true_positive.sum() / max(1, action.sum())
    recall = true_positive.sum() / max(1, own_positive.sum())
    oracle_gain = np.maximum(own, 0).sum()
    selected_gain = own[action].sum()
    distribution = lambda values: {
        str(key): int(value) for key, value in sorted(Counter(values.tolist()).items())
    }
    correlation = None
    if len(rows) > 1 and np.std(eager) > 0 and np.std(own) > 0:
        correlation = float(np.corrcoef(eager, own)[0, 1])
    return {
        "rows": len(rows),
        "eager_advantage_counts": distribution(eager),
        "on_policy_advantage_counts": distribution(own),
        "exact_advantage_agreement": float(np.mean(eager == own)),
        "sign_agreement_all_rows": float(np.mean(eager_sign == own_sign)),
        "sign_agreement_union_nonzero": (
            float(np.mean(eager_sign[union_nonzero] == own_sign[union_nonzero]))
            if union_nonzero.any() else None
        ),
        "opposite_nonzero_sign_fraction_union_nonzero": (
            float(np.mean((eager_sign[union_nonzero] * own_sign[union_nonzero]) < 0))
            if union_nonzero.any() else None
        ),
        "eager_positive_onpolicy_nonpositive": int(((eager > 0) & (own <= 0)).sum()),
        "eager_nonpositive_onpolicy_positive": int(((eager <= 0) & (own > 0)).sum()),
        "pearson_advantage": correlation,
        "predictor_action": {
            "execute_count": int(action.sum()),
            "execute_rate": float(action.mean()),
            "on_policy_positive_count": int(own_positive.sum()),
            "positive_precision": float(precision),
            "positive_recall": float(recall),
            "harmful_execute_count": int((action & (own < 0)).sum()),
            "zero_execute_count": int((action & (own == 0)).sum()),
            "selected_signed_advantage_sum": int(selected_gain),
            "positive_oracle_advantage_sum": int(oracle_gain),
            "on_policy_action_regret": int(oracle_gain - selected_gain),
        },
    }


def run(old_root, da_root, sequences_path, output_root):
    for path in (old_root, da_root, sequences_path, output_root):
        OLD.reject_test_path(path)
    if output_root.exists():
        raise FileExistsError(output_root)
    temporary = output_root.with_name(f".{output_root.name}.incomplete")
    temporary.mkdir(parents=True)
    started = time.perf_counter()
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    sequences = OLD.SequenceFeatures(sequences_path)
    policies = load_policies(old_root, da_root)
    dense = CHRON.DEFAULT_DENSE_ROOT
    indices, shard_count = CHRON.completed_shard_indices(dense)
    assignments = json.loads((dense / "fit_calibration_split.json").read_text())["assignments"]
    vocab = json.loads(CHRON.DEFAULT_VOCAB.read_text())
    blank_id = vocab.index("<blank>")
    outputs = {key: [] for key in policies}
    trace_paths = {
        key: temporary / f"{key}_on_policy_audit.jsonl.gz" for key in policies
    }
    handles = {
        key: gzip.open(path, "wt", encoding="utf-8", compresslevel=5)
        for key, path in trace_paths.items()
    }
    sample_count = 0
    try:
        for shard in indices:
            results, logits, _ = DATA_BUILDER.BUILDER.validate_dense_shard(
                dense, shard, shard_count, verify_hashes=False
            )
            for name in results:
                if assignments[name] != "calibration" or name not in sequences.slices:
                    continue
                sample_count += 1
                for key, policy in policies.items():
                    rows = audit_sample(
                        name, results[name], logits[name], vocab, blank_id,
                        policy, sequences,
                    )
                    outputs[key].extend(rows)
                    for row in rows:
                        handles[key].write(json.dumps(row, separators=(",", ":")) + "\n")
    finally:
        for handle in handles.values():
            handle.close()
    metrics = {
        "scope": "partial32 train-calibration on-policy advantage audit smoke",
        "samples": sample_count,
        "horizon": HORIZON,
        "gpu_used": False,
        "dev_used": False,
        "test_used": False,
        "labels": {
            "eager": "execute/skip then fixed eager continuation for 16 candidates",
            "on_policy": "execute/skip then the same frozen predictor continuation for 16 candidates",
            "future_reference_usage": "offline label only",
        },
        "policies": {key: summarize(rows) for key, rows in outputs.items()},
        "runtime_seconds": time.perf_counter() - started,
        "limitations": [
            "32/56 non-random train shards",
            "calibration repeatedly inspected by earlier smokes",
            "future-aware reference labels are not deployable inputs",
            "no training, dev/test, wall-time, or latency claim",
        ],
    }
    metrics_path = temporary / "metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2) + "\n")
    manifest = {
        "status": "complete",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "metrics_sha256": sha256_file(metrics_path),
        "traces": {
            key: {"rows": len(outputs[key]), "sha256": sha256_file(path)}
            for key, path in trace_paths.items()
        },
    }
    (temporary / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    temporary.replace(output_root)
    return metrics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--old-root", type=Path, default=OLD_ROOT)
    parser.add_argument("--da-root", type=Path, default=DA_ROOT)
    parser.add_argument("--sequences", type=Path, default=SEQUENCES)
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    args = parser.parse_args()
    result = run(
        args.old_root.resolve(), args.da_root.resolve(),
        args.sequences.resolve(), args.output_root.resolve(),
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
