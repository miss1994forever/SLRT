#!/usr/bin/env python3
"""CPU-only terminal-advantage audit on a frozen partial32 calibration subset.

Future dense logits, the reference, and the true endpoint are used only to
construct offline diagnostic labels/oracles.  Every deployed-policy feature
and action remains causal and contains neither total length nor EOS.
"""
import argparse
import gzip
import hashlib
import importlib.util
import json
import os
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
OLD_ROOT = ROOT / "results/phoenix-2014t_ISLR/p3_partial32_rollout_predictor_closed_loop_smoke_v2_49faacc3"
PI1_ROOT = ROOT / "results/phoenix-2014t_ISLR/p3_partial32_on_policy_pi1_smoke_v1_49faacc3"
OUTPUT_ROOT = ROOT / "results/phoenix-2014t_ISLR/p3_partial32_terminal_advantage_audit_smoke_v1_49faacc3"
SEQUENCES = Path("/tmp/phoenix_p2_train_center_sequences_v1.npz")
KIND = "B2_bookkeeping_prefix"
SUBSET_SIZE = 128
SUBSET_SEED = "terminal-advantage-audit-v1"
HORIZON = 16


def imp(name):
    path = ROOT / f"tools/{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


OLD = imp("analyze_phoenix_partial_rollout_predictor")
SELF = imp("build_phoenix_partial_rollout_dagger_self_dataset")
PI1 = imp("analyze_phoenix_partial_on_policy_pi1")
AUDIT = imp("analyze_phoenix_partial_on_policy_advantage")
CHRON, BUILDER = OLD.DATA_BUILDER.CHRON, OLD.DATA_BUILDER


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def stable_rank(name):
    return hashlib.sha256(f"{SUBSET_SEED}\0{name}".encode("utf-8")).hexdigest()


def load_pi1(root=PI1_ROOT):
    checkpoint = torch.load(root / "B2_PI1.pt", map_location="cpu")
    if checkpoint["variant"] != KIND or checkpoint["policy_iteration"] != 1:
        raise ValueError("unexpected PI1 checkpoint")
    stats = checkpoint["normalization_fit_only"]
    model = OLD.RolloutPredictor(
        len(stats["bookkeeping"][0]), len(stats["prefix"][0]),
        len(stats["visual"][0]), True, False,
    )
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    metrics = json.loads((root / "metrics.json").read_text())
    threshold = float(metrics["training"]["primary_threshold"])
    return model, stats, threshold


def load_policies():
    return {
        "B2_old": (KIND, *SELF.load_old(KIND, OLD_ROOT)),
        "B2_PI1_primary": (KIND, *load_pi1(PI1_ROOT)),
    }


def frozen_config(sample_ids):
    return {
        "experiment": "terminal-advantage audit smoke",
        "created_before_outcome_computation": True,
        "cpu_only": True,
        "train_partition": "partial32 frozen calibration only",
        "subset_selection": {
            "rule": "lowest sha256(seed + NUL + sample_id)",
            "seed": SUBSET_SEED,
            "size": SUBSET_SIZE,
            "sample_ids": sample_ids,
            "sample_ids_sha256": hashlib.sha256("\n".join(sample_ids).encode()).hexdigest(),
        },
        "policies": {
            "B2_old": {"checkpoint_sha256": sha256_file(OLD_ROOT / f"{KIND}.pt")},
            "B2_PI1_primary": {"checkpoint_sha256": sha256_file(PI1_ROOT / "B2_PI1.pt"), "threshold": 0.0},
        },
        "protocol": {
            "skeleton": "start % 4 == 0", "bonus_accrual": "1/3",
            "initial_tokens": 0, "capacity": 2, "eligible": 1,
            "forced": 2, "eos_topup": False, "unknown_T_EOS_for_actions": True,
            "local_horizon": HORIZON,
        },
        "terminal_label": "execute/skip now, then same frozen base policy to true sample end",
        "terminal_policy_improvement": "at each autonomous state act iff A_terminal^base > 0; forced actions unchanged",
        "forbidden": ["GPU/CUDA", "dev", "test", "training", "PI2", "T/EOS in state or deployable action"],
    }


def calibration_ids():
    dense = CHRON.DEFAULT_DENSE_ROOT
    indices, shard_count = CHRON.completed_shard_indices(dense)
    if len(indices) != 32:
        raise ValueError("requires the frozen partial32 dense set")
    split = json.loads((dense / "fit_calibration_split.json").read_text())
    assignments = split["assignments"]
    sequences = OLD.SequenceFeatures(SEQUENCES)
    names = []
    for index in indices:
        results, _, _ = BUILDER.BUILDER.validate_dense_shard(
            dense, index, shard_count, verify_hashes=False
        )
        names.extend(name for name in results
                     if assignments[name] == "calibration" and name in sequences.slices)
    return sorted(set(names))


def preregister(output_root=OUTPUT_ROOT):
    if "dev" in str(output_root).lower() or "test" in str(output_root).lower():
        raise ValueError("dev/test path forbidden")
    output_root.mkdir(parents=True, exist_ok=False)
    names = calibration_ids()
    if len(names) < SUBSET_SIZE:
        raise ValueError("not enough calibration samples")
    chosen = sorted(names, key=lambda x: (stable_rank(x), x))[:SUBSET_SIZE]
    config = frozen_config(chosen)
    path = output_root / "resolved_config_preregistered.json"
    path.write_text(json.dumps(config, indent=2) + "\n")
    (output_root / "workers").mkdir()
    return config


def load_config(output_root=OUTPUT_ROOT):
    path = output_root / "resolved_config_preregistered.json"
    config = json.loads(path.read_text())
    ids = config["subset_selection"]["sample_ids"]
    expected = hashlib.sha256("\n".join(ids).encode()).hexdigest()
    if expected != config["subset_selection"]["sample_ids_sha256"]:
        raise ValueError("frozen subset hash mismatch")
    if config != frozen_config(ids):
        # created_utc is intentionally absent, making this exact comparison stable.
        raise ValueError("preregistered protocol changed")
    return config


def branch_to_end(policy, sequences, name, probabilities, selected, balance,
                  current, execute_current, blank_id):
    """Branch without mutating selected, then use exactly the frozen base policy."""
    kind, model, stats, threshold = policy
    return AUDIT.branch_future(
        kind, model, stats, threshold, sequences, name, probabilities, blank_id,
        selected, balance, current, execute_current, horizon=len(probabilities),
    )


def advantage_from_selections(probabilities, execute, skip, reference, vocab, blank_id):
    execute_hyp = BUILDER.ORACLE.decode_probabilities(probabilities, execute, vocab, blank_id)
    skip_hyp = BUILDER.ORACLE.decode_probabilities(probabilities, skip, vocab, blank_id)
    execute_error = BUILDER.ORACLE.error_count(reference, execute_hyp)
    skip_error = BUILDER.ORACLE.error_count(reference, skip_hyp)
    return int(skip_error - execute_error), int(skip_error), int(execute_error)


def terminal_advantage(policy, sequences, name, probabilities, selected, balance,
                       current, reference, vocab, blank_id):
    execute = branch_to_end(policy, sequences, name, probabilities, selected,
                            balance, current, True, blank_id)
    skip = branch_to_end(policy, sequences, name, probabilities, selected,
                         balance, current, False, blank_id)
    return advantage_from_selections(probabilities, execute, skip, reference, vocab, blank_id)


def run_frozen_policy(name, probabilities, reference, vocab, blank_id, policy, sequences,
                      collect_labels=False):
    kind, model, stats, threshold = policy
    selected, balance, rows = [], 0.0, []
    forced_actions = autonomous = triggers = 0
    total = len(probabilities)
    for start in range(total):
        if CHRON.is_skeleton(start):
            selected.append(start)
            continue
        balance = CHRON.accrue_token(balance)
        if not CHRON.token_eligible(balance):
            continue
        forced = CHRON.token_forced(balance)
        execute = forced
        if forced:
            forced_actions += 1
        else:
            autonomous += 1
            row = OLD.online_feature_row(probabilities, selected, start, balance, blank_id)
            score = OLD.online_score(kind, model, stats, sequences, name, row, start)
            execute = score > threshold
            triggers += int(execute)
            if collect_labels:
                local, _, _ = AUDIT.on_policy_advantage(
                    kind, model, stats, threshold, sequences, name, probabilities,
                    selected, balance, start, reference, vocab, blank_id,
                )
                terminal, skip_error, execute_error = terminal_advantage(
                    policy, sequences, name, probabilities, selected, balance,
                    start, reference, vocab, blank_id,
                )
                rows.append({
                    "sample_id": name, "candidate_window_start": start,
                    "remaining_windows_to_eos": total - 1 - start,
                    "action_execute": bool(execute), "score": float(score),
                    "local16_advantage": int(local),
                    "terminal_advantage": int(terminal),
                    "terminal_skip_error": skip_error,
                    "terminal_execute_error": execute_error,
                    "provenance": {
                        "state_and_action_strictly_causal": True,
                        "T_EOS_absent_from_state_action": True,
                        "future_reference_endpoint_label_only": True,
                        "same_frozen_policy_continuation": True,
                        "forced_state": False,
                    },
                })
        if execute:
            selected.append(start)
            balance -= 1.0
    _, counts = CHRON.decode_counts(probabilities, selected, reference, vocab, blank_id)
    return {
        "selected": selected, "counts": counts, "dense_windows": total,
        "actual_window_rate": len(selected) / total, "unspent_bonus_tokens": balance,
        "forced_actions": forced_actions, "autonomous_decisions": autonomous,
        "autonomous_triggers": triggers,
        "coverage": CHRON.coverage_metrics(selected, total), "audit_rows": rows,
    }


def run_terminal_oracle(name, probabilities, reference, vocab, blank_id, base_policy, sequences):
    selected, balance, trace = [], 0.0, []
    forced_actions = autonomous = positive_actions = 0
    total = len(probabilities)
    for start in range(total):
        if CHRON.is_skeleton(start):
            selected.append(start)
            continue
        balance = CHRON.accrue_token(balance)
        if not CHRON.token_eligible(balance):
            continue
        forced = CHRON.token_forced(balance)
        execute = forced
        if forced:
            forced_actions += 1
        else:
            autonomous += 1
            advantage, skip_error, execute_error = terminal_advantage(
                base_policy, sequences, name, probabilities, selected, balance,
                start, reference, vocab, blank_id,
            )
            execute = advantage > 0
            positive_actions += int(execute)
            trace.append({"candidate_window_start": start,
                          "terminal_advantage_under_base": advantage,
                          "action_execute": bool(execute),
                          "skip_error": skip_error, "execute_error": execute_error})
        if execute:
            selected.append(start)
            balance -= 1.0
    _, counts = CHRON.decode_counts(probabilities, selected, reference, vocab, blank_id)
    return {
        "selected": selected, "counts": counts, "dense_windows": total,
        "actual_window_rate": len(selected) / total, "unspent_bonus_tokens": balance,
        "forced_actions": forced_actions, "autonomous_decisions": autonomous,
        "positive_oracle_actions": positive_actions,
        "coverage": CHRON.coverage_metrics(selected, total), "trace": trace,
    }


def uniform_result(probabilities, reference, vocab, blank_id):
    selected, balance = CHRON.online_uniform_schedule(len(probabilities))
    _, counts = CHRON.decode_counts(probabilities, selected, reference, vocab, blank_id)
    return {"selected": selected, "counts": counts, "dense_windows": len(probabilities),
            "actual_window_rate": len(selected) / len(probabilities),
            "unspent_bonus_tokens": balance,
            "coverage": CHRON.coverage_metrics(selected, len(probabilities))}


def process_sample(name, result, logits, policies, sequences, vocab, blank_id):
    probabilities = BUILDER.ORACLE.softmax_rows(np.asarray(logits))
    reference = BUILDER.REPLAY.clean_phoenix_2014_trans(result["gls_ref"])
    starts = [int(x["start"]) for x in result["adaptive_stride_metadata"]]
    if starts != list(range(len(probabilities))):
        raise ValueError(f"non-dense coordinates: {name}")
    output = {"sample_id": name, "uniform": uniform_result(
        probabilities, reference, vocab, blank_id), "policies": {}}
    for key, policy in policies.items():
        output["policies"][key] = {
            "base": run_frozen_policy(name, probabilities, reference, vocab, blank_id,
                                      policy, sequences, collect_labels=True),
            "terminal_oracle": run_terminal_oracle(
                name, probabilities, reference, vocab, blank_id, policy, sequences),
        }
    return output


def worker(output_root, worker_index, workers):
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    config = load_config(output_root)
    chosen = config["subset_selection"]["sample_ids"]
    assigned = {name for pos, name in enumerate(chosen) if pos % workers == worker_index}
    policies = load_policies()
    sequences = OLD.SequenceFeatures(SEQUENCES)
    vocab = json.loads(CHRON.DEFAULT_VOCAB.read_text())
    blank_id = vocab.index("<blank>")
    dense = CHRON.DEFAULT_DENSE_ROOT
    indices, shard_count = CHRON.completed_shard_indices(dense)
    final = output_root / "workers" / f"worker-{worker_index:02d}-of-{workers:02d}.jsonl.gz"
    marker = final.with_suffix(".complete.json")
    if final.exists() or marker.exists():
        raise FileExistsError(final)
    temporary = final.with_name(f".{final.name}.incomplete-{os.getpid()}")
    started, done = time.perf_counter(), 0
    with gzip.open(temporary, "wt", encoding="utf-8", compresslevel=5) as handle:
        for index in indices:
            results, logits, _ = BUILDER.BUILDER.validate_dense_shard(
                dense, index, shard_count, verify_hashes=False)
            for name in results:
                if name not in assigned:
                    continue
                value = process_sample(name, results[name], logits[name], policies,
                                       sequences, vocab, blank_id)
                handle.write(json.dumps(value, separators=(",", ":")) + "\n")
                done += 1
                print(json.dumps({"worker": worker_index, "done": done,
                                  "assigned": len(assigned), "sample": name,
                                  "elapsed_seconds": time.perf_counter() - started}), flush=True)
    if done != len(assigned):
        raise RuntimeError(f"worker {worker_index}: expected {len(assigned)}, got {done}")
    temporary.replace(final)
    info = {"status": "complete", "worker": worker_index, "workers": workers,
            "samples": done, "seconds": time.perf_counter() - started,
            "sha256": sha256_file(final), "bytes": final.stat().st_size}
    marker.write_text(json.dumps(info, indent=2) + "\n")
    return info


def sign(value):
    return (int(value) > 0) - (int(value) < 0)


def label_summary(rows):
    local = np.asarray([x["local16_advantage"] for x in rows], dtype=np.int16)
    terminal = np.asarray([x["terminal_advantage"] for x in rows], dtype=np.int16)
    ls, ts = np.sign(local), np.sign(terminal)
    union = (ls != 0) | (ts != 0)
    correlation = None
    if len(rows) > 1 and np.std(local) > 0 and np.std(terminal) > 0:
        correlation = float(np.corrcoef(local, terminal)[0, 1])
    distribution = lambda x: {str(k): int(v) for k, v in sorted(Counter(x.tolist()).items())}
    return {
        "states": len(rows), "local16_counts": distribution(local),
        "terminal_counts": distribution(terminal),
        "exact_advantage_agreement": float(np.mean(local == terminal)) if len(rows) else None,
        "sign_agreement_all": float(np.mean(ls == ts)) if len(rows) else None,
        "union_nonzero_states": int(union.sum()),
        "sign_agreement_union_nonzero": float(np.mean(ls[union] == ts[union])) if union.any() else None,
        "direct_opposite_signs": int(((ls * ts) < 0).sum()),
        "local_zero_terminal_nonzero": int(((ls == 0) & (ts != 0)).sum()),
        "local_nonzero_terminal_zero": int(((ls != 0) & (ts == 0)).sum()),
        "local_positive_terminal_nonpositive": int(((local > 0) & (terminal <= 0)).sum()),
        "local_nonpositive_terminal_positive": int(((local <= 0) & (terminal > 0)).sum()),
        "pearson_advantage": correlation,
    }


def distance_bin(remaining):
    if remaining <= 15:
        return "0-15"
    if remaining <= 31:
        return "16-31"
    if remaining <= 63:
        return "32-63"
    return "64+"


def aggregate_strategy(rows, uniform_rows):
    counts = CHRON.aggregate_counts([x["counts"] for x in rows])
    uniform = CHRON.aggregate_counts([x["counts"] for x in uniform_rows])
    dense = sum(x["dense_windows"] for x in rows)
    return {
        "metrics": counts,
        "delta_wer_pp_vs_online_uniform": counts["wer"] - uniform["wer"],
        "error_delta_vs_online_uniform": counts["error"] - uniform["error"],
        "executed_windows": sum(len(x["selected"]) for x in rows),
        "dense_windows": dense,
        "actual_window_rate": sum(len(x["selected"]) for x in rows) / dense,
        "max_gap": max(x["coverage"]["max_gap_including_endpoints"] for x in rows),
        "paired_bootstrap_vs_online_uniform": CHRON.paired_bootstrap(
            [x["counts"] for x in rows], [x["counts"] for x in uniform_rows]),
    }


def finalize(output_root, workers):
    config = load_config(output_root)
    records, worker_info = [], []
    for index in range(workers):
        path = output_root / "workers" / f"worker-{index:02d}-of-{workers:02d}.jsonl.gz"
        marker = path.with_suffix(".complete.json")
        info = json.loads(marker.read_text())
        if info["sha256"] != sha256_file(path):
            raise ValueError(f"worker hash mismatch: {index}")
        worker_info.append(info)
        with gzip.open(path, "rt", encoding="utf-8") as handle:
            records.extend(json.loads(line) for line in handle)
    expected = config["subset_selection"]["sample_ids"]
    if sorted(x["sample_id"] for x in records) != sorted(expected):
        raise ValueError("worker outputs do not exactly cover frozen subset")
    uniform_rows = [x["uniform"] for x in records]
    uniform_summary = aggregate_strategy(uniform_rows, uniform_rows)
    full_metrics = json.loads((PI1_ROOT / "metrics.json").read_text())["closed_loop"]["policies"]
    policies = {}
    for key in ("B2_old", "B2_PI1_primary"):
        base = [x["policies"][key]["base"] for x in records]
        oracle = [x["policies"][key]["terminal_oracle"] for x in records]
        audit_rows = [row for sample in base for row in sample.pop("audit_rows")]
        base_summary = aggregate_strategy(base, uniform_rows)
        oracle_summary = aggregate_strategy(oracle, uniform_rows)
        oracle_summary["delta_wer_pp_vs_base"] = (
            oracle_summary["metrics"]["wer"] - base_summary["metrics"]["wer"])
        oracle_summary["error_delta_vs_base"] = (
            oracle_summary["metrics"]["error"] - base_summary["metrics"]["error"])
        oracle_summary["paired_bootstrap_vs_base"] = CHRON.paired_bootstrap(
            [x["counts"] for x in oracle], [x["counts"] for x in base])
        by_distance = {}
        for label in ("0-15", "16-31", "32-63", "64+"):
            group = [row for row in audit_rows
                     if distance_bin(row["remaining_windows_to_eos"]) == label]
            by_distance[label] = label_summary(group)
        policies[key] = {
            "local16_vs_terminal_labels": label_summary(audit_rows),
            "by_remaining_windows_to_eos": by_distance,
            "base_subset": base_summary,
            "terminal_policy_improvement_oracle_subset": oracle_summary,
            "full526_reference": full_metrics[key],
        }
    result = {
        "scope": "preregistered partial32 train-calibration terminal-advantage audit smoke",
        "samples": len(records), "gpu_used": False, "dev_used": False,
        "test_used": False, "trained_model": False,
        "subset": config["subset_selection"], "online_uniform_subset": uniform_summary,
        "online_uniform_full526_reference": full_metrics["online_uniform"],
        "policies": policies,
        "interpretation_rule": {
            "horizon_mismatch": "terminal oracle improves and local16/terminal union-nonzero sign agreement is low",
            "stop_current_utility": "terminal oracle does not improve",
            "other_bottleneck": "sign agreement is high but learned policy still fails",
        },
        "limitations": [
            "32/56 non-random train shards", "train calibration repeatedly inspected",
            "deterministic 128-sample subset, not redrawn after outcomes",
            "terminal labels/oracles use future, reference, and true endpoint and are not deployable",
            "no dev/test claim", "no formal wall-time or submission-latency conclusion",
        ],
        "workers": worker_info,
    }
    temporary = output_root / ".metrics.json.incomplete"
    temporary.write_text(json.dumps(result, indent=2) + "\n")
    temporary.replace(output_root / "metrics.json")
    manifest = {"status": "complete", "created_utc": datetime.now(timezone.utc).isoformat(),
                "config_sha256": sha256_file(output_root / "resolved_config_preregistered.json"),
                "metrics_sha256": sha256_file(output_root / "metrics.json"),
                "worker_outputs": worker_info}
    temporary = output_root / ".manifest.json.incomplete"
    temporary.write_text(json.dumps(manifest, indent=2) + "\n")
    temporary.replace(output_root / "manifest.json")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("preregister", "worker", "finalize"))
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    parser.add_argument("--worker-index", type=int, default=0)
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args()
    output = args.output_root.resolve()
    if args.mode == "preregister":
        value = preregister(output)
    elif args.mode == "worker":
        value = worker(output, args.worker_index, args.workers)
    else:
        value = finalize(output, args.workers)
    print(json.dumps(value, indent=2))


if __name__ == "__main__":
    main()
