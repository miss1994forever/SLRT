#!/usr/bin/env python3
"""Preregistered CPU-only predictor smoke for terminal advantage.

The true endpoint, reference, and future dense logits are used only to build
offline terminal-advantage labels.  Deployable features and actions contain
only bookkeeping and decoder-prefix information available before the current
candidate is executed.
"""
import argparse, gzip, hashlib, importlib.util, json, math, os, random, sys, time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
OLD_ROOT = ROOT / "results/phoenix-2014t_ISLR/p3_partial32_rollout_predictor_closed_loop_smoke_v2_49faacc3"
AUDIT_ROOT = ROOT / "results/phoenix-2014t_ISLR/p3_partial32_terminal_advantage_audit_smoke_v1_49faacc3"
OUTPUT_ROOT = ROOT / "results/phoenix-2014t_ISLR/p3_partial32_terminal_predictor_smoke_v1_49faacc3"
DATA_ROOT = OUTPUT_ROOT.with_name(OUTPUT_ROOT.name + "_dataset")
SEQUENCES = Path("/tmp/phoenix_p2_train_center_sequences_v1.npz")
KIND0, KIND1 = "B0_bookkeeping", "B2_bookkeeping_prefix"
FIT_SIZE, CAL_SIZE = 512, 128
SUBSET_SEED = "terminal-predictor-fit-v1"
TRAIN_SEED = 261016


def imp(name):
    path = ROOT / f"tools/{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


OLD = imp("analyze_phoenix_partial_rollout_predictor")
SELF = imp("build_phoenix_partial_rollout_dagger_self_dataset")
AUDIT = imp("analyze_phoenix_partial_terminal_advantage_audit")
CHRON, BUILDER = OLD.DATA_BUILDER.CHRON, OLD.DATA_BUILDER


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def stable_rank(name):
    return hashlib.sha256(f"{SUBSET_SEED}\0{name}".encode()).hexdigest()


def available_ids():
    dense = CHRON.DEFAULT_DENSE_ROOT
    indices, shard_count = CHRON.completed_shard_indices(dense)
    if len(indices) != 32:
        raise ValueError("requires frozen partial32 dense shards")
    split = json.loads((dense / "fit_calibration_split.json").read_text())
    assignments = split["assignments"]
    sequences = OLD.SequenceFeatures(SEQUENCES)
    fit = []
    calibration = []
    for index in indices:
        results, _, _ = BUILDER.BUILDER.validate_dense_shard(
            dense, index, shard_count, verify_hashes=False)
        for name in results:
            if name not in sequences.slices:
                continue
            (fit if assignments[name] == "fit" else calibration).append(name)
    return sorted(set(fit)), sorted(set(calibration)), assignments


def frozen_config():
    fit, calibration, assignments = available_ids()
    audit_config = json.loads((AUDIT_ROOT / "resolved_config_preregistered.json").read_text())
    cal = audit_config["subset_selection"]["sample_ids"]
    if len(cal) != CAL_SIZE or not set(cal).issubset(calibration):
        raise ValueError("frozen audit calibration subset unavailable")
    chosen = sorted(fit, key=lambda x: (stable_rank(x), x))[:FIT_SIZE]
    if len(chosen) != FIT_SIZE:
        raise ValueError("not enough fit samples")
    fit_sources = {BUILDER.source_video(x) for x in chosen}
    cal_sources = {BUILDER.source_video(x) for x in cal}
    if fit_sources & cal_sources:
        raise ValueError("fit/calibration source-video overlap")
    return {
        "experiment": "strict-causal terminal-advantage predictor smoke",
        "created_before_label_or_outcome_computation": True,
        "cpu_only": True,
        "partial_dense_shards": 32,
        "base_policy": {
            "name": KIND1,
            "checkpoint_sha256": sha256_file(OLD_ROOT / f"{KIND1}.pt"),
            "threshold_source": "frozen fit-only threshold in old metrics",
        },
        "subsets": {
            "fit": {"rule": "lowest sha256(seed+NUL+sample_id)", "seed": SUBSET_SEED,
                    "size": FIT_SIZE, "sample_ids": chosen,
                    "sha256": hashlib.sha256("\n".join(chosen).encode()).hexdigest()},
            "calibration": {"rule": "exact frozen terminal-audit 128", "size": CAL_SIZE,
                    "sample_ids": cal,
                    "sha256": hashlib.sha256("\n".join(cal).encode()).hexdigest()},
            "source_video_disjoint": True,
        },
        "protocol": {"skeleton": "start%4==0", "bonus_accrual": "1/3",
            "initial_tokens": 0, "capacity": 2, "eligible": 1, "forced": 2,
            "eos_topup": False, "unknown_T_EOS_for_actions": True,
            "terminal_label": "execute/skip now then same frozen B2-old to true end",
            "training_rows": "autonomous eligible non-forced only"},
        "models": {"T0": "bookkeeping-only", "T1": "bookkeeping+past-selected decoder prefix",
            "architecture": "same small MLP as rollout predictor", "epochs": 5,
            "optimizer": "AdamW(lr=1e-3,weight_decay=1e-4)",
            "loss": "SmoothL1 signed terminal advantage + 0.25 capped-positive BCE",
            "same_training_budget": True},
        "gate": ["T1 PR-AUC > T0", "T1 recall@10 > T0", "T1 NDCG@10 > T0",
                 "T1 score-decile Spearman > 0", "T1 highest decile mean advantage > lowest"],
        "primary_threshold": "fit regression expected terminal advantage > 0",
        "sensitivity_threshold": "fit regression-score 90th percentile; report only",
        "forbidden": ["GPU/CUDA", "dev", "test", "visual", "LM",
            "current candidate logits as input", "future logits as input", "reference as input",
            "T/EOS/remaining windows as input", "calibration threshold tuning", "PI2/DA"],
    }


def preregister(root=DATA_ROOT):
    if root.exists():
        raise FileExistsError(root)
    config = frozen_config()
    root.mkdir(parents=True)
    (root / "workers").mkdir()
    (root / "resolved_config_preregistered.json").write_text(json.dumps(config, indent=2) + "\n")
    return config


def load_config(root=DATA_ROOT):
    config = json.loads((root / "resolved_config_preregistered.json").read_text())
    for split in ("fit", "calibration"):
        ids = config["subsets"][split]["sample_ids"]
        if hashlib.sha256("\n".join(ids).encode()).hexdigest() != config["subsets"][split]["sha256"]:
            raise ValueError("frozen subset hash mismatch")
    if config != frozen_config():
        raise ValueError("preregistered configuration changed")
    return config


def causal_feature(row):
    return {"bookkeeping": row["bookkeeping"], "prefix": row["prefix"]}


def sample_rows(name, partition, result, logits, base, sequences, vocab, blank_id):
    probabilities = BUILDER.ORACLE.softmax_rows(np.asarray(logits))
    reference = BUILDER.REPLAY.clean_phoenix_2014_trans(result["gls_ref"])
    selected, balance, rows = [], 0.0, []
    counts = Counter(samples=1, dense=len(probabilities))
    for start in range(len(probabilities)):
        if CHRON.is_skeleton(start):
            selected.append(start); counts["skeleton"] += 1; continue
        balance = CHRON.accrue_token(balance)
        if not CHRON.token_eligible(balance):
            counts["ineligible"] += 1; continue
        forced = CHRON.token_forced(balance)
        execute = forced
        if forced:
            counts["forced"] += 1
        else:
            row = OLD.online_feature_row(probabilities, selected, start, balance, blank_id)
            score = OLD.online_score(KIND1, base[1], base[2], sequences, name, row, start)
            execute = score > base[3]
            advantage, skip_error, execute_error = AUDIT.terminal_advantage(
                base, sequences, name, probabilities, selected, balance, start,
                reference, vocab, blank_id)
            rows.append({"sample_id": name, "source_video_id": BUILDER.source_video(name),
                "partition": partition, "candidate_window_start": start,
                **causal_feature(row),
                "label": {"terminal_signed_advantage": advantage, "positive": advantage > 0,
                          "skip_error": skip_error, "execute_error": execute_error},
                "base": {"score": score, "action_execute": bool(execute)},
                "provenance": {"base_own_past_selected_prefix": True, "forced_state": False,
                    "future_reference_endpoint_label_only": True,
                    "current_candidate_logits_absent_from_input": True,
                    "T_EOS_remaining_absent_from_input": True,
                    "continuation_same_frozen_B2_old": True}})
            counts["autonomous"] += 1
            counts["positive"] += int(advantage > 0)
            counts["zero"] += int(advantage == 0)
            counts["negative"] += int(advantage < 0)
            # Distance is label-side reporting only and is never serialized as a feature.
            remain = len(probabilities) - 1 - start
            bucket = "0-15" if remain <= 15 else "16-31" if remain <= 31 else "32-63" if remain <= 63 else "64+"
            counts[f"distance_{bucket}_rows"] += 1
            counts[f"distance_{bucket}_positive"] += int(advantage > 0)
            counts[f"distance_{bucket}_zero"] += int(advantage == 0)
            counts[f"distance_{bucket}_negative"] += int(advantage < 0)
        if execute:
            selected.append(start); balance -= 1.0
    counts["executed"] += len(selected)
    return rows, counts


def worker(root, worker_index, workers):
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    torch.set_num_threads(1); torch.use_deterministic_algorithms(True)
    config = load_config(root)
    all_ids = [(x, "fit") for x in config["subsets"]["fit"]["sample_ids"]] + [
               (x, "calibration") for x in config["subsets"]["calibration"]["sample_ids"]]
    assigned = {x: p for pos, (x, p) in enumerate(all_ids) if pos % workers == worker_index}
    final = root / "workers" / f"worker-{worker_index:02d}-of-{workers:02d}.jsonl.gz"
    marker = final.with_suffix(".complete.json")
    if final.exists() and marker.exists() and json.loads(marker.read_text())["sha256"] == sha256_file(final):
        return json.loads(marker.read_text()) | {"resumed": True}
    if final.exists() or marker.exists():
        raise FileExistsError(final)
    base = (KIND1, *SELF.load_old(KIND1, OLD_ROOT))
    sequences = OLD.SequenceFeatures(SEQUENCES)
    vocab = json.loads(CHRON.DEFAULT_VOCAB.read_text()); blank_id = vocab.index("<blank>")
    dense = CHRON.DEFAULT_DENSE_ROOT; indices, shard_count = CHRON.completed_shard_indices(dense)
    temporary = final.with_name("." + final.name + f".incomplete-{os.getpid()}")
    total, done, started = Counter(), 0, time.perf_counter()
    with gzip.open(temporary, "wt", encoding="utf-8", compresslevel=5) as handle:
        for index in indices:
            results, logits, _ = BUILDER.BUILDER.validate_dense_shard(dense, index, shard_count, verify_hashes=False)
            for name in results:
                if name not in assigned:
                    continue
                rows, counts = sample_rows(name, assigned[name], results[name], logits[name],
                                           base, sequences, vocab, blank_id)
                for row in rows:
                    handle.write(json.dumps(row, separators=(",", ":")) + "\n")
                total.update(counts); done += 1
                print(json.dumps({"worker": worker_index, "done": done, "assigned": len(assigned),
                                  "sample": name, "elapsed_seconds": time.perf_counter()-started}), flush=True)
    if done != len(assigned):
        raise RuntimeError(f"worker {worker_index}: expected {len(assigned)}, got {done}")
    temporary.replace(final)
    info = {"status": "complete", "worker": worker_index, "workers": workers,
            "samples": done, "counts": dict(total), "seconds": time.perf_counter()-started,
            "sha256": sha256_file(final), "bytes": final.stat().st_size}
    marker.write_text(json.dumps(info, indent=2) + "\n")
    return info


def finalize_dataset(root, workers):
    config = load_config(root); counts = Counter(); info = []
    for index in range(workers):
        path = root / "workers" / f"worker-{index:02d}-of-{workers:02d}.jsonl.gz"
        marker = path.with_suffix(".complete.json")
        value = json.loads(marker.read_text())
        if value["sha256"] != sha256_file(path): raise ValueError("worker hash mismatch")
        counts.update(value["counts"]); info.append(value)
    if counts["samples"] != FIT_SIZE + CAL_SIZE: raise ValueError("sample count mismatch")
    manifest = {"status": "complete", "created_utc": datetime.now(timezone.utc).isoformat(),
        "config_sha256": sha256_file(root / "resolved_config_preregistered.json"),
        "counts": dict(counts), "positive_prevalence": counts["positive"] / counts["autonomous"],
        "workers": info}
    temporary = root / ".dataset_manifest.json.incomplete"
    temporary.write_text(json.dumps(manifest, indent=2) + "\n"); temporary.replace(root / "dataset_manifest.json")
    return manifest


def iter_rows(root):
    manifest = json.loads((root / "dataset_manifest.json").read_text())
    if manifest["status"] != "complete": raise ValueError("dataset incomplete")
    for info in manifest["workers"]:
        path = root / "workers" / f"worker-{info['worker']:02d}-of-{info['workers']:02d}.jsonl.gz"
        if sha256_file(path) != info["sha256"]: raise ValueError("dataset hash failure")
        with gzip.open(path, "rt", encoding="utf-8") as handle:
            for line in handle:
                yield json.loads(line)


def load_data(root):
    grouped = {"fit": [], "calibration": []}
    sources = {"fit": set(), "calibration": set()}
    for row in iter_rows(root):
        p = row["provenance"]
        if not (p["base_own_past_selected_prefix"] and not p["forced_state"] and
                p["future_reference_endpoint_label_only"] and
                p["current_candidate_logits_absent_from_input"] and
                p["T_EOS_remaining_absent_from_input"] and
                p["continuation_same_frozen_B2_old"]):
            raise ValueError("causality provenance failure")
        value = {"bookkeeping": OLD.bookkeeping_features(row), "prefix": OLD.prefix_features(row),
                 "visual": np.zeros((31, 1), dtype=np.float16),
                 "advantage": float(row["label"]["terminal_signed_advantage"]),
                 "positive": float(row["label"]["positive"])}
        grouped[row["partition"]].append(value); sources[row["partition"]].add(row["source_video_id"])
    if sources["fit"] & sources["calibration"]: raise ValueError("source-video leakage")
    return {key: OLD.stack_rows(value) for key, value in grouped.items()}, sources


def gate(metrics):
    t0, t1 = metrics["T0"], metrics["T1"]
    checks = {"T1_PR_AUC_gt_T0": t1["pr_auc_regression_score"] > t0["pr_auc_regression_score"],
        "T1_recall_at_10pct_gt_T0": t1["positive_recall_at_top_10pct"] > t0["positive_recall_at_top_10pct"],
        "T1_NDCG_at_10pct_gt_T0": t1["ndcg_at_top_10pct"] > t0["ndcg_at_top_10pct"],
        "T1_decile_spearman_gt_0": t1["score_decile_spearman"] > 0,
        "T1_highest_decile_gt_lowest": t1["highest_minus_lowest_decile_mean_advantage"] > 0}
    return {"passed": all(checks.values()), "checks": checks,
            "action": "run_closed_loop" if all(checks.values()) else "no_go_stop_before_closed_loop"}


def run_policy(name, probabilities, reference, vocab, blank_id, kind, model, stats, threshold, sequences):
    result = OLD.run_predictor_policy(kind, model, threshold, stats, sequences, name,
                                      probabilities, reference, vocab, blank_id)
    result["dense_windows"] = len(probabilities)
    return result


def closed_loop(policies, config):
    sequences = OLD.SequenceFeatures(SEQUENCES)
    chosen = set(config["subsets"]["calibration"]["sample_ids"])
    vocab = json.loads(CHRON.DEFAULT_VOCAB.read_text()); blank = vocab.index("<blank>")
    dense = CHRON.DEFAULT_DENSE_ROOT; indices, shard_count = CHRON.completed_shard_indices(dense)
    outputs = {key: [] for key in ["online_uniform", *policies]}
    for index in indices:
        results, logits, _ = BUILDER.BUILDER.validate_dense_shard(dense, index, shard_count, verify_hashes=False)
        for name in results:
            if name not in chosen: continue
            p = BUILDER.ORACLE.softmax_rows(np.asarray(logits[name]))
            ref = BUILDER.REPLAY.clean_phoenix_2014_trans(results[name]["gls_ref"])
            outputs["online_uniform"].append(AUDIT.uniform_result(p, ref, vocab, blank))
            for key, (kind, model, stats, threshold) in policies.items():
                outputs[key].append(run_policy(name, p, ref, vocab, blank, kind, model, stats, threshold, sequences))
    uniform = outputs["online_uniform"]
    summaries = {"online_uniform": AUDIT.aggregate_strategy(uniform, uniform)}
    for key in policies:
        summaries[key] = OLD.summarize_policy(outputs[key], uniform, key, policies[key][3])
    return summaries


def analyze(data_root, output_root):
    if output_root.exists(): raise FileExistsError(output_root)
    config = load_config(data_root)
    temporary = output_root.with_name("." + output_root.name + f".incomplete-{os.getpid()}")
    temporary.mkdir(parents=True)
    (temporary / "resolved_config_preregistered.json").write_text(json.dumps(config, indent=2) + "\n")
    random.seed(TRAIN_SEED); np.random.seed(TRAIN_SEED); torch.manual_seed(TRAIN_SEED)
    torch.set_num_threads(min(8, os.cpu_count() or 1)); torch.use_deterministic_algorithms(True)
    data, sources = load_data(data_root); fit, cal = data["fit"], data["calibration"]
    stats = OLD.normalization(fit); fitn, caln = OLD.normalize(fit, stats), OLD.normalize(cal, stats)
    models, offline, training = {}, {}, {}
    for label, kind in (("T0", KIND0), ("T1", KIND1)):
        model, history, pos_weight = OLD.train_variant(kind, fitn, epochs=5, batch_size=1024, seed=TRAIN_SEED)
        fit_score, _ = OLD.predict(model, fitn)
        score, positive = OLD.predict(model, caln)
        offline[label] = OLD.evaluation_metrics(caln["positive"], caln["advantage"], score, positive)
        sensitivity = float(np.quantile(fit_score, .90))
        training[label] = {"variant": kind, "history": history, "positive_weight": pos_weight,
                           "primary_threshold": 0.0, "fit_90pct_sensitivity_threshold": sensitivity}
        models[label] = (kind, model, stats, 0.0)
        torch.save({"variant": kind, "state_dict": model.state_dict(),
                    "normalization_fit_only": stats, "training": training[label]}, temporary / f"{label}.pt")
    decision = gate(offline)
    closed = None
    if decision["passed"]:
        policies = {"T0_primary": models["T0"], "T1_primary": models["T1"],
            "T1_fit90_sensitivity": (models["T1"][0], models["T1"][1], models["T1"][2],
                                      training["T1"]["fit_90pct_sensitivity_threshold"])}
        closed = closed_loop(policies, config)
        audit_metrics = json.loads((AUDIT_ROOT / "metrics.json").read_text())
        # These exact frozen-128 references were already computed under the identical protocol.
        closed["B2_old"] = audit_metrics["policies"]["B2_old"]["base_subset"]
        closed["terminal_oracle"] = audit_metrics["policies"]["B2_old"]["terminal_policy_improvement_oracle_subset"]
        uwer = closed["online_uniform"]["metrics"]["wer"]
        ower = closed["terminal_oracle"]["metrics"]["wer"]
        available = uwer - ower
        for key in ("T0_primary", "T1_primary", "T1_fit90_sensitivity"):
            closed[key]["terminal_oracle_gap_recovery_fraction"] = ((uwer - closed[key]["metrics"]["wer"]) / available
                                                                    if available > 0 else None)
        t1 = closed["T1_primary"]
        closed["decision"] = {
            "directional_success": t1["metrics"]["wer"] < closed["B2_old"]["metrics"]["wer"] and t1["metrics"]["wer"] < uwer,
            "strong_go": t1["delta_wer_pp_vs_online_uniform"] <= -0.5 and
                t1["paired_bootstrap_vs_online_uniform"]["ci95"][1] < 0}
    manifest = json.loads((data_root / "dataset_manifest.json").read_text())
    result = {"scope": "preregistered partial32 terminal predictor smoke", "gpu_used": False,
        "dev_used": False, "test_used": False, "fit_samples": FIT_SIZE,
        "calibration_samples": CAL_SIZE, "dataset": manifest, "offline_predictability": offline,
        "gate": decision, "training": training, "closed_loop": closed,
        "limitations": ["32/56 non-random train shards", "train calibration repeatedly inspected",
            "fixed 128 calibration subset is unusually hard and does not represent full 526",
            "terminal labels use future dense logits, reference, and true endpoint only as offline supervision",
            "no formal dev/test, wall-time, or submission-latency conclusion"]}
    (temporary / "metrics.json").write_text(json.dumps(result, indent=2) + "\n")
    out_manifest = {"status": "complete", "created_utc": datetime.now(timezone.utc).isoformat(),
        "config_sha256": sha256_file(temporary / "resolved_config_preregistered.json"),
        "metrics_sha256": sha256_file(temporary / "metrics.json"),
        "model_sha256": {x: sha256_file(temporary / f"{x}.pt") for x in ("T0", "T1")}}
    (temporary / "manifest.json").write_text(json.dumps(out_manifest, indent=2) + "\n")
    temporary.replace(output_root)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("preregister", "worker", "finalize-dataset", "analyze"))
    parser.add_argument("--data-root", type=Path, default=DATA_ROOT)
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    parser.add_argument("--worker-index", type=int, default=0); parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args(); os.environ["CUDA_VISIBLE_DEVICES"] = ""
    if args.mode == "preregister": value = preregister(args.data_root.resolve())
    elif args.mode == "worker": value = worker(args.data_root.resolve(), args.worker_index, args.workers)
    elif args.mode == "finalize-dataset": value = finalize_dataset(args.data_root.resolve(), args.workers)
    else: value = analyze(args.data_root.resolve(), args.output_root.resolve())
    print(json.dumps(value, indent=2))


if __name__ == "__main__": main()
