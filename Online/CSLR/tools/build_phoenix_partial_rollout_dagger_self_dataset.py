#!/usr/bin/env python3
"""Build one frozen DA1 self-trajectory dataset for B2 and B3 (CPU only).

The old predictors choose actions on their own chronological fit trajectories.
Reference transcripts and future dense logits are consulted only after reaching
an autonomous state, to attach the same fixed rollout-16 signed-advantage label.
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
OLD_RESULT = ROOT / "results/phoenix-2014t_ISLR/p3_partial32_rollout_predictor_closed_loop_smoke_v2_49faacc3"
OUTPUT_ROOT = ROOT / "results/phoenix-2014t_ISLR/p3_partial32_rollout_dagger_da1_smoke_self_v1_49faacc3"
SEQUENCES = Path("/tmp/phoenix_p2_train_center_sequences_v1.npz")
KINDS = ("B2_bookkeeping_prefix", "B3_bookkeeping_visual_tcn_prefix")


def import_tool(name):
    path = ROOT / f"tools/{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


ANALYZE = import_tool("analyze_phoenix_partial_rollout_predictor")
BUILDER = ANALYZE.DATA_BUILDER
CHRON, ORACLE, REPLAY = BUILDER.CHRON, BUILDER.ORACLE, BUILDER.REPLAY


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_old(kind, old_root=OLD_RESULT):
    checkpoint = torch.load(old_root / f"{kind}.pt", map_location="cpu")
    use_prefix, use_visual = ANALYZE.VARIANTS[kind]
    stats = checkpoint["normalization_fit_only"]
    model = ANALYZE.RolloutPredictor(
        len(stats["bookkeeping"][0]), len(stats["prefix"][0]),
        len(stats["visual"][0]), use_prefix, use_visual,
    )
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    metrics = json.loads((old_root / "metrics.json").read_text())
    return model, stats, float(metrics["fit_only_thresholds"][kind])


def self_records(name, result, logits, vocab, blank_id, kind, model, stats, threshold, sequences):
    probabilities = ORACLE.softmax_rows(np.asarray(logits))
    reference = REPLAY.clean_phoenix_2014_trans(result["gls_ref"])
    starts = [int(x["start"]) for x in result["adaptive_stride_metadata"]]
    if starts != list(range(len(probabilities))):
        raise ValueError(f"non-dense coordinates: {name}")
    selected, balance, rows = [], 0.0, []
    counts = Counter()
    for start in range(len(probabilities)):
        if CHRON.is_skeleton(start):
            selected.append(start); counts["skeleton"] += 1
            continue
        balance = CHRON.accrue_token(balance)
        if not CHRON.token_eligible(balance):
            counts["ineligible"] += 1
            continue
        forced = CHRON.token_forced(balance)
        row = ANALYZE.online_feature_row(probabilities, selected, start, balance, blank_id)
        score = None
        if not forced:
            score = ANALYZE.online_score(kind, model, stats, sequences, name, row, start)
        execute = bool(forced or score > threshold)
        if not forced:
            advantage, skip_error, execute_error = CHRON.rollout_advantage(
                probabilities, selected, balance, start, reference, vocab, blank_id,
                horizon=CHRON.ROLLOUT_HORIZON,
            )
            rows.append({
                "sample_id": name,
                "source_video_id": BUILDER.source_video(name),
                "partition": "fit",
                "trajectory_source": kind,
                "candidate_window_start": start,
                "visible_original_frame_upper": min(len(probabilities) - 1, start + CHRON.LOOKAHEAD_FRAMES),
                "bookkeeping": row["bookkeeping"],
                "prefix": row["prefix"],
                "label": {"rollout16_signed_advantage": int(advantage),
                          "rollout16_positive": bool(advantage > 0),
                          "skip_error": int(skip_error), "execute_error": int(execute_error)},
                "old_predictor": {"score": float(score), "fit_only_threshold": float(threshold),
                                  "action_execute": execute},
                "provenance": {
                    "prefix_uses_only_this_predictor_own_past_selected": True,
                    "teacher_prefix_reuse": False,
                    "future_dense_logits_and_reference_used_only_for_label": True,
                    "current_candidate_expensive_logits_absent_from_inputs": True,
                    "total_length_and_eos_absent_from_inputs": True,
                },
            })
            counts["autonomous"] += 1
            counts["positive"] += int(advantage > 0)
            counts["execute_autonomous"] += int(execute)
        else:
            counts["forced"] += 1
        if execute:
            selected.append(start); balance -= 1.0
    counts["executed"] += len(selected)
    counts["dense"] += len(probabilities)
    return rows, counts


def build_shard(dense_root, output_root, shard_index, shard_count, assignments, vocab, blank_id,
                models, sequences):
    result, logits, _ = BUILDER.BUILDER.validate_dense_shard(
        dense_root, shard_index, shard_count, verify_hashes=False
    )
    summaries = {}
    for kind in KINDS:
        final = output_root / kind / f"self-{shard_index:05d}-of-{shard_count:05d}.jsonl.gz"
        marker = final.with_suffix(".complete.json")
        final.parent.mkdir(parents=True, exist_ok=True)
        if final.is_file() and marker.is_file():
            value = json.loads(marker.read_text())
            if value["sha256"] != sha256_file(final):
                raise ValueError(f"hash mismatch: {final}")
            summaries[kind] = {**value, "resumed": True}; continue
        temporary = final.with_name(f".{final.name}.incomplete-{os.getpid()}")
        total = Counter(); sample_count = 0; started = time.perf_counter()
        with gzip.open(temporary, "wt", encoding="utf-8", compresslevel=5) as handle:
            for name in result:
                if assignments[name] != "fit" or name not in sequences.slices:
                    continue
                rows, counts = self_records(name, result[name], logits[name], vocab, blank_id,
                                            kind, *models[kind], sequences)
                for row in rows:
                    handle.write(json.dumps(row, separators=(",", ":")) + "\n")
                total.update(counts); sample_count += 1
        temporary.replace(final)
        value = {"kind": kind, "shard": shard_index, "samples": sample_count,
                 "counts": dict(total), "sha256": sha256_file(final), "bytes": final.stat().st_size,
                 "elapsed_seconds": time.perf_counter() - started,
                 "created_utc": datetime.now(timezone.utc).isoformat()}
        marker.write_text(json.dumps(value, indent=2) + "\n")
        summaries[kind] = {**value, "resumed": False}
    return summaries


def run(dense_root, output_root, old_root, sequence_path, worker_id=None, workers=1,
        finalize_only=False):
    for path in (dense_root, output_root, old_root, sequence_path): ANALYZE.reject_test_path(path)
    output_root.mkdir(parents=True, exist_ok=True)
    indices, shard_count = CHRON.completed_shard_indices(dense_root)
    if len(indices) != 32: raise ValueError(f"expected frozen partial32, got {len(indices)}")
    split = json.loads((dense_root / "fit_calibration_split.json").read_text())
    assignments = split["assignments"]
    if {BUILDER.source_video(k) for k,v in assignments.items() if v == "fit"} & \
       {BUILDER.source_video(k) for k,v in assignments.items() if v == "calibration"}:
        raise ValueError("fit/calibration source overlap")
    vocab = json.loads(CHRON.DEFAULT_VOCAB.read_text()); blank_id = vocab.index("<blank>")
    config = {"status": "running", "dataset_aggregation_round": 1, "cpu_only": True,
              "trajectory_models": list(KINDS), "teacher_self_source_weight": [0.5, 0.5],
              "protocol": {"skeleton_period": 4, "bonus_accrual": 1/3, "initial": 0,
                           "capacity": 2, "eligible": 1, "forced": 2, "eos_topup": False,
                           "rollout_horizon": 16, "lookahead_frames": 8},
              "inputs_exclude": ["T", "EOS", "reference", "future_logits", "current_candidate_logits"]}
    config_path = output_root / "resolved_config_preregistered.json"
    if not config_path.exists(): config_path.write_text(json.dumps(config, indent=2) + "\n")
    if finalize_only:
        models = sequences = None; work = []
    else:
        torch.set_num_threads(1); torch.use_deterministic_algorithms(True)
        models = {kind: load_old(kind, old_root) for kind in KINDS}
        sequences = ANALYZE.SequenceFeatures(sequence_path)
        work = [x for p,x in enumerate(indices) if worker_id is None or p % workers == worker_id]
    for position, index in enumerate(work, 1):
        value = build_shard(dense_root, output_root, index, shard_count, assignments, vocab,
                            blank_id, models, sequences)
        print(json.dumps({"worker": worker_id, "completed": position, "total": len(work),
                          "shard": index, "summaries": value}), flush=True)
    if worker_id is not None: return
    manifest = {**config, "status": "complete", "completed_utc": datetime.now(timezone.utc).isoformat(),
                "kinds": {}}
    for kind in KINDS:
        values=[]; counts=Counter()
        for index in indices:
            final=output_root/kind/f"self-{index:05d}-of-{shard_count:05d}.jsonl.gz"
            marker=final.with_suffix(".complete.json")
            if not final.is_file() or not marker.is_file(): raise RuntimeError(f"missing {final}")
            value=json.loads(marker.read_text())
            if value["sha256"] != sha256_file(final): raise ValueError(f"hash mismatch {final}")
            values.append(value); counts.update(value["counts"])
        manifest["kinds"][kind]={"counts":dict(counts), "shards":values,
                                 "positive_prevalence":counts["positive"]/max(1,counts["autonomous"])}
    (output_root/"dataset_manifest.json").write_text(json.dumps(manifest,indent=2)+"\n")


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dense-root",type=Path,default=CHRON.DEFAULT_DENSE_ROOT)
    parser.add_argument("--output-root",type=Path,default=OUTPUT_ROOT)
    parser.add_argument("--old-root",type=Path,default=OLD_RESULT)
    parser.add_argument("--sequences",type=Path,default=SEQUENCES)
    parser.add_argument("--worker-id",type=int); parser.add_argument("--workers",type=int,default=1)
    parser.add_argument("--finalize-only",action="store_true")
    args=parser.parse_args()
    if args.worker_id is not None and not 0 <= args.worker_id < args.workers: raise ValueError("bad worker")
    run(args.dense_root.resolve(),args.output_root.resolve(),args.old_root.resolve(),args.sequences.resolve(),
        args.worker_id,args.workers,args.finalize_only)

if __name__ == "__main__": main()
