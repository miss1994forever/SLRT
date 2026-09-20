#!/usr/bin/env python3
"""Build fit-only block labels for the robust-continuation oracle.

References and future dense replay are teacher-only.  Serialized predictor
inputs contain bookkeeping and decoder-prefix summaries from paid past
windows; cheap causal visual histories remain in the separately audited P2
sequence archive.
"""
import gzip
import hashlib
import importlib.util
import json
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "results/phoenix-2014t_ISLR"
OUTPUT = BASE / "p3_partial32_robust_continuation_fit512_dataset_v1_49faacc3"
ORACLE_RESULT = BASE / "p3_partial32_robust_continuation_oracle_fit512_v2_49faacc3"


def imp(name):
    path = ROOT / f"tools/{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


ROBUST = imp("analyze_phoenix_robust_continuation_oracle")
BLOCK, CHRON, BUILDER = ROBUST.BLOCK, ROBUST.CHRON, ROBUST.BUILDER


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def preregister():
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    oracle_metrics = json.loads((ORACLE_RESULT / "metrics.json").read_text())
    ids = ROBUST.fit_ids()
    config = {
        "experiment": "fit512 robust-continuation counterfactual block-label dataset",
        "created_before_dataset_outcomes": True,
        "scope": oracle_metrics["scope"],
        "sample_ids_sha256": hashlib.sha256("\n".join(ids).encode()).hexdigest(),
        "checkpoint_sha256": oracle_metrics["checkpoint_sha256"],
        "teacher": {
            "past": "closed-loop robust-oracle selected past",
            "future_continuations": ["fixed offset2 center", "fixed offset3 late"],
            "side_label": "strictly positive minimum terminal edit-error improvement versus center",
            "selection": "maximum minimum improvement, tie left then right, otherwise center",
            "uses_reference_future_and_EOS": True,
        },
        "predictor_inputs": {
            "stored": ["bookkeeping", "past-paid-window decoder-prefix summary"],
            "external_audited_archive": "/tmp/phoenix_p2_train_center_sequences_v1.npz",
            "candidate_expensive_logits": False,
            "reference_future_total_length_or_EOS": False,
            "decision_arrival": "offset3, so all three candidate positions have arrived",
        },
        "intended_use": "source-disjoint fit-only cross-validation; no dev/test model selection",
        "oracle_metrics_sha256": sha256_file(ORACLE_RESULT / "metrics.json"),
        "forbidden": ["GPU", "dev", "test", "calibration tuning", "git commit"],
    }
    OUTPUT.mkdir(parents=True)
    (OUTPUT / "resolved_config_preregistered.json").write_text(json.dumps(config, indent=2) + "\n")
    return config


def run():
    config = preregister()
    wanted = set(ROBUST.fit_ids())
    vocab = json.loads(CHRON.DEFAULT_VOCAB.read_text())
    blank = vocab.index("<blank>")
    final = OUTPUT / "robust-block-labels.jsonl.gz"
    temp = OUTPUT / ".robust-block-labels.jsonl.gz.incomplete"
    counts = Counter()
    seen = set()
    started = time.perf_counter()
    indices, shard_count = CHRON.completed_shard_indices(CHRON.DEFAULT_DENSE_ROOT)
    with gzip.open(temp, "wt", encoding="utf-8", compresslevel=5) as handle:
        for shard in indices:
            results, logits, _ = BUILDER.BUILDER.validate_dense_shard(
                CHRON.DEFAULT_DENSE_ROOT, shard, shard_count, verify_hashes=False
            )
            for name in results:
                if name not in wanted:
                    continue
                probabilities = BUILDER.ORACLE.softmax_rows(np.asarray(logits[name]))
                reference = BUILDER.REPLAY.clean_phoenix_2014_trans(results[name]["gls_ref"])
                selected = []
                sample_counts = Counter()
                for start in range(0, len(probabilities), 4):
                    selected.append(start)
                    if start + 3 >= len(probabilities):
                        continue
                    candidates = [start + 1, start + 2, start + 3]
                    futures = [
                        ROBUST.continuation(len(probabilities), start + 4, 2),
                        ROBUST.continuation(len(probabilities), start + 4, 3),
                    ]
                    choice, errors, robust = ROBUST.robust_choice(
                        probabilities, selected, candidates, futures, reference, vocab, blank
                    )
                    row = {
                        "type": "block",
                        "sample_id": name,
                        "source_video_id": BUILDER.source_video(name),
                        "block_start": start,
                        "decision_arrival": start + 3,
                        "candidate_starts": candidates,
                        "predictor_inputs": BLOCK.candidate_features(
                            probabilities, selected, candidates, blank
                        ),
                        "label": {
                            "terminal_errors_center_future": errors[0],
                            "terminal_errors_late_future": errors[1],
                            "robust_reward_by_candidate": robust,
                            "strictly_beneficial_side_mask": [robust[0] > 0, False, robust[2] > 0],
                            "teacher_choice": choice,
                        },
                        "provenance": {
                            "predictor_prefix_past_selected_only": True,
                            "candidate_expensive_logits_absent": True,
                            "reference_future_EOS_label_only": True,
                            "unknown_EOS_deployment_compatible_inputs": True,
                        },
                    }
                    handle.write(json.dumps(row, separators=(",", ":")) + "\n")
                    selected.append(candidates[choice])
                    sample_counts["blocks"] += 1
                    sample_counts["side_choices"] += choice != 1
                uniform_counts = BLOCK.decode_counts(
                    probabilities, BLOCK.block_uniform(len(probabilities)), reference, vocab, blank
                )
                robust_counts = BLOCK.decode_counts(
                    probabilities, selected, reference, vocab, blank
                )
                sample = {
                    "type": "sample",
                    "sample_id": name,
                    "source_video_id": BUILDER.source_video(name),
                    "dense_windows": len(probabilities),
                    "selected_windows": len(selected),
                    "ref_len": int(uniform_counts["ref_len"]),
                    "uniform_error": int(uniform_counts["error"]),
                    "robust_error": int(robust_counts["error"]),
                    **dict(sample_counts),
                }
                handle.write(json.dumps(sample, separators=(",", ":")) + "\n")
                counts.update(sample_counts)
                counts.update({
                    "samples": 1,
                    "uniform_error": sample["uniform_error"],
                    "robust_error": sample["robust_error"],
                    "ref_len": sample["ref_len"],
                })
                seen.add(name)
                if len(seen) % 25 == 0 or len(seen) == len(wanted):
                    print(json.dumps({"samples_done": len(seen), "samples_expected": len(wanted),
                                      "elapsed_seconds": round(time.perf_counter() - started, 1)}), flush=True)
    if seen != wanted:
        raise RuntimeError("fit sample coverage mismatch")
    oracle_metrics = json.loads((ORACLE_RESULT / "metrics.json").read_text())
    observed = {
        "uniform_error": counts["uniform_error"],
        "robust_error": counts["robust_error"],
        "blocks": counts["blocks"],
        "side_choices": counts["side_choices"],
    }
    expected = {
        "uniform_error": oracle_metrics["uniform"]["errors"],
        "robust_error": oracle_metrics["robust_oracle"]["errors"],
        "blocks": oracle_metrics["schedule"]["blocks"],
        "side_choices": oracle_metrics["schedule"]["side_choices"],
    }
    if observed != expected:
        raise RuntimeError(f"oracle reproduction mismatch: {observed} != {expected}")
    temp.replace(final)
    manifest = {
        "status": "complete",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "counts": dict(counts),
        "oracle_reproduction": observed,
        "config_sha256": sha256_file(OUTPUT / "resolved_config_preregistered.json"),
        "data_sha256": sha256_file(final),
        "bytes": final.stat().st_size,
        "elapsed_seconds": time.perf_counter() - started,
    }
    (OUTPUT / "dataset_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


if __name__ == "__main__":
    print(json.dumps(run(), indent=2))
