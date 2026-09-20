#!/usr/bin/env python3
"""Fit-only robust-continuation oracle for structured online scheduling.

A non-center bonus is chosen only when it strictly improves terminal edit error
under both frozen center and late-offset future continuations.  The oracle is
reference/future aware and is never a deployable policy.
"""
import gzip
import hashlib
import importlib.util
import json
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "results/phoenix-2014t_ISLR"
DATA = BASE / "p3_partial32_structured_block_smoke_v1_49faacc3_dataset"
OUTPUT = BASE / "p3_partial32_robust_continuation_oracle_fit512_v2_49faacc3"
SEED = 261025


def imp(name):
    path = ROOT / f"tools/{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


BLOCK = imp("analyze_phoenix_partial_structured_block")
CHRON, BUILDER = BLOCK.CHRON, BLOCK.BUILDER


def fit_ids():
    config = json.loads((DATA / "resolved_config_preregistered.json").read_text())
    return sorted(config["subsets"]["fit"]["sample_ids"])


def continuation(total, next_skeleton, bonus_offset):
    selected = []
    for start in range(next_skeleton, total, 4):
        selected.append(start)
        if start + 3 < total:
            selected.append(start + bonus_offset)
    return selected


def decode_error(probabilities, selected, reference, vocab, blank):
    hypothesis = BUILDER.ORACLE.decode_probabilities(probabilities, selected, vocab, blank, span=15.0)
    return int(BUILDER.ORACLE.error_count(reference, hypothesis))


def robust_choice(probabilities, selected, candidates, futures, reference, vocab, blank):
    errors = []
    for future in futures:
        errors.append([decode_error(probabilities, selected + [candidate] + future,
                                    reference, vocab, blank) for candidate in candidates])
    rewards = np.asarray([[row[1] - row[index] for index in range(3)] for row in errors], dtype=np.int64)
    robust = rewards.min(axis=0)
    eligible = [index for index in (0, 2) if robust[index] > 0]
    if not eligible:
        return 1, errors, robust.tolist()
    # Maximize worst-case benefit; deterministic tie-break left then right.
    chosen = max(eligible, key=lambda index: (int(robust[index]), -index))
    return chosen, errors, robust.tolist()


def source_bootstrap(per_sample, reps=10000):
    by_source = defaultdict(list)
    for row in per_sample:
        by_source[row["source_video_id"]].append(row)
    sources = sorted(by_source)
    source_deltas = np.asarray([
        sum(item["robust_error"] - item["uniform_error"] for item in by_source[source])
        for source in sources
    ], dtype=np.int64)
    source_refs = np.asarray([
        sum(item["ref_len"] for item in by_source[source]) for source in sources
    ], dtype=np.int64)
    rng = np.random.default_rng(SEED)
    values = []
    for _ in range(reps):
        index = rng.integers(0, len(sources), size=len(sources))
        values.append(100.0 * source_deltas[index].sum() / max(1, source_refs[index].sum()))
    return [float(np.quantile(values, 0.025)), float(np.quantile(values, 0.975))]


def preregister():
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    ids = fit_ids()
    sources = sorted({BUILDER.source_video(sample_id) for sample_id in ids})
    dense_protocol = json.loads((CHRON.DEFAULT_DENSE_ROOT / "protocol_manifest.json").read_text())
    checkpoint = next(value for path, value in dense_protocol["inputs"].items() if path.endswith("/ckpts/best.ckpt"))
    config = {
        "experiment": "fit512 robust-continuation structured oracle",
        "created_before_oracle_outcomes": True,
        "scope": {"samples": len(ids), "sources": len(sources),
                  "sample_ids_sha256": hashlib.sha256("\n".join(ids).encode()).hexdigest()},
        "checkpoint_sha256": checkpoint["sha256"],
        "protocol": {"hard_skeleton_offset": 0, "one_bonus_offsets": [1, 2, 3],
                     "fallback": "offset2 center", "target_rate": "approximately 50%",
                     "max_gap_including_endpoints": 4, "decoder_span": 15, "unknown_EOS": True},
        "oracle": {
            "future_continuations": ["fixed offset2 center", "fixed offset3 late"],
            "eligible_side": "strict terminal edit-error improvement over center under both continuations",
            "selection": "maximize minimum improvement; ties left then right; otherwise center",
            "future_reference_EOS": "oracle label only, never deployment input",
        },
        "go": "delta WER versus fixed-center <= -0.5 pp and source-cluster bootstrap upper < 0",
        "forbidden": ["GPU", "dev", "test", "calibration", "evaluation", "predictor training", "git commit"],
        "classification": "exploratory fit-only target-headroom audit",
    }
    OUTPUT.mkdir(parents=True)
    (OUTPUT / "resolved_config_preregistered.json").write_text(json.dumps(config, indent=2) + "\n")
    return config


def run():
    config = preregister()
    wanted = set(fit_ids())
    vocab = json.loads(CHRON.DEFAULT_VOCAB.read_text())
    blank = vocab.index("<blank>")
    per_sample = []
    aggregate = Counter()
    seen = set()
    started = time.perf_counter()
    indices, shard_count = CHRON.completed_shard_indices(CHRON.DEFAULT_DENSE_ROOT)
    for shard in indices:
        results, logits, _ = BUILDER.BUILDER.validate_dense_shard(
            CHRON.DEFAULT_DENSE_ROOT, shard, shard_count, verify_hashes=False
        )
        for name in results:
            if name not in wanted:
                continue
            p = BUILDER.ORACLE.softmax_rows(np.asarray(logits[name]))
            reference = BUILDER.REPLAY.clean_phoenix_2014_trans(results[name]["gls_ref"])
            uniform = BLOCK.block_uniform(len(p))
            selected = []
            local = Counter()
            for start in range(0, len(p), 4):
                selected.append(start)
                if start + 3 >= len(p):
                    continue
                candidates = [start + 1, start + 2, start + 3]
                futures = [continuation(len(p), start + 4, 2), continuation(len(p), start + 4, 3)]
                chosen, _, robust = robust_choice(p, selected, candidates, futures, reference, vocab, blank)
                selected.append(candidates[chosen])
                local["blocks"] += 1
                local["side_choices"] += chosen != 1
                local["left_choices"] += chosen == 0
                local["right_choices"] += chosen == 2
                local["worst_case_reward"] += int(robust[chosen])
            uniform_hypothesis, uniform_counts = CHRON.decode_counts(p, uniform, reference, vocab, blank)
            robust_hypothesis, robust_counts = CHRON.decode_counts(p, selected, reference, vocab, blank)
            del uniform_hypothesis, robust_hypothesis
            coverage = CHRON.coverage_metrics(selected, len(p))
            if coverage["max_gap_including_endpoints"] > 4:
                raise RuntimeError("coverage invariant violated")
            row = {"sample_id": name, "source_video_id": BUILDER.source_video(name),
                   "uniform_error": int(uniform_counts["error"]), "robust_error": int(robust_counts["error"]),
                   "ref_len": int(uniform_counts["ref_len"]), "selected_windows": len(selected),
                   "dense_windows": len(p), **dict(local)}
            per_sample.append(row)
            aggregate.update({"uniform_error": row["uniform_error"], "robust_error": row["robust_error"],
                              "ref_len": row["ref_len"], "selected_windows": row["selected_windows"],
                              "dense_windows": row["dense_windows"], **dict(local)})
            seen.add(name)
            if len(seen) % 25 == 0 or len(seen) == len(wanted):
                print(json.dumps({"samples_done": len(seen), "samples_expected": len(wanted),
                                  "elapsed_seconds": round(time.perf_counter() - started, 1)}), flush=True)
    if seen != wanted:
        raise RuntimeError("fit sample coverage mismatch")
    uniform_wer = 100.0 * aggregate["uniform_error"] / aggregate["ref_len"]
    robust_wer = 100.0 * aggregate["robust_error"] / aggregate["ref_len"]
    ci = source_bootstrap(per_sample)
    value = {
        "scope": config["scope"], "checkpoint_sha256": config["checkpoint_sha256"],
        "uniform": {"errors": aggregate["uniform_error"], "wer": uniform_wer},
        "robust_oracle": {"errors": aggregate["robust_error"], "wer": robust_wer,
                          "delta_errors": aggregate["robust_error"] - aggregate["uniform_error"],
                          "delta_wer_pp": robust_wer - uniform_wer,
                          "source_cluster_bootstrap_95pct_delta_wer_pp": ci},
        "schedule": {"blocks": aggregate["blocks"], "side_choices": aggregate["side_choices"],
                     "left_choices": aggregate["left_choices"], "right_choices": aggregate["right_choices"],
                     "side_choice_rate": aggregate["side_choices"] / max(1, aggregate["blocks"]),
                     "selected_windows": aggregate["selected_windows"],
                     "dense_windows": aggregate["dense_windows"],
                     "window_rate": aggregate["selected_windows"] / aggregate["dense_windows"],
                     "summed_worst_case_local_reward": aggregate["worst_case_reward"]},
        "gate": {"passed": robust_wer - uniform_wer <= -0.5 and ci[1] < 0,
                 "checks": {"delta_wer_le_minus_0_5pp": robust_wer - uniform_wer <= -0.5,
                            "bootstrap_upper_lt_0": ci[1] < 0}},
        "limitations": ["fit-only exploratory audit", "reference/future-aware non-deployable oracle",
                        "partial32-derived dense replay", "robust labels are likely substantially sparser"],
        "elapsed_seconds": time.perf_counter() - started,
    }
    (OUTPUT / "metrics.json").write_text(json.dumps(value, indent=2) + "\n")
    with gzip.open(OUTPUT / "per_sample.jsonl.gz", "wt", encoding="utf-8") as handle:
        for row in per_sample:
            handle.write(json.dumps(row, separators=(",", ":")) + "\n")
    return value


if __name__ == "__main__":
    print(json.dumps(run(), indent=2))
