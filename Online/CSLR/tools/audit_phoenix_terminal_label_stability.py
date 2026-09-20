#!/usr/bin/env python3
"""Fit-only sensitivity audit for structured-block terminal labels.

This is diagnostic, not a new scheduler or a model-selection experiment.  The
same already-selected past is evaluated under two frozen alternative targets:
late-offset future continuation and a narrower decoder span.
"""
import argparse
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
OUTPUT = BASE / "p3_partial32_terminal_label_stability_fit96_v1_49faacc3"
FULL_OUTPUT = BASE / "p3_partial32_terminal_label_stability_fullfit_v2_49faacc3"
SEED = "terminal-label-stability-v1"
N_SOURCES = 96


def imp(name):
    path = ROOT / f"tools/{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


BLOCK = imp("analyze_phoenix_partial_structured_block")
CHRON, BUILDER = BLOCK.CHRON, BLOCK.BUILDER


def selected_ids(full_fit=False):
    config = json.loads((DATA / "resolved_config_preregistered.json").read_text())
    groups = defaultdict(list)
    for sample_id in config["subsets"]["fit"]["sample_ids"]:
        groups[BUILDER.source_video(sample_id)].append(sample_id)
    if full_fit:
        return sorted(sample for samples in groups.values() for sample in samples)
    ranked = sorted(groups, key=lambda source: hashlib.sha256(f"{SEED}\0{source}".encode()).hexdigest())
    return sorted(min(groups[source], key=lambda sample: hashlib.sha256(f"{SEED}\0{sample}".encode()).hexdigest())
                  for source in ranked[:N_SOURCES])


def late_continuation(total, next_skeleton):
    selected = []
    for start in range(next_skeleton, total, 4):
        selected.append(start)
        if start + 3 < total:
            selected.append(start + 3)
    return selected


def center_continuation(total, next_skeleton):
    return BLOCK.center_continuation(total, next_skeleton)


def stable_best(errors):
    """Prefer fixed center on ties, then left, then right."""
    minimum = min(errors)
    return next(index for index in (1, 0, 2) if errors[index] == minimum)


def signs(errors):
    center = errors[1]
    return tuple(int(np.sign(center - errors[index])) for index in (0, 2))


def terminal_errors(probabilities, reference, vocab, blank, selected_past, candidates, future, span):
    output = []
    for candidate in candidates:
        hypothesis = BUILDER.ORACLE.decode_probabilities(
            probabilities, selected_past + [candidate] + future, vocab, blank, span=span
        )
        output.append(int(BUILDER.ORACLE.error_count(reference, hypothesis)))
    return output


def preregister(full_fit=False):
    output = FULL_OUTPUT if full_fit else OUTPUT
    if output.exists():
        raise FileExistsError(output)
    ids = selected_ids(full_fit)
    sources = {BUILDER.source_video(sample_id) for sample_id in ids}
    dense_protocol = json.loads((CHRON.DEFAULT_DENSE_ROOT / "protocol_manifest.json").read_text())
    checkpoint = next(value for path, value in dense_protocol["inputs"].items() if path.endswith("/ckpts/best.ckpt"))
    config = {
        "purpose": "diagnose stability of terminal edit-error labels, not select a scheduler",
        "scope": (f"fit-only, all {len(ids)} samples from {len(sources)} source videos" if full_fit
                  else "fit-only, one deterministic sample from each of 96 source videos"),
        "sample_ids": ids,
        "sample_ids_sha256": hashlib.sha256("\n".join(ids).encode()).hexdigest(),
        "source_video_count": len(sources),
        "checkpoint_sha256": checkpoint["sha256"],
        "baseline": "span15, fixed-center future continuation, stored oracle past",
        "alternative_future": "span15, fixed-offset3 future continuation, same past",
        "alternative_decoder": "span7, fixed-center future continuation, same past",
        "statistics": ["informative blocks", "strictly beneficial side actions", "benefit sign agreement",
                       "best-action agreement with center tie-break", "positive-side survival",
                       "source-cluster bootstrap intervals for survival and informative action agreement"],
        "interpretation": "descriptive sensitivity only; alternate future or decoder is not a deployment policy",
        "forbidden": ["GPU", "dev", "test", "calibration", "evaluation", "predictor training", "checkpoint loading"],
    }
    output.mkdir(parents=True)
    (output / "resolved_config_preregistered.json").write_text(json.dumps(config, indent=2) + "\n")
    return config


def source_bootstrap(per_source, label, numerator, denominator, reps=2000):
    sources = sorted(per_source)
    rows = np.asarray([[per_source[source][f"{label}_{numerator}"],
                        per_source[source][f"{label}_{denominator}"]] for source in sources], dtype=np.int64)
    rng = np.random.default_rng(261024)
    estimates = []
    for _ in range(reps):
        sampled = rows[rng.integers(0, len(rows), size=len(rows))].sum(axis=0)
        if sampled[1]:
            estimates.append(sampled[0] / sampled[1])
    return [float(np.quantile(estimates, 0.025)), float(np.quantile(estimates, 0.975))]


def run(full_fit=False):
    config = preregister(full_fit)
    output = FULL_OUTPUT if full_fit else OUTPUT
    wanted = set(config["sample_ids"])
    by_sample = defaultdict(list)
    sample_info = {}
    manifest = json.loads((DATA / "dataset_manifest.json").read_text())
    for worker in manifest["workers"]:
        path = DATA / "workers" / f"worker-{worker['worker']:02d}-of-{worker['workers']:02d}.jsonl.gz"
        with gzip.open(path, "rt", encoding="utf-8") as handle:
            for line in handle:
                row = json.loads(line)
                if row["partition"] != "fit" or row["sample_id"] not in wanted:
                    continue
                if row["type"] == "block":
                    by_sample[row["sample_id"]].append(row)
                else:
                    sample_info[row["sample_id"]] = row
    if set(by_sample) != wanted or set(sample_info) != wanted:
        raise RuntimeError("selected fit samples missing from structured-block dataset")

    vocab = json.loads(CHRON.DEFAULT_VOCAB.read_text())
    blank = vocab.index("<blank>")
    counts = Counter()
    per_source = defaultdict(Counter)
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
            probabilities = BUILDER.ORACLE.softmax_rows(np.asarray(logits[name]))
            reference = BUILDER.REPLAY.clean_phoenix_2014_trans(results[name]["gls_ref"])
            chosen = sample_info[name]["oracle"]["selected"]
            local = Counter()
            for row in by_sample[name]:
                start = row["block_start"]
                past = [position for position in chosen if position <= start]
                candidates = row["candidate_starts"]
                baseline = list(map(int, row["label"]["terminal_errors"]))
                future_center = center_continuation(len(probabilities), start + 4)
                future_late = late_continuation(len(probabilities), start + 4)
                check = terminal_errors(probabilities, reference, vocab, blank, past, candidates, future_center, 15.0)
                if check != baseline:
                    raise RuntimeError(f"baseline replay mismatch for {name} block {start}: {check} != {baseline}")
                alternatives = {
                    "late_future": terminal_errors(probabilities, reference, vocab, blank, past, candidates, future_late, 15.0),
                    "span7": terminal_errors(probabilities, reference, vocab, blank, past, candidates, future_center, 7.0),
                }
                base_signs = signs(baseline)
                base_best = stable_best(baseline)
                local["blocks"] += 1
                local["baseline_informative"] += len(set(baseline)) > 1
                local["baseline_positive_sides"] += sum(value > 0 for value in base_signs)
                for label, errors in alternatives.items():
                    other_signs = signs(errors)
                    local[f"{label}_informative"] += len(set(errors)) > 1
                    local[f"{label}_positive_sides"] += sum(value > 0 for value in other_signs)
                    local[f"{label}_best_agree_all"] += stable_best(errors) == base_best
                    if len(set(baseline)) > 1 or len(set(errors)) > 1:
                        local[f"{label}_best_agree_union_informative"] += stable_best(errors) == base_best
                        local[f"{label}_union_informative"] += 1
                    for before, after in zip(base_signs, other_signs):
                        if before != 0 or after != 0:
                            local[f"{label}_union_nonzero_sides"] += 1
                            local[f"{label}_sign_agree_union_nonzero"] += before == after
                        if before > 0:
                            local[f"{label}_base_positive_sides"] += 1
                            local[f"{label}_base_positive_survives"] += after > 0
                            local[f"{label}_base_positive_reverses"] += after < 0
            per_source[BUILDER.source_video(name)].update(local)
            counts.update(local)
            seen.add(name)
            if len(seen) % 25 == 0 or len(seen) == len(wanted):
                print(json.dumps({"samples_done": len(seen), "samples_expected": len(wanted),
                                  "elapsed_seconds": round(time.perf_counter()-started, 1)}), flush=True)
    if seen != wanted or len(per_source) != config["source_video_count"]:
        raise RuntimeError("dense replay did not cover all selected fit samples and sources")
    summary = {"scope": config["scope"], "checkpoint_sha256": config["checkpoint_sha256"],
               "samples": len(seen), "source_videos": len(per_source), "counts": dict(counts),
               "rates": {
                   label: {
                       "best_agreement_all": counts[f"{label}_best_agree_all"] / counts["blocks"],
                       "best_agreement_union_informative": counts[f"{label}_best_agree_union_informative"] / max(1, counts[f"{label}_union_informative"]),
                       "sign_agreement_union_nonzero": counts[f"{label}_sign_agree_union_nonzero"] / max(1, counts[f"{label}_union_nonzero_sides"]),
                       "baseline_positive_survival": counts[f"{label}_base_positive_survives"] / max(1, counts[f"{label}_base_positive_sides"]),
                   } for label in ("late_future", "span7")
               },
               "source_cluster_bootstrap_95pct": {
                   label: {
                       "best_agreement_union_informative": source_bootstrap(
                           per_source, label, "best_agree_union_informative", "union_informative"),
                       "baseline_positive_survival": source_bootstrap(
                           per_source, label, "base_positive_survives", "base_positive_sides"),
                   } for label in ("late_future", "span7")
               },
               "elapsed_seconds": time.perf_counter()-started,
               "limitations": ["partial32 fit-only selected sources", "labels are reference-aware and non-deployable",
                               "changing the decoder changes the target task itself"]}
    (output / "metrics.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--full-fit", action="store_true")
    args = parser.parse_args()
    print(json.dumps(run(full_fit=args.full_fit), indent=2))
