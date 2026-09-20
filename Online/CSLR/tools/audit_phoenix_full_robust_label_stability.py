#!/usr/bin/env python3
"""Complete-fit continuation stability audit from frozen robust labels."""
import gzip
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "results/phoenix-2014t_ISLR"
DATA = BASE / "p3_fulltrain_robust_continuation_fit6378_dataset_v2_49faacc3"
OUTPUT = BASE / "p3_fulltrain_robust_label_stability_fit6378_v1_49faacc3"
SEED = 261033


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def stable_best(errors):
    minimum = min(errors)
    return next(index for index in (1, 0, 2) if errors[index] == minimum)


def side_signs(errors):
    return tuple(int(np.sign(errors[1] - errors[index])) for index in (0, 2))


def frozen_config():
    manifest = json.loads((DATA / "dataset_manifest.json").read_text())
    return {
        "experiment": "complete-fit future-continuation label-stability audit",
        "created_before_summary_computation": True,
        "scope": manifest["scope"],
        "blocks": int(manifest["counts"]["blocks"]),
        "expected_robust_positive_sides": int(
            manifest["counts"]["positive_side_labels"]
        ),
        "dataset_manifest_sha256": sha256_file(DATA / "dataset_manifest.json"),
        "comparison": "same closed-loop robust past and span15 decoder; fixed-center versus fixed-late future",
        "statistics": ["informative blocks", "strictly beneficial side actions", "sign agreement",
                       "best-action agreement", "center-future positive survival", "strict reversals",
                       "source-cluster bootstrap intervals"],
        "interpretation": "descriptive target stability; reference/future/EOS-aware and nondeployable",
        "forbidden": ["calibration outcomes", "dev", "test", "predictor tuning", "git commit"],
    }


def preregister():
    config = frozen_config()
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    OUTPUT.mkdir(parents=True)
    (OUTPUT / "resolved_config_preregistered.json").write_text(json.dumps(config, indent=2) + "\n")
    return config


def source_bootstrap(per_source, numerator, denominator, reps=10000):
    sources = sorted(per_source)
    rows = np.asarray([[per_source[source][numerator], per_source[source][denominator]]
                       for source in sources], dtype=np.int64)
    rng = np.random.default_rng(SEED)
    estimates = []
    for _ in range(reps):
        sampled = rows[rng.integers(0, len(rows), size=len(rows))].sum(axis=0)
        if sampled[1]:
            estimates.append(sampled[0] / sampled[1])
    return [float(np.quantile(estimates, 0.025)), float(np.quantile(estimates, 0.975))]


def run():
    config = preregister()
    counts = Counter()
    per_source = defaultdict(Counter)
    samples = set()
    for path in sorted((DATA / "shards").glob("labels-*.jsonl.gz")):
        with gzip.open(path, "rt", encoding="utf-8") as handle:
            for line in handle:
                row = json.loads(line)
                if row["type"] != "block":
                    continue
                samples.add(row["sample_id"])
                center = row["label"]["terminal_errors_center_future"]
                late = row["label"]["terminal_errors_late_future"]
                center_sign = side_signs(center)
                late_sign = side_signs(late)
                local = Counter()
                local["blocks"] += 1
                local["center_informative_blocks"] += len(set(center)) > 1
                local["late_informative_blocks"] += len(set(late)) > 1
                local["center_positive_sides"] += sum(value > 0 for value in center_sign)
                local["late_positive_sides"] += sum(value > 0 for value in late_sign)
                local["best_agree_all"] += stable_best(center) == stable_best(late)
                if len(set(center)) > 1 or len(set(late)) > 1:
                    local["union_informative_blocks"] += 1
                    local["best_agree_union_informative"] += stable_best(center) == stable_best(late)
                for before, after in zip(center_sign, late_sign):
                    if before != 0 or after != 0:
                        local["union_nonzero_sides"] += 1
                        local["sign_agree_union_nonzero"] += before == after
                    if before > 0:
                        local["center_positive_denominator"] += 1
                        local["center_positive_survives"] += after > 0
                        local["center_positive_reverses"] += after < 0
                    local["robust_positive_sides"] += before > 0 and after > 0
                counts.update(local)
                per_source[row["source_video_id"]].update(local)
    if counts["blocks"] != config["blocks"] or len(samples) != config["scope"]["samples"]:
        raise RuntimeError("complete-fit block or sample coverage mismatch")
    if len(per_source) != config["scope"]["sources"]:
        raise RuntimeError("complete-fit source coverage mismatch")
    if counts["robust_positive_sides"] != config["expected_robust_positive_sides"]:
        raise RuntimeError("robust-positive count does not reproduce the frozen dataset")
    value = {
        "scope": config["scope"], "samples": len(samples), "sources": len(per_source),
        "counts": dict(counts),
        "rates": {
            "best_agreement_all": counts["best_agree_all"] / counts["blocks"],
            "best_agreement_union_informative": (
                counts["best_agree_union_informative"] / max(1, counts["union_informative_blocks"])),
            "sign_agreement_union_nonzero": (
                counts["sign_agree_union_nonzero"] / max(1, counts["union_nonzero_sides"])),
            "center_positive_survival": (
                counts["center_positive_survives"] / max(1, counts["center_positive_denominator"])),
            "center_positive_reversal": (
                counts["center_positive_reverses"] / max(1, counts["center_positive_denominator"])),
        },
        "source_cluster_bootstrap_95pct": {
            "best_agreement_union_informative": source_bootstrap(
                per_source, "best_agree_union_informative", "union_informative_blocks"),
            "center_positive_survival": source_bootstrap(
                per_source, "center_positive_survives", "center_positive_denominator"),
        },
        "partial32_reference": {"center_positive_survival": 0.5,
                                "ci95": [0.36534090909090905, 0.6190886699507387]},
        "limitations": ["fit-only descriptive audit", "reference/future/EOS-aware labels",
                        "same decoder span15 only; decoder sensitivity is a different target"],
    }
    (OUTPUT / "metrics.json").write_text(json.dumps(value, indent=2) + "\n")
    return value


if __name__ == "__main__":
    print(json.dumps(run(), indent=2))
