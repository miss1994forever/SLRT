#!/usr/bin/env python3
"""Analyze frozen-checkpoint P7 context-margin and frame-count grid."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import pickle
import random
from collections import Counter, defaultdict
from pathlib import Path


CSLR_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = CSLR_ROOT.parents[1]
INPUT_ROOT = CSLR_ROOT / "results/phoenix-2014t_ISLR/p7_temporal_grid_frozen_v1"
INPUT = INPUT_ROOT / "per_example.jsonl.gz"
SUMMARY = INPUT_ROOT / "summary.json"
OUTPUT = INPUT_ROOT / "analysis.json"
TRAIN_BAGS = REPO_ROOT / "data/phoenix_2014t/phoenix_iso_center_label_bag2items.train"
TRAIN_META = REPO_ROOT / "data/phoenix_2014t/phoenix14t.train"
PRIMARY_HEAD = "weighted_main"
COMPARISONS = {
    "D_margin0_to16": ["D_margin2_to16", "D_margin4_to16", "D_margin6_to16"],
    "E_context16": ["E_context8", "E_context12", "E_context24", "E_context32"],
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def frequency_bucket(count: int) -> str:
    if count <= 10:
        return "rare_1_10"
    if count <= 50:
        return "medium_11_50"
    return "common_51_plus"


def normalized_duration_bucket(row: dict) -> str:
    if str(row["reference"]) == "<blank>":
        return "blank"
    duration = int(row["duration"])
    if duration <= 6:
        return "5-6"
    if duration <= 9:
        return "7-9"
    return "10+"


def aggregate(rows: list[dict]) -> dict:
    if not rows:
        return {"n": 0}
    n = len(rows)
    return {
        "n": n,
        "top1": sum(bool(row["correct"]) for row in rows) / n,
        "top5": sum(bool(row["top5_correct"]) for row in rows) / n,
        "blank_prediction_rate": sum(row["prediction"] == "<blank>" for row in rows) / n,
        "mean_correct_probability": sum(float(row["correct_probability"]) for row in rows) / n,
    }


def stratify(rows: list[dict]) -> dict:
    nonblank = [row for row in rows if row["reference"] != "<blank>"]
    result = {"all_items": aggregate(rows), "nonblank": aggregate(nonblank)}
    for field, values in (
        ("duration_bucket", rows), ("frequency_bucket", nonblank), ("signer", rows)
    ):
        groups = defaultdict(list)
        for row in values:
            groups[str(row[field])].append(row)
        result[f"by_{field}"] = {
            key: aggregate(group) for key, group in sorted(groups.items())
        }
    return result


def subset(rows: dict[str, dict], name: str) -> dict[str, dict]:
    if name == "nonblank":
        return {key: row for key, row in rows.items() if row["reference"] != "<blank>"}
    if name == "5-6":
        return {key: row for key, row in rows.items() if row["duration_bucket"] == "5-6"}
    if name == "common":
        return {key: row for key, row in rows.items() if row["frequency_bucket"] == "common_51_plus"}
    raise ValueError(name)


def bootstrap(baseline: dict[str, dict], candidate: dict[str, dict], repetitions: int, seed: int) -> dict:
    if set(baseline) != set(candidate):
        raise RuntimeError("paired sample mismatch")
    clusters = defaultdict(list)
    observed_top1, observed_probability = [], []
    for sample_id in sorted(baseline):
        before, after = baseline[sample_id], candidate[sample_id]
        if before["source"] != after["source"]:
            raise RuntimeError(f"source mismatch: {sample_id}")
        value = (
            float(bool(after["correct"])) - float(bool(before["correct"])),
            float(after["correct_probability"]) - float(before["correct_probability"]),
        )
        clusters[before["source"]].append(value)
        observed_top1.append(value[0])
        observed_probability.append(value[1])
    keys = sorted(clusters)
    rng = random.Random(seed)
    top1_draws, probability_draws = [], []
    for _ in range(repetitions):
        sampled = [clusters[rng.choice(keys)] for _ in keys]
        flat = [value for cluster in sampled for value in cluster]
        top1_draws.append(sum(value[0] for value in flat) / len(flat))
        probability_draws.append(sum(value[1] for value in flat) / len(flat))
    top1_draws.sort(); probability_draws.sort()
    low = int(0.025 * repetitions); high = min(repetitions - 1, int(0.975 * repetitions))
    return {
        "samples": len(observed_top1),
        "source_clusters": len(keys),
        "top1_delta": sum(observed_top1) / len(observed_top1),
        "top1_ci95": [top1_draws[low], top1_draws[high]],
        "correct_probability_delta": sum(observed_probability) / len(observed_probability),
        "correct_probability_ci95": [probability_draws[low], probability_draws[high]],
        "bootstrap_repetitions": repetitions,
        "seed": seed,
    }


def changes(baseline: dict[str, dict], candidate: dict[str, dict], limit: int) -> dict:
    improved, regressed = [], []
    for sample_id in sorted(baseline):
        before, after = baseline[sample_id], candidate[sample_id]
        record = {
            "sample_id": sample_id,
            "source": before["source"],
            "duration": before["duration"],
            "reference": before["reference"],
            "baseline_hypothesis": before["prediction"],
            "candidate_hypothesis": after["prediction"],
        }
        if not before["correct"] and after["correct"]:
            improved.append(record)
        elif before["correct"] and not after["correct"]:
            regressed.append(record)
    order = lambda row: (row["duration"], row["sample_id"])
    return {
        "improvement_count": len(improved),
        "regression_count": len(regressed),
        "improvements": sorted(improved, key=order)[:limit],
        "regressions": sorted(regressed, key=order)[:limit],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=INPUT)
    parser.add_argument("--summary", type=Path, default=SUMMARY)
    parser.add_argument("--train-bags", type=Path, default=TRAIN_BAGS)
    parser.add_argument("--train-meta", type=Path, default=TRAIN_META)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--bootstrap-repetitions", type=int, default=10000)
    parser.add_argument("--example-limit", type=int, default=20)
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args()
    if args.verify:
        result = json.loads(args.output.read_text())
        for record in result["inputs"].values():
            assert sha256(Path(record["path"])) == record["sha256"]
        assert result["scope"].endswith("no dev/test")
        print(json.dumps({"verified": True, "output": str(args.output.resolve())}, sort_keys=True))
        return
    if args.bootstrap_repetitions < 1000:
        raise SystemExit("--bootstrap-repetitions must be >= 1000")
    with gzip.open(args.input, "rt", encoding="utf-8") as handle:
        rows = [json.loads(line) for line in handle]
    with args.train_bags.open("rb") as handle:
        bags = pickle.load(handle)
    bases = [next(item for item in values if int(item.get("aug", 0)) == 0) for values in bags.values()]
    frequencies = Counter(str(item["label"]) for item in bases if str(item["label"]) != "<blank>")
    with gzip.open(args.train_meta, "rb") as handle:
        metadata = pickle.load(handle)
    signers = {str(item["name"]): str(item["signer"]) for item in metadata}
    grouped = defaultdict(list)
    for row in rows:
        row["duration_bucket"] = normalized_duration_bucket(row)
        row["frequency"] = frequencies[str(row["reference"])]
        row["frequency_bucket"] = frequency_bucket(row["frequency"])
        row["signer"] = signers[str(row["source"])]
        grouped[(row["candidate"], row["head"])].append(row)
    primary = {}
    results = {}
    for (candidate, head), values in sorted(grouped.items()):
        results[f"{candidate}::{head}"] = stratify(values)
        if head == PRIMARY_HEAD:
            mapping = {str(row["sample_id"]): row for row in values}
            if len(mapping) != len(values):
                raise RuntimeError(f"duplicate sample IDs: {candidate}")
            primary[candidate] = mapping
    comparisons = {}
    examples = {}
    comparison_index = 0
    for baseline, candidates in COMPARISONS.items():
        for candidate in candidates:
            comparison_index += 1
            key = f"{candidate}_minus_{baseline}"
            comparisons[key] = {
                name: bootstrap(
                    subset(primary[baseline], name), subset(primary[candidate], name),
                    args.bootstrap_repetitions, 73060 + 10 * comparison_index + index,
                )
                for index, name in enumerate(("nonblank", "5-6", "common"))
            }
            examples[key] = changes(
                subset(primary[baseline], "nonblank"),
                subset(primary[candidate], "nonblank"), args.example_limit,
            )
    runtime = json.loads(args.summary.read_text())["runtime"]
    output = {
        "schema_version": 1,
        "scope": "frozen checkpoint source-disjoint train-only D/E sensitivity; no dev/test",
        "primary_head": PRIMARY_HEAD,
        "primary_population": "nonblank",
        "results": results,
        "paired_source_cluster_bootstrap": comparisons,
        "representative_changes": examples,
        "runtime": runtime,
        "inputs": {
            "predictions": {"path": str(args.input.resolve()), "sha256": sha256(args.input)},
            "summary": {"path": str(args.summary.resolve()), "sha256": sha256(args.summary)},
            "train_bags": {"path": str(args.train_bags.resolve()), "sha256": sha256(args.train_bags)},
            "train_meta": {"path": str(args.train_meta.resolve()), "sha256": sha256(args.train_meta)},
        },
    }
    args.output.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"output": str(args.output.resolve()), "rows": len(rows)}, sort_keys=True))


if __name__ == "__main__":
    main()
