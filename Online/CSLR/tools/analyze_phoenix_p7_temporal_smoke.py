#!/usr/bin/env python3
"""Analyze trained P7 A/B/C recognizers on train-only source-disjoint calibration.

The primary comparison is frozen before result inspection: every model is evaluated
with the same context_center16 input and weighted_main head.  Other input/head
combinations remain diagnostics.
"""

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
EVAL_ROOT = CSLR_ROOT / "results/phoenix-2014t_ISLR/p7_temporal_smoke_eval_v1"
TRAIN_BAGS = REPO_ROOT / "data/phoenix_2014t/phoenix_iso_center_label_bag2items.train"
TRAIN_META = REPO_ROOT / "data/phoenix_2014t/phoenix14t.train"
OUTPUT = EVAL_ROOT / "analysis.json"
CANDIDATES = ["A_current_fixed16", "B_aligned_fixed16", "C_inside_even16"]
PRIMARY_INPUT = "context_center16"
BASE_ONLY_INPUT = "base_pad16"
PRIMARY_HEAD = "weighted_main"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def duration_bucket(duration: int, label: str) -> str:
    if label == "<blank>":
        return "blank"
    if duration <= 6:
        return "5-6"
    if duration <= 9:
        return "7-9"
    return "10+"


def frequency_bucket(count: int) -> str:
    if count <= 10:
        return "rare_1_10"
    if count <= 50:
        return "medium_11_50"
    return "common_51_plus"


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
    nonblank = [row for row in rows if str(row["label"]) != "<blank>"]
    result = {"all_items": aggregate(rows), "nonblank": aggregate(nonblank)}
    for field in ["duration_bucket", "signer"]:
        groups = defaultdict(list)
        for row in rows:
            groups[str(row[field])].append(row)
        result[f"by_{field}"] = {
            key: aggregate(value) for key, value in sorted(groups.items())
        }
    frequency_groups = defaultdict(list)
    for row in nonblank:
        frequency_groups[str(row["frequency_bucket"])].append(row)
    result["by_frequency_bucket"] = {
        key: aggregate(value) for key, value in sorted(frequency_groups.items())
    }
    return result


def cluster_bootstrap_delta(
    baseline: dict[str, dict], candidate: dict[str, dict], *, repetitions: int, seed: int
) -> dict:
    if set(baseline) != set(candidate):
        raise RuntimeError("candidate sample IDs differ")
    by_source = defaultdict(list)
    observed = []
    for sample_id in sorted(baseline):
        before, after = baseline[sample_id], candidate[sample_id]
        if before["source"] != after["source"]:
            raise RuntimeError(f"source mismatch for {sample_id}")
        delta = float(bool(after["correct"])) - float(bool(before["correct"]))
        by_source[str(before["source"])].append(delta)
        observed.append(delta)
    clusters = sorted(by_source)
    rng = random.Random(seed)
    draws = []
    for _ in range(repetitions):
        total = 0.0
        count = 0
        for _ in clusters:
            cluster = by_source[rng.choice(clusters)]
            total += sum(cluster)
            count += len(cluster)
        draws.append(total / count)
    draws.sort()
    low = draws[int(0.025 * repetitions)]
    high = draws[min(repetitions - 1, int(0.975 * repetitions))]
    return {
        "metric": "paired top1 delta; candidate minus A",
        "clusters": len(clusters),
        "samples": len(observed),
        "delta": sum(observed) / len(observed),
        "ci95": [low, high],
        "bootstrap_repetitions": repetitions,
        "seed": seed,
    }


def changed_examples(baseline: dict[str, dict], candidate: dict[str, dict], limit: int) -> dict:
    improvements, regressions = [], []
    for sample_id in sorted(baseline):
        before, after = baseline[sample_id], candidate[sample_id]
        record = {
            "sample_id": sample_id,
            "source": before["source"],
            "duration": before["duration"],
            "reference": before["reference"],
            "A_hypothesis": before["prediction"],
            "candidate_hypothesis": after["prediction"],
            "A_correct_probability": before["correct_probability"],
            "candidate_correct_probability": after["correct_probability"],
        }
        if not before["correct"] and after["correct"]:
            improvements.append(record)
        elif before["correct"] and not after["correct"]:
            regressions.append(record)
    order = lambda row: (row["duration"], row["sample_id"])
    return {
        "improvements": sorted(improvements, key=order)[:limit],
        "regressions": sorted(regressions, key=order)[:limit],
        "improvement_count": len(improvements),
        "regression_count": len(regressions),
    }


def select_rows(rows: dict[str, dict], subset: str) -> dict[str, dict]:
    if subset == "nonblank":
        return {key: row for key, row in rows.items() if str(row["label"]) != "<blank>"}
    if subset == "5-6":
        return {key: row for key, row in rows.items() if row["duration_bucket"] == "5-6"}
    raise ValueError(subset)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval-root", type=Path, default=EVAL_ROOT)
    parser.add_argument("--train-bags", type=Path, default=TRAIN_BAGS)
    parser.add_argument("--train-meta", type=Path, default=TRAIN_META)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--bootstrap-repetitions", type=int, default=10000)
    parser.add_argument("--example-limit", type=int, default=20)
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args()
    if args.verify:
        result = json.loads(args.output.read_text())
        for record in result["inputs"]["predictions"].values():
            assert sha256(Path(record["path"])) == record["sha256"]
        assert result["scope"].endswith("official dev/test forbidden")
        print(json.dumps({"verified": True, "output": str(args.output.resolve())}, sort_keys=True))
        return

    if args.bootstrap_repetitions < 1000:
        raise SystemExit("--bootstrap-repetitions must be >= 1000")
    with args.train_bags.open("rb") as handle:
        bags = pickle.load(handle)
    base_items = [next(item for item in items if int(item.get("aug", 0)) == 0) for items in bags.values()]
    frequencies = Counter(str(item["label"]) for item in base_items if str(item["label"]) != "<blank>")
    with gzip.open(args.train_meta, "rb") as handle:
        train_meta = pickle.load(handle)
    signers = {str(item["name"]): str(item["signer"]) for item in train_meta}

    input_records = {}
    rows_by_candidate = {}
    for candidate in CANDIDATES:
        path = args.eval_root / candidate / "per_example.jsonl.gz"
        if not path.is_file():
            raise FileNotFoundError(path)
        with gzip.open(path, "rt", encoding="utf-8") as handle:
            rows = [json.loads(line) for line in handle]
        for row in rows:
            label = str(row["label"])
            source = str(row["source"])
            row["duration_bucket"] = duration_bucket(int(row["duration"]), label)
            row["frequency"] = frequencies[label]
            row["frequency_bucket"] = frequency_bucket(row["frequency"])
            row["signer"] = signers[source]
        rows_by_candidate[candidate] = rows
        input_records[candidate] = {"path": str(path.resolve()), "sha256": sha256(path)}

    diagnostic = {}
    primary = {}
    base_only = {}
    for candidate, rows in rows_by_candidate.items():
        groups = defaultdict(list)
        for row in rows:
            groups[(row["input"], row["head"])].append(row)
        diagnostic[candidate] = {
            f"{input_name}::{head}": stratify(values)
            for (input_name, head), values in sorted(groups.items())
        }
        selected = [
            row for row in rows
            if row["input"] == PRIMARY_INPUT and row["head"] == PRIMARY_HEAD
        ]
        if not selected:
            raise RuntimeError(f"missing primary rows for {candidate}")
        primary[candidate] = {str(row["sample_id"]): row for row in selected}
        base_selected = [
            row for row in rows
            if row["input"] == BASE_ONLY_INPUT and row["head"] == PRIMARY_HEAD
        ]
        if not base_selected:
            raise RuntimeError(f"missing base-only rows for {candidate}")
        base_only[candidate] = {str(row["sample_id"]): row for row in base_selected}

    baseline = primary[CANDIDATES[0]]
    comparisons = {}
    examples = {}
    for index, candidate in enumerate(CANDIDATES[1:], start=1):
        comparison = {}
        for input_index, (input_name, collection) in enumerate(
            ((PRIMARY_INPUT, primary), (BASE_ONLY_INPUT, base_only))
        ):
            for subset_index, subset in enumerate(("nonblank", "5-6")):
                comparison[f"{input_name}::{subset}"] = cluster_bootstrap_delta(
                    select_rows(collection[CANDIDATES[0]], subset),
                    select_rows(collection[candidate], subset),
                    repetitions=args.bootstrap_repetitions,
                    seed=73040 + 100 * input_index + 10 * subset_index + index,
                )
        comparisons[f"{candidate}_minus_{CANDIDATES[0]}"] = comparison
        examples[candidate] = changed_examples(
            select_rows(baseline, "nonblank"),
            select_rows(primary[candidate], "nonblank"),
            args.example_limit,
        )

    result = {
        "schema_version": 1,
        "scope": "trained recognizers on source-disjoint train-only calibration; official dev/test forbidden",
        "primary_comparison": {
            "deployment_input": PRIMARY_INPUT,
            "required_isolated_input": BASE_ONLY_INPUT,
            "head": PRIMARY_HEAD,
            "accuracy_population": "nonblank; blank reported separately",
        },
        "frequency_bins": {
            "rare_1_10": "1-10 base bags",
            "medium_11_50": "11-50 base bags",
            "common_51_plus": ">=51 base bags",
        },
        "primary_results": {
            candidate: stratify(list(rows.values())) for candidate, rows in primary.items()
        },
        "required_base_only_results": {
            candidate: stratify(list(rows.values())) for candidate, rows in base_only.items()
        },
        "paired_cluster_bootstrap": comparisons,
        "representative_changes": examples,
        "diagnostic_results": diagnostic,
        "inputs": {
            "predictions": input_records,
            "train_bags": {"path": str(args.train_bags.resolve()), "sha256": sha256(args.train_bags)},
            "train_meta": {"path": str(args.train_meta.resolve()), "sha256": sha256(args.train_meta)},
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"output": str(args.output.resolve()), "candidates": CANDIDATES}, sort_keys=True))


if __name__ == "__main__":
    main()
