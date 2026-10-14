#!/usr/bin/env python3
"""Frequency, signer, and duration strata for the frozen P7 sensitivity run."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import pickle
from collections import Counter, defaultdict
from pathlib import Path


CSLR_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = CSLR_ROOT.parents[1]
INPUT = CSLR_ROOT / "results/phoenix-2014t_ISLR/p7_input_sensitivity_train_v1/per_example.jsonl.gz"
TRAIN_BAGS = REPO_ROOT / "data/phoenix_2014t/phoenix_iso_center_label_bag2items.train"
TRAIN_META = REPO_ROOT / "data/phoenix_2014t/phoenix14t.train"
OUTPUT = CSLR_ROOT / "results/phoenix-2014t_ISLR/p7_input_sensitivity_train_v1/stratified.json"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def frequency_bin(count: int) -> str:
    if count <= 10:
        return "rare_1_10"
    if count <= 50:
        return "medium_11_50"
    return "common_51_plus"


def aggregate(rows):
    return {
        "n": len(rows),
        "top1": sum(bool(row["top1_correct"]) for row in rows) / len(rows),
        "top5": sum(bool(row["top5_correct"]) for row in rows) / len(rows),
        "blank_prediction_rate": sum(row["pred"] == "<blank>" for row in rows) / len(rows),
        "mean_correct_probability": sum(float(row["correct_probability"]) for row in rows) / len(rows),
        "mean_correct_margin": sum(float(row["correct_margin"]) for row in rows) / len(rows),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=INPUT)
    parser.add_argument("--train-bags", type=Path, default=TRAIN_BAGS)
    parser.add_argument("--train-meta", type=Path, default=TRAIN_META)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args()
    if args.verify:
        result = json.loads(args.output.read_text())
        assert result["inputs"]["predictions"]["sha256"] == sha256(args.input)
        assert result["scope"].endswith("no dev/test")
        print(json.dumps({"verified": True, "output": str(args.output)}, sort_keys=True))
        return

    with gzip.open(args.input, "rt", encoding="utf-8") as handle:
        rows = [json.loads(line) for line in handle]
    with args.train_bags.open("rb") as handle:
        bags = pickle.load(handle)
    base_items = [next(item for item in items if int(item.get("aug", 0)) == 0) for items in bags.values()]
    frequencies = Counter(str(item["label"]) for item in base_items if str(item["label"]) != "<blank>")
    with gzip.open(args.train_meta, "rb") as handle:
        metadata = pickle.load(handle)
    signer = {str(item["name"]): str(item["signer"]) for item in metadata}

    grouped = defaultdict(list)
    for row in rows:
        row["frequency"] = frequencies[str(row["label"])]
        row["frequency_bin"] = frequency_bin(row["frequency"])
        row["signer"] = signer[str(row["source"])]
        grouped[(row["candidate"], row["head"])].append(row)

    results = {}
    for key, values in sorted(grouped.items()):
        entry = {"overall": aggregate(values)}
        for field in ["duration_bucket", "frequency_bin", "signer"]:
            strata = defaultdict(list)
            for row in values:
                strata[str(row[field])].append(row)
            entry[f"by_{field}"] = {name: aggregate(group) for name, group in sorted(strata.items())}
        results["::".join(key)] = entry
    output = {
        "schema_version": 1,
        "scope": "frozen-checkpoint train-only sensitivity stratification; no dev/test",
        "frequency_bins": {"rare_1_10": "1-10 base bags", "medium_11_50": "11-50 base bags", "common_51_plus": ">=51 base bags"},
        "inputs": {
            "predictions": {"path": str(args.input.resolve()), "sha256": sha256(args.input)},
            "train_bags": {"path": str(args.train_bags.resolve()), "sha256": sha256(args.train_bags)},
            "train_meta": {"path": str(args.train_meta.resolve()), "sha256": sha256(args.train_meta)},
        },
        "results": results,
    }
    args.output.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"output": str(args.output), "rows": len(rows)}, sort_keys=True))


if __name__ == "__main__":
    main()
