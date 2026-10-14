#!/usr/bin/env python3
"""Paired source-cluster analysis for the P7 train-only sensitivity pilot."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import numpy as np


CSLR_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = CSLR_ROOT / "results/phoenix-2014t_ISLR/p7_input_sensitivity_train_v1/per_example.jsonl.gz"
DEFAULT_OUTPUT = CSLR_ROOT / "results/phoenix-2014t_ISLR/p7_input_sensitivity_train_v1/comparison.json"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def percentile(values: List[float], q: float) -> float:
    values = sorted(values)
    pos = (len(values) - 1) * q
    lo, hi = int(pos), min(int(pos) + 1, len(values) - 1)
    frac = pos - lo
    return values[lo] * (1 - frac) + values[hi] * frac


def paired_comparison(
    rows: List[Dict[str, Any]],
    left: Tuple[str, str],
    right: Tuple[str, str],
    short_only: bool,
    seed: int,
    bootstrap: int,
) -> Dict[str, Any]:
    index = {
        (r["sample_id"], r["candidate"], r["head"]): r
        for r in rows
        if not short_only or int(r["duration"]) <= 6
    }
    ids = sorted({k[0] for k in index if k[1:] == left} & {k[0] for k in index if k[1:] == right})
    pairs = [(index[(sid, *left)], index[(sid, *right)]) for sid in ids]
    if not pairs:
        raise ValueError((left, right, short_only))

    def delta(ps: Iterable[Tuple[Dict[str, Any], Dict[str, Any]]], field: str) -> float:
        ps = list(ps)
        return sum(float(a[field]) - float(b[field]) for a, b in ps) / len(ps)

    by_source = defaultdict(list)
    for pair in pairs:
        assert pair[0]["source"] == pair[1]["source"]
        by_source[pair[0]["source"]].append(pair)
    sources = sorted(by_source)
    source_counts = np.asarray([len(by_source[source]) for source in sources], dtype=np.float64)
    rng = np.random.default_rng(seed)
    sampled_indices = rng.integers(0, len(sources), size=(bootstrap, len(sources)))
    boot = {"top1_correct": [], "correct_probability": [], "correct_margin": []}
    for field in boot:
        source_sums = np.asarray([
            sum(float(a[field]) - float(b[field]) for a, b in by_source[source])
            for source in sources
        ])
        numer = source_sums[sampled_indices].sum(axis=1)
        denom = source_counts[sampled_indices].sum(axis=1)
        boot[field] = (numer / denom).tolist()

    fields = {}
    for field, values in boot.items():
        fields[field] = {
            "delta_left_minus_right": delta(pairs, field),
            "source_cluster_ci95": [percentile(values, 0.025), percentile(values, 0.975)],
        }
    return {
        "left": {"candidate": left[0], "head": left[1]},
        "right": {"candidate": right[0], "head": right[1]},
        "short_only_duration_le_6": short_only,
        "samples": len(pairs),
        "sources": len(sources),
        "left_correct_right_wrong": sum(a["top1_correct"] and not b["top1_correct"] for a, b in pairs),
        "left_wrong_right_correct": sum(not a["top1_correct"] and b["top1_correct"] for a, b in pairs),
        "metrics": fields,
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    p.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    p.add_argument("--bootstrap", type=int, default=2000)
    p.add_argument("--seed", type=int, default=73022)
    p.add_argument("--verify", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if args.verify:
        data = json.loads(args.output.read_text())
        assert data["input_sha256"] == sha256(args.input)
        assert data["scope"].startswith("paired train-only")
        assert len(data["comparisons"]) == 12
        print(json.dumps({"verified": True, "output": str(args.output)}, sort_keys=True))
        return

    with gzip.open(args.input, "rt", encoding="utf-8") as f:
        rows = [json.loads(line) for line in f]
    definitions = [
        (("base_pad16", "ensemble_foreground"), ("base_pad16", "ensemble_main")),
        (("inside_even16", "ensemble_main"), ("base_pad16", "ensemble_main")),
        (("context_center16", "ensemble_main"), ("base_pad16", "ensemble_main")),
        (("base_pad16", "weighted_foreground"), ("base_pad16", "weighted_main")),
        (("inside_even16", "weighted_main"), ("base_pad16", "weighted_main")),
        (("context_center16", "weighted_main"), ("base_pad16", "weighted_main")),
    ]
    comparisons = []
    for left, right in definitions:
        for short_only in [False, True]:
            comparisons.append(paired_comparison(
                rows, left, right, short_only, args.seed + len(comparisons), args.bootstrap
            ))
    out = {
        "schema_version": 1,
        "scope": "paired train-only input sensitivity; recognizer in-sample; no dev/test opened",
        "input": str(args.input.resolve()),
        "input_sha256": sha256(args.input),
        "bootstrap_replicates": args.bootstrap,
        "comparisons": comparisons,
        "interpretation_limit": "These deltas establish checkpoint input sensitivity, not the causal effect of retraining with each construction."
    }
    args.output.write_text(json.dumps(out, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"output": str(args.output)}, sort_keys=True))


if __name__ == "__main__":
    main()
