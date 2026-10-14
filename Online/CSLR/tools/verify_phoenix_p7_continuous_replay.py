#!/usr/bin/env python3
"""Verify P7 continuous replay against frozen P6 per-sample outcomes."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import pickle
from pathlib import Path

import numpy as np

from analyze_phoenix_p7_continuous_smoke import (
    P6_MODELS,
    P6_RULE,
    VOCAB,
    FrozenRidgeBundle,
    clean_phoenix_2014_trans,
    decode,
    frozen_schedules,
    sentence_row,
    sha256,
)


CSLR_ROOT = Path(__file__).resolve().parents[1]
BASE = CSLR_ROOT / "results/phoenix-2014t_ISLR"
FROZEN = BASE / "p6_k_history_online_quantile_oof_v1_49faacc3/per_sample.jsonl.gz"
SHARD = BASE / "train_dense_stride1_v1_49faacc3/shards/shard-00000-of-00056"
OUTPUT = BASE / "p7_continuous_replay_parity_v1/summary.json"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--frozen", type=Path, default=FROZEN)
    parser.add_argument("--shard", type=Path, default=SHARD)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args()
    if args.verify:
        result = json.loads(args.output.read_text())
        for record in result["inputs"].values():
            assert sha256(Path(record["path"])) == record["sha256"]
        assert result["checks"]["zero_mismatches"]
        assert result["scope"].endswith("no dev/test")
        print(json.dumps({"verified": True, "output": str(args.output.resolve())}, sort_keys=True))
        return

    with gzip.open(args.frozen, "rt", encoding="utf-8") as handle:
        frozen = {row["sample"]: row for row in map(json.loads, handle)}
    results_path = args.shard / "train_results.pkl"
    logits_path = args.shard / "train_logits.pkl"
    with results_path.open("rb") as handle:
        results = pickle.load(handle)
    with logits_path.open("rb") as handle:
        logits = pickle.load(handle)
    vocab = json.loads(VOCAB.read_text())
    blank = vocab.index("<blank>")
    bundle = FrozenRidgeBundle(P6_MODELS, P6_RULE)
    mismatches = []
    checked = 0
    for name in sorted(set(results) & set(frozen)):
        values = np.asarray(logits[name], np.float32)
        starts = [int(item["start"]) for item in results[name]["adaptive_stride_metadata"]]
        selected = frozen_schedules(values, name, blank, bundle)["quantile"]
        hypothesis = clean_phoenix_2014_trans(decode(values, starts, selected, vocab, blank))
        reference = clean_phoenix_2014_trans(results[name]["gls_ref"])
        observed = sentence_row(name, reference, hypothesis, len(selected))
        expected = frozen[name]
        if (
            observed["error"] != expected["quantile"]
            or observed["windows"] != expected["quantile_windows"]
        ):
            mismatches.append({
                "sample": name,
                "observed_error": observed["error"],
                "expected_error": expected["quantile"],
                "observed_windows": observed["windows"],
                "expected_windows": expected["quantile_windows"],
            })
        checked += 1
    if checked < 50 or mismatches:
        raise RuntimeError({"checked": checked, "mismatches": mismatches[:10]})
    script = Path(__file__).resolve()
    replay = CSLR_ROOT / "tools/analyze_phoenix_p7_continuous_smoke.py"
    output = {
        "schema_version": 1,
        "scope": "frozen P6 train-fit replay parity; no dev/test",
        "checked_samples": checked,
        "mismatch_count": len(mismatches),
        "checks": {
            "at_least_50_samples": checked >= 50,
            "quantile_window_count_exact": True,
            "quantile_sentence_error_count_exact": True,
            "zero_mismatches": not mismatches,
        },
        "inputs": {
            "frozen_per_sample": {"path": str(args.frozen.resolve()), "sha256": sha256(args.frozen)},
            "shard_results": {"path": str(results_path.resolve()), "sha256": sha256(results_path)},
            "shard_logits": {"path": str(logits_path.resolve()), "sha256": sha256(logits_path)},
            "p6_rule": {"path": str(P6_RULE.resolve()), "sha256": sha256(P6_RULE)},
            "p6_models": {"path": str(P6_MODELS.resolve()), "sha256": sha256(P6_MODELS)},
            "vocab": {"path": str(VOCAB.resolve()), "sha256": sha256(VOCAB)},
            "replay_script": {"path": str(replay), "sha256": sha256(replay)},
            "verification_script": {"path": str(script), "sha256": sha256(script)},
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"output": str(args.output.resolve()), "checked": checked}, sort_keys=True))


if __name__ == "__main__":
    main()
