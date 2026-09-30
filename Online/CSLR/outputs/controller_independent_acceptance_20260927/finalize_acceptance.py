#!/usr/bin/env python3
"""Check saved block trace against frozen per-sample records and seal provenance."""
import gzip
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from statistics import median

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
BASE = ROOT / "results/phoenix-2014t_ISLR"
FROZEN_ROWS = BASE / "p6_k_history_online_quantile_oof_v1_49faacc3/per_sample.jsonl.gz"


def sha(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    report = json.loads((OUT / "full_report.json").read_text())
    if not report["passed"] or not report["complete_fit"]:
        raise ValueError("direct full replay did not pass")
    with gzip.open(FROZEN_ROWS, "rt") as stream:
        frozen = {row["sample"]: row for row in map(json.loads, stream)}
    state = defaultdict(lambda: {"original_scores": [], "release_scores": [],
                                 "original_token": 0.0, "forced": 0, "autonomous": 0, "bonus": 0})
    mismatch = 0
    blocks = 0
    output = OUT / "trace_full_enriched.jsonl.gz"
    with (OUT / "trace_full.jsonl").open() as source, gzip.open(output, "wt", compresslevel=6) as target:
        for line in source:
            row = json.loads(line)
            sample = row["sample"]
            s = state[sample]
            if row["block"] != len(s["original_scores"]):
                raise ValueError("nonsequential block trace")
            original_threshold = median(s["original_scores"]) if s["original_scores"] else None
            release_threshold = median(s["release_scores"]) if s["release_scores"] else None
            original_token_before = s["original_token"]
            s["original_token"] = min(2.0, original_token_before + 0.5)
            if row["original_action"]:
                if s["original_token"] >= 2.0:
                    s["forced"] += 1
                else:
                    s["autonomous"] += 1
                s["bonus"] += 1
                s["original_token"] -= 1.0
            row.update({"original_threshold": original_threshold,
                        "release_threshold": release_threshold,
                        "original_token_before": original_token_before,
                        "original_token_after": s["original_token"],
                        "coverage_window": row["start"],
                        "bonus_window": row["start"] + 2})
            mismatch += (row["original_action"] != row["release_action"] or
                         row["token_after"] != s["original_token"] or
                         row["threshold"] != original_threshold)
            s["original_scores"].append(row["original_score"])
            s["release_scores"].append(row["release_score"])
            blocks += 1
            target.write(json.dumps(row, separators=(",", ":")) + "\n")
    for name, s in state.items():
        record = frozen[name]
        if (s["forced"] != record["forced"] or s["autonomous"] != record["autonomous"] or
            s["original_token"] != record["unspent"] or
            s["bonus"] + len(s["original_scores"]) + (record["full_center_windows"] % 2) != record["quantile_windows"]):
            mismatch += 1
    if len(state) != 6378 or blocks != 183401 or len(frozen) != len(state):
        raise ValueError("coverage differs from frozen fit identity")
    result = {"passed": mismatch == 0, "samples": len(state), "blocks": blocks,
              "action_token_and_frozen_count_mismatches": mismatch,
              "source_script_sha256": sha(OUT / "independent_acceptance.py"),
              "finalize_script_sha256": sha(Path(__file__)),
              "full_report_sha256": sha(OUT / "full_report.json"),
              "trace_full_sha256": sha(OUT / "trace_full.jsonl"),
              "trace_full_enriched_sha256": sha(output),
              "frozen_per_sample_sha256": sha(FROZEN_ROWS),
              "split_manifest_sha256": sha(BASE / "train_dense_stride1_v1_49faacc3/fit_calibration_split.json"),
              "release_model_sha256": report["model"]["model_sha256"],
              "frozen_script_sha256": report["frozen_script_sha256"],
              "release_controller_sha256": report["controller_sha256"]}
    (OUT / "final_acceptance.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    if not result["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
