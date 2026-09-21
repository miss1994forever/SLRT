#!/usr/bin/env python3
"""Measure schedule-level latency for a wait-k G2T run.

This analysis deliberately does not report seconds.  The saved CSL-Daily G2T
inputs contain a final gloss sentence and a video frame count, but no timestamp
for when each gloss became stable.  The script therefore reports token-level
AL/AP and an explicitly labelled uniform-duration frame proxy.
"""

import argparse
import json
import pickle
from pathlib import Path

import numpy as np
import sentencepiece as spm


def percentile_summary(values):
    values = np.asarray(values, dtype=np.float64)
    if values.size == 0:
        return {key: None for key in ("mean", "p50", "p90", "p95", "p99")}
    return {
        "mean": float(np.mean(values)),
        "p50": float(np.percentile(values, 50)),
        "p90": float(np.percentile(values, 90)),
        "p95": float(np.percentile(values, 95)),
        "p99": float(np.percentile(values, 99)),
    }


def waitk_trace(source_length, target_length, wait_k):
    if source_length <= 0 or target_length <= 0:
        raise ValueError("source and target lengths must be positive")
    target_steps = np.arange(target_length, dtype=np.float64)
    return np.minimum(wait_k + target_steps, source_length)


def average_proportion(source_length, target_length, wait_k):
    reads = waitk_trace(source_length, target_length, wait_k)
    return float(np.sum(reads) / (source_length * target_length))


def average_lagging(source_length, target_length, wait_k):
    """Standard AL; return None if output ends before all source is consumed."""
    tau = max(1, source_length - wait_k + 1)
    if target_length < tau:
        return None
    target_per_source = target_length / source_length
    steps = np.arange(tau, dtype=np.float64)
    reads = np.minimum(wait_k + steps, source_length)
    return float(np.mean(reads - steps / target_per_source))


def load_pickle(path):
    with path.open("rb") as handle:
        return pickle.load(handle)


def analyze_split(source_path, result_path, processor, wait_values):
    sources = {row["name"]: row for row in load_pickle(source_path)}
    results = load_pickle(result_path)
    missing = sorted(set(results) - set(sources))
    if missing:
        raise KeyError(f"{len(missing)} result samples have no source metadata")

    base = []
    for name, result in results.items():
        source = sources[name]
        source_length = len(source["gloss"].split())
        target_length = len(processor.encode(result["txt_hyp"], out_type=int))
        if source_length <= 0 or target_length <= 0:
            continue
        base.append(
            {
                "name": name,
                "source_length": source_length,
                "target_length": target_length,
                "num_frames": int(source["num_frames"]),
            }
        )

    schedules = {}
    for wait_k in wait_values:
        als = []
        aps = []
        initial_fractions = []
        uniform_frame_proxies = []
        source_exhaustion_steps = []
        undefined_al = 0
        for row in base:
            source_length = row["source_length"]
            target_length = row["target_length"]
            al = average_lagging(source_length, target_length, wait_k)
            if al is None:
                undefined_al += 1
            else:
                als.append(al)
            aps.append(average_proportion(source_length, target_length, wait_k))
            initial_read = min(wait_k, source_length)
            initial_fraction = initial_read / source_length
            initial_fractions.append(initial_fraction)
            uniform_frame_proxies.append(row["num_frames"] * initial_fraction)
            source_exhaustion_steps.append(max(1, source_length - wait_k + 1))

        schedules[str(wait_k)] = {
            "average_lagging_gloss_tokens": percentile_summary(als),
            "average_proportion": percentile_summary(aps),
            "initial_source_fraction": percentile_summary(initial_fractions),
            "source_exhaustion_target_step": percentile_summary(source_exhaustion_steps),
            "uniform_duration_first_output_frame_proxy": percentile_summary(
                uniform_frame_proxies
            ),
            "undefined_average_lagging_samples": undefined_al,
        }

    return {
        "samples": len(base),
        "source_gloss_length": percentile_summary(
            [row["source_length"] for row in base]
        ),
        "hypothesis_sentencepiece_length": percentile_summary(
            [row["target_length"] for row in base]
        ),
        "video_frame_count": percentile_summary([row["num_frames"] for row in base]),
        "schedules": schedules,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dev-source", type=Path, required=True)
    parser.add_argument("--test-source", type=Path, required=True)
    parser.add_argument("--dev-results", type=Path, required=True)
    parser.add_argument("--test-results", type=Path, required=True)
    parser.add_argument("--sentencepiece-model", type=Path, required=True)
    parser.add_argument("--wait-k", type=int, nargs="+", default=[1, 2, 4])
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    processor = spm.SentencePieceProcessor(model_file=str(args.sentencepiece_model))
    report = {
        "metric_scope": "schedule-level latency over saved final gloss sequences",
        "target_unit": "mBART SentencePiece token",
        "source_unit": "predicted gloss token",
        "definitions": {
            "read_position": "g(t)=min(k+t-1, source_length), with t one-indexed",
            "average_lagging": (
                "mean_t[g(t)-(t-1)/(target_length/source_length)] through the "
                "first target step that has consumed the source"
            ),
            "average_proportion": "sum_t g(t)/(source_length*target_length)",
        },
        "limitations": [
            "Saved inputs contain no per-gloss stable-arrival timestamps.",
            "The uniform-duration frame value is a proxy, not measured frame latency.",
            "No camera, CSLR, queueing, controller, or network time is included.",
            "This does not measure irreversible stable-prefix commit latency.",
        ],
        "splits": {
            "dev": analyze_split(
                args.dev_source,
                args.dev_results,
                processor,
                args.wait_k,
            ),
            "test": analyze_split(
                args.test_source,
                args.test_results,
                processor,
                args.wait_k,
            ),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
