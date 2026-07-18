#!/usr/bin/env python3
"""Evaluate span-weighted decoding from saved *dev* logits.

This utility intentionally accepts a dev result directory and refuses paths
containing a test component, so it can be used for parameter selection without
accidentally consulting the held-out test set.
"""

import argparse
import json
import pickle
import sys
from itertools import groupby
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from utils.adaptive_stride import span_weighted_predictions
from utils.metrics import wer_list
from utils.phoenix_cleanup import clean_phoenix_2014_trans


def map_phoenix_gloss(gloss):
    # Keep this mapping byte-for-byte equivalent to prediction_slide.index2token.
    for prefix, offset in (("neg-", 4), ("poss-", 5), ("negalp-", 7), ("loc-", 7), ("cl-", 7)):
        if gloss.startswith(prefix):
            return prefix + gloss[offset:].upper()
    return gloss.upper()


def decode(logits, metadata, vocab, blank_id, span, min_weight):
    if len(logits) != len(metadata):
        raise ValueError(f"logit/metadata length mismatch: {len(logits)} != {len(metadata)}")
    centers = [float(item["start"]) + 7.5 for item in metadata]
    token_ids = span_weighted_predictions(
        torch.from_numpy(logits), centers, span=span, min_weight=min_weight
    ).tolist()
    collapsed = [token for token, _ in groupby(token_ids) if token != blank_id]
    return " ".join(map_phoenix_gloss(vocab[token]) for token in collapsed)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("result_dir", type=Path)
    parser.add_argument("--vocab", type=Path, required=True)
    parser.add_argument("--spans", type=float, nargs="+", default=[7, 9, 11, 13, 15])
    parser.add_argument("--min-weight", type=float, default=0.05)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    resolved_parts = {part.lower() for part in args.result_dir.resolve().parts}
    if "test" in resolved_parts or any("test" == part.split("_")[-1] for part in resolved_parts):
        raise ValueError("This tuning utility refuses test result paths")

    with (args.result_dir / "dev_results.pkl").open("rb") as handle:
        results = pickle.load(handle)
    with (args.result_dir / "dev_logits.pkl").open("rb") as handle:
        logits = pickle.load(handle)
    with args.vocab.open("r", encoding="utf-8") as handle:
        vocab = json.load(handle)
    blank_id = vocab.index("<blank>")

    references = []
    hypotheses = {span: [] for span in args.spans}
    stride_counts = {}
    clip_count = 0
    for name, sample in results.items():
        if name not in logits:
            raise KeyError(f"missing logits for {name}")
        metadata = sample["adaptive_stride_metadata"]
        clip_count += len(metadata)
        references.append(clean_phoenix_2014_trans(sample["gls_ref"]))
        for item in metadata:
            stride = int(item["stride"])
            stride_counts[stride] = stride_counts.get(stride, 0) + 1
        for span in args.spans:
            hypothesis = decode(logits[name], metadata, vocab, blank_id, span, args.min_weight)
            hypotheses[span].append(clean_phoenix_2014_trans(hypothesis))

    summary = {
        "result_dir": str(args.result_dir),
        "sample_count": len(results),
        "clip_count": clip_count,
        "stride_counts": dict(sorted(stride_counts.items())),
        "spans": {},
    }
    for span in args.spans:
        metrics = wer_list(references, hypotheses[span])
        summary["spans"][str(span)] = {
            key: float(metrics[key]) if key in ("wer", "del", "ins", "sub") else int(metrics[key])
            for key in ("wer", "del", "ins", "sub", "ref_len", "error")
        }

    rendered = json.dumps(summary, ensure_ascii=False, indent=2)
    print(rendered)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
