#!/usr/bin/env python3
"""Benchmark G2T-only runtime for a trained wait-k checkpoint.

The benchmark measures compute after glosses are already available.  It does
not include camera capture, CSLR, gloss stabilization, or queueing latency.
"""

import argparse
import json
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import torch

SLT_ROOT = Path(__file__).resolve().parents[1]
if str(SLT_ROOT) not in sys.path:
    sys.path.insert(0, str(SLT_ROOT))

from modelling.model import build_model
from utils.misc import load_config, make_logger


def summary(values):
    values = np.asarray(values, dtype=np.float64)
    return {
        "mean_ms": float(np.mean(values)),
        "p50_ms": float(np.percentile(values, 50)),
        "p90_ms": float(np.percentile(values, 90)),
        "p95_ms": float(np.percentile(values, 95)),
        "p99_ms": float(np.percentile(values, 99)),
    }


def generate(model, glosses, num_beams, max_length, device):
    tokenized = model.gloss_tokenizer(batch_gls_seq=glosses)
    input_ids = tokenized["input_ids"].to(device)
    attention_mask = tokenized["attention_mask"].to(device)
    inputs_embeds = model.translation_network.prepare_gloss_inputs(input_ids)
    return model.translation_network.generate(
        inputs_embeds=inputs_embeds,
        attention_mask=attention_mask,
        num_beams=num_beams,
        max_length=max_length,
        length_penalty=1,
    )


def timed_generate(model, glosses, num_beams, max_length, device):
    torch.cuda.synchronize(device)
    started = time.perf_counter()
    output = generate(model, glosses, num_beams, max_length, device)
    torch.cuda.synchronize(device)
    return (time.perf_counter() - started) * 1000.0, output


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--wait-k", type=int, default=2)
    parser.add_argument("--num-beams", type=int, default=5)
    parser.add_argument("--max-length", type=int, default=100)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--max-samples", type=int, default=300)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; refusing to label CPU timing as GPU latency")
    device = torch.device("cuda:0")
    cfg = load_config(str(args.config))
    cfg["device"] = device
    make_logger(str(args.output.parent), log_file="waitk_runtime_benchmark.log")
    model = build_model(cfg)
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint["model_state"], strict=True)
    model.eval()

    with args.source.open("rb") as handle:
        rows = pickle.load(handle)
    rows = [row for row in rows if row["gloss"].strip()]
    if args.max_samples > 0:
        rows = rows[: args.warmup + args.max_samples]
    if len(rows) <= args.warmup:
        raise ValueError("not enough samples after applying warmup")

    first_token_ms = []
    complete_sentence_ms = []
    with torch.inference_mode():
        for index, row in enumerate(rows):
            gloss_tokens = row["gloss"].split()
            prefix = " ".join(gloss_tokens[: args.wait_k])
            first_ms, _ = timed_generate(
                model,
                [prefix],
                args.num_beams,
                2,
                device,
            )
            full_ms, _ = timed_generate(
                model,
                [row["gloss"]],
                args.num_beams,
                args.max_length,
                device,
            )
            if index >= args.warmup:
                first_token_ms.append(first_ms)
                complete_sentence_ms.append(full_ms)

    report = {
        "scope": "G2T-only batch-1 synchronized host wall time",
        "gpu": torch.cuda.get_device_name(device),
        "wait_k": args.wait_k,
        "num_beams": args.num_beams,
        "warmup_samples": args.warmup,
        "measured_samples": len(first_token_ms),
        "first_token_after_k_glosses_available": summary(first_token_ms),
        "complete_sentence_from_final_gloss_sequence": summary(complete_sentence_ms),
        "excluded": [
            "time until the first k stable glosses arrive",
            "camera and video preprocessing",
            "CSLR inference and stabilization",
            "queueing, controller, and network time",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
