#!/usr/bin/env python3
"""Measure the compatible Phoenix wait-2 text stage on fixed CSLR gloss traces."""
import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from modelling.model import build_model
from utils.misc import load_config, make_logger


def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(b)
    return h.hexdigest()


def generate(model, gloss, max_length):
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    tokenized = model.gloss_tokenizer(batch_gls_seq=[gloss])
    ids = tokenized["input_ids"].to("cuda")
    mask = tokenized["attention_mask"].to("cuda")
    embeds = model.translation_network.prepare_gloss_inputs(ids)
    result = model.translation_network.generate(inputs_embeds=embeds, attention_mask=mask, num_beams=5, max_length=max_length, length_penalty=1)
    torch.cuda.synchronize()
    return time.perf_counter()-t0, result["decoded_sequences"][0]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    plan = json.loads(args.plan.read_text())
    if os.environ.get("CUDA_VISIBLE_DEVICES") != plan["gpu_uuid"] or plan["gpu_uuid"] != "GPU-91f9e38c-0a59-15c0-bd08-6a4f89da07ed":
        raise ValueError("text GPU identity mismatch")
    source = Path(plan["source"])
    if sha(source) != plan["source_sha256"] or sha(__file__) != plan["script_sha256"]:
        raise ValueError("text plan identity mismatch")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable")
    cfg = load_config(str(ROOT / "configs/g2t_wait2.yaml"))
    cfg["device"] = torch.device("cuda")
    make_logger(str(args.output.parent), log_file="p6_wait2_text_20260928.log")
    model = build_model(cfg)
    ckpt = torch.load(ROOT / "results/g2t_wait2/ckpts/best.ckpt", map_location="cuda")
    model.load_state_dict(ckpt["model_state"], strict=True)
    model.eval()
    rows = [json.loads(line) for line in source.read_text().splitlines()]
    rows = [row for row in rows if row["repetition"] == plan["repetition"]]
    output = []
    with torch.inference_mode():
        for row in rows:
            committed = [e for e in row["events"] if e["kind"] == "paid_window" and len(e["committed_gloss"]) >= 2]
            if not committed:
                raise RuntimeError("two stable glosses never arrived")
            first = committed[0]
            prefix = " ".join(first["committed_gloss"][:2])
            first_s, first_text = generate(model, prefix, 3)
            final_s, final_text = generate(model, " ".join(row["hypothesis"]), 100)
            result = {"sample": row["sample"], "policy": row["policy"], "repetition": row["repetition"],
                      "two_stable_gloss_ready_s": first["t_s"], "first_text_compute_s": first_s,
                      "first_text_if_uncontended_s": first["t_s"] + first_s,
                      "first_text": first_text, "final_gloss_ready_s": row["wall_s"],
                      "final_text_compute_s": final_s, "final_text": final_text,
                      "scope": "G2T computation after recorded gloss readiness; non-concurrent diagnostic"}
            output.append(result)
            print(json.dumps(result, ensure_ascii=False), flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as f:
        for item in output:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
