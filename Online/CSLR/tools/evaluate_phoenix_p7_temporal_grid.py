#!/usr/bin/env python3
"""Frozen-checkpoint evaluation of P7 context-margin and frame-count grid."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import pickle
import sys
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import torch


CSLR_ROOT = Path(__file__).resolve().parents[1]
if str(CSLR_ROOT) not in sys.path:
    sys.path.insert(0, str(CSLR_ROOT))

from dataset.Dataloader import build_dataloader  # noqa: E402
from tools.islr_dev_diagnostic import load_model  # noqa: E402
from utils.misc import load_config, make_logger, move_to_device, set_seed  # noqa: E402


ASSETS = CSLR_ROOT / "results/phoenix-2014t_ISLR/p7_temporal_grid_assets_v1"
CHECKPOINT = CSLR_ROOT / "results/phoenix-2014t_ISLR/ckpts/best.ckpt"
OUTPUT = CSLR_ROOT / "results/phoenix-2014t_ISLR/p7_temporal_grid_frozen_v1"
WEIGHTS = (0.10, 0.70, 0.20)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def duration_bucket(item):
    if str(item["label"]) == "<blank>":
        return "blank"
    duration = int(item["base_end"]) - int(item["base_start"])
    if duration <= 6:
        return "1-6"
    if duration <= 9:
        return "7-9"
    if duration <= 15:
        return "10-15"
    return "16+"


def aggregate(rows):
    return {
        "n": len(rows),
        "top1": sum(row["correct"] for row in rows) / len(rows),
        "top5": sum(row["top5_correct"] for row in rows) / len(rows),
        "blank_prediction_rate": sum(row["prediction"] == "<blank>" for row in rows) / len(rows),
        "mean_correct_probability": sum(row["correct_probability"] for row in rows) / len(rows),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--assets", type=Path, default=ASSETS)
    parser.add_argument("--checkpoint", type=Path, default=CHECKPOINT)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args()
    args.output = args.output.resolve()
    manifest_path = args.output / "manifest.json"
    if args.verify:
        manifest = json.loads(manifest_path.read_text())
        for relative, digest in manifest["outputs"].items():
            assert sha256(args.output / relative) == digest, relative
        print(json.dumps({"verified": True, "manifest": str(manifest_path)}, sort_keys=True))
        return

    args.output.mkdir(parents=True, exist_ok=True)
    make_logger(str(args.output), log_file="evaluation.log")
    set_seed(73034)
    asset_manifest = json.loads((args.assets / "manifest.json").read_text())
    rows = []
    runtime = {}
    for candidate, config_record in sorted(asset_manifest["configs"].items()):
        config_path = Path(config_record["path"])
        cfg = load_config(str(config_path))
        cfg["device"] = torch.device(args.device)
        cfg["training"]["batch_size"] = args.batch_size
        model = load_model(cfg, str(args.checkpoint.resolve()), cfg["device"])
        loader, _ = build_dataloader(cfg, "train", is_train=False, val_distributed=False)
        metadata = {
            item["name"]: {
                "source": str(item["video_file"]), "reference": str(item["label"]),
                "duration": int(item["base_end"]) - int(item["base_start"]),
                "duration_bucket": duration_bucket(item),
                "sample_id": (
                    f"{item['video_file']}|{int(item['base_start'])}:{int(item['base_end'])}"
                    f"|{item['label']}"
                ),
            }
            for item in loader.dataset.annotation
        }
        vocab = loader.dataset.vocab
        torch.cuda.reset_peak_memory_stats(cfg["device"])
        forward_seconds = 0.0
        forward_batches = 0
        forward_items = 0
        with torch.no_grad():
            for batch in loader:
                batch = move_to_device(batch, cfg["device"])
                torch.cuda.synchronize(cfg["device"])
                started = time.perf_counter()
                output = model(
                    is_train=False, labels=batch["labels"],
                    sgn_videos=batch["sgn_videos"], sgn_keypoints=batch["sgn_keypoints"]
                )
                torch.cuda.synchronize(cfg["device"])
                forward_seconds += time.perf_counter() - started
                forward_batches += 1
                forward_items += len(batch["names"])
                streams = [output[f"{name}_gloss_logits"] for name in ["rgb", "keypoint", "fuse"]]
                logits_by_head = {
                    "ensemble_main": sum(x.softmax(-1) for x in streams).clamp_min(1e-12).log(),
                    "weighted_main": sum(w * x.softmax(-1) for w, x in zip(WEIGHTS, streams)).clamp_min(1e-12).log(),
                }
                for head, logits in logits_by_head.items():
                    probs = logits.softmax(-1)
                    top = probs.topk(5, dim=-1)
                    for i, name in enumerate(batch["names"]):
                        reference_id = int(batch["labels"][i])
                        prediction_id = int(top.indices[i, 0])
                        rows.append({
                            **metadata[name], "candidate": candidate, "head": head,
                            "prediction": vocab[prediction_id], "correct": prediction_id == reference_id,
                            "top5_correct": reference_id in top.indices[i].tolist(),
                            "correct_probability": float(probs[i, reference_id]),
                        })
        frames = int(cfg["data"]["num_output_frames"])
        runtime[candidate] = {
            "input_frames": frames,
            "nominal_centered_lookahead_frames": frames // 2,
            "batch_size": args.batch_size,
            "items": forward_items,
            "batches": forward_batches,
            "model_forward_seconds": forward_seconds,
            "items_per_second": forward_items / forward_seconds,
            "mean_batch_milliseconds": 1000 * forward_seconds / forward_batches,
            "peak_allocated_bytes": int(torch.cuda.max_memory_allocated(cfg["device"])),
            "peak_reserved_bytes": int(torch.cuda.max_memory_reserved(cfg["device"])),
            "timing_scope": "synchronized model forward only; data loading and host-to-device transfer excluded",
        }
        del model
        torch.cuda.empty_cache()

    grouped = defaultdict(list)
    for row in rows:
        grouped[(row["candidate"], row["head"])].append(row)
    summary = {"checkpoint": str(args.checkpoint.resolve()), "runtime": runtime, "results": {}}
    for (candidate, head), values in sorted(grouped.items()):
        buckets = defaultdict(list)
        for row in values:
            buckets[row["duration_bucket"]].append(row)
        summary["results"][f"{candidate}::{head}"] = {
            "overall": aggregate(values),
            "by_duration_bucket": {key: aggregate(value) for key, value in sorted(buckets.items())},
        }
    summary_path = args.output / "summary.json"
    rows_path = args.output / "per_example.jsonl.gz"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    with gzip.open(rows_path, "wt", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    outputs = [summary_path.name, rows_path.name]
    manifest = {
        "schema_version": 1,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "frozen checkpoint, source-disjoint train-only sensitivity; no dev/test",
        "inputs": {
            "assets": {"path": str((args.assets / "manifest.json").resolve()), "sha256": sha256(args.assets / "manifest.json")},
            "checkpoint": {"path": str(args.checkpoint.resolve()), "sha256": sha256(args.checkpoint)},
        },
        "outputs": {name: sha256(args.output / name) for name in outputs},
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"summary": str(summary_path), "rows": len(rows)}, sort_keys=True))


if __name__ == "__main__":
    main()
