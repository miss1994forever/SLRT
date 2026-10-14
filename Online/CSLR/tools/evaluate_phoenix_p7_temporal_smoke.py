#!/usr/bin/env python3
"""Evaluate P7 short-trained recognizers on source-disjoint train calibration data."""

from __future__ import annotations

import argparse
import copy
import gzip
import hashlib
import json
import pickle
import sys
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


ASSETS = CSLR_ROOT / "results/phoenix-2014t_ISLR/p7_temporal_debug_smoke_assets_v1"
OUTPUT = CSLR_ROOT / "results/phoenix-2014t_ISLR/p7_temporal_smoke_eval_v1"
CANDIDATES = ["A_current_fixed16", "B_aligned_fixed16", "C_inside_even16"]
CONFIGS = {
    name: CSLR_ROOT / "configs" / f"p7_temporal_debug_smoke_{name}.yaml"
    for name in CANDIDATES
}
CHECKPOINTS = {
    name: CSLR_ROOT / "results/phoenix-2014t_ISLR/p7_temporal_smoke_v1" / name / "ckpts/best.ckpt"
    for name in CANDIDATES
}
WEIGHTS = (0.10, 0.70, 0.20)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def bucket(item):
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


def build_inputs(calibration, output: Path):
    variants = {"context_center16": [], "base_pad16": [], "inside_even16": []}
    metadata = {}
    for idx, source in enumerate(calibration):
        sample_id = f"c{idx:05d}"
        metadata[sample_id] = {
            "sample_id": sample_id,
            "source": str(source["video_file"]),
            "label": str(source["label"]),
            "duration": int(source["base_end"]) - int(source["base_start"]),
            "duration_bucket": bucket(source),
        }
        context = copy.deepcopy(source)
        context["name"] = f"p7eval_{sample_id}_{context['name']}"
        variants["context_center16"].append(context)
        for variant in ["base_pad16", "inside_even16"]:
            item = copy.deepcopy(source)
            item["start"] = int(item["base_start"])
            item["end"] = int(item["base_end"])
            item["seq_len"] = item["end"] - item["start"]
            item["aug"] = 0
            item["name"] = f"p7eval_{sample_id}_{item['label']}_{item['video_file']}_[{item['start']}:{item['end']}]"
            variants[variant].append(item)
    paths = {}
    for name, rows in variants.items():
        path = output / f"{name}.train.pkl"
        with path.open("wb") as handle:
            pickle.dump(rows, handle, protocol=pickle.HIGHEST_PROTOCOL)
        paths[name] = path
    return paths, metadata


def input_config(config_path: Path, annotation: Path, variant: str, batch_size: int, workers: int):
    cfg = load_config(str(config_path))
    cfg["data"]["train"] = str(annotation.resolve())
    cfg["training"]["batch_size"] = batch_size
    cfg["training"]["num_workers"] = workers
    test_mode = "even" if variant == "inside_even16" else "pad"
    cfg["data"]["transform_cfg"]["index_setting"] = ["consecutive", "pad", "central", test_mode]
    return cfg


def heads(output):
    streams = [output[f"{name}_gloss_logits"] for name in ["rgb", "keypoint", "fuse"]]
    equal = sum(x.softmax(-1) for x in streams).clamp_min(1e-12).log()
    weighted = sum(w * x.softmax(-1) for w, x in zip(WEIGHTS, streams)).clamp_min(1e-12).log()
    result = {"ensemble_main": equal, "weighted_main": weighted}
    foreground = []
    for name in ["rgb", "keypoint", "fuse"]:
        values = output.get(f"{name}_split_logits")
        if not values:
            foreground = []
            break
        foreground.append(values[0])
    if foreground:
        result["ensemble_foreground"] = sum(x.softmax(-1) for x in foreground).clamp_min(1e-12).log()
        result["weighted_foreground"] = sum(
            w * x.softmax(-1) for w, x in zip(WEIGHTS, foreground)
        ).clamp_min(1e-12).log()
    return result


def aggregate(rows):
    if not rows:
        return {"n": 0}
    return {
        "n": len(rows),
        "top1": sum(x["correct"] for x in rows) / len(rows),
        "top5": sum(x["top5_correct"] for x in rows) / len(rows),
        "blank_prediction_rate": sum(x["prediction"] == "<blank>" for x in rows) / len(rows),
        "mean_correct_probability": sum(x["correct_probability"] for x in rows) / len(rows),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--calibration", type=Path, default=ASSETS / "calibration_context16.pkl")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--model", choices=CANDIDATES)
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

    if args.model is None:
        raise SystemExit("--model is required; run one model per output directory")
    args.output.mkdir(parents=True, exist_ok=True)
    make_logger(str(args.output), log_file="evaluation.log")
    set_seed(73032)
    with args.calibration.open("rb") as handle:
        calibration = pickle.load(handle)
    paths, metadata = build_inputs(calibration, args.output)
    config_path, checkpoint = CONFIGS[args.model], CHECKPOINTS[args.model]
    rows = []
    for variant in ["context_center16", "base_pad16", "inside_even16"]:
        cfg = input_config(config_path, paths[variant], variant, args.batch_size, args.num_workers)
        cfg["device"] = torch.device(args.device)
        model = load_model(cfg, str(checkpoint.resolve()), cfg["device"])
        loader, _ = build_dataloader(cfg, "train", is_train=False, val_distributed=False)
        vocab = loader.dataset.vocab
        with torch.no_grad():
            for batch in loader:
                batch = move_to_device(batch, cfg["device"])
                output = model(
                    is_train=False, labels=batch["labels"],
                    sgn_videos=batch["sgn_videos"], sgn_keypoints=batch["sgn_keypoints"],
                    epoch=None, temp_idx=batch["temp_idx"],
                )
                for head, logits in heads(output).items():
                    probs = logits.softmax(-1)
                    top = probs.topk(5, dim=-1)
                    for i, name in enumerate(batch["names"]):
                        sample_id = name.split("_", 2)[1]
                        ref = int(batch["labels"][i])
                        pred = int(top.indices[i, 0])
                        rows.append({
                            **metadata[sample_id], "model": args.model, "input": variant, "head": head,
                            "reference": vocab[ref], "prediction": vocab[pred], "correct": pred == ref,
                            "top5_correct": ref in top.indices[i].tolist(),
                            "correct_probability": float(probs[i, ref]),
                        })
        del model
        torch.cuda.empty_cache()

    grouped = defaultdict(list)
    for row in rows:
        grouped[(row["input"], row["head"])].append(row)
    summary = {"model": args.model, "calibration_items": len(calibration), "results": {}}
    for (variant, head), values in sorted(grouped.items()):
        by_bucket = defaultdict(list)
        for row in values:
            by_bucket[row["duration_bucket"]].append(row)
        summary["results"][f"{variant}::{head}"] = {
            "overall": aggregate(values),
            "by_duration_bucket": {name: aggregate(group) for name, group in sorted(by_bucket.items())},
        }
    summary_path = args.output / "summary.json"
    rows_path = args.output / "per_example.jsonl.gz"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    with gzip.open(rows_path, "wt", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    output_names = [summary_path.name, rows_path.name] + [path.name for path in paths.values()]
    manifest = {
        "schema_version": 1,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "source-disjoint train-only calibration; official dev/test forbidden",
        "model": args.model,
        "inputs": {
            "calibration": {"path": str(args.calibration.resolve()), "sha256": sha256(args.calibration)},
            "config": {"path": str(config_path.resolve()), "sha256": sha256(config_path)},
            "checkpoint": {"path": str(checkpoint.resolve()), "sha256": sha256(checkpoint)},
        },
        "outputs": {name: sha256(args.output / name) for name in output_names},
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"model": args.model, "summary": str(summary_path)}, sort_keys=True))


if __name__ == "__main__":
    main()
