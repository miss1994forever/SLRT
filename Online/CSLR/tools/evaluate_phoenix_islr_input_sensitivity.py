#!/usr/bin/env python3
"""Frozen-checkpoint, train-only temporal input sensitivity pilot.

This tool never opens dev or test annotations/keypoints.  It selects a fixed,
duration-stratified set of nonblank training glosses and evaluates paired input
constructions with the frozen ISLR checkpoint:

* base_pad16: isolated proxy frames with centered endpoint padding;
* inside_even16: the same proxy frames evenly replicated to 16;
* context_center16: the existing continuous-video 16-frame augmentation nearest
  the proxy midpoint.

For base/context inputs it also evaluates the checkpoint's ``split_nonblk``
foreground-pooled logits.  These are sensitivity diagnostics, not held-out
generalization estimates, because the recognizer saw the training videos.
"""

from __future__ import annotations

import argparse
import copy
import gzip
import hashlib
import json
import os
import pickle
import random
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import torch


CSLR_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = CSLR_ROOT.parents[1]
if str(CSLR_ROOT) not in sys.path:
    sys.path.insert(0, str(CSLR_ROOT))

from dataset.Dataloader import build_dataloader  # noqa: E402
from tools.islr_dev_diagnostic import load_model  # noqa: E402
from utils.misc import load_config, make_logger, move_to_device, set_seed  # noqa: E402


DEFAULT_CONFIG = CSLR_ROOT / "configs/phoenix-2014t_ISLR.yaml"
DEFAULT_CKPT = CSLR_ROOT / "results/phoenix-2014t_ISLR/ckpts/best.ckpt"
DEFAULT_TRAIN = REPO_ROOT / "data/phoenix_2014t/phoenix_iso_center_label_bag2items.train"
DEFAULT_TRAIN_KEYPOINTS = REPO_ROOT / "data/phoenix_2014t/keypoints_hrnet_dark_coco_wholebody_iso.train.finite.pkl"
DEFAULT_OUTPUT = CSLR_ROOT / "results/phoenix-2014t_ISLR/p7_input_sensitivity_train_v1"
PRIMARY_WEIGHTS = (0.10, 0.70, 0.20)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def duration_bucket(duration: int) -> str:
    if duration <= 6:
        return "1-6"
    if duration <= 9:
        return "7-9"
    if duration <= 15:
        return "10-15"
    return "16+"


def stable_key(seed: int, item: Dict[str, Any]) -> str:
    payload = f"{seed}|{item['video_file']}|{item['start']}|{item['end']}|{item['label']}"
    return hashlib.sha256(payload.encode()).hexdigest()


def select_examples(
    bags: Dict[str, List[Dict[str, Any]]], per_bucket: int, seed: int
) -> List[Tuple[str, Dict[str, Any], List[Dict[str, Any]]]]:
    grouped: Dict[str, List[Tuple[Dict[str, Any], List[Dict[str, Any]]]]] = defaultdict(list)
    for items in bags.values():
        bases = [x for x in items if int(x.get("aug", 0)) == 0]
        if len(bases) != 1:
            raise AssertionError("one base item required")
        base = bases[0]
        if str(base["label"]) == "<blank>":
            continue
        duration = int(base["base_end"]) - int(base["base_start"])
        grouped[duration_bucket(duration)].append((base, items))

    selected: List[Tuple[str, Dict[str, Any], List[Dict[str, Any]]]] = []
    for bucket_name in ["1-6", "7-9", "10-15", "16+"]:
        rows = sorted(grouped[bucket_name], key=lambda x: stable_key(seed, x[0]))
        # Prefer distinct continuous source videos, then fill any remainder.
        chosen, deferred, seen_sources = [], [], set()
        for row in rows:
            source = str(row[0]["video_file"])
            if source not in seen_sources and len(chosen) < per_bucket:
                chosen.append(row)
                seen_sources.add(source)
            else:
                deferred.append(row)
        if len(chosen) < per_bucket:
            chosen.extend(deferred[: per_bucket - len(chosen)])
        if len(chosen) != per_bucket:
            raise RuntimeError(f"not enough examples for {bucket_name}: {len(chosen)}")
        for base, items in chosen:
            sid = f"s{len(selected):05d}"
            selected.append((sid, base, items))
    return selected


def closest_context(base: Dict[str, Any], items: List[Dict[str, Any]]) -> Dict[str, Any]:
    augmented = [x for x in items if int(x.get("aug", 0)) == 1]
    midpoint = (int(base["base_start"]) + int(base["base_end"]) - 1) / 2
    return min(
        augmented,
        key=lambda x: (abs(((int(x["start"]) + int(x["end"]) - 1) / 2) - midpoint), int(x["start"])),
    )


def build_candidate_files(
    selected: List[Tuple[str, Dict[str, Any], List[Dict[str, Any]]]], output: Path
) -> Tuple[Dict[str, Path], Dict[str, Dict[str, Any]]]:
    annotations: Dict[str, List[Dict[str, Any]]] = {
        "base_pad16": [],
        "inside_even16": [],
        "context_center16": [],
    }
    metadata: Dict[str, Dict[str, Any]] = {}
    for sid, base, items in selected:
        duration = int(base["base_end"]) - int(base["base_start"])
        metadata[sid] = {
            "sample_id": sid,
            "source": str(base["video_file"]),
            "label": str(base["label"]),
            "duration": duration,
            "duration_bucket": duration_bucket(duration),
            "base_start": int(base["base_start"]),
            "base_end": int(base["base_end"]),
        }
        for candidate in ["base_pad16", "inside_even16"]:
            item = copy.deepcopy(base)
            item["name"] = f"p7_{sid}_{item['name']}"
            annotations[candidate].append(item)
        context = copy.deepcopy(closest_context(base, items))
        context["name"] = f"p7_{sid}_{context['name']}"
        annotations["context_center16"].append(context)

    paths = {}
    for candidate, rows in annotations.items():
        path = output / f"{candidate}.train.pkl"
        with path.open("wb") as f:
            pickle.dump(rows, f, protocol=pickle.HIGHEST_PROTOCOL)
        paths[candidate] = path
    (output / "selection.json").write_text(
        json.dumps({"samples": list(metadata.values())}, indent=2, sort_keys=True) + "\n"
    )
    return paths, metadata


def candidate_cfg(base_cfg: Dict[str, Any], annotation: Path, candidate: str, batch_size: int, workers: int) -> Dict[str, Any]:
    cfg = copy.deepcopy(base_cfg)
    cfg["data"]["train"] = str(annotation.resolve())
    cfg["training"]["batch_size"] = batch_size
    cfg["training"]["num_workers"] = workers
    if candidate == "inside_even16":
        cfg["data"]["transform_cfg"]["index_setting"] = ["consecutive", "pad", "central", "even"]
    else:
        cfg["data"]["transform_cfg"]["index_setting"] = ["consecutive", "pad", "central", "pad"]
    return cfg


def ensemble_logits(output: Dict[str, Any], suffix: str = "gloss_logits") -> torch.Tensor:
    logits = [output[f"{stream}_{suffix}"] for stream in ["rgb", "keypoint", "fuse"]]
    return sum(x.softmax(dim=-1) for x in logits).clamp_min(1e-12).log()


def foreground_ensemble_logits(output: Dict[str, Any]) -> Optional[torch.Tensor]:
    logits = []
    for stream in ["rgb", "keypoint", "fuse"]:
        values = output[f"{stream}_split_logits"]
        if not values:
            return None
        logits.append(values[0])
    return sum(x.softmax(dim=-1) for x in logits).clamp_min(1e-12).log()


def weighted_logits(output: Dict[str, Any], foreground: bool = False) -> Optional[torch.Tensor]:
    logits = []
    for stream in ["rgb", "keypoint", "fuse"]:
        if foreground:
            values = output[f"{stream}_split_logits"]
            if not values:
                return None
            logits.append(values[0])
        else:
            logits.append(output[f"{stream}_gloss_logits"])
    probs = sum(w * x.softmax(dim=-1) for w, x in zip(PRIMARY_WEIGHTS, logits))
    return probs.clamp_min(1e-12).log()


def record_predictions(
    logits: torch.Tensor,
    labels: torch.Tensor,
    names: List[str],
    vocab: List[str],
    candidate: str,
    head: str,
    metadata: Dict[str, Dict[str, Any]],
) -> List[Dict[str, Any]]:
    probs = logits.softmax(dim=-1)
    top5 = probs.topk(k=5, dim=-1)
    rows = []
    for i, name in enumerate(names):
        sid = name.split("_", 2)[1]
        ref = int(labels[i])
        pred = int(top5.indices[i, 0])
        wrong = probs[i].clone()
        wrong[ref] = -1
        row = {
            **metadata[sid],
            "candidate": candidate,
            "head": head,
            "ref_id": ref,
            "pred_id": pred,
            "pred": vocab[pred],
            "top1_correct": pred == ref,
            "top5_correct": ref in [int(x) for x in top5.indices[i]],
            "correct_probability": float(probs[i, ref].cpu()),
            "correct_margin": float((probs[i, ref] - wrong.max()).cpu()),
            "blank_prediction": pred == 0,
        }
        rows.append(row)
    return rows


def aggregate(rows: Iterable[Dict[str, Any]]) -> Dict[str, Any]:
    rows = list(rows)
    if not rows:
        return {"n": 0}
    return {
        "n": len(rows),
        "top1": sum(r["top1_correct"] for r in rows) / len(rows),
        "top5": sum(r["top5_correct"] for r in rows) / len(rows),
        "blank_rate": sum(r["blank_prediction"] for r in rows) / len(rows),
        "mean_correct_probability": sum(r["correct_probability"] for r in rows) / len(rows),
        "mean_correct_margin": sum(r["correct_margin"] for r in rows) / len(rows),
    }


def summarize(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    grouped: Dict[Tuple[str, str], List[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(row["candidate"], row["head"])].append(row)
    out = {}
    for (candidate, head), values in sorted(grouped.items()):
        key = f"{candidate}::{head}"
        by_bucket = defaultdict(list)
        by_duration = defaultdict(list)
        for row in values:
            by_bucket[row["duration_bucket"]].append(row)
            by_duration[str(row["duration"])].append(row)
        out[key] = {
            "overall": aggregate(values),
            "by_duration_bucket": {k: aggregate(v) for k, v in sorted(by_bucket.items())},
            "by_duration": {k: aggregate(v) for k, v in sorted(by_duration.items(), key=lambda x: int(x[0]))},
        }
    return out


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    p.add_argument("--checkpoint", type=Path, default=DEFAULT_CKPT)
    p.add_argument("--train", type=Path, default=DEFAULT_TRAIN)
    p.add_argument("--train-keypoints", type=Path, default=DEFAULT_TRAIN_KEYPOINTS)
    p.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    p.add_argument("--per-bucket", type=int, default=512)
    p.add_argument("--seed", type=int, default=73021)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--verify", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    args.output = args.output.resolve()
    summary_path = args.output / "summary.json"
    manifest_path = args.output / "manifest.json"
    if args.verify:
        manifest = json.loads(manifest_path.read_text())
        for name, digest in manifest["outputs"].items():
            assert sha256(args.output / name) == digest, name
        for name, info in manifest["inputs"].items():
            assert sha256(Path(info["path"])) == info["sha256"], name
        summary = json.loads(summary_path.read_text())
        assert summary["scope"].startswith("train-only")
        assert summary["samples"] == summary["per_bucket"] * 4
        print(json.dumps({"verified": True, "manifest": str(manifest_path)}, sort_keys=True))
        return

    args.output.mkdir(parents=True, exist_ok=True)
    make_logger(str(args.output), log_file="input_sensitivity.log")
    with args.train.open("rb") as f:
        bags = pickle.load(f)
    selected = select_examples(bags, args.per_bucket, args.seed)
    candidate_paths, metadata = build_candidate_files(selected, args.output)

    original_cwd = Path.cwd()
    os.chdir(CSLR_ROOT)
    try:
        cfg = load_config(str(args.config.resolve()))
        cfg["device"] = torch.device(args.device)
        cfg["rank"] = 0
        cfg["local_rank"] = 0
        cfg["world_size"] = 1
        cfg["data"]["keypoint_file"] = str(args.train_keypoints.resolve())
        set_seed(args.seed)
        model = load_model(cfg, str(args.checkpoint.resolve()), cfg["device"])

        all_rows: List[Dict[str, Any]] = []
        vocab = None
        for candidate in ["base_pad16", "inside_even16", "context_center16"]:
            c_cfg = candidate_cfg(cfg, candidate_paths[candidate], candidate, args.batch_size, args.num_workers)
            dataloader, _ = build_dataloader(c_cfg, "train", task=c_cfg["task"], is_train=False, val_distributed=False)
            vocab = dataloader.dataset.vocab
            with torch.no_grad():
                for batch in dataloader:
                    batch = move_to_device(batch, cfg["device"])
                    output = model(
                        is_train=False,
                        labels=batch["labels"],
                        sgn_videos=batch["sgn_videos"],
                        sgn_keypoints=batch["sgn_keypoints"],
                        epoch=None,
                        temp_idx=batch["temp_idx"],
                    )
                    labels = batch["labels"].detach().cpu()
                    heads = {
                        "rgb_main": output["rgb_gloss_logits"],
                        "keypoint_main": output["keypoint_gloss_logits"],
                        "fuse_main": output["fuse_gloss_logits"],
                        "ensemble_main": ensemble_logits(output),
                        "weighted_main": weighted_logits(output),
                        "ensemble_foreground": foreground_ensemble_logits(output),
                        "weighted_foreground": weighted_logits(output, foreground=True),
                    }
                    for stream in ["rgb", "keypoint", "fuse"]:
                        values = output[f"{stream}_split_logits"]
                        if values:
                            heads[f"{stream}_foreground"] = values[0]
                    for head, logits in heads.items():
                        if logits is not None:
                            all_rows.extend(record_predictions(
                                logits, labels, batch["names"], vocab, candidate, head, metadata
                            ))
            del dataloader
    finally:
        os.chdir(original_cwd)

    with gzip.open(args.output / "per_example.jsonl.gz", "wt", encoding="utf-8") as f:
        for row in all_rows:
            f.write(json.dumps(row, sort_keys=True) + "\n")
    summary = {
        "schema_version": 1,
        "identity": "frozen ISLR train temporal-input sensitivity pilot",
        "scope": "train-only annotations and train-only keypoints; dev/test not opened",
        "warning": "recognizer saw these training videos; results measure paired input sensitivity, not held-out generalization",
        "samples": len(selected),
        "per_bucket": args.per_bucket,
        "seed": args.seed,
        "primary_weights_rgb_keypoint_fuse": list(PRIMARY_WEIGHTS),
        "metrics": summarize(all_rows),
    }
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")

    input_paths = {
        "config": args.config.resolve(),
        "checkpoint": args.checkpoint.resolve(),
        "train_annotations": args.train.resolve(),
        "train_keypoints": args.train_keypoints.resolve(),
        "evaluator": Path(__file__).resolve(),
        "video_loader": CSLR_ROOT / "dataset/VideoLoader.py",
        "dataloader": CSLR_ROOT / "dataset/Dataloader.py",
        "visual_head": CSLR_ROOT / "modelling/Visualhead.py",
    }
    output_names = ["summary.json", "selection.json", "per_example.jsonl.gz"] + [p.name for p in candidate_paths.values()]
    manifest = {
        "schema_version": 1,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "command": "evaluate_phoenix_islr_input_sensitivity.py",
        "inputs": {k: {"path": str(v), "sha256": sha256(v)} for k, v in input_paths.items()},
        "outputs": {name: sha256(args.output / name) for name in output_names},
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"summary": str(summary_path), "manifest": str(manifest_path)}, sort_keys=True))


if __name__ == "__main__":
    main()
