#!/usr/bin/env python3
"""Run zero-training reliability diagnostics on isolated dev only."""

import argparse
import csv
import json
import math
import os
import sys
from pathlib import Path

import numpy as np
import torch


CSLR_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CSLR_ROOT))

from dataset.Dataloader import build_dataloader
from tools.islr_dev_diagnostic import load_model
from utils.misc import load_config, make_logger, move_to_device, set_seed
from utils.reliability_analysis import binary_auc, quantile_bins, safe_correlation, summarize_signal


STREAMS = ("rgb", "keypoint", "fuse")
LOGIT_KEYS = {
    "rgb": "rgb_gloss_logits",
    "keypoint": "keypoint_gloss_logits",
    "fuse": "fuse_gloss_logits",
}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default=str(CSLR_ROOT / "configs" / "csl-daily-top-800_ISLR_full_stable.yaml"),
    )
    parser.add_argument(
        "--ckpt",
        default=str(CSLR_ROOT / "results" / "csl-daily-top-800_ISLR_full_stable" / "ckpts" / "best.ckpt"),
    )
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--max-batches", type=int)
    parser.add_argument("--keypoint-confidence-threshold", type=float, default=0.2)
    parser.add_argument(
        "--output-dir",
        default=str(
            CSLR_ROOT
            / "results"
            / "csl-daily-top-800_ISLR_full_stable"
            / "diagnostics"
            / "reliability_r1_dev"
        ),
    )
    return parser.parse_args()


def keypoint_quality(keypoints, threshold):
    """Return visibility, confidence, and normalized median motion per item."""
    if isinstance(keypoints, (list, tuple)):
        keypoints = keypoints[-1]
    confidence = keypoints[..., 2]
    visible = torch.isfinite(confidence) & (confidence >= threshold)
    visible_ratio = visible.float().mean(dim=(1, 2))
    finite_conf = torch.where(torch.isfinite(confidence), confidence, torch.zeros_like(confidence))
    mean_confidence = finite_conf.mean(dim=(1, 2))

    coords = keypoints[..., :2]
    valid_pair = visible[:, 1:] & visible[:, :-1]
    displacement = torch.linalg.norm(coords[:, 1:] - coords[:, :-1], dim=-1)
    extent = coords.amax(dim=(1, 2)) - coords.amin(dim=(1, 2))
    scale = torch.linalg.norm(extent, dim=-1).clamp_min(1e-6)
    normalized = displacement / scale[:, None, None]
    normalized = torch.where(valid_pair, normalized, torch.full_like(normalized, float("nan")))
    motion = torch.nanmedian(normalized.flatten(1), dim=1).values
    motion = torch.nan_to_num(motion, nan=0.0, posinf=0.0, neginf=0.0)
    return visible_ratio, mean_confidence, motion


def js_divergence(probabilities):
    stacked = torch.stack(probabilities, dim=0).clamp_min(1e-12)
    mean = stacked.mean(dim=0).clamp_min(1e-12)
    return (stacked * (stacked.log() - mean.log())).sum(dim=-1).mean(dim=0)


def analyze(records):
    summary = {"num_examples": len(records), "streams": {}}
    for stream in STREAMS:
        confidence = [row[f"{stream}_max_prob"] for row in records]
        correct = [row[f"{stream}_correct"] for row in records]
        entropy_confidence = [1.0 - row[f"{stream}_normalized_entropy"] for row in records]
        stream_summary = summarize_signal(confidence, correct)
        stream_summary["entropy_correctness_auc"] = binary_auc(entropy_confidence, correct)
        stream_summary["margin_correctness_auc"] = binary_auc(
            [row[f"{stream}_margin"] for row in records], correct
        )
        stream_summary["label_aware_blank_probability_correctness_auc"] = binary_auc(
            [
                row[f"{stream}_blank_prob"]
                if row["label_gloss"] == "<blank>"
                else 1.0 - row[f"{stream}_blank_prob"]
                for row in records
            ],
            correct,
        )
        nonblank = [row for row in records if row["label_gloss"] != "<blank>"]
        stream_summary["nonblank_negative_blank_probability_correctness_auc"] = binary_auc(
            [-row[f"{stream}_blank_prob"] for row in nonblank],
            [row[f"{stream}_correct"] for row in nonblank],
        )
        summary["streams"][stream] = stream_summary

    predictions = np.asarray([[row[f"{stream}_pred"] for stream in STREAMS] for row in records])
    correct = np.asarray([[row[f"{stream}_correct"] for stream in STREAMS] for row in records], dtype=bool)
    confidence = np.asarray([[row[f"{stream}_max_prob"] for stream in STREAMS] for row in records])
    entropy = np.asarray([[row[f"{stream}_normalized_entropy"] for stream in STREAMS] for row in records])
    all_agree = np.all(predictions == predictions[:, :1], axis=1)
    max_conf_choice = confidence.argmax(axis=1)
    min_entropy_choice = entropy.argmin(axis=1)
    row_index = np.arange(len(records))
    summary["cross_stream"] = {
        "all_agree_rate": float(all_agree.mean()),
        "all_agree_accuracy": float(correct[all_agree, 0].mean()) if all_agree.any() else None,
        "disagreement_rate": float((~all_agree).mean()),
        "oracle_any_stream_accuracy": float(correct.any(axis=1).mean()),
        "max_confidence_selector_accuracy": float(correct[row_index, max_conf_choice].mean()),
        "min_entropy_selector_accuracy": float(correct[row_index, min_entropy_choice].mean()),
        "max_confidence_selector_accuracy_on_disagreement": (
            float(correct[row_index[~all_agree], max_conf_choice[~all_agree]].mean())
            if (~all_agree).any()
            else None
        ),
        "mean_js_divergence": float(np.mean([row["stream_js_divergence"] for row in records])),
        "js_divergence_any_correct_auc": binary_auc(
            [-row["stream_js_divergence"] for row in records], correct.any(axis=1)
        ),
    }

    kp_correct = [row["keypoint_correct"] for row in records]
    for signal in ("keypoint_visible_ratio", "keypoint_mean_confidence", "keypoint_motion"):
        values = [row[signal] for row in records]
        summary.setdefault("keypoint_quality", {})[signal] = {
            "correctness_auc": binary_auc(values, kp_correct),
            "correctness_correlation": safe_correlation(values, kp_correct),
            "quartiles": quantile_bins(values, kp_correct),
        }
    return summary


def main():
    args = parse_args()
    if args.batch_size < 1 or args.num_workers < 0:
        raise ValueError("batch size must be positive and num workers non-negative")
    if not 0 <= args.keypoint_confidence_threshold <= 1:
        raise ValueError("keypoint confidence threshold must be in [0, 1]")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    make_logger(str(output_dir), log_file="reliability_r1_dev.log")
    cfg = load_config(args.config)
    cfg["device"] = torch.device(args.device)
    cfg.update({"rank": 0, "local_rank": 0, "world_size": 1})
    cfg["training"]["batch_size"] = args.batch_size
    cfg["training"]["num_workers"] = args.num_workers
    set_seed(cfg["training"].get("random_seed", 42))
    model = load_model(cfg, args.ckpt, cfg["device"])
    dataloader, _ = build_dataloader(
        cfg, "dev", task=cfg["task"], is_train=False, val_distributed=False
    )
    vocab = dataloader.dataset.vocab
    blank_id = vocab.index("<blank>")
    records = []

    with torch.no_grad():
        for step, batch in enumerate(dataloader):
            if args.max_batches is not None and step >= args.max_batches:
                break
            visible, kp_confidence, motion = keypoint_quality(
                batch["sgn_keypoints"], args.keypoint_confidence_threshold
            )
            names = list(batch["names"])
            labels = batch["labels"].clone()
            batch = move_to_device(batch, cfg["device"])
            output = model(
                is_train=False,
                labels=batch["labels"],
                sgn_videos=batch["sgn_videos"],
                sgn_keypoints=batch["sgn_keypoints"],
                epoch=None,
            )
            stream_data = {}
            for stream in STREAMS:
                probability = output[LOGIT_KEYS[stream]].softmax(dim=-1)
                top2_prob, top2_id = probability.topk(2, dim=-1)
                entropy = -(probability.clamp_min(1e-12) * probability.clamp_min(1e-12).log()).sum(dim=-1)
                stream_data[stream] = {
                    "probability": probability,
                    "pred": top2_id[:, 0],
                    "max_prob": top2_prob[:, 0],
                    "margin": top2_prob[:, 0] - top2_prob[:, 1],
                    "normalized_entropy": entropy / math.log(probability.shape[-1]),
                    "blank_prob": probability[:, blank_id],
                }
            divergence = js_divergence([stream_data[s]["probability"] for s in STREAMS])
            for index, name in enumerate(names):
                row = {
                    "name": name,
                    "label": int(labels[index]),
                    "label_gloss": vocab[int(labels[index])],
                    "keypoint_visible_ratio": float(visible[index]),
                    "keypoint_mean_confidence": float(kp_confidence[index]),
                    "keypoint_motion": float(motion[index]),
                    "stream_js_divergence": float(divergence[index]),
                }
                for stream in STREAMS:
                    data = stream_data[stream]
                    pred = int(data["pred"][index])
                    row.update(
                        {
                            f"{stream}_pred": pred,
                            f"{stream}_pred_gloss": vocab[pred],
                            f"{stream}_correct": int(pred == int(labels[index])),
                            f"{stream}_max_prob": float(data["max_prob"][index]),
                            f"{stream}_margin": float(data["margin"][index]),
                            f"{stream}_normalized_entropy": float(data["normalized_entropy"][index]),
                            f"{stream}_blank_prob": float(data["blank_prob"][index]),
                        }
                    )
                records.append(row)
            print(f"R1 dev batches: {step + 1}/{len(dataloader)} examples={len(records)}", flush=True)

    summary = {
        "protocol": {
            "split": "dev",
            "config": os.path.abspath(args.config),
            "checkpoint": os.path.abspath(args.ckpt),
            "batch_size": args.batch_size,
            "keypoint_confidence_threshold": args.keypoint_confidence_threshold,
            "max_batches": args.max_batches,
            "no_training": True,
        },
        **analyze(records),
    }
    with (output_dir / "reliability_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)
    if records:
        with (output_dir / "reliability_records.csv").open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(records[0]))
            writer.writeheader()
            writer.writerows(records)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
