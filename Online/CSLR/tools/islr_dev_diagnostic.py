#!/usr/bin/env python3

import argparse
import csv
import json
import os
import pickle
import sys
from collections import Counter, defaultdict

import torch


CSLR_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if CSLR_ROOT not in sys.path:
    sys.path.insert(0, CSLR_ROOT)

from dataset.Dataloader import build_dataloader
from dataset.Dataset import build_dataset
from modelling.model import build_model
from utils.misc import load_config, make_logger, move_to_device, neq_load_customized, set_seed


DEFAULT_CONFIG = os.path.join(
    CSLR_ROOT, "configs", "csl-daily-top-800_ISLR_full_stable.yaml"
)
DEFAULT_CKPT = os.path.join(
    CSLR_ROOT, "results", "csl-daily-top-800_ISLR_full_stable", "ckpts", "best.ckpt"
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run detailed isolated dev diagnostics for an ISLR checkpoint."
    )
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--ckpt", default=DEFAULT_CKPT)
    parser.add_argument("--split", default="dev", choices=["dev", "test", "train"])
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--max-batches", type=int, default=None)
    parser.add_argument("--topk", type=int, default=10)
    parser.add_argument("--min-class-count", type=int, default=5)
    parser.add_argument("--weighted-fusion", action="store_true")
    parser.add_argument("--rgb-weight", type=float, default=0.10)
    parser.add_argument("--keypoint-weight", type=float, default=0.70)
    parser.add_argument("--fuse-weight", type=float, default=0.20)
    parser.add_argument("--weight-grid-search", action="store_true")
    parser.add_argument(
        "--output-dir",
        default=os.path.join(
            CSLR_ROOT,
            "results",
            "csl-daily-top-800_ISLR_full_stable",
            "diagnostics",
            "dev_best",
        ),
    )
    return parser.parse_args()


def validate_weights(rgb_weight, keypoint_weight, fuse_weight):
    weights = [rgb_weight, keypoint_weight, fuse_weight]
    if any(w < 0 for w in weights):
        raise ValueError(f"Fusion weights must be non-negative: {weights}")
    total = sum(weights)
    if total <= 0:
        raise ValueError("At least one fusion weight must be positive")
    return [w / total for w in weights]


def default_weight_grid():
    rgb_values = [0.0, 0.05, 0.10, 0.15, 0.20]
    keypoint_values = [0.50, 0.60, 0.70, 0.80, 0.90]
    grid = []
    for rgb_w in rgb_values:
        for keypoint_w in keypoint_values:
            fuse_w = round(1.0 - rgb_w - keypoint_w, 10)
            if fuse_w >= 0:
                grid.append((rgb_w, keypoint_w, fuse_w))
    return grid


def strip_module_prefix(state_dict):
    if not state_dict:
        return state_dict
    if all(k.startswith("module.") for k in state_dict.keys()):
        return {k[len("module.") :]: v for k, v in state_dict.items()}
    return state_dict


def load_model(cfg, ckpt_path, device):
    dataset = build_dataset(cfg["data"], "train", task=cfg["task"])
    vocab = dataset.vocab
    cls_num = len(vocab)

    word_emb_tab = None
    if dataset.word_emb_tab is not None:
        word_emb_tab = torch.stack(
            [torch.from_numpy(dataset.word_emb_tab[w]) for w in vocab], dim=0
        ).float().to(device)
    del dataset

    model = build_model(cfg, cls_num, word_emb_tab=word_emb_tab)
    state = torch.load(ckpt_path, map_location="cpu")
    state_dict = strip_module_prefix(state.get("model_state", state))
    try:
        model.load_state_dict(state_dict)
    except RuntimeError:
        # Fall back to the project's tolerant loader for older checkpoints.
        neq_load_customized(model, state_dict, verbose=True)
    model.to(device)
    model.eval()
    return model


def empty_stream_stats():
    return {
        "n": 0,
        "top1": 0,
        "top5": 0,
        "top10": 0,
        "blank_n": 0,
        "blank_as_blank": 0,
        "blank_as_gloss": 0,
        "nonblank_n": 0,
        "nonblank_as_nonblank": 0,
        "nonblank_as_blank": 0,
        "nonblank_top1": 0,
        "nonblank_top5": 0,
        "nonblank_top10": 0,
        "signness_correct": 0,
        "pred_blank": 0,
        "pred_nonblank": 0,
        "entropy_sum": 0.0,
        "max_prob_sum": 0.0,
    }


def pct(num, den):
    return None if den == 0 else 100.0 * float(num) / float(den)


def summarize_stream(raw, class_totals, class_correct, pred_counter, confusion, vocab, min_class_count):
    out = {
        "num_examples": raw["n"],
        "top1_acc": pct(raw["top1"], raw["n"]),
        "top5_acc": pct(raw["top5"], raw["n"]),
        "top10_acc": pct(raw["top10"], raw["n"]),
        "signness_acc": pct(raw["signness_correct"], raw["n"]),
        "blank_recall": pct(raw["blank_as_blank"], raw["blank_n"]),
        "blank_to_gloss_rate": pct(raw["blank_as_gloss"], raw["blank_n"]),
        "nonblank_sign_recall": pct(raw["nonblank_as_nonblank"], raw["nonblank_n"]),
        "gloss_to_blank_rate": pct(raw["nonblank_as_blank"], raw["nonblank_n"]),
        "nonblank_top1_acc": pct(raw["nonblank_top1"], raw["nonblank_n"]),
        "nonblank_top5_acc": pct(raw["nonblank_top5"], raw["nonblank_n"]),
        "nonblank_top10_acc": pct(raw["nonblank_top10"], raw["nonblank_n"]),
        "pred_blank_rate": pct(raw["pred_blank"], raw["n"]),
        "pred_nonblank_rate": pct(raw["pred_nonblank"], raw["n"]),
        "avg_entropy": None if raw["n"] == 0 else raw["entropy_sum"] / raw["n"],
        "avg_max_prob": None if raw["n"] == 0 else raw["max_prob_sum"] / raw["n"],
    }

    supported_classes = [
        idx for idx, count in class_totals.items() if count >= min_class_count and vocab[idx] != "<blank>"
    ]
    class_acc = []
    for idx in supported_classes:
        class_acc.append(
            {
                "class_id": idx,
                "gloss": vocab[idx],
                "count": class_totals[idx],
                "top1_acc": pct(class_correct[idx], class_totals[idx]),
            }
        )
    out["worst_classes"] = sorted(class_acc, key=lambda x: x["top1_acc"])[:20]
    out["best_classes"] = sorted(class_acc, key=lambda x: x["top1_acc"], reverse=True)[:20]
    out["top_predicted"] = [
        {"class_id": idx, "gloss": vocab[idx], "count": count}
        for idx, count in pred_counter.most_common(20)
    ]
    out["top_confusions"] = [
        {
            "ref_id": ref,
            "ref": vocab[ref],
            "pred_id": pred,
            "pred": vocab[pred],
            "count": count,
        }
        for (ref, pred), count in confusion.most_common(30)
    ]
    return out


def update_stats_for_logits(
    logits,
    labels,
    names,
    stream_key,
    stats,
    class_totals,
    class_correct,
    pred_counter,
    confusion,
    examples,
    vocab,
    blank_id,
    topk_size,
):
    prob = logits.softmax(dim=-1)
    max_prob, pred = prob.max(dim=-1)
    entropy = -(prob.clamp_min(1e-12) * prob.clamp_min(1e-12).log()).sum(dim=-1)
    topk = torch.topk(logits, k=min(topk_size, logits.shape[-1]), dim=-1).indices.cpu()
    pred_cpu = pred.detach().cpu()
    max_prob_cpu = max_prob.detach().cpu()
    entropy_cpu = entropy.detach().cpu()

    for i, ref in enumerate(labels.tolist()):
        pred_i = int(pred_cpu[i])
        top = [int(x) for x in topk[i].tolist()]
        ref_is_blank = ref == blank_id
        pred_is_blank = pred_i == blank_id
        s = stats[stream_key]
        s["n"] += 1
        s["top1"] += int(pred_i == ref)
        s["top5"] += int(ref in top[:5])
        s["top10"] += int(ref in top[:10])
        s["signness_correct"] += int(ref_is_blank == pred_is_blank)
        s["pred_blank"] += int(pred_is_blank)
        s["pred_nonblank"] += int(not pred_is_blank)
        s["entropy_sum"] += float(entropy_cpu[i])
        s["max_prob_sum"] += float(max_prob_cpu[i])
        pred_counter[stream_key][pred_i] += 1
        class_totals[stream_key][ref] += 1

        if pred_i == ref:
            class_correct[stream_key][ref] += 1
        else:
            confusion[stream_key][(ref, pred_i)] += 1

        if ref_is_blank:
            s["blank_n"] += 1
            s["blank_as_blank"] += int(pred_is_blank)
            s["blank_as_gloss"] += int(not pred_is_blank)
        else:
            s["nonblank_n"] += 1
            s["nonblank_as_nonblank"] += int(not pred_is_blank)
            s["nonblank_as_blank"] += int(pred_is_blank)
            s["nonblank_top1"] += int(pred_i == ref)
            s["nonblank_top5"] += int(ref in top[:5])
            s["nonblank_top10"] += int(ref in top[:10])

        if pred_i != ref and len(examples[stream_key]) < 200:
            examples[stream_key].append(
                {
                    "name": names[i],
                    "ref_id": ref,
                    "ref": vocab[ref],
                    "pred_id": pred_i,
                    "pred": vocab[pred_i],
                    "ref_is_blank": ref_is_blank,
                    "pred_is_blank": pred_is_blank,
                    "max_prob": float(max_prob_cpu[i]),
                    "entropy": float(entropy_cpu[i]),
                    "top10": [vocab[j] for j in top[:10]],
                }
            )


def weighted_logits_from_output(output, rgb_weight, keypoint_weight, fuse_weight):
    rgb_w, keypoint_w, fuse_w = validate_weights(rgb_weight, keypoint_weight, fuse_weight)
    rgb_prob = output["rgb_gloss_logits"].softmax(dim=-1)
    keypoint_prob = output["keypoint_gloss_logits"].softmax(dim=-1)
    fuse_prob = output["fuse_gloss_logits"].softmax(dim=-1)
    weighted_prob = rgb_w * rgb_prob + keypoint_w * keypoint_prob + fuse_w * fuse_prob
    return weighted_prob.clamp_min(1e-12).log()


def run_diagnostic(args):
    os.makedirs(args.output_dir, exist_ok=True)
    make_logger(args.output_dir, log_file=f"{args.split}_diagnostic.log")

    cfg = load_config(args.config)
    cfg["device"] = torch.device(args.device)
    cfg["rank"] = 0
    cfg["local_rank"] = 0
    cfg["world_size"] = 1
    cfg["training"]["num_workers"] = args.num_workers
    if args.batch_size is not None:
        cfg["training"]["batch_size"] = args.batch_size

    set_seed(cfg["training"].get("random_seed", 42))
    model = load_model(cfg, args.ckpt, cfg["device"])
    dataloader, _ = build_dataloader(
        cfg, args.split, task=cfg["task"], is_train=False, val_distributed=False
    )
    vocab = dataloader.dataset.vocab
    blank_id = vocab.index("<blank>")

    stream_keys = ["rgb", "keypoint", "fuse", "ensemble_last"]
    weight_grid = default_weight_grid() if args.weight_grid_search else []
    if args.weighted_fusion:
        stream_keys.append("weighted")
    for idx, _ in enumerate(weight_grid):
        stream_keys.append(f"weighted_grid_{idx:03d}")

    stats = {key: empty_stream_stats() for key in stream_keys}
    class_totals = {key: Counter() for key in stream_keys}
    class_correct = {key: Counter() for key in stream_keys}
    pred_counter = {key: Counter() for key in stream_keys}
    confusion = {key: Counter() for key in stream_keys}
    examples = {key: [] for key in stream_keys}

    with torch.no_grad():
        for step, batch in enumerate(dataloader):
            if args.max_batches is not None and step >= args.max_batches:
                break
            batch = move_to_device(batch, cfg["device"])
            output = model(
                is_train=False,
                labels=batch["labels"],
                sgn_videos=batch["sgn_videos"],
                sgn_keypoints=batch["sgn_keypoints"],
                epoch=None,
            )
            labels = batch["labels"].detach().cpu()
            names = batch["names"]

            base_logits = {
                "rgb": output.get("rgb_gloss_logits"),
                "keypoint": output.get("keypoint_gloss_logits"),
                "fuse": output.get("fuse_gloss_logits"),
                "ensemble_last": output.get("ensemble_last_gloss_logits"),
            }
            for stream_key, logits in base_logits.items():
                if logits is None:
                    continue
                update_stats_for_logits(
                    logits,
                    labels,
                    names,
                    stream_key,
                    stats,
                    class_totals,
                    class_correct,
                    pred_counter,
                    confusion,
                    examples,
                    vocab,
                    blank_id,
                    args.topk,
                )

            if args.weighted_fusion:
                weighted_logits = weighted_logits_from_output(
                    output, args.rgb_weight, args.keypoint_weight, args.fuse_weight
                )
                update_stats_for_logits(
                    weighted_logits,
                    labels,
                    names,
                    "weighted",
                    stats,
                    class_totals,
                    class_correct,
                    pred_counter,
                    confusion,
                    examples,
                    vocab,
                    blank_id,
                    args.topk,
                )

            for grid_idx, (rgb_w, keypoint_w, fuse_w) in enumerate(weight_grid):
                weighted_logits = weighted_logits_from_output(output, rgb_w, keypoint_w, fuse_w)
                update_stats_for_logits(
                    weighted_logits,
                    labels,
                    names,
                    f"weighted_grid_{grid_idx:03d}",
                    stats,
                    class_totals,
                    class_correct,
                    pred_counter,
                    confusion,
                    examples,
                    vocab,
                    blank_id,
                    args.topk,
                )

    summary = {
        "config": os.path.abspath(args.config),
        "ckpt": os.path.abspath(args.ckpt),
        "split": args.split,
        "device": str(cfg["device"]),
        "vocab_size": len(vocab),
        "blank_id": blank_id,
        "num_examples_seen": next(iter(stats.values()))["n"],
        "weighted_fusion": {
            "enabled": bool(args.weighted_fusion),
            "rgb_weight": validate_weights(
                args.rgb_weight, args.keypoint_weight, args.fuse_weight
            )[0],
            "keypoint_weight": validate_weights(
                args.rgb_weight, args.keypoint_weight, args.fuse_weight
            )[1],
            "fuse_weight": validate_weights(
                args.rgb_weight, args.keypoint_weight, args.fuse_weight
            )[2],
        },
        "streams": {},
    }
    grid_rows = []
    for key in stream_keys:
        if stats[key]["n"] == 0:
            continue
        stream_summary = summarize_stream(
            stats[key],
            class_totals[key],
            class_correct[key],
            pred_counter[key],
            confusion[key],
            vocab,
            args.min_class_count,
        )
        summary["streams"][key] = stream_summary
        if key.startswith("weighted_grid_"):
            grid_idx = int(key.rsplit("_", 1)[-1])
            rgb_w, keypoint_w, fuse_w = weight_grid[grid_idx]
            grid_rows.append(
                {
                    "stream": key,
                    "rgb_weight": rgb_w,
                    "keypoint_weight": keypoint_w,
                    "fuse_weight": fuse_w,
                    **{
                        metric: stream_summary.get(metric)
                        for metric in [
                            "nonblank_top1_acc",
                            "signness_acc",
                            "top1_acc",
                            "gloss_to_blank_rate",
                            "blank_to_gloss_rate",
                            "pred_blank_rate",
                            "nonblank_top5_acc",
                            "nonblank_top10_acc",
                        ]
                    },
                }
            )

    summary_path = os.path.join(args.output_dir, f"{args.split}_diagnostic_summary.json")
    examples_path = os.path.join(args.output_dir, f"{args.split}_diagnostic_examples.pkl")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    with open(examples_path, "wb") as f:
        pickle.dump(examples, f, protocol=pickle.HIGHEST_PROTOCOL)

    csv_path = os.path.join(args.output_dir, f"{args.split}_diagnostic_streams.csv")
    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        fieldnames = [
            "stream",
            "top1_acc",
            "top5_acc",
            "top10_acc",
            "signness_acc",
            "blank_recall",
            "blank_to_gloss_rate",
            "nonblank_sign_recall",
            "gloss_to_blank_rate",
            "nonblank_top1_acc",
            "nonblank_top5_acc",
            "nonblank_top10_acc",
            "pred_blank_rate",
            "avg_max_prob",
            "avg_entropy",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for stream, data in summary["streams"].items():
            writer.writerow({"stream": stream, **{k: data.get(k) for k in fieldnames if k != "stream"}})

    grid_path = None
    if grid_rows:
        grid_rows = sorted(
            grid_rows,
            key=lambda row: (
                -(row["nonblank_top1_acc"] or 0),
                -(row["signness_acc"] or 0),
                -(row["top1_acc"] or 0),
                row["gloss_to_blank_rate"] if row["gloss_to_blank_rate"] is not None else 1e9,
            ),
        )
        grid_path = os.path.join(args.output_dir, f"{args.split}_weight_grid.csv")
        with open(grid_path, "w", encoding="utf-8", newline="") as f:
            fieldnames = [
                "rank",
                "stream",
                "rgb_weight",
                "keypoint_weight",
                "fuse_weight",
                "nonblank_top1_acc",
                "signness_acc",
                "top1_acc",
                "gloss_to_blank_rate",
                "blank_to_gloss_rate",
                "pred_blank_rate",
                "nonblank_top5_acc",
                "nonblank_top10_acc",
            ]
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for rank, row in enumerate(grid_rows, start=1):
                writer.writerow({"rank": rank, **row})

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"\nSaved summary: {summary_path}")
    print(f"Saved stream CSV: {csv_path}")
    if grid_path:
        print(f"Saved weight grid CSV: {grid_path}")
    print(f"Saved mistake examples: {examples_path}")


def main():
    args = parse_args()
    run_diagnostic(args)


if __name__ == "__main__":
    main()
