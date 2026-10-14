#!/usr/bin/env python3
"""Audit Phoenix ISLR temporal construction without reading dev/test video data.

The audit is deliberately annotation-only.  It quantifies how a fixed 16-frame
input, bag sampling, surrounding-context augmentation, and the current
``temp_idx`` foreground coordinates interact on the training split.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import pickle
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List


REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_TRAIN = REPO_ROOT / "data/phoenix_2014t/phoenix_iso_center_label_bag2items.train"
DEFAULT_CONFIG = REPO_ROOT / "Online/CSLR/configs/phoenix-2014t_ISLR.yaml"
DEFAULT_CKPT = REPO_ROOT / "Online/CSLR/results/phoenix-2014t_ISLR/ckpts/best.ckpt"
DEFAULT_OUTPUT = REPO_ROOT / "Online/CSLR/results/phoenix-2014t_ISLR/p7_temporal_input_audit_v1"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def bucket(duration: int) -> str:
    if duration <= 6:
        return "1-6"
    if duration <= 9:
        return "7-9"
    if duration <= 15:
        return "10-15"
    return "16+"


def mean(values: Iterable[float]) -> float:
    values = list(values)
    return sum(values) / len(values) if values else 0.0


def weighted_mean(pairs: Iterable[tuple[float, float]]) -> float:
    pairs = list(pairs)
    den = sum(w for _, w in pairs)
    return sum(v * w for v, w in pairs) / den if den else 0.0


def padding_alignment(duration: int, input_frames: int = 16, ratio: int = 2) -> Dict[str, Any]:
    """Compare fixed temp_idx bins with random train padding positions.

    VisualHead upsamples the final S3D feature sequence to input_frames/ratio
    bins.  Bin centers are used only as a coordinate proxy; convolutional
    receptive fields are wider, so these are not exact feature attributions.
    """
    assert 0 < duration < input_frames
    bins = input_frames // ratio
    remain = input_frames - duration
    planned_start = remain // (2 * ratio)
    planned_end = planned_start + duration // ratio
    planned_start = max(0, min(planned_start, bins - 1))
    planned_end = max(planned_start + 1, min(planned_end, bins))
    planned_centers = [ratio * j + (ratio - 1) / 2 for j in range(planned_start, planned_end)]

    # np.random.randint(0, remain) in the current training loader.
    rows: List[Dict[str, float]] = []
    for pad_left in range(remain):
        lo, hi = pad_left, pad_left + duration
        actual = [ratio * j + (ratio - 1) / 2 for j in range(bins) if lo <= ratio * j + (ratio - 1) / 2 < hi]
        selected_valid = sum(lo <= c < hi for c in planned_centers)
        purity = selected_valid / len(planned_centers)
        recall = selected_valid / len(actual) if actual else 0.0
        rows.append({"pad_left": pad_left, "purity": purity, "recall": recall})
    return {
        "duration": duration,
        "possible_train_pad_left": list(range(remain)),
        "planned_temp_idx": [planned_start, planned_end],
        "proxy_mean_purity": mean(r["purity"] for r in rows),
        "proxy_mean_recall": mean(r["recall"] for r in rows),
        "probability_all_planned_bins_are_foreground": mean(r["purity"] == 1.0 for r in rows),
        "warning": "bin-center coordinate proxy; S3D temporal receptive fields are wider"
    }


def analyze(train_path: Path, input_frames: int, bag_size: int) -> Dict[str, Any]:
    with train_path.open("rb") as f:
        bags = pickle.load(f)

    bag_counts = Counter()
    duration_counts: Dict[str, Counter] = defaultdict(Counter)
    bag_size_counts: Dict[str, Counter] = defaultdict(Counter)
    expected_base_clips: Dict[str, Counter] = defaultdict(Counter)
    expected_aug_clips: Dict[str, Counter] = defaultdict(Counter)
    aug_fraction_pairs: Dict[str, Dict[str, List[tuple[float, float]]]] = defaultdict(lambda: defaultdict(list))
    edge_aug_expected = Counter()
    relation_violations = Counter()

    for items in bags.values():
        bases = [x for x in items if int(x.get("aug", 0)) == 0]
        if len(bases) != 1:
            raise AssertionError(f"expected one base item, found {len(bases)}")
        base = bases[0]
        split_kind = "blank" if str(base["label"]) == "<blank>" else "nonblank"
        duration = int(base["base_end"]) - int(base["base_start"])
        dur_bucket = bucket(duration)
        n = len(items)
        multiplicity = bag_size / n  # exact expectation under current sampling/duplication

        bag_counts[split_kind] += 1
        duration_counts[split_kind][duration] += 1
        bag_size_counts[split_kind][n] += 1
        expected_base_clips[split_kind][dur_bucket] += multiplicity
        expected_aug_clips[split_kind][dur_bucket] += multiplicity * (n - 1)
        if n != duration + 1:
            relation_violations[split_kind] += 1

        for item in items:
            if int(item.get("aug", 0)) == 0:
                continue
            start, end = int(item["start"]), int(item["end"])
            raw_len = end - start
            overlap = max(0, min(end, int(base["base_end"])) - max(start, int(base["base_start"])))
            raw_fraction = overlap / raw_len if raw_len else 0.0
            aug_fraction_pairs[split_kind][dur_bucket].append((raw_fraction, multiplicity))
            if raw_len < input_frames:
                edge_aug_expected[split_kind] += multiplicity

    padding_by_duration: Dict[str, Dict[str, Any]] = {}
    for duration in sorted(set(duration_counts["nonblank"])):
        if duration < input_frames:
            padding_by_duration[str(duration)] = padding_alignment(duration, input_frames=input_frames)

    expected_base_total = {k: sum(v.values()) for k, v in expected_base_clips.items()}
    expected_aug_total = {k: sum(v.values()) for k, v in expected_aug_clips.items()}
    short_nonblank = sum(c for d, c in duration_counts["nonblank"].items() if d <= 6)

    return {
        "schema_version": 1,
        "identity": "Phoenix-2014T ISLR temporal-input construction audit",
        "scope": "train annotations only; no dev/test media, labels, keypoints, or predictions opened",
        "constants": {
            "input_frames": input_frames,
            "bag_size": bag_size,
            "foreground_coordinate_ratio": 2,
            "training_short_clip_padding": "repeat endpoint frames; random pad_left in [0, 16-duration)",
            "evaluation_short_clip_padding": "repeat endpoint frames; centered"
        },
        "bags": {
            "total": len(bags),
            "by_label_kind": dict(bag_counts),
            "annotations_total": sum(len(v) for v in bags.values()),
            "clips_sampled_per_epoch": len(bags) * bag_size,
            "nonblank_short_duration_le_6": short_nonblank,
            "nonblank_short_fraction": short_nonblank / bag_counts["nonblank"],
            "bag_size_equals_duration_plus_one_violations": dict(relation_violations)
        },
        "base_duration_histogram": {k: dict(sorted(v.items())) for k, v in duration_counts.items()},
        "bag_size_histogram": {k: dict(sorted(v.items())) for k, v in bag_size_counts.items()},
        "expected_epoch_sampling": {
            "base_clips_by_duration_bucket": {k: dict(v) for k, v in expected_base_clips.items()},
            "augmented_clips_by_duration_bucket": {k: dict(v) for k, v in expected_aug_clips.items()},
            "base_clips_total": expected_base_total,
            "augmented_clips_total": expected_aug_total,
            "base_fraction": {
                k: expected_base_total[k] / (expected_base_total[k] + expected_aug_total[k])
                for k in expected_base_total
            }
        },
        "augmented_raw_target_fraction_weighted_mean": {
            kind: {dur: weighted_mean(pairs) for dur, pairs in by_dur.items()}
            for kind, by_dur in aug_fraction_pairs.items()
        },
        "expected_edge_augmented_clips_per_epoch": dict(edge_aug_expected),
        "short_base_padding_foreground_coordinate_proxy": padding_by_duration,
        "implementation_findings": [
            "The current loader randomizes left/right endpoint padding for short training clips.",
            "Dataloader temp_idx for an unaugmented short base assumes centered padding and is computed before the random pad offset is known.",
            "Therefore split_nonblk foreground pooling can select shifted bins for short base clips; magnitude is quantified by the coordinate proxy.",
            "For augmented clips away from sentence edges, temp_idx is tied to the correct raw overlap inside a 16-frame context window.",
            "The main whole-window classification loss still pools all temporal features, including context; split_nonblk is an auxiliary foreground classification loss rather than a hard mask on the main logits."
        ]
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--train", type=Path, default=DEFAULT_TRAIN)
    p.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    p.add_argument("--checkpoint", type=Path, default=DEFAULT_CKPT)
    p.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    p.add_argument("--input-frames", type=int, default=16)
    p.add_argument("--bag-size", type=int, default=6)
    p.add_argument("--verify", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    summary_path = args.output / "summary.json"
    manifest_path = args.output / "manifest.json"
    if args.verify:
        summary = json.loads(summary_path.read_text())
        manifest = json.loads(manifest_path.read_text())
        assert sha256(summary_path) == manifest["outputs"]["summary.json"]
        for name, info in manifest["inputs"].items():
            assert sha256(Path(info["path"])) == info["sha256"], name
        assert summary["scope"].startswith("train annotations only")
        assert summary["bags"]["total"] == 85379
        print(json.dumps({"verified": True, "manifest": str(manifest_path)}, sort_keys=True))
        return

    args.output.mkdir(parents=True, exist_ok=True)
    summary = analyze(args.train, args.input_frames, args.bag_size)
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    manifest = {
        "schema_version": 1,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "command": "audit_phoenix_islr_temporal_inputs.py",
        "inputs": {
            "train_annotations": {"path": str(args.train.resolve()), "sha256": sha256(args.train)},
            "training_config": {"path": str(args.config.resolve()), "sha256": sha256(args.config)},
            "baseline_checkpoint": {"path": str(args.checkpoint.resolve()), "sha256": sha256(args.checkpoint)}
        },
        "outputs": {"summary.json": sha256(summary_path)}
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"summary": str(summary_path), "manifest": str(manifest_path)}, sort_keys=True))


if __name__ == "__main__":
    main()
