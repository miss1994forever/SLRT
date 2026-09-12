#!/usr/bin/env python3
"""Build train-only causal keypoint features and sign-center proxy targets.

The command accepts a *train-only* keypoint pickle.  It refuses a mapping with
any non-train key, preventing accidental use of the repository's monolithic
train/dev/test keypoint artifact.  Core functions are importable for tests.
"""

import argparse
import gzip
import hashlib
import json
import pickle
from pathlib import Path

import numpy as np


POSE = list(range(11))
MOUTH_HALF = list(range(71, 91, 2))
HANDS = list(range(91, 133))
SELECTED = np.asarray(POSE + MOUTH_HALF + HANDS, dtype=np.int64)
FEATURE_NAMES = (
    "valid_fraction",
    "motion_now",
    "motion_mean_4",
    "motion_max_4",
    "motion_mean_16",
    "motion_max_16",
    "motion_mean_48",
    "motion_max_48",
    "motion_delta",
    "bbox_width",
    "bbox_height",
)
EXCLUDED_TARGET_VIDEOS = {
    "train/25August_2009_Tuesday_heute-3301",
}


def reject_test_path(path):
    parts = [part.lower() for part in Path(path).resolve().parts]
    if any(part == "test" or part.startswith("test_") or part.endswith("_test") for part in parts):
        raise ValueError(f"test paths are forbidden: {path}")


def assert_train_only_keys(mapping):
    invalid = [key for key in mapping if not str(key).startswith("train/")]
    if invalid:
        raise ValueError(f"keypoint mapping is not train-only; first forbidden key: {invalid[0]}")


def sanitize_keypoints(array, confidence_threshold=0.2):
    """Return finite coordinates and an explicit validity mask.

    Invalid landmark-frames are set to neutral coordinates only after their
    mask is recorded; all features use the mask and expose valid_fraction.
    """
    keypoints = np.asarray(array, dtype=np.float32)
    if keypoints.ndim != 3 or keypoints.shape[-1] != 3:
        raise ValueError("keypoints must have shape [T, N, 3]")
    selected = keypoints[:, SELECTED]
    finite = np.isfinite(selected).all(axis=-1)
    valid = finite & (selected[..., 2] >= confidence_threshold)
    xy = selected[..., :2].copy()
    xy[~valid] = 0.0
    return xy, valid


def _past_stat(values, width, reducer):
    result = np.empty_like(values, dtype=np.float32)
    for index in range(len(values)):
        begin = max(0, index - width + 1)
        result[index] = reducer(values[begin : index + 1])
    return result


def causal_features(array, confidence_threshold=0.2, raw_width=210.0, raw_height=260.0):
    """Compute features at t using only keypoints at indices <= t."""
    xy, valid = sanitize_keypoints(array, confidence_threshold)
    length = len(xy)
    motion = np.zeros(length, dtype=np.float32)
    for index in range(1, length):
        pair_valid = valid[index] & valid[index - 1]
        if pair_valid.any():
            delta = (xy[index, pair_valid] - xy[index - 1, pair_valid]) / np.asarray([raw_width, raw_height])
            motion[index] = np.median(np.linalg.norm(delta, axis=-1))
    current_min = np.zeros((length, 2), dtype=np.float32)
    current_max = np.zeros((length, 2), dtype=np.float32)
    for index in range(length):
        if valid[index].any():
            current_min[index] = xy[index, valid[index]].min(axis=0)
            current_max[index] = xy[index, valid[index]].max(axis=0)
    extent = (current_max - current_min) / np.asarray([raw_width, raw_height], dtype=np.float32)
    previous = np.concatenate([motion[:1], motion[:-1]])
    features = np.column_stack(
        [
            valid.mean(axis=1),
            motion,
            _past_stat(motion, 4, np.mean),
            _past_stat(motion, 4, np.max),
            _past_stat(motion, 16, np.mean),
            _past_stat(motion, 16, np.max),
            _past_stat(motion, 48, np.mean),
            _past_stat(motion, 48, np.max),
            motion - previous,
            extent[:, 0],
            extent[:, 1],
        ]
    ).astype(np.float32)
    if not np.isfinite(features).all():
        raise ValueError("feature construction produced a non-finite value")
    return features


def midpoint_targets(length, segments, radius=1):
    target = np.zeros(int(length), dtype=np.uint8)
    centers = []
    for item in segments:
        if item["label"] == "<blank>":
            continue
        start, end = int(item["start"]), int(item["end"])
        if not 0 <= start < end <= length:
            raise ValueError("segment is empty or outside the video")
        center = (start + end - 1) // 2
        centers.append(center)
        target[max(0, center - radius) : min(length, center + radius + 1)] = 1
    return target, np.asarray(centers, dtype=np.int32)


def deterministic_group_split(names, calibration_fraction=0.2, seed=20260912):
    fit, calibration = [], []
    threshold = int(calibration_fraction * (1 << 64))
    for name in sorted(names):
        value = int.from_bytes(hashlib.sha256(f"{seed}:{name}".encode()).digest()[:8], "big")
        (calibration if value < threshold else fit).append(name)
    if not fit or not calibration:
        raise ValueError("group split produced an empty partition")
    return fit, calibration


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-keypoints", type=Path, required=True, help="must contain train/ keys only")
    parser.add_argument("--train-bags", type=Path, required=True)
    parser.add_argument("--train-meta", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True, help="compressed npz; use /tmp for full data")
    parser.add_argument("--split-output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260912)
    args = parser.parse_args()
    for path in (args.train_keypoints, args.train_bags, args.train_meta):
        reject_test_path(path)
    with args.train_keypoints.open("rb") as handle:
        keypoints = pickle.load(handle)
    assert_train_only_keys(keypoints)
    with args.train_bags.open("rb") as handle:
        bags = pickle.load(handle)
    with gzip.open(args.train_meta, "rb") as handle:
        meta = pickle.load(handle)
    segments = {}
    for records in bags.values():
        base = [item for item in records if int(item.get("aug", -1)) == 0]
        if len(base) != 1:
            raise ValueError("every bag must have exactly one base record")
        segments.setdefault(base[0]["video_file"], []).append(base[0])
    all_names = [item["name"] for item in meta]
    if set(all_names) != set(keypoints) or set(all_names) != set(segments):
        raise ValueError("train metadata, segments, and keypoints do not have identical video sets")
    names = [name for name in all_names if name not in EXCLUDED_TARGET_VIDEOS]
    fit, calibration = deterministic_group_split(names, seed=args.seed)
    row_names, frame_indices, matrices, targets = [], [], [], []
    for item in meta:
        name, length = item["name"], int(item["num_frames"])
        if name in EXCLUDED_TARGET_VIDEOS:
            continue
        array = keypoints[name]
        if len(array) != length:
            raise ValueError(f"keypoint length mismatch: {name}")
        matrix = causal_features(array)
        target, _ = midpoint_targets(length, segments[name])
        matrices.append(matrix)
        targets.append(target)
        row_names.extend([name] * length)
        frame_indices.extend(range(length))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output,
        features=np.concatenate(matrices),
        targets=np.concatenate(targets),
        video_names=np.asarray(row_names),
        frame_indices=np.asarray(frame_indices, dtype=np.int32),
        feature_names=np.asarray(FEATURE_NAMES),
    )
    args.split_output.parent.mkdir(parents=True, exist_ok=True)
    args.split_output.write_text(
        json.dumps({"seed": args.seed, "fit_videos": fit, "calibration_videos": calibration}, indent=2) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
