#!/usr/bin/env python3
"""Build train-only causal keypoint sequences for Phoenix P2."""

import argparse
import gzip
import importlib.util
import json
import pickle
from pathlib import Path

import numpy as np

_P1_PATH = Path(__file__).resolve().with_name("build_phoenix_p1_center_features.py")
_P1_SPEC = importlib.util.spec_from_file_location("build_phoenix_p1_center_features", _P1_PATH)
p1_features = importlib.util.module_from_spec(_P1_SPEC)
_P1_SPEC.loader.exec_module(p1_features)


POSE = np.arange(11, dtype=np.int64)
LEFT_HAND = np.arange(91, 112, dtype=np.int64)
RIGHT_HAND = np.arange(112, 133, dtype=np.int64)
HAND_ANCHORS = np.asarray([0, 4, 8, 12, 16, 20], dtype=np.int64)
RAW_SIZE = np.asarray([210.0, 260.0], dtype=np.float32)
CONFIDENCE_MAX = 1.5


def _names():
    names = []
    for axis in ("x", "y"):
        names.extend(f"pose_{index}_{axis}" for index in range(11))
    names.extend(f"pose_{index}_valid" for index in range(11))
    names.extend(f"pose_{index}_confidence" for index in range(11))
    for side in ("left", "right"):
        for axis in ("x", "y"):
            names.extend(f"{side}_hand_local_{index}_{axis}" for index in range(21))
        names.extend(f"{side}_hand_{index}_valid" for index in range(21))
        for axis in ("x", "y"):
            names.extend(f"{side}_anchor_{index}_{axis}" for index in HAND_ANCHORS)
    names.extend([
        "left_valid_fraction", "right_valid_fraction",
        "left_mean_confidence", "right_mean_confidence",
        "left_centroid_relative_shoulder_x", "left_centroid_relative_shoulder_y",
        "right_centroid_relative_shoulder_x", "right_centroid_relative_shoulder_y",
        "interhand_distance", "pose_motion", "left_global_motion", "right_global_motion",
        "left_shape_change", "right_shape_change", "interhand_distance_change",
        "left_confidence_change", "right_confidence_change",
    ])
    return tuple(names)


FEATURE_NAMES = _names()


def _masked_mean(xy, valid):
    count = valid.sum(axis=1, keepdims=True)
    return (xy * valid[..., None]).sum(axis=1) / np.maximum(count, 1)


def _median_motion(current, previous, pair_valid):
    values = np.linalg.norm((current - previous) / RAW_SIZE, axis=-1)
    result = np.zeros(len(current), dtype=np.float32)
    for frame in range(1, len(current)):
        if pair_valid[frame].any():
            result[frame] = np.median(values[frame, pair_valid[frame]])
    return result


def frame_features(array, confidence_threshold=0.2):
    """Return features at t that use only keypoints from frames <= t."""
    keypoints = np.asarray(array, dtype=np.float32)
    if keypoints.ndim != 3 or keypoints.shape[1:] != (133, 3):
        raise ValueError("keypoints must have shape [T, 133, 3]")
    finite = np.isfinite(keypoints).all(axis=-1)
    valid = finite & (keypoints[..., 2] >= confidence_threshold)
    xy = keypoints[..., :2].copy()
    xy[~valid] = 0.0
    confidence = np.clip(np.where(finite, keypoints[..., 2], 0.0), 0.0, CONFIDENCE_MAX) / CONFIDENCE_MAX

    pose_xy, pose_valid = xy[:, POSE], valid[:, POSE]
    left_xy, left_valid = xy[:, LEFT_HAND], valid[:, LEFT_HAND]
    right_xy, right_valid = xy[:, RIGHT_HAND], valid[:, RIGHT_HAND]
    pose_global = pose_xy / RAW_SIZE

    local_parts = []
    anchor_parts = []
    centroids = []
    hand_motion = []
    shape_motion = []
    mean_confidence = []
    for hand_xy, hand_valid, indices in (
        (left_xy, left_valid, LEFT_HAND), (right_xy, right_valid, RIGHT_HAND)
    ):
        centroid = _masked_mean(hand_xy, hand_valid)
        wrist = np.where(hand_valid[:, :1], hand_xy[:, 0], centroid)
        local = (hand_xy - wrist[:, None, :]) / RAW_SIZE
        local[~hand_valid] = 0.0
        anchors = hand_xy[:, HAND_ANCHORS] / RAW_SIZE
        anchors[~hand_valid[:, HAND_ANCHORS]] = 0.0
        previous_xy = np.concatenate([hand_xy[:1], hand_xy[:-1]])
        previous_local = np.concatenate([local[:1], local[:-1]])
        pair_valid = hand_valid & np.concatenate([hand_valid[:1], hand_valid[:-1]])
        local_parts.append(local)
        anchor_parts.append(anchors)
        centroids.append(centroid)
        hand_motion.append(_median_motion(hand_xy, previous_xy, pair_valid))
        shape_motion.append(_median_motion(local * RAW_SIZE, previous_local * RAW_SIZE, pair_valid))
        mean_confidence.append(confidence[:, indices].mean(axis=1))

    shoulders_valid = pose_valid[:, 5] & pose_valid[:, 6]
    pose_centroid = _masked_mean(pose_xy, pose_valid)
    shoulder_center = (pose_xy[:, 5] + pose_xy[:, 6]) / 2.0
    shoulder_center = np.where(shoulders_valid[:, None], shoulder_center, pose_centroid)
    centroid_relative = [(value - shoulder_center) / RAW_SIZE for value in centroids]
    both_hands = left_valid.any(axis=1) & right_valid.any(axis=1)
    interhand = np.linalg.norm((centroids[0] - centroids[1]) / RAW_SIZE, axis=1)
    interhand[~both_hands] = 0.0
    previous_pose = np.concatenate([pose_xy[:1], pose_xy[:-1]])
    pose_pair_valid = pose_valid & np.concatenate([pose_valid[:1], pose_valid[:-1]])
    pose_motion = _median_motion(pose_xy, previous_pose, pose_pair_valid)

    scalars = np.column_stack([
        left_valid.mean(axis=1), right_valid.mean(axis=1),
        mean_confidence[0], mean_confidence[1],
        centroid_relative[0], centroid_relative[1], interhand, pose_motion,
        hand_motion[0], hand_motion[1], shape_motion[0], shape_motion[1],
        interhand - np.concatenate([interhand[:1], interhand[:-1]]),
        mean_confidence[0] - np.concatenate([mean_confidence[0][:1], mean_confidence[0][:-1]]),
        mean_confidence[1] - np.concatenate([mean_confidence[1][:1], mean_confidence[1][:-1]]),
    ]).astype(np.float32)
    matrix = np.column_stack([
        pose_global[:, :, 0], pose_global[:, :, 1], pose_valid, confidence[:, POSE],
        local_parts[0][:, :, 0], local_parts[0][:, :, 1], left_valid,
        anchor_parts[0][:, :, 0], anchor_parts[0][:, :, 1],
        local_parts[1][:, :, 0], local_parts[1][:, :, 1], right_valid,
        anchor_parts[1][:, :, 0], anchor_parts[1][:, :, 1], scalars,
    ]).astype(np.float32)
    if matrix.shape[1] != len(FEATURE_NAMES) or not np.isfinite(matrix).all():
        raise ValueError("invalid P2 feature matrix")
    return matrix


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-keypoints", type=Path, required=True)
    parser.add_argument("--train-bags", type=Path, required=True)
    parser.add_argument("--train-meta", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--split-output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260912)
    args = parser.parse_args()
    for path in (args.train_keypoints, args.train_bags, args.train_meta):
        p1_features.reject_test_path(path)
    with args.train_keypoints.open("rb") as handle:
        keypoints = pickle.load(handle)
    p1_features.assert_train_only_keys(keypoints)
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
        raise ValueError("train metadata, segments, and keypoints differ")
    names = [name for name in all_names if name not in p1_features.EXCLUDED_TARGET_VIDEOS]
    fit, calibration = p1_features.deterministic_group_split(names, seed=args.seed)
    matrices, targets, offsets = [], [], [0]
    for item in meta:
        name, length = item["name"], int(item["num_frames"])
        if name in p1_features.EXCLUDED_TARGET_VIDEOS:
            continue
        if len(keypoints[name]) != length:
            raise ValueError(f"keypoint length mismatch: {name}")
        matrix = frame_features(keypoints[name])
        target, _ = p1_features.midpoint_targets(length, segments[name])
        matrices.append(matrix.astype(np.float16))
        targets.append(target)
        offsets.append(offsets[-1] + length)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output, features=np.concatenate(matrices), targets=np.concatenate(targets),
        video_names=np.asarray(names), video_offsets=np.asarray(offsets, dtype=np.int64),
        feature_names=np.asarray(FEATURE_NAMES),
    )
    args.split_output.parent.mkdir(parents=True, exist_ok=True)
    args.split_output.write_text(json.dumps({"seed": args.seed, "fit_videos": fit,
                                              "calibration_videos": calibration}, indent=2) + "\n")


if __name__ == "__main__":
    main()
