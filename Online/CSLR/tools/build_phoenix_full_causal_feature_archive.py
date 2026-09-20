#!/usr/bin/env python3
"""Build a resumable, train-only P2 causal pose/hand/motion archive.

This is a feature-only companion to ``build_phoenix_p2_center_sequences.py``.
It intentionally includes every train sample because midpoint-target validity is
irrelevant to the downstream decoder-utility predictors.  The numerical frame
feature definition is imported unchanged from the frozen P2 implementation.
"""

import argparse
import gzip
import hashlib
import importlib.util
import json
import os
import pickle
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


CSLR_ROOT = Path(__file__).resolve().parents[1]
SLRT_ROOT = CSLR_ROOT.parents[1]
DEFAULT_KEYPOINTS = (
    SLRT_ROOT / "data/phoenix_2014t/keypoints_hrnet_dark_coco_wholebody_iso.train.pkl"
)
DEFAULT_META = SLRT_ROOT / "data/phoenix_2014t/phoenix14t.train"
DEFAULT_DENSE = CSLR_ROOT / "results/phoenix-2014t_ISLR/train_dense_stride1_v1_49faacc3"
DEFAULT_OUTPUT = (
    CSLR_ROOT
    / "results/phoenix-2014t_ISLR/p3_fulltrain_causal_pose_hand_motion_v1_49faacc3"
)
EXPECTED_SAMPLES = 7096
EXPECTED_FRAMES = 827354


def _import_p2():
    path = Path(__file__).resolve().with_name("build_phoenix_p2_center_sequences.py")
    spec = importlib.util.spec_from_file_location("frozen_p2_features", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


P2 = _import_p2()


def sha256_file(path, chunk_size=8 * 1024 * 1024):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def names_sha256(names):
    return hashlib.sha256("\n".join(names).encode("utf-8")).hexdigest()


def load_train_meta(path):
    P2.p1_features.reject_test_path(path)
    with gzip.open(path, "rb") as handle:
        records = pickle.load(handle)
    if not records or any(not str(row["name"]).startswith("train/") for row in records):
        raise ValueError("metadata is not unambiguously train-only")
    return records


def expected_dense_shard(records, index, shard_samples, shard_count, dense_root):
    left = index * shard_samples
    subset = records[left : left + shard_samples]
    directory = dense_root / "shards" / f"shard-{index:05d}-of-{shard_count:05d}"
    complete_path = directory / "complete.json"
    results_path = directory / "train_results.pkl"
    if not complete_path.is_file() or not results_path.is_file():
        raise FileNotFoundError(f"dense shard is incomplete: {directory}")
    complete = json.loads(complete_path.read_text(encoding="utf-8"))
    expected_range = [left, left + len(subset)]
    if complete.get("shard_index") != index or complete.get("sample_range") != expected_range:
        raise ValueError(f"dense shard marker has the wrong identity: {directory}")
    recorded = complete.get("artifacts", {}).get("train_results.pkl", {}).get("sha256")
    if recorded != sha256_file(results_path):
        raise ValueError(f"dense result hash differs from completion marker: {directory}")
    with results_path.open("rb") as handle:
        results = pickle.load(handle)
    expected_names = [row["name"] for row in subset]
    if list(results) != expected_names:
        raise ValueError(f"dense shard order differs from train metadata: {index}")
    for row in subset:
        starts = [int(item["start"]) for item in results[row["name"]]["adaptive_stride_metadata"]]
        if starts != list(range(int(row["num_frames"]))):
            raise ValueError(f"dense coordinates are not stride-1: {row['name']}")
    return subset


def shard_paths(output_root, index, shard_count):
    stem = f"features-{index:05d}-of-{shard_count:05d}"
    return output_root / "shards" / f"{stem}.npz", output_root / "shards" / f"{stem}.complete.json"


def inspect_feature_npz(path):
    with np.load(path, allow_pickle=False) as data:
        required = {"features", "video_names", "video_offsets", "feature_names"}
        if set(data.files) != required:
            raise ValueError(f"unexpected archive members: {path}")
        features = data["features"]
        names = [str(value) for value in data["video_names"].tolist()]
        offsets = np.asarray(data["video_offsets"], dtype=np.int64)
        feature_names = [str(value) for value in data["feature_names"].tolist()]
        if features.dtype != np.float16 or features.ndim != 2:
            raise ValueError(f"unexpected feature tensor: {path}")
        if feature_names != list(P2.FEATURE_NAMES):
            raise ValueError(f"feature definition mismatch: {path}")
        if len(offsets) != len(names) + 1 or offsets[0] != 0 or offsets[-1] != len(features):
            raise ValueError(f"invalid offsets: {path}")
        if np.any(np.diff(offsets) <= 0) or not np.isfinite(features).all():
            raise ValueError(f"invalid lengths or non-finite features: {path}")
        return {
            "samples": len(names),
            "frames": int(len(features)),
            "names": names,
            "lengths": np.diff(offsets).astype(int).tolist(),
            "feature_count": int(features.shape[1]),
        }


def valid_completed_feature_shard(output_root, index, shard_count, records):
    archive, marker_path = shard_paths(output_root, index, shard_count)
    if not archive.is_file() or not marker_path.is_file():
        return None
    marker = json.loads(marker_path.read_text(encoding="utf-8"))
    if marker.get("status") != "complete" or marker.get("sha256") != sha256_file(archive):
        return None
    info = inspect_feature_npz(archive)
    expected_names = [row["name"] for row in records]
    expected_lengths = [int(row["num_frames"]) for row in records]
    if info["names"] != expected_names or info["lengths"] != expected_lengths:
        return None
    return marker


def build_feature_shard(output_root, index, shard_count, records, keypoints):
    completed = valid_completed_feature_shard(output_root, index, shard_count, records)
    if completed is not None:
        return completed | {"resumed": True}
    archive, marker_path = shard_paths(output_root, index, shard_count)
    if archive.exists() or marker_path.exists():
        raise FileExistsError(f"incomplete prior feature shard requires audit: {archive}")
    matrices = []
    offsets = [0]
    names = []
    for row in records:
        name, length = row["name"], int(row["num_frames"])
        array = keypoints[name]
        if len(array) != length:
            raise ValueError(f"keypoint length mismatch: {name}")
        matrix = P2.frame_features(array).astype(np.float16)
        matrices.append(matrix)
        names.append(name)
        offsets.append(offsets[-1] + length)
    feature_matrix = np.concatenate(matrices, axis=0)
    temporary = archive.with_name(f".{archive.name}.incomplete-{os.getpid()}")
    with temporary.open("wb") as handle:
        np.savez_compressed(
            handle,
            features=feature_matrix,
            video_names=np.asarray(names),
            video_offsets=np.asarray(offsets, dtype=np.int64),
            feature_names=np.asarray(P2.FEATURE_NAMES),
        )
    os.replace(temporary, archive)
    info = inspect_feature_npz(archive)
    marker = {
        "status": "complete",
        "shard": index,
        "shards": shard_count,
        "samples": info["samples"],
        "frames": info["frames"],
        "names_sha256": names_sha256(names),
        "bytes": archive.stat().st_size,
        "sha256": sha256_file(archive),
    }
    marker_path.write_text(json.dumps(marker, indent=2) + "\n", encoding="utf-8")
    return marker


def assemble_archive(output_root, records, shard_count):
    archive_path = output_root / "train_causal_pose_hand_motion_sequences.npz"
    expected_names = [row["name"] for row in records]
    expected_lengths = [int(row["num_frames"]) for row in records]
    if archive_path.is_file():
        info = inspect_feature_npz(archive_path)
        if info["names"] != expected_names or info["lengths"] != expected_lengths:
            raise ValueError("existing assembled archive has the wrong sample identity")
        return archive_path, info

    total_frames = sum(expected_lengths)
    matrix = np.empty((total_frames, len(P2.FEATURE_NAMES)), dtype=np.float16)
    cursor = 0
    observed_names = []
    for index in range(shard_count):
        left = index * 128
        subset = records[left : left + 128]
        marker = valid_completed_feature_shard(output_root, index, shard_count, subset)
        if marker is None:
            raise RuntimeError(f"feature shard {index} is not valid and complete")
        shard_path, _ = shard_paths(output_root, index, shard_count)
        with np.load(shard_path, allow_pickle=False) as data:
            values = data["features"]
            matrix[cursor : cursor + len(values)] = values
            cursor += len(values)
            observed_names.extend(str(value) for value in data["video_names"].tolist())
    if cursor != total_frames or observed_names != expected_names:
        raise RuntimeError("assembled feature coverage differs from metadata")
    offsets = np.concatenate([[0], np.cumsum(expected_lengths, dtype=np.int64)])
    temporary = archive_path.with_name(f".{archive_path.name}.incomplete-{os.getpid()}")
    with temporary.open("wb") as handle:
        np.savez_compressed(
            handle,
            features=matrix,
            video_names=np.asarray(expected_names),
            video_offsets=offsets,
            feature_names=np.asarray(P2.FEATURE_NAMES),
        )
    os.replace(temporary, archive_path)
    return archive_path, inspect_feature_npz(archive_path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-keypoints", type=Path, default=DEFAULT_KEYPOINTS)
    parser.add_argument("--train-meta", type=Path, default=DEFAULT_META)
    parser.add_argument("--dense-root", type=Path, default=DEFAULT_DENSE)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    for path in (args.train_keypoints, args.train_meta, args.dense_root):
        P2.p1_features.reject_test_path(path)
    records = load_train_meta(args.train_meta)
    if len(records) != EXPECTED_SAMPLES or sum(int(row["num_frames"]) for row in records) != EXPECTED_FRAMES:
        raise ValueError("unexpected full-train metadata scope")
    dense_manifest_path = args.dense_root / "protocol_manifest.json"
    dense_manifest = json.loads(dense_manifest_path.read_text(encoding="utf-8"))
    if dense_manifest.get("split") != "train" or dense_manifest.get("samples") != EXPECTED_SAMPLES:
        raise ValueError("dense replay is not the expected train-only scope")
    shard_samples = int(dense_manifest["shard_samples"])
    if shard_samples != 128:
        raise ValueError("this archive version is frozen to 128-sample dense shards")
    shard_count = (len(records) + shard_samples - 1) // shard_samples

    args.output_root.mkdir(parents=True, exist_ok=True)
    (args.output_root / "shards").mkdir(exist_ok=True)
    with args.train_keypoints.open("rb") as handle:
        keypoints = pickle.load(handle)
    P2.p1_features.assert_train_only_keys(keypoints)
    expected_names = [row["name"] for row in records]
    if set(keypoints) != set(expected_names):
        raise ValueError("train keypoint and metadata name sets differ")

    shard_markers = []
    for index in range(shard_count):
        subset = expected_dense_shard(records, index, shard_samples, shard_count, args.dense_root)
        marker = build_feature_shard(args.output_root, index, shard_count, subset, keypoints)
        shard_markers.append(marker)
        print(f"feature shard {index + 1}/{shard_count}: {marker['samples']} samples, {marker['frames']} frames", flush=True)

    archive_path, info = assemble_archive(args.output_root, records, shard_count)
    if info["samples"] != EXPECTED_SAMPLES or info["frames"] != EXPECTED_FRAMES:
        raise RuntimeError("final archive coverage mismatch")
    manifest = {
        "schema_version": 1,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "complete Phoenix-2014T train-only cheap causal pose/hand/motion frame features",
        "samples": info["samples"],
        "frames": info["frames"],
        "feature_count": info["feature_count"],
        "sample_ids_sha256": names_sha256(expected_names),
        "inputs": {
            str(path.resolve()): {"bytes": path.stat().st_size, "sha256": sha256_file(path)}
            for path in (
                args.train_keypoints,
                args.train_meta,
                dense_manifest_path,
                Path(P2.__file__),
                Path(__file__),
            )
        },
        "output": {
            "path": str(archive_path.resolve()),
            "bytes": archive_path.stat().st_size,
            "sha256": sha256_file(archive_path),
            "dtype": "float16",
            "feature_names": list(P2.FEATURE_NAMES),
        },
        "shards": shard_markers,
        "causality": {
            "per_frame_t_visible_keypoints": "frames 0..t only",
            "temporal_differences": "current frame t minus previous frame t-1 only",
            "future_frames": False,
            "reference_gloss_or_alignment": False,
            "islr_logits_or_decoder_state": False,
            "known_EOS": False,
            "downstream_history_note": "a 31-step history ending at decision availability remains past-only",
        },
        "p2_compatibility": {
            "feature_function": "build_phoenix_p2_center_sequences.frame_features unchanged",
            "midpoint_targets_included": False,
            "all_7096_samples_included": True,
            "difference_from_old_p2_archive": (
                "the old supervised midpoint archive excluded one target-inconsistent sample; "
                "this feature-only archive includes it without constructing a midpoint target"
            ),
        },
        "device": "CPU only; CUDA_VISIBLE_DEVICES was empty",
        "split_policy": "train only; dev and test forbidden",
    }
    manifest_path = args.output_root / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"archive": str(archive_path), "manifest": str(manifest_path),
                      "samples": info["samples"], "frames": info["frames"],
                      "sha256": manifest["output"]["sha256"]}, indent=2))


if __name__ == "__main__":
    main()
