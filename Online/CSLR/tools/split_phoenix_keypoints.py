#!/usr/bin/env python3
"""Split the monolithic Phoenix keypoint pickle into isolated split files.

The source is read-only. Existing outputs are never overwritten: they are
loaded and validated against the source instead. New outputs are published
atomically with a no-clobber hard link from a temporary file.
"""

import argparse
import hashlib
import json
import os
import pickle
import tempfile
from collections import Counter
from pathlib import Path

import numpy as np


SPLITS = ("train", "dev", "test")
SCHEMA_VERSION = 1


def sha256_file(path, chunk_size=8 * 1024 * 1024):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def key_split(key):
    if not isinstance(key, str):
        raise ValueError(f"key must be a string, got {type(key).__name__}")
    prefix, separator, remainder = key.partition("/")
    if not separator or not remainder or prefix not in SPLITS:
        raise ValueError(f"invalid split-qualified key: {key!r}")
    return prefix


def partition_mapping(mapping):
    if not isinstance(mapping, dict):
        raise ValueError(f"source must contain a dict, got {type(mapping).__name__}")
    partitions = {split: {} for split in SPLITS}
    for key, value in mapping.items():
        partitions[key_split(key)][key] = value
    if any(not partitions[split] for split in SPLITS):
        raise ValueError("source must contain at least one item for every split")
    return partitions


def array_summary(mapping):
    dtype_counts = Counter()
    ndim_counts = Counter()
    trailing_shape_counts = Counter()
    total_frames = 0
    for key, value in mapping.items():
        if not isinstance(value, np.ndarray):
            raise ValueError(f"{key}: expected ndarray, got {type(value).__name__}")
        if value.ndim < 1:
            raise ValueError(f"{key}: keypoint array must have a frame dimension")
        total_frames += int(value.shape[0])
        dtype_counts[str(value.dtype)] += 1
        ndim_counts[str(value.ndim)] += 1
        trailing_shape_counts[str(list(value.shape[1:]))] += 1
    return {
        "videos": len(mapping),
        "total_frames": total_frames,
        "dtype_counts": dict(sorted(dtype_counts.items())),
        "ndim_counts": dict(sorted(ndim_counts.items())),
        "trailing_shape_counts": dict(sorted(trailing_shape_counts.items())),
    }


def assert_same_values(expected, actual, split):
    if not isinstance(actual, dict):
        raise ValueError(f"{split} output must contain a dict")
    if set(actual) != set(expected):
        missing = sorted(set(expected) - set(actual))[:1]
        extra = sorted(set(actual) - set(expected))[:1]
        raise ValueError(f"{split} key mismatch; missing={missing}, extra={extra}")
    for key in expected:
        left, right = expected[key], actual[key]
        if not isinstance(right, np.ndarray):
            raise ValueError(f"{key}: output value is not an ndarray")
        if left.shape != right.shape or left.dtype != right.dtype:
            raise ValueError(f"{key}: output shape or dtype differs from source")
        if not np.array_equal(left, right, equal_nan=True):
            raise ValueError(f"{key}: output values differ from source")


def publish_pickle_no_clobber(value, destination):
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        return "existing"
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb", prefix=f".{destination.name}.", suffix=".tmp",
            dir=destination.parent, delete=False,
        ) as handle:
            temporary = Path(handle.name)
            pickle.dump(value, handle, protocol=pickle.HIGHEST_PROTOCOL)
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(temporary, 0o644)
        try:
            os.link(temporary, destination)
        except FileExistsError:
            return "existing"
        return "created"
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def publish_text_no_clobber(text, destination):
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        if destination.read_text(encoding="utf-8") != text:
            raise ValueError(f"existing manifest differs: {destination}")
        return "existing"
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", prefix=f".{destination.name}.",
            suffix=".tmp", dir=destination.parent, delete=False,
        ) as handle:
            temporary = Path(handle.name)
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(temporary, 0o644)
        try:
            os.link(temporary, destination)
        except FileExistsError:
            if destination.read_text(encoding="utf-8") != text:
                raise ValueError(f"concurrently created manifest differs: {destination}")
            return "existing"
        return "created"
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def output_paths(source, output_dir):
    stem = source.name[:-4] if source.name.endswith(".pkl") else source.name
    return {split: output_dir / f"{stem}.{split}.pkl" for split in SPLITS}


def split_file(source, output_dir, manifest_path):
    source = Path(source).resolve()
    output_dir = Path(output_dir).resolve()
    manifest_path = Path(manifest_path).resolve()
    with source.open("rb") as handle:
        complete = pickle.load(handle)
    partitions = partition_mapping(complete)
    destinations = output_paths(source, output_dir)
    statuses = {}
    # Validate every pre-existing artifact before creating any missing one.
    for split in SPLITS:
        if destinations[split].exists():
            with destinations[split].open("rb") as handle:
                existing = pickle.load(handle)
            if any(key_split(key) != split for key in existing):
                raise ValueError(f"{split} output contains a key from another split")
            assert_same_values(partitions[split], existing, split)
    for split in SPLITS:
        statuses[split] = publish_pickle_no_clobber(partitions[split], destinations[split])

    reloaded = {}
    for split in SPLITS:
        with destinations[split].open("rb") as handle:
            reloaded[split] = pickle.load(handle)
        if any(key_split(key) != split for key in reloaded[split]):
            raise ValueError(f"{split} output contains a key from another split")
        assert_same_values(partitions[split], reloaded[split], split)

    key_sets = {split: set(reloaded[split]) for split in SPLITS}
    for index, left in enumerate(SPLITS):
        for right in SPLITS[index + 1 :]:
            if not key_sets[left].isdisjoint(key_sets[right]):
                raise ValueError(f"outputs {left} and {right} are not disjoint")
    union = set().union(*(key_sets[split] for split in SPLITS))
    if union != set(complete):
        raise ValueError("output key union does not equal source key set")

    manifest = {
        "schema_version": SCHEMA_VERSION,
        "operation": "split_monolithic_phoenix_keypoints",
        "source": {
            "path": str(source),
            "bytes": source.stat().st_size,
            "sha256": sha256_file(source),
            "videos": len(complete),
        },
        "outputs": {},
        "integrity": {
            "all_keys_split_qualified": True,
            "pairwise_disjoint": True,
            "union_equals_source": True,
            "shape_dtype_and_values_equal_source": True,
        },
    }
    for split in SPLITS:
        path = destinations[split]
        manifest["outputs"][split] = {
            "path": str(path),
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
            "artifact_status": "created_or_preexisting_and_fully_validated",
            **array_summary(reloaded[split]),
        }
    text = json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    publish_text_no_clobber(text, manifest_path)
    manifest["run_write_status"] = statuses
    return manifest


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    args = parser.parse_args()
    result = split_file(args.source, args.output_dir, args.manifest)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
