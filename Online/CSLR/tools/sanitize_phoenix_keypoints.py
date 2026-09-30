#!/usr/bin/env python3
"""Create a finite Phoenix keypoint artifact without altering the source.

Any joint whose x or y coordinate is non-finite is represented as a missing
joint by setting its complete ``(x, y, confidence)`` triplet to zero.  This is
conservative: a corrupted coordinate can never be consumed as a valid point,
and every unaffected value remains byte-for-byte equal at the ndarray level.
"""

import argparse
import hashlib
import json
import os
import pickle
import tempfile
from pathlib import Path

import numpy as np


SCHEMA_VERSION = 1


def sha256_file(path, chunk_size=8 * 1024 * 1024):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sanitize_mapping(mapping):
    if not isinstance(mapping, dict):
        raise ValueError(f"source must contain a dict, got {type(mapping).__name__}")
    sanitized = {}
    repairs = []
    for key, value in mapping.items():
        if not isinstance(key, str) or not isinstance(value, np.ndarray):
            raise ValueError(f"{key!r}: expected string key and ndarray value")
        if value.ndim != 3 or value.shape[-1] < 3:
            raise ValueError(f"{key}: expected [frames, joints, >=3], got {value.shape}")
        bad_joints = ~np.isfinite(value[..., :2]).all(axis=-1)
        if not bad_joints.any():
            sanitized[key] = value
            continue
        repaired = value.copy()
        for frame, joint in np.argwhere(bad_joints):
            before = repaired[frame, joint, :3].astype(np.float64).tolist()
            bad_components = np.flatnonzero(~np.isfinite(repaired[frame, joint, :2])).tolist()
            repaired[frame, joint, :3] = 0
            repairs.append({
                "sample": key,
                "frame": int(frame),
                "joint": int(joint),
                "nonfinite_xy_components": bad_components,
                "before_xyz": [str(item) for item in before],
                "after_xyz": ["0.0", "0.0", "0.0"],
            })
        sanitized[key] = repaired
    return sanitized, repairs


def assert_valid_repair(source, sanitized, repairs):
    if set(source) != set(sanitized):
        raise ValueError("sanitized key set differs from source")
    repair_sites = {(item["sample"], item["frame"], item["joint"]) for item in repairs}
    observed_sites = set()
    for key in source:
        left, right = source[key], sanitized[key]
        if left.shape != right.shape or left.dtype != right.dtype:
            raise ValueError(f"{key}: shape or dtype changed")
        if not np.isfinite(right).all():
            raise ValueError(f"{key}: sanitized output still contains non-finite values")
        changed_joints = np.any(left != right, axis=-1)
        for frame, joint in np.argwhere(changed_joints):
            site = (key, int(frame), int(joint))
            observed_sites.add(site)
            if site not in repair_sites:
                raise ValueError(f"{key}: unexpected change at frame={frame}, joint={joint}")
            if not np.all(right[frame, joint, :3] == 0):
                raise ValueError(f"{key}: repaired joint is not zeroed")
        unchanged = ~changed_joints
        if not np.array_equal(left[unchanged], right[unchanged], equal_nan=True):
            raise ValueError(f"{key}: an unaffected value changed")
    if observed_sites != repair_sites:
        raise ValueError("manifest repair sites do not match changed joints")


def atomic_pickle(value, destination):
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile("wb", dir=destination.parent, delete=False) as handle:
            temporary = Path(handle.name)
            pickle.dump(value, handle, protocol=pickle.HIGHEST_PROTOCOL)
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(temporary, 0o644)
        os.replace(temporary, destination)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def sanitize_file(source, destination, manifest_path):
    source, destination = Path(source).resolve(), Path(destination).resolve()
    manifest_path = Path(manifest_path).resolve()
    if source == destination:
        raise ValueError("source and destination must differ")
    with source.open("rb") as handle:
        original = pickle.load(handle)
    sanitized, repairs = sanitize_mapping(original)
    assert_valid_repair(original, sanitized, repairs)
    atomic_pickle(sanitized, destination)
    with destination.open("rb") as handle:
        reloaded = pickle.load(handle)
    assert_valid_repair(original, reloaded, repairs)
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "operation": "zero_nonfinite_phoenix_keypoint_joints",
        "policy": "if x or y is non-finite, set the complete x/y/confidence triplet to zero",
        "source": {"path": str(source), "sha256": sha256_file(source), "bytes": source.stat().st_size},
        "output": {"path": str(destination), "sha256": sha256_file(destination), "bytes": destination.stat().st_size},
        "videos": len(original),
        "repaired_samples": len({item["sample"] for item in repairs}),
        "repaired_joints": len(repairs),
        "nonfinite_xy_scalars": sum(len(item["nonfinite_xy_components"]) for item in repairs),
        "repairs": repairs,
        "validation": {
            "all_output_values_finite": True,
            "keys_shapes_and_dtypes_preserved": True,
            "unaffected_values_preserved": True,
            "repaired_triplets_zeroed": True,
        },
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--destination", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(sanitize_file(args.source, args.destination, args.manifest), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
