#!/usr/bin/env python3
"""Build a CSL-Daily training annotation that excludes quarantined samples."""

import argparse
import gzip
import hashlib
import json
import os
import pickle
import tempfile
from pathlib import Path


SCHEMA_VERSION = 1


def sha256_file(path, chunk_size=1024 * 1024):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def filter_annotations(records, quarantine_names):
    if not isinstance(records, list):
        raise ValueError(f"annotation must contain a list, got {type(records).__name__}")
    names = [record.get("name") for record in records]
    if len(names) != len(set(names)):
        raise ValueError("annotation contains duplicate sample names")
    missing = sorted(set(quarantine_names) - set(names))
    if missing:
        raise ValueError(f"quarantined samples not found: {missing}")
    kept = [record for record in records if record["name"] not in quarantine_names]
    removed = [record for record in records if record["name"] in quarantine_names]
    return kept, removed


def write_gzip_pickle(value, destination):
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile("wb", dir=destination.parent, delete=False) as raw:
            temporary = Path(raw.name)
            with gzip.GzipFile(fileobj=raw, mode="wb", mtime=0) as handle:
                pickle.dump(value, handle, protocol=pickle.HIGHEST_PROTOCOL)
            raw.flush()
            os.fsync(raw.fileno())
        os.chmod(temporary, 0o644)
        os.replace(temporary, destination)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def build(source, destination, quarantine_path, manifest_path):
    source, destination = Path(source).resolve(), Path(destination).resolve()
    quarantine_path, manifest_path = Path(quarantine_path).resolve(), Path(manifest_path).resolve()
    quarantine = json.loads(quarantine_path.read_text(encoding="utf-8"))
    names = [item["name"] for item in quarantine["samples"]]
    with gzip.open(source, "rb") as handle:
        records = pickle.load(handle)
    kept, removed = filter_annotations(records, set(names))
    write_gzip_pickle(kept, destination)
    with gzip.open(destination, "rb") as handle:
        reloaded = pickle.load(handle)
    if [item["name"] for item in reloaded] != [item["name"] for item in kept]:
        raise ValueError("written annotation differs from filtered records")
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "operation": "exclude_unrecoverable_csl_daily_training_samples",
        "source": {"path": str(source), "sha256": sha256_file(source), "records": len(records)},
        "quarantine": {"path": str(quarantine_path), "sha256": sha256_file(quarantine_path)},
        "output": {"path": str(destination), "sha256": sha256_file(destination), "records": len(reloaded)},
        "removed": [{"name": item["name"], "num_frames": item["num_frames"]} for item in removed],
        "validation": {"order_preserved": True, "all_quarantined_samples_removed": True},
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--destination", type=Path, required=True)
    parser.add_argument("--quarantine", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(build(args.source, args.destination, args.quarantine, args.manifest), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
