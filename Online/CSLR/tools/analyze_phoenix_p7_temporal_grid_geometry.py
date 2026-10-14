#!/usr/bin/env python3
"""Summarize foreground/context geometry for the preregistered P7 grid."""

from __future__ import annotations

import argparse
import hashlib
import json
import pickle
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import yaml


CSLR_ROOT = Path(__file__).resolve().parents[1]
ASSETS = CSLR_ROOT / "results/phoenix-2014t_ISLR/p7_temporal_grid_assets_v1"
OUTPUT = CSLR_ROOT / "results/phoenix-2014t_ISLR/p7_temporal_grid_geometry_v1"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def bucket(item):
    if str(item["label"]) == "<blank>":
        return "blank"
    duration = int(item["base_end"]) - int(item["base_start"])
    if duration <= 6:
        return "1-6"
    if duration <= 9:
        return "7-9"
    if duration <= 15:
        return "10-15"
    return "16+"


def output_coordinates(item, frames):
    start, end = int(item["start"]), int(item["end"])
    coords = np.arange(start, end)
    if len(coords) >= frames:
        offset = (len(coords) - frames) // 2
        return coords[offset:offset + frames]
    remaining = frames - len(coords)
    left = remaining // 2
    right = remaining - left
    return np.concatenate([np.repeat(coords[0], left), coords, np.repeat(coords[-1], right)])


def stats(values):
    return {
        "n": len(values),
        "mean_foreground_fraction": float(np.mean(values)),
        "median_foreground_fraction": float(np.median(values)),
        "p10_foreground_fraction": float(np.quantile(values, 0.1)),
        "fraction_below_half_foreground": float(np.mean(np.asarray(values) < 0.5)),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--assets", type=Path, default=ASSETS)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args()
    args.output = args.output.resolve()
    manifest_path = args.output / "manifest.json"
    if args.verify:
        manifest = json.loads(manifest_path.read_text())
        for relative, digest in manifest["outputs"].items():
            assert sha256(args.output / relative) == digest, relative
        print(json.dumps({"verified": True, "manifest": str(manifest_path)}, sort_keys=True))
        return

    args.output.mkdir(parents=True, exist_ok=True)
    asset_manifest_path = args.assets / "manifest.json"
    asset_manifest = json.loads(asset_manifest_path.read_text())
    summary = {"candidates": {}}
    for candidate, record in sorted(asset_manifest["configs"].items()):
        cfg = yaml.safe_load(Path(record["path"]).read_text())
        frames = int(cfg["data"]["num_output_frames"])
        with Path(cfg["data"]["train"]).open("rb") as handle:
            annotations = pickle.load(handle)
        grouped = defaultdict(list)
        for item in annotations:
            coords = output_coordinates(item, frames)
            foreground = np.logical_and(coords >= int(item["base_start"]), coords < int(item["base_end"]))
            grouped[bucket(item)].append(float(np.mean(foreground)))
            grouped["overall"].append(float(np.mean(foreground)))
        summary["candidates"][candidate] = {
            "input_frames": frames,
            "centered_online_lookahead_frames": frames // 2,
            "nominal_frame_compute_vs_16": frames / 16,
            "geometry": {key: stats(value) for key, value in sorted(grouped.items())},
        }
    summary_path = args.output / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    manifest = {
        "schema_version": 1,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "annotation-only geometry; no model outcomes and no dev/test",
        "inputs": {"assets": {"path": str(asset_manifest_path.resolve()), "sha256": sha256(asset_manifest_path)}},
        "outputs": {summary_path.name: sha256(summary_path)},
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"summary": str(summary_path)}, sort_keys=True))


if __name__ == "__main__":
    main()
