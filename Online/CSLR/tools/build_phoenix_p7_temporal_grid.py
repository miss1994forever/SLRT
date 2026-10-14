#!/usr/bin/env python3
"""Build train-only controlled-context and frame-count candidates for P7."""

from __future__ import annotations

import argparse
import copy
import gzip
import hashlib
import json
import pickle
from datetime import datetime, timezone
from pathlib import Path

import yaml


CSLR_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = CSLR_ROOT.parents[1]
CALIBRATION = CSLR_ROOT / "results/phoenix-2014t_ISLR/p7_temporal_debug_smoke_assets_v1/calibration_context16.pkl"
OUTPUT = CSLR_ROOT / "results/phoenix-2014t_ISLR/p7_temporal_grid_assets_v1"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def make_item(source, start: int, end: int, tag: str):
    item = copy.deepcopy(source)
    item["start"], item["end"], item["seq_len"] = start, end, end - start
    item["aug"] = 0 if start == int(item["base_start"]) and end == int(item["base_end"]) else 1
    item["name"] = f"p7grid_{tag}_{item['label']}_{item['video_file']}_[{start}:{end}]"
    return item


def centered_window(midpoint: float, width: int, raw_length: int):
    if raw_length <= width:
        return 0, raw_length
    start = round(midpoint - width / 2)
    start = max(0, min(start, raw_length - width))
    return start, start + width


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--calibration", type=Path, default=CALIBRATION)
    parser.add_argument("--train-meta", type=Path, default=REPO_ROOT / "data/phoenix_2014t/phoenix14t.train")
    parser.add_argument("--base-config", type=Path, default=CSLR_ROOT / "configs/phoenix-2014t_ISLR.yaml")
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args()
    args.output = args.output.resolve()
    manifest_path = args.output / "manifest.json"
    if args.verify:
        manifest = json.loads(manifest_path.read_text())
        for relative, digest in manifest["outputs"].items():
            assert sha256(args.output / relative) == digest, relative
        for record in manifest["configs"].values():
            assert sha256(Path(record["path"])) == record["sha256"]
        print(json.dumps({"verified": True, "manifest": str(manifest_path)}, sort_keys=True))
        return

    args.output.mkdir(parents=True, exist_ok=True)
    with args.calibration.open("rb") as handle:
        calibration = pickle.load(handle)
    with gzip.open(args.train_meta, "rb") as handle:
        metadata = pickle.load(handle)
    raw_lengths = {str(item["name"]): int(item["num_frames"]) for item in metadata}
    variants = {}
    for margin in [0, 2, 4, 6]:
        name = f"D_margin{margin}_to16"
        rows = []
        for item in calibration:
            raw = raw_lengths[str(item["video_file"])]
            start = max(0, int(item["base_start"]) - margin)
            end = min(raw, int(item["base_end"]) + margin)
            rows.append(make_item(item, start, end, name))
        variants[name] = {"rows": rows, "frames": 16}
    for frames in [8, 12, 16, 24, 32]:
        name = f"E_context{frames}"
        rows = []
        for item in calibration:
            raw = raw_lengths[str(item["video_file"])]
            midpoint = (int(item["base_start"]) + int(item["base_end"]) - 1) / 2
            start, end = centered_window(midpoint, frames, raw)
            rows.append(make_item(item, start, end, name))
        variants[name] = {"rows": rows, "frames": frames}

    base = yaml.safe_load(args.base_config.read_text())
    for key in ["zip_file", "vocab_file"]:
        base["data"][key] = str((CSLR_ROOT / base["data"][key]).resolve())
    base["data"]["keypoint_file"] = str(
        (REPO_ROOT / "data/phoenix_2014t/keypoints_hrnet_dark_coco_wholebody_iso.train.finite.pkl").resolve()
    )
    rn = base["model"]["RecognitionNetwork"]
    rn["GlossTokenizer"]["gloss2id_file"] = str((CSLR_ROOT / rn["GlossTokenizer"]["gloss2id_file"]).resolve())
    for stream in ["s3d", "keypoint_s3d"]:
        rn[stream]["pretrained_ckpt"] = str((CSLR_ROOT / rn[stream]["pretrained_ckpt"]).resolve())

    outputs, configs = [], {}
    for name, definition in variants.items():
        annotation_path = args.output / f"{name}.train.pkl"
        with annotation_path.open("wb") as handle:
            pickle.dump(definition["rows"], handle, protocol=pickle.HIGHEST_PROTOCOL)
        outputs.append(annotation_path.name)
        cfg = copy.deepcopy(base)
        cfg["data"]["train"] = str(annotation_path.resolve())
        cfg["data"]["dev"] = "/tmp/P7_DEV_ACCESS_FORBIDDEN"
        cfg["data"]["test"] = "/tmp/P7_TEST_ACCESS_FORBIDDEN"
        cfg["data"]["num_output_frames"] = definition["frames"]
        cfg["data"]["raw_vlens_split_overrides"] = {"dev": "train"}
        cfg["data"]["transform_cfg"]["index_setting"] = ["consecutive", "pad", "central", "pad"]
        cfg["training"]["batch_size"] = 16
        cfg["training"]["num_workers"] = 4
        config_path = CSLR_ROOT / "configs" / f"p7_temporal_grid_{name}.yaml"
        config_path.write_text(yaml.safe_dump(cfg, sort_keys=False))
        configs[name] = {"path": str(config_path.resolve()), "sha256": sha256(config_path)}

    manifest = {
        "schema_version": 1,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "source-disjoint train-only controlled context and temporal scale grid",
        "candidates": {
            name: {"frames": value["frames"], "items": len(value["rows"])}
            for name, value in variants.items()
        },
        "configs": configs,
        "inputs": {
            "calibration": {"path": str(args.calibration.resolve()), "sha256": sha256(args.calibration)},
            "train_meta": {"path": str(args.train_meta.resolve()), "sha256": sha256(args.train_meta)},
            "base_config": {"path": str(args.base_config.resolve()), "sha256": sha256(args.base_config)},
        },
        "outputs": {name: sha256(args.output / name) for name in outputs},
        "test_opened": False,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"candidates": len(variants), "manifest": str(manifest_path)}, sort_keys=True))


if __name__ == "__main__":
    main()
