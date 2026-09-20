#!/usr/bin/env python3
"""Generate resumable Phoenix train stride-1 logits without touching test."""

import argparse
import gzip
import hashlib
import json
import os
import pickle
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import yaml


CSLR_ROOT = Path(__file__).resolve().parents[1]
SLRT_ROOT = CSLR_ROOT.parents[1]
DEFAULT_CONFIG = CSLR_ROOT / "configs/slide_phoenix-2014t.yaml"
DEFAULT_META = SLRT_ROOT / "data/phoenix_2014t/phoenix14t.train"
DEFAULT_KEYPOINTS = (
    SLRT_ROOT / "data/phoenix_2014t/keypoints_hrnet_dark_coco_wholebody_iso.train.pkl"
)
DEFAULT_CHECKPOINT = CSLR_ROOT / "results/phoenix-2014t_ISLR/ckpts/best.ckpt"
DEFAULT_VOCAB = SLRT_ROOT / "data/phoenix_2014t/phoenix_iso_with_blank.vocab"
DEFAULT_VIDEO_ZIP = SLRT_ROOT / "data/phoenix_2014t/PHOENIX2014T_videos.zip"
DEFAULT_KEYPOINT_SPLIT_MANIFEST = (
    SLRT_ROOT
    / "data/phoenix_2014t/keypoints_hrnet_dark_coco_wholebody_iso.split_manifest.json"
)
DEFAULT_OUTPUT = (
    CSLR_ROOT / "results/phoenix-2014t_ISLR/train_dense_stride1_v1_49faacc3"
)
FAULTY_GPU_UUIDS = {
    "GPU-dbd35875-dfa5-43f1-0cf0-f88ccb529c8a",
    "GPU-06afe121-c4ce-b981-bb86-399e4a85ae83",
}


def sha256_file(path, chunk_size=8 * 1024 * 1024):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_train_meta(path):
    lowered = [part.lower() for part in Path(path).resolve().parts]
    if any(part == "test" or part.startswith("test_") or part.endswith("_test") for part in lowered):
        raise ValueError("test paths are forbidden")
    with gzip.open(path, "rb") as handle:
        records = pickle.load(handle)
    if not records or any(not str(row["name"]).startswith("train/") for row in records):
        raise ValueError("metadata is not an unambiguous train split")
    return records


def source_video_id(name):
    stem, separator, suffix = name.rpartition("-")
    return stem if separator and suffix.isdigit() else name


def build_fit_calibration(records, calibration_fraction=0.1, seed=260916):
    groups = {}
    for row in records:
        groups.setdefault(source_video_id(row["name"]), []).append(row["name"])
    ordered = sorted(
        groups,
        key=lambda group: hashlib.sha256(f"{seed}:{group}".encode("utf-8")).hexdigest(),
    )
    target = round(len(records) * calibration_fraction)
    calibration_groups = set()
    count = 0
    for group in ordered:
        if count >= target:
            break
        calibration_groups.add(group)
        count += len(groups[group])
    assignment = {
        name: ("calibration" if group in calibration_groups else "fit")
        for group, names in groups.items()
        for name in names
    }
    return {
        "schema_version": 1,
        "unit": "source_video_id obtained by removing the final numeric segment suffix",
        "seed": seed,
        "target_calibration_fraction": calibration_fraction,
        "samples": {
            "fit": sum(value == "fit" for value in assignment.values()),
            "calibration": sum(value == "calibration" for value in assignment.values()),
        },
        "source_videos": {
            "fit": sum(group not in calibration_groups for group in groups),
            "calibration": len(calibration_groups),
        },
        "assignments": assignment,
    }


def validate_gpu_uuid(value):
    if not value or "," in value or not value.startswith("GPU-"):
        raise ValueError("provide exactly one GPU UUID")
    if value in FAULTY_GPU_UUIDS:
        raise ValueError("refusing a known faulty GPU")


def validate_completed_shard(shard_dir, expected_records):
    results_path = shard_dir / "train_results.pkl"
    logits_path = shard_dir / "train_logits.pkl"
    runtime_path = shard_dir / "train_runtime_profile.json"
    if not all(path.is_file() for path in (results_path, logits_path, runtime_path)):
        raise FileNotFoundError("completed shard lacks required artifacts")
    with results_path.open("rb") as handle:
        results = pickle.load(handle)
    with logits_path.open("rb") as handle:
        logits = pickle.load(handle)
    expected_names = [row["name"] for row in expected_records]
    if list(results) != expected_names or list(logits) != expected_names:
        raise ValueError("shard sample order differs from train metadata")
    for row in expected_records:
        name = row["name"]
        if len(logits[name]) != int(row["num_frames"]):
            raise ValueError(f"dense logit length mismatch: {name}")
        starts = [int(item["start"]) for item in results[name]["adaptive_stride_metadata"]]
        if starts != list(range(int(row["num_frames"]))):
            raise ValueError(f"non-dense coordinates: {name}")
        if results[name]["gls_ref"] != row["gloss"]:
            raise ValueError(f"reference mismatch: {name}")
    return {
        "samples": len(expected_records),
        "windows": sum(int(row["num_frames"]) for row in expected_records),
        "artifacts": {
            path.name: {"bytes": path.stat().st_size, "sha256": sha256_file(path)}
            for path in (results_path, logits_path, runtime_path)
        },
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpu-uuid", required=True)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--train-meta", type=Path, default=DEFAULT_META)
    parser.add_argument("--train-keypoints", type=Path, default=DEFAULT_KEYPOINTS)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--shard-samples", type=int, default=128)
    parser.add_argument("--start-shard", type=int, default=0)
    parser.add_argument("--stop-shard", type=int)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    validate_gpu_uuid(args.gpu_uuid)
    if args.shard_samples < 1 or args.start_shard < 0:
        parser.error("invalid shard bounds")
    paths = [
        args.config,
        args.train_meta,
        args.train_keypoints,
        args.checkpoint,
        DEFAULT_VOCAB,
        DEFAULT_VIDEO_ZIP,
        DEFAULT_KEYPOINT_SPLIT_MANIFEST,
    ]
    for path in paths:
        if not Path(path).is_file():
            raise FileNotFoundError(path)
    records = load_train_meta(args.train_meta)
    shard_count = (len(records) + args.shard_samples - 1) // args.shard_samples
    stop = shard_count if args.stop_shard is None else min(args.stop_shard, shard_count)
    if args.start_shard >= stop:
        raise ValueError("empty shard range")
    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    model_dir = output_root / "model"
    checkpoint_dir = model_dir / "ckpts"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_link = checkpoint_dir / "best.ckpt"
    if not checkpoint_link.exists():
        checkpoint_link.symlink_to(args.checkpoint.resolve())
    elif checkpoint_link.resolve() != args.checkpoint.resolve():
        raise ValueError("checkpoint link targets a different model")

    split_manifest_path = output_root / "fit_calibration_split.json"
    split_manifest = build_fit_calibration(records)
    rendered_split = json.dumps(split_manifest, ensure_ascii=False, indent=2) + "\n"
    if split_manifest_path.exists() and split_manifest_path.read_text(encoding="utf-8") != rendered_split:
        raise ValueError("existing fit/calibration split differs")
    split_manifest_path.write_text(rendered_split, encoding="utf-8")

    manifest_path = output_root / "protocol_manifest.json"
    manifest = {
        "manifest_version": 1,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "split": "train",
        "git_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=CSLR_ROOT, text=True
        ).strip(),
        "gpu_uuid": args.gpu_uuid,
        "shard_samples": args.shard_samples,
        "samples": len(records),
        "windows": sum(int(row["num_frames"]) for row in records),
        "inputs": {
            str(Path(path).resolve()): {
                "bytes": Path(path).stat().st_size,
                "sha256": sha256_file(path),
            }
            for path in [Path(__file__), *paths]
        },
        "fit_calibration_split": {
            "path": str(split_manifest_path),
            "sha256": sha256_file(split_manifest_path),
        },
    }
    if manifest_path.exists():
        existing = json.loads(manifest_path.read_text(encoding="utf-8"))
        for key in ("split", "git_commit", "shard_samples", "samples", "windows", "inputs"):
            if existing[key] != manifest[key]:
                raise ValueError(f"existing protocol differs at {key}")
        manifest = existing
    else:
        manifest_path.write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )

    base_config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    base_config["data"]["keypoint_file"] = str(args.train_keypoints.resolve())
    base_config["data"]["sampling"] = {"mode": "fixed", "fixed_stride": 1}
    base_config["data"]["stride"] = 1
    base_config["data"]["adaptive_stride"]["enabled"] = False
    base_config.setdefault("postprocess", {}).setdefault("span_weighted_voting", {}).update(
        {"enabled": True, "vote_span_frames": 15, "kernel": "triangular", "min_weight": 0.05}
    )
    base_config["runtime_profile"] = {"enabled": True}
    base_config["training"]["model_dir"] = str(model_dir.resolve())
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = args.gpu_uuid
    env.setdefault("OMP_NUM_THREADS", "1")
    env.setdefault("PYTHONUNBUFFERED", "1")

    for shard_index in range(args.start_shard, stop):
        left = shard_index * args.shard_samples
        right = min(len(records), left + args.shard_samples)
        shard_records = records[left:right]
        shard_dir = output_root / "shards" / f"shard-{shard_index:05d}-of-{shard_count:05d}"
        complete_path = shard_dir / "complete.json"
        if complete_path.is_file() and args.resume:
            validation = validate_completed_shard(shard_dir, shard_records)
            print(f"skip validated complete shard {shard_index + 1}/{shard_count}: {validation}")
            continue
        if shard_dir.exists() and any(shard_dir.iterdir()):
            raise FileExistsError(f"refusing non-empty incomplete shard: {shard_dir}")
        shard_dir.mkdir(parents=True, exist_ok=True)
        metadata_path = shard_dir / "train_metadata.pkl.gz"
        with gzip.open(metadata_path, "wb", compresslevel=6) as handle:
            pickle.dump(shard_records, handle, protocol=pickle.HIGHEST_PROTOCOL)
        config = dict(base_config)
        config["data"] = dict(base_config["data"])
        config["data"]["train"] = str(metadata_path.resolve())
        config_path = shard_dir / "resolved_config.yaml"
        config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
        command = [
            sys.executable,
            "prediction_slide.py",
            "--config",
            str(config_path),
            "--output_dir",
            str(shard_dir),
            "--split",
            "train",
            "--pred_src",
            "ensemble",
            "--sampling_mode",
            "fixed",
            "--fixed_stride",
            "1",
            "--span_weighted_voting",
            "1",
            "--vote_span_frames",
            "15",
            "--span_min_weight",
            "0.05",
            "--save_fea",
            "0",
            "--eval_setting",
            f"train_dense_shard_{shard_index:05d}",
        ]
        (shard_dir / "command.json").write_text(
            json.dumps(command, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        started = time.perf_counter()
        with (shard_dir / "stdout.log").open("w", encoding="utf-8") as output:
            process = subprocess.run(
                command, cwd=CSLR_ROOT, env=env, stdout=output, stderr=subprocess.STDOUT
            )
        elapsed = time.perf_counter() - started
        if process.returncode != 0:
            raise RuntimeError(f"shard {shard_index} failed; see {shard_dir / 'stdout.log'}")
        validation = validate_completed_shard(shard_dir, shard_records)
        completion = {
            "shard_index": shard_index,
            "sample_range": [left, right],
            "wall_seconds": elapsed,
            **validation,
        }
        complete_path.write_text(
            json.dumps(completion, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        print(f"completed shard {shard_index + 1}/{shard_count}: {completion}", flush=True)


if __name__ == "__main__":
    main()
