#!/usr/bin/env python3
"""Build source-disjoint, train-only P7 temporal smoke assets and configs."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import pickle
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Tuple

import yaml


CSLR_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = CSLR_ROOT.parents[1]
DEFAULT_CONFIG = CSLR_ROOT / "configs/phoenix-2014t_ISLR.yaml"
DEFAULT_TRAIN = REPO_ROOT / "data/phoenix_2014t/phoenix_iso_center_label_bag2items.train"
DEFAULT_KEYPOINTS = REPO_ROOT / "data/phoenix_2014t/keypoints_hrnet_dark_coco_wholebody_iso.train.finite.pkl"
DEFAULT_OUTPUT = CSLR_ROOT / "results/phoenix-2014t_ISLR/p7_temporal_debug_smoke_assets_v1"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def stable_hash(seed: int, value: str) -> str:
    return hashlib.sha256(f"{seed}|{value}".encode()).hexdigest()


def get_base(items: List[Dict[str, Any]]) -> Dict[str, Any]:
    bases = [x for x in items if int(x.get("aug", 0)) == 0]
    if len(bases) != 1:
        raise AssertionError(len(bases))
    return bases[0]


def duration_bucket(duration: int) -> str:
    if duration <= 6:
        return "1-6"
    if duration <= 9:
        return "7-9"
    if duration <= 15:
        return "10-15"
    return "16+"


def source_split(bags: Dict[str, List[Dict[str, Any]]], seed: int, calibration_percent: int) -> Tuple[set, set]:
    source_labels = defaultdict(set)
    for items in bags.values():
        base = get_base(items)
        source_labels[str(base["video_file"])].add(str(base["label"]))
    calibration = {
        source for source in source_labels
        if int(stable_hash(seed, source)[:8], 16) % 100 < calibration_percent
    }
    fit = set(source_labels) - calibration
    all_labels = set().union(*source_labels.values())
    fit_labels = set().union(*(source_labels[s] for s in fit))
    for label in sorted(all_labels - fit_labels):
        if label in fit_labels:
            continue
        candidates = sorted((s for s in calibration if label in source_labels[s]), key=lambda s: stable_hash(seed + 1, s))
        if not candidates:
            raise RuntimeError(f"cannot place {label} in fit")
        moved = candidates[0]
        calibration.remove(moved)
        fit.add(moved)
        fit_labels.update(source_labels[moved])
    return fit, calibration


def choose_smoke_bags(
    bags: Dict[str, List[Dict[str, Any]]], fit_sources: set, target: int, seed: int
) -> Dict[str, List[Dict[str, Any]]]:
    eligible = [(str(k), v) for k, v in bags.items() if str(get_base(v)["video_file"]) in fit_sources]
    by_label = defaultdict(list)
    for key, items in eligible:
        by_label[str(get_base(items)["label"])].append((key, items))
    chosen: Dict[str, List[Dict[str, Any]]] = {}
    # Preserve at least one bag for every fit label before hash filling.
    for label, rows in sorted(by_label.items()):
        key, items = min(rows, key=lambda x: stable_hash(seed, x[0]))
        chosen[key] = items
    for key, items in sorted(eligible, key=lambda x: stable_hash(seed + 2, x[0])):
        if len(chosen) >= target:
            break
        chosen.setdefault(key, items)
    if len(chosen) != target:
        raise RuntimeError(f"requested {target}, got {len(chosen)}")
    return chosen


def closest_context(base: Dict[str, Any], items: List[Dict[str, Any]]) -> Dict[str, Any]:
    midpoint = (int(base["base_start"]) + int(base["base_end"]) - 1) / 2
    augmented = [x for x in items if int(x.get("aug", 0)) == 1]
    return min(augmented, key=lambda x: abs(((int(x["start"]) + int(x["end"]) - 1) / 2) - midpoint))


def choose_calibration(
    bags: Dict[str, List[Dict[str, Any]]], calibration_sources: set, per_bucket: int, blank_count: int, seed: int
) -> List[Dict[str, Any]]:
    grouped = defaultdict(list)
    blanks = []
    for items in bags.values():
        base = get_base(items)
        if str(base["video_file"]) not in calibration_sources:
            continue
        if str(base["label"]) == "<blank>":
            blanks.append((base, items))
        else:
            duration = int(base["base_end"]) - int(base["base_start"])
            grouped[duration_bucket(duration)].append((base, items))
    selected = []
    for bucket_name in ["1-6", "7-9", "10-15", "16+"]:
        rows = sorted(grouped[bucket_name], key=lambda x: stable_hash(seed, x[0]["name"]))[:per_bucket]
        if len(rows) != per_bucket:
            raise RuntimeError((bucket_name, len(rows)))
        selected.extend(rows)
    selected.extend(sorted(blanks, key=lambda x: stable_hash(seed, x[0]["name"]))[:blank_count])
    output = []
    for idx, (base, items) in enumerate(selected):
        item = copy.deepcopy(closest_context(base, items))
        item["name"] = f"p7cal_{idx:05d}_{item['name']}"
        output.append(item)
    return output


def absolutize_config_paths(cfg: Dict[str, Any]) -> None:
    for key in ["zip_file", "vocab_file"]:
        cfg["data"][key] = str((CSLR_ROOT / cfg["data"][key]).resolve())
    rn = cfg["model"]["RecognitionNetwork"]
    rn["GlossTokenizer"]["gloss2id_file"] = str((CSLR_ROOT / rn["GlossTokenizer"]["gloss2id_file"]).resolve())
    for stream in ["s3d", "keypoint_s3d"]:
        rn[stream]["pretrained_ckpt"] = str((CSLR_ROOT / rn[stream]["pretrained_ckpt"]).resolve())


def make_config(
    base_cfg: Dict[str, Any], candidate: str, train_path: Path, calibration_path: Path,
    keypoints: Path, epochs: int, workers: int, output_root: Path
) -> Dict[str, Any]:
    cfg = copy.deepcopy(base_cfg)
    absolutize_config_paths(cfg)
    cfg["data"]["train"] = str(train_path.resolve())
    cfg["data"]["dev"] = str(calibration_path.resolve())
    cfg["data"]["test"] = "/tmp/P7_TEST_ACCESS_FORBIDDEN"
    cfg["data"]["keypoint_file"] = str(keypoints.resolve())
    # The calibration examples are held out from the training split, so their
    # source sentence lengths also come from train metadata.
    cfg["data"]["raw_vlens_split_overrides"] = {"dev": "train"}
    cfg["training"]["model_dir"] = str((output_root / candidate).resolve())
    cfg["training"]["overwrite"] = True
    cfg["training"]["total_epoch"] = epochs
    cfg["training"]["num_workers"] = workers
    cfg["training"]["disable_wandb"] = True
    cfg["training"]["keep_last_ckpts"] = epochs + 1
    # Keep the baseline scheduler horizon.  Setting t_max equal to a short
    # smoke epoch count makes the final smoke epoch have zero learning rate
    # and is not an equal-budget proxy for the 100-epoch baseline.
    cfg["training"]["random_seed"] = 73031
    cfg["training"].pop("debug", None)
    cfg["training"]["p7_temporal_candidate"] = candidate
    cfg["data"].pop("align_temp_idx_with_padding", None)
    if candidate == "A_current_fixed16":
        cfg["data"]["transform_cfg"]["index_setting"] = ["consecutive", "pad", "central", "pad"]
    elif candidate == "B_aligned_fixed16":
        cfg["data"]["transform_cfg"]["index_setting"] = ["consecutive", "center_pad", "central", "pad"]
        cfg["data"]["align_temp_idx_with_padding"] = True
    elif candidate == "C_inside_even16":
        cfg["data"]["transform_cfg"]["index_setting"] = ["consecutive", "even", "central", "pad"]
        cfg["data"]["align_temp_idx_with_padding"] = True
    else:
        raise ValueError(candidate)
    return cfg


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    p.add_argument("--train", type=Path, default=DEFAULT_TRAIN)
    p.add_argument("--keypoints", type=Path, default=DEFAULT_KEYPOINTS)
    p.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    p.add_argument("--fit-bags", type=int, default=16384)
    p.add_argument("--calibration-percent", type=int, default=20)
    p.add_argument("--calibration-per-duration-bucket", type=int, default=256)
    p.add_argument("--calibration-blanks", type=int, default=256)
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--seed", type=int, default=73030)
    p.add_argument("--verify", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    args.output = args.output.resolve()
    manifest_path = args.output / "manifest.json"
    if args.verify:
        manifest = json.loads(manifest_path.read_text())
        for name, digest in manifest["outputs"].items():
            assert sha256(args.output / name) == digest, name
        for candidate, record in manifest["configs"].items():
            path = Path(record["path"])
            assert sha256(path) == record["sha256"], candidate
        assert set(manifest["fit_sources"]).isdisjoint(manifest["calibration_sources"])
        assert manifest["test_policy"] == "forbidden"
        print(json.dumps({"verified": True, "manifest": str(manifest_path)}, sort_keys=True))
        return

    args.output.mkdir(parents=True, exist_ok=True)
    with args.train.open("rb") as f:
        bags = pickle.load(f)
    fit_sources, calibration_sources = source_split(bags, args.seed, args.calibration_percent)
    smoke = choose_smoke_bags(bags, fit_sources, args.fit_bags, args.seed)
    calibration = choose_calibration(
        bags, calibration_sources, args.calibration_per_duration_bucket,
        args.calibration_blanks, args.seed
    )
    inside = {k: [copy.deepcopy(get_base(v))] for k, v in smoke.items()}

    current_path = args.output / "fit_current_bags.pkl"
    inside_path = args.output / "fit_inside_only_bags.pkl"
    calibration_path = args.output / "calibration_context16.pkl"
    for path, obj in [(current_path, smoke), (inside_path, inside), (calibration_path, calibration)]:
        with path.open("wb") as f:
            pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)

    base_cfg = yaml.safe_load(args.config.read_text())
    config_names = {}
    output_root = CSLR_ROOT / "results/phoenix-2014t_ISLR/p7_temporal_smoke_v1"
    for candidate, train_path in [
        ("A_current_fixed16", current_path),
        ("B_aligned_fixed16", current_path),
        ("C_inside_even16", inside_path),
    ]:
        cfg = make_config(
            base_cfg, candidate, train_path, calibration_path, args.keypoints,
            args.epochs, args.workers, output_root
        )
        config_path = CSLR_ROOT / "configs" / f"p7_temporal_debug_smoke_{candidate}.yaml"
        config_path.write_text(yaml.safe_dump(cfg, sort_keys=False))
        config_names[candidate] = {
            "path": str(config_path.resolve()),
            "sha256": sha256(config_path),
        }

    fit_labels = Counter(str(get_base(v)["label"]) for v in smoke.values())
    calibration_labels = Counter(str(x["label"]) for x in calibration)
    outputs = [current_path.name, inside_path.name, calibration_path.name]
    manifest = {
        "schema_version": 1,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "source-disjoint train-only smoke; no dev/test opened",
        "test_policy": "forbidden",
        "seed": args.seed,
        "fit_sources": sorted(fit_sources),
        "calibration_sources": sorted(calibration_sources),
        "fit_bags": len(smoke),
        "calibration_items": len(calibration),
        "fit_label_count": len(fit_labels),
        "calibration_label_count": len(calibration_labels),
        "configs": config_names,
        "inputs": {
            "base_config": {"path": str(args.config.resolve()), "sha256": sha256(args.config)},
            "train_annotations": {"path": str(args.train.resolve()), "sha256": sha256(args.train)},
            "train_keypoints": {"path": str(args.keypoints.resolve()), "sha256": sha256(args.keypoints)},
        },
        "outputs": {name: sha256(args.output / name) for name in outputs},
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"manifest": str(manifest_path), "configs": config_names}, sort_keys=True))


if __name__ == "__main__":
    main()
