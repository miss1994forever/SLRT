#!/usr/bin/env python3
"""Build a fixed, train-only continuous WER screen for P7 recognizers."""

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
ASSETS = CSLR_ROOT / "results/phoenix-2014t_ISLR/p7_temporal_debug_smoke_assets_v1"
OUTPUT = CSLR_ROOT / "results/phoenix-2014t_ISLR/p7_temporal_continuous_smoke_assets_v1"
CANDIDATES = ["A_current_fixed16", "B_aligned_fixed16", "C_inside_even16"]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def stable(seed: int, value: str) -> str:
    return hashlib.sha256(f"{seed}|{value}".encode()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-manifest", type=Path, default=ASSETS / "manifest.json")
    parser.add_argument("--continuous-train", type=Path, default=REPO_ROOT / "data/phoenix_2014t/phoenix14t.train")
    parser.add_argument("--slide-config", type=Path, default=CSLR_ROOT / "configs/slide_phoenix-2014t.yaml")
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--sentences", type=int, default=256)
    parser.add_argument("--seed", type=int, default=73033)
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
    source_manifest = json.loads(args.source_manifest.read_text())
    allowed = set(source_manifest["calibration_sources"])
    with gzip.open(args.continuous_train, "rb") as handle:
        annotations = pickle.load(handle)
    eligible = [item for item in annotations if str(item["name"]) in allowed]
    selected = sorted(eligible, key=lambda item: stable(args.seed, str(item["name"])))[: args.sentences]
    if len(selected) != args.sentences:
        raise RuntimeError((len(selected), args.sentences))
    annotation_path = args.output / "calibration_sentences_256.train.gz"
    with gzip.open(annotation_path, "wb") as handle:
        pickle.dump(selected, handle, protocol=pickle.HIGHEST_PROTOCOL)

    slide = yaml.safe_load(args.slide_config.read_text())
    slide["data"]["train"] = str(annotation_path.resolve())
    slide["data"]["dev"] = "/tmp/P7_DEV_ACCESS_FORBIDDEN"
    slide["data"]["test"] = "/tmp/P7_TEST_ACCESS_FORBIDDEN"
    slide["data"]["keypoint_file"] = str((REPO_ROOT / "data/phoenix_2014t/keypoints_hrnet_dark_coco_wholebody_iso.train.finite.pkl").resolve())
    for key in ["zip_file", "vocab_file"]:
        slide["data"][key] = str((CSLR_ROOT / slide["data"][key]).resolve())
    configs = {}
    for candidate in CANDIDATES:
        train_config = CSLR_ROOT / "configs" / f"p7_temporal_debug_smoke_{candidate}.yaml"
        trained = yaml.safe_load(train_config.read_text())
        cfg = copy.deepcopy(slide)
        cfg["model"] = trained["model"]
        cfg["training"]["model_dir"] = str(
            (CSLR_ROOT / "results/phoenix-2014t_ISLR/p7_temporal_continuous_smoke_v1" / candidate).resolve()
        )
        cfg["training"]["disable_wandb"] = True
        # Record synchronized model-forward time and peak CUDA memory for the
        # deployment cost comparison.  This does not change logits or decoding.
        cfg["runtime_profile"] = {"enabled": True}
        config_path = CSLR_ROOT / "configs" / f"p7_temporal_continuous_smoke_{candidate}.yaml"
        config_path.write_text(yaml.safe_dump(cfg, sort_keys=False))
        configs[candidate] = {"path": str(config_path.resolve()), "sha256": sha256(config_path)}

    outputs = [annotation_path.name]
    manifest = {
        "schema_version": 1,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "256 deterministic source-disjoint train sentences; official dev/test forbidden",
        "seed": args.seed,
        "sentence_names": [str(item["name"]) for item in selected],
        "reference_gloss_tokens": sum(len(str(item["gloss"]).split()) for item in selected),
        "configs": configs,
        "inputs": {
            "source_manifest": {"path": str(args.source_manifest.resolve()), "sha256": sha256(args.source_manifest)},
            "continuous_train": {"path": str(args.continuous_train.resolve()), "sha256": sha256(args.continuous_train)},
            "slide_config": {"path": str(args.slide_config.resolve()), "sha256": sha256(args.slide_config)},
        },
        "outputs": {name: sha256(args.output / name) for name in outputs},
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"sentences": len(selected), "manifest": str(manifest_path)}, sort_keys=True))


if __name__ == "__main__":
    main()
