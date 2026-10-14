#!/usr/bin/env python3
"""Audit whether the preregistered P7 A/B/C smoke comparison is fair."""

from __future__ import annotations

import argparse
import hashlib
import json
import pickle
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml


CSLR_ROOT = Path(__file__).resolve().parents[1]
ASSETS = CSLR_ROOT / "results/phoenix-2014t_ISLR/p7_temporal_debug_smoke_assets_v1"
OUTPUT = CSLR_ROOT / "results/phoenix-2014t_ISLR/p7_abc_fairness_audit_v1"
CANDIDATES = ["A_current_fixed16", "B_aligned_fixed16", "C_inside_even16"]
CONFIGS = {
    name: CSLR_ROOT / "configs" / f"p7_temporal_debug_smoke_{name}.yaml"
    for name in CANDIDATES
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def flatten(value: Any, prefix: str = "") -> dict[str, Any]:
    if isinstance(value, dict):
        result = {}
        for key, child in value.items():
            path = f"{prefix}.{key}" if prefix else str(key)
            result.update(flatten(child, path))
        return result
    if isinstance(value, list):
        return {prefix: value}
    return {prefix: value}


def base(items: list[dict[str, Any]]) -> dict[str, Any]:
    rows = [item for item in items if int(item.get("aug", 0)) == 0]
    if len(rows) != 1:
        raise AssertionError(f"expected one base item, got {len(rows)}")
    return rows[0]


def bag_identity(bags: dict[str, list[dict[str, Any]]]) -> dict[str, tuple[Any, ...]]:
    result = {}
    for key, items in bags.items():
        item = base(items)
        result[str(key)] = (
            str(item["video_file"]), str(item["label"]),
            int(item["base_start"]), int(item["base_end"]),
        )
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args()
    output = args.output.resolve()
    manifest_path = output / "manifest.json"
    if args.verify:
        manifest = json.loads(manifest_path.read_text())
        for name, digest in manifest["outputs"].items():
            assert sha256(output / name) == digest, name
        assert manifest["status"] == "pass"
        print(json.dumps({"verified": True, "manifest": str(manifest_path)}, sort_keys=True))
        return

    configs = {name: yaml.safe_load(path.read_text()) for name, path in CONFIGS.items()}
    flat = {name: flatten(cfg) for name, cfg in configs.items()}
    all_keys = sorted(set().union(*(values.keys() for values in flat.values())))
    differing = {
        key: {name: flat[name].get(key, "<missing>") for name in CANDIDATES}
        for key in all_keys
        if len({json.dumps(flat[name].get(key, "<missing>"), sort_keys=True) for name in CANDIDATES}) > 1
    }
    allowed = {
        "data.train",
        "data.transform_cfg.index_setting",
        "data.align_temp_idx_with_padding",
        "training.model_dir",
        "training.p7_temporal_candidate",
    }
    unexpected = sorted(set(differing) - allowed)

    current_path = ASSETS / "fit_current_bags.pkl"
    inside_path = ASSETS / "fit_inside_only_bags.pkl"
    calibration_path = ASSETS / "calibration_context16.pkl"
    with current_path.open("rb") as handle:
        current = pickle.load(handle)
    with inside_path.open("rb") as handle:
        inside = pickle.load(handle)
    with calibration_path.open("rb") as handle:
        calibration = pickle.load(handle)
    current_identity = bag_identity(current)
    inside_identity = bag_identity(inside)
    identity_match = current_identity == inside_identity
    fit_sources = {row[0] for row in current_identity.values()}
    calibration_sources = {str(item["video_file"]) for item in calibration}

    expected_indices = {
        "A_current_fixed16": ["consecutive", "pad", "central", "pad"],
        "B_aligned_fixed16": ["consecutive", "center_pad", "central", "pad"],
        "C_inside_even16": ["consecutive", "even", "central", "pad"],
    }
    candidate_contract = {}
    for name, cfg in configs.items():
        candidate_contract[name] = {
            "index_setting": cfg["data"]["transform_cfg"]["index_setting"],
            "align_temp_idx_with_padding": cfg["data"].get("align_temp_idx_with_padding", False),
            "random_seed": cfg["training"]["random_seed"],
            "batch_size": cfg["training"]["batch_size"],
            "total_epoch": cfg["training"]["total_epoch"],
            "optimizer": cfg["training"]["optimization"],
            "test_path": cfg["data"]["test"],
        }

    checks = {
        "only_allowed_config_fields_differ": not unexpected,
        "candidate_index_settings_match_contract": all(
            candidate_contract[name]["index_setting"] == expected_indices[name]
            for name in CANDIDATES
        ),
        "A_has_legacy_unaligned_padding": not candidate_contract["A_current_fixed16"]["align_temp_idx_with_padding"],
        "B_and_C_align_temp_idx": all(candidate_contract[name]["align_temp_idx_with_padding"] for name in CANDIDATES[1:]),
        "A_and_B_share_training_asset": configs[CANDIDATES[0]]["data"]["train"] == configs[CANDIDATES[1]]["data"]["train"],
        "C_uses_inside_only_asset": Path(configs[CANDIDATES[2]]["data"]["train"]).resolve() == inside_path.resolve(),
        "A_C_base_bag_identity_exact_match": identity_match,
        "fit_bag_count_equal": len(current) == len(inside) == 16384,
        "C_has_one_base_item_per_bag": all(len(items) == 1 and int(items[0].get("aug", 0)) == 0 for items in inside.values()),
        "fit_and_calibration_sources_disjoint": fit_sources.isdisjoint(calibration_sources),
        "all_test_paths_forbidden": all(cfg["data"]["test"] == "/tmp/P7_TEST_ACCESS_FORBIDDEN" for cfg in configs.values()),
        "output_directories_absent": all(not Path(cfg["training"]["model_dir"]).exists() for cfg in configs.values()),
    }
    status = "pass" if all(checks.values()) else "fail"
    summary = {
        "schema_version": 1,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": status,
        "scope": "source-disjoint train-only A/B/C smoke fairness; dev and test not read",
        "checks": checks,
        "unexpected_config_differences": unexpected,
        "allowed_config_differences": differing,
        "candidate_contract": candidate_contract,
        "asset_statistics": {
            "fit_bags": len(current),
            "fit_sources": len(fit_sources),
            "calibration_items": len(calibration),
            "calibration_sources": len(calibration_sources),
            "current_items_per_bag": dict(sorted(Counter(len(v) for v in current.values()).items())),
            "inside_items_per_bag": dict(sorted(Counter(len(v) for v in inside.values()).items())),
        },
        "inputs": {
            **{str(path.resolve()): sha256(path) for path in CONFIGS.values()},
            str(current_path.resolve()): sha256(current_path),
            str(inside_path.resolve()): sha256(inside_path),
            str(calibration_path.resolve()): sha256(calibration_path),
        },
        "interpretation": (
            "A/B/C have identical model, optimizer, scheduler, seed, batch, epochs and calibration. "
            "Their only substantive differences are the preregistered temporal input construction and "
            "the matching foreground-coordinate alignment flag. C keeps the exact same base bag identities."
        ),
    }
    output.mkdir(parents=True, exist_ok=True)
    summary_path = output / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    manifest = {
        "schema_version": 1,
        "status": status,
        "outputs": {"summary.json": sha256(summary_path)},
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"status": status, "summary": str(summary_path), "checks": checks}, indent=2, sort_keys=True))
    if status != "pass":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
