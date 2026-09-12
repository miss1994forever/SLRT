#!/usr/bin/env python3
"""One-shot dev detection evaluation for the frozen P1 predictor.

The test split is forbidden. Scheduler replay is explicitly skipped when the
train-calibration gate stored in the frozen model is not qualified.
"""

import argparse
import importlib.util
import json
import pickle
from pathlib import Path

import numpy as np
from scipy.special import expit


CSLR_ROOT = Path(__file__).resolve().parents[1]


def import_tool(name):
    path = CSLR_ROOT / f"tools/{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


FEATURES = import_tool("build_phoenix_p1_center_features")
TRAIN = import_tool("train_phoenix_p1_center_predictor")


def reject_test_path(path):
    parts = [part.lower() for part in Path(path).resolve().parts]
    if any(part == "test" or part.startswith("test_") or part.endswith("_test") for part in parts):
        raise ValueError(f"test paths are forbidden: {path}")


def validate_freeze(root):
    freeze_path = Path(root) / "predev_freeze_manifest.json"
    freeze = json.loads(freeze_path.read_text(encoding="utf-8"))
    if freeze.get("status") != "predev_frozen" or freeze.get("test_opened_or_run") is not False:
        raise ValueError("invalid pre-dev freeze status")
    for path_text, record in freeze["outputs"].items():
        path = Path(path_text)
        if path.stat().st_size != record["bytes"] or TRAIN.sha256_file(path) != record["sha256"]:
            raise ValueError(f"frozen output hash mismatch: {path}")
    return freeze


def first_centers(records, video_names, lengths):
    first = {}
    for record in records:
        first.setdefault(int(record["bag"]), record)
    centers = {name: [] for name in video_names}
    segments = {name: [] for name in video_names}
    for _, record in sorted(first.items()):
        name = record["video_file"]
        if name not in centers:
            continue
        start = int(record.get("base_start", record["start"]))
        end = int(record.get("base_end", record["end"]))
        if not 0 <= start < end <= lengths[name]:
            raise ValueError(f"invalid dev segment: {name}")
        segments[name].append(record)
        if record["label"] != "<blank>":
            centers[name].append((start + end - 1) // 2)
    return centers, segments


def evaluate(dev_keypoints, dev_centers, model):
    with Path(dev_keypoints).open("rb") as handle:
        keypoints = pickle.load(handle)
    if not isinstance(keypoints, dict) or any(not str(key).startswith("dev/") for key in keypoints):
        raise ValueError("keypoint mapping is not dev-only")
    with Path(dev_centers).open("rb") as handle:
        records = pickle.load(handle)
    names = list(keypoints)
    lengths = {name: len(keypoints[name]) for name in names}
    centers, segments = first_centers(records, names, lengths)
    matrices, targets = [], []
    scores_by_video = {}
    mean = np.asarray(model["mean_fit_only"], dtype=np.float64)
    scale = np.asarray(model["scale_fit_only"], dtype=np.float64)
    coef = np.asarray(model["coefficients"], dtype=np.float64)
    if list(model["feature_names"]) != list(FEATURES.FEATURE_NAMES):
        raise ValueError("frozen model feature identity differs from implementation")
    for name in names:
        matrix = FEATURES.causal_features(keypoints[name])
        target, rebuilt_centers = FEATURES.midpoint_targets(lengths[name], segments[name])
        if rebuilt_centers.tolist() != centers[name]:
            raise ValueError(f"center reconstruction mismatch: {name}")
        probability = expit(((matrix.astype(np.float64) - mean) / scale) @ coef + model["intercept"])
        matrices.append(matrix)
        targets.append(target)
        scores_by_video[name] = probability
    target = np.concatenate(targets)
    probability = np.concatenate([scores_by_video[name] for name in names])
    frame = TRAIN.binary_metrics(target, probability)
    if model["gate"]["qualified"]:
        event = TRAIN.event_metrics(scores_by_video, centers, model["threshold"])
        scheduler = {"status": "eligible_but_not_implemented_in_detection_evaluator"}
    else:
        event = None
        scheduler = {
            "status": "skipped_no_go",
            "reason": "train-calibration gate failed before dev was opened",
            "uniform_reference": {"wer": 23.058446757405925, "errors": 864},
            "offline_perfect_center_reference": {"wer": 21.537229783827062, "errors": 807},
        }
    return {
        "schema_version": 1,
        "identity_warning": "Targets are alignment-derived midpoint proxies, not manual frame-level ground truth.",
        "selection_policy": "one-shot dev evaluation; no dev value changes model, features, normalization, threshold, event policy, or gate",
        "videos": len(names), "frames": int(sum(lengths.values())),
        "proxy_centers": int(sum(len(value) for value in centers.values())),
        "frame_metrics": frame, "event_metrics_at_frozen_threshold": event,
        "scheduler_replay": scheduler, "test_opened_or_run": False,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dev-keypoints", type=Path, required=True)
    parser.add_argument("--dev-centers", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    for path in (args.dev_keypoints, args.dev_centers, args.output_root):
        reject_test_path(path)
    freeze = validate_freeze(args.output_root)
    model_path = args.output_root / "model.json"
    model = json.loads(model_path.read_text(encoding="utf-8"))
    result = evaluate(args.dev_keypoints, args.dev_centers, model)
    result["predev_freeze_manifest_sha256"] = TRAIN.sha256_file(
        args.output_root / "predev_freeze_manifest.json")
    result["inputs"] = {
        str(path.resolve()): {"bytes": path.stat().st_size, "sha256": TRAIN.sha256_file(path)}
        for path in (args.dev_keypoints, args.dev_centers)
    }
    output = args.output_root / "aggregate/dev_detection_metrics.json"
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
