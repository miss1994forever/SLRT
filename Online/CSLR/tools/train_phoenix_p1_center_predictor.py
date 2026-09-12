#!/usr/bin/env python3
"""Fit and freeze the P1 train-only causal center predictor.

No dev or test path is accepted. The fixed model is a standardized,
class-balanced logistic regression with C=1 (L2 coefficient 1 in the
equivalent summed-loss objective), fitted by deterministic CPU L-BFGS.
"""

import argparse
import hashlib
import json
import pickle
import platform
from pathlib import Path

import numpy as np
import scipy
from scipy.optimize import minimize
from scipy.special import expit


THRESHOLDS = np.arange(0.01, 1.0, 0.01, dtype=np.float64)
REFRACTORY = 4
EVENT_TOLERANCE = 1


def reject_dev_or_test_path(path):
    parts = [part.lower() for part in Path(path).resolve().parts]
    forbidden = ("dev", "test")
    if any(part == word or part.startswith(word + "_") or part.endswith("_" + word)
           for part in parts for word in forbidden):
        raise ValueError(f"dev/test paths are forbidden before freeze: {path}")


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def binary_metrics(target, probability, bins=15):
    target = np.asarray(target, dtype=np.uint8)
    probability = np.asarray(probability, dtype=np.float64)
    positives = int(target.sum())
    negatives = len(target) - positives
    order = np.argsort(probability, kind="mergesort")
    ranks = np.empty(len(order), dtype=np.float64)
    ranks[order] = np.arange(1, len(order) + 1)
    # Average ranks for tied probabilities.
    sorted_probability = probability[order]
    begin = 0
    while begin < len(order):
        end = begin + 1
        while end < len(order) and sorted_probability[end] == sorted_probability[begin]:
            end += 1
        ranks[order[begin:end]] = 0.5 * (begin + 1 + end)
        begin = end
    auroc = (ranks[target == 1].sum() - positives * (positives + 1) / 2) / (positives * negatives)
    descending = np.argsort(-probability, kind="mergesort")
    ordered_target = target[descending]
    precision = np.cumsum(ordered_target) / np.arange(1, len(target) + 1)
    auprc = precision[ordered_target == 1].sum() / positives
    brier = np.mean((probability - target) ** 2)
    ece = 0.0
    bin_rows = []
    edges = np.linspace(0.0, 1.0, bins + 1)
    for index in range(bins):
        selected = ((probability >= edges[index]) &
                    (probability < edges[index + 1] if index + 1 < bins else probability <= 1.0))
        count = int(selected.sum())
        if count:
            confidence = float(probability[selected].mean())
            frequency = float(target[selected].mean())
            ece += count / len(target) * abs(confidence - frequency)
            bin_rows.append({"left": float(edges[index]), "right": float(edges[index + 1]),
                             "count": count, "mean_probability": confidence,
                             "positive_fraction": frequency})
    return {"rows": len(target), "positives": positives,
            "positive_fraction": positives / len(target), "auroc": float(auroc),
            "auprc": float(auprc), "random_auprc": positives / len(target),
            "brier": float(brier), "ece_15_equal_width": float(ece), "ece_bins": bin_rows}


def fit_logistic(features, target, l2=1.0, max_iterations=200):
    features = np.asarray(features, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    count = len(target)
    positives = target.sum()
    sample_weight = np.where(target == 1, count / (2.0 * positives),
                             count / (2.0 * (count - positives)))

    def objective(parameters):
        coef, intercept = parameters[:-1], parameters[-1]
        logits = features @ coef + intercept
        loss = np.sum(sample_weight * (np.logaddexp(0.0, logits) - target * logits)) / count
        loss += 0.5 * l2 * np.dot(coef, coef) / count
        residual = sample_weight * (expit(logits) - target) / count
        gradient = np.concatenate([features.T @ residual + l2 * coef / count,
                                   [residual.sum()]])
        return float(loss), gradient

    initial = np.zeros(features.shape[1] + 1, dtype=np.float64)
    result = minimize(objective, initial, method="L-BFGS-B", jac=True,
                      options={"maxiter": max_iterations, "ftol": 1e-12, "gtol": 1e-8,
                               "maxls": 40})
    if not result.success:
        raise RuntimeError(f"L-BFGS failed: {result.message}")
    return result.x[:-1], float(result.x[-1]), {
        "success": bool(result.success), "message": str(result.message),
        "iterations": int(result.nit), "function_evaluations": int(result.nfev),
        "final_objective": float(result.fun),
    }


def events_from_scores(scores, threshold, refractory=REFRACTORY):
    scores = np.asarray(scores)
    above = scores >= threshold
    crossings = np.flatnonzero(above & np.concatenate(([True], ~above[:-1])))
    events = []
    for frame in crossings:
        if not events or int(frame) - events[-1] >= refractory:
            events.append(int(frame))
    return events


def event_metrics(scores_by_video, centers_by_video, threshold, tolerance=EVENT_TOLERANCE,
                  refractory=REFRACTORY):
    true_count = event_count = hit_true = hit_events = 0
    signed_errors = []
    for name, scores in scores_by_video.items():
        centers = np.asarray(centers_by_video[name], dtype=np.int64)
        events = np.asarray(events_from_scores(scores, threshold, refractory), dtype=np.int64)
        true_count += len(centers)
        event_count += len(events)
        if len(events) and len(centers):
            distance = np.abs(centers[:, None] - events[None, :])
            nearest_event = distance.argmin(axis=1)
            matched_true = distance[np.arange(len(centers)), nearest_event] <= tolerance
            hit_true += int(matched_true.sum())
            signed_errors.extend((events[nearest_event[matched_true]] - centers[matched_true]).tolist())
            hit_events += int((distance.min(axis=0) <= tolerance).sum())
    recall = hit_true / true_count if true_count else 0.0
    precision = hit_events / event_count if event_count else 0.0
    errors = np.asarray(signed_errors, dtype=np.int64)
    return {
        "threshold": float(threshold), "refractory_frames": int(refractory),
        "matching_tolerance_frames": int(tolerance), "true_centers": int(true_count),
        "emitted_events": int(event_count), "centers_hit": int(hit_true),
        "events_hitting_center": int(hit_events), "center_recall": float(recall),
        "event_precision": float(precision),
        "event_f1": float(2 * recall * precision / (recall + precision)) if recall + precision else 0.0,
        "matched_signed_error_mean": float(errors.mean()) if len(errors) else None,
        "matched_absolute_error_mean": float(np.abs(errors).mean()) if len(errors) else None,
        "positive_delay_fraction": float(np.mean(errors > 0)) if len(errors) else None,
        "zero_delay_fraction": float(np.mean(errors == 0)) if len(errors) else None,
        "negative_delay_fraction": float(np.mean(errors < 0)) if len(errors) else None,
    }


def centers_from_bags(path, names):
    with Path(path).open("rb") as handle:
        bags = pickle.load(handle)
    result = {name: [] for name in names}
    for records in bags.values():
        base = [item for item in records if int(item.get("aug", -1)) == 0]
        if len(base) != 1:
            raise ValueError("every bag must have exactly one base record")
        item = base[0]
        name = item["video_file"]
        if name in result and item["label"] != "<blank>":
            result[name].append((int(item["start"]) + int(item["end"]) - 1) // 2)
    for name in result:
        result[name].sort()
    return result


def score_by_video(probability, video_names, selected_names):
    selected_names = set(selected_names)
    result = {}
    for name in sorted(selected_names):
        result[name] = probability[video_names == name]
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--split", type=Path, required=True)
    parser.add_argument("--train-bags", type=Path, required=True)
    parser.add_argument("--train-keypoints", type=Path, required=True)
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    for path in (args.features, args.split, args.train_bags, args.train_keypoints,
                 args.split_manifest, args.output_root):
        reject_dev_or_test_path(path)
    data = np.load(args.features)
    features = data["features"]
    target = data["targets"]
    video_names = data["video_names"]
    feature_names = data["feature_names"].tolist()
    split = json.loads(args.split.read_text(encoding="utf-8"))
    fit_names, calibration_names = split["fit_videos"], split["calibration_videos"]
    if len(fit_names) != 5655 or len(calibration_names) != 1440:
        raise ValueError("frozen video split sizes changed")
    fit_mask = np.isin(video_names, fit_names)
    calibration_mask = np.isin(video_names, calibration_names)
    if np.any(fit_mask & calibration_mask) or not np.all(fit_mask | calibration_mask):
        raise ValueError("feature rows do not form the frozen fit/calibration partition")
    mean = features[fit_mask].mean(axis=0, dtype=np.float64)
    scale = features[fit_mask].std(axis=0, dtype=np.float64)
    scale[scale == 0] = 1.0
    standardized = (features.astype(np.float64) - mean) / scale
    coef, intercept, optimizer = fit_logistic(standardized[fit_mask], target[fit_mask])
    probability = expit(standardized @ coef + intercept)
    centers = centers_from_bags(args.train_bags, set(fit_names) | set(calibration_names))
    calibration_scores = score_by_video(probability, video_names, calibration_names)
    threshold_rows = [event_metrics(calibration_scores, centers, threshold)
                      for threshold in THRESHOLDS]
    feasible = [row for row in threshold_rows if row["center_recall"] >= 0.75 and
                row["positive_delay_fraction"] is not None and
                row["positive_delay_fraction"] <= 0.25]
    selected = max(feasible, key=lambda row: (row["threshold"], row["event_precision"],
                                               -row["threshold"])) if feasible else None
    gate = {
        "qualified": selected is not None,
        "requirements": {"center_recall_minimum": 0.75,
                         "positive_delay_fraction_maximum": 0.25},
        "selection": "highest threshold on fixed 0.01..0.99 grid satisfying both; precision tie-break",
        "selected_threshold": selected["threshold"] if selected else None,
        "reason": "qualified" if selected else "no calibration threshold satisfied both requirements",
    }
    model = {
        "schema_version": 1, "model": "standardized_logistic_regression",
        "feature_names": feature_names, "mean_fit_only": mean.tolist(),
        "scale_fit_only": scale.tolist(), "coefficients": coef.tolist(),
        "intercept": intercept, "class_weight": "balanced from fit rows only",
        "regularization": {"C": 1.0, "summed_loss_l2_coefficient": 1.0,
                           "normalized_objective_l2_coefficient": 1.0 / int(fit_mask.sum()),
                           "intercept_penalized": False},
        "optimizer": optimizer, "execution_device": "cpu",
        "threshold": selected["threshold"] if selected else None,
        "threshold_grid": {"start": 0.01, "stop_inclusive": 0.99, "step": 0.01},
        "event_policy": "upward probability threshold crossing, then 4-frame refractory",
        "refractory_frames": REFRACTORY, "gate": gate,
    }
    fit_scores = score_by_video(probability, video_names, fit_names)
    metrics = {
        "schema_version": 1,
        "metric_warning": "Frame metrics are dominated by class imbalance and are secondary to event-level center recall/precision/delay.",
        "fit_frame": binary_metrics(target[fit_mask], probability[fit_mask]),
        "calibration_frame": binary_metrics(target[calibration_mask], probability[calibration_mask]),
        "threshold_search": threshold_rows, "selected_calibration_event": selected,
        "gate": gate,
        "fit_event_at_selected_threshold": event_metrics(fit_scores, centers, selected["threshold"])
        if selected else None,
    }
    root = args.output_root
    aggregate = root / "aggregate"
    aggregate.mkdir(parents=True, exist_ok=True)
    model_path = root / "model.json"
    metrics_path = aggregate / "train_calibration_metrics.json"
    model_path.write_text(json.dumps(model, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    metrics_path.write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    freeze = {
        "schema_version": 1, "status": "predev_frozen", "test_opened_or_run": False,
        "inputs": {str(path.resolve()): {"bytes": path.stat().st_size, "sha256": sha256_file(path)}
                   for path in (args.features, args.split, args.train_bags, args.train_keypoints,
                                args.split_manifest)},
        "outputs": {str(path.resolve()): {"bytes": path.stat().st_size, "sha256": sha256_file(path)}
                    for path in (model_path, metrics_path)},
        "runtime": {"device": "cpu", "python": platform.python_version(),
                    "numpy": np.__version__, "scipy": scipy.__version__},
        "dev_permission": "model, normalization, threshold grid, event policy and gate are frozen; dev detection may now run once",
        "scheduler_permission": "only if gate.qualified is true",
    }
    freeze_path = root / "predev_freeze_manifest.json"
    freeze_path.write_text(json.dumps(freeze, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"optimizer": optimizer, "gate": gate,
                      "calibration_frame": metrics["calibration_frame"],
                      "selected_calibration_event": selected,
                      "freeze_manifest_sha256": sha256_file(freeze_path)}, indent=2))


if __name__ == "__main__":
    main()
