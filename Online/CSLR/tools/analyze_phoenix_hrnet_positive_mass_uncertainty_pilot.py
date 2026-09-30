#!/usr/bin/env python3
"""Outcome-blind gates for the HRNet positive-mass uncertainty pilot."""
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "results/phoenix-2014t_ISLR/p4_hrnet_positive_mass_uncertainty_pilot_v1_49faacc3"
CONFIG = ROOT / "configs/p4_hrnet_positive_mass_uncertainty_pilot_v1_preregistered.json"
DESTINATION = OUTPUT / "positive_mass_diagnostics.json"


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main():
    if DESTINATION.exists():
        raise FileExistsError(DESTINATION)
    config = json.loads(CONFIG.read_text())
    metrics_path = OUTPUT / "metrics.json"
    metrics = json.loads(metrics_path.read_text())
    if metrics["wer_utility_reference_or_heldout_read"] is not False:
        raise RuntimeError("outcome leakage marker failure")
    parts = [np.load(path) for path in sorted(OUTPUT.glob("worker-??-of-05.npz"))]
    stats = np.concatenate([part["statistics"] for part in parts])
    boxes = np.concatenate([part["bboxes"] for part in parts])
    timings = np.concatenate([part["timings_seconds"] for part in parts])
    valid = np.isfinite(stats[..., :8]).all(axis=2) & (stats[..., 8] > 1e-12)
    varx, vary, cov = stats[..., 2], stats[..., 3], stats[..., 4]
    min_eigenvalue = .5 * (varx + vary - np.sqrt((varx - vary) ** 2 + 4 * cov ** 2))
    trace = varx + vary
    hands = slice(91, 133)
    hand_valid = valid[:, hands]
    hand_trace = trace[:, hands][hand_valid]
    hand_entropy = stats[:, hands, 7][hand_valid]
    hand_peak = stats[:, hands, 5][hand_valid]
    frame_hand_trace = np.nanmedian(np.where(hand_valid, trace[:, hands], np.nan), axis=1)
    bbox_area = np.maximum(boxes[:, 2] - boxes[:, 0], 0) * np.maximum(boxes[:, 3] - boxes[:, 1], 0)
    trace_peak_rho = float(spearmanr(hand_trace, hand_peak).statistic)
    trace_bbox_rho = float(spearmanr(frame_hand_trace, bbox_area).statistic)
    gates = config["outcome_blind_gates"]
    values = {
        "invalid_keypoint_fraction": float(1.0 - valid.mean()),
        "valid_covariance_finite_fraction": float(np.isfinite(np.stack([varx[valid], vary[valid], cov[valid]], 1)).all(1).mean()),
        "normalized_covariance_min_eigenvalue": float(np.nanmin(min_eigenvalue)),
        "hand_trace_iqr": float(np.quantile(hand_trace, .75) - np.quantile(hand_trace, .25)),
        "hand_entropy_iqr": float(np.quantile(hand_entropy, .75) - np.quantile(hand_entropy, .25)),
        "spearman_hand_trace_vs_raw_peak": trace_peak_rho,
        "spearman_frame_median_hand_trace_vs_bbox_area": trace_bbox_rho,
        "statistics_median_ms_per_frame": float(np.median(timings[:, 3]) * 1000),
    }
    passed = {
        "invalid_mass": values["invalid_keypoint_fraction"] < gates["invalid_keypoint_fraction_strictly_less_than"],
        "finite_covariance": values["valid_covariance_finite_fraction"] >= gates["valid_covariance_finite_fraction_at_least"],
        "covariance_psd": values["normalized_covariance_min_eigenvalue"] >= gates["normalized_covariance_min_eigenvalue_at_least"],
        "trace_nonconstant": values["hand_trace_iqr"] > gates["hand_trace_iqr_strictly_greater_than"],
        "entropy_nonconstant": values["hand_entropy_iqr"] > gates["hand_entropy_iqr_strictly_greater_than"],
        "not_raw_peak_redundant": abs(trace_peak_rho) < gates["absolute_spearman_hand_trace_vs_raw_peak_strictly_less_than"],
        "not_bbox_size_dominated": abs(trace_bbox_rho) < gates["absolute_spearman_frame_median_hand_trace_vs_bbox_area_strictly_less_than"],
        "statistics_latency": values["statistics_median_ms_per_frame"] <= gates["statistics_median_ms_per_frame_at_most"],
    }
    quantiles = {}
    for name, value in {"hand_trace": hand_trace, "hand_entropy": hand_entropy,
                        "hand_probability_peak": stats[:, hands, 6][hand_valid],
                        "hand_positive_mass": stats[:, hands, 8][hand_valid]}.items():
        quantiles[name] = {str(q): float(np.quantile(value, q)) for q in (0, .05, .25, .5, .75, .95, 1)}
    result = {
        "status": "complete", "created_utc": datetime.now(timezone.utc).isoformat(),
        "train_only": True, "outcomes_read": False, "frames": int(len(stats)),
        "keypoint_observations": int(valid.size), "valid_keypoint_observations": int(valid.sum()),
        "values": values, "distribution_quantiles": quantiles, "gates_passed": passed,
        "all_gates_passed": bool(all(passed.values())),
        "decision": "unlock full-train extraction" if all(passed.values()) else "stop before full-train extraction",
        "config_sha256": sha256_file(CONFIG), "metrics_sha256": sha256_file(metrics_path),
    }
    for part in parts:
        part.close()
    DESTINATION.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
