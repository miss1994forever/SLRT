#!/usr/bin/env python3
"""Outcome-blind numeric diagnostics for the frozen HRNet heatmap pilot."""
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "results/phoenix-2014t_ISLR/p4_hrnet_heatmap_uncertainty_pilot_v1_49faacc3"
METRICS = OUTPUT / "metrics.json"
DESTINATION = OUTPUT / "normalization_diagnostics.json"


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def distribution_summary(probability):
    probability /= np.maximum(probability.sum(axis=(-2, -1), keepdims=True), 1e-30)
    yy = np.arange(96, dtype=np.float32)[None, None, :, None]
    xx = np.arange(72, dtype=np.float32)[None, None, None, :]
    mux = (probability * xx).sum(axis=(-2, -1)); muy = (probability * yy).sum(axis=(-2, -1))
    dx, dy = xx - mux[..., None, None], yy - muy[..., None, None]
    trace = (probability * (dx * dx + dy * dy)).sum(axis=(-2, -1))
    entropy = -(probability * np.log(np.maximum(probability, 1e-30))).sum(axis=(-2, -1))
    return {"median_covariance_trace": float(np.median(trace)),
            "median_entropy": float(np.median(entropy)),
            "median_probability_peak": float(np.median(probability.max(axis=(-2, -1))))}


def main():
    if DESTINATION.exists():
        raise FileExistsError(DESTINATION)
    if not json.loads(METRICS.read_text())["wer_utility_reference_or_heldout_read"] is False:
        raise RuntimeError("pilot leakage marker failure")
    parts = [np.load(path) for path in sorted(OUTPUT.glob("worker-??-of-05.npz"))]
    predicted = np.concatenate([part["predicted_keypoints"] for part in parts])
    frozen = np.concatenate([part["frozen_keypoints"] for part in parts])
    stats = np.concatenate([part["statistics"] for part in parts])
    raw = np.concatenate([part["raw_hand_heatmaps"].astype(np.float32) for part in parts])
    groups = {"body": slice(0, 23), "face": slice(23, 91), "hands": slice(91, 133)}
    agreement = {}
    for name, group in groups.items():
        valid = ((predicted[:, group, 2] > 0.2) & (frozen[:, group, 2] > 0.2)
                 & np.isfinite(predicted[:, group, :2]).all(2) & np.isfinite(frozen[:, group, :2]).all(2))
        distance = np.linalg.norm(predicted[:, group, :2] - frozen[:, group, :2], axis=2)[valid]
        agreement[name] = {"observations": int(len(distance)), "median_pixel_distance": float(np.median(distance)),
                           "p90_pixel_distance": float(np.quantile(distance, .9)),
                           "p95_pixel_distance": float(np.quantile(distance, .95))}
    normalization = {"positive_mass": distribution_summary(np.maximum(raw, 0))}
    for temperature in (1.0, 0.25, 0.1, 0.05):
        shifted = (raw - raw.max(axis=(-2, -1), keepdims=True)) / temperature
        normalization[f"softmax_T{temperature:g}"] = distribution_summary(np.exp(shifted))
    result = {
        "status": "complete", "created_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "100 preregistered diagnostic hand heatmaps plus all 4498 coordinate pairs",
        "gpu_used": False, "outcomes_read": False,
        "metrics_sha256": sha256_file(METRICS),
        "coordinate_agreement_confidence_gt_0.2": agreement,
        "confidence_correlation": {
            "all": float(np.corrcoef(predicted[..., 2].ravel(), frozen[..., 2].ravel())[0, 1]),
            "hands": float(np.corrcoef(predicted[:, 91:, 2].ravel(), frozen[:, 91:, 2].ravel())[0, 1]),
        },
        "softmax_T1_all_keypoints": {"entropy_theoretical_uniform": float(np.log(96 * 72)),
                                      "entropy_observed_median": float(np.median(stats[..., 7]))},
        "hand_normalization_sensitivity": normalization,
        "decision": "T=1 spatial softmax is nearly uniform for HRNet score maps; do not freeze it as the production covariance. Preregister positive-mass normalization and temperature sensitivity before any downstream outcome.",
    }
    for part in parts: part.close()
    DESTINATION.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
