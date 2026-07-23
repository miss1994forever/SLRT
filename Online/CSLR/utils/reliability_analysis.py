"""Pure analysis helpers for reliability diagnostics (no model dependency)."""

from typing import Dict, Iterable, List

import numpy as np


def binary_auc(scores: Iterable[float], labels: Iterable[bool]):
    """AUROC using average ranks; returns None when either class is absent."""
    scores = np.asarray(list(scores), dtype=np.float64)
    labels = np.asarray(list(labels), dtype=bool)
    positives = int(labels.sum())
    negatives = int((~labels).sum())
    if positives == 0 or negatives == 0:
        return None
    order = np.argsort(scores, kind="mergesort")
    sorted_scores = scores[order]
    ranks = np.empty(len(scores), dtype=np.float64)
    start = 0
    while start < len(scores):
        end = start + 1
        while end < len(scores) and sorted_scores[end] == sorted_scores[start]:
            end += 1
        ranks[order[start:end]] = (start + 1 + end) / 2.0
        start = end
    rank_sum = ranks[labels].sum()
    return float((rank_sum - positives * (positives + 1) / 2.0) / (positives * negatives))


def expected_calibration_error(confidence, correct, bins=10):
    confidence = np.asarray(confidence, dtype=np.float64)
    correct = np.asarray(correct, dtype=np.float64)
    total = len(confidence)
    if total == 0:
        return None
    error = 0.0
    for index in range(bins):
        low, high = index / bins, (index + 1) / bins
        selected = (confidence >= low) & (confidence < high if index < bins - 1 else confidence <= high)
        if selected.any():
            error += selected.mean() * abs(correct[selected].mean() - confidence[selected].mean())
    return float(error)


def safe_correlation(x, y):
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if len(x) < 2 or np.std(x) == 0 or np.std(y) == 0:
        return None
    return float(np.corrcoef(x, y)[0, 1])


def selective_accuracy(confidence, correct, coverages=(0.25, 0.5, 0.75, 1.0)):
    confidence = np.asarray(confidence, dtype=np.float64)
    correct = np.asarray(correct, dtype=np.float64)
    order = np.argsort(-confidence, kind="mergesort")
    output = {}
    for coverage in coverages:
        count = max(1, int(np.ceil(len(order) * coverage)))
        output[str(coverage)] = {
            "count": count,
            "accuracy": float(correct[order[:count]].mean()),
            "threshold": float(confidence[order[count - 1]]),
        }
    return output


def quantile_bins(values, outcomes, bins=4):
    values = np.asarray(values, dtype=np.float64)
    outcomes = np.asarray(outcomes, dtype=np.float64)
    if len(values) == 0:
        return []
    edges = np.quantile(values, np.linspace(0, 1, bins + 1))
    output: List[Dict] = []
    for index in range(bins):
        selected = (values >= edges[index]) & (
            values <= edges[index + 1] if index == bins - 1 else values < edges[index + 1]
        )
        output.append(
            {
                "low": float(edges[index]),
                "high": float(edges[index + 1]),
                "count": int(selected.sum()),
                "outcome_mean": None if not selected.any() else float(outcomes[selected].mean()),
            }
        )
    return output


def summarize_signal(confidence, correct):
    confidence = np.asarray(confidence, dtype=np.float64)
    correct = np.asarray(correct, dtype=bool)
    return {
        "count": int(len(confidence)),
        "accuracy": float(correct.mean()),
        "mean_confidence": float(confidence.mean()),
        "correctness_auc": binary_auc(confidence, correct),
        "confidence_correctness_correlation": safe_correlation(confidence, correct.astype(float)),
        "ece_10_bins": expected_calibration_error(confidence, correct, bins=10),
        "selective_accuracy": selective_accuracy(confidence, correct),
    }
