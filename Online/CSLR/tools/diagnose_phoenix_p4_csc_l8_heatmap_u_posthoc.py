#!/usr/bin/env python3
"""Post-hoc attribution of the frozen failed CSC-L8 K+U ridge probe."""
import json
from pathlib import Path

import numpy as np

import analyze_phoenix_p4_csc_l8_heatmap_u_identifiability as probe
import analyze_phoenix_p4_csc_l8_identifiability_probe as prior


def score(prediction, target, winner, baseline):
    residual = np.asarray(target - prediction, np.float64)
    centered = np.asarray(target - baseline, np.float64)
    return {
        "pooled_r2": float(1.0 - np.sum(residual * residual) / np.sum(centered * centered)),
        "winner_accuracy": float(np.mean(prediction.argmax(axis=1) + 1 == winner)),
    }


def main():
    with np.load(prior.TARGETS, allow_pickle=False) as packed:
        block_indices = np.asarray(packed["block_indices"], np.int64)
        folds = np.asarray(packed["folds"], np.int8)
        target = np.asarray(packed["js_t1"], np.float32)
        winner = np.asarray(packed["winners"], np.int8)
    names, offsets = probe.load_fit_index()
    with np.load(prior.HISTORY, allow_pickle=False) as packed:
        arrivals = np.asarray(packed["decision_arrivals"], np.int64)[block_indices]
        sample_indices = np.asarray(packed["sample_indices"], np.int64)[block_indices]
    lengths = np.diff(offsets)
    visible = np.minimum(lengths[sample_indices] - 1, arrivals + prior.VISIBLE_OFFSET)
    frame_features = np.load(probe.FRAME_FEATURES, mmap_mode="r")
    summaries = np.empty((len(block_indices), 384), np.float32)
    delta = np.arange(prior.HISTORY_LENGTH - 1, -1, -1, dtype=np.int64)
    for left in range(0, len(summaries), 4096):
        right = min(left + 4096, len(summaries))
        local = np.maximum(visible[left:right, None] - delta[None, :], 0)
        index = offsets[sample_indices[left:right], None] + local
        values = np.asarray(frame_features[index], np.float32)
        summaries[left:right] = np.concatenate([
            values[:, -1], values.mean(1), values.std(1), values[:, -1] - values[:, 0]
        ], axis=1)
    projection = probe.projection_matrix(261121)
    matrix_u = summaries @ projection
    matrix_k = prior.build_k(block_indices, names)
    matrix = np.concatenate([matrix_k, matrix_u], axis=1)
    prediction = np.empty_like(target)
    baseline = np.empty_like(target)
    records = []
    for fold in range(5):
        train, evaluate = folds != fold, folds == fold
        mean = matrix[train].mean(axis=0, dtype=np.float64).astype(np.float32)
        scale = matrix[train].std(axis=0, dtype=np.float64).astype(np.float32)
        scale[scale < 1e-6] = 1.0
        xtrain = np.asarray((matrix[train] - mean) / scale, np.float64)
        xeval = np.asarray((matrix[evaluate] - mean) / scale, np.float64)
        ymean = target[train].mean(axis=0, dtype=np.float64)
        ytrain = np.asarray(target[train] - ymean, np.float64)
        gram = xtrain.T @ xtrain
        gram.flat[::len(gram) + 1] += 10.0
        weights = np.linalg.solve(gram, xtrain.T @ ytrain)
        prediction[evaluate] = (xeval @ weights + ymean).astype(np.float32)
        baseline[evaluate] = ymean.astype(np.float32)
        records.append((train, evaluate, scale[68:], weights[68:]))
    full = score(prediction, target, winner, baseline)
    with np.load(probe.FRAME_INDEX, allow_pickle=False) as packed:
        feature_names = [str(value) for value in packed["feature_names"].tolist()]

    def attribution(frame_columns):
        columns = np.concatenate([np.asarray(frame_columns, np.int64) + 96 * step for step in range(4)])
        component = summaries[:, columns] @ projection[columns]
        removed = prediction.copy()
        squared = 0.0
        count = 0
        for train, evaluate, scale_u, weights_u in records:
            center = component[train].mean(axis=0, dtype=np.float64)
            contribution = (np.asarray(component[evaluate] - center, np.float64)
                            / np.asarray(scale_u, np.float64)) @ weights_u
            removed[evaluate] -= contribution.astype(np.float32)
            squared += float(np.sum(contribution * contribution))
            count += contribution.size
        without = score(removed, target, winner, baseline)
        return {
            "without_component": without,
            "full_minus_without_r2": full["pooled_r2"] - without["pooled_r2"],
            "full_minus_without_winner_accuracy": full["winner_accuracy"] - without["winner_accuracy"],
            "prediction_contribution_rms": float(np.sqrt(squared / count)),
        }

    groups = {}
    for group, begin, end in probe.GROUPS:
        prefix = f"{group}."
        groups[group] = attribution([i for i, name in enumerate(feature_names) if name.startswith(prefix)])
    quantities = {}
    for quantity in probe.QUANTITIES:
        marker = f".{quantity}."
        quantities[quantity] = attribution([i for i, name in enumerate(feature_names) if marker in name])
    result = {
        "status": "posthoc_explanatory_only", "policy_outcome_read": False,
        "warning": "Do not use these attributions to retune the failed frozen v2 probe.",
        "full_KU": full, "groups": groups, "quantities": quantities,
    }
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
