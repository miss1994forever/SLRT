#!/usr/bin/env python3
"""Train and freeze the P2 train-only causal TCN center predictor."""

import argparse
import importlib.util
import json
import math
import os
import platform
import random
import shutil
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


ROOT = Path(__file__).resolve().parents[1]


def import_tool(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / f"tools/{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


P1 = import_tool("train_phoenix_p1_center_predictor")
FEATURES = import_tool("build_phoenix_p2_center_sequences")
SEED = 20260912
EPOCHS = 8
BATCH_SIZE = 64
CHANNELS = 32
DILATIONS = (1, 2, 4, 8)
DROPOUT = 0.1
LEARNING_RATE = 0.002
WEIGHT_DECAY = 0.0001


class CausalBlock(nn.Module):
    def __init__(self, channels, dilation, dropout):
        super().__init__()
        self.padding = 2 * dilation
        self.conv = nn.Conv1d(channels, channels, kernel_size=3, dilation=dilation)
        self.dropout = nn.Dropout(dropout)

    def forward(self, value):
        update = self.conv(F.pad(value, (self.padding, 0)))
        return value + self.dropout(F.gelu(update))


class CenterTCN(nn.Module):
    def __init__(self, input_features, channels=CHANNELS, dilations=DILATIONS, dropout=DROPOUT):
        super().__init__()
        self.input_projection = nn.Conv1d(input_features, channels, kernel_size=1)
        self.blocks = nn.ModuleList([CausalBlock(channels, dilation, dropout)
                                     for dilation in dilations])
        self.output = nn.Conv1d(channels, 1, kernel_size=1)

    def forward(self, value):
        value = F.gelu(self.input_projection(value))
        for block in self.blocks:
            value = block(value)
        return self.output(value).squeeze(1)


def set_determinism(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False


def video_slices(names, offsets):
    return {str(name): slice(int(offsets[index]), int(offsets[index + 1]))
            for index, name in enumerate(names)}


def fit_statistics(features, slices, fit_names):
    total = np.zeros(features.shape[1], dtype=np.float64)
    square = np.zeros(features.shape[1], dtype=np.float64)
    rows = 0
    for name in fit_names:
        value = features[slices[name]].astype(np.float64)
        total += value.sum(axis=0)
        square += np.square(value).sum(axis=0)
        rows += len(value)
    mean = total / rows
    variance = np.maximum(square / rows - np.square(mean), 0.0)
    scale = np.sqrt(variance)
    scale[scale == 0] = 1.0
    return mean.astype(np.float32), scale.astype(np.float32), rows


def make_batches(names, slices, batch_size, rng):
    ordered = sorted(names, key=lambda name: slices[name].stop - slices[name].start)
    buckets = [ordered[index:index + batch_size] for index in range(0, len(ordered), batch_size)]
    rng.shuffle(buckets)
    return buckets


def batch_tensors(batch_names, features, targets, slices, mean, scale, device):
    lengths = [slices[name].stop - slices[name].start for name in batch_names]
    maximum = max(lengths)
    matrix = np.zeros((len(batch_names), maximum, features.shape[1]), dtype=np.float32)
    target = np.zeros((len(batch_names), maximum), dtype=np.float32)
    mask = np.zeros((len(batch_names), maximum), dtype=np.bool_)
    for row, name in enumerate(batch_names):
        source = slices[name]
        length = lengths[row]
        matrix[row, :length] = (features[source].astype(np.float32) - mean) / scale
        target[row, :length] = targets[source]
        mask[row, :length] = True
    x = torch.from_numpy(matrix).transpose(1, 2).to(device)
    y = torch.from_numpy(target).to(device)
    valid = torch.from_numpy(mask).to(device)
    return x, y, valid


def score_videos(model, names, features, slices, mean, scale, device):
    result = {}
    model.eval()
    with torch.no_grad():
        for name in names:
            value = (features[slices[name]].astype(np.float32) - mean) / scale
            tensor = torch.from_numpy(value).transpose(0, 1).unsqueeze(0).to(device)
            result[name] = torch.sigmoid(model(tensor))[0].cpu().numpy().astype(np.float64)
    return result


def sha_record(path):
    path = Path(path)
    return {"bytes": path.stat().st_size, "sha256": P1.sha256_file(path)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sequences", type=Path, required=True)
    parser.add_argument("--split", type=Path, required=True)
    parser.add_argument("--train-bags", type=Path, required=True)
    parser.add_argument("--train-keypoints", type=Path, required=True)
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    for path in vars(args).values():
        P1.reject_dev_or_test_path(path)
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("P2 requires exactly one UUID-bound healthy CUDA device")
    set_determinism()
    device = torch.device("cuda:0")
    data = np.load(args.sequences)
    features, targets = data["features"], data["targets"]
    names, offsets = data["video_names"].tolist(), data["video_offsets"]
    feature_names = data["feature_names"].tolist()
    if feature_names != list(FEATURES.FEATURE_NAMES):
        raise ValueError("feature identity differs from implementation")
    split = json.loads(args.split.read_text())
    fit_names, calibration_names = split["fit_videos"], split["calibration_videos"]
    if len(fit_names) != 5655 or len(calibration_names) != 1440:
        raise ValueError("frozen P1 group split sizes changed")
    if set(names) != set(fit_names) | set(calibration_names):
        raise ValueError("sequence names do not match frozen split")
    slices = video_slices(names, offsets)
    mean, scale, fit_rows = fit_statistics(features, slices, fit_names)
    positive_rows = sum(int(targets[slices[name]].sum()) for name in fit_names)
    pos_weight = (fit_rows - positive_rows) / positive_rows
    model = CenterTCN(features.shape[1]).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_weight, device=device), reduction="none")
    history = []
    for epoch in range(EPOCHS):
        model.train()
        rng = random.Random(SEED + epoch)
        loss_sum = valid_rows = 0.0
        for batch_names in make_batches(fit_names, slices, BATCH_SIZE, rng):
            x, y, valid = batch_tensors(batch_names, features, targets, slices, mean, scale, device)
            optimizer.zero_grad(set_to_none=True)
            loss_values = criterion(model(x), y)
            loss = loss_values[valid].mean()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            loss_sum += float(loss_values[valid].detach().sum().cpu())
            valid_rows += int(valid.sum().cpu())
        history.append({"epoch": epoch + 1, "fit_weighted_bce": loss_sum / valid_rows})
        print(json.dumps(history[-1]), flush=True)

    calibration_scores = score_videos(model, calibration_names, features, slices, mean, scale, device)
    centers = P1.centers_from_bags(args.train_bags, set(fit_names) | set(calibration_names))
    threshold_rows = [P1.event_metrics(calibration_scores, centers, threshold)
                      for threshold in P1.THRESHOLDS]
    feasible = [row for row in threshold_rows if row["center_recall"] >= 0.75 and
                row["positive_delay_fraction"] is not None and
                row["positive_delay_fraction"] <= 0.25]
    selected = max(feasible, key=lambda row: (row["threshold"], row["event_precision"])) if feasible else None
    gate = {
        "qualified": selected is not None,
        "requirements": {"center_recall_minimum": 0.75,
                         "positive_delay_fraction_maximum": 0.25},
        "selected_threshold": selected["threshold"] if selected else None,
        "reason": "qualified" if selected else "no calibration threshold satisfied both requirements",
    }
    calibration_target = np.concatenate([targets[slices[name]] for name in calibration_names])
    calibration_probability = np.concatenate([calibration_scores[name] for name in calibration_names])
    metrics = {
        "schema_version": 1, "training_history": history,
        "calibration_frame": P1.binary_metrics(calibration_target, calibration_probability),
        "threshold_search": threshold_rows, "selected_calibration_event": selected, "gate": gate,
    }
    config = {
        "schema_version": 1, "stage": "P2 train-only causal sign-interior predictor",
        "seed": SEED, "features": feature_names, "confidence_threshold": 0.2,
        "model": {"type": "causal_residual_tcn", "channels": CHANNELS, "kernel_size": 3,
                  "dilations": list(DILATIONS), "dropout": DROPOUT, "receptive_field": 31},
        "training": {"epochs": EPOCHS, "batch_size_videos": BATCH_SIZE,
                     "optimizer": "AdamW", "learning_rate": LEARNING_RATE,
                     "weight_decay": WEIGHT_DECAY, "gradient_clip": 1.0,
                     "loss": "fit-only class-balanced BCEWithLogitsLoss", "model_selection": "none"},
        "calibration": {"thresholds": "0.01..0.99 step 0.01", "refractory_frames": 4,
                        "matching_tolerance_frames": 1, "gate": gate},
        "test_policy": "forbidden",
    }
    root = args.output_root
    aggregate = root / "aggregate"
    aggregate.mkdir(parents=True, exist_ok=True)
    split_copy = root / "fit_calibration_split.json"
    if args.split.resolve() != split_copy.resolve():
        shutil.copyfile(args.split, split_copy)
    config_path = root / "resolved_config.json"
    metrics_path = aggregate / "train_calibration_metrics.json"
    model_path = root / "model.pt"
    config_path.write_text(json.dumps(config, indent=2, sort_keys=True) + "\n")
    metrics_path.write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n")
    torch.save({"state_dict": model.state_dict(), "mean_fit_only": mean,
                "scale_fit_only": scale, "feature_names": feature_names, "config": config}, model_path)
    inputs = [args.sequences, args.split, args.train_bags, args.train_keypoints,
              args.split_manifest, args.protocol]
    outputs = [split_copy, config_path, metrics_path, model_path]
    freeze = {
        "schema_version": 1, "status": "predev_frozen", "gate": gate,
        "test_opened_or_run": False, "dev_opened_or_run": False,
        "inputs": {str(path.resolve()): sha_record(path) for path in inputs},
        "outputs": {str(path.resolve()): sha_record(path) for path in outputs},
        "runtime": {"device": torch.cuda.get_device_name(0),
                    "visible_device_uuid": os.environ.get("CUDA_VISIBLE_DEVICES"),
                    "python": platform.python_version(), "numpy": np.__version__,
                    "torch": torch.__version__},
        "dev_permission": "one-shot dev detection and scheduler only if gate.qualified is true",
    }
    freeze_path = root / "predev_freeze_manifest.json"
    freeze_path.write_text(json.dumps(freeze, indent=2, sort_keys=True) + "\n")
    code_paths = [Path(__file__), ROOT / "tools/build_phoenix_p2_center_sequences.py"]
    manifest = {
        "schema_version": 1, "scope": "P2 train-fit/calibration causal TCN center predictor",
        "split_policy": "train fit/calibration only; dev conditional on frozen gate; test forbidden",
        "inputs": {str(path.resolve()): P1.sha256_file(path) for path in inputs},
        "code": {str(path.relative_to(ROOT.parent)): P1.sha256_file(path) for path in code_paths},
        "outputs": {str(path.relative_to(root)): P1.sha256_file(path)
                    for path in outputs + [freeze_path]},
        "decision": {"gate": "go" if gate["qualified"] else "no_go",
                     "selected_threshold": gate["selected_threshold"]},
        "runtime": freeze["runtime"],
    }
    manifest_path = root / "protocol_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"gate": gate, "calibration_frame": metrics["calibration_frame"],
                      "manifest_sha256": P1.sha256_file(manifest_path)}, indent=2), flush=True)


if __name__ == "__main__":
    main()
