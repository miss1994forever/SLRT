#!/usr/bin/env python3
"""Positive-mass v2 pilot: aggregate heatmap statistics with torch on CUDA."""
import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np

import run_phoenix_hrnet_heatmap_uncertainty_pilot as base


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/p4_hrnet_positive_mass_gpu_stats_pilot_v2_preregistered.json"
OUTPUT = ROOT / "results/phoenix-2014t_ISLR/p4_hrnet_positive_mass_gpu_stats_pilot_v2_49faacc3"
SELECTION = OUTPUT / "selection.json"
PRIOR_SELECTION = ROOT / "results/phoenix-2014t_ISLR/p4_hrnet_positive_mass_uncertainty_pilot_v1_49faacc3/selection.json"
_GRID = None


def configure_base():
    base.CONFIG = CONFIG
    base.OUTPUT = OUTPUT
    base.SELECTION = SELECTION
    base.heatmap_stats = heatmap_stats


def heatmap_stats(heatmap):
    global _GRID
    import torch
    value = np.asarray(heatmap, np.float32)
    if value.shape != (133, 96, 72):
        raise RuntimeError(f"unexpected heatmap shape: {value.shape}")
    if _GRID is None:
        x = torch.arange(72, device="cuda", dtype=torch.float32)
        y = torch.arange(96, device="cuda", dtype=torch.float32)
        _GRID = (x, y, x * x, y * y)
    x, y, x2, y2 = _GRID
    h = torch.as_tensor(value, device="cuda")
    positive = h.clamp_min(0)
    mass = positive.sum(dim=(1, 2))
    finite = torch.isfinite(h).all(dim=2).all(dim=1)
    valid = finite & (mass > 1e-12)
    safe_mass = torch.where(valid, mass, torch.ones_like(mass))
    xmass = positive.sum(dim=1)
    ymass = positive.sum(dim=2)
    mux = (xmass @ x) / safe_mass
    muy = (ymass @ y) / safe_mass
    ex2 = (xmass @ x2) / safe_mass
    ey2 = (ymass @ y2) / safe_mass
    exy = ((positive @ x) @ y) / safe_mass
    varx = (ex2 - mux * mux).clamp_min(0)
    vary = (ey2 - muy * muy).clamp_min(0)
    cov = exy - mux * muy
    raw_peak = h.amax(dim=(1, 2))
    probability_peak = positive.amax(dim=(1, 2)) / safe_mass
    entropy = torch.log(safe_mass) - torch.special.xlogy(positive, positive).sum(dim=(1, 2)) / safe_mass
    result = torch.stack([
        mux / 72.0, muy / 96.0, varx / (72.0 ** 2), vary / (96.0 ** 2),
        cov / (72.0 * 96.0), raw_peak, probability_peak, entropy, mass, mux, muy,
    ], dim=1)
    result[~valid, :8] = torch.nan
    return result.cpu().numpy()


def prepare():
    configure_base()
    base.prepare()
    current = json.loads(SELECTION.read_text())
    prior = json.loads(PRIOR_SELECTION.read_text())
    if current["rows"] != prior["rows"] or current["selected_frames"] != prior["selected_frames"]:
        raise RuntimeError("v2 sample is not identical to positive-mass v1")
    identity_path = OUTPUT / "preregistered_identity.json"
    identity = json.loads(identity_path.read_text())
    identity["prior_selection_sha256"] = base.sha256_file(PRIOR_SELECTION)
    identity["selection_rows_identical_to_prior"] = True
    identity_path.write_text(json.dumps(identity, indent=2) + "\n")


def launch():
    workers = json.loads(CONFIG.read_text())["gpu_policy"]["allowed"]
    processes = []
    logs = []
    for index, gpu in enumerate(workers):
        log_path = OUTPUT / f"worker-{index:02d}.log"
        handle = log_path.open("w")
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = gpu["uuid"]
        command = [sys.executable, str(Path(__file__).resolve()), "--worker", str(index),
                   "--workers", str(len(workers)), "--gpu-uuid", gpu["uuid"]]
        processes.append((index, subprocess.Popen(command, stdout=handle, stderr=subprocess.STDOUT, env=env), handle))
        logs.append(str(log_path))
    failures = []
    for index, process, handle in processes:
        code = process.wait()
        handle.close()
        if code != 0:
            failures.append({"worker": index, "exit_code": code})
    if failures:
        raise RuntimeError({"worker_failures": failures, "logs": logs})
    configure_base()
    base.aggregate()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prepare", action="store_true")
    parser.add_argument("--launch", action="store_true")
    parser.add_argument("--worker", type=int)
    parser.add_argument("--workers", type=int)
    parser.add_argument("--gpu-uuid")
    args = parser.parse_args()
    if args.prepare:
        prepare()
    elif args.launch:
        launch()
    elif args.worker is not None:
        configure_base()
        base.run_worker(args.worker, args.workers, args.gpu_uuid)
    else:
        parser.error("choose --prepare, --launch, or --worker")


if __name__ == "__main__":
    main()
