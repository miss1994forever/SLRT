#!/usr/bin/env python3
"""Freeze the actual P7 A/B/C fair-training launch identity."""

from __future__ import annotations

import hashlib
import json
import platform
from datetime import datetime, timezone
from pathlib import Path

import torch


CSLR_ROOT = Path(__file__).resolve().parents[1]
BASE = CSLR_ROOT / "results/phoenix-2014t_ISLR"
RUN_ROOT = BASE / "p7_temporal_smoke_v1"
OUTPUT = RUN_ROOT / "launch_manifest.json"
FAIRNESS = BASE / "p7_abc_fairness_audit_v1/summary.json"
PREFLIGHT = BASE / "p7_cuda_recovery_probe_20261012/training_preflight.json"
CANDIDATES = {
    "A_current_fixed16": {
        "config": "p7_temporal_debug_smoke_A_current_fixed16.yaml",
        "port": 29681,
        "gpus": [
            {"pci": "0000:01:00.0", "uuid": "GPU-dbd35875-dfa5-43f1-0cf0-f88ccb529c8a"},
            {"pci": "0000:41:00.0", "uuid": "GPU-2c6a50cb-f770-c785-dc1d-7f8cc7b7b9aa"},
        ],
    },
    "B_aligned_fixed16": {
        "config": "p7_temporal_debug_smoke_B_aligned_fixed16.yaml",
        "port": 29682,
        "gpus": [
            {"pci": "0000:61:00.0", "uuid": "GPU-91f9e38c-0a59-15c0-bd08-6a4f89da07ed"},
            {"pci": "0000:81:00.0", "uuid": "GPU-e1683bce-0e4f-68bc-54cc-4a2f62f55631"},
        ],
    },
    "C_inside_even16": {
        "config": "p7_temporal_debug_smoke_C_inside_even16.yaml",
        "port": 29683,
        "gpus": [
            {"pci": "0000:a1:00.0", "uuid": "GPU-4f5bee72-bad2-f8c9-2656-58e4d94c3c98"},
            {"pci": "0000:e1:00.0", "uuid": "GPU-8c92fc9f-aaa8-cd49-7196-a08cca1c6c7b"},
        ],
    },
}
FORBIDDEN = {"pci": "0000:25:00.0", "uuid": "GPU-06afe121-c4ce-b981-bb86-399e4a85ae83"}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def snapshot_digest(directory: Path) -> dict:
    files = sorted(
        path for path in directory.iterdir()
        if path.is_file() and (path.suffix == ".py" or path.suffix in {".yaml", ".yml"})
    )
    digest = hashlib.sha256()
    records = {}
    for path in files:
        value = sha256(path)
        records[path.name] = value
        digest.update(path.name.encode())
        digest.update(b"\0")
        digest.update(value.encode())
        digest.update(b"\n")
    return {"tree_sha256": digest.hexdigest(), "files": records}


def main() -> None:
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    fairness = json.loads(FAIRNESS.read_text())
    preflight = json.loads(PREFLIGHT.read_text())
    if fairness["status"] != "pass" or preflight["status"] != "pass":
        raise RuntimeError("fairness audit and CUDA preflight must pass")
    selected = {(row["pci_bus_id"].lower(), row["uuid"]) for row in preflight["selected"]}
    records = {}
    for candidate, specification in CANDIDATES.items():
        config = CSLR_ROOT / "configs" / specification["config"]
        directory = RUN_ROOT / candidate
        copied = directory / specification["config"]
        if not copied.is_file() or sha256(copied) != sha256(config):
            raise RuntimeError(f"training config snapshot mismatch: {candidate}")
        requested = {(gpu["pci"].lower(), gpu["uuid"]) for gpu in specification["gpus"]}
        if not requested <= selected:
            raise RuntimeError(f"GPU assignment did not pass preflight: {candidate}")
        visible = ",".join(gpu["uuid"] for gpu in specification["gpus"])
        command = [
            "env", "CUDA_DEVICE_ORDER=PCI_BUS_ID", f"CUDA_VISIBLE_DEVICES={visible}",
            "OMP_NUM_THREADS=1", "WANDB_MODE=disabled",
            "/mnt/workspace/conda_envs/haojun/envs/slrt_legacy/bin/python", "-u", "-m",
            "torch.distributed.run", "--nproc_per_node=2",
            f"--master_port={specification['port']}", "training.py",
            f"--config=configs/{specification['config']}",
        ]
        records[candidate] = {
            "model_dir": str(directory.resolve()),
            "config": {"path": str(config.resolve()), "sha256": sha256(config)},
            "copied_config": {"path": str(copied.resolve()), "sha256": sha256(copied)},
            "gpus": specification["gpus"],
            "command": command,
            "training_snapshot": snapshot_digest(directory),
        }
    assigned = [gpu for record in records.values() for gpu in record["gpus"]]
    if any(gpu["pci"].lower() == FORBIDDEN["pci"] or gpu["uuid"] == FORBIDDEN["uuid"] for gpu in assigned):
        raise RuntimeError("forbidden GPU assigned")
    result = {
        "schema_version": 1,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "source-disjoint train-only P7 A/B/C fair short training; official dev/test not read",
        "status_at_creation": "running",
        "world_size_per_candidate": 2,
        "candidates": records,
        "forbidden_gpu": FORBIDDEN,
        "checks": {
            "fairness_audit_passed": True,
            "cuda_preflight_passed": True,
            "copied_configs_match_sources": True,
            "all_assigned_gpus_passed_preflight": True,
            "forbidden_gpu_not_assigned": True,
            "candidate_gpu_sets_disjoint": len({gpu["uuid"] for gpu in assigned}) == len(assigned),
        },
        "inputs": {
            "fairness_audit": {"path": str(FAIRNESS.resolve()), "sha256": sha256(FAIRNESS)},
            "cuda_preflight": {"path": str(PREFLIGHT.resolve()), "sha256": sha256(PREFLIGHT)},
        },
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "cuda_runtime": torch.version.cuda,
        },
    }
    OUTPUT.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"output": str(OUTPUT.resolve()), "candidates": len(records)}, sort_keys=True))


if __name__ == "__main__":
    main()
