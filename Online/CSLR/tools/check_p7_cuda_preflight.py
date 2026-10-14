#!/usr/bin/env python3
"""Fail closed unless selected P7 GPUs are safe and CUDA can initialize."""

from __future__ import annotations

import argparse
import ctypes
import json
import os
import subprocess
from pathlib import Path


FORBIDDEN_PCI = "0000:25:00.0"
FORBIDDEN_UUID = "GPU-06afe121-c4ce-b981-bb86-399e4a85ae83"


def normalize_pci(value: str) -> str:
    fields = value.strip().lower().split(":")
    if len(fields) == 3:
        domain = fields[0][-4:]
        return f"{domain}:{fields[1]}:{fields[2]}"
    raise ValueError(f"unexpected PCI bus ID: {value!r}")


def gpu_inventory() -> list[dict[str, str]]:
    command = [
        "nvidia-smi",
        "--query-gpu=index,pci.bus_id,uuid,name",
        "--format=csv,noheader,nounits",
    ]
    result = subprocess.run(command, check=True, capture_output=True, text=True)
    rows = []
    for line in result.stdout.splitlines():
        if not line.strip():
            continue
        index, pci, uuid, name = [part.strip() for part in line.split(",", 3)]
        rows.append({"logical_index": index, "pci_bus_id": normalize_pci(pci), "uuid": uuid, "name": name})
    return rows


def cuda_error_name(driver, code: int) -> str:
    value = ctypes.c_char_p()
    if driver.cuGetErrorName(code, ctypes.byref(value)) == 0 and value.value:
        return value.value.decode()
    return "UNKNOWN"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--uuids", required=True, help="Comma-separated verified healthy GPU UUIDs")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    requested = [item.strip() for item in args.uuids.split(",") if item.strip()]
    if not requested or len(requested) != len(set(requested)):
        raise SystemExit("--uuids must contain one or more unique UUIDs")

    report = {
        "schema_version": 1,
        "forbidden": {"pci_bus_id": FORBIDDEN_PCI, "uuid": FORBIDDEN_UUID},
        "requested_uuids": requested,
        "checks": {},
        "status": "fail",
    }
    try:
        inventory = gpu_inventory()
        report["inventory"] = inventory
        by_uuid = {row["uuid"]: row for row in inventory}
        missing = [uuid for uuid in requested if uuid not in by_uuid]
        selected = [by_uuid[uuid] for uuid in requested if uuid in by_uuid]
        report["selected"] = selected
        report["checks"]["all_requested_uuids_visible"] = not missing
        report["checks"]["forbidden_uuid_not_selected"] = FORBIDDEN_UUID not in requested
        report["checks"]["forbidden_pci_not_selected"] = all(row["pci_bus_id"] != FORBIDDEN_PCI for row in selected)
        if not all(report["checks"].values()):
            report["error"] = {"missing_uuids": missing}
        else:
            os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
            os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(requested)
            driver = ctypes.CDLL("libcuda.so.1")
            code = int(driver.cuInit(0))
            report["cuda_driver"] = {"cuInit_return_code": code, "error_name": cuda_error_name(driver, code)}
            report["checks"]["cuda_driver_initializes"] = code == 0
            if code == 0:
                import torch

                available = bool(torch.cuda.is_available())
                count = int(torch.cuda.device_count())
                report["pytorch"] = {
                    "version": torch.__version__,
                    "cuda_runtime": torch.version.cuda,
                    "is_available": available,
                    "device_count": count,
                }
                report["checks"]["pytorch_available"] = available
                report["checks"]["pytorch_count_matches"] = count == len(requested)
            report["status"] = "pass" if all(report["checks"].values()) else "fail"
    except Exception as error:
        report["error"] = {"type": type(error).__name__, "message": str(error)}

    rendered = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered)
    print(rendered, end="")
    if report["status"] != "pass":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
