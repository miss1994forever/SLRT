#!/usr/bin/env python3
"""Write fsynced stage markers while probing one CUDA device."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--log", type=Path, required=True)
    args = parser.parse_args()
    args.log.parent.mkdir(parents=True, exist_ok=True)
    args.log.write_text("")

    def record(stage: str, **values) -> None:
        row = {"stage": stage, **values}
        with args.log.open("a") as handle:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
            handle.flush()
            os.fsync(handle.fileno())

    record("start", visible=os.environ.get("CUDA_VISIBLE_DEVICES"))
    import torch

    record("torch_imported", torch=torch.__version__, runtime=torch.version.cuda)
    available = bool(torch.cuda.is_available())
    record("is_available", value=available)
    count = int(torch.cuda.device_count())
    record("device_count", value=count)
    if not available or count != 1:
        raise RuntimeError(f"unexpected CUDA state: available={available}, count={count}")
    name = torch.cuda.get_device_name(0)
    record("device_named", value=name)
    tensor = torch.arange(1024, device="cuda", dtype=torch.float32)
    record("tensor_allocated")
    value = float((tensor * tensor).sum())
    record("kernel_complete", value=value)
    torch.cuda.synchronize()
    record("synchronized", status="pass")


if __name__ == "__main__":
    main()
