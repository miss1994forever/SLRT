"""Unified, deterministic temporal-window sampling for Online CSLR inference."""

import math
from typing import Dict, List, Optional, Tuple

import torch

from utils.adaptive_stride import adaptive_window_starts


SAMPLING_MODES = ("fixed", "uniform_rate", "adaptive_motion")


def resolve_sampling_config(
    sampling_config: Optional[Dict],
    fixed_stride: int = 1,
    adaptive_config: Optional[Dict] = None,
) -> Dict:
    """Resolve new sampling fields while preserving legacy adaptive settings."""
    sampling = dict(sampling_config or {})
    adaptive = dict(adaptive_config or {})
    mode = sampling.get("mode")
    if mode is None:
        enabled = adaptive.get("enabled", adaptive.get("enable", False))
        mode = "adaptive_motion" if enabled else "fixed"
    if mode not in SAMPLING_MODES:
        raise ValueError(f"sampling mode must be one of {SAMPLING_MODES}, got {mode!r}")

    resolved = dict(adaptive) if mode == "adaptive_motion" else {}
    resolved.update(sampling)
    resolved["mode"] = mode
    resolved.setdefault("fixed_stride", int(fixed_stride))
    resolved.setdefault("uniform_mean_stride", 1.0)
    return resolved


def _validate_total_frames(total_frames: int) -> int:
    total_frames = int(total_frames)
    if total_frames < 0:
        raise ValueError("total_frames must be non-negative")
    return total_frames


def _fixed_starts(total_frames: int, stride: int) -> Tuple[List[int], List[Dict]]:
    if stride < 1:
        raise ValueError("fixed_stride must be at least 1")
    starts = list(range(0, total_frames, stride)) or [0]
    metadata = [
        {
            "start": int(start),
            "stride": int(stride),
            "motion": 0.0,
            "quality": 1.0,
            "sampling_mode": "fixed",
        }
        for start in starts
    ]
    return starts, metadata


def _uniform_rate_starts(total_frames: int, mean_stride: float) -> Tuple[List[int], List[Dict]]:
    if not math.isfinite(mean_stride) or mean_stride < 1.0:
        raise ValueError("uniform_mean_stride must be finite and at least 1.0")
    starts: List[int] = []
    index = 0
    while True:
        start = int(math.floor(index * mean_stride + 0.5))
        if start >= total_frames:
            break
        if not starts or start > starts[-1]:
            starts.append(start)
        index += 1
    if not starts:
        starts = [0]

    metadata = []
    for index, start in enumerate(starts):
        if index + 1 < len(starts):
            stride = starts[index + 1] - start
        else:
            next_start = int(math.floor((index + 1) * mean_stride + 0.5))
            stride = max(1, next_start - start)
        metadata.append(
            {
                "start": int(start),
                "stride": int(stride),
                "motion": 0.0,
                "quality": 1.0,
                "sampling_mode": "uniform_rate",
            }
        )
    return starts, metadata


def generate_window_starts(
    mode: str,
    total_frames: int,
    keypoints: Optional[torch.Tensor],
    config: Optional[Dict] = None,
) -> Tuple[List[int], List[Dict]]:
    """Generate starts and common metadata for a registered sampling mode."""
    total_frames = _validate_total_frames(total_frames)
    config = dict(config or {})
    if mode not in SAMPLING_MODES:
        raise ValueError(f"sampling mode must be one of {SAMPLING_MODES}, got {mode!r}")

    if mode == "fixed":
        return _fixed_starts(total_frames, int(config.get("fixed_stride", 1)))
    if mode == "uniform_rate":
        return _uniform_rate_starts(total_frames, float(config.get("uniform_mean_stride", 1.0)))

    if keypoints is None:
        raise ValueError("adaptive_motion sampling requires keypoints")
    adaptive_config = dict(config)
    adaptive_config.pop("mode", None)
    adaptive_config.pop("fixed_stride", None)
    adaptive_config.pop("uniform_mean_stride", None)
    adaptive_config["enabled"] = True
    starts, metadata = adaptive_window_starts(total_frames, keypoints, adaptive_config)
    metadata = [dict(item, sampling_mode="adaptive_motion") for item in metadata]
    return starts, metadata
