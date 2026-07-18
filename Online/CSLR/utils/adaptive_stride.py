"""Causal, robust adaptive-stride utilities for Online CSLR inference.

This module does not depend on model weights.  It only chooses temporal window
starts from keypoint motion and provides time-aware voting for window logits.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import torch


@dataclass(frozen=True)
class AdaptiveStrideConfig:
    enabled: bool = False
    min_stride: int = 1
    max_stride: int = 3
    keypoint_confidence_threshold: float = 0.2
    ema_decay: float = 0.4
    quantile_low: float = 0.2
    quantile_high: float = 0.7
    calibration_window_frames: int = 48
    warmup_frames: int = 16
    min_valid_keypoints: int = 4
    epsilon: float = 1e-6

    @classmethod
    def from_dict(cls, value: Optional[Dict]) -> "AdaptiveStrideConfig":
        value = value or {}
        # Accept the previous prototype names so old commands remain runnable.
        enabled = value.get("enabled", value.get("enable", False))
        confidence = value.get(
            "keypoint_confidence_threshold", value.get("confidence_threshold", value.get("conf_thr", 0.2))
        )
        ema_decay = value.get("ema_decay", value.get("ema_beta", 0.4))
        cfg = cls(
            enabled=bool(enabled),
            min_stride=int(value.get("min_stride", 1)),
            max_stride=int(value.get("max_stride", 3)),
            keypoint_confidence_threshold=float(confidence),
            ema_decay=float(ema_decay),
            quantile_low=float(value.get("quantile_low", 0.2)),
            quantile_high=float(value.get("quantile_high", 0.7)),
            calibration_window_frames=int(value.get("calibration_window_frames", 48)),
            warmup_frames=int(value.get("warmup_frames", 16)),
            min_valid_keypoints=int(value.get("min_valid_keypoints", 4)),
            epsilon=float(value.get("epsilon", 1e-6)),
        )
        cfg.validate()
        return cfg

    def validate(self) -> None:
        if self.min_stride < 1 or self.max_stride < self.min_stride:
            raise ValueError("adaptive stride requires 1 <= min_stride <= max_stride")
        if not 0.0 <= self.keypoint_confidence_threshold <= 1.0:
            raise ValueError("keypoint_confidence_threshold must be in [0, 1]")
        if not 0.0 <= self.ema_decay < 1.0:
            raise ValueError("ema_decay must be in [0, 1)")
        if not 0.0 <= self.quantile_low < self.quantile_high <= 1.0:
            raise ValueError("quantiles must satisfy 0 <= low < high <= 1")
        if self.calibration_window_frames < 2 or self.warmup_frames < 1:
            raise ValueError("calibration_window_frames >= 2 and warmup_frames >= 1 are required")
        if self.min_valid_keypoints < 1 or self.epsilon <= 0:
            raise ValueError("min_valid_keypoints and epsilon must be positive")


def _robust_frame_speeds(keypoints: torch.Tensor, cfg: AdaptiveStrideConfig) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return normalized median motion and valid-keypoint ratio for each frame."""
    if keypoints.ndim != 3 or keypoints.shape[0] == 0:
        raise ValueError("keypoints must have shape [T, K, C]")
    total_frames, keypoint_count = keypoints.shape[:2]
    speed = torch.zeros(total_frames, dtype=torch.float32, device=keypoints.device)
    quality = torch.zeros_like(speed)
    if total_frames == 1:
        return speed, quality

    coords = keypoints[..., :2].float()
    finite = torch.isfinite(coords).all(dim=-1)
    if keypoints.shape[-1] >= 3:
        confidence = keypoints[..., 2].float()
        finite = finite & torch.isfinite(confidence)
        visible = finite & (confidence >= cfg.keypoint_confidence_threshold)
    else:
        visible = finite

    for frame in range(1, total_frames):
        valid = visible[frame - 1] & visible[frame]
        valid_count = int(valid.sum().item())
        quality[frame] = valid_count / max(keypoint_count, 1)
        if valid_count < cfg.min_valid_keypoints:
            speed[frame] = float("nan")
            continue

        previous = coords[frame - 1, valid]
        current = coords[frame, valid]
        combined = torch.cat([previous, current], dim=0)
        extent = combined.amax(dim=0) - combined.amin(dim=0)
        scale = torch.linalg.norm(extent).clamp_min(cfg.epsilon)
        displacement = torch.linalg.norm(current - previous, dim=-1) / scale
        # Median makes isolated pose-estimator jumps much less influential.
        speed[frame] = displacement.median()

    speed[0] = speed[1] if torch.isfinite(speed[1]) else 0.0
    quality[0] = quality[1]
    return speed, quality


def _causal_ema(values: torch.Tensor, decay: float) -> torch.Tensor:
    smoothed = torch.empty_like(values)
    previous = torch.tensor(0.0, dtype=values.dtype, device=values.device)
    initialized = False
    for index, value in enumerate(values):
        if not torch.isfinite(value):
            smoothed[index] = previous if initialized else float("nan")
            continue
        previous = value if not initialized else decay * previous + (1.0 - decay) * value
        initialized = True
        smoothed[index] = previous
    return smoothed


def adaptive_window_starts(
    total_frames: int,
    keypoints: torch.Tensor,
    config: Optional[Dict] = None,
) -> Tuple[List[int], List[Dict[str, float]]]:
    """Generate strictly increasing causal starts and per-window diagnostics."""
    cfg = AdaptiveStrideConfig.from_dict(config)
    if total_frames <= 0:
        return [0], [{"start": 0, "stride": 1, "motion": 0.0, "quality": 0.0}]
    if not cfg.enabled:
        starts = list(range(total_frames))
        return starts, [
            {"start": int(start), "stride": 1, "motion": 0.0, "quality": 1.0} for start in starts
        ]

    speed, quality = _robust_frame_speeds(keypoints[:total_frames], cfg)
    speed = _causal_ema(speed, cfg.ema_decay)
    starts: List[int] = []
    metadata: List[Dict[str, float]] = []
    current = 0
    while current < total_frames:
        history_start = max(0, current - cfg.calibration_window_frames + 1)
        history = speed[history_start : current + 1]
        history = history[torch.isfinite(history)]
        current_speed = speed[current]

        insufficient_quality = float(quality[current].item()) * keypoints.shape[1] < cfg.min_valid_keypoints
        if (
            current < cfg.warmup_frames
            or history.numel() < cfg.warmup_frames
            or not torch.isfinite(current_speed)
            or insufficient_quality
        ):
            stride = cfg.min_stride
            motion = 1.0  # conservative: uncertain motion is treated as fast
        else:
            low = torch.quantile(history, cfg.quantile_low)
            high = torch.quantile(history, cfg.quantile_high)
            span = high - low
            if not torch.isfinite(span) or float(span.item()) <= cfg.epsilon:
                # A stable low-motion history is a hold; an uncertain history was
                # already handled above and falls back to min_stride.
                motion = 0.0
            else:
                motion = float(torch.clamp((current_speed - low) / span, 0.0, 1.0).item())
            stride = int(round(cfg.max_stride - motion * (cfg.max_stride - cfg.min_stride)))
            stride = max(cfg.min_stride, min(cfg.max_stride, stride))

        starts.append(current)
        metadata.append(
            {
                "start": int(current),
                "stride": int(stride),
                "motion": float(motion),
                "speed": float(current_speed.item()) if torch.isfinite(current_speed) else float("nan"),
                "quality": float(quality[current].item()),
            }
        )
        current += stride

    return starts, metadata


def span_weighted_predictions(
    logits: torch.Tensor,
    centers: Sequence[float],
    span: float = 13.0,
    min_weight: float = 0.05,
) -> torch.Tensor:
    """Vote over a fixed temporal span using triangular probability weights."""
    if logits.ndim != 2 or logits.shape[0] != len(centers):
        raise ValueError("logits must be [N, C] and centers must contain N entries")
    if span <= 0 or not 0.0 < min_weight <= 1.0:
        raise ValueError("span must be positive and min_weight must be in (0, 1]")
    if logits.shape[0] == 0:
        return torch.empty(0, dtype=torch.long, device=logits.device)

    center_tensor = torch.as_tensor(centers, dtype=logits.dtype, device=logits.device)
    probabilities = logits.softmax(dim=-1)
    radius = span / 2.0
    predictions = []
    for center in center_tensor:
        distance = torch.abs(center_tensor - center)
        selected = distance <= radius
        weights = torch.clamp(1.0 - distance[selected] / max(radius, 1e-6), min=min_weight)
        weights = weights / weights.sum().clamp_min(1e-12)
        score = (probabilities[selected] * weights.unsqueeze(-1)).sum(dim=0)
        predictions.append(score.argmax())
    return torch.stack(predictions)
