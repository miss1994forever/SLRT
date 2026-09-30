"""Frozen P6 online-quantile decision rule, driven by arriving windows.

The caller runs offset 0 and calls ``start_block`` when it becomes available,
then calls ``complete_block`` after offset 3 is known to exist. ``score`` must
use only paid-window posteriors, including the current offset 0. An incomplete
final block receives coverage only.
"""

from dataclasses import dataclass, field
from hashlib import sha256
import json
from pathlib import Path
from statistics import median
from typing import Callable, List, Optional


@dataclass
class OnlineQuantileController:
    score: Callable[[int, List[int]], float]
    selected: List[int] = field(default_factory=list)
    past_scores: List[float] = field(default_factory=list)
    balance: float = 0.0
    forced: int = 0
    autonomous: int = 0
    next_start: int = 0
    pending_start: Optional[int] = None
    finished: bool = False

    def start_block(self, start: int) -> None:
        """Record coverage when offset 0 is actually executed."""
        if self.finished or self.pending_start is not None or start != self.next_start:
            raise ValueError("offset0 must arrive once, in order")
        self.selected.append(start)
        self.pending_start = start

    def complete_block(self, start: int) -> bool:
        """Return whether to run buffered offset 2 for a complete block.

        Call only after all four positions of this block are available. The
        method has no video-length or end-of-stream argument.
        """
        if self.finished or start != self.pending_start:
            raise ValueError("block must have paid offset0 and be completed once")
        value = float(self.score(start, self.selected[:-1]))
        if value != value or value in (float("inf"), float("-inf")):
            raise ValueError("score must be finite")
        self.balance = min(2.0, self.balance + 0.5)
        execute = False
        if self.balance >= 1.0:
            if self.balance >= 2.0:
                execute = True
                self.forced += 1
            elif self.past_scores and value >= median(self.past_scores):
                execute = True
                self.autonomous += 1
        if execute:
            self.selected.append(start + 2)
            self.balance -= 1.0
        self.past_scores.append(value)
        self.pending_start = None
        self.next_start += 4
        return execute

    def finish(self) -> List[int]:
        """Close an incomplete tail; never spend or top up tokens."""
        if self.finished:
            raise ValueError("stream already finished")
        self.finished = True
        return sorted(self.selected)


def k_history_feature(frame_features, selected, current, previous_offsets):
    """153-dimensional frozen K feature from paid windows and current offset 0.

    Each frame feature is the frozen 36-dimensional posterior summary. The
    caller must provide summaries only for windows actually run by the policy.
    """
    visible = (selected + [current])[-16:]
    rows = [frame_features[index] for index in visible]
    if not rows or any(len(row) != 36 for row in rows):
        raise ValueError("expected 36-dimensional paid-window features")
    n = len(rows)
    mean = [sum(row[j] for row in rows) / n for j in range(36)]
    std = [(sum((row[j] - mean[j]) ** 2 for row in rows) / n) ** 0.5 for j in range(36)]
    counts = [previous_offsets.count(offset) / max(1, len(previous_offsets)) for offset in (1, 2, 3)]
    recent = previous_offsets[-4:]
    last = [0.0] * (4 - len(recent)) + [offset - 2 for offset in recent]
    from math import log1p
    return (list(rows[-1]) + mean + std + [rows[-1][j] - rows[0][j] for j in range(36)]
            + [log1p(current), log1p(len(selected))] + counts + last)


class FrozenRidgeBundle:
    """Five source-OOF ridge models with checked frozen asset identity."""

    def __init__(self, path, config_path):
        config_bytes = Path(config_path).read_bytes()
        payload = json.loads(Path(path).read_text())
        config = json.loads(config_bytes)
        if payload["config_sha256"] != sha256(config_bytes).hexdigest():
            raise ValueError("model/config identity mismatch")
        if payload["fit_dataset_sha256"] != config["fixed_assets"]["fit_p6_dataset_sha256"]:
            raise ValueError("model/fit dataset identity mismatch")
        if len(payload["folds"]) != 5:
            raise ValueError("expected five source folds")
        self.folds = payload["folds"]
        for fold in self.folds:
            if any(len(fold[key]) != 153 for key in ("mean", "scale", "weights")):
                raise ValueError("invalid K model dimension")

    def score(self, feature, fold):
        if len(feature) != 153 or not 0 <= fold < 5:
            raise ValueError("invalid feature or fold")
        model = self.folds[fold]
        return model["intercept"] + sum(
            (value - mean) / scale * weight for value, mean, scale, weight in
            zip(feature, model["mean"], model["scale"], model["weights"]))


def ridge_score_callback(bundle, fold, paid_frame_features):
    """Bind one frozen source fold to paid 36-D posterior summaries.

    Insert offset0 into ``paid_frame_features`` before completing each block.
    Insert offset2 only if ``complete_block`` returns true.
    """
    def score(start, selected):
        offsets = [2 for index in selected if index % 4 == 2]
        feature = k_history_feature(paid_frame_features, selected, start, offsets)
        return bundle.score(feature, fold)
    return score
