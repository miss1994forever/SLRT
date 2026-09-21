#!/usr/bin/env python3
"""Decoder-conditioned continuous counterfactual-value OOF experiment."""
import argparse
import gzip
import hashlib
import importlib.util
import json
import math
import os
import random
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.stats import spearmanr


ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "results/phoenix-2014t_ISLR"
DATA = BASE / "p3_fulltrain_robust_continuation_fit6378_dataset_v2_49faacc3"
DENSE = BASE / "train_dense_stride1_v1_49faacc3"
KROOT = BASE / "p3_fulltrain_robust_predictor_oof_v1_49faacc3"
FINGER = BASE / "p3_fulltrain_finger_rgb_tinycnn_robust_predictor_oof_v1_49faacc3"
FACE = BASE / "p3_fulltrain_face_rgb_tinycnn_robust_predictor_oof_v1_49faacc3"
OUTPUT = BASE / "p3_fulltrain_decoder_conditioned_value_oof_v1_49faacc3"
CONFIG = OUTPUT / "resolved_config_preregistered.json"
HISTORY = OUTPUT / "paid_decoder_history_blocks.npz"
EXECUTED = OUTPUT / "paid_decoder_logits_float16.npy"
HISTORY_MANIFEST = OUTPUT / "history_manifest.json"
OOF = OUTPUT / "oof_scores.npz"
METRICS = OUTPUT / "metrics.json"

FOLDS = 5
HISTORY_LENGTH = 16
PCA_DIM = 32
PCA_Q = 40
PCA_NITER = 3
EPOCHS = 20
BATCH_BLOCKS = 1024
NEUTRAL_RATIO = 4
SEEDS = [261050, 261051, 261052]
PCA_SEED = 261060
TOP_K = 769
EXPECTED_GPU = {"pci_bus_id": "00000000:61:00.0",
                "uuid": "GPU-91f9e38c-0a59-15c0-bd08-6a4f89da07ed"}


def imp(name):
    path = ROOT / f"tools/{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


FULL = imp("analyze_phoenix_full_robust_predictor_oof")
PARTIAL = FULL.PARTIAL
BUILDDATA = imp("build_phoenix_full_robust_continuation_dataset")


def utcnow():
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_sha(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def source_fold(source):
    return PARTIAL.source_fold(source)


def verify_cuda_device():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable; GPU stages must run outside the sandbox")
    if torch.cuda.device_count() != 1 or os.environ.get("CUDA_VISIBLE_DEVICES") != EXPECTED_GPU["uuid"]:
        raise RuntimeError({"visible": os.environ.get("CUDA_VISIBLE_DEVICES"), "expected": EXPECTED_GPU})
    props = torch.cuda.get_device_properties(0)
    return {**EXPECTED_GPU, "name": props.name, "total_memory_bytes": props.total_memory}


def frozen_config():
    required = [
        DATA / "dataset_manifest.json", DENSE / "protocol_manifest.json",
        KROOT / "oof_scores.npz", FINGER / "resolved_config_preregistered.json",
        FINGER / "feature_manifest.json", FACE / "resolved_config_preregistered.json",
        FACE / "feature_manifest.json",
    ]
    if any(not path.is_file() for path in required):
        raise FileNotFoundError("required frozen upstream asset is absent")
    identities = {str(path): sha256_file(path) for path in required}
    return {
        "experiment": "DCV-v1 decoder-conditioned continuous counterfactual-value source-disjoint OOF",
        "created_before_history_archive_or_DCV_outcomes": True,
        "motivation": "test candidate visual information conditional on already-paid decoder state, rather than improve RGB appearance alone",
        "scope": {"samples": 6378, "sources": 578, "blocks": 183401, "side_rows": 366802},
        "upstream_identities": identities,
        "face_and_finger_family_completion": "required artifacts present; their metrics are not read during preregistration",
        "decision_state": {
            "trajectory": "same frozen robust-oracle train-fit trajectory already used to create B0/K/HF/RF OOF rows",
            "paid_history": "last 16 actually selected windows through current block offset0",
            "allowed_logits": "only logits of already-paid selected windows",
            "forbidden_logits": "current unexecuted offset1/offset2/offset3 candidates",
            "history_metadata": ["offset-mod4 one-hot (4)", "log gap", "log absolute index",
                                 "raw-logit softmax entropy", "top1-top2 margin", "blank probability",
                                 "four deterministic top1-token hash signs"],
            "history_metadata_width": 13,
            "B0": "49-D candidate bookkeeping plus past-paid prefix summary",
        },
        "candidate_visual": {
            "finger": "frozen fold-specific 264-D four-segment feature from FINGER_RGB",
            "face": "frozen fold-specific 132-D four-segment feature from FACE_RGB",
            "combined_width": 396,
            "candidate_window": "candidate_start+[-7,...,+8] with the frozen bounded-lookahead padding rules",
            "no_encoder_retraining": True,
        },
        "fold_preprocessing": {
            "decoder_logits": {"method": "torch randomized low-rank PCA", "dimensions": PCA_DIM,
                               "q": PCA_Q, "niter": PCA_NITER,
                               "fit": "already-paid executions from outer-train sources only",
                               "seed": "261060+fold"},
            "standardization": "B0, visual, PCA coordinates, and history metadata fit on outer-train sources only",
        },
        "model": {
            "history": "GRU(input=45=PCA32+meta13, hidden=32)",
            "visual_tower": "Linear(396,64)-SiLU-Linear(64,32)",
            "base_tower": "Linear(49,32)-SiLU",
            "context": "Linear(base32+history32,32)-SiLU",
            "interaction": "concat visual32, context32, elementwise product32, absolute difference32",
            "head": "Linear(128,64)-SiLU-Linear(64,2)",
            "outputs": ["center-continuation center-minus-side edit-error advantage",
                        "late-continuation center-minus-side edit-error advantage"],
            "score": "minimum of the two predicted advantages",
        },
        "training": {
            "block_sampling": "all blocks with any nonzero two-continuation side advantage plus up to four times as many neutral blocks, resampled deterministically each epoch",
            "loss": "SmoothL1 on both continuous advantages weighted by 1+4*abs(target), 4x penalty when target<=0 but prediction>0; plus 0.5x center-anchor hinge and 0.5x within-block ordering hinge",
            "center_anchor_margin": 0.5, "neutral_deadzone": 0.25,
            "optimizer": "AdamW", "lr": 0.001, "weight_decay": 0.0001,
            "epochs": EPOCHS, "batch_blocks": BATCH_BLOCKS, "seeds": SEEDS,
            "early_stopping": False, "scheduler": None,
        },
        "evaluation": {
            "primary_comparison": "DCV minus frozen K signed top-769 utility",
            "diagnostics": ["PR-AUC", "recall/precision@769", "positive utility", "harmful actions",
                            "source-cluster bootstrap 95% CI", "continuous-target MAE and Spearman"],
            "top_K": TOP_K, "strong_go": "DCV signed top-769 utility >=249 errors",
            "HF_RF_comparisons": "diagnostic only after DCV OOF is complete; cannot replace K comparison or gate",
        },
        "gpu": {**EXPECTED_GPU, "forbidden_pci": ["00000000:25:00.0", "00000000:41:00.0"],
                "gpu0_forbidden": True},
        "forbidden": ["calibration/dev/test", "closed-loop before gate", "unexecuted candidate logits",
                      "reference/future/EOS as predictor inputs", "post-OOF loss/model/PCA/history tuning", "git commit"],
        "output_directory": str(OUTPUT),
    }


def preregister():
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    OUTPUT.mkdir(parents=True)
    value = frozen_config()
    CONFIG.write_text(json.dumps(value, indent=2) + "\n")
    return value


def load_config():
    value = json.loads(CONFIG.read_text())
    if value != frozen_config():
        raise RuntimeError("preregistered configuration changed")
    return value


def token_hash4(token):
    digest = hashlib.sha256(str(int(token)).encode()).digest()
    return [1.0 if digest[i] & 1 else -1.0 for i in range(4)]


def execution_meta(logit, start, previous_start, blank):
    value = np.asarray(logit, np.float32)
    shifted = value - value.max()
    probability = np.exp(shifted); probability /= probability.sum()
    top = np.argpartition(probability, -2)[-2:]
    top = top[np.argsort(probability[top])]
    top1 = int(top[-1])
    entropy = float(-np.sum(probability * np.log(np.maximum(probability, 1e-12))))
    gap = int(start + 1) if previous_start is None else int(start - previous_start)
    offset = [float(start % 4 == index) for index in range(4)]
    return np.asarray(offset + [math.log1p(gap), math.log1p(int(start) + 1), entropy,
                         float(probability[top[-1]] - probability[top[-2]]), float(probability[blank]),
                         *token_hash4(top1)], np.float32)


def parse_label_shard(path):
    blocks = defaultdict(list)
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            if row.get("type") == "block":
                blocks[row["sample_id"]].append(row)
    return blocks


def build_history():
    load_config()
    if any(path.exists() for path in (HISTORY, EXECUTED, HISTORY_MANIFEST)):
        raise FileExistsError("history assets are non-overwriting")
    vocab = json.loads(BUILDDATA.CHRON.DEFAULT_VOCAB.read_text())
    blank = vocab.index("<blank>")
    base_parts, target_parts, reward_parts, fold_parts, source_parts = [], [], [], [], []
    history_parts, length_parts, sample_parts, starts_parts, arrival_parts = [], [], [], [], []
    executed_logits, executed_meta, executed_folds = [], [], []
    source_to_index = {}; sample_to_index = {}; sample_names = []
    started = time.perf_counter(); block_count = 0
    label_paths = FULL.label_paths()
    for number, label_path in enumerate(label_paths):
        shard = int(label_path.name.split("-")[1])
        blocks = parse_label_shard(label_path)
        results, logits, _ = BUILDDATA.BUILDER.BUILDER.validate_dense_shard(DENSE, shard, len(label_paths), verify_hashes=False)
        for name, rows in blocks.items():
            rows.sort(key=lambda row: int(row["block_start"]))
            source = rows[0]["source_video_id"]; fold = source_fold(source)
            source_index = source_to_index.setdefault(source, len(source_to_index))
            if name not in sample_to_index:
                sample_to_index[name] = len(sample_names); sample_names.append(name)
            sample_index = sample_to_index[name]
            sample_logits = np.asarray(logits[name], np.float32)
            selected_starts, selected_global = [], []
            previous = None
            for row in rows:
                skeleton = int(row["block_start"])
                for start in (skeleton,):
                    selected_starts.append(start); selected_global.append(len(executed_logits))
                    executed_logits.append(np.asarray(sample_logits[start], np.float16))
                    executed_meta.append(execution_meta(sample_logits[start], start, previous, blank))
                    executed_folds.append(fold); previous = start
                available = selected_global[-HISTORY_LENGTH:]
                history_index = np.full(HISTORY_LENGTH, -1, np.int32)
                history_index[-len(available):] = available
                bases, targets, rewards = [], [], []
                for candidate in (0, 2):
                    inputs = row["predictor_inputs"][candidate]
                    bases.append(np.concatenate([np.asarray(inputs["bookkeeping"], np.float32),
                                                 np.asarray(inputs["prefix"], np.float32)]))
                    center = row["label"]["terminal_errors_center_future"]
                    late = row["label"]["terminal_errors_late_future"]
                    targets.append([int(center[1]) - int(center[candidate]), int(late[1]) - int(late[candidate])])
                    rewards.append(int(row["label"]["robust_reward_by_candidate"][candidate]))
                base_parts.append(bases); target_parts.append(targets); reward_parts.append(rewards)
                fold_parts.append(fold); source_parts.append(source_index); history_parts.append(history_index)
                length_parts.append(len(available)); sample_parts.append(sample_index)
                starts_parts.append([int(row["candidate_starts"][0]), int(row["candidate_starts"][2])])
                arrival_parts.append(int(row["decision_arrival"])); block_count += 1
                choice = int(row["label"]["teacher_choice"]); chosen = int(row["candidate_starts"][choice])
                selected_starts.append(chosen); selected_global.append(len(executed_logits))
                executed_logits.append(np.asarray(sample_logits[chosen], np.float16))
                executed_meta.append(execution_meta(sample_logits[chosen], chosen, previous, blank))
                executed_folds.append(fold); previous = chosen
        print(json.dumps({"history_shards": number + 1, "blocks": block_count,
                          "executions": len(executed_logits), "seconds": round(time.perf_counter() - started, 1)}), flush=True)
    base = np.asarray(base_parts, np.float32); targets = np.asarray(target_parts, np.int8)
    rewards = np.asarray(reward_parts, np.int8); folds = np.asarray(fold_parts, np.int8)
    sources = np.asarray(source_parts, np.int16); history = np.asarray(history_parts, np.int32)
    if base.shape != (183401, 2, 49) or targets.shape != (183401, 2, 2):
        raise RuntimeError({"base": base.shape, "targets": targets.shape})
    if not np.array_equal(targets.min(axis=2), rewards):
        raise RuntimeError("continuous targets do not reproduce frozen robust rewards")
    old = np.load(KROOT / "oof_scores.npz")
    if not np.array_equal(rewards.reshape(-1), old["rewards"]) or not np.array_equal(np.repeat(folds, 2), old["folds"]):
        raise RuntimeError("frozen OOF reward/fold row parity failure")
    logits_array = np.asarray(executed_logits, np.float16)
    np.save(EXECUTED, logits_array)
    np.savez_compressed(HISTORY, base=base, targets=targets, rewards=rewards, folds=folds,
                        source_indices=sources, history_indices=history,
                        history_lengths=np.asarray(length_parts, np.int8),
                        sample_indices=np.asarray(sample_parts, np.int16), candidate_starts=np.asarray(starts_parts, np.int32),
                        decision_arrivals=np.asarray(arrival_parts, np.int32), sample_names=np.asarray(sample_names),
                        executed_meta=np.asarray(executed_meta, np.float32), executed_folds=np.asarray(executed_folds, np.int8))
    counts = Counter(map(int, rewards.reshape(-1)))
    nonzero_cont = int(np.any(targets != 0, axis=2).sum())
    manifest = {"status": "complete", "created_utc": utcnow(), "blocks": len(base), "side_rows": 2 * len(base),
                "samples": len(sample_names), "sources": len(source_to_index), "executions": len(logits_array),
                "vocab_width": logits_array.shape[1], "history_length": HISTORY_LENGTH,
                "robust_reward_counts": dict(sorted(counts.items())),
                "blocks_with_any_nonzero_continuation_advantage": nonzero_cont,
                "files": {"history": {"path": str(HISTORY), "bytes": HISTORY.stat().st_size, "sha256": sha256_file(HISTORY)},
                          "executed_logits": {"path": str(EXECUTED), "bytes": EXECUTED.stat().st_size, "sha256": sha256_file(EXECUTED)}},
                "row_parity_with_frozen_K": True, "candidate_unexecuted_logits_in_inputs": False,
                "seconds": time.perf_counter() - started}
    HISTORY_MANIFEST.write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def generic_summary(frames, start, arrival):
    n = len(frames); offsets = np.arange(-7, 9, dtype=np.int64) + int(start)
    deadline = min(n - 1, int(arrival) + 8)
    if min(int(offsets[-1]), n - 1) > deadline:
        raise RuntimeError("candidate visibility deadline violation")
    return np.asarray(frames[np.clip(offsets, 0, n - 1)], np.float32).reshape(4, 4, -1).mean(1).reshape(-1)


def feature_archive(root, stem):
    index = np.load(root / "ssl_crop_index.npz")
    names = [str(value) for value in index["video_names"].tolist()]
    offsets = np.asarray(index["video_offsets"], np.int64)
    slices = {name: slice(int(offsets[i]), int(offsets[i + 1])) for i, name in enumerate(names)}
    path = root / f"{stem}"
    return np.load(path, mmap_mode="r"), slices


def visual_matrix(data, fold):
    finger, fslices = feature_archive(FINGER, f"finger_frame_embeddings_fold{fold}_float16.npy")
    face, rslices = feature_archive(FACE, f"face_frame_embeddings_fold{fold}_float16.npy")
    sample_names = [str(value) for value in data["sample_names"].tolist()]
    output = np.empty((len(data["folds"]), 2, 396), np.float32)
    offsets = np.arange(-7, 9, dtype=np.int64)
    for sample_index, name in enumerate(sample_names):
        blocks = np.flatnonzero(data["sample_indices"] == sample_index)
        if not len(blocks):
            continue
        starts = data["candidate_starts"][blocks].astype(np.int64)
        arrival = data["decision_arrivals"][blocks].astype(np.int64)
        def summarize(frames):
            n = len(frames)
            indices = starts[:, :, None] + offsets[None, None, :]
            deadline = np.minimum(n - 1, arrival + 8)
            if np.any(np.minimum(indices[:, :, -1], n - 1) > deadline[:, None]):
                raise RuntimeError("candidate visibility deadline violation")
            values = np.asarray(frames[np.clip(indices, 0, n - 1)], np.float32)
            return values.reshape(len(blocks), 2, 4, 4, -1).mean(3).reshape(len(blocks), 2, -1)
        output[blocks] = np.concatenate([summarize(finger[fslices[name]]), summarize(face[rslices[name]])], axis=2)
    return output


def mean_scale(values):
    mean = values.mean(axis=0, dtype=np.float64).astype(np.float32)
    scale = values.std(axis=0, dtype=np.float64).astype(np.float32); scale[scale < 1e-6] = 1.0
    return mean, scale


class DCV(nn.Module):
    def __init__(self):
        super().__init__()
        self.history = nn.GRU(PCA_DIM + 13, 32, batch_first=True)
        self.visual = nn.Sequential(nn.Linear(396, 64), nn.SiLU(), nn.Linear(64, 32))
        self.base = nn.Sequential(nn.Linear(49, 32), nn.SiLU())
        self.context = nn.Sequential(nn.Linear(64, 32), nn.SiLU())
        self.head = nn.Sequential(nn.Linear(128, 64), nn.SiLU(), nn.Linear(64, 2))

    def forward(self, base, visual, history, lengths):
        out, _ = self.history(history)
        h = out[torch.arange(len(out), device=out.device), lengths - 1]
        b, c, _ = base.shape
        v = self.visual(visual.reshape(b * c, -1)).reshape(b, c, 32)
        bstate = self.base(base.reshape(b * c, -1)).reshape(b, c, 32)
        context = self.context(torch.cat([bstate, h[:, None, :].expand(-1, c, -1)], dim=2))
        fused = torch.cat([v, context, v * context, torch.abs(v - context)], dim=2)
        return self.head(fused.reshape(b * c, -1)).reshape(b, c, 2)


def dcv_loss(pred, target):
    per = F.smooth_l1_loss(pred, target, reduction="none")
    weight = 1.0 + 4.0 * torch.abs(target)
    weight = weight * torch.where((target <= 0) & (pred > 0), torch.full_like(weight, 4.0), torch.ones_like(weight))
    regression = (per * weight).mean()
    p = pred.min(2).values; t = target.min(2).values
    anchor = torch.where(t > 0, F.relu(0.5 - p), torch.where(t < 0, F.relu(0.5 + p), F.relu(torch.abs(p) - 0.25))).mean()
    difference = t[:, 0] - t[:, 1]
    ordering = (F.relu(torch.minimum(torch.ones_like(difference), torch.abs(difference)) -
                       torch.sign(difference) * (p[:, 0] - p[:, 1])) * (difference != 0)).sum() / (difference != 0).sum().clamp_min(1)
    return regression + 0.5 * anchor + 0.5 * ordering


def spearman(x, y):
    value = spearmanr(np.asarray(x), np.asarray(y)).statistic
    return float(value) if np.isfinite(value) else None


def run_oof():
    cfg = load_config(); gpu = verify_cuda_device()
    if not HISTORY_MANIFEST.is_file() or OOF.exists() or METRICS.exists():
        raise RuntimeError("complete history required; OOF is non-overwriting")
    device = torch.device("cuda")
    torch.set_num_threads(min(8, os.cpu_count() or 1))
    packed = np.load(HISTORY)
    data = {key: packed[key] for key in packed.files}
    packed.close()
    executed = np.load(EXECUTED, mmap_mode="r")
    blocks = len(data["folds"]); scores = np.full((len(SEEDS), blocks, 2), np.nan, np.float32)
    predictions = np.full((len(SEEDS), blocks, 2, 2), np.nan, np.float32)
    fold_records = []; started = time.perf_counter()
    for fold in range(FOLDS):
        train_blocks = data["folds"] != fold; eval_blocks = ~train_blocks
        exec_train = data["executed_folds"] != fold
        torch.manual_seed(PCA_SEED + fold)
        xtrain = torch.from_numpy(np.asarray(executed[exec_train], np.float32)).to(device)
        pca_mean = xtrain.mean(0); centered = xtrain - pca_mean
        _, singular, components = torch.pca_lowrank(centered, q=PCA_Q, center=False, niter=PCA_NITER)
        components = components[:, :PCA_DIM]
        pca_scale = (singular[:PCA_DIM] / math.sqrt(max(1, len(xtrain) - 1))).clamp_min(1e-6)
        explained_fraction = float(singular[:PCA_DIM].square().sum() / centered.square().sum())
        pca_path = OUTPUT / f"decoder_pca_fold{fold}.npz"
        np.savez_compressed(pca_path, mean=pca_mean.cpu().numpy(), components=components.cpu().numpy(),
                            scale=pca_scale.cpu().numpy())
        projected = np.empty((len(executed), PCA_DIM), np.float32)
        with torch.no_grad():
            for left in range(0, len(executed), 8192):
                value = torch.from_numpy(np.asarray(executed[left:left + 8192], np.float32)).to(device)
                projected[left:left + len(value)] = (((value - pca_mean) @ components) / pca_scale).cpu().numpy()
        del xtrain, centered
        visual = visual_matrix(data, fold)
        bm, bs = mean_scale(data["base"][train_blocks].reshape(-1, 49))
        vm, vs = mean_scale(visual[train_blocks].reshape(-1, 396))
        mm, ms = mean_scale(data["executed_meta"][exec_train])
        base = (data["base"] - bm) / bs; visual = (visual - vm) / vs
        exec_features = np.concatenate([projected, (data["executed_meta"] - mm) / ms], axis=1).astype(np.float32)
        history_index = data["history_indices"]
        history = exec_features[np.maximum(history_index, 0)]; history[history_index < 0] = 0
        base_t = torch.from_numpy(np.asarray(base, np.float32)).to(device)
        visual_t = torch.from_numpy(np.asarray(visual, np.float32)).to(device)
        history_t = torch.from_numpy(history).to(device)
        lengths_t = torch.from_numpy(data["history_lengths"].astype(np.int64)).to(device)
        targets_t = torch.from_numpy(data["targets"].astype(np.float32)).to(device)
        informative = np.flatnonzero(train_blocks & np.any(data["targets"] != 0, axis=(1, 2)))
        neutral = np.flatnonzero(train_blocks & ~np.any(data["targets"] != 0, axis=(1, 2)))
        eval_index = np.flatnonzero(eval_blocks)
        seed_losses = []
        for seed_index, seed in enumerate(SEEDS):
            random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
            model = DCV().to(device); optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.0001)
            generator = torch.Generator().manual_seed(seed + fold)
            losses = []
            for _ in range(EPOCHS):
                neutral_count = min(len(neutral), NEUTRAL_RATIO * len(informative))
                chosen_neutral = neutral[torch.randperm(len(neutral), generator=generator)[:neutral_count].numpy()]
                chosen = np.concatenate([informative, chosen_neutral])
                chosen = chosen[torch.randperm(len(chosen), generator=generator).numpy()]
                total = 0.0
                for left in range(0, len(chosen), BATCH_BLOCKS):
                    idx = torch.from_numpy(chosen[left:left + BATCH_BLOCKS]).to(device)
                    pred = model(base_t[idx], visual_t[idx], history_t[idx], lengths_t[idx])
                    loss = dcv_loss(pred, targets_t[idx])
                    optimizer.zero_grad(set_to_none=True); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); optimizer.step()
                    total += float(loss.detach()) * len(idx)
                losses.append(total / len(chosen))
            model.eval(); out = []
            with torch.no_grad():
                for left in range(0, len(eval_index), 4096):
                    idx = torch.from_numpy(eval_index[left:left + 4096]).to(device)
                    out.append(model(base_t[idx], visual_t[idx], history_t[idx], lengths_t[idx]).cpu().numpy())
            out = np.concatenate(out)
            predictions[seed_index, eval_index] = out
            scores[seed_index, eval_index] = out.min(axis=2)
            seed_losses.append(losses)
            del model, optimizer
        fold_records.append({"fold": fold, "train_blocks": int(train_blocks.sum()), "eval_blocks": int(eval_blocks.sum()),
                             "informative_train_blocks": len(informative), "sampled_neutral_per_epoch": min(len(neutral), NEUTRAL_RATIO * len(informative)),
                             "PCA_train_executions": int(exec_train.sum()),
                             "PCA_explained_fraction": explained_fraction,
                             "PCA_file": str(pca_path), "PCA_sha256": sha256_file(pca_path),
                             "seed_epoch_losses": seed_losses})
        print(json.dumps({"dcv_fold": fold, "seconds": round(time.perf_counter() - started, 1)}), flush=True)
        del base_t, visual_t, history_t, targets_t, history, exec_features, visual, projected
        torch.cuda.empty_cache()
    score = scores.mean(0).reshape(-1); pred = predictions.mean(0).reshape(-1, 2)
    rewards = data["rewards"].reshape(-1).astype(np.int64); positive = (rewards > 0).astype(np.float32)
    old = np.load(KROOT / "oof_scores.npz")
    metric = {"DCV": PARTIAL.metrics(positive, rewards, score),
              "K": PARTIAL.metrics(positive, rewards, old["signed_K"]),
              "B0": PARTIAL.metrics(positive, rewards, old["signed_B0"])}
    source_rows = np.repeat(data["source_indices"], 2)
    ci = FULL.source_bootstrap_delta(rewards, source_rows, score, old["signed_K"], TOP_K)
    utility = metric["DCV"]["top_K_signed_robust_utility"]; delta = utility - metric["K"]["top_K_signed_robust_utility"]
    actual = data["targets"].reshape(-1, 2).astype(np.float32)
    gate = utility >= 249
    value = {"created_utc": utcnow(), "device": gpu, "rows": len(rewards), "blocks": blocks, "sources": 578,
             "model_parameters": sum(p.numel() for p in DCV().parameters()), "folds": fold_records,
             "metrics": metric, "DCV_minus_K_top_K_utility": delta,
             "DCV_minus_K_source_bootstrap_95pct": ci,
             "continuous_targets": {"center_MAE": float(np.mean(np.abs(pred[:, 0] - actual[:, 0]))),
                                    "late_MAE": float(np.mean(np.abs(pred[:, 1] - actual[:, 1]))),
                                    "center_spearman": spearman(pred[:, 0], actual[:, 0]),
                                    "late_spearman": spearman(pred[:, 1], actual[:, 1])},
             "gate": {"threshold_errors": 249, "DCV_utility": utility, "passed": gate,
                      "decision": "strong-go" if gate else ("ranking-only" if delta > 0 and ci[0] > 0 else "no-go"),
                      "action": "unlock_separate_frozen_closed_loop" if gate else "do_not_run_closed_loop"},
             "elapsed_seconds": time.perf_counter() - started}
    np.savez_compressed(OOF, folds=np.repeat(data["folds"], 2), rewards=rewards, source_indices=source_rows,
                        signed_B0=old["signed_B0"], signed_K=old["signed_K"], dcv_score=score,
                        dcv_prediction=pred, continuous_targets=actual)
    value["oof_scores"] = {"path": str(OOF), "bytes": OOF.stat().st_size, "sha256": sha256_file(OOF)}
    METRICS.write_text(json.dumps(value, indent=2) + "\n")
    return value


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("preregister", "build-history", "run-oof"))
    args = parser.parse_args()
    functions = {"preregister": preregister, "build-history": build_history, "run-oof": run_oof}
    print(json.dumps(functions[args.command](), indent=2))


if __name__ == "__main__":
    main()
