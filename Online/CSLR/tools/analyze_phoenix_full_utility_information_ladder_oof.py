#!/usr/bin/env python3
"""Privileged, nondeployable information-sufficiency ladder for robust utility."""
import argparse
import hashlib
import importlib.util
import json
import math
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from scipy.stats import spearmanr


ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "results/phoenix-2014t_ISLR"
DATA = BASE / "p3_fulltrain_robust_continuation_fit6378_dataset_v2_49faacc3"
DENSE = BASE / "train_dense_stride1_v1_49faacc3"
KROOT = BASE / "p3_fulltrain_robust_predictor_oof_v1_49faacc3"
FINGER = BASE / "p3_fulltrain_finger_rgb_tinycnn_robust_predictor_oof_v1_49faacc3"
FACE = BASE / "p3_fulltrain_face_rgb_tinycnn_robust_predictor_oof_v1_49faacc3"
DCVROOT = BASE / "p3_fulltrain_decoder_conditioned_value_oof_v1_49faacc3"
OUTPUT = BASE / "p3_fulltrain_utility_information_ladder_oof_v1_49faacc3"
CONFIG = OUTPUT / "resolved_config_preregistered.json"
DENSE_LOGITS = OUTPUT / "fit_dense_logits_float16.npy"
DENSE_META = OUTPUT / "fit_dense_logit_meta_float32.npy"
DENSE_INDEX = OUTPUT / "dense_index.npz"
DENSE_MANIFEST = OUTPUT / "dense_manifest.json"
OOF = OUTPUT / "oof_scores.npz"
METRICS = OUTPUT / "metrics.json"

FOLDS = 5
PCA_DIM, PCA_Q, PCA_NITER = 32, 40, 3
PCA_SEED = 261080
SEEDS = [261070, 261071, 261072]
EPOCHS, BATCH_BLOCKS, NEUTRAL_RATIO = 20, 1024, 4
TOP_K = 769
EXPECTED_GPU = {"pci_bus_id": "00000000:61:00.0",
                "uuid": "GPU-91f9e38c-0a59-15c0-bd08-6a4f89da07ed"}


def imp(name):
    path = ROOT / f"tools/{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec); sys.modules[spec.name] = module; spec.loader.exec_module(module)
    return module


DCV = imp("analyze_phoenix_full_decoder_conditioned_value_oof")
FULL, PARTIAL, BUILDDATA = DCV.FULL, DCV.PARTIAL, DCV.BUILDDATA


def utcnow():
    from datetime import datetime, timezone
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_cuda_device():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable; run GPU stages outside sandbox")
    if torch.cuda.device_count() != 1 or os.environ.get("CUDA_VISIBLE_DEVICES") != EXPECTED_GPU["uuid"]:
        raise RuntimeError({"visible": os.environ.get("CUDA_VISIBLE_DEVICES"), "expected": EXPECTED_GPU})
    props = torch.cuda.get_device_properties(0)
    return {**EXPECTED_GPU, "name": props.name, "total_memory_bytes": props.total_memory}


def frozen_config():
    required = [DATA / "dataset_manifest.json", DENSE / "protocol_manifest.json", KROOT / "oof_scores.npz",
                FINGER / "feature_manifest.json", FACE / "feature_manifest.json",
                DCVROOT / "resolved_config_preregistered.json", DCVROOT / "history_manifest.json", DCVROOT / "oof_scores.npz"]
    if any(not path.is_file() for path in required):
        raise FileNotFoundError("required frozen upstream asset absent")
    return {
        "experiment": "full-fit privileged utility information-sufficiency ladder OOF",
        "created_before_dense_archive_or_ladder_outcomes": True,
        "scope": {"samples": 6378, "sources": 578, "blocks": 183401, "side_rows": 366802},
        "upstream_sha256": {str(path): sha256_file(path) for path in required},
        "status": "nondeployable diagnostic only; no tier may claim compute savings",
        "common_inputs": {
            "B0": "49-D bookkeeping and past-paid prefix summary",
            "visual": "frozen fold-specific finger264 + face132",
            "paid_history": "last 16 paid logits through current offset0, projected by fold PCA and pooled last/mean/std/delta",
            "paid_metadata": "13-D per execution pooled last/mean/std/delta",
            "common_width": 625,
        },
        "tiers": {
            "L0": "reuse completed DCV OOF; deployable reference point",
            "L1": {"extra": "the evaluated side candidate's complete expensive ISLR logit PCA32 + entropy/margin/blank", "width": 660},
            "L2": {"extra": "all left/center/right candidate descriptors plus side-minus-center PCA32", "width": 762},
            "L3": {"extra": "L2 plus last/mean/std/delta PCA summaries for complete fixed-offset2 and fixed-offset3 future continuations", "width": 1018},
        },
        "causality_interpretation": {"L1": "current candidate expensive computation, forbidden to a scheduler deciding whether to compute it",
                                     "L2": "all three current candidates computed", "L3": "complete future observed",
                                     "reference_input": False, "EOS_input": "L3 implicitly uses full future endpoint and is strictly noncausal"},
        "PCA": {"dimensions": PCA_DIM, "q": PCA_Q, "niter": PCA_NITER,
                "fit": "all dense logits from outer-train sources only", "seed": "261080+fold"},
        "future_summary": {"continuations": ["offset2", "offset3"],
                           "definition": "from next block skeleton to physical end, skeleton plus frozen bonus offset",
                           "pooling": "last/mean/std/last-minus-first of fold-PCA logits"},
        "model": "per-candidate MLP Linear(input,128)-SiLU-Linear(128,64)-SiLU-Linear(64,2)",
        "outputs": ["center-future center-minus-side error", "late-future center-minus-side error"],
        "score": "minimum predicted advantage",
        "training": {"loss_and_sampling": "exact DCV continuous loss; all informative blocks plus 4x neutral resample per epoch",
                     "epochs": EPOCHS, "batch_blocks": BATCH_BLOCKS, "optimizer": "AdamW", "lr": 0.001,
                     "weight_decay": 0.0001, "seeds": SEEDS, "scheduler": None, "early_stopping": False},
        "evaluation": {"folds": "exact robust-oof-v1 source folds", "top_K": TOP_K,
                       "primary": "signed top-769 utility for each tier", "strong_information_threshold": 249,
                       "decision": ["lowest L1/L2 tier >=249: candidate information sufficient; proceed to distillation",
                                    "only L3 >=249: utility requires future; reject low-latency predictor",
                                    "no privileged tier >=249: abandon per-window utility prediction"]},
        "gpu": {**EXPECTED_GPU, "forbidden_pci": ["00000000:25:00.0", "00000000:41:00.0"], "gpu0_forbidden": True},
        "forbidden": ["calibration/dev/test", "closed loop", "reference as predictor input", "post-outcome tier/model/loss tuning", "git commit"],
        "output_directory": str(OUTPUT),
    }


def preregister():
    if OUTPUT.exists(): raise FileExistsError(OUTPUT)
    OUTPUT.mkdir(parents=True); value = frozen_config(); CONFIG.write_text(json.dumps(value, indent=2) + "\n"); return value


def load_config():
    value = json.loads(CONFIG.read_text())
    if value != frozen_config(): raise RuntimeError("preregistered configuration changed")
    return value


def logit_meta(values, blank):
    output = np.empty((len(values), 3), np.float32)
    for left in range(0, len(values), 2048):
        x = np.asarray(values[left:left + 2048], np.float32); x -= x.max(1, keepdims=True)
        p = np.exp(x); p /= p.sum(1, keepdims=True)
        top = np.partition(p, -2, axis=1)[:, -2:]
        output[left:left + len(x), 0] = -np.sum(p * np.log(np.maximum(p, 1e-12)), axis=1)
        output[left:left + len(x), 1] = top.max(1) - top.min(1)
        output[left:left + len(x), 2] = p[:, blank]
    return output


def build_dense():
    load_config()
    if any(path.exists() for path in (DENSE_LOGITS, DENSE_META, DENSE_INDEX, DENSE_MANIFEST)):
        raise FileExistsError("dense ladder cache is non-overwriting")
    packed = np.load(DCV.HISTORY); sample_names = [str(x) for x in packed["sample_names"].tolist()]
    sample_indices = np.asarray(packed["sample_indices"], np.int64); folds = np.asarray(packed["folds"], np.int8); packed.close()
    sample_fold = np.empty(len(sample_names), np.int8)
    for index in range(len(sample_names)):
        values = np.unique(folds[sample_indices == index])
        if len(values) != 1: raise RuntimeError("sample fold identity failure")
        sample_fold[index] = values[0]
    wanted = set(sample_names); by_name = {}; started = time.perf_counter()
    indices, shard_count = BUILDDATA.CHRON.completed_shard_indices(DENSE)
    for number, shard in enumerate(indices):
        results, logits, _ = BUILDDATA.BUILDER.BUILDER.validate_dense_shard(DENSE, shard, shard_count, verify_hashes=False)
        for name in results:
            if name in wanted: by_name[name] = np.asarray(logits[name], np.float16)
        print(json.dumps({"dense_shards": number + 1, "fit_samples": len(by_name), "seconds": round(time.perf_counter() - started, 1)}), flush=True)
    if set(by_name) != wanted: raise RuntimeError("fit dense coverage failure")
    offsets = np.zeros(len(sample_names) + 1, np.int64); offsets[1:] = np.cumsum([len(by_name[name]) for name in sample_names])
    width = by_name[sample_names[0]].shape[1]
    dense = np.lib.format.open_memmap(DENSE_LOGITS, mode="w+", dtype=np.float16, shape=(int(offsets[-1]), width))
    frame_folds = np.empty(int(offsets[-1]), np.int8)
    for i, name in enumerate(sample_names):
        dense[offsets[i]:offsets[i + 1]] = by_name[name]; frame_folds[offsets[i]:offsets[i + 1]] = sample_fold[i]
    dense.flush(); del by_name
    vocab = json.loads(BUILDDATA.CHRON.DEFAULT_VOCAB.read_text()); blank = vocab.index("<blank>")
    meta = logit_meta(dense, blank); np.save(DENSE_META, meta)
    np.savez_compressed(DENSE_INDEX, sample_names=np.asarray(sample_names), offsets=offsets, sample_folds=sample_fold, frame_folds=frame_folds)
    manifest = {"status": "complete", "created_utc": utcnow(), "samples": len(sample_names), "frames": len(dense),
                "vocab_width": width, "files": {
                    "logits": {"path": str(DENSE_LOGITS), "bytes": DENSE_LOGITS.stat().st_size, "sha256": sha256_file(DENSE_LOGITS)},
                    "meta": {"path": str(DENSE_META), "bytes": DENSE_META.stat().st_size, "sha256": sha256_file(DENSE_META)},
                    "index": {"path": str(DENSE_INDEX), "bytes": DENSE_INDEX.stat().st_size, "sha256": sha256_file(DENSE_INDEX)}},
                "calibration_dev_test_read": False, "seconds": time.perf_counter() - started}
    DENSE_MANIFEST.write_text(json.dumps(manifest, indent=2) + "\n"); return manifest


def pool4(values):
    return np.concatenate([values[-1], values.mean(0), values.std(0), values[-1] - values[0]])


def common_and_privileged(data, projected_dense, projected_paid, dense_meta, offsets, fold):
    visual = DCV.visual_matrix(data, fold)
    paid = np.concatenate([projected_paid, data["executed_meta"]], axis=1)
    history = paid[np.maximum(data["history_indices"], 0)]; history[data["history_indices"] < 0] = 0
    pooled = np.empty((len(history), 4 * paid.shape[1]), np.float32)
    for i, length in enumerate(data["history_lengths"]): pooled[i] = pool4(history[i, -int(length):])
    common = np.concatenate([data["base"], visual, np.repeat(pooled[:, None, :], 2, axis=1)], axis=2)
    candidate_desc = np.empty((len(common), 3, 35), np.float32)
    futures = np.empty((len(common), 2, 128), np.float32)
    for sample_index in range(len(data["sample_names"])):
        blocks = np.flatnonzero(data["sample_indices"] == sample_index)
        if not len(blocks): continue
        begin, end = int(offsets[sample_index]), int(offsets[sample_index + 1]); n = end - begin
        starts3 = np.stack([data["candidate_starts"][blocks, 0], data["candidate_starts"][blocks, 0] + 1,
                            data["candidate_starts"][blocks, 1]], axis=1).astype(np.int64)
        global3 = begin + starts3
        candidate_desc[blocks] = np.concatenate([projected_dense[global3], dense_meta[global3]], axis=2)
        for local, block in enumerate(blocks):
            next_skeleton = int(data["decision_arrivals"][block]) + 1
            for continuation_index, bonus in enumerate((2, 3)):
                selected = []
                for skeleton in range(next_skeleton, n, 4):
                    selected.append(skeleton)
                    if skeleton + 3 < n: selected.append(skeleton + bonus)
                values = projected_dense[begin + np.asarray(selected, np.int64)] if selected else projected_dense[begin + np.asarray([n - 1])]
                futures[block, continuation_index] = pool4(values)
    side_desc = candidate_desc[:, [0, 2], :]
    all3 = np.repeat(candidate_desc.reshape(len(common), 1, -1), 2, axis=1)
    relative = candidate_desc[:, [0, 2], :32] - candidate_desc[:, 1:2, :32]
    l1 = np.concatenate([common, side_desc], axis=2)
    l2 = np.concatenate([common, all3, relative], axis=2)
    future_both = np.repeat(futures.reshape(len(common), 1, -1), 2, axis=1)
    l3 = np.concatenate([l2, future_both], axis=2)
    return {"L1": l1, "L2": l2, "L3": l3}


class TierMLP(nn.Module):
    def __init__(self, width):
        super().__init__(); self.net = nn.Sequential(nn.Linear(width, 128), nn.SiLU(), nn.Linear(128, 64), nn.SiLU(), nn.Linear(64, 2))
    def forward(self, value):
        b, c, w = value.shape; return self.net(value.reshape(b * c, w)).reshape(b, c, 2)


def normalize(matrix, train):
    mean, scale = DCV.mean_scale(matrix[train].reshape(-1, matrix.shape[-1])); return np.asarray((matrix - mean) / scale, np.float32)


def train_tier(matrix, data, fold, scores, predictions):
    train = data["folds"] != fold; evaluate = ~train
    x = torch.from_numpy(normalize(matrix, train)).cuda(); target = torch.from_numpy(data["targets"].astype(np.float32)).cuda()
    informative = np.flatnonzero(train & np.any(data["targets"] != 0, axis=(1, 2)))
    neutral = np.flatnonzero(train & ~np.any(data["targets"] != 0, axis=(1, 2))); eval_index = np.flatnonzero(evaluate)
    losses = []
    for seed_index, seed in enumerate(SEEDS):
        random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
        model = TierMLP(matrix.shape[-1]).cuda(); optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.0001)
        generator = torch.Generator().manual_seed(seed + fold); history = []
        for _ in range(EPOCHS):
            count = min(len(neutral), NEUTRAL_RATIO * len(informative))
            chosen_neutral = neutral[torch.randperm(len(neutral), generator=generator)[:count].numpy()]
            chosen = np.concatenate([informative, chosen_neutral]); chosen = chosen[torch.randperm(len(chosen), generator=generator).numpy()]
            total = 0.0
            for left in range(0, len(chosen), BATCH_BLOCKS):
                idx = torch.from_numpy(chosen[left:left + BATCH_BLOCKS]).cuda()
                pred = model(x[idx]); loss = DCV.dcv_loss(pred, target[idx])
                optimizer.zero_grad(set_to_none=True); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); optimizer.step()
                total += float(loss.detach()) * len(idx)
            history.append(total / len(chosen))
        model.eval(); out = []
        with torch.no_grad():
            for left in range(0, len(eval_index), 4096):
                idx = torch.from_numpy(eval_index[left:left + 4096]).cuda(); out.append(model(x[idx]).cpu().numpy())
        out = np.concatenate(out); predictions[seed_index, eval_index] = out; scores[seed_index, eval_index] = out.min(2); losses.append(history)
        del model, optimizer
    del x, target; torch.cuda.empty_cache(); return losses


def run_oof():
    load_config(); gpu = verify_cuda_device()
    if not DENSE_MANIFEST.is_file() or OOF.exists() or METRICS.exists(): raise RuntimeError("complete dense cache required; OOF non-overwriting")
    packed = np.load(DCV.HISTORY); data = {key: packed[key] for key in packed.files}; packed.close()
    dense_index = np.load(DENSE_INDEX); offsets = np.asarray(dense_index["offsets"], np.int64); frame_folds = np.asarray(dense_index["frame_folds"], np.int8)
    dense = np.load(DENSE_LOGITS, mmap_mode="r"); dense_meta = np.load(DENSE_META, mmap_mode="r"); paid = np.load(DCV.EXECUTED, mmap_mode="r")
    scores = {name: np.full((len(SEEDS), len(data["folds"]), 2), np.nan, np.float32) for name in ("L1", "L2", "L3")}
    predictions = {name: np.full((len(SEEDS), len(data["folds"]), 2, 2), np.nan, np.float32) for name in scores}
    fold_records = []; started = time.perf_counter()
    for fold in range(FOLDS):
        train_frame = frame_folds != fold; torch.manual_seed(PCA_SEED + fold)
        xtrain = torch.from_numpy(np.asarray(dense[train_frame], np.float32)).cuda(); mean = xtrain.mean(0); centered = xtrain - mean
        _, singular, components = torch.pca_lowrank(centered, q=PCA_Q, center=False, niter=PCA_NITER); components = components[:, :PCA_DIM]
        scale = (singular[:PCA_DIM] / math.sqrt(max(1, len(xtrain) - 1))).clamp_min(1e-6)
        explained = float(singular[:PCA_DIM].square().sum() / centered.square().sum())
        pca_path = OUTPUT / f"dense_pca_fold{fold}.npz"; np.savez_compressed(pca_path, mean=mean.cpu().numpy(), components=components.cpu().numpy(), scale=scale.cpu().numpy())
        def project(array):
            result = np.empty((len(array), PCA_DIM), np.float32)
            with torch.no_grad():
                for left in range(0, len(array), 8192):
                    value = torch.from_numpy(np.asarray(array[left:left + 8192], np.float32)).cuda()
                    result[left:left + len(value)] = (((value - mean) @ components) / scale).cpu().numpy()
            return result
        projected_dense = project(dense); projected_paid = project(paid); del xtrain, centered
        matrices = common_and_privileged(data, projected_dense, projected_paid, dense_meta, offsets, fold)
        tier_record = {}
        for name in ("L1", "L2", "L3"):
            tier_record[name] = {"width": matrices[name].shape[-1], "parameters": sum(p.numel() for p in TierMLP(matrices[name].shape[-1]).parameters()),
                                 "seed_losses": train_tier(matrices[name], data, fold, scores[name], predictions[name])}
            del matrices[name]
        fold_records.append({"fold": fold, "PCA_train_frames": int(train_frame.sum()), "PCA_explained_fraction": explained,
                             "PCA_sha256": sha256_file(pca_path), "tiers": tier_record})
        print(json.dumps({"ladder_fold": fold, "seconds": round(time.perf_counter() - started, 1)}), flush=True)
        del projected_dense, projected_paid; torch.cuda.empty_cache()
    rewards = data["rewards"].reshape(-1).astype(np.int64); positive = (rewards > 0).astype(np.float32); source_rows = np.repeat(data["source_indices"], 2)
    old = np.load(KROOT / "oof_scores.npz"); dcv_old = np.load(DCVROOT / "oof_scores.npz")
    final_scores = {name: value.mean(0).reshape(-1) for name, value in scores.items()}
    metric = {"K": PARTIAL.metrics(positive, rewards, old["signed_K"]), "L0_DCV": PARTIAL.metrics(positive, rewards, dcv_old["dcv_score"])}
    ci = {}; continuous = {}; actual = data["targets"].reshape(-1, 2).astype(np.float32)
    for name in ("L1", "L2", "L3"):
        metric[name] = PARTIAL.metrics(positive, rewards, final_scores[name]); ci[name] = FULL.source_bootstrap_delta(rewards, source_rows, final_scores[name], old["signed_K"], TOP_K)
        pred = predictions[name].mean(0).reshape(-1, 2)
        continuous[name] = {"center_spearman": float(spearmanr(pred[:, 0], actual[:, 0]).statistic),
                            "late_spearman": float(spearmanr(pred[:, 1], actual[:, 1]).statistic)}
    passing = [name for name in ("L1", "L2", "L3") if metric[name]["top_K_signed_robust_utility"] >= 249]
    if any(name in passing for name in ("L1", "L2")): decision = "candidate_information_sufficient_proceed_to_separate_distillation"
    elif "L3" in passing: decision = "future_required_reject_low_latency_predictor"
    else: decision = "abandon_per_window_utility_prediction"
    value = {"created_utc": utcnow(), "device": gpu, "metrics": metric, "tier_minus_K_source_bootstrap_95pct": ci,
             "continuous_target_spearman": continuous, "folds": fold_records,
             "gate": {"threshold_errors": 249, "passing_tiers": passing, "decision": decision, "closed_loop": False},
             "elapsed_seconds": time.perf_counter() - started}
    np.savez_compressed(OOF, rewards=rewards, folds=np.repeat(data["folds"], 2), source_indices=source_rows,
                        signed_K=old["signed_K"], L0_DCV=dcv_old["dcv_score"],
                        L1=final_scores["L1"], L2=final_scores["L2"], L3=final_scores["L3"], continuous_targets=actual)
    value["oof_scores"] = {"path": str(OOF), "bytes": OOF.stat().st_size, "sha256": sha256_file(OOF)}
    METRICS.write_text(json.dumps(value, indent=2) + "\n"); return value


def main():
    parser = argparse.ArgumentParser(); parser.add_argument("command", choices=("preregister", "build-dense", "run-oof")); args = parser.parse_args()
    functions = {"preregister": preregister, "build-dense": build_dense, "run-oof": run_oof}; print(json.dumps(functions[args.command](), indent=2))


if __name__ == "__main__": main()
