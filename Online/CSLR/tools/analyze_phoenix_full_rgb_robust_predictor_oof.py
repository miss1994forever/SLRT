#!/usr/bin/env python3
"""Frozen fit-only RGB-vs-skeleton robust-utility predictor experiment.

The RGB descriptor is deliberately small and deployment-oriented: current-frame
pose-guided hand crops followed by a fixed (untrained) low-frequency YCrCb DCT.
No decoder outcome, reference, future frame, true EOS, or expensive ISLR feature
is used as a predictor input.
"""
import argparse
import gzip
import hashlib
import importlib.util
import io
import json
import math
import os
import pickle
import resource
import sys
import time
import zipfile
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parents[1]
BASE = ROOT / "results/phoenix-2014t_ISLR"
DATA = BASE / "p3_fulltrain_robust_continuation_fit6378_dataset_v2_49faacc3"
POSE = (BASE / "p3_fulltrain_causal_pose_hand_motion_v1_49faacc3"
        / "train_causal_pose_hand_motion_sequences.npz")
SKELETON = BASE / "p3_fulltrain_robust_predictor_oof_v1_49faacc3"
OUTPUT = BASE / "p3_fulltrain_rgb_dct_robust_predictor_oof_v1_49faacc3"
ZIP_PATH = REPO / "data/phoenix_2014t/PHOENIX2014T_videos.zip"
KEYPOINTS = REPO / "data/phoenix_2014t/keypoints_hrnet_dark_coco_wholebody_iso.train.pkl"
SPLIT = BASE / "train_dense_stride1_v1_49faacc3/fit_calibration_split.json"
CHECKPOINT = BASE / "ckpts/best.ckpt"
FEATURES = OUTPUT / "train_fit_rgb_hand_dct_sequences.npz"
FEATURE_MANIFEST = OUTPUT / "rgb_feature_manifest.json"
AUDIT = OUTPUT / "rgb_visibility_cost_audit.json"
CONFIG = OUTPUT / "resolved_config_preregistered.json"
OOF = OUTPUT / "oof_scores.npz"
METRICS = OUTPUT / "metrics.json"

FOLDS = 5
HISTORY = 31
VISIBLE_OFFSET = 8
BINARY_SEEDS = [261026, 261027, 261028]
SIGNED_SEEDS = [261029, 261030, 261031]
LUMA_COORDS = [(0, 0), (0, 1), (1, 0), (2, 0), (1, 1), (0, 2),
               (0, 3), (1, 2), (2, 1), (3, 0), (4, 0), (3, 1),
               (2, 2), (1, 3), (0, 4), (0, 5)]
CHROMA_COORDS = [(0, 0), (0, 1), (1, 0), (1, 1)]
PER_HAND_WIDTH = len(LUMA_COORDS) + 2 * len(CHROMA_COORDS) + 1
FRAME_WIDTH = 2 * PER_HAND_WIDTH
EXPECTED = {
    "video_sha256": "49faacc304666a75cb51e3e2d335dfbead8d08e8dd5ff834c66c690e1175d457",
    "keypoints_sha256": "18a045bfe7790064e06a9016d0949d7f2f8243ae62ddad4f08d3fb5ea3d5b763",
    "checkpoint_sha256": "b3390f0dc4b6a826b53c88d3309b1d98fb75e5b58a5ec3f5779628cf3d51767b",
    "pose_sha256": "ab659a6e5f1f9f545fe705318ff9be5d7bc2d843acad8930d446b4f83e4dbcb3",
    "skeleton_oof_sha256": "9665b83e11a8c6c415142820d5428ec37f8ac234b3b02b68a23d91d8a94eb29a",
}


def imp(name):
    path = ROOT / f"tools/{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


FULL = imp("analyze_phoenix_full_robust_predictor_oof")
PARTIAL = FULL.PARTIAL
HAND = imp("audit_phoenix_new_observable_preview")


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


def fit_ids_in_pose_order():
    split = json.loads(SPLIT.read_text())
    pose = np.load(POSE)
    names = [str(x) for x in pose["video_names"].tolist()]
    ids = [name for name in names if split["assignments"].get(name) == "fit"]
    if len(ids) != 6378 or len(set(ids)) != len(ids):
        raise RuntimeError("frozen fit sample identity mismatch")
    digest = hashlib.sha256("\n".join(sorted(ids)).encode()).hexdigest()
    if digest != "ea9f022ef1160bc3be5786fc31ec745b7007222c400864bbe16cb5101a6ec28e":
        # The robust dataset hash is based on its canonical order, so also
        # accept the pose order only after checking the exact set below.
        dataset_ids = set()
        for path in FULL.label_paths():
            with gzip.open(path, "rt", encoding="utf-8") as handle:
                for line in handle:
                    row = json.loads(line)
                    if row.get("type") == "block":
                        dataset_ids.add(row["sample_id"])
        if set(ids) != dataset_ids:
            raise RuntimeError("fit split and robust-label sample sets differ")
    return ids


def encoder_identity():
    eye = np.eye(32, dtype=np.float32)
    basis = np.stack([cv2.dct(row.reshape(32, 1)).ravel() for row in eye])
    payload = {
        "name": "fixed-ycrcb-dct-v1",
        "crop_size": [32, 32],
        "luma_coords": [list(value) for value in LUMA_COORDS],
        "chroma_coords": [list(value) for value in CHROMA_COORDS],
        "normalization": "RGB uint8 -> YCrCb float32 / 255; orthonormal OpenCV DCT",
        "basis_sha256": hashlib.sha256(basis.tobytes()).hexdigest(),
        "learned_weights": False,
    }
    payload["encoder_identity_sha256"] = canonical_sha(payload)
    return payload


def member_name(name, frame):
    return f"images/{name.split('/', 1)[1]}/images{frame + 1:04d}.png"


def hand_descriptor(crop, valid):
    if not valid:
        return np.zeros(PER_HAND_WIDTH, dtype=np.float32)
    color = cv2.cvtColor(crop, cv2.COLOR_RGB2YCrCb).astype(np.float32) / 255.0
    planes = [cv2.dct(color[:, :, channel]) for channel in range(3)]
    values = [planes[0][i, j] for i, j in LUMA_COORDS]
    values.extend(planes[1][i, j] for i, j in CHROMA_COORDS)
    values.extend(planes[2][i, j] for i, j in CHROMA_COORDS)
    values.append(1.0)
    return np.asarray(values, dtype=np.float32)


def frame_descriptor(frame, points):
    values = []
    valid = []
    for side in ("left", "right"):
        crop, ok = HAND.hand_crop(frame, points, side, size=32)
        values.append(hand_descriptor(crop, ok))
        valid.append(ok)
    return np.concatenate(values), valid


def run_audit():
    if OUTPUT.exists():
        raise FileExistsError(f"audit requires a fresh output directory: {OUTPUT}")
    OUTPUT.mkdir(parents=True)
    actual = {
        "video_sha256": sha256_file(ZIP_PATH),
        "keypoints_sha256": sha256_file(KEYPOINTS),
        "checkpoint_sha256": sha256_file(CHECKPOINT),
        "pose_sha256": sha256_file(POSE),
        "skeleton_oof_sha256": sha256_file(SKELETON / "oof_scores.npz"),
    }
    if actual != EXPECTED:
        raise RuntimeError({"identity_mismatch": actual, "expected": EXPECTED})
    ids = fit_ids_in_pose_order()
    chosen = sorted(ids, key=lambda x: hashlib.sha256(f"rgb-dct-cost-v1\0{x}".encode()).digest())[:32]
    with KEYPOINTS.open("rb") as handle:
        keypoints = pickle.load(handle)
    decode_ms, crop_encode_ms, valid = [], [], []
    decoded_bytes = 0
    zf = zipfile.ZipFile(ZIP_PATH)
    for name in chosen:
        points = keypoints[name]
        start = max(0, (len(points) - 16) // 2)
        tick = time.perf_counter()
        frames = []
        for frame_index in range(start, min(start + 16, len(points))):
            with zf.open(member_name(name, frame_index)) as handle:
                raw = handle.read()
            decoded_bytes += len(raw)
            frames.append(np.asarray(Image.open(io.BytesIO(raw)).convert("RGB")))
        decode_ms.append(1000 * (time.perf_counter() - tick))
        tick = time.perf_counter()
        for offset, frame in enumerate(frames):
            _, pair_valid = frame_descriptor(frame, points[start + offset])
            valid.extend(pair_valid)
        crop_encode_ms.append(1000 * (time.perf_counter() - tick))
    zf.close()
    frames = sum(min(16, len(keypoints[name])) for name in chosen)
    audit = {
        "status": "complete",
        "created_utc": utcnow(),
        "scope": "read-only train-fit RGB visibility, identity, causality, and CPU cost audit",
        "outcomes_read": False,
        "calibration_dev_test_read": False,
        "identities": actual,
        "fit_samples": len(ids),
        "cost_sample": {
            "selection": "lowest SHA256(rgb-dct-cost-v1+NUL+fit sample id)",
            "samples": len(chosen),
            "frames": frames,
            "decoded_png_bytes": decoded_bytes,
            "decode_ms_per_frame": float(sum(decode_ms) / frames),
            "two_hand_crop_plus_dct_ms_per_frame": float(sum(crop_encode_ms) / frames),
            "valid_hand_fraction": float(np.mean(valid)),
            "peak_process_rss_mib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0,
        },
        "deployment": {
            "execution_frequency": "every arrived RGB frame, including frames in windows later skipped by the scheduler",
            "candidate_deadline": "history ends at min(last real frame, candidate start + 8), identical to K",
            "causal_inputs": "same-frame RGB plus same-frame HRNet hand points; no temporal fill",
            "cpu_gpu_transfer": "none for descriptor; CPU float vector may be appended to the scheduler input",
            "descriptor_gpu_memory_bytes": 0,
            "approximate_dct_arithmetic_per_frame": "393216 MACs for six separable 32x32 2-D DCTs, excluding decode/crop resize",
            "pose_detector_cost": "not included; K already requires the same online pose stream",
            "risk": "continuous PNG/RGB decode and pose inference remain mandatory and can offset saved ISLR windows",
        },
        "forbidden_rejected": {
            "dense_islr_logits": True,
            "candidate_islr_backbone_embeddings": True,
            "future_frames_reference_true_eos_total_length": True,
            "reason": "these are unavailable before the scheduler decision or are outcome leakage",
        },
        "encoder": encoder_identity(),
    }
    AUDIT.write_text(json.dumps(audit, indent=2) + "\n")
    return audit


def frozen_config():
    if not AUDIT.is_file():
        raise RuntimeError("run the visibility/cost audit before preregistration")
    audit = json.loads(AUDIT.read_text())
    manifest = json.loads((DATA / "dataset_manifest.json").read_text())
    return {
        "experiment": "complete-fit RGB-DCT versus frozen skeleton robust predictor OOF",
        "created_before_rgb_feature_generation_or_oof_outcomes": True,
        "scope": manifest["scope"],
        "rows": 2 * int(manifest["counts"]["blocks"]),
        "class_counts": {"beneficial": 769, "harmful": 2015, "neutral": 364018},
        "inputs": {
            "video": {"path": str(ZIP_PATH), "sha256": audit["identities"]["video_sha256"]},
            "keypoints": {"path": str(KEYPOINTS), "sha256": audit["identities"]["keypoints_sha256"]},
            "checkpoint_identity_only_not_predictor_input": audit["identities"]["checkpoint_sha256"],
            "robust_dataset_manifest_sha256": sha256_file(DATA / "dataset_manifest.json"),
            "pose_archive_sha256": audit["identities"]["pose_sha256"],
            "frozen_skeleton_oof_sha256": audit["identities"]["skeleton_oof_sha256"],
            "visibility_cost_audit_sha256": sha256_file(AUDIT),
        },
        "rgb_encoder": {
            **encoder_identity(),
            "crop": "each hand from same-frame HRNet points confidence>=0.2; square 1.5xbbox, minimum 24 px; resize 32x32",
            "invalid_crop": "all 24 DCT coefficients and validity flag are zero; no future/past fill",
            "output": "16 Y plus 4 Cr plus 4 Cb DCT coefficients and one validity flag per hand; 50/frame; float16 archive",
        },
        "temporal_visibility": {
            "decision": "after offset3 candidate arrives; same bounded lookahead as K",
            "last_visible_frame": "min(last real frame, candidate start + 8)",
            "history": 31,
            "left_padding": "repeat first visible real frame",
            "pooling": ["last", "mean", "std", "last-minus-first"],
            "R_width": 49 + 4 * FRAME_WIDTH,
        },
        "feature_sets": {
            "B0": "reuse frozen saved OOF score; 49 bookkeeping+decoder-prefix inputs",
            "K": "reuse frozen saved OOF score; B0 plus 68 skeleton-history summary inputs",
            "R": "B0 plus 200 RGB-DCT-history summary inputs; primary candidate",
            "KR": "B0 plus K and R histories; preregistered secondary complement diagnostic; not gate-eligible",
        },
        "folds": "exact frozen robust-oof-v1 5-fold deterministic source mapping",
        "normalization": "mean/std independently fit on each OOF training fold only; no PCA/codebook",
        "objectives": {
            "signed_primary": "harmful/neutral/beneficial inverse-frequency weighted cross entropy",
            "binary_diagnostic": "beneficial versus rest weighted BCE",
            "network": "MLP 32-16; 20 fixed epochs; AdamW lr 0.002 weight_decay 0.0001; batch 8192",
            "binary_seeds": BINARY_SEEDS,
            "signed_seeds": SIGNED_SEEDS,
        },
        "K": 769,
        "primary_comparison": "R minus K signed top-K utility with source-cluster bootstrap 95% CI",
        "candidate_priority": ["R"],
        "KR_status": "secondary diagnostic fixed now; cannot replace R as the confirmatory comparison or unlock closed loop",
        "closed_loop_gate": "R signed top-769 robust utility >=249 errors",
        "forbidden": ["calibration", "dev", "test", "closed-loop outcome before gate", "threshold tuning",
                      "feature/crop/history/architecture tuning", "expensive ISLR logits or embeddings as R", "git commit"],
        "output_directory": str(OUTPUT),
    }


def preregister():
    if not OUTPUT.is_dir() or not AUDIT.is_file():
        raise RuntimeError("fresh audit directory is missing")
    if CONFIG.exists() or FEATURES.exists() or FEATURE_MANIFEST.exists() or OOF.exists() or METRICS.exists():
        raise FileExistsError("preregistration must precede all feature/outcome artifacts")
    value = frozen_config()
    CONFIG.write_text(json.dumps(value, indent=2) + "\n")
    return value


def load_config():
    value = json.loads(CONFIG.read_text())
    if value != frozen_config():
        raise RuntimeError("preregistered configuration has changed")
    return value


def build_features():
    cfg = load_config()
    if FEATURES.exists() or FEATURE_MANIFEST.exists() or OOF.exists() or METRICS.exists():
        raise FileExistsError("RGB feature archive is non-overwriting")
    ids = fit_ids_in_pose_order()
    with KEYPOINTS.open("rb") as handle:
        keypoints = pickle.load(handle)
    if any(name not in keypoints for name in ids):
        raise RuntimeError("fit RGB/keypoint sample coverage mismatch")
    offsets = np.zeros(len(ids) + 1, dtype=np.int64)
    offsets[1:] = np.cumsum([len(keypoints[name]) for name in ids])
    features = np.empty((int(offsets[-1]), FRAME_WIDTH), dtype=np.float16)
    invalid_hands = invalid_frames = decoded_bytes = 0
    encode_seconds = decode_seconds = 0.0
    zf = zipfile.ZipFile(ZIP_PATH)
    started = time.perf_counter()
    for number, name in enumerate(ids, 1):
        points = keypoints[name]
        base = int(offsets[number - 1])
        for frame_index, current_points in enumerate(points):
            tick = time.perf_counter()
            with zf.open(member_name(name, frame_index)) as handle:
                raw = handle.read()
            frame = np.asarray(Image.open(io.BytesIO(raw)).convert("RGB"))
            decode_seconds += time.perf_counter() - tick
            decoded_bytes += len(raw)
            tick = time.perf_counter()
            descriptor, pair_valid = frame_descriptor(frame, current_points)
            encode_seconds += time.perf_counter() - tick
            failures = 2 - int(sum(pair_valid))
            invalid_hands += failures
            invalid_frames += failures > 0
            features[base + frame_index] = descriptor.astype(np.float16)
        if number % 100 == 0 or number == len(ids):
            print(json.dumps({"feature_samples": number, "frames": int(offsets[number]),
                              "elapsed_seconds": round(time.perf_counter() - started, 1)}), flush=True)
    zf.close()
    temp = OUTPUT / ".train_fit_rgb_hand_dct_sequences.incomplete.npz"
    np.savez_compressed(temp, features=features, video_names=np.asarray(ids), video_offsets=offsets,
                        feature_names=np.asarray([f"rgb_dct_{i}" for i in range(FRAME_WIDTH)]))
    temp.replace(FEATURES)
    manifest = {
        "status": "complete", "created_utc": utcnow(), "samples": len(ids),
        "sources": len({name.rsplit("-", 1)[0] for name in ids}), "frames": len(features),
        "feature_shape": list(features.shape), "dtype": str(features.dtype),
        "sample_ids_sha256": hashlib.sha256("\n".join(ids).encode()).hexdigest(),
        "input_sha256": {"video": cfg["inputs"]["video"]["sha256"],
                           "keypoints": cfg["inputs"]["keypoints"]["sha256"]},
        "output": {"path": str(FEATURES), "bytes": FEATURES.stat().st_size,
                    "sha256": sha256_file(FEATURES)},
        "invalid_hand_crops": invalid_hands, "total_hand_crops": 2 * len(features),
        "invalid_hand_crop_rate": invalid_hands / (2 * len(features)),
        "frames_with_any_invalid_hand": invalid_frames,
        "invalid_frame_rate": invalid_frames / len(features),
        "decoded_png_bytes": decoded_bytes, "decode_seconds": decode_seconds,
        "crop_and_dct_seconds": encode_seconds, "total_seconds": time.perf_counter() - started,
        "causality": "each row uses only same-frame RGB and same-frame HRNet points; invalid crops zeroed without temporal fill",
        "encoder_identity_sha256": cfg["rgb_encoder"]["encoder_identity_sha256"],
    }
    FEATURE_MANIFEST.write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


class RGBArchive:
    def __init__(self):
        manifest = json.loads(FEATURE_MANIFEST.read_text())
        if sha256_file(FEATURES) != manifest["output"]["sha256"]:
            raise RuntimeError("RGB archive hash mismatch")
        data = np.load(FEATURES)
        self.features = data["features"]
        names = [str(x) for x in data["video_names"].tolist()]
        offsets = np.asarray(data["video_offsets"], dtype=np.int64)
        self.slices = {name: slice(int(offsets[i]), int(offsets[i + 1])) for i, name in enumerate(names)}

    def history(self, name, start):
        frames = np.asarray(self.features[self.slices[name]], dtype=np.float16)
        visible = min(len(frames) - 1, int(start) + VISIBLE_OFFSET)
        left = max(0, visible - HISTORY + 1)
        values = frames[left:visible + 1]
        if len(values) < HISTORY:
            values = np.concatenate([np.repeat(values[:1], HISTORY - len(values), axis=0), values])
        return values


def load_rows():
    rgb = RGBArchive()
    pose = PARTIAL.OLD.SequenceFeatures(POSE)
    count = 366802
    base = np.empty((count, 49), dtype=np.float32)
    skeleton = np.empty((count, 4 * pose.width), dtype=np.float32)
    appearance = np.empty((count, 4 * FRAME_WIDTH), dtype=np.float32)
    rewards = np.empty(count, dtype=np.int64)
    folds = np.empty(count, dtype=np.int8)
    source_indices = np.empty(count, dtype=np.int16)
    source_to_index = {}
    cursor = 0
    classes = Counter()
    for path in FULL.label_paths():
        with gzip.open(path, "rt", encoding="utf-8") as handle:
            for line in handle:
                row = json.loads(line)
                if row.get("type") != "block":
                    continue
                k = PARTIAL.visual_summary(pose.history(row["sample_id"], row["decision_arrival"]))
                r = PARTIAL.visual_summary(rgb.history(row["sample_id"], row["decision_arrival"]))
                source = row["source_video_id"]
                source_index = source_to_index.setdefault(source, len(source_to_index))
                fold = PARTIAL.source_fold(source)
                for candidate in (0, 2):
                    inputs = row["predictor_inputs"][candidate]
                    b = np.concatenate([np.asarray(inputs["bookkeeping"], np.float32),
                                        np.asarray(inputs["prefix"], np.float32)])
                    reward = int(row["label"]["robust_reward_by_candidate"][candidate])
                    base[cursor], skeleton[cursor], appearance[cursor] = b, k, r
                    rewards[cursor], folds[cursor], source_indices[cursor] = reward, fold, source_index
                    classes["beneficial" if reward > 0 else "harmful" if reward < 0 else "neutral"] += 1
                    cursor += 1
    if cursor != count or classes != Counter({"neutral": 364018, "harmful": 2015, "beneficial": 769}):
        raise RuntimeError({"rows": cursor, "classes": classes})
    old = np.load(SKELETON / "oof_scores.npz")
    for key, current in (("folds", folds), ("rewards", rewards), ("source_indices", source_indices)):
        if not np.array_equal(old[key], current):
            raise RuntimeError(f"frozen OOF row mapping mismatch: {key}")
    return {"base": base, "skeleton": skeleton, "rgb": appearance, "rewards": rewards,
            "folds": folds, "source_indices": source_indices, "old": old}


def train_feature_set(matrix, rewards, folds, device):
    positive = (rewards > 0).astype(np.float32)
    signed = np.where(rewards < 0, 0, np.where(rewards > 0, 2, 1)).astype(np.int64)
    scores = {name: np.full(len(rewards), np.nan, np.float32) for name in ("binary", "signed")}
    summaries = []
    for fold in range(FOLDS):
        evaluate = folds == fold
        train = ~evaluate
        train_x, eval_x = FULL.normalized_tensors(matrix, train, evaluate, device)
        binary_y = torch.from_numpy(positive[train]).to(device)
        signed_y = torch.from_numpy(signed[train]).to(device)
        scores["binary"][evaluate] = np.mean([
            FULL.train_binary(train_x, binary_y, eval_x, seed, device) for seed in BINARY_SEEDS
        ], axis=0)
        scores["signed"][evaluate] = np.mean([
            FULL.train_signed(train_x, signed_y, eval_x, seed, device) for seed in SIGNED_SEEDS
        ], axis=0)
        summaries.append({"fold": fold, "train_rows": int(train.sum()), "eval_rows": int(evaluate.sum()),
                          "train_beneficial": int(positive[train].sum()),
                          "eval_beneficial": int(positive[evaluate].sum())})
        del train_x, eval_x, binary_y, signed_y
    if any(np.isnan(x).any() for x in scores.values()):
        raise RuntimeError("OOF score coverage failure")
    return scores, summaries


def run_oof(device_name):
    cfg = load_config()
    if not FEATURE_MANIFEST.is_file() or OOF.exists() or METRICS.exists():
        raise RuntimeError("complete features required and OOF is non-overwriting")
    device = torch.device(device_name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("requested CUDA is unavailable")
    torch.set_num_threads(min(8, os.cpu_count() or 1))
    torch.use_deterministic_algorithms(True)
    values = load_rows()
    rewards, folds = values["rewards"], values["folds"]
    positive = (rewards > 0).astype(np.float32)
    matrices = {
        "R": np.concatenate([values["base"], values["rgb"]], axis=1),
        "KR": np.concatenate([values["base"], values["skeleton"], values["rgb"]], axis=1),
    }
    scores = {
        "binary": {"B0": values["old"]["binary_B0"], "K": values["old"]["binary_K"]},
        "signed": {"B0": values["old"]["signed_B0"], "K": values["old"]["signed_K"]},
    }
    fold_summaries = {}
    started = time.perf_counter()
    for name in ("R", "KR"):
        trained, fold_summaries[name] = train_feature_set(matrices[name], rewards, folds, device)
        for objective in ("binary", "signed"):
            scores[objective][name] = trained[objective]
        print(json.dumps({"trained": name, "elapsed_seconds": round(time.perf_counter() - started, 1)}), flush=True)
    metric = {objective: {name: PARTIAL.metrics(positive, rewards, array)
                          for name, array in variants.items()} for objective, variants in scores.items()}
    k = int(cfg["K"])
    ci_rk = FULL.source_bootstrap_delta(rewards, values["source_indices"],
                                        scores["signed"]["R"], scores["signed"]["K"], k)
    ci_krk = FULL.source_bootstrap_delta(rewards, values["source_indices"],
                                         scores["signed"]["KR"], scores["signed"]["K"], k)
    utility = metric["signed"]["R"]["top_K_signed_robust_utility"]
    r_minus_k = utility - metric["signed"]["K"]["top_K_signed_robust_utility"]
    gate = utility >= 249
    decision = "strong-go" if gate else ("ranking-only" if r_minus_k > 0 and ci_rk[0] > 0 else "no-go")
    value = {
        "created_utc": utcnow(),
        "device": {"requested": device_name,
                   "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
                   "torch_cuda_name": torch.cuda.get_device_name(device) if device.type == "cuda" else None},
        "rows": len(rewards), "sources": 578,
        "class_counts": dict(sorted(Counter(map(int, rewards)).items())),
        "feature_widths": {"B0": 49, "K": 117, "R": matrices["R"].shape[1], "KR": matrices["KR"].shape[1]},
        "folds": fold_summaries, "metrics": metric,
        "signed_R_minus_K_top_K_utility": r_minus_k,
        "signed_R_minus_K_top_K_utility_source_bootstrap_95pct": ci_rk,
        "signed_KR_minus_K_top_K_utility_source_bootstrap_95pct_secondary": ci_krk,
        "gate": {"threshold_errors": 249, "R_utility": utility, "passed": gate,
                 "decision": decision,
                 "action": "run_frozen_unknown_EOS_closed_loop" if gate else "do_not_run_closed_loop"},
        "limitations": ["fit-only OOF ranking", "offset3 bounded candidate lookahead",
                        "pose detector cost excluded because shared with K", "no held-out or final WER claim"],
        "elapsed_seconds": time.perf_counter() - started,
    }
    METRICS.write_text(json.dumps(value, indent=2) + "\n")
    np.savez_compressed(OOF, folds=folds, rewards=rewards, source_indices=values["source_indices"],
                        binary_B0=scores["binary"]["B0"], binary_K=scores["binary"]["K"],
                        binary_R=scores["binary"]["R"], binary_KR=scores["binary"]["KR"],
                        signed_B0=scores["signed"]["B0"], signed_K=scores["signed"]["K"],
                        signed_R=scores["signed"]["R"], signed_KR=scores["signed"]["KR"])
    return value


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("audit", "preregister", "build-features", "run-oof"))
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    if args.command == "audit":
        value = run_audit()
    elif args.command == "preregister":
        value = preregister()
    elif args.command == "build-features":
        value = build_features()
    else:
        value = run_oof(args.device)
    print(json.dumps(value, indent=2))


if __name__ == "__main__":
    main()
