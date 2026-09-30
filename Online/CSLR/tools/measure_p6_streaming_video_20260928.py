#!/usr/bin/env python3
"""Bounded train-fit video replay with actual on-demand ISLR window inference."""
import argparse
import gzip
import hashlib
import json
import logging
import pickle
import statistics
import sys
import time
import zipfile
from collections import deque
from io import BytesIO
from pathlib import Path

import numpy as np
import torch
import yaml
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from modelling.model import build_model
import utils.misc as misc
from utils.adaptive_stride import span_weighted_predictions
from utils.online_quantile_controller import FrozenRidgeBundle, OnlineQuantileController, ridge_score_callback


ROOT = Path(__file__).resolve().parents[1]
SLRT = ROOT.parents[1]
BASE = ROOT / "results/phoenix-2014t_ISLR"
CONFIG = ROOT / "configs/slide_phoenix-2014t.yaml"
MODEL = ROOT / "configs/frozen_models/p6_online_quantile_fold_models_v1.json"
RULE = ROOT / "configs/p6_k_history_online_quantile_oof_v1_preregistered.json"
SPLIT = BASE / "train_dense_stride1_v1_49faacc3/fit_calibration_split.json"
META = SLRT / "data/phoenix_2014t/phoenix14t.train"
KPS = SLRT / "data/phoenix_2014t/keypoints_hrnet_dark_coco_wholebody_iso.train.pkl"
VIDEO = SLRT / "data/phoenix_2014t/PHOENIX2014T_videos.zip"
CKPT = BASE / "ckpts/best.ckpt"
VOCAB = SLRT / "data/phoenix_2014t/phoenix_iso_with_blank.vocab"
FORBIDDEN = {"GPU-dbd35875-dfa5-43f1-0cf0-f88ccb529c8a", "GPU-06afe121-c4ce-b981-bb86-399e4a85ae83", "GPU-2c6a50cb-f770-c785-dc1d-7f8cc7b7b9aa"}


def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(b)
    return h.hexdigest()


def source_fold(name):
    source = name.rpartition("-")[0]
    return int.from_bytes(hashlib.sha256(f"robust-oof-v1\0{source}".encode()).digest()[:8], "big") % 5


def check_plan(plan):
    if plan["controller_sha256"] != sha(ROOT / "utils/online_quantile_controller.py"):
        raise ValueError("controller snapshot changed")
    for key, path in (("model_sha256", MODEL), ("rule_sha256", RULE), ("split_sha256", SPLIT), ("script_sha256", Path(__file__))):
        if plan[key] != sha(path):
            raise ValueError(f"plan identity changed: {key}")
    if plan["gpu_uuid"] in FORBIDDEN or not plan["gpu_uuid"].startswith("GPU-"):
        raise ValueError("GPU not allowed")
    assignments = json.loads(SPLIT.read_text())["assignments"]
    names = plan["samples"]
    if not names or len(names) != len(set(names)) or any(assignments.get(n) != "fit" or not n.startswith("train/") for n in names):
        raise ValueError("sample whitelist is not exclusively train-fit")
    if plan["fps"] <= 0 or plan["repetitions"] < 1:
        raise ValueError("invalid replay plan")


def load_inputs(plan):
    with gzip.open(META, "rb") as f:
        rows = pickle.load(f)
    by_name = {r["name"]: r for r in rows if r["name"] in plan["samples"]}
    if set(by_name) != set(plan["samples"]):
        raise ValueError("train metadata missing whitelist member")
    with open(KPS, "rb") as f:
        all_kps = pickle.load(f)
    parts = {"pose": list(range(11)), "mouth_half": list(range(71, 91, 2)), "hand": list(range(91, 133))}
    indices = sum((parts[k] for k in sorted(["pose", "mouth_half", "hand"])), [])
    return by_name, {n: (all_kps[n]["keypoints"] if isinstance(all_kps[n], dict) else all_kps[n])[:, indices] for n in plan["samples"]}


def frame(zf, name, index):
    if not name.startswith("train/"):
        raise ValueError("forbidden video member")
    # The ZIP omits split prefixes. Only names verified against fit assignments
    # may be mapped to members; no ZIP enumeration or other split is loaded.
    member = f"images/{name[len('train/'):]}/images{index+1:04d}.png"
    with zf.open(member) as f:
        return np.asarray(Image.open(BytesIO(f.read())).convert("RGB")).copy()


def feature(prob, blank, bucket, sign):
    top = np.argpartition(prob, -8)[-8:]
    result = np.zeros(36, np.float32)
    result[:32] = np.bincount(bucket[top], weights=prob[top] * sign[top], minlength=32)
    ordered = np.sort(prob[top])
    result[32] = -np.sum(prob * np.log(np.maximum(prob, 1e-12)))
    result[33] = prob[blank]
    result[34] = ordered[-1]
    result[35] = ordered[-1] - ordered[-2]
    return result.tolist()


def decode(logits, starts, vocab, blank):
    if not logits:
        return []
    tensor = torch.as_tensor(np.asarray(logits), dtype=torch.float32)
    pred = span_weighted_predictions(tensor, [s + 7.5 for s in starts], span=15, min_weight=.05).tolist()
    collapsed = []
    for p in pred:
        if not collapsed or p != collapsed[-1]:
            collapsed.append(p)
    return [vocab[p].upper() for p in collapsed if p != blank]


def committed_decode(logits, starts, vocab, blank):
    if not starts:
        return []
    tensor = torch.as_tensor(np.asarray(logits), dtype=torch.float32)
    pred = span_weighted_predictions(tensor, [s + 7.5 for s in starts], span=15, min_weight=.05).tolist()
    stable = [p for p, s in zip(pred, starts) if s + 7.5 <= starts[-1]]
    collapsed = []
    for p in stable:
        if not collapsed or p != collapsed[-1]:
            collapsed.append(p)
    return [vocab[p].upper() for p in collapsed if p != blank]


def edit_distance(a, b):
    row = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        next_row = [i]
        for j, y in enumerate(b, 1):
            next_row.append(min(next_row[-1] + 1, row[j] + 1, row[j-1] + (x != y)))
        row = next_row
    return row[-1]


def run_one(plan, name, meta, kps, zf, model, bundle, vocab, policy, repetition):
    n = int(meta["num_frames"])
    rng = np.random.default_rng(261130)
    bucket = rng.integers(0, 32, len(vocab))
    sign = rng.choice(np.asarray([-1., 1.], np.float32), size=len(vocab))
    blank = vocab.index("<blank>")
    paid = {}
    scores = ridge_score_callback(bundle, source_fold(name), paid)
    controller = OnlineQuantileController(scores) if policy == "quantile" else None
    balance = 0.
    starts, logits, events = [], [], []
    first_output = first_commit = None
    previous_commit = []
    controller_s = preprocess_s = model_s = wait_s = 0.
    base = time.perf_counter()

    def execute(start, arrived):
        nonlocal preprocess_s, model_s, first_output, first_commit, previous_commit
        # Dense sliding_windows uses seven left and eight right edge copies.
        ids = [min(n-1, max(0, start + j - 7)) for j in range(16)]
        if ids[-1] > arrived:
            raise RuntimeError("future frame requested")
        t0 = time.perf_counter()
        rgb = torch.from_numpy(np.stack([cache_lookup[i] for i in ids])).permute(0, 3, 1, 2).unsqueeze(0).float().div_(255)
        kp = torch.from_numpy(np.asarray(kps[ids])).unsqueeze(0).float()
        preprocess_s += time.perf_counter() - t0
        t0 = time.perf_counter()
        with torch.inference_mode():
            output = model(is_train=False, labels=torch.zeros(1, dtype=torch.long, device="cuda"), sgn_videos=[rgb.to("cuda")], sgn_keypoints=[kp.to("cuda")], epoch=None)
            logit = output["ensemble_last_gloss_logits"].float().detach().cpu().numpy().reshape(-1)
        torch.cuda.synchronize()
        model_s += time.perf_counter() - t0
        if logit.ndim != 1:
            raise RuntimeError("expected one posterior per paid window")
        probs = np.exp(logit - np.max(logit)); probs /= probs.sum()
        paid[start] = feature(probs, blank, bucket, sign)
        starts.append(start); logits.append(logit)
        now = time.perf_counter()
        if first_output is None:
            first_output = now - base
        if starts[-1] - starts[0] >= 7.5:
            current = committed_decode(logits, starts, vocab, blank)
            if current != previous_commit:
                previous_commit = current
                if first_commit is None and current:
                    first_commit = now - base
        events.append({"kind": "paid_window", "start": start, "arrival_frame": arrived, "t_s": now-base, "model_s": model_s, "committed_gloss": previous_commit.copy()})

    cache_lookup = {}
    bonus_available = {}
    cache_wait_s = 0.
    next_start = 0
    with torch.inference_mode():
        for i in range(n):
            due = base + i / plan["fps"]
            now = time.perf_counter()
            if now < due:
                time.sleep(due-now)
            wait_s += max(0., time.perf_counter() - now)
            t0 = time.perf_counter()
            cache_lookup[i] = frame(zf, name, i)
            if len(cache_lookup) > 25:
                cache_lookup.pop(i-25, None)
            preprocess_s += time.perf_counter() - t0
            events.append({"kind": "frame_arrival", "frame": i, "t_s": time.perf_counter()-base})
            # Window start s becomes available after frame s+8, or EOS right pad.
            while next_start < n and (next_start + 8 <= i or i == n-1):
                s = next_start
                if s % 4 == 0:
                    execute(s, i)
                    if controller:
                        t0 = time.perf_counter(); controller.start_block(s); controller_s += time.perf_counter()-t0
                if s % 4 == 2:
                    bonus_available[s] = time.perf_counter()
                if s % 4 == 3:
                    b = s - 3
                    t0 = time.perf_counter()
                    if controller:
                        bonus = controller.complete_block(b)
                    else:
                        balance = min(2., balance + .5)
                        bonus = balance >= 1.
                        if bonus: balance -= 1.
                    controller_s += time.perf_counter()-t0
                    delay = time.perf_counter() - bonus_available.pop(b+2)
                    if bonus:
                        cache_wait_s += delay
                        execute(b+2, i)
                    events.append({"kind": "block_decision", "block": b, "arrival_frame": i, "t_s": time.perf_counter()-base, "bonus": bool(bonus), "cache_wait_s": delay})
                next_start += 1
        if controller:
            controller.finish()
    final = decode(logits, starts, vocab, blank)
    ref = str(meta["gloss"]).split()
    return {"sample": name, "policy": policy, "repetition": repetition, "frames": n, "windows": len(starts), "bonuses": sum(s%4 == 2 for s in starts), "errors": edit_distance(ref, final), "reference": ref, "hypothesis": final, "first_output_s": first_output, "first_stable_commit_s": first_commit, "wall_s": time.perf_counter()-base, "controller_s": controller_s, "preprocess_s": preprocess_s, "model_s": model_s, "pacing_wait_s": wait_s, "cache_wait_s": cache_wait_s, "events": events}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--plan", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--smoke", action="store_true")
    args = p.parse_args()
    plan = json.loads(args.plan.read_text()); check_plan(plan)
    if args.output.exists():
        raise FileExistsError(args.output)
    cfg = yaml.safe_load(CONFIG.read_text()); cfg["device"] = torch.device("cuda")
    cfg["model"]["RecognitionNetwork"]["keypoint_s3d"]["in_channel"] = 63
    misc.logger = logging.getLogger("p6_stream_video")
    vocab = json.loads(VOCAB.read_text()); model = build_model(cfg, len(vocab), word_emb_tab=None)
    state = torch.load(CKPT, map_location="cuda"); model.load_state_dict(state["model_state"], strict=True); model.eval()
    bundle = FrozenRidgeBundle(MODEL, RULE)
    meta, keypoints = load_inputs(plan)
    names = plan["samples"][:1] if args.smoke else plan["samples"]
    repetitions = 1 if args.smoke else plan["repetitions"]
    rows = []
    with zipfile.ZipFile(VIDEO) as zf:
        for r in range(repetitions):
            for name in names:
                for policy in (("quantile", "uniform") if r % 2 == 0 else ("uniform", "quantile")):
                    result = run_one(plan, name, meta[name], keypoints[name], zf, model, bundle, vocab, policy, r)
                    rows.append(result)
                    print(json.dumps({k:v for k,v in result.items() if k != "events"}), flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
