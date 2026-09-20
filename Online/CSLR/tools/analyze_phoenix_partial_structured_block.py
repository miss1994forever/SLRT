#!/usr/bin/env python3
"""CPU-only structured 4-window block scheduling smoke.

Each completed block executes its offset-0 uniform skeleton and exactly one of
offsets 1/2/3.  The choice is made only when offset 3 arrives, so it requires
at most two candidate-window arrivals of bounded lookahead but never the EOS.
"""
import argparse, gzip, hashlib, importlib.util, json, os, random, sys, time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
OLD_ROOT = ROOT / "results/phoenix-2014t_ISLR/p3_partial32_rollout_predictor_closed_loop_smoke_v2_49faacc3"
TERMINAL_ROOT = ROOT / "results/phoenix-2014t_ISLR/p3_partial32_terminal_predictor_smoke_v1_49faacc3"
TERMINAL_DATA = TERMINAL_ROOT.with_name(TERMINAL_ROOT.name + "_dataset")
AUDIT_ROOT = ROOT / "results/phoenix-2014t_ISLR/p3_partial32_terminal_advantage_audit_smoke_v1_49faacc3"
OUTPUT_ROOT = ROOT / "results/phoenix-2014t_ISLR/p3_partial32_structured_block_smoke_v1_49faacc3"
DATA_ROOT = OUTPUT_ROOT.with_name(OUTPUT_ROOT.name + "_dataset")
KIND = "B2_bookkeeping_prefix"
SEED = 261017


def imp(name):
    path = ROOT / f"tools/{name}.py"; spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec); sys.modules[spec.name] = module; spec.loader.exec_module(module)
    return module


OLD = imp("analyze_phoenix_partial_rollout_predictor")
SELF = imp("build_phoenix_partial_rollout_dagger_self_dataset")
AUDIT = imp("analyze_phoenix_partial_terminal_advantage_audit")
CHRON, BUILDER = OLD.DATA_BUILDER.CHRON, OLD.DATA_BUILDER


def sha256_file(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for b in iter(lambda: f.read(1024 * 1024), b""): h.update(b)
    return h.hexdigest()


def config():
    prior = json.loads((TERMINAL_DATA / "resolved_config_preregistered.json").read_text())
    return {"experiment": "structured 4-window block scheduling smoke",
      "created_before_outcome_computation": True, "cpu_only": True,
      "subsets": prior["subsets"],
      "protocol": {"block": [0, 1, 2, 3], "hard_skeleton_offset": 0,
        "choose_exactly_one_bonus_from_offsets": [1, 2, 3],
        "uniform_bonus_offset": 2, "tail_incomplete_block_bonus": "none",
        "decision_time": "when offset3 arrives", "bounded_lookahead_candidate_arrivals": 2,
        "unknown_EOS": True, "EOS_topup": False, "target_rate": "approximately 50%",
        "max_gap": 4, "decoder": "fixed registered decoder"},
      "oracle": {"name": "receding terminal block oracle",
        "candidate_value": "final edit error after candidate plus fixed center-block continuation",
        "tie_order_offsets": [2, 1, 3], "future/reference/EOS": "label/oracle only"},
      "predictors": {"T0": "shared candidate scorer; bookkeeping only",
        "T1": "same scorer; bookkeeping + past-selected decoder prefix", "epochs": 5,
        "loss": "uniform-target cross entropy over all minimum-error candidates",
        "primary_action": "argmax among three; no threshold"},
      "offline_gate": ["oracle error < structured uniform error",
        "T1 optimal-set accuracy > T0", "T1 mean regret < T0",
        "T1 mean true reward versus center > 0"],
      "forbidden": ["GPU/CUDA", "dev", "test", "visual", "LM", "candidate expensive logits as input",
        "future/reference/EOS as input", "calibration tuning", "git commit"]}


def preregister(root):
    if root.exists(): raise FileExistsError(root)
    root.mkdir(parents=True); (root / "workers").mkdir()
    (root / "resolved_config_preregistered.json").write_text(json.dumps(config(), indent=2) + "\n")
    return config()


def load_config(root):
    value = json.loads((root / "resolved_config_preregistered.json").read_text())
    if value != config(): raise ValueError("preregistered configuration changed")
    return value


def block_uniform(total):
    selected = list(range(0, total, 4))
    selected += [s + 2 for s in range(0, total, 4) if s + 3 < total]
    return sorted(selected)


def center_continuation(total, next_skeleton):
    out = []
    for s in range(next_skeleton, total, 4):
        out.append(s)
        if s + 3 < total: out.append(s + 2)
    return out


def candidate_features(probabilities, selected, candidates, blank_id):
    values = []
    for start in candidates:
        row = OLD.online_feature_row(probabilities, selected, start, 1.0, blank_id)
        values.append({"bookkeeping": OLD.bookkeeping_features(row).tolist(),
                       "prefix": OLD.prefix_features(row).tolist()})
    return values


def decode_counts(probabilities, selected, reference, vocab, blank_id):
    _, counts = CHRON.decode_counts(probabilities, sorted(selected), reference, vocab, blank_id)
    return counts


def oracle_sample(name, result, logits, partition, vocab, blank_id, collect_rows=True):
    p = BUILDER.ORACLE.softmax_rows(np.asarray(logits)); ref = BUILDER.REPLAY.clean_phoenix_2014_trans(result["gls_ref"])
    selected, rows = [], []
    tie_rank = {2: 0, 1: 1, 3: 2}
    for s in range(0, len(p), 4):
        selected.append(s)
        if s + 3 >= len(p): continue
        candidates = [s + 1, s + 2, s + 3]
        future = center_continuation(len(p), s + 4)
        errors = []
        for candidate in candidates:
            counts = decode_counts(p, selected + [candidate] + future, ref, vocab, blank_id)
            errors.append(int(counts["error"]))
        minimum = min(errors)
        best = min((i for i, e in enumerate(errors) if e == minimum), key=lambda i: tie_rank[candidates[i] - s])
        if collect_rows:
            rows.append({"sample_id": name, "source_video_id": BUILDER.source_video(name), "partition": partition,
              "block_start": s, "decision_arrival": s + 3, "candidate_starts": candidates,
              "features": candidate_features(p, selected, candidates, blank_id),
              "label": {"terminal_errors": errors, "minimum_error": minimum,
                        "optimal_mask": [e == minimum for e in errors], "teacher_choice": best,
                        "center_reward_by_candidate": [errors[1] - e for e in errors]},
              "provenance": {"hard_skeleton_present": True, "prefix_past_selected_only": True,
                "candidate_logits_absent_from_inputs": True, "future_reference_EOS_label_only": True,
                "decision_after_offset3_arrival": True}})
        selected.append(candidates[best])
    uniform = block_uniform(len(p))
    return {"sample_id": name, "partition": partition, "dense_windows": len(p),
      "uniform": {"selected": uniform, "counts": decode_counts(p, uniform, ref, vocab, blank_id),
                  "coverage": CHRON.coverage_metrics(uniform, len(p))},
      "oracle": {"selected": selected, "counts": decode_counts(p, selected, ref, vocab, blank_id),
                 "coverage": CHRON.coverage_metrics(selected, len(p))}, "rows": rows}


def worker(root, index, workers):
    os.environ["CUDA_VISIBLE_DEVICES"] = ""; torch.set_num_threads(1)
    cfg = load_config(root)
    all_ids = [(x, "fit") for x in cfg["subsets"]["fit"]["sample_ids"]] + [(x, "calibration") for x in cfg["subsets"]["calibration"]["sample_ids"]]
    assigned = {x: p for pos, (x, p) in enumerate(all_ids) if pos % workers == index}
    final = root / "workers" / f"worker-{index:02d}-of-{workers:02d}.jsonl.gz"; marker = final.with_suffix(".complete.json")
    if final.exists() and marker.exists() and json.loads(marker.read_text())["sha256"] == sha256_file(final): return json.loads(marker.read_text()) | {"resumed": True}
    if final.exists() or marker.exists(): raise FileExistsError(final)
    vocab = json.loads(CHRON.DEFAULT_VOCAB.read_text()); blank = vocab.index("<blank>")
    dense = CHRON.DEFAULT_DENSE_ROOT; indices, shard_count = CHRON.completed_shard_indices(dense)
    temp = final.with_name("." + final.name + f".incomplete-{os.getpid()}"); done = 0; started = time.perf_counter(); counts = Counter()
    with gzip.open(temp, "wt", encoding="utf-8", compresslevel=5) as out:
        for shard in indices:
            results, logits, _ = BUILDER.BUILDER.validate_dense_shard(dense, shard, shard_count, verify_hashes=False)
            for name in results:
                if name not in assigned: continue
                value = oracle_sample(name, results[name], logits[name], assigned[name], vocab, blank)
                for row in value.pop("rows"): out.write(json.dumps({"type": "block", **row}, separators=(",", ":")) + "\n"); counts["blocks"] += 1
                out.write(json.dumps({"type": "sample", **value}, separators=(",", ":")) + "\n")
                counts["samples"] += 1; done += 1
                print(json.dumps({"worker": index, "done": done, "assigned": len(assigned), "elapsed": time.perf_counter()-started}), flush=True)
    if done != len(assigned): raise RuntimeError("worker coverage mismatch")
    temp.replace(final); info = {"status": "complete", "worker": index, "workers": workers, "samples": done,
      "counts": dict(counts), "seconds": time.perf_counter()-started, "sha256": sha256_file(final), "bytes": final.stat().st_size}
    marker.write_text(json.dumps(info, indent=2) + "\n"); return info


def finalize_dataset(root, workers):
    cfg = load_config(root); info = []; counts = Counter()
    for i in range(workers):
        path = root / "workers" / f"worker-{i:02d}-of-{workers:02d}.jsonl.gz"; m = json.loads(path.with_suffix(".complete.json").read_text())
        if sha256_file(path) != m["sha256"]: raise ValueError("worker hash mismatch")
        info.append(m); counts.update(m["counts"])
    if counts["samples"] != 640: raise ValueError("sample count mismatch")
    value = {"status": "complete", "created_utc": datetime.now(timezone.utc).isoformat(), "counts": dict(counts),
             "config_sha256": sha256_file(root / "resolved_config_preregistered.json"), "workers": info}
    (root / "dataset_manifest.json").write_text(json.dumps(value, indent=2) + "\n"); return value


def load_rows(root):
    manifest = json.loads((root / "dataset_manifest.json").read_text()); blocks = {"fit": [], "calibration": []}; samples = {"fit": [], "calibration": []}
    for info in manifest["workers"]:
        path = root / "workers" / f"worker-{info['worker']:02d}-of-{info['workers']:02d}.jsonl.gz"
        if sha256_file(path) != info["sha256"]: raise ValueError("hash mismatch")
        with gzip.open(path, "rt", encoding="utf-8") as f:
            for line in f:
                x = json.loads(line); target = blocks if x.pop("type") == "block" else samples
                target[x["partition"]].append(x)
    return blocks, samples, manifest


class BlockScorer(nn.Module):
    def __init__(self, width):
        super().__init__(); self.net = nn.Sequential(nn.Linear(width, 48), nn.GELU(), nn.Linear(48, 32), nn.GELU(), nn.Linear(32, 1))
    def forward(self, x): return self.net(x).squeeze(-1)


def arrays(rows, use_prefix, stats=None):
    raw = []
    for row in rows:
        raw.append(np.asarray([f["bookkeeping"] + (f["prefix"] if use_prefix else []) for f in row["features"]], dtype=np.float32))
    x = np.stack(raw); errors = np.asarray([r["label"]["terminal_errors"] for r in rows], dtype=np.float32)
    optimal = np.asarray([r["label"]["optimal_mask"] for r in rows], dtype=np.float32)
    if stats is None:
        mean = x.mean(axis=(0, 1), dtype=np.float64).astype(np.float32); scale = x.std(axis=(0, 1), dtype=np.float64).astype(np.float32); scale[scale < 1e-6] = 1
        stats = (mean, scale)
    x = (x - stats[0]) / stats[1]
    return x, errors, optimal, stats


def train(rows, use_prefix):
    x, errors, optimal, stats = arrays(rows, use_prefix); torch.manual_seed(SEED); model = BlockScorer(x.shape[2]); optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    target = optimal / optimal.sum(axis=1, keepdims=True); generator = torch.Generator().manual_seed(SEED); history = []
    for epoch in range(5):
        order = torch.randperm(len(x), generator=generator); total = 0.
        for left in range(0, len(x), 512):
            idx = order[left:left+512].numpy(); logits = model(torch.from_numpy(x[idx]).float()); t = torch.from_numpy(target[idx]).float()
            loss = -(t * F.log_softmax(logits, dim=1)).sum(dim=1).mean(); optimizer.zero_grad(); loss.backward(); optimizer.step(); total += float(loss) * len(idx)
        history.append(total / len(x))
    model.eval(); return model, stats, history


def choose(model, stats, row, use_prefix):
    x, _, _, _ = arrays([row], use_prefix, stats)
    with torch.no_grad(): score = model(torch.from_numpy(x).float())[0].numpy()
    return int(np.argmax(score)), score.tolist()


def offline_metrics(model, stats, rows, use_prefix):
    chosen = []; regrets = []; rewards = []; optimal_hits = []
    for row in rows:
        c, _ = choose(model, stats, row, use_prefix); e = row["label"]["terminal_errors"]
        chosen.append(c); regrets.append(e[c] - min(e)); rewards.append(e[1] - e[c]); optimal_hits.append(e[c] == min(e))
    return {"blocks": len(rows), "optimal_set_accuracy": float(np.mean(optimal_hits)), "mean_terminal_error_regret": float(np.mean(regrets)),
            "total_terminal_error_regret": int(sum(regrets)), "mean_true_reward_vs_center": float(np.mean(rewards)),
            "choice_counts": {str(k): int(v) for k, v in sorted(Counter(chosen).items())}}


def predictor_sample(name, result, logits, model, stats, use_prefix, vocab, blank):
    p = BUILDER.ORACLE.softmax_rows(np.asarray(logits)); ref = BUILDER.REPLAY.clean_phoenix_2014_trans(result["gls_ref"]); selected = []
    triggers = Counter()
    for s in range(0, len(p), 4):
        selected.append(s)
        if s + 3 >= len(p): continue
        candidates = [s+1, s+2, s+3]; fake = {"features": candidate_features(p, selected, candidates, blank), "label": {"terminal_errors": [0,0,0], "optimal_mask": [1,1,1]}}
        c, _ = choose(model, stats, fake, use_prefix); selected.append(candidates[c]); triggers[c] += 1
    return {"selected": selected, "counts": decode_counts(p, selected, ref, vocab, blank), "dense_windows": len(p),
            "coverage": CHRON.coverage_metrics(selected, len(p)), "choice_counts": dict(triggers)}


def summarize(rows, uniform):
    counts = CHRON.aggregate_counts([x["counts"] for x in rows]); base = CHRON.aggregate_counts([x["counts"] for x in uniform]); dense = sum(x["dense_windows"] for x in rows)
    return {"metrics": counts, "delta_wer_pp_vs_structured_uniform": counts["wer"] - base["wer"], "error_delta_vs_structured_uniform": counts["error"] - base["error"],
      "executed_windows": sum(len(x["selected"]) for x in rows), "dense_windows": dense, "actual_window_rate": sum(len(x["selected"]) for x in rows)/dense,
      "max_gap": max(x["coverage"]["max_gap_including_endpoints"] for x in rows),
      "paired_bootstrap_vs_structured_uniform": CHRON.paired_bootstrap([x["counts"] for x in rows], [x["counts"] for x in uniform])}


def analyze(data_root, output_root):
    if output_root.exists(): raise FileExistsError(output_root)
    cfg = load_config(data_root); blocks, samples, manifest = load_rows(data_root)
    models = {}; offline = {}; training = {}
    for label, use_prefix in (("T0", False), ("T1", True)):
        model, stats, history = train(blocks["fit"], use_prefix); models[label] = (model, stats, use_prefix); training[label] = history; offline[label] = offline_metrics(model, stats, blocks["calibration"], use_prefix)
    uniform_rows = [{**x["uniform"], "dense_windows": x["dense_windows"]} for x in samples["calibration"]]
    oracle_rows = [{**x["oracle"], "dense_windows": x["dense_windows"]} for x in samples["calibration"]]
    uniform_summary = summarize(uniform_rows, uniform_rows); oracle_summary = summarize(oracle_rows, uniform_rows)
    gate_checks = {"oracle_error_lt_uniform": oracle_summary["metrics"]["error"] < uniform_summary["metrics"]["error"],
      "T1_accuracy_gt_T0": offline["T1"]["optimal_set_accuracy"] > offline["T0"]["optimal_set_accuracy"],
      "T1_regret_lt_T0": offline["T1"]["mean_terminal_error_regret"] < offline["T0"]["mean_terminal_error_regret"],
      "T1_reward_vs_center_gt_0": offline["T1"]["mean_true_reward_vs_center"] > 0}
    gate = {"passed": all(gate_checks.values()), "checks": gate_checks}
    closed = None
    if gate["passed"]:
        chosen_ids = set(cfg["subsets"]["calibration"]["sample_ids"]); vocab = json.loads(CHRON.DEFAULT_VOCAB.read_text()); blank = vocab.index("<blank>")
        outputs = {"T0": [], "T1": []}; dense = CHRON.DEFAULT_DENSE_ROOT; indices, n = CHRON.completed_shard_indices(dense)
        for shard in indices:
            results, logits, _ = BUILDER.BUILDER.validate_dense_shard(dense, shard, n, verify_hashes=False)
            for name in results:
                if name not in chosen_ids: continue
                for label in outputs: outputs[label].append(predictor_sample(name, results[name], logits[name], *models[label], vocab, blank))
        closed = {label: summarize(rows, uniform_rows) for label, rows in outputs.items()}
    temp = output_root.with_name("." + output_root.name + f".incomplete-{os.getpid()}"); temp.mkdir(parents=True)
    result = {"scope": "partial32 structured block scheduling smoke", "gpu_used": False, "dev_used": False, "test_used": False,
      "dataset": manifest, "structured_uniform": uniform_summary, "structured_block_oracle": oracle_summary,
      "offline_predictors": offline, "training": training, "gate": gate, "closed_loop": closed,
      "B2_old_frozen128_reference": json.loads((AUDIT_ROOT/"metrics.json").read_text())["policies"]["B2_old"]["base_subset"],
      "limitations": ["32/56 non-random train shards", "train calibration repeatedly inspected", "fixed hard 128 subset",
        "oracle labels use future/reference/EOS", "block choice has up to two candidate-arrival bounded lookahead", "no dev/test/wall-time/latency claim"]}
    (temp/"resolved_config_preregistered.json").write_text(json.dumps(cfg, indent=2)+"\n"); (temp/"metrics.json").write_text(json.dumps(result, indent=2)+"\n")
    for label, (model, stats, use_prefix) in models.items(): torch.save({"state_dict": model.state_dict(), "stats": stats, "use_prefix": use_prefix}, temp/f"{label}.pt")
    (temp/"manifest.json").write_text(json.dumps({"status":"complete","created_utc":datetime.now(timezone.utc).isoformat(),"metrics_sha256":sha256_file(temp/"metrics.json")},indent=2)+"\n"); temp.replace(output_root); return result


def main():
    p=argparse.ArgumentParser(); p.add_argument("mode",choices=("preregister","worker","finalize-dataset","analyze")); p.add_argument("--data-root",type=Path,default=DATA_ROOT); p.add_argument("--output-root",type=Path,default=OUTPUT_ROOT); p.add_argument("--worker-index",type=int,default=0); p.add_argument("--workers",type=int,default=1); a=p.parse_args(); os.environ["CUDA_VISIBLE_DEVICES"]=""
    if a.mode=="preregister": x=preregister(a.data_root.resolve())
    elif a.mode=="worker": x=worker(a.data_root.resolve(),a.worker_index,a.workers)
    elif a.mode=="finalize-dataset": x=finalize_dataset(a.data_root.resolve(),a.workers)
    else: x=analyze(a.data_root.resolve(),a.output_root.resolve())
    print(json.dumps(x,indent=2))


if __name__=="__main__": main()
