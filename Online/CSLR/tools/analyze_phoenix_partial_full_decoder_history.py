#!/usr/bin/env python3
"""Fit-only OOF smoke using the ordered logits of already executed windows."""
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
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
FRESH = ROOT / "results/phoenix-2014t_ISLR/p3_partial32_fresh_terminal_listwise_exploratory_v1_49faacc3"
DATA = FRESH.with_name(FRESH.name + "_dataset")
OUTPUT = ROOT / "results/phoenix-2014t_ISLR/p3_partial32_full_decoder_history_fit2541_oof_v1_49faacc3"
SEED = 261026
FOLD_SEED = 261022
FOLDS = 5
HISTORY = 16
PCA_DIM = 16
PCA_Q = 24
PCA_NITER = 3
META_DIM = 14
EPOCHS = 20
BATCH = 128
FALSE_OVERRIDE_COST = 4.0
COVERAGES = (0.005, 0.01, 0.02, 0.05, 0.10)


def imp(name):
    path = ROOT / f"tools/{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec); sys.modules[spec.name] = module; spec.loader.exec_module(module)
    return module


FRESHMOD = imp("analyze_phoenix_partial_fresh_terminal_listwise")
SELECTIVE = imp("analyze_phoenix_partial_selective_residual")
BLOCK, CHRON, BUILDER = FRESHMOD.BLOCK, FRESHMOD.CHRON, FRESHMOD.BUILDER


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""): digest.update(chunk)
    return digest.hexdigest()


def split_spec():
    return json.loads((FRESH / "resolved_config_preregistered.json").read_text())["split"]


def source_fold(source):
    value = hashlib.sha256(f"{FOLD_SEED}\0{source}".encode()).hexdigest()
    return int(value[:16], 16) % FOLDS


def config():
    split = split_spec()
    return {
        "experiment": "fit-only full paid decoder-history residual predictor source-group OOF",
        "created_before_OOF_outcomes": True, "cpu_only": True,
        "scope": {"fit_samples": 2541, "fit_sources": 235, "fit_blocks": 73446,
                  "informative_terminal_blocks": 971, "split": split, "terminal_rows": str(DATA),
                  "dense_logits": str(CHRON.DEFAULT_DENSE_ROOT)},
        "old_state_audit": {
            "B2": "9 bookkeeping scalars + 40 aggregate prefix features (length/path length, aggregate/last entropy, delta, blank/repeat flags, hash32); no ordered per-executed-window logits",
            "Tpreview": "the same 49-d summary plus a candidate-local 31x17 pose/hand preview; no ordered paid ISLR-logit history",
            "new_information": "ordered last 16 already-executed full ISLR logits plus per-execution offset/gap/time and prefix token/entropy/change trajectory",
        },
        "behavior_history": {
            "frozen_policy": "structured fixed-center: execute offset0 skeleton and offset2 center in every complete block",
            "decision_state": "past fixed-center executions plus current block's already-paid offset0 skeleton",
            "history_length": HISTORY, "allowed_current_logits": "offset0 skeleton only",
            "forbidden_current_logits": ["offset1", "offset2", "offset3"],
            "tail": "terminal-row complete blocks only; unknown EOS and no tail top-up",
            "prefix_reconstruction": "incremental span15 decoder state using only executions available at that step",
            "per_step_metadata": ["offset0/offset2 one-hot", "log gap", "log absolute window time",
                "prefix mean entropy", "prefix last entropy", "entropy change", "prefix changed",
                "prefix last-token blank", "log prefix length", "four deterministic token-hash signs"],
        },
        "PCA": {"dimensions": PCA_DIM, "method": "torch randomized low-rank PCA", "q": PCA_Q,
                "niter": PCA_NITER, "fit": "all executed logits from source-fold training videos only",
                "normalization": "PCA coordinates and metadata standardized within source-fold training executions"},
        "models": {
            "summary_control": "shared candidate MLP 49->128->32->1 (10561 parameters)",
            "history": "GRU(input=30=PCA16+meta14, hidden=32) then shared candidate MLP (49+32)->48->1 (~10225 parameters)",
            "output": "offset1/3 score minus center score", "parameter_matching": "within 5%",
            "training": "informative blocks only; SmoothL1 signed center-minus-side advantage, 4x false-positive override cost",
            "optimizer": "AdamW(lr=1e-3, weight_decay=1e-4)", "epochs": EPOCHS, "batch": BATCH, "seed": SEED,
        },
        "OOF": {"folds": FOLDS, "assignment": "sha256(261022+NUL+source_video_id) mod5",
                "coverages": list(COVERAGES), "threshold": "method-specific global OOF positive max-side-score quantile",
                "pass": "history at some coverage has reward>0, all folds>=0, >=20 overrides, reward>summary and regret<summary"},
        "cost": "report amortized heldout PCA projection and batched GRU/scorer CPU milliseconds per block; excludes already-paid ISLR inference",
        "after_pass": "still requires on-policy closed-loop validation because histories here come from frozen fixed-center behavior",
        "forbidden": ["GPU/CUDA", "dev", "test", "calibration/evaluation outcomes", "future/reference/EOS", "unexecuted candidate logits", "git commit"],
    }


def preregister(root):
    if root.exists(): raise FileExistsError(root)
    root.mkdir(parents=True); value = config()
    (root / "resolved_config_preregistered.json").write_text(json.dumps(value, indent=2) + "\n")
    return {"status":"preregistered", "fit_samples":2541, "fit_blocks":73446, "history":HISTORY,
            "PCA_dimensions":PCA_DIM, "epochs":EPOCHS, "coverages":list(COVERAGES)}


def loadcfg(root):
    value = json.loads((root / "resolved_config_preregistered.json").read_text())
    if value != config(): raise ValueError("preregistered configuration changed")
    return value


def load_fit_rows():
    manifest = json.loads((DATA / "dataset_manifest.json").read_text()); rows = []
    for info in manifest["workers"]:
        path = DATA / "workers" / f"worker-{info['worker']:02d}-of-{info['workers']:02d}.jsonl.gz"
        if sha(path) != info["sha256"]: raise ValueError("terminal dataset hash mismatch")
        with gzip.open(path, "rt", encoding="utf-8") as handle:
            for line in handle:
                row = json.loads(line)
                if row["partition"] == "fit": rows.append(row)
    rows.sort(key=lambda x: (x["sample_id"], x["block_start"]))
    return rows, manifest


def collapse(tokens, blank):
    out = []; previous = None
    for token in tokens:
        token = int(token)
        if token != previous and token != blank: out.append(token)
        previous = token
    return tuple(out)


def token_hash4(token):
    digest = hashlib.sha256(str(int(token)).encode()).digest()
    return [1.0 if digest[i] & 1 else -1.0 for i in range(4)]


def execution_trace(logits, blank):
    probabilities = BUILDER.ORACLE.softmax_rows(np.asarray(logits, np.float32))
    starts = np.arange(0, len(probabilities), 2, dtype=np.int64)
    raw_scores = np.zeros((len(starts), probabilities.shape[1]), np.float32)
    tokens = np.full(len(starts), blank, np.int64); entropies = np.zeros(len(starts), np.float32)
    meta = np.zeros((len(starts), META_DIM), np.float32); previous_prefix = (); previous_mean = 0.0
    for k, start in enumerate(starts):
        near = np.flatnonzero(np.abs(starts[:k+1] - start) <= 7.5)
        for j in near:
            distance = abs(int(starts[j]) - int(start)); weight = max(1.0-distance/7.5, 0.05)
            raw_scores[j] += np.float32(weight) * probabilities[start]
        raw_scores[k] = 0
        for j in near:
            distance = abs(int(starts[j]) - int(start)); weight = max(1.0-distance/7.5, 0.05)
            raw_scores[k] += np.float32(weight) * probabilities[starts[j]]
        for j in near:
            q = raw_scores[j] / max(float(raw_scores[j].sum()), 1e-12)
            tokens[j] = int(q.argmax()); entropies[j] = float(-np.sum(q*np.log(np.maximum(q, 1e-12))))
        prefix = collapse(tokens[:k+1], blank); mean_entropy = float(entropies[:k+1].mean()); last_token = prefix[-1] if prefix else blank
        offset0 = int(start % 4 == 0); gap = int(start-starts[k-1]) if k else int(start+1)
        meta[k] = np.asarray([float(offset0), float(1-offset0), math.log1p(gap), math.log1p(int(start)+1),
            mean_entropy, float(entropies[k]), mean_entropy-previous_mean, float(prefix != previous_prefix),
            float(last_token == blank), math.log1p(len(prefix)), *token_hash4(last_token)], np.float32)
        previous_prefix, previous_mean = prefix, mean_entropy
    return starts, np.asarray(logits, np.float16)[starts], meta


def build_data():
    rows, manifest = load_fit_rows(); wanted = {x["sample_id"] for x in rows}; vocab = json.loads(CHRON.DEFAULT_VOCAB.read_text()); blank = vocab.index("<blank>")
    dense = {}; indices, workers = CHRON.completed_shard_indices(CHRON.DEFAULT_DENSE_ROOT)
    for shard in indices:
        results, logits, _ = BUILDER.BUILDER.validate_dense_shard(CHRON.DEFAULT_DENSE_ROOT, shard, workers, verify_hashes=False)
        for name in results:
            if name in wanted: dense[name] = logits[name]
    if set(dense) != wanted: raise ValueError("dense fit coverage mismatch")
    logits_parts = []; meta_parts = []; exec_folds = []; sample_exec = {}; offset = 0
    sample_source = {row["sample_id"]:row["source_video_id"] for row in rows}
    for name in sorted(wanted):
        starts, values, meta = execution_trace(dense[name], blank); count = len(starts)
        sample_exec[name] = (starts, np.arange(offset, offset+count, dtype=np.int32)); offset += count
        logits_parts.append(values); meta_parts.append(meta); exec_folds.extend([source_fold(sample_source[name])]*count)
    base=[]; errors=[]; sources=[]; samples=[]; history_indices=np.full((len(rows), HISTORY), -1, np.int32); lengths=[]
    for i,row in enumerate(rows):
        base.append(np.asarray([x["bookkeeping"]+x["prefix"] for x in row["features"]], np.float32)); errors.append(row["label"]["terminal_errors"])
        sources.append(row["source_video_id"]); samples.append(row["sample_id"])
        starts, global_idx = sample_exec[row["sample_id"]]; available = global_idx[starts <= int(row["block_start"])][-HISTORY:]
        history_indices[i,:len(available)] = available; lengths.append(len(available))
    errors=np.asarray(errors,np.float32); result={"base":np.asarray(base,np.float32),"errors":errors,"advantage":errors[:,1:2]-errors[:,[0,2]],
        "informative":np.any(errors!=errors[:,1:2],axis=1),"source":np.asarray(sources),"sample":np.asarray(samples),
        "history_indices":history_indices,"lengths":np.asarray(lengths,np.int64),"executed_logits":np.concatenate(logits_parts),
        "executed_meta":np.concatenate(meta_parts),"exec_folds":np.asarray(exec_folds,np.int64),"dataset_manifest":manifest}
    if (len(set(samples)),len(set(sources)),len(rows),int(result["informative"].sum())) != (2541,235,73446,971): raise ValueError("frozen fit scale mismatch")
    return result


def fit_stats(data, train_mask, fold):
    base=data["base"][train_mask]; bm=base.mean((0,1),dtype=np.float64).astype(np.float32); bs=base.std((0,1),dtype=np.float64).astype(np.float32);bs[bs<1e-6]=1
    em=data["exec_folds"]!=fold; meta=data["executed_meta"][em]; mm=meta.mean(0,dtype=np.float64).astype(np.float32);ms=meta.std(0,dtype=np.float64).astype(np.float32);ms[ms<1e-6]=1
    values=torch.from_numpy(np.asarray(data["executed_logits"][em],np.float32));mean=values.mean(0);centered=values-mean;torch.manual_seed(SEED+fold)
    _,singular,vectors=torch.pca_lowrank(centered,q=PCA_Q,center=False,niter=PCA_NITER);components=vectors[:,:PCA_DIM].contiguous();variance=singular[:PCA_DIM].square()/max(1,len(values)-1);total=centered.square().sum()/max(1,len(values)-1)
    return {"base":(bm,bs),"meta":(mm,ms),"pca_mean":mean.numpy(),"pca_components":components.numpy(),
            "pca_scale":variance.sqrt().clamp_min(1e-6).numpy(),
            "pca_explained":float(variance.sum()/total),"pca_train_executions":int(em.sum())}


def fold_arrays(data, stats):
    tick=time.perf_counter(); projected=((np.asarray(data["executed_logits"],np.float32)-stats["pca_mean"])@stats["pca_components"])/stats["pca_scale"]; projection_seconds=time.perf_counter()-tick
    projected=projected.astype(np.float32)
    meta=((data["executed_meta"]-stats["meta"][0])/stats["meta"][1]).astype(np.float32)
    history=np.zeros((len(data["base"]),HISTORY,PCA_DIM+META_DIM),np.float32); valid=data["history_indices"]>=0
    idx=data["history_indices"][valid]; history[valid]=np.concatenate([projected[idx],meta[idx]],axis=1)
    base=((data["base"]-stats["base"][0])/stats["base"][1]).astype(np.float32)
    return base,history,projection_seconds


class SummaryMLP(nn.Module):
    def __init__(self): super().__init__(); self.net=nn.Sequential(nn.Linear(49,128),nn.GELU(),nn.Linear(128,32),nn.GELU(),nn.Linear(32,1))
    def forward(self,base,history,lengths):
        b,c,w=base.shape;return self.net(base.reshape(b*c,w)).reshape(b,c)


class HistoryGRU(nn.Module):
    def __init__(self):
        super().__init__();self.gru=nn.GRU(PCA_DIM+META_DIM,32,batch_first=True);self.score=nn.Sequential(nn.Linear(49+32,48),nn.GELU(),nn.Linear(48,1))
    def forward(self,base,history,lengths):
        output,_=self.gru(history);last=output[torch.arange(len(output)),lengths-1];expanded=last[:,None,:].expand(-1,3,-1);value=torch.cat([base,expanded],2);return self.score(value).squeeze(-1)


def parameter_count(model): return sum(x.numel() for x in model.parameters())


def train_model(model,base,history,lengths,data,train_mask,seed):
    idx=np.flatnonzero(train_mask&data["informative"]);optimizer=torch.optim.AdamW(model.parameters(),lr=1e-3,weight_decay=1e-4);generator=torch.Generator().manual_seed(seed);losses=[]
    for _ in range(EPOCHS):
        order=idx[torch.randperm(len(idx),generator=generator).numpy()];total=0.
        for left in range(0,len(order),BATCH):
            part=order[left:left+BATCH];score=model(torch.from_numpy(base[part]).float(),torch.from_numpy(history[part]).float(),torch.from_numpy(lengths[part]));pred=score[:,[0,2]]-score[:,1:2];target=torch.from_numpy(data["advantage"][part]).float()
            per=F.smooth_l1_loss(pred,target,reduction="none");weight=torch.where((target<=0)&(pred>0),torch.full_like(per,FALSE_OVERRIDE_COST),torch.ones_like(per));loss=(per*weight).mean();optimizer.zero_grad();loss.backward();optimizer.step();total+=float(loss)*len(part)
        losses.append(total/len(order))
    model.eval();return losses


def predict(model,base,history,lengths,mask):
    idx=np.flatnonzero(mask);out=[];tick=time.perf_counter()
    with torch.no_grad():
        for left in range(0,len(idx),1024):
            part=idx[left:left+1024];score=model(torch.from_numpy(base[part]).float(),torch.from_numpy(history[part]).float(),torch.from_numpy(lengths[part]));out.append((score[:,[0,2]]-score[:,1:2]).numpy())
    return idx,np.concatenate(out),time.perf_counter()-tick


def reports(data,scores,folds):
    return {str(c):{"nominal_coverage":c,**SELECTIVE.policy_metrics(data,scores,SELECTIVE.candidate_threshold(scores,c),folds)} for c in COVERAGES}


def run_oof(root):
    loadcfg(root);(root/"oof_started.marker").write_text(datetime.now(timezone.utc).isoformat()+"\n");os.environ["CUDA_VISIBLE_DEVICES"]="";random.seed(SEED);np.random.seed(SEED);torch.manual_seed(SEED);torch.set_num_threads(min(8,os.cpu_count() or 1));torch.use_deterministic_algorithms(True)
    started=time.perf_counter();data=build_data();folds=np.asarray([source_fold(x) for x in data["source"]],np.int64);summary_scores=np.full((len(folds),2),np.nan,np.float32);history_scores=np.full_like(summary_scores,np.nan);fold_reports={}
    for fold in range(FOLDS):
        train,held=folds!=fold,folds==fold;stats=fit_stats(data,train,fold);base,history,projection_seconds=fold_arrays(data,stats);torch.manual_seed(SEED+fold);summary=SummaryMLP();torch.manual_seed(SEED+fold);full=HistoryGRU()
        summary_history=train_model(summary,base,history,data["lengths"],data,train,SEED+100+fold);full_history=train_model(full,base,history,data["lengths"],data,train,SEED+100+fold)
        sidx,sscore,summary_seconds=predict(summary,base,history,data["lengths"],held);hidx,hscore,history_seconds=predict(full,base,history,data["lengths"],held)
        summary_scores[sidx]=sscore;history_scores[hidx]=hscore;held_exec=data["exec_folds"]==fold
        fold_reports[str(fold)]={"train_sources":len(set(data["source"][train])),"heldout_sources":len(set(data["source"][held])),"train_blocks":int(train.sum()),"heldout_blocks":int(held.sum()),"informative_train_blocks":int((train&data["informative"]).sum()),
            "PCA_explained_variance_ratio":stats["pca_explained"],"PCA_train_executions":stats["pca_train_executions"],"heldout_executions":int(held_exec.sum()),
            "controller_CPU":{"full_cache_PCA_projection_ms_per_execution":1000*projection_seconds/len(data["executed_logits"]),"summary_inference_ms_per_block":1000*summary_seconds/held.sum(),"history_GRU_inference_ms_per_block":1000*history_seconds/held.sum()},
            "training_history":{"summary":summary_history,"history":full_history}}
        print(json.dumps({"fold_complete":fold,"PCA_explained":stats["pca_explained"],"seconds":time.perf_counter()-started}),flush=True)
    if np.isnan(summary_scores).any() or np.isnan(history_scores).any():raise RuntimeError("incomplete OOF")
    summary_report=reports(data,summary_scores,folds);history_report=reports(data,history_scores,folds);comparison=[]
    for coverage in COVERAGES:
        key=str(coverage);a,b=summary_report[key],history_report[key];checks={"aggregate_reward_gt_0":b["cumulative_terminal_reward"]>0,"every_fold_reward_ge_0":all(v>=0 for v in b["fold_reward"].values()),"at_least_20_overrides":b["overrides"]>=20,"reward_gt_summary":b["cumulative_terminal_reward"]>a["cumulative_terminal_reward"],"regret_lt_summary":b["total_terminal_regret"]<a["total_terminal_regret"]}
        comparison.append({"nominal_coverage":coverage,"passed":all(checks.values()),"checks":checks,"history_minus_summary_reward":b["cumulative_terminal_reward"]-a["cumulative_terminal_reward"],"history_minus_summary_regret":b["total_terminal_regret"]-a["total_terminal_regret"]})
    passing=[x for x in comparison if x["passed"]];selected=sorted(passing,key=lambda x:(-history_report[str(x["nominal_coverage"])]["cumulative_terminal_reward"],x["nominal_coverage"]))[0] if passing else None
    lengths=data["lengths"];result={"scope":"fit2541 full paid decoder-history source-group OOF","gpu_used":False,"dev_used":False,"test_used":False,"calibration_or_evaluation_outcomes_read":False,
        "data":{"samples":len(set(data["sample"])),"sources":len(set(data["source"])),"blocks":len(data["errors"]),"informative_blocks":int(data["informative"].sum()),"executed_history_records":len(data["executed_logits"]),"assembly_seconds":time.perf_counter()-started},
        "history_availability":{"maximum":HISTORY,"mean":float(lengths.mean()),"median":float(np.median(lengths)),"minimum":int(lengths.min()),"full_length_blocks":int((lengths==HISTORY).sum()),"full_length_rate":float(np.mean(lengths==HISTORY))},
        "models":{"summary_parameters":parameter_count(SummaryMLP()),"history_parameters":parameter_count(HistoryGRU())},"empty_policy":SELECTIVE.empty_metrics(data,folds),"summary_control":summary_report,"full_history":history_report,"comparison":comparison,"fold_reports":fold_reports,
        "gate":{"passed":selected is not None,"selected":selected},"distribution_warning":"offline histories come from frozen fixed-center behavior; any pass still requires on-policy closed-loop validation","decision":"eligible for separate on-policy validation" if selected else "stop full-history predictor; do not read calibration"}
    (root/"metrics.json").write_text(json.dumps(result,indent=2)+"\n");(root/"manifest.json").write_text(json.dumps({"status":"fit_oof_complete","created_utc":datetime.now(timezone.utc).isoformat(),"gate_passed":selected is not None,"calibration_outcomes_read":False,"evaluation_outcomes_read":False,"metrics_sha256":sha(root/"metrics.json")},indent=2)+"\n");return result


def main():
    parser=argparse.ArgumentParser();parser.add_argument("mode",choices=("preregister","oof"));parser.add_argument("--output-root",type=Path,default=OUTPUT);args=parser.parse_args();os.environ["CUDA_VISIBLE_DEVICES"]="";root=args.output_root.resolve();result=preregister(root) if args.mode=="preregister" else run_oof(root);print(json.dumps(result,indent=2))


if __name__=="__main__":main()
