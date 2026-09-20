#!/usr/bin/env python3
"""Two-stage CPU-only causal phase/interior-entry-hazard scheduling smoke."""
import argparse
import gzip
import hashlib
import importlib.util
import json
import os
import pickle
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parents[1]
FRESH = ROOT / "results/phoenix-2014t_ISLR/p3_partial32_fresh_terminal_listwise_exploratory_v1_49faacc3"
DATA = FRESH.with_name(FRESH.name + "_dataset")
HOG_ROOT = ROOT / "results/phoenix-2014t_ISLR/p3_partial32_hog_preview_fit512_oof_v1_49faacc3"
OUTPUT = ROOT / "results/phoenix-2014t_ISLR/p3_partial32_phase_hazard_two_stage_smoke_v1_49faacc3"
BAGS = REPO / "data/phoenix_2014t/phoenix_iso_center_label_bag2items.train"
KEYPOINTS = REPO / "data/phoenix_2014t/keypoints_hrnet_dark_coco_wholebody_iso.train.pkl"
POSE_ARCHIVE = Path("/tmp/phoenix_p2_train_center_sequences_v1.npz")
SEED = 261024
FOLD_SEED = 261022
FOLDS = 5
TARGETS = ("broad_interior", "interior_entry_hazard4", "interior_entry_hazard8")
LOOKAHEADS = (0, 4, 8)


def imp(name):
    path = ROOT / f"tools/{name}.py"; spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec); sys.modules[spec.name] = module; spec.loader.exec_module(module); return module


FRESHMOD = imp("analyze_phoenix_partial_fresh_terminal_listwise")
BLOCK, CHRON, BUILDER = FRESHMOD.BLOCK, FRESHMOD.CHRON, FRESHMOD.BUILDER
HOGMOD = imp("analyze_phoenix_partial_hog_preview_oof")


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""): digest.update(chunk)
    return digest.hexdigest()


def split_and_fit512():
    cfg = json.loads((FRESH / "resolved_config_preregistered.json").read_text())
    return cfg["split"], cfg["fit_512_scaling_subset"]["sample_ids"]


def config():
    split, fit_ids = split_and_fit512()
    return {
        "experiment": "two-stage causal broad-phase/interior-entry-hazard scheduling smoke",
        "created_before_stage_A_or_predictor_results": True, "cpu_only": True,
        "data": {"fit": "frozen hash512 subset", "fit_samples": 512, "fit_sources": 211,
                 "calibration_samples": 459, "calibration_sources": 50,
                 "fit_ids_sha256": hashlib.sha256("\n".join(fit_ids).encode()).hexdigest(), "split": split,
                 "alignment": str(BAGS), "pose": str(POSE_ARCHIVE), "HOG_fit_cache": str(HOG_ROOT / "hog_preview_fit512_fp16.npz")},
        "protocol": {"block_offsets": [0,1,2,3], "hard_skeleton_offset": 0, "bonus_offsets": [1,2,3],
                     "default_center_offset": 2, "one_bonus_per_complete_block": True, "span": 15,
                     "actual_window_rate": "approximately 50%", "max_gap": 3, "unknown_EOS": True,
                     "target_oracle_tie_order": [2,1,3]},
        "stage_A": {
            "targets": {
                "broad_interior": "candidate center frame lies at normalized nonblank-gloss phase in [0.30,0.70]",
                "interior_entry_hazard4": "currently outside broad interior and first entry occurs in frames t+1..t+4",
                "interior_entry_hazard8": "currently outside broad interior and first entry occurs in frames t+1..t+8",
            },
            "distinction": "interior-entry hazard, not gloss-boundary hazard",
            "candidate_time": "raw frame indexed by sliding-window start; label future is oracle/training only",
            "oracle_policy": "choose center when center target=1; otherwise left then right among target-positive sides; center if all zero",
            "pass_per_target": "fit block reward>0, fit closed-loop reward>0, all five fit-source-fold closed-loop rewards>=0, calibration block reward>0, calibration closed-loop reward>0",
            "failure": "if no target passes, stop before predictor training",
        },
        "stage_B_frozen_before_stage_A": {
            "inputs": "49-d bookkeeping/past prefix + 68-d fixed pose summary + 72-d fixed HOG summary",
            "representation": "pose: 31x17 ending candidate availability+lookahead; HOG: latest 16 frames ending candidate availability+lookahead; each temporal mean,std,last-first,max",
            "lookaheads": {"0": "through candidate start+8", "4": "through start+12; +4-frame algorithmic delay", "8": "through start+16; +8-frame algorithmic delay"},
            "architecture": "shared candidate MLP Linear(189,48)-GELU-Linear(48,32)-GELU-Linear(32,1)",
            "training": "binary BCEWithLogits on all candidates; AdamW lr1e-3 weight_decay1e-4; 20 epochs; batch128; identical architecture/hyperparameters all targets/lookaheads",
            "normalization": "fit within each source-group fold only",
            "folds": "same sha256(261022+NUL+source) mod5 as selective residual",
            "schedule": "one bonus: argmax predicted target probability, center wins exact ties",
            "ranking_metrics": ["candidate PR-AUC", "global recall@top10%", "mean block NDCG on positive blocks"],
            "OOF_gate": "aggregate closed-loop terminal reward>0 and every source-fold reward>=0",
            "calibration_gate": "frozen full-fit model closed-loop terminal reward>0; no model/target changes on calibration",
            "evaluation": "only a later/explicit step may run the already-open evaluation once if both gates pass",
        },
        "forbidden": ["GPU/CUDA", "dev", "test", "evaluation outcomes", "deployment access to alignment labels/reference/full logits/EOS", "git commit"],
    }


def preregister(root):
    if root.exists(): raise FileExistsError(root)
    root.mkdir(parents=True); (root / "resolved_config_preregistered.json").write_text(json.dumps(config(), indent=2)+"\n")
    return {"status":"preregistered", "fit_samples":512, "calibration_samples":459, "targets":list(TARGETS), "lookaheads":list(LOOKAHEADS)}


def loadcfg(root):
    value=json.loads((root/"resolved_config_preregistered.json").read_text())
    if value != config(): raise ValueError("preregistered config changed")
    return value


def source_fold(source):
    return int(hashlib.sha256(f"{FOLD_SEED}\0{source}".encode()).hexdigest()[:16],16)%FOLDS


def alignment_targets():
    with BAGS.open("rb") as handle: bags=pickle.load(handle)
    segments={}
    for records in bags.values():
        base=[x for x in records if int(x.get("aug",-1))==0]
        if len(base)!=1: raise ValueError("bag base record mismatch")
        item=base[0]; segments.setdefault(item["video_file"],[]).append(item)
    with KEYPOINTS.open("rb") as handle: keypoints=pickle.load(handle)
    targets={}
    for name,items in segments.items():
        length=len(keypoints[name]); interior=np.zeros(length,bool)
        for item in items:
            if item["label"]=="<blank>": continue
            start,end=int(item["start"]),int(item["end"])
            if not 0<=start<end<=length: raise ValueError(f"invalid segment {name}")
            denom=max(1,end-start-1)
            phase=np.arange(end-start,dtype=np.float32)/denom
            interior[start:end] |= (phase>=.30)&(phase<=.70)
        values={"broad_interior":interior}
        for horizon,key in ((4,"interior_entry_hazard4"),(8,"interior_entry_hazard8")):
            hazard=np.zeros(length,bool)
            entries=np.flatnonzero(interior & ~np.concatenate(([False],interior[:-1])))
            for entry in entries:
                left=max(0,int(entry)-horizon); hazard[left:int(entry)]=True
            values[key]=hazard
        targets[name]=values
    return targets


def load_rows(fit_ids,cal_ids):
    wanted={"fit":set(fit_ids),"calibration":set(cal_ids)};out={"fit":[],"calibration":[]}
    manifest=json.loads((DATA/"dataset_manifest.json").read_text())
    for info in manifest["workers"]:
        path=DATA/"workers"/f"worker-{info['worker']:02d}-of-{info['workers']:02d}.jsonl.gz"
        if sha(path)!=info["sha256"]:raise ValueError("row hash mismatch")
        with gzip.open(path,"rt",encoding="utf-8") as handle:
            for line in handle:
                row=json.loads(line);part=row["partition"]
                if row["sample_id"] in wanted.get(part,set()):out[part].append(row)
    return out


def target_values(label_map,name,target,candidates):
    value=label_map[name][target]
    return [bool(value[min(max(0,int(c)),len(value)-1)]) for c in candidates]


def target_choice(values):
    if values[1]:return 1
    if values[0]:return 0
    if values[2]:return 2
    return 1


def block_report(rows,label_map,target,folds=False):
    reward=[];regret=[];center_regret=[];sources=[];choices=Counter();positive_blocks=0
    for row in rows:
        values=target_values(label_map,row["sample_id"],target,row["candidate_starts"]);choice=target_choice(values);errors=row["label"]["terminal_errors"]
        reward.append(errors[1]-errors[choice]);regret.append(errors[choice]-min(errors));center_regret.append(errors[1]-min(errors));sources.append(row["source_video_id"]);choices[choice]+=1;positive_blocks+=int(any(values))
    result={"blocks":len(rows),"target_positive_blocks":positive_blocks,"choice_counts":{str(k):v for k,v in sorted(choices.items())},"cumulative_terminal_reward":int(sum(reward)),"total_terminal_regret":int(sum(regret)),"fixed_center_regret":int(sum(center_regret))}
    if folds:
        result["fold_reward"]={str(f):int(sum(r for r,s in zip(reward,sources) if source_fold(s)==f)) for f in range(FOLDS)}
    return result


def scheduled_sample(name,result,logits,label_map,target,vocab,blank):
    p=BUILDER.ORACLE.softmax_rows(np.asarray(logits));ref=BUILDER.REPLAY.clean_phoenix_2014_trans(result["gls_ref"]);selected=[];choices=Counter()
    for start in range(0,len(p),4):
        selected.append(start)
        if start+3>=len(p):continue
        candidates=[start+1,start+2,start+3];values=target_values(label_map,name,target,candidates);choice=target_choice(values);selected.append(candidates[choice]);choices[choice]+=1
    return {"sample_id":name,"selected":selected,"counts":BLOCK.decode_counts(p,selected,ref,vocab,blank),"dense_windows":len(p),"coverage":CHRON.coverage_metrics(selected,len(p)),"choice_counts":dict(choices)}


def closed_loop(ids,label_map,targets):
    wanted=set(ids);vocab=json.loads(CHRON.DEFAULT_VOCAB.read_text());blank=vocab.index("<blank>");out={t:[] for t in targets};uniform=[];dense=CHRON.DEFAULT_DENSE_ROOT;indices,n=CHRON.completed_shard_indices(dense)
    for shard in indices:
        results,logits,_=BUILDER.BUILDER.validate_dense_shard(dense,shard,n,verify_hashes=False)
        for name in results:
            if name not in wanted:continue
            base=BLOCK.oracle_sample(name,results[name],logits[name],"train",vocab,blank,collect_rows=False);uniform.append({"sample_id":name,**base["uniform"],"dense_windows":base["dense_windows"]})
            for target in targets:out[target].append(scheduled_sample(name,results[name],logits[name],label_map,target,vocab,blank))
    if len(uniform)!=len(wanted):raise ValueError("closed-loop coverage mismatch")
    by={x["sample_id"]:x for x in uniform};reports={}
    for target,rows in out.items():
        ordered=[by[x["sample_id"]] for x in rows];summary=BLOCK.summarize(rows,ordered);sample_reward={x["sample_id"]:by[x["sample_id"]]["counts"]["error"]-x["counts"]["error"] for x in rows}
        reports[target]={"summary":summary,"cumulative_error_reward":int(sum(sample_reward.values())),"sample_reward":sample_reward}
    return reports


def stage_a(root):
    cfg=loadcfg(root);(root/"stage_a_started.marker").write_text(datetime.now(timezone.utc).isoformat()+"\n");fit_ids=cfg["data"]["split"]["partitions"]["fit"]["sample_ids"]
    frozen=set(split_and_fit512()[1]);fit_ids=[x for x in fit_ids if x in frozen];cal_ids=cfg["data"]["split"]["partitions"]["calibration"]["sample_ids"]
    labels=alignment_targets();rows=load_rows(fit_ids,cal_ids);fit_closed=closed_loop(fit_ids,labels,TARGETS);cal_closed=closed_loop(cal_ids,labels,TARGETS);reports={};passing=[]
    for target in TARGETS:
        fit_block=block_report(rows["fit"],labels,target,True);cal_block=block_report(rows["calibration"],labels,target,False);sample_reward=fit_closed[target]["sample_reward"]
        fold_reward={str(f):int(sum(value for name,value in sample_reward.items() if source_fold(BUILDER.source_video(name))==f)) for f in range(FOLDS)}
        checks={"fit_block_reward_gt_0":fit_block["cumulative_terminal_reward"]>0,"fit_closed_reward_gt_0":fit_closed[target]["cumulative_error_reward"]>0,
                "all_fit_fold_closed_rewards_ge_0":all(v>=0 for v in fold_reward.values()),"cal_block_reward_gt_0":cal_block["cumulative_terminal_reward"]>0,
                "cal_closed_reward_gt_0":cal_closed[target]["cumulative_error_reward"]>0}
        passed=all(checks.values());passing.append(target) if passed else None
        reports[target]={"fit_block":fit_block,"fit_closed_loop":{k:v for k,v in fit_closed[target].items() if k!="sample_reward"}|{"fold_reward":fold_reward},"calibration_block":cal_block,
                         "calibration_closed_loop":{k:v for k,v in cal_closed[target].items() if k!="sample_reward"},"gate":{"passed":passed,"checks":checks}}
    result={"stage":"A target alignment/oracle audit","gpu_used":False,"dev_used":False,"test_used":False,"evaluation_outcomes_read":False,"targets":reports,"gate":{"passed":bool(passing),"passing_targets":passing},
            "decision":"proceed to preregistered predictor stage B" if passing else "stop: no phase/hazard target aligned with terminal utility"}
    (root/"stage_a_metrics.json").write_text(json.dumps(result,indent=2)+"\n")
    if not passing:
        (root/"metrics.json").write_text(json.dumps({**result,"stage_B":None},indent=2)+"\n");(root/"manifest.json").write_text(json.dumps({"status":"stopped_at_stage_A","created_utc":datetime.now(timezone.utc).isoformat(),"stage_a_sha256":sha(root/"stage_a_metrics.json"),"calibration_used_for_target_alignment":True,"predictor_trained":False,"evaluation_outcomes_read":False},indent=2)+"\n")
    return result


def main():
    parser=argparse.ArgumentParser();parser.add_argument("mode",choices=("preregister","stage-a"));parser.add_argument("--output-root",type=Path,default=OUTPUT);args=parser.parse_args();os.environ["CUDA_VISIBLE_DEVICES"]="";root=args.output_root.resolve();result=preregister(root) if args.mode=="preregister" else stage_a(root);print(json.dumps(result,indent=2))


if __name__=="__main__":main()
