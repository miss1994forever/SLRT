#!/usr/bin/env python3
"""Fit512 source-group OOF smoke for self-supervised cheap-feature surprise."""
import argparse
import gzip
import hashlib
import importlib.util
import json
import os
import random
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT=Path(__file__).resolve().parents[1]
FRESH=ROOT/"results/phoenix-2014t_ISLR/p3_partial32_fresh_terminal_listwise_exploratory_v1_49faacc3"
DATA=FRESH.with_name(FRESH.name+"_dataset")
HOG_ROOT=ROOT/"results/phoenix-2014t_ISLR/p3_partial32_hog_preview_fit512_oof_v1_49faacc3"
POSE_ARCHIVE=Path("/tmp/phoenix_p2_train_center_sequences_v1.npz")
OUTPUT=ROOT/"results/phoenix-2014t_ISLR/p3_partial32_self_supervised_surprise_fit512_oof_v1_49faacc3"
SEED=261027
FOLD_SEED=261022
FOLDS=5
EPOCHS=15
HIDDEN=32
MOTION_INDICES=(9,10,11,12,13,14)


def imp(name):
    path=ROOT/f"tools/{name}.py";spec=importlib.util.spec_from_file_location(name,path);module=importlib.util.module_from_spec(spec);sys.modules[spec.name]=module;spec.loader.exec_module(module);return module


FRESHMOD=imp("analyze_phoenix_partial_fresh_terminal_listwise")
HOGMOD=imp("analyze_phoenix_partial_hog_preview_oof")
PREVIEW=FRESHMOD.PREVIEW


def sha(path):
    digest=hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda:handle.read(1024*1024),b""):digest.update(chunk)
    return digest.hexdigest()


def frozen_ids():
    cfg=json.loads((FRESH/"resolved_config_preregistered.json").read_text());ids=cfg["fit_512_scaling_subset"]["sample_ids"]
    if len(ids)!=512:raise ValueError("frozen subset mismatch")
    return ids


def source_fold(source):
    return int(hashlib.sha256(f"{FOLD_SEED}\0{source}".encode()).hexdigest()[:16],16)%FOLDS


def config():
    ids=frozen_ids();hog=json.loads((HOG_ROOT/"cache_manifest.json").read_text())
    return {"experiment":"fit-only self-supervised causal surprise/change-point source-group OOF","created_before_OOF_terminal_evaluation":True,"cpu_only":True,
      "scope":{"fit_samples":512,"fit_sources":211,"sample_ids_sha256":hashlib.sha256("\n".join(ids).encode()).hexdigest(),"terminal_rows":str(DATA),"pose":str(POSE_ARCHIVE),"HOG":str(HOG_ROOT/"hog_preview_fit512_fp16.npz"),"HOG_sha256":hog["sha256"]},
      "cheap_sequence":{"per_candidate_width":89,"pose_hand_motion":"17-d audited feature vector at candidate availability (raw frame start+8)","HOG":"frozen 72-d two-hand HOG temporal summary over start-7..start+8","time_index":"dense candidate starts from 0 through the last complete terminal block; no EOS queried","normalization":"mean/std from training sources inside each fold only"},
      "self_supervision":{"model":"GRU(input89,hidden32) + Linear(32,89)","task":"one-step prediction: state after x[t-1] predicts x[t]","loss":"Huber(delta=1) over standardized cheap features","epochs":EPOCHS,"optimizer":"AdamW(lr=1e-3,weight_decay=1e-4)","batching":"one source-safe video sequence per optimizer step","terminal_labels_used_for_training":False,"uncertainty":"none"},
      "surprise":{"definition":"mean per-dimension Huber residual of one-step prediction","standardization":"residual mean/std estimated on training-source sequences after self-supervised training","cold_start":"t=0 score zero; never a bonus candidate"},
      "policy":{"blocks":"offset0 skeleton + exactly one of offset1/2/3 bonus","decision":"after offset3 cheap feature is available","bounded_lookahead_candidate_arrivals":2,"choice":"argmax score; center wins exact ties","budget":"same one bonus per complete block, maxgap3","unknown_EOS":True},
      "controls":{"fixed_center":"always offset2","motion_magnitude":"L2 of fold-standardized pose_motion,left/right_global_motion,left/right_shape_change,interhand_distance_change","raw_feature_difference":"L2 of fold-standardized x[t]-x[t-1] over all 89 dimensions"},
      "OOF":{"folds":5,"assignment":"sha256(261022+NUL+source_video_id) mod5","terminal_use":"held fold schedule evaluation only after predictions frozen","reports":["positive-utility PR-AUC","nonneutral-informativeness PR-AUC","positive/informative enrichment","fold reward","positive/harmful/neutral choices","reward/regret"],"pass":"learned aggregate reward>0, every fold>=0, >=20 nonneutral side choices, reward>motion and raw-novelty, regret<motion and raw-novelty"},
      "cost":"report batched causal GRU CPU milliseconds per candidate; cached pose/HOG extraction cost excluded and must be added in a real stream",
      "forbidden":["GPU/CUDA","dev","test","calibration/evaluation outcomes","terminal/full-logit/reference/future/EOS training inputs","git commit"]}


def preregister(root):
    if root.exists():raise FileExistsError(root)
    root.mkdir(parents=True);value=config();(root/"resolved_config_preregistered.json").write_text(json.dumps(value,indent=2)+"\n");return {"status":"preregistered","samples":512,"sources":211,"feature_width":89,"epochs":EPOCHS,"folds":FOLDS}


def loadcfg(root):
    value=json.loads((root/"resolved_config_preregistered.json").read_text())
    if value!=config():raise ValueError("preregistered configuration changed")
    return value


def load_rows():
    wanted=set(frozen_ids());manifest=json.loads((DATA/"dataset_manifest.json").read_text());rows=[]
    for info in manifest["workers"]:
        path=DATA/"workers"/f"worker-{info['worker']:02d}-of-{info['workers']:02d}.jsonl.gz"
        if sha(path)!=info["sha256"]:raise ValueError("row hash mismatch")
        with gzip.open(path,"rt",encoding="utf-8") as handle:
            for line in handle:
                row=json.loads(line)
                if row["partition"]=="fit" and row["sample_id"] in wanted:rows.append(row)
    rows.sort(key=lambda x:(x["sample_id"],x["block_start"]));return rows


def build_data():
    rows=load_rows();pose=PREVIEW.PreviewArchive(POSE_ARCHIVE);hog=HOGMOD.HogArchive(HOG_ROOT);grouped={}
    for row in rows:grouped.setdefault(row["sample_id"],[]).append(row)
    features=[];slices={};offset=0;source_by_sample={};row_indices=[]
    for name in frozen_ids():
        sample_rows=grouped[name];source_by_sample[name]=sample_rows[0]["source_video_id"];length=max(max(r["candidate_starts"]) for r in sample_rows)+1;sequence=[]
        for start in range(length):
            pose_now=np.asarray(pose.history(name,start)[-1],np.float32);hog_now=hog.summary(name,start);sequence.append(np.concatenate([pose_now,hog_now]))
        sequence=np.asarray(sequence,np.float32);features.append(sequence);slices[name]=slice(offset,offset+length);offset+=length
    features=np.concatenate(features);errors=[];sources=[];candidates=[]
    for row in rows:
        sl=slices[row["sample_id"]];candidates.append([sl.start+int(x) for x in row["candidate_starts"]]);errors.append(row["label"]["terminal_errors"]);sources.append(row["source_video_id"])
    errors=np.asarray(errors,np.float32);return {"features":features,"slices":slices,"source_by_sample":source_by_sample,"errors":errors,"advantage":errors[:,1:2]-errors,
      "informative":np.any(errors!=errors[:,1:2],axis=1),"source":np.asarray(sources),"candidates":np.asarray(candidates,np.int64),"rows":rows}


class Predictor(nn.Module):
    def __init__(self):super().__init__();self.gru=nn.GRU(89,HIDDEN,batch_first=True);self.head=nn.Linear(HIDDEN,89)
    def forward(self,value):
        output,_=self.gru(value);return self.head(output)


def fold_stats(data,fold):
    pieces=[data["features"][sl] for name,sl in data["slices"].items() if source_fold(data["source_by_sample"][name])!=fold];all_values=np.concatenate(pieces);mean=all_values.mean(0,dtype=np.float64).astype(np.float32);scale=all_values.std(0,dtype=np.float64).astype(np.float32);scale[scale<1e-6]=1;return mean,scale


def train_predictor(data,fold,stats):
    names=[x for x in frozen_ids() if source_fold(data["source_by_sample"][x])!=fold];torch.manual_seed(SEED+fold);model=Predictor();optimizer=torch.optim.AdamW(model.parameters(),lr=1e-3,weight_decay=1e-4);generator=np.random.default_rng(SEED+100+fold);history=[]
    for _ in range(EPOCHS):
        order=generator.permutation(len(names));total=0.;count=0
        for pos in order:
            sequence=(data["features"][data["slices"][names[int(pos)]]]-stats[0])/stats[1]
            if len(sequence)<2:continue
            value=torch.from_numpy(sequence[:-1][None]).float();target=torch.from_numpy(sequence[1:][None]).float();pred=model(value);loss=F.smooth_l1_loss(pred,target);optimizer.zero_grad();loss.backward();optimizer.step();total+=float(loss)*(len(sequence)-1);count+=len(sequence)-1
        history.append(total/count)
    model.eval();return model,history


def sequence_surprise(model,sequence):
    scores=np.zeros(len(sequence),np.float32)
    if len(sequence)<2:return scores
    with torch.no_grad():pred=model(torch.from_numpy(sequence[:-1][None]).float())[0];target=torch.from_numpy(sequence[1:]).float();residual=F.smooth_l1_loss(pred,target,reduction="none").mean(1).numpy();scores[1:]=residual
    return scores


def train_residual_stats(model,data,fold,stats):
    values=[]
    for name,sl in data["slices"].items():
        if source_fold(data["source_by_sample"][name])!=fold:values.append(sequence_surprise(model,(data["features"][sl]-stats[0])/stats[1])[1:])
    values=np.concatenate(values);return float(values.mean()),float(max(values.std(),1e-6))


def held_scores(model,data,fold,stats,residual_stats):
    learned=np.zeros(len(data["features"]),np.float32);motion=np.zeros_like(learned);novelty=np.zeros_like(learned);count=0;tick=time.perf_counter()
    for name,sl in data["slices"].items():
        if source_fold(data["source_by_sample"][name])!=fold:continue
        sequence=(data["features"][sl]-stats[0])/stats[1];raw=sequence_surprise(model,sequence);learned[sl]=(raw-residual_stats[0])/residual_stats[1];motion[sl]=np.linalg.norm(sequence[:,MOTION_INDICES],axis=1);novelty_sequence=np.zeros(len(sequence),np.float32);novelty_sequence[1:]=np.linalg.norm(sequence[1:]-sequence[:-1],axis=1);novelty[sl]=novelty_sequence;count+=len(sequence)
    elapsed=time.perf_counter()-tick;held_blocks=np.asarray([source_fold(x)==fold for x in data["source"]]);idx=data["candidates"][held_blocks]
    return held_blocks,{"learned":learned[idx],"motion":motion[idx],"raw_novelty":novelty[idx]},1000*elapsed/count


def choose(scores):
    # np.argmax gives offset1 on exact ties; explicitly prefer center, then left, then right.
    center=scores[:,1];left=scores[:,0];right=scores[:,2];choice=np.ones(len(scores),np.int64);choice[left>center]=0;best=np.maximum(left,center);choice[right>best]=2;return choice


def policy_metrics(data,scores,folds):
    picked=choose(scores);errors=data["errors"];selected=errors[np.arange(len(errors)),picked];center=errors[:,1];minimum=errors.min(1);reward=center-selected;side=picked!=1;positive=(reward>0)&side;harmful=(reward<0)&side;neutral=(reward==0)&side
    return {"blocks":len(errors),"choice_counts":{str(k):int(v) for k,v in zip(*np.unique(picked,return_counts=True))},"side_choices":int(side.sum()),"positive_choices":int(positive.sum()),"harmful_choices":int(harmful.sum()),"neutral_side_choices":int(neutral.sum()),"nonneutral_side_choices":int((positive|harmful).sum()),"cumulative_terminal_reward":int(reward.sum()),"total_terminal_regret":int(np.sum(selected-minimum)),"fixed_center_regret":int(np.sum(center-minimum)),"fold_reward":{str(f):int(reward[folds==f].sum()) for f in range(FOLDS)}}


def average_precision(label,score):
    label=np.asarray(label,bool);score=np.asarray(score);positives=int(label.sum())
    if not positives:return None
    order=np.argsort(-score,kind="stable");ranked=label[order];precision=np.cumsum(ranked)/np.arange(1,len(ranked)+1);return float(precision[ranked].sum()/positives)


def ranking_metrics(data,scores):
    advantage=data["advantage"];positive=advantage>0;informative=advantage!=0;flat=scores.reshape(-1);choice=choose(scores);picked_adv=advantage[np.arange(len(advantage)),choice];side=choice!=1
    base_positive=float(positive.mean());base_informative=float(informative.mean());selected_positive=float(np.mean(picked_adv[side]>0)) if side.any() else 0.;selected_informative=float(np.mean(picked_adv[side]!=0)) if side.any() else 0.
    return {"positive_utility_PR_AUC":average_precision(positive.reshape(-1),flat),"nonneutral_informativeness_PR_AUC":average_precision(informative.reshape(-1),flat),"candidate_positive_rate":base_positive,"selected_side_positive_rate":selected_positive,"positive_enrichment":selected_positive/base_positive if base_positive else None,"candidate_informative_rate":base_informative,"selected_side_informative_rate":selected_informative,"informativeness_enrichment":selected_informative/base_informative if base_informative else None}


def run_oof(root):
    loadcfg(root);(root/"oof_started.marker").write_text(datetime.now(timezone.utc).isoformat()+"\n");os.environ["CUDA_VISIBLE_DEVICES"]="";random.seed(SEED);np.random.seed(SEED);torch.manual_seed(SEED);torch.set_num_threads(min(8,os.cpu_count() or 1));torch.use_deterministic_algorithms(True);started=time.perf_counter();data=build_data();folds=np.asarray([source_fold(x) for x in data["source"]],np.int64);score_map={k:np.full((len(data["errors"]),3),np.nan,np.float32) for k in ("learned","motion","raw_novelty")};fold_reports={}
    for fold in range(FOLDS):
        stats=fold_stats(data,fold);model,history=train_predictor(data,fold,stats);residual_stats=train_residual_stats(model,data,fold,stats);mask,scores,cost=held_scores(model,data,fold,stats,residual_stats)
        for key in score_map:score_map[key][mask]=scores[key]
        fold_reports[str(fold)]={"train_sources":len({x for x in data["source_by_sample"].values() if source_fold(x)!=fold}),"heldout_sources":len({x for x in data["source_by_sample"].values() if source_fold(x)==fold}),"train_sequences":sum(source_fold(x)!=fold for x in data["source_by_sample"].values()),"heldout_sequences":sum(source_fold(x)==fold for x in data["source_by_sample"].values()),"residual_train_mean":residual_stats[0],"residual_train_std":residual_stats[1],"controller_CPU_ms_per_candidate_including_sequence_loop":cost,"training_history":history};print(json.dumps({"fold_complete":fold,"last_loss":history[-1],"seconds":time.perf_counter()-started}),flush=True)
    if any(np.isnan(x).any() for x in score_map.values()):raise RuntimeError("incomplete OOF")
    reports={key:{"policy":policy_metrics(data,scores,folds),"ranking":ranking_metrics(data,scores)} for key,scores in score_map.items()};fixed={"policy":policy_metrics(data,np.tile(np.asarray([[0.,1.,0.]],np.float32),(len(data["errors"]),1)),folds)};learned=reports["learned"]["policy"];motion=reports["motion"]["policy"];raw=reports["raw_novelty"]["policy"]
    checks={"aggregate_reward_gt_0":learned["cumulative_terminal_reward"]>0,"every_fold_reward_ge_0":all(x>=0 for x in learned["fold_reward"].values()),"at_least_20_nonneutral_side_choices":learned["nonneutral_side_choices"]>=20,"reward_gt_motion":learned["cumulative_terminal_reward"]>motion["cumulative_terminal_reward"],"reward_gt_raw_novelty":learned["cumulative_terminal_reward"]>raw["cumulative_terminal_reward"],"regret_lt_motion":learned["total_terminal_regret"]<motion["total_terminal_regret"],"regret_lt_raw_novelty":learned["total_terminal_regret"]<raw["total_terminal_regret"]};passed=all(checks.values())
    result={"scope":"frozen512 fit-only self-supervised surprise source-group OOF","gpu_used":False,"dev_used":False,"test_used":False,"calibration_or_evaluation_outcomes_read":False,"data":{"samples":len(data["slices"]),"sources":len(set(data["source"])),"blocks":len(data["errors"]),"informative_blocks":int(data["informative"].sum()),"candidate_sequence_steps":len(data["features"]),"feature_assembly_and_total_seconds":time.perf_counter()-started},"fixed_center":fixed,"methods":reports,"fold_reports":fold_reports,"gate":{"passed":passed,"checks":checks},"decision":"eligible for separate calibration" if passed else "stop self-supervised surprise; do not read calibration"}
    (root/"metrics.json").write_text(json.dumps(result,indent=2)+"\n");(root/"manifest.json").write_text(json.dumps({"status":"fit_oof_complete","created_utc":datetime.now(timezone.utc).isoformat(),"gate_passed":passed,"calibration_outcomes_read":False,"evaluation_outcomes_read":False,"metrics_sha256":sha(root/"metrics.json")},indent=2)+"\n");return result


def main():
    parser=argparse.ArgumentParser();parser.add_argument("mode",choices=("preregister","oof"));parser.add_argument("--output-root",type=Path,default=OUTPUT);args=parser.parse_args();os.environ["CUDA_VISIBLE_DEVICES"]="";root=args.output_root.resolve();result=preregister(root) if args.mode=="preregister" else run_oof(root);print(json.dumps(result,indent=2))


if __name__=="__main__":main()
