#!/usr/bin/env python3
"""Fit512 predictor-complementarity audit with two frozen AND/veto rules."""
import argparse
import hashlib
import importlib.util
import json
import math
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
OUTPUT=ROOT/"results/phoenix-2014t_ISLR/p3_partial32_predictor_complementarity_fit512_oof_v1_49faacc3"
SEED=261028
FOLD_SEED=261022
FOLDS=5
HAZARD_EPOCHS=20
HISTORY_EPOCHS=20
BATCH=128


def imp(name):
    path=ROOT/f"tools/{name}.py";spec=importlib.util.spec_from_file_location(name,path);module=importlib.util.module_from_spec(spec);sys.modules[spec.name]=module;spec.loader.exec_module(module);return module


SURPRISE=imp("analyze_phoenix_partial_self_supervised_surprise")
HISTORYMOD=imp("analyze_phoenix_partial_full_decoder_history")
PHASE=imp("analyze_phoenix_partial_phase_hazard_smoke")
CHRON,BUILDER=SURPRISE.FRESHMOD.CHRON,SURPRISE.FRESHMOD.BUILDER


def sha(path):
    digest=hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda:handle.read(1024*1024),b""):digest.update(chunk)
    return digest.hexdigest()


def source_fold(source):
    return int(hashlib.sha256(f"{FOLD_SEED}\0{source}".encode()).hexdigest()[:16],16)%FOLDS


def config():
    ids=SURPRISE.frozen_ids()
    return {"experiment":"fit512 predictor complementarity source-group OOF audit","created_before_combination_terminal_results":True,"cpu_only":True,
      "scope":{"samples":512,"sources":211,"sample_ids_sha256":hashlib.sha256("\n".join(ids).encode()).hexdigest(),"folds":5,"assignment":"sha256(261022+NUL+source_video_id) mod5","calibration_evaluation":"never read"},
      "component_reconstruction":{"surprise":"exact preregistered 89-d causal GRU one-step predictor rerun on same folds; terminal-free training","hazard4":"entry into broad interior phase[.30,.70] within next4 raw frames; 98-d current candidate input = bookkeeping9 + pose17 + HOG72; MLP98->48->32->1, BCE","hazard_threshold":"within each fold, threshold is the training prediction quantile whose positive count matches training hazard-label prevalence; terminal labels forbidden","history":"same frozen fixed-center behavior history, last16 paid logits PCA16+meta14, GRU32 residual; rerun only common512 blocks/sources with terminal residual training"},
      "components":{"surprise_choice":"argmax offsets1/2/3, center wins exact ties","hazard_choice":"argmax predicted hazard offsets1/2/3, center wins ties","history_choice":"highest offset1/3 residual if positive, otherwise center"},
      "combination_A":{"name":"hazard4 gate AND learned-surprise offset","rule":"gate opens iff max candidate hazard probability reaches the fold's hazard-only threshold; if open use surprise choice, otherwise center","forbidden":"no terminal threshold tuning, OR, weights, or offset replacement by hazard"},
      "combination_B":{"name":"full-history residual veto AND learned-surprise offset","rule":"if surprise selects offset1/3, allow only when history residual for that same side is >0; otherwise center","forbidden":"no terminal threshold tuning, OR, weights, or alternative side substitution"},
      "pair_oracle":{"choices":"minimum terminal error among center, component-1 choice, component-2 choice","deployable":False,"stopping":"if its reward is not strictly above the best single component, stop that combination before deployment-rule scoring"},
      "reports":["component choices and agreement/disagreement","positive/harmful/neutral override overlap","signed reward Pearson correlation","pair-oracle reward/regret","deployment five-fold reward/regret and deltas"],
      "pass":"deployment aggregate reward>0, every fold>=0, >=20 side overrides, reward strictly above both components, regret strictly below both components",
      "distribution_warning":"history comes from frozen fixed-center behavior; any pass still requires on-policy closed-loop validation",
      "forbidden":["GPU/CUDA","dev","test","calibration/evaluation outcomes","deployment future/reference/EOS/unexecuted logits","fusion network","weight/rule search","git commit"]}


def preregister(root):
    if root.exists():raise FileExistsError(root)
    root.mkdir(parents=True);value=config();(root/"resolved_config_preregistered.json").write_text(json.dumps(value,indent=2)+"\n");return {"status":"preregistered","samples":512,"sources":211,"combinations":["A_hazard_AND_surprise","B_history_veto_surprise"]}


def loadcfg(root):
    value=json.loads((root/"resolved_config_preregistered.json").read_text())
    if value!=config():raise ValueError("preregistered config changed")
    return value


class HazardMLP(nn.Module):
    def __init__(self):super().__init__();self.net=nn.Sequential(nn.Linear(98,48),nn.GELU(),nn.Linear(48,32),nn.GELU(),nn.Linear(32,1))
    def forward(self,value):
        b,c,w=value.shape;return self.net(value.reshape(b*c,w)).reshape(b,c)


def hazard_arrays(data):
    bookkeeping=np.asarray([[f["bookkeeping"] for f in row["features"]] for row in data["rows"]],np.float32);candidate=data["features"][data["candidates"]];value=np.concatenate([bookkeeping,candidate],2)
    label_map=PHASE.alignment_targets();labels=[]
    for row in data["rows"]:labels.append(PHASE.target_values(label_map,row["sample_id"],"interior_entry_hazard4",row["candidate_starts"]))
    return value,np.asarray(labels,np.float32)


def train_hazard(value,labels,train,seed):
    selected=value[train];mean=selected.mean((0,1),dtype=np.float64).astype(np.float32);scale=selected.std((0,1),dtype=np.float64).astype(np.float32);scale[scale<1e-6]=1;normalized=(value-mean)/scale
    torch.manual_seed(seed);model=HazardMLP();positive=float(labels[train].sum());negative=float(labels[train].size-positive);pos_weight=torch.tensor(negative/max(1.,positive));optimizer=torch.optim.AdamW(model.parameters(),lr=1e-3,weight_decay=1e-4);idx=np.flatnonzero(train);generator=torch.Generator().manual_seed(seed);history=[]
    for _ in range(HAZARD_EPOCHS):
        order=idx[torch.randperm(len(idx),generator=generator).numpy()];total=0.
        for left in range(0,len(order),BATCH):
            part=order[left:left+BATCH];logit=model(torch.from_numpy(normalized[part]).float());target=torch.from_numpy(labels[part]).float();loss=F.binary_cross_entropy_with_logits(logit,target,pos_weight=pos_weight);optimizer.zero_grad();loss.backward();optimizer.step();total+=float(loss)*len(part)
        history.append(total/len(order))
    model.eval()
    with torch.no_grad():train_prob=torch.sigmoid(model(torch.from_numpy(normalized[train]).float())).numpy();all_prob=torch.sigmoid(model(torch.from_numpy(normalized).float())).numpy()
    k=max(1,int(labels[train].sum()));ordered=np.sort(train_prob.reshape(-1))[::-1];threshold=float(ordered[min(k-1,len(ordered)-1)])
    return all_prob,threshold,history,{"train_hazard_positives":int(labels[train].sum()),"train_candidates":int(labels[train].size),"threshold_rule":"top-K where K equals training hazard positives"}


def build_common_history(data):
    wanted=set(data["slices"]);dense={};indices,workers=CHRON.completed_shard_indices(CHRON.DEFAULT_DENSE_ROOT)
    for shard in indices:
        results,logits,_=BUILDER.BUILDER.validate_dense_shard(CHRON.DEFAULT_DENSE_ROOT,shard,workers,verify_hashes=False)
        for name in results:
            if name in wanted:dense[name]=logits[name]
    vocab=json.loads(CHRON.DEFAULT_VOCAB.read_text());blank=vocab.index("<blank>");parts=[];meta_parts=[];exec_folds=[];mapping={};offset=0;sample_source={row["sample_id"]:row["source_video_id"] for row in data["rows"]}
    for name in sorted(wanted):
        starts,values,meta=HISTORYMOD.execution_trace(dense[name],blank);count=len(starts);mapping[name]=(starts,np.arange(offset,offset+count,dtype=np.int32));offset+=count;parts.append(values);meta_parts.append(meta);exec_folds.extend([source_fold(sample_source[name])]*count)
    indices_array=np.full((len(data["rows"]),HISTORYMOD.HISTORY),-1,np.int32);base=[]
    for i,row in enumerate(data["rows"]):
        base.append(np.asarray([x["bookkeeping"]+x["prefix"] for x in row["features"]],np.float32));starts,global_idx=mapping[row["sample_id"]];available=global_idx[starts<=int(row["block_start"])][-HISTORYMOD.HISTORY:];indices_array[i,:len(available)]=available
    return {"base":np.asarray(base,np.float32),"errors":data["errors"],"advantage":data["errors"][:,1:2]-data["errors"][:,[0,2]],"informative":data["informative"],"source":data["source"],"sample":np.asarray([x["sample_id"] for x in data["rows"]]),"history_indices":indices_array,"lengths":np.sum(indices_array>=0,1).astype(np.int64),"executed_logits":np.concatenate(parts),"executed_meta":np.concatenate(meta_parts),"exec_folds":np.asarray(exec_folds,np.int64)}


def train_surprise_fold(data,fold):
    stats=SURPRISE.fold_stats(data,fold);model,history=SURPRISE.train_predictor(data,fold,stats);residual=SURPRISE.train_residual_stats(model,data,fold,stats);mask,scores,cost=SURPRISE.held_scores(model,data,fold,stats,residual);return mask,scores["learned"],history,cost


def train_history_fold(data,fold):
    folds=np.asarray([source_fold(x) for x in data["source"]]);train,held=folds!=fold,folds==fold;stats=HISTORYMOD.fit_stats(data,train,fold);base,history,projection_seconds=HISTORYMOD.fold_arrays(data,stats);torch.manual_seed(SEED+fold);model=HISTORYMOD.HistoryGRU();old_epochs=HISTORYMOD.EPOCHS;HISTORYMOD.EPOCHS=HISTORY_EPOCHS
    try:losses=HISTORYMOD.train_model(model,base,history,data["lengths"],data,train,SEED+300+fold)
    finally:HISTORYMOD.EPOCHS=old_epochs
    idx,scores,inference=HISTORYMOD.predict(model,base,history,data["lengths"],held);return idx,scores,losses,{"PCA_explained":stats["pca_explained"],"projection_ms_per_execution":1000*projection_seconds/len(data["executed_logits"]),"inference_ms_per_block":1000*inference/held.sum()}


def choose_three(scores):return SURPRISE.choose(scores)


def history_choice(scores):
    side=np.argmax(scores,1);confidence=scores[np.arange(len(scores)),side];return np.where(confidence>0,np.where(side==0,0,2),1)


def choice_metrics(data,choice,folds):
    errors=data["errors"];picked=errors[np.arange(len(errors)),choice];center=errors[:,1];minimum=errors.min(1);reward=center-picked;side=choice!=1
    return {"choice_counts":{str(k):int(v) for k,v in zip(*np.unique(choice,return_counts=True))},"side_overrides":int(side.sum()),"positive_overrides":int(np.sum(side&(reward>0))),"harmful_overrides":int(np.sum(side&(reward<0))),"neutral_overrides":int(np.sum(side&(reward==0))),"cumulative_terminal_reward":int(reward.sum()),"total_terminal_regret":int(np.sum(picked-minimum)),"fixed_center_regret":int(np.sum(center-minimum)),"fold_reward":{str(f):int(reward[folds==f].sum()) for f in range(FOLDS)},"reward_array":reward}


def clean(metrics):return {k:v for k,v in metrics.items() if k!="reward_array"}


def component_audit(first,second):
    r1,r2=first["reward_array"],second["reward_array"];side1=first["choice"]!=1;side2=second["choice"]!=1;both=side1&side2
    correlation=float(np.corrcoef(r1,r2)[0,1]) if r1.std()>0 and r2.std()>0 else None
    return {"exact_choice_agreement":float(np.mean(first["choice"]==second["choice"])),"disagreement_blocks":int(np.sum(first["choice"]!=second["choice"])),"both_side_overlap":int(both.sum()),"same_side_overlap":int(np.sum(both&(first["choice"]==second["choice"]))),"opposite_side_overlap":int(np.sum(both&(first["choice"]!=second["choice"]))),"positive_override_overlap":int(np.sum((r1>0)&side1&(r2>0)&side2)),"harmful_override_overlap":int(np.sum((r1<0)&side1&(r2<0)&side2)),"neutral_override_overlap":int(np.sum((r1==0)&side1&(r2==0)&side2)),"signed_reward_Pearson":correlation}


def pair_oracle(data,choice1,choice2,folds):
    errors=data["errors"];choice=np.ones(len(errors),np.int64)
    for i in range(len(errors)):
        candidates=[1,int(choice1[i]),int(choice2[i])];choice[i]=min(candidates,key=lambda x:(errors[i,x],0 if x==1 else 1))
    return choice,choice_metrics(data,choice,folds)


def combo_report(data,folds,name,c1,c2,combined):
    m1=choice_metrics(data,c1,folds);m2=choice_metrics(data,c2,folds);oracle_choice,oracle=pair_oracle(data,c1,c2,folds);upper_beats=oracle["cumulative_terminal_reward"]>max(m1["cumulative_terminal_reward"],m2["cumulative_terminal_reward"])
    result={"name":name,"component_1":clean(m1),"component_2":clean(m2),"complementarity":component_audit({"choice":c1,**m1},{"choice":c2,**m2}),"pair_oracle_non_deployable":clean(oracle),"pair_oracle_beats_best_component":upper_beats}
    if not upper_beats:result.update({"deployment":None,"gate":{"passed":False,"reason":"pair oracle does not beat best component"}});return result
    deployed=choice_metrics(data,combined,folds);checks={"aggregate_reward_gt_0":deployed["cumulative_terminal_reward"]>0,"every_fold_reward_ge_0":all(v>=0 for v in deployed["fold_reward"].values()),"at_least_20_overrides":deployed["side_overrides"]>=20,"reward_gt_both_components":deployed["cumulative_terminal_reward"]>max(m1["cumulative_terminal_reward"],m2["cumulative_terminal_reward"]),"regret_lt_both_components":deployed["total_terminal_regret"]<min(m1["total_terminal_regret"],m2["total_terminal_regret"])}
    result.update({"deployment":clean(deployed),"relative":{"reward_minus_component1":deployed["cumulative_terminal_reward"]-m1["cumulative_terminal_reward"],"reward_minus_component2":deployed["cumulative_terminal_reward"]-m2["cumulative_terminal_reward"],"regret_minus_component1":deployed["total_terminal_regret"]-m1["total_terminal_regret"],"regret_minus_component2":deployed["total_terminal_regret"]-m2["total_terminal_regret"]},"gate":{"passed":all(checks.values()),"checks":checks}});return result


def run(root):
    loadcfg(root);(root/"oof_started.marker").write_text(datetime.now(timezone.utc).isoformat()+"\n");os.environ["CUDA_VISIBLE_DEVICES"]="";random.seed(SEED);np.random.seed(SEED);torch.manual_seed(SEED);torch.set_num_threads(min(8,os.cpu_count() or 1));torch.use_deterministic_algorithms(True);started=time.perf_counter();data=SURPRISE.build_data();hazard_x,hazard_y=hazard_arrays(data);history_data=build_common_history(data);folds=np.asarray([source_fold(x) for x in data["source"]],np.int64);surprise_scores=np.full((len(folds),3),np.nan,np.float32);hazard_prob=np.full_like(surprise_scores,np.nan);hazard_threshold=np.full(len(folds),np.nan,np.float32);history_scores=np.full((len(folds),2),np.nan,np.float32);fold_reports={}
    for fold in range(FOLDS):
        train,held=folds!=fold,folds==fold;mask,sscore,shistory,scost=train_surprise_fold(data,fold);surprise_scores[mask]=sscore;hprob,threshold,hhistory,hscale=train_hazard(hazard_x,hazard_y,train,SEED+200+fold);hazard_prob[held]=hprob[held];hazard_threshold[held]=threshold;idx,hscore,hihistory,hcost=train_history_fold(history_data,fold);history_scores[idx]=hscore
        fold_reports[str(fold)]={"surprise_last_loss":shistory[-1],"surprise_controller_ms_per_candidate":scost,"hazard_last_loss":hhistory[-1],"hazard_threshold":threshold,"hazard_scale":hscale,"history_last_loss":hihistory[-1],"history_cost":hcost};print(json.dumps({"fold_complete":fold,"hazard_threshold":threshold,"seconds":time.perf_counter()-started}),flush=True)
    if np.isnan(surprise_scores).any() or np.isnan(hazard_prob).any() or np.isnan(history_scores).any():raise RuntimeError("incomplete common OOF")
    surprise_choice=choose_three(surprise_scores);hazard_choice=choose_three(hazard_prob);hist_choice=history_choice(history_scores)
    gate_open=hazard_prob.max(1)>=hazard_threshold;combo_a=np.where(gate_open,surprise_choice,1);combo_a=np.where(surprise_choice==1,1,combo_a)
    combo_b=np.ones(len(folds),np.int64);left=(surprise_choice==0)&(history_scores[:,0]>0);right=(surprise_choice==2)&(history_scores[:,1]>0);combo_b[left]=0;combo_b[right]=2
    report_a=combo_report(data,folds,"A hazard4 gate AND surprise",hazard_choice,surprise_choice,combo_a);report_b=combo_report(data,folds,"B history veto AND surprise",hist_choice,surprise_choice,combo_b);passed=report_a["gate"]["passed"] or report_b["gate"]["passed"]
    result={"scope":"frozen512 fit-only predictor complementarity source-group OOF","gpu_used":False,"dev_used":False,"test_used":False,"calibration_or_evaluation_outcomes_read":False,"data":{"samples":len(data["slices"]),"sources":len(set(data["source"])),"blocks":len(data["errors"]),"informative_blocks":int(data["informative"].sum()),"runtime_seconds":time.perf_counter()-started},"combination_A":report_a,"combination_B":report_b,"fold_reports":fold_reports,"gate":{"passed":passed},"decision":"eligible for separate calibration/on-policy audit" if passed else "stop complementarity combinations; do not read calibration"}
    (root/"metrics.json").write_text(json.dumps(result,indent=2)+"\n");(root/"manifest.json").write_text(json.dumps({"status":"fit_oof_complete","created_utc":datetime.now(timezone.utc).isoformat(),"gate_passed":passed,"calibration_outcomes_read":False,"evaluation_outcomes_read":False,"metrics_sha256":sha(root/"metrics.json")},indent=2)+"\n");return result


def main():
    parser=argparse.ArgumentParser();parser.add_argument("mode",choices=("preregister","oof"));parser.add_argument("--output-root",type=Path,default=OUTPUT);args=parser.parse_args();os.environ["CUDA_VISIBLE_DEVICES"]="";root=args.output_root.resolve();result=preregister(root) if args.mode=="preregister" else run(root);print(json.dumps(result,indent=2))


if __name__=="__main__":main()
