#!/usr/bin/env python3
"""One-shot, CPU-only DA1 smoke for chronological rollout predictors."""
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
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[1]
TEACHER_ROOT = ROOT / "results/phoenix-2014t_ISLR/p3_partial32_rollout_predictor_dataset_smoke_v1_49faacc3"
SELF_ROOT = ROOT / "results/phoenix-2014t_ISLR/p3_partial32_rollout_dagger_da1_smoke_self_v1_49faacc3"
OLD_ROOT = ROOT / "results/phoenix-2014t_ISLR/p3_partial32_rollout_predictor_closed_loop_smoke_v2_49faacc3"
OUTPUT_ROOT = ROOT / "results/phoenix-2014t_ISLR/p3_partial32_rollout_dagger_da1_smoke_v1_49faacc3"
SEQUENCES = Path("/tmp/phoenix_p2_train_center_sequences_v1.npz")
SEED = 260924
KINDS = ("B2_bookkeeping_prefix", "B3_bookkeeping_visual_tcn_prefix")


def import_tool(name):
    path=ROOT/f"tools/{name}.py"; spec=importlib.util.spec_from_file_location(name,path)
    module=importlib.util.module_from_spec(spec); sys.modules[spec.name]=module; spec.loader.exec_module(module)
    return module

OLD=import_tool("analyze_phoenix_partial_rollout_predictor")
SELF=import_tool("build_phoenix_partial_rollout_dagger_self_dataset")
CHRON, DATA_BUILDER = OLD.DATA_BUILDER.CHRON, OLD.DATA_BUILDER


def sha256_file(path):
    digest=hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda:handle.read(1024*1024),b""): digest.update(block)
    return digest.hexdigest()


def set_determinism():
    random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
    torch.set_num_threads(min(8,os.cpu_count() or 1)); torch.use_deterministic_algorithms(True)


def row_value(row,sequences):
    name=row["sample_id"]
    return {"bookkeeping":OLD.bookkeeping_features(row),"prefix":OLD.prefix_features(row),
            "visual":sequences.history(name,row["candidate_window_start"]).astype(np.float16),
            "advantage":float(row["label"]["rollout16_signed_advantage"]),
            "positive":float(row["label"]["rollout16_positive"])}


def load_teacher(teacher_root,sequences):
    _,paths=OLD.completed_paths(teacher_root); values={"fit":[],"calibration":[]}; sources={"fit":set(),"calibration":set()}
    for row in OLD.iter_rows(paths):
        if row["sample_id"] not in sequences.slices: continue
        values[row["partition"]].append(row_value(row,sequences)); sources[row["partition"]].add(row["source_video_id"])
    if sources["fit"] & sources["calibration"]: raise ValueError("fit/cal source overlap")
    return {key:OLD.stack_rows(value) for key,value in values.items()},sources


def load_self(self_root,kind,sequences):
    manifest=json.loads((self_root/"dataset_manifest.json").read_text())
    if manifest["status"]!="complete" or manifest["dataset_aggregation_round"]!=1: raise ValueError("not atomic DA1 data")
    rows=[]
    for info in manifest["kinds"][kind]["shards"]:
        matches=list((self_root/kind).glob(f"self-{info['shard']:05d}-of-*.jsonl.gz"))
        if len(matches)!=1 or sha256_file(matches[0])!=info["sha256"]: raise ValueError("self hash mismatch")
        with gzip.open(matches[0],"rt",encoding="utf-8") as handle:
            for line in handle:
                row=json.loads(line); provenance=row["provenance"]
                if (row["partition"]!="fit" or provenance["teacher_prefix_reuse"] or
                    not provenance["prefix_uses_only_this_predictor_own_past_selected"] or
                    not provenance["current_candidate_expensive_logits_absent_from_inputs"]):
                    raise ValueError("self causality failure")
                rows.append(row_value(row,sequences))
    return OLD.stack_rows(rows),manifest["kinds"][kind]


def source_balanced_stats(teacher,self_data):
    stats={}
    for key in ("bookkeeping","prefix","visual"):
        axes=(0,1) if key=="visual" else 0
        mt=teacher[key].mean(axis=axes,dtype=np.float64); ms=self_data[key].mean(axis=axes,dtype=np.float64)
        vt=teacher[key].var(axis=axes,dtype=np.float64); vs=self_data[key].var(axis=axes,dtype=np.float64)
        mean=.5*(mt+ms); var=.5*(vt+(mt-mean)**2)+.5*(vs+(ms-mean)**2)
        scale=np.sqrt(var).astype(np.float32); scale[scale<1e-6]=1
        stats[key]=(mean.astype(np.float32),scale)
    return stats


def source_balanced_row_weights(n_teacher,n_self):
    if n_teacher<=0 or n_self<=0: raise ValueError("both sources required")
    return np.concatenate([np.full(n_teacher,.5/n_teacher),np.full(n_self,.5/n_self)]).astype(np.float64)


def concatenate(teacher,self_data,stats):
    output={key:np.concatenate([teacher[key],self_data[key]]) for key in teacher}
    output=OLD.normalize(output,stats)
    output["source_weight"]=source_balanced_row_weights(len(teacher["advantage"]),len(self_data["advantage"]))
    return output


def train_da1(kind,fit,epochs=5,batch_size=1024):
    torch.manual_seed(SEED); use_prefix,use_visual=OLD.VARIANTS[kind]
    model=OLD.RolloutPredictor(fit["bookkeeping"].shape[1],fit["prefix"].shape[1],fit["visual"].shape[2],use_prefix,use_visual)
    optimizer=torch.optim.AdamW(model.parameters(),lr=1e-3,weight_decay=1e-4)
    w=fit["source_weight"]; prevalence=float((w*fit["positive"]).sum()/w.sum())
    pos_weight=min(20.0,(1-prevalence)/max(prevalence,1e-12))
    generator=torch.Generator().manual_seed(SEED); history=[]
    for epoch in range(epochs):
        model.train(); order=torch.randperm(len(w),generator=generator); totals=np.zeros(3)
        for left in range(0,len(order),batch_size):
            idx=order[left:left+batch_size].numpy(); weight=torch.from_numpy(w[idx]).float()
            args=[torch.from_numpy(fit[key][idx]).float() for key in ("bookkeeping","prefix","visual")]
            target=torch.from_numpy(fit["advantage"][idx]); positive=torch.from_numpy(fit["positive"][idx])
            optimizer.zero_grad(set_to_none=True); pred,logit=model(*args)
            reg=F.smooth_l1_loss(pred,target,reduction="none"); bce=F.binary_cross_entropy_with_logits(logit,positive,pos_weight=torch.tensor(pos_weight),reduction="none")
            denominator=weight.sum(); reg_loss=(reg*weight).sum()/denominator; bce_loss=(bce*weight).sum()/denominator
            loss=reg_loss+.25*bce_loss; loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),1); optimizer.step()
            totals += [float(loss.detach()),float(reg_loss.detach()),float(bce_loss.detach())]
        history.append({"epoch":epoch+1,"mean_batch_total":totals[0]/math.ceil(len(w)/batch_size),
                        "mean_batch_smooth_l1":totals[1]/math.ceil(len(w)/batch_size),
                        "mean_batch_weighted_bce":totals[2]/math.ceil(len(w)/batch_size)})
        print(json.dumps({"kind":kind,"DA":1,**history[-1]}),flush=True)
    model.eval(); return model,history,pos_weight


def weighted_quantile(values,weights,q=.9):
    order=np.argsort(values,kind="stable"); values=np.asarray(values)[order]; weights=np.asarray(weights)[order]
    return float(values[np.searchsorted(np.cumsum(weights),q*weights.sum(),side="left")])


def ks_distance(a,b):
    a=np.sort(np.asarray(a)); b=np.sort(np.asarray(b)); values=np.sort(np.concatenate([a,b]))
    return float(np.max(np.abs(np.searchsorted(a,values,side="right")/len(a)-np.searchsorted(b,values,side="right")/len(b))))


def load_old(kind): return SELF.load_old(kind,OLD_ROOT)


def run_policy(kind,model,threshold,stats,sequences,name,probabilities,reference,vocab,blank_id):
    selected=[]; balance=0.; forced=0; scores=[]; books=[]; prefixes=[]
    for start in range(len(probabilities)):
        if CHRON.is_skeleton(start): selected.append(start); continue
        balance=CHRON.accrue_token(balance)
        if not CHRON.token_eligible(balance): continue
        is_forced=CHRON.token_forced(balance); execute=is_forced
        if is_forced: forced+=1
        else:
            row=OLD.online_feature_row(probabilities,selected,start,balance,blank_id)
            books.append((OLD.bookkeeping_features(row)-stats["bookkeeping"][0])/stats["bookkeeping"][1])
            prefixes.append((OLD.prefix_features(row)-stats["prefix"][0])/stats["prefix"][1])
            score=OLD.online_score(kind,model,stats,sequences,name,row,start); scores.append(score); execute=score>threshold
        if execute: selected.append(start); balance-=1
    hypothesis,counts=CHRON.decode_counts(probabilities,selected,reference,vocab,blank_id)
    return {"selected":selected,"counts":counts,"hypothesis":hypothesis,"unspent_bonus_tokens":balance,
            "forced_actions":forced,"scores":scores,"bookkeeping":books,"prefix":prefixes,
            "coverage":CHRON.coverage_metrics(selected,len(probabilities))}


def summarize(rows,uniform,threshold,teacher_scores,teacher_book,teacher_prefix):
    counts=CHRON.aggregate_counts([r["counts"] for r in rows]); base=CHRON.aggregate_counts([r["counts"] for r in uniform])
    scores=np.asarray([x for r in rows for x in r["scores"]]); books=np.asarray([x for r in rows for x in r["bookkeeping"]]); prefixes=np.asarray([x for r in rows for x in r["prefix"]])
    total=sum(r["dense_windows"] for r in rows); executed=sum(len(r["selected"]) for r in rows)
    return {"metrics":counts,"delta_wer_pp_vs_online_uniform":counts["wer"]-base["wer"],
            "error_delta_vs_online_uniform":counts["error"]-base["error"],"actual_window_rate":executed/total,
            "executed_windows":executed,"dense_windows":total,
            "max_gap":max(r["coverage"]["max_gap_including_endpoints"] for r in rows),
            "unspent_tokens_total":sum(r["unspent_bonus_tokens"] for r in rows),
            "forced_actions":sum(r["forced_actions"] for r in rows),
            "autonomous_execute_rate":float((scores>threshold).mean()),
            "paired_bootstrap_vs_online_uniform":CHRON.paired_bootstrap([r["counts"] for r in rows],[r["counts"] for r in uniform]),
            "teacher_to_self_drift":{"teacher_expected_trigger_rate":float((teacher_scores>threshold).mean()),
               "absolute_execute_rate_shift":float(abs((scores>threshold).mean()-(teacher_scores>threshold).mean())),
               "score_ks":ks_distance(teacher_scores,scores),
               "score_mean_teacher":float(np.mean(teacher_scores)),"score_mean_self":float(np.mean(scores)),
               "mean_abs_standardized_bookkeeping_shift":float(np.abs(books.mean(0)-teacher_book.mean(0)).mean()),
               "mean_abs_standardized_prefix_shift":float(np.abs(prefixes.mean(0)-teacher_prefix.mean(0)).mean())}}


def closed_loop(sequences,models,stats,thresholds,cal_teacher):
    dense=CHRON.DEFAULT_DENSE_ROOT; indices,shard_count=CHRON.completed_shard_indices(dense)
    assignments=json.loads((dense/"fit_calibration_split.json").read_text())["assignments"]
    vocab=json.loads(CHRON.DEFAULT_VOCAB.read_text()); blank_id=vocab.index("<blank>")
    outputs={key:[] for key in ["online_uniform","myopic_oracle","rollout16_oracle",*models]}
    for shard in indices:
        results,logits,_=DATA_BUILDER.BUILDER.validate_dense_shard(dense,shard,shard_count,verify_hashes=False)
        for name in results:
            if assignments[name]!="calibration" or name not in sequences.slices: continue
            probabilities=DATA_BUILDER.ORACLE.softmax_rows(np.asarray(logits[name])); reference=DATA_BUILDER.REPLAY.clean_phoenix_2014_trans(results[name]["gls_ref"]); total=len(probabilities)
            selected,balance=CHRON.online_uniform_schedule(total); _,counts=CHRON.decode_counts(probabilities,selected,reference,vocab,blank_id)
            outputs["online_uniform"].append({"selected":selected,"counts":counts,"dense_windows":total,"unspent_bonus_tokens":balance,"forced_actions":0,"coverage":CHRON.coverage_metrics(selected,total)})
            for key,(kind,model) in models.items():
                value=run_policy(kind,model,thresholds[key],stats[key],sequences,name,probabilities,reference,vocab,blank_id); value["dense_windows"]=total; outputs[key].append(value)
            for policy in ("myopic","rollout16"):
                value=CHRON.run_policy(probabilities,reference,vocab,blank_id,policy); outputs[f"{policy}_oracle"].append({"selected":value["selected"],"counts":value["counts"],"dense_windows":total,"unspent_bonus_tokens":value["unspent_bonus_tokens"],"forced_actions":sum(x["forced_by_token_capacity"] for x in value["trace"]),"coverage":value["coverage"]})
    uniform=outputs["online_uniform"]; base_counts=CHRON.aggregate_counts([r["counts"] for r in uniform]); summaries={"online_uniform":{"metrics":base_counts,"actual_window_rate":sum(len(r["selected"]) for r in uniform)/sum(r["dense_windows"] for r in uniform)}}
    rollout_counts=CHRON.aggregate_counts([r["counts"] for r in outputs["rollout16_oracle"]]); oracle_gain=base_counts["wer"]-rollout_counts["wer"]
    for key in models:
        normalized=OLD.normalize(cal_teacher,stats[key]); teacher_score,_=OLD.predict(models[key][1],normalized)
        summary=summarize(outputs[key],uniform,thresholds[key],teacher_score,normalized["bookkeeping"],normalized["prefix"])
        summary["fraction_of_rollout_oracle_wer_gap_recovered"]=(base_counts["wer"]-summary["metrics"]["wer"])/oracle_gain
        summaries[key]=summary
    for key in ("myopic_oracle","rollout16_oracle"):
        counts=CHRON.aggregate_counts([r["counts"] for r in outputs[key]])
        summaries[key]={"metrics":counts,"delta_wer_pp_vs_online_uniform":counts["wer"]-base_counts["wer"],"error_delta_vs_online_uniform":counts["error"]-base_counts["error"],"actual_window_rate":sum(len(r["selected"]) for r in outputs[key])/sum(r["dense_windows"] for r in outputs[key]),"paired_bootstrap_vs_online_uniform":CHRON.paired_bootstrap([r["counts"] for r in outputs[key]],[r["counts"] for r in uniform])}
    for da,old in (("B2_DA1","B2_old"),("B3_DA1","B3_old")):
        summaries[da]["delta_wer_pp_vs_corresponding_old"]=summaries[da]["metrics"]["wer"]-summaries[old]["metrics"]["wer"]
        summaries[da]["paired_bootstrap_vs_corresponding_old"]=CHRON.paired_bootstrap([r["counts"] for r in outputs[da]],[r["counts"] for r in outputs[old]])
    return {"samples":len(uniform),"policies":summaries}


def run(teacher_root,self_root,old_root,sequences_path,output_root):
    for path in (teacher_root,self_root,old_root,sequences_path,output_root): OLD.reject_test_path(path)
    if output_root.exists(): raise FileExistsError(output_root)
    temporary=output_root.with_name(f".{output_root.name}.incomplete-{os.getpid()}"); temporary.mkdir(parents=True)
    config={"dataset_aggregation_round":1,"source_weight":{"teacher":.5,"self":.5},"variants":list(KINDS),"epochs":5,"batch_size":1024,"optimizer":"AdamW","learning_rate":1e-3,"weight_decay":1e-4,"loss":"SmoothL1 + 0.25 capped-positive-weight BCE","threshold":"fit aggregated source-balanced score 90th percentile","calibration_threshold_tuning":False,"protocol":{"skeleton_period":4,"accrual":1/3,"capacity":2,"rollout_horizon":16,"lookahead":8,"unknown_eos_no_topup":True}}
    (temporary/"resolved_config_preregistered.json").write_text(json.dumps(config,indent=2)+"\n")
    set_determinism(); started=time.perf_counter(); sequences=OLD.SequenceFeatures(sequences_path)
    teacher,sources=load_teacher(teacher_root,sequences); cal=teacher["calibration"]
    models={}; stats_map={}; thresholds={}; held={}; train_summary={}; old_models={}
    for kind in KINDS:
        self_data,self_info=load_self(self_root,kind,sequences); stats=source_balanced_stats(teacher["fit"],self_data); fit=concatenate(teacher["fit"],self_data,stats)
        model,history,pos_weight=train_da1(kind,fit); teacher_norm=OLD.normalize(teacher["fit"],stats); self_norm=OLD.normalize(self_data,stats)
        teacher_score,_=OLD.predict(model,teacher_norm); self_score,_=OLD.predict(model,self_norm)
        threshold=weighted_quantile(np.concatenate([teacher_score,self_score]),fit["source_weight"],.9)
        cal_norm=OLD.normalize(cal,stats); score,positive=OLD.predict(model,cal_norm)
        held[f"{kind}_DA1"]=OLD.evaluation_metrics(cal_norm["positive"],cal_norm["advantage"],score,positive)
        key="B2_DA1" if kind.startswith("B2") else "B3_DA1"; models[key]=(kind,model); stats_map[key]=stats; thresholds[key]=threshold
        train_summary[key]={"teacher_rows":len(teacher_score),"self_rows":len(self_score),"teacher_positive_rate":float(teacher["fit"]["positive"].mean()),"self_positive_rate":float(self_data["positive"].mean()),"positive_class_weight":pos_weight,"threshold":threshold,"teacher_trigger_rate":float((teacher_score>threshold).mean()),"self_trigger_rate":float((self_score>threshold).mean()),"score_ks_teacher_self":ks_distance(teacher_score,self_score),"mean_abs_standardized_bookkeeping_teacher_self":float(np.abs(teacher_norm["bookkeeping"].mean(0)-self_norm["bookkeeping"].mean(0)).mean()),"mean_abs_standardized_prefix_teacher_self":float(np.abs(teacher_norm["prefix"].mean(0)-self_norm["prefix"].mean(0)).mean()),"history":history,"self_manifest_counts":self_info["counts"]}
        torch.save({"state_dict":model.state_dict(),"normalization_fit_only_source_balanced":stats,"variant":kind,"DA_round":1},temporary/f"{key}.pt")
        old_model,old_stats,old_threshold=load_old(kind); oldkey="B2_old" if kind.startswith("B2") else "B3_old"; models[oldkey]=(kind,old_model); stats_map[oldkey]=old_stats; thresholds[oldkey]=old_threshold
        old_cal=OLD.normalize(cal,old_stats); oscore,opos=OLD.predict(old_model,old_cal); held[oldkey]=OLD.evaluation_metrics(old_cal["positive"],old_cal["advantage"],oscore,opos)
    loop=closed_loop(sequences,models,stats_map,thresholds,cal)
    decisions={}
    for prefix in ("B2","B3"):
        old=loop["policies"][f"{prefix}_old"]; da=loop["policies"][f"{prefix}_DA1"]
        drift_old=old["teacher_to_self_drift"]; drift_da=da["teacher_to_self_drift"]
        drift_improved=(drift_da["absolute_execute_rate_shift"]<drift_old["absolute_execute_rate_shift"] or drift_da["score_ks"]<drift_old["score_ks"] or drift_da["mean_abs_standardized_bookkeeping_shift"]<drift_old["mean_abs_standardized_bookkeeping_shift"])
        wer_improved=da["metrics"]["wer"]<old["metrics"]["wer"]
        decisions[prefix]={"drift_improved":drift_improved,"closed_loop_wer_improved":wer_improved,"supports_covariate_shift_explanation":drift_improved and wer_improved,"strong_go_vs_uniform":da["delta_wer_pp_vs_online_uniform"]<=-.5 and da["paired_bootstrap_vs_online_uniform"]["ci95"][1]<0}
    visual_increment={"DA1_B3_minus_B2_PR_AUC":held["B3_bookkeeping_visual_tcn_prefix_DA1"]["pr_auc_regression_score"]-held["B2_bookkeeping_prefix_DA1"]["pr_auc_regression_score"],"DA1_B3_minus_B2_WER_pp":loop["policies"]["B3_DA1"]["metrics"]["wer"]-loop["policies"]["B2_DA1"]["metrics"]["wer"]}
    result={"scope":"single preregistered partial32 train-only DA1 smoke; not a research result","gpu_used":False,"dev_used":False,"test_used":False,"dataset_aggregation_rounds_run":1,"fit_calibration_source_disjoint":not bool(sources["fit"]&sources["calibration"]),"training":train_summary,"held_out_calibration":held,"closed_loop":loop,"decision":decisions,"visual_increment":visual_increment,"limitations":["32/56 non-random train shards","calibration repeatedly inspected by prior smokes","future-aware rollout teacher","no formal dev/test result","no real wall-time or submission-latency conclusion"],"runtime_seconds":time.perf_counter()-started}
    (temporary/"metrics.json").write_text(json.dumps(result,indent=2)+"\n"); manifest={"status":"complete","DA_round":1,"created_utc":datetime.now(timezone.utc).isoformat(),"metrics_sha256":sha256_file(temporary/"metrics.json"),"config_sha256":sha256_file(temporary/"resolved_config_preregistered.json"),"self_manifest_sha256":sha256_file(self_root/"dataset_manifest.json")}; (temporary/"manifest.json").write_text(json.dumps(manifest,indent=2)+"\n"); temporary.replace(output_root); return result


def main():
    parser=argparse.ArgumentParser(description=__doc__); parser.add_argument("--teacher-root",type=Path,default=TEACHER_ROOT); parser.add_argument("--self-root",type=Path,default=SELF_ROOT); parser.add_argument("--old-root",type=Path,default=OLD_ROOT); parser.add_argument("--sequences",type=Path,default=SEQUENCES); parser.add_argument("--output-root",type=Path,default=OUTPUT_ROOT); args=parser.parse_args(); result=run(args.teacher_root.resolve(),args.self_root.resolve(),args.old_root.resolve(),args.sequences.resolve(),args.output_root.resolve()); print(json.dumps(result,indent=2))
if __name__=="__main__": main()
