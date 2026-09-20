#!/usr/bin/env python3
"""One preregistered CPU-only approximate policy-improvement smoke for B2.

Build mode labels states visited by the frozen original B2 policy with
16-candidate A^pi0.  Analyze mode trains exactly one same-capacity B2 model,
uses predicted expected advantage > 0 as the primary action rule, and audits
the resulting policy on the held-out train-calibration partition.
"""
import argparse, gzip, hashlib, importlib.util, json, math, os, random, sys, time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
OLD_ROOT = ROOT / "results/phoenix-2014t_ISLR/p3_partial32_rollout_predictor_closed_loop_smoke_v2_49faacc3"
DA_ROOT = ROOT / "results/phoenix-2014t_ISLR/p3_partial32_rollout_dagger_da1_smoke_v1_49faacc3"
AUDIT_ROOT = ROOT / "results/phoenix-2014t_ISLR/p3_partial32_on_policy_advantage_smoke_v1_49faacc3"
OUTPUT_ROOT = ROOT / "results/phoenix-2014t_ISLR/p3_partial32_on_policy_pi1_smoke_v1_49faacc3"
DATA_ROOT = OUTPUT_ROOT.with_name(OUTPUT_ROOT.name + "_dataset")
SEQUENCES = Path("/tmp/phoenix_p2_train_center_sequences_v1.npz")
KIND = "B2_bookkeeping_prefix"
SEED = 260924

def imp(name):
    path=ROOT/f"tools/{name}.py"; spec=importlib.util.spec_from_file_location(name,path)
    module=importlib.util.module_from_spec(spec); sys.modules[spec.name]=module; spec.loader.exec_module(module); return module

OLD=imp("analyze_phoenix_partial_rollout_predictor")
SELF=imp("build_phoenix_partial_rollout_dagger_self_dataset")
DA=imp("analyze_phoenix_partial_rollout_dagger_da1")
AUDIT=imp("analyze_phoenix_partial_on_policy_advantage")
CHRON, BUILDER = OLD.DATA_BUILDER.CHRON, OLD.DATA_BUILDER

def sha256(path):
    h=hashlib.sha256()
    with Path(path).open("rb") as f:
        for b in iter(lambda:f.read(1024*1024),b""): h.update(b)
    return h.hexdigest()

def config():
    return {"experiment":"B2-PI1", "policy_iterations":1, "cpu_only":True,
      "base_policy":"original B2_bookkeeping_prefix and fit-only threshold",
      "training_trajectory":"pi0 own fit states only", "old_eager_labels_mixed":False,
      "architecture":"unchanged B2 bookkeeping+prefix MLP", "epochs":5,
      "optimizer":"AdamW(lr=1e-3,weight_decay=1e-4)",
      "loss":"SmoothL1 + 0.25 capped-positive-weight BCE",
      "primary_threshold":"predicted regression expected advantage > 0",
      "sensitivity_threshold":"fit-only regression-score 90th percentile",
      "calibration_threshold_tuning":False,
      "protocol":{"skeleton":"start%4==0","bonus_accrual":1/3,"initial":0,
        "capacity":2,"eligible":1,"forced":2,"eos_topup":False,"horizon":16,
        "unknown_T_EOS":True},
      "inputs":["bookkeeping","past-selected decoder prefix"],
      "forbidden_inputs":["current candidate logits","future logits","reference","T","EOS"]}

def preregister(root):
    root.mkdir(parents=True,exist_ok=True)
    p=root/"resolved_config_preregistered.json"
    if not p.exists(): p.write_text(json.dumps(config(),indent=2)+"\n")
    elif json.loads(p.read_text()) != config(): raise ValueError("preregistered config changed")

def feature_value(row):
    # Visual is structurally present because the common model API expects it,
    # but B2 has use_visual=False and never reads it.
    return {"bookkeeping":OLD.bookkeeping_features(row), "prefix":OLD.prefix_features(row),
      "visual":np.zeros((31,17),dtype=np.float16)}

def sample_rows(name,result,logits,vocab,blank_id,model,stats,threshold,sequences):
    probabilities=BUILDER.ORACLE.softmax_rows(np.asarray(logits)); reference=BUILDER.REPLAY.clean_phoenix_2014_trans(result["gls_ref"])
    selected=[]; balance=0.; rows=[]; counts=Counter()
    for start in range(len(probabilities)):
        if CHRON.is_skeleton(start): selected.append(start); counts["skeleton"]+=1; continue
        balance=CHRON.accrue_token(balance)
        if not CHRON.token_eligible(balance): counts["ineligible"]+=1; continue
        forced=CHRON.token_forced(balance); execute=forced
        if not forced:
            row=OLD.online_feature_row(probabilities,selected,start,balance,blank_id)
            score=OLD.online_score(KIND,model,stats,sequences,name,row,start); execute=score>threshold
            advantage,skip_error,execute_error=AUDIT.on_policy_advantage(
                KIND,model,stats,threshold,sequences,name,probabilities,selected,balance,start,reference,vocab,blank_id)
            rows.append({"sample_id":name,"source_video_id":BUILDER.source_video(name),"partition":"fit",
              "trajectory_source":"frozen_B2_old_pi0","candidate_window_start":start,
              "bookkeeping":row["bookkeeping"],"prefix":row["prefix"],
              "label":{"on_policy16_signed_advantage":advantage,"positive":advantage>0,
                "skip_error":skip_error,"execute_error":execute_error},
              "pi0":{"score":score,"threshold":threshold,"action_execute":execute},
              "provenance":{"pi0_own_past_selected_prefix":True,"forced_state":False,
                "future_and_reference_label_only":True,"current_candidate_logits_absent":True,
                "T_EOS_absent":True,"continuation_same_frozen_pi0":True}})
            counts["autonomous"]+=1; counts["positive"]+=int(advantage>0); counts["negative"]+=int(advantage<0)
        else: counts["forced"]+=1
        if execute: selected.append(start); balance-=1
    counts["samples"]+=1; counts["dense"]+=len(probabilities); counts["executed"]+=len(selected)
    return rows,counts

def build_worker(data_root,worker,workers):
    preregister(data_root); torch.set_num_threads(1); torch.use_deterministic_algorithms(True)
    dense=CHRON.DEFAULT_DENSE_ROOT; indices,shard_count=CHRON.completed_shard_indices(dense)
    if len(indices)!=32: raise ValueError("requires frozen partial32")
    split=json.loads((dense/"fit_calibration_split.json").read_text()); assignments=split["assignments"]
    fit_sources={BUILDER.source_video(k) for k,v in assignments.items() if v=="fit"}; cal_sources={BUILDER.source_video(k) for k,v in assignments.items() if v=="calibration"}
    if fit_sources & cal_sources: raise ValueError("fit/cal source overlap")
    vocab=json.loads(CHRON.DEFAULT_VOCAB.read_text()); blank_id=vocab.index("<blank>")
    model,stats,threshold=SELF.load_old(KIND,OLD_ROOT); sequences=OLD.SequenceFeatures(SEQUENCES)
    work=[x for pos,x in enumerate(indices) if pos%workers==worker]
    for pos,index in enumerate(work,1):
        final=data_root/f"pi0-on-policy-{index:05d}-of-{shard_count:05d}.jsonl.gz"; marker=final.with_suffix(".complete.json")
        if final.exists() and marker.exists() and json.loads(marker.read_text())["sha256"]==sha256(final):
            print(json.dumps({"worker":worker,"resumed":True,"shard":index,"done":pos,"total":len(work)}),flush=True); continue
        results,logits,_=BUILDER.BUILDER.validate_dense_shard(dense,index,shard_count,verify_hashes=False)
        tmp=final.with_name("."+final.name+f".incomplete-{os.getpid()}"); total=Counter(); started=time.perf_counter()
        with gzip.open(tmp,"wt",encoding="utf-8",compresslevel=5) as out:
            for name in results:
                if assignments[name]!="fit" or name not in sequences.slices: continue
                rows,c=sample_rows(name,results[name],logits[name],vocab,blank_id,model,stats,threshold,sequences)
                for row in rows: out.write(json.dumps(row,separators=(",",":"))+"\n")
                total.update(c)
        tmp.replace(final); info={"shard":index,"counts":dict(total),"sha256":sha256(final),"bytes":final.stat().st_size,"seconds":time.perf_counter()-started}
        marker.write_text(json.dumps(info,indent=2)+"\n")
        print(json.dumps({"worker":worker,"resumed":False,"shard":index,"done":pos,"total":len(work),**info}),flush=True)

def finalize(data_root):
    preregister(data_root); indices,shard_count=CHRON.completed_shard_indices(CHRON.DEFAULT_DENSE_ROOT); shards=[]; counts=Counter()
    for index in indices:
        p=data_root/f"pi0-on-policy-{index:05d}-of-{shard_count:05d}.jsonl.gz"; m=p.with_suffix(".complete.json")
        if not p.exists() or not m.exists(): raise RuntimeError(f"missing shard {index}")
        x=json.loads(m.read_text());
        if x["sha256"]!=sha256(p): raise ValueError(f"hash mismatch {index}")
        shards.append(x); counts.update(x["counts"])
    manifest={"status":"complete","created_utc":datetime.now(timezone.utc).isoformat(),"config_sha256":sha256(data_root/"resolved_config_preregistered.json"),"counts":dict(counts),"positive_prevalence":counts["positive"]/counts["autonomous"],"shards":shards}
    (data_root/"dataset_manifest.json").write_text(json.dumps(manifest,indent=2)+"\n")

def load_fit(data_root):
    man=json.loads((data_root/"dataset_manifest.json").read_text());
    if man["status"]!="complete": raise ValueError("dataset incomplete")
    rows=[]
    for info in man["shards"]:
        matches=list(data_root.glob(f"pi0-on-policy-{info['shard']:05d}-of-*.jsonl.gz"));
        if len(matches)!=1 or sha256(matches[0])!=info["sha256"]: raise ValueError("dataset hash failure")
        with gzip.open(matches[0],"rt",encoding="utf-8") as f:
            for line in f:
                row=json.loads(line); p=row["provenance"]
                if not all((p["pi0_own_past_selected_prefix"],not p["forced_state"],p["future_and_reference_label_only"],p["current_candidate_logits_absent"],p["T_EOS_absent"],p["continuation_same_frozen_pi0"])): raise ValueError("causality failure")
                v=feature_value(row); v.update({"advantage":float(row["label"]["on_policy16_signed_advantage"]),"positive":float(row["label"]["positive"])}); rows.append(v)
    return OLD.stack_rows(rows),man

def policy(kind,model,stats,threshold,sequences,name,p,ref,vocab,blank_id):
    selected=[]; balance=0.; forced=auto=trigger=0; scores=[]
    for start in range(len(p)):
        if CHRON.is_skeleton(start): selected.append(start); continue
        balance=CHRON.accrue_token(balance)
        if not CHRON.token_eligible(balance): continue
        f=CHRON.token_forced(balance); execute=f
        if f: forced+=1
        else:
            auto+=1; row=OLD.online_feature_row(p,selected,start,balance,blank_id); score=OLD.online_score(kind,model,stats,sequences,name,row,start); scores.append(score); execute=score>threshold; trigger+=int(execute)
        if execute: selected.append(start); balance-=1
    _,counts=CHRON.decode_counts(p,selected,ref,vocab,blank_id)
    return {"selected":selected,"counts":counts,"dense_windows":len(p),"unspent_bonus_tokens":balance,"forced_actions":forced,"autonomous_decisions":auto,"autonomous_triggers":trigger,"scores":scores,"coverage":CHRON.coverage_metrics(selected,len(p))}

def summarize(rows,uniform):
    c=CHRON.aggregate_counts([r["counts"] for r in rows]); u=CHRON.aggregate_counts([r["counts"] for r in uniform]); dense=sum(r["dense_windows"] for r in rows); auto=sum(r.get("autonomous_decisions",0) for r in rows)
    return {"metrics":c,"delta_wer_pp_vs_online_uniform":c["wer"]-u["wer"],"error_delta_vs_online_uniform":c["error"]-u["error"],"executed_windows":sum(len(r["selected"]) for r in rows),"dense_windows":dense,"actual_window_rate":sum(len(r["selected"]) for r in rows)/dense,"max_gap":max(r["coverage"]["max_gap_including_endpoints"] for r in rows),"unspent_tokens_total":sum(r["unspent_bonus_tokens"] for r in rows),"forced_actions":sum(r.get("forced_actions",0) for r in rows),"autonomous_decisions":auto,"autonomous_trigger_rate":sum(r.get("autonomous_triggers",0) for r in rows)/max(1,auto),"paired_bootstrap_vs_online_uniform":CHRON.paired_bootstrap([r["counts"] for r in rows],[r["counts"] for r in uniform])}

def calibration_labels(base,sequences):
    kind,model,stats,threshold=base; dense=CHRON.DEFAULT_DENSE_ROOT; indices,n=CHRON.completed_shard_indices(dense); assignments=json.loads((dense/"fit_calibration_split.json").read_text())["assignments"]; vocab=json.loads(CHRON.DEFAULT_VOCAB.read_text()); blank=vocab.index("<blank>"); values=[]
    for shard in indices:
        results,logits,_=BUILDER.BUILDER.validate_dense_shard(dense,shard,n,verify_hashes=False)
        for name in results:
            if assignments[name]!="calibration" or name not in sequences.slices: continue
            p=BUILDER.ORACLE.softmax_rows(np.asarray(logits[name])); ref=BUILDER.REPLAY.clean_phoenix_2014_trans(results[name]["gls_ref"]); selected=[]; balance=0.
            for start in range(len(p)):
                if CHRON.is_skeleton(start): selected.append(start); continue
                balance=CHRON.accrue_token(balance)
                if not CHRON.token_eligible(balance): continue
                forced=CHRON.token_forced(balance); row=OLD.online_feature_row(p,selected,start,balance,blank); execute=forced
                if not forced:
                    score=OLD.online_score(kind,model,stats,sequences,name,row,start); execute=score>threshold
                    a,_,_=AUDIT.on_policy_advantage(kind,model,stats,threshold,sequences,name,p,selected,balance,start,ref,vocab,blank)
                    v=feature_value(row); v.update({"advantage":float(a),"positive":float(a>0)}); values.append(v)
                if execute: selected.append(start); balance-=1
    return OLD.stack_rows(values)

def analyze(data_root,output_root):
    if output_root.exists(): raise FileExistsError(output_root)
    tmp=output_root.with_name("."+output_root.name+f".incomplete-{os.getpid()}"); tmp.mkdir(parents=True); (tmp/"resolved_config_preregistered.json").write_text(json.dumps(config(),indent=2)+"\n")
    random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED); torch.set_num_threads(min(8,os.cpu_count() or 1)); torch.use_deterministic_algorithms(True); started=time.perf_counter()
    fit,manifest=load_fit(data_root); stats=OLD.normalization(fit); fitn=OLD.normalize(fit,stats); model,history,pos_weight=OLD.train_variant(KIND,fitn,epochs=5,batch_size=1024,seed=SEED); fit_score,_=OLD.predict(model,fitn); sensitivity=float(np.quantile(fit_score,.90)); primary=0.
    sequences=OLD.SequenceFeatures(SEQUENCES); old=(KIND,*SELF.load_old(KIND,OLD_ROOT)); da_model,da_stats,da_thr=AUDIT.load_da(KIND,"B2_DA1",DA_ROOT); da=(KIND,da_model,da_stats,da_thr); pi=(KIND,model,stats,primary); pis=(KIND,model,stats,sensitivity)
    cal=calibration_labels(old,sequences); caln=OLD.normalize(cal,stats); score,pos=OLD.predict(model,caln); offline=OLD.evaluation_metrics(caln["positive"],caln["advantage"],score,pos)
    dense=CHRON.DEFAULT_DENSE_ROOT; indices,n=CHRON.completed_shard_indices(dense); assignments=json.loads((dense/"fit_calibration_split.json").read_text())["assignments"]; vocab=json.loads(CHRON.DEFAULT_VOCAB.read_text()); blank=vocab.index("<blank>")
    policies={"B2_old":old,"B2_DA1":da,"B2_PI1_primary":pi,"B2_PI1_fit90_sensitivity":pis}; outputs={k:[] for k in ["online_uniform","myopic_oracle","rollout16_eager_oracle",*policies]}
    pi1_audit=[]
    for shard in indices:
        results,logits,_=BUILDER.BUILDER.validate_dense_shard(dense,shard,n,verify_hashes=False)
        for name in results:
            if assignments[name]!="calibration" or name not in sequences.slices: continue
            p=BUILDER.ORACLE.softmax_rows(np.asarray(logits[name])); ref=BUILDER.REPLAY.clean_phoenix_2014_trans(results[name]["gls_ref"]); total=len(p); sel,bal=CHRON.online_uniform_schedule(total); _,c=CHRON.decode_counts(p,sel,ref,vocab,blank); outputs["online_uniform"].append({"selected":sel,"counts":c,"dense_windows":total,"unspent_bonus_tokens":bal,"coverage":CHRON.coverage_metrics(sel,total)})
            for key,value in policies.items(): outputs[key].append(policy(*value,sequences,name,p,ref,vocab,blank))
            for source,key in (("myopic","myopic_oracle"),("rollout16","rollout16_eager_oracle")):
                v=CHRON.run_policy(p,ref,vocab,blank,source); outputs[key].append({"selected":v["selected"],"counts":v["counts"],"dense_windows":total,"unspent_bonus_tokens":v["unspent_bonus_tokens"],"forced_actions":sum(x["forced_by_token_capacity"] for x in v["trace"]),"coverage":v["coverage"]})
            # A^pi1 audit on the primary PI1 trajectory.
            selected=[]; balance=0.
            for start in range(total):
                if CHRON.is_skeleton(start): selected.append(start); continue
                balance=CHRON.accrue_token(balance)
                if not CHRON.token_eligible(balance): continue
                forced=CHRON.token_forced(balance); row=OLD.online_feature_row(p,selected,start,balance,blank); execute=forced
                if not forced:
                    pred=OLD.online_score(KIND,model,stats,sequences,name,row,start); execute=pred>0
                    own,_,_=AUDIT.on_policy_advantage(KIND,model,stats,0.,sequences,name,p,selected,balance,start,ref,vocab,blank)
                    pi1_audit.append({"score":pred,"advantage":own,"execute":execute})
                if execute: selected.append(start); balance-=1
    uniform=outputs["online_uniform"]; summaries={k:summarize(v,uniform) for k,v in outputs.items()}; old_rows=outputs["B2_old"]
    for key in ("B2_PI1_primary","B2_PI1_fit90_sensitivity"):
        s=summaries[key]; oldc=summaries["B2_old"]["metrics"]; s["delta_wer_pp_vs_B2_old"]=s["metrics"]["wer"]-oldc["wer"]; s["error_delta_vs_B2_old"]=s["metrics"]["error"]-oldc["error"]; s["paired_bootstrap_vs_B2_old"]=CHRON.paired_bootstrap([r["counts"] for r in outputs[key]],[r["counts"] for r in old_rows])
    gain=summaries["online_uniform"]["metrics"]["wer"]-summaries["rollout16_eager_oracle"]["metrics"]["wer"]
    for key in policies: summaries[key]["fraction_of_eager_rollout_oracle_WER_gap_recovered"]=(summaries["online_uniform"]["metrics"]["wer"]-summaries[key]["metrics"]["wer"])/gain if gain>0 else None
    labels=np.asarray([x["advantage"]>0 for x in pi1_audit]); adv=np.asarray([x["advantage"] for x in pi1_audit]); sc=np.asarray([x["score"] for x in pi1_audit]); act=np.asarray([x["execute"] for x in pi1_audit]); pi1_off=OLD.evaluation_metrics(labels,adv,sc,sc); pi1_off["action_positive_precision"]=float((act&labels).sum()/max(1,act.sum())); pi1_off["action_positive_recall"]=float((act&labels).sum()/max(1,labels.sum())); pi1_off["harmful_execute_count"]=int((act&(adv<0)).sum()); pi1_off["selected_signed_advantage_sum"]=float(adv[act].sum()); pi1_off["action_regret"]=float(np.maximum(adv,0).sum()-adv[act].sum())
    primary_summary=summaries["B2_PI1_primary"]; directional=primary_summary["metrics"]["wer"]<summaries["B2_old"]["metrics"]["wer"] and primary_summary["metrics"]["wer"]<summaries["online_uniform"]["metrics"]["wer"]; strong=primary_summary["delta_wer_pp_vs_online_uniform"]<=-.5 and primary_summary["paired_bootstrap_vs_online_uniform"]["ci95"][1]<0
    result={"scope":"one preregistered partial32 train-only B2 approximate policy-improvement smoke","gpu_used":False,"dev_used":False,"test_used":False,"policy_iterations_run":1,"fit_dataset":manifest["counts"],"training":{"history":history,"positive_class_weight":pos_weight,"primary_threshold":0.,"fit90_sensitivity_threshold":sensitivity},"held_out_A_pi0_offline":offline,"closed_loop":{"samples":len(uniform),"policies":summaries},"held_out_A_pi1_audit":pi1_off,"decision":{"directional_success":directional,"strong_go":strong,"stop_without_PI2":not directional,"rule":"PI1 primary must beat B2-old and online uniform; strong-go additionally <=-0.5pp and CI upper<0"},"limitations":["32/56 non-random train shards","train-calibration repeatedly inspected by prior smokes","future-aware reference labels are offline supervision only","no formal dev/test result","no real wall-time or submission-latency conclusion"],"runtime_seconds":time.perf_counter()-started}
    torch.save({"state_dict":model.state_dict(),"normalization_fit_only":stats,"variant":KIND,"policy_iteration":1},tmp/"B2_PI1.pt"); (tmp/"metrics.json").write_text(json.dumps(result,indent=2)+"\n"); (tmp/"manifest.json").write_text(json.dumps({"status":"complete","created_utc":datetime.now(timezone.utc).isoformat(),"metrics_sha256":sha256(tmp/"metrics.json"),"checkpoint_sha256":sha256(tmp/"B2_PI1.pt"),"dataset_manifest_sha256":sha256(data_root/"dataset_manifest.json")},indent=2)+"\n"); tmp.replace(output_root); return result

def main():
    p=argparse.ArgumentParser(); p.add_argument("mode",choices=["build","finalize","analyze"]); p.add_argument("--data-root",type=Path,default=DATA_ROOT); p.add_argument("--output-root",type=Path,default=OUTPUT_ROOT); p.add_argument("--worker",type=int,default=0); p.add_argument("--workers",type=int,default=1); a=p.parse_args()
    if a.mode=="build": build_worker(a.data_root.resolve(),a.worker,a.workers)
    elif a.mode=="finalize": finalize(a.data_root.resolve())
    else: print(json.dumps(analyze(a.data_root.resolve(),a.output_root.resolve()),indent=2))
if __name__=="__main__": main()
