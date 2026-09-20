#!/usr/bin/env python3
"""CPU-only audit of dense full-ISLR teacher targets for modality upgrading."""
import argparse, hashlib, importlib.util, json, math, os, random, sys, time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT=Path(__file__).resolve().parents[1]
BLOCK_ROOT=ROOT/"results/phoenix-2014t_ISLR/p3_partial32_structured_block_smoke_v1_49faacc3"
BLOCK_DATA=BLOCK_ROOT.with_name(BLOCK_ROOT.name+"_dataset")
PREVIEW_ROOT=ROOT/"results/phoenix-2014t_ISLR/p3_partial32_structured_block_preview_smoke_v1_49faacc3"
SEQUENCES=Path("/tmp/phoenix_p2_train_center_sequences_v1.npz")
OUTPUT_ROOT=ROOT/"results/phoenix-2014t_ISLR/p3_partial32_dense_upgrade_target_audit_v1_49faacc3"
TARGETS=("prefix_changed","prefix_edit_distance","prefix_entropy_reduction","full_margin","full_nonblank_confidence")
SEED,EPOCHS=261019,5

def imp(name):
 p=ROOT/f"tools/{name}.py";s=importlib.util.spec_from_file_location(name,p);m=importlib.util.module_from_spec(s);sys.modules[s.name]=m;s.loader.exec_module(m);return m
BLOCK=imp("analyze_phoenix_partial_structured_block");OLD=BLOCK.OLD;CHRON,BUILDER=BLOCK.CHRON,BLOCK.BUILDER

def sha256_file(path):
 h=hashlib.sha256()
 with Path(path).open("rb") as f:
  for b in iter(lambda:f.read(1024*1024),b""):h.update(b)
 return h.hexdigest()

def config():
 bc=json.loads((BLOCK_DATA/"resolved_config_preregistered.json").read_text())
 return {"experiment":"dense modality-upgrade teacher-target audit","created_before_outcome_computation":True,"cpu_only":True,"subsets":bc["subsets"],"block_protocol":bc["protocol"],
  "teacher_targets":{
   "prefix_changed":"adding full candidate changes fixed-decoder best prefix",
   "prefix_edit_distance":"token edit distance between decoder prefix before/after full candidate",
   "prefix_entropy_reduction":"mean entropy(past selected full logits)-mean entropy(after adding candidate)",
   "full_margin":"candidate full-softmax top1 minus top2 probability",
   "full_nonblank_confidence":"1-candidate full-softmax blank probability"},
  "teacher_only":["candidate full ISLR logits","decoder after action","future continuation","reference","EOS"],
  "deployable_predictor_inputs":["candidate bookkeeping","31-frame causal cheap pose/handshape preview ending at candidate s+8"],
  "predictor":{"model":"two-layer MLP","epochs":EPOCHS,"continuous_high_value_label":"strictly above fit 80th percentile","prefix_changed_label":">0"},
  "gate_A_alignment":["top target quintile terminal reward > bottom target quintile","equal-budget target block oracle error < fixed-center error"],
  "gate_B_predictability":["Spearman(prediction,target)>0.1","predicted top10% true target mean > overall mean","PR-AUC > high-target prevalence","recall@10% > 0.1"],
  "decision_availability":{"wait_for_offset3":True,"bounded_lookahead_video_frames":2,"unknown_EOS":True,"EOS_topup":False},
  "forbidden":["GPU/CUDA","dev","test","full logits/decoder-after-action/reference/future/EOS as predictor input","git commit"],
  "omitted_targets":{"sequence_loss":"no reliable frame-to-gloss alignment","cheap_vs_full_disagreement":"no cached cheap gloss-classification head"}}

def preregister(root):
 if root.exists():raise FileExistsError(root)
 root.mkdir(parents=True);x=config();(root/"resolved_config_preregistered.json").write_text(json.dumps(x,indent=2)+"\n");return x

def entropy_rows(p):return -np.sum(p*np.log(np.maximum(p,1e-12)),axis=1)

def target_values(p,selected,candidates,vocab,blank):
 before=BUILDER.ORACLE.decode_probabilities(p,selected,vocab,blank);past_entropy=float(entropy_rows(p[selected]).mean()) if selected else 0.;out={k:[] for k in TARGETS}
 for c in candidates:
  after=BUILDER.ORACLE.decode_probabilities(p,selected+[c],vocab,blank);dist=BUILDER.ORACLE.error_count(before,after);e=float(entropy_rows(p[[c]])[0]);new=(past_entropy*len(selected)+e)/(len(selected)+1);top=np.partition(p[c],-2)[-2:]
  out["prefix_changed"].append(float(dist>0));out["prefix_edit_distance"].append(float(dist));out["prefix_entropy_reduction"].append(float(past_entropy-new));out["full_margin"].append(float(top.max()-top.min()));out["full_nonblank_confidence"].append(float(1-p[c,blank]))
 return out

def preview_feature(row,j,sequences):
 b=np.asarray(row["features"][j]["bookkeeping"],np.float32);h=sequences.history(row["sample_id"],row["candidate_starts"][j]).astype(np.float32)
 return np.concatenate([b,h[-1],h.mean(0)]).astype(np.float32)

def enrich(blocks,samples):
 by_sample={split:defaultdict(list) for split in blocks}
 for split in blocks:
  for row in blocks[split]:by_sample[split][row["sample_id"]].append(row)
 ids=set(by_sample["fit"])|set(by_sample["calibration"]);targets={};target_schedules={k:[] for k in TARGETS};vocab=json.loads(CHRON.DEFAULT_VOCAB.read_text());blank=vocab.index("<blank>");dense=CHRON.DEFAULT_DENSE_ROOT;indices,n=CHRON.completed_shard_indices(dense)
 for shard in indices:
  results,logits,_=BUILDER.BUILDER.validate_dense_shard(dense,shard,n,verify_hashes=False)
  for name in results:
   if name not in ids:continue
   split="fit" if name in by_sample["fit"] else "calibration";p=BUILDER.ORACLE.softmax_rows(np.asarray(logits[name]));ref=BUILDER.REPLAY.clean_phoenix_2014_trans(results[name]["gls_ref"])
   selected=[]
   for row in sorted(by_sample[split][name],key=lambda x:x["block_start"]):
    selected.append(row["block_start"]);vals=target_values(p,selected,row["candidate_starts"],vocab,blank)
    for j,c in enumerate(row["candidate_starts"]):targets[(name,row["block_start"],j)]=dict((k,vals[k][j]) for k in TARGETS)
    selected.append(row["candidate_starts"][row["label"]["teacher_choice"]])
   if split=="calibration":
    for target in TARGETS:
     chosen=[]
     for s in range(0,len(p),4):
      chosen.append(s)
      if s+3>=len(p):continue
      cand=[s+1,s+2,s+3];values=target_values(p,chosen,cand,vocab,blank)[target];best=max(range(3),key=lambda j:(values[j],j==1,-j));chosen.append(cand[best])
     target_schedules[target].append({"sample_id":name,"selected":chosen,"counts":BLOCK.decode_counts(p,chosen,ref,vocab,blank),"dense_windows":len(p),"coverage":CHRON.coverage_metrics(chosen,len(p))})
 return targets,target_schedules

def alignment(rows,targets,target):
 values=[];rewards=[]
 for row in rows:
  center=row["label"]["terminal_errors"][1]
  for j,e in enumerate(row["label"]["terminal_errors"]):values.append(targets[(row["sample_id"],row["block_start"],j)][target]);rewards.append(center-e)
 v=np.asarray(values);r=np.asarray(rewards);order=np.argsort(v,kind="stable");k=max(1,int(math.ceil(.2*len(v))));rho=OLD.spearman(v,r)
 return {"candidate_rows":len(v),"target_summary":{"mean":float(v.mean()),"std":float(v.std()),"min":float(v.min()),"max":float(v.max()),"positive_count":int((v>0).sum()),"positive_prevalence":float((v>0).mean()),"unique_values":int(len(np.unique(v)))},
  "terminal_reward_spearman":rho,"bottom_quintile_mean_terminal_reward":float(r[order[:k]].mean()),"top_quintile_mean_terminal_reward":float(r[order[-k:]].mean()),"top_minus_bottom_reward":float(r[order[-k:]].mean()-r[order[:k]].mean())}

class MLP(nn.Module):
 def __init__(self,w):super().__init__();self.net=nn.Sequential(nn.Linear(w,48),nn.GELU(),nn.Linear(48,24),nn.GELU(),nn.Linear(24,1))
 def forward(self,x):return self.net(x).squeeze(1)

def train_predict(xfit,yfit,xcal,target):
 mean=xfit.mean(0);scale=xfit.std(0);scale[scale<1e-6]=1;xf=(xfit-mean)/scale;xc=(xcal-mean)/scale
 binary=target=="prefix_changed";threshold=0. if binary else float(np.quantile(yfit,.8));highfit=yfit>threshold;highcal=None
 ym,ys=(0.,1.) if binary else (float(yfit.mean()),float(yfit.std()) or 1.);torch.manual_seed(SEED);model=MLP(xf.shape[1]);opt=torch.optim.AdamW(model.parameters(),lr=1e-3,weight_decay=1e-4);g=torch.Generator().manual_seed(SEED)
 for epoch in range(EPOCHS):
  order=torch.randperm(len(xf),generator=g)
  for left in range(0,len(order),1024):
   idx=order[left:left+1024].numpy();pred=model(torch.from_numpy(xf[idx]).float())
   if binary:
    pos=float(highfit.sum());pw=min(20.,(len(highfit)-pos)/max(pos,1.));loss=F.binary_cross_entropy_with_logits(pred,torch.from_numpy(highfit[idx].astype(np.float32)),pos_weight=torch.tensor(pw))
   else:loss=F.smooth_l1_loss(pred,torch.from_numpy(((yfit[idx]-ym)/ys).astype(np.float32)))
   opt.zero_grad();loss.backward();opt.step()
 with torch.no_grad():score=model(torch.from_numpy(xc).float()).numpy()
 if not binary:score=score*ys+ym
 highcal=ycal_global>threshold; k=max(1,int(math.ceil(.1*len(score))));chosen=np.argsort(-score,kind="stable")[:k]
 return {"fit_high_threshold":threshold,"cal_rows":len(score),"high_rows":int(highcal.sum()),"high_prevalence":float(highcal.mean()),"pr_auc":OLD.average_precision(highcal,score),
  "recall_at_top10pct":float(highcal[chosen].sum()/highcal.sum()) if highcal.any() else None,"spearman_prediction_target":OLD.spearman(score,ycal_global),
  "overall_true_target_mean":float(ycal_global.mean()),"predicted_top10pct_true_target_mean":float(ycal_global[chosen].mean()),"top10_mean_enrichment":float(ycal_global[chosen].mean()-ycal_global.mean())}

ycal_global=None
def analyze(root):
 global ycal_global
 cfg=json.loads((root/"resolved_config_preregistered.json").read_text());
 if cfg!=config():raise ValueError("preregister changed")
 random.seed(SEED);np.random.seed(SEED);torch.manual_seed(SEED);torch.set_num_threads(min(8,os.cpu_count() or 1));torch.use_deterministic_algorithms(True)
 blocks,samples,_=BLOCK.load_rows(BLOCK_DATA);sequences=OLD.SequenceFeatures(SEQUENCES);targets,schedules=enrich(blocks,samples);uniform=[{**x["uniform"],"dense_windows":x["dense_windows"]} for x in samples["calibration"]];uniform_by_name={x["sample_id"]:{**x["uniform"],"dense_windows":x["dense_windows"]} for x in samples["calibration"]};uniform_summary=BLOCK.summarize(uniform,uniform)
 features={split:[] for split in blocks};ys={split:{k:[] for k in TARGETS} for split in blocks}
 for split in blocks:
  for row in blocks[split]:
   for j in range(3):
    features[split].append(preview_feature(row,j,sequences))
    for target in TARGETS:ys[split][target].append(targets[(row["sample_id"],row["block_start"],j)][target])
 features={k:np.stack(v) for k,v in features.items()};report={}
 for target in TARGETS:
  yf=np.asarray(ys["fit"][target],np.float32);ycal_global=np.asarray(ys["calibration"][target],np.float32);align=alignment(blocks["calibration"],targets,target);matched_uniform=[uniform_by_name[x["sample_id"]] for x in schedules[target]];schedule=BLOCK.summarize(schedules[target],matched_uniform);pred=train_predict(features["fit"],yf,features["calibration"],target)
  checksA={"top_reward_gt_bottom":align["top_minus_bottom_reward"]>0,"target_schedule_error_lt_center":schedule["metrics"]["error"]<uniform_summary["metrics"]["error"]};checksB={"spearman_gt_0.1":pred["spearman_prediction_target"]>.1,"top10_enrichment_gt_0":pred["top10_mean_enrichment"]>0,"pr_auc_gt_prevalence":pred["pr_auc"]>pred["high_prevalence"],"recall_at_10_gt_0.1":pred["recall_at_top10pct"]>.1}
  report[target]={"alignment":align,"equal_budget_target_oracle":schedule,"gate_A":{"passed":all(checksA.values()),"checks":checksA},"predictability":pred,"gate_B":{"passed":all(checksB.values()),"checks":checksB},"eligible_for_future_closed_loop":all(checksA.values()) and all(checksB.values())}
 result={"scope":"partial32 dense modality-upgrade target audit","gpu_used":False,"dev_used":False,"test_used":False,"structured_uniform":uniform_summary,"targets":report,
  "closed_loop_run":False,"closed_loop_rule":"this is a preregistered target audit; only A+B targets may proceed in a separate frozen experiment",
  "limitations":["32/56 non-random train shards","train calibration repeatedly inspected","fixed hard 128 subset","teacher targets use full ISLR outputs only as labels","predictor uses cached pose/hand preview whose detector runtime is excluded","bounded-lookahead-2, not zero-lookahead","no formal wall-time/latency claim"]}
 (root/"metrics.json").write_text(json.dumps(result,indent=2)+"\n");(root/"manifest.json").write_text(json.dumps({"status":"complete","created_utc":datetime.now(timezone.utc).isoformat(),"metrics_sha256":sha256_file(root/"metrics.json")},indent=2)+"\n");return result

def main():
 p=argparse.ArgumentParser();p.add_argument("mode",choices=("preregister","analyze"));p.add_argument("--output-root",type=Path,default=OUTPUT_ROOT);a=p.parse_args();os.environ["CUDA_VISIBLE_DEVICES"]="";root=a.output_root.resolve();x=preregister(root) if a.mode=="preregister" else analyze(root);print(json.dumps(x,indent=2))
if __name__=="__main__":main()
