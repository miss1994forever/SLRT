#!/usr/bin/env python3
"""Post-selection exploratory closed loop for the frozen nonblank surrogate."""
import argparse, hashlib, importlib.util, json, os, random, sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT=Path(__file__).resolve().parents[1]
AUDIT_ROOT=ROOT/"results/phoenix-2014t_ISLR/p3_partial32_dense_upgrade_target_audit_v1_49faacc3"
BLOCK_ROOT=ROOT/"results/phoenix-2014t_ISLR/p3_partial32_structured_block_smoke_v1_49faacc3"
BLOCK_DATA=BLOCK_ROOT.with_name(BLOCK_ROOT.name+"_dataset")
SEQUENCES=Path("/tmp/phoenix_p2_train_center_sequences_v1.npz")
OUTPUT_ROOT=ROOT/"results/phoenix-2014t_ISLR/p3_partial32_nonblank_surrogate_closed_loop_exploratory_v1_49faacc3"

def imp(name):
 p=ROOT/f"tools/{name}.py";s=importlib.util.spec_from_file_location(name,p);m=importlib.util.module_from_spec(s);sys.modules[s.name]=m;s.loader.exec_module(m);return m
AUDIT=imp("analyze_phoenix_partial_dense_upgrade_targets");BLOCK=AUDIT.BLOCK;OLD=AUDIT.OLD;CHRON,BUILDER=AUDIT.CHRON,AUDIT.BUILDER

def sha256_file(path):
 h=hashlib.sha256()
 with Path(path).open("rb") as f:
  for b in iter(lambda:f.read(1024*1024),b""):h.update(b)
 return h.hexdigest()

def config():
 ac=json.loads((AUDIT_ROOT/"resolved_config_preregistered.json").read_text())
 return {"experiment":"post-selection exploratory full-nonblank surrogate closed loop","created_before_closed_loop":True,"cpu_only":True,
  "selection_bias":"target was selected using the same fixed 128 calibration samples; not confirmatory",
  "fresh_split_required_for_confirmation":True,"target":"full_nonblank_confidence","audit_metrics_sha256":sha256_file(AUDIT_ROOT/"metrics.json"),
  "subsets":ac["subsets"],"protocol":ac["block_protocol"],"decision":{"wait_for_offset3":True,"bounded_lookahead_video_frames":2,"unknown_EOS":True,"tail_topup":False},
  "frozen_model":{"inputs":ac["deployable_predictor_inputs"],"architecture":ac["predictor"]["model"],"epochs":ac["predictor"]["epochs"],"seed":AUDIT.SEED,"optimizer":"AdamW(lr=1e-3,weight_decay=1e-4)","loss":"SmoothL1 on fit-standardized nonblank confidence"},
  "policy":"choose predicted-highest candidate among offsets1/2/3; always retain offset0 skeleton",
  "stop_rule":"if predictor WER is not lower than fixed-center uniform, stop conditional-modality surrogate route",
  "forbidden":["GPU/CUDA","dev","test","full logits as deploy input","reference/future/EOS as deploy input","hyperparameter tuning","git commit"]}

def preregister(root):
 if root.exists():raise FileExistsError(root)
 root.mkdir(parents=True);x=config();(root/"resolved_config_preregistered.json").write_text(json.dumps(x,indent=2)+"\n");return x

def frozen_train(x,y):
 mean=x.mean(0);scale=x.std(0);scale[scale<1e-6]=1;xf=(x-mean)/scale;ym=float(y.mean());ys=float(y.std()) or 1.;torch.manual_seed(AUDIT.SEED);model=AUDIT.MLP(x.shape[1]);opt=torch.optim.AdamW(model.parameters(),lr=1e-3,weight_decay=1e-4);g=torch.Generator().manual_seed(AUDIT.SEED);history=[]
 for epoch in range(AUDIT.EPOCHS):
  order=torch.randperm(len(xf),generator=g);total=0.
  for left in range(0,len(order),1024):
   idx=order[left:left+1024].numpy();pred=model(torch.from_numpy(xf[idx]).float());loss=F.smooth_l1_loss(pred,torch.from_numpy(((y[idx]-ym)/ys).astype(np.float32)));opt.zero_grad();loss.backward();opt.step();total+=float(loss)*len(idx)
  history.append(total/len(xf))
 model.eval();return model,{"mean":mean,"scale":scale,"target_mean":ym,"target_scale":ys},history

def online_features(name,p,selected,candidates,blank,sequences):
 out=[]
 for c in candidates:
  row=OLD.online_feature_row(p,selected,c,1.,blank);b=OLD.bookkeeping_features(row);h=sequences.history(name,c).astype(np.float32);out.append(np.concatenate([b,h[-1],h.mean(0)]))
 return np.stack(out).astype(np.float32)

def run_sample(name,result,logits,model,stats,sequences,vocab,blank):
 p=BUILDER.ORACLE.softmax_rows(np.asarray(logits));ref=BUILDER.REPLAY.clean_phoenix_2014_trans(result["gls_ref"]);selected=[];choice=Counter();wait=[]
 for s in range(0,len(p),4):
  selected.append(s)
  if s+3>=len(p):continue
  candidates=[s+1,s+2,s+3];x=(online_features(name,p,selected,candidates,blank,sequences)-stats["mean"])/stats["scale"]
  with torch.no_grad():score=model(torch.from_numpy(x).float()).numpy()*stats["target_scale"]+stats["target_mean"]
  j=int(np.argmax(score));selected.append(candidates[j]);choice[j+1]+=1;wait.append(2-j)
 return {"sample_id":name,"selected":selected,"counts":BLOCK.decode_counts(p,selected,ref,vocab,blank),"dense_windows":len(p),"coverage":CHRON.coverage_metrics(selected,len(p)),"choice_offsets":dict(choice),"extra_wait_frames":wait}

def analyze(root):
 cfg=json.loads((root/"resolved_config_preregistered.json").read_text());
 if cfg!=config():raise ValueError("preregister changed")
 random.seed(AUDIT.SEED);np.random.seed(AUDIT.SEED);torch.manual_seed(AUDIT.SEED);torch.set_num_threads(min(8,os.cpu_count() or 1));torch.use_deterministic_algorithms(True)
 blocks,samples,_=BLOCK.load_rows(BLOCK_DATA);sequences=OLD.SequenceFeatures(SEQUENCES);targets,_=AUDIT.enrich(blocks,samples)
 x=[];y=[]
 for row in blocks["fit"]:
  for j in range(3):x.append(AUDIT.preview_feature(row,j,sequences));y.append(targets[(row["sample_id"],row["block_start"],j)]["full_nonblank_confidence"])
 model,stats,history=frozen_train(np.stack(x),np.asarray(y,np.float32))
 ids=set(cfg["subsets"]["calibration"]["sample_ids"]);vocab=json.loads(CHRON.DEFAULT_VOCAB.read_text());blank=vocab.index("<blank>");rows=[];dense=CHRON.DEFAULT_DENSE_ROOT;indices,n=CHRON.completed_shard_indices(dense)
 for shard in indices:
  results,logits,_=BUILDER.BUILDER.validate_dense_shard(dense,shard,n,verify_hashes=False)
  for name in results:
   if name in ids:rows.append(run_sample(name,results[name],logits[name],model,stats,sequences,vocab,blank))
 uniform=[{**x["uniform"],"dense_windows":x["dense_windows"]} for x in samples["calibration"]];byname={x["sample_id"]:{**x["uniform"],"dense_windows":x["dense_windows"]} for x in samples["calibration"]};matched=[byname[x["sample_id"]] for x in rows];predictor=BLOCK.summarize(rows,matched)
 choices=Counter();wait=[]
 for row in rows:choices.update(row["choice_offsets"]);wait.extend(row["extra_wait_frames"])
 prior=json.loads((BLOCK_ROOT/"metrics.json").read_text());audit=json.loads((AUDIT_ROOT/"metrics.json").read_text());u=prior["structured_uniform"];delta=predictor["delta_wer_pp_vs_structured_uniform"]
 result={"scope":"post-selection exploratory partial32 closed loop","confirmatory":False,"gpu_used":False,"dev_used":False,"test_used":False,
  "fixed_center_uniform":u,"predicted_nonblank_policy":predictor,"full_nonblank_target_oracle":audit["targets"]["full_nonblank_confidence"]["equal_budget_target_oracle"],"terminal_block_oracle":prior["structured_block_oracle"],
  "choice_and_delay":{"choice_offset_counts":{str(k):v for k,v in sorted(choices.items())},"blocks":len(wait),"extra_wait_frames_definition":"decision at offset3 availability minus chosen candidate availability","mean_extra_wait_frames":float(np.mean(wait)),"max_extra_wait_frames":int(max(wait)),"lookahead_class":"bounded-lookahead-2"},
  "training_loss":history,"decision":{"directionally_better_than_uniform":delta<0,"stop_surrogate_route":delta>=0,"next_if_better":"fresh source-video-disjoint split confirmation; no more tuning on fixed128"},
  "walltime_statement":"no end-to-end wall-time claim: cached preview excludes real pose/hand detector and feature extraction",
  "limitations":["target selected on same fixed128 (post-selection bias)","32/56 non-random train shards","calibration repeatedly inspected","bounded-lookahead-2","cached detector cost excluded","no dev/test or confirmatory claim"]}
 (root/"metrics.json").write_text(json.dumps(result,indent=2)+"\n");torch.save({"state_dict":model.state_dict(),"stats":stats,"history":history},root/"model.pt");(root/"manifest.json").write_text(json.dumps({"status":"complete","created_utc":datetime.now(timezone.utc).isoformat(),"metrics_sha256":sha256_file(root/"metrics.json"),"model_sha256":sha256_file(root/"model.pt")},indent=2)+"\n");return result

def main():
 p=argparse.ArgumentParser();p.add_argument("mode",choices=("preregister","analyze"));p.add_argument("--output-root",type=Path,default=OUTPUT_ROOT);a=p.parse_args();os.environ["CUDA_VISIBLE_DEVICES"]="";root=a.output_root.resolve();x=preregister(root) if a.mode=="preregister" else analyze(root);print(json.dumps(x,indent=2))
if __name__=="__main__":main()
