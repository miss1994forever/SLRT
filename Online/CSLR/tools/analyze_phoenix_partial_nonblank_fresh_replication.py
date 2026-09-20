#!/usr/bin/env python3
"""CPU-only fresh-disjoint partial32 replication of the frozen nonblank policy."""
import argparse,gzip,hashlib,importlib.util,json,os,pickle,random,sys,time
from collections import Counter
from datetime import datetime,timezone
from pathlib import Path
import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[1]
OLD_DATA=ROOT/"results/phoenix-2014t_ISLR/p3_partial32_terminal_predictor_smoke_v1_49faacc3_dataset"
AUDIT_ROOT=ROOT/"results/phoenix-2014t_ISLR/p3_partial32_dense_upgrade_target_audit_v1_49faacc3"
BLOCK_ROOT=ROOT/"results/phoenix-2014t_ISLR/p3_partial32_structured_block_smoke_v1_49faacc3"
SEQUENCES=Path("/tmp/phoenix_p2_train_center_sequences_v1.npz")
OUTPUT_ROOT=ROOT/"results/phoenix-2014t_ISLR/p3_partial32_nonblank_fresh_disjoint_replication_v1_49faacc3"
DATA_ROOT=OUTPUT_ROOT.with_name(OUTPUT_ROOT.name+"_train_dataset")
SPLIT_SEED="nonblank-fresh-replication-v1"

def imp(n):
 p=ROOT/f"tools/{n}.py";s=importlib.util.spec_from_file_location(n,p);m=importlib.util.module_from_spec(s);sys.modules[s.name]=m;s.loader.exec_module(m);return m
POL=imp("analyze_phoenix_partial_nonblank_closed_loop");AUDIT=POL.AUDIT;BLOCK=POL.BLOCK;OLD=POL.OLD;CHRON,BUILDER=POL.CHRON,POL.BUILDER

def sha(path):
 h=hashlib.sha256()
 with Path(path).open("rb") as f:
  for b in iter(lambda:f.read(1024*1024),b""):h.update(b)
 return h.hexdigest()

def source(x):return BUILDER.source_video(x)

def completed_input_names():
 dense=CHRON.DEFAULT_DENSE_ROOT;indices,n=CHRON.completed_shard_indices(dense);names=[]
 for i in indices:
  p=dense/"shards"/f"shard-{i:05d}-of-{n:05d}"/"train_metadata.pkl.gz"
  with gzip.open(p,"rb") as f:records=pickle.load(f)
  names.extend(str(x["name"]) for x in records)
 return sorted(set(names))

def split_spec():
 old=json.loads((OLD_DATA/"resolved_config_preregistered.json").read_text());oldfit=set(old["subsets"]["fit"]["sample_ids"]);oldeval=set(old["subsets"]["calibration"]["sample_ids"]);excluded_sources={source(x) for x in oldeval};available=completed_input_names();out={k:[] for k in ("fit","calibration","evaluation")};dropped=[];excluded=[]
 for x in available:
  if source(x) in excluded_sources:excluded.append(x);continue
  b=int(hashlib.sha256(f"{SPLIT_SEED}\0{source(x)}".encode()).hexdigest()[:8],16)%20;part="evaluation" if b<3 else "calibration" if b<6 else "fit"
  if x in oldfit and part!="fit":dropped.append(x);continue
  out[part].append(x)
 sources={k:{source(x) for x in v} for k,v in out.items()}
 if sources["fit"]&sources["calibration"] or sources["fit"]&sources["evaluation"] or sources["calibration"]&sources["evaluation"]:raise ValueError("source overlap")
 return {"rule":"source bucket = sha256(seed+NUL+source) mod20; 0-2 eval,3-5 cal,6-19 fit","seed":SPLIT_SEED,
  "partitions":{k:{"sample_ids":v,"samples":len(v),"sources":len(sources[k]),"ids_sha256":hashlib.sha256("\n".join(v).encode()).hexdigest()} for k,v in out.items()},
  "contamination_audit":{"old_evaluation_samples":len(oldeval),"old_evaluation_sources_excluded":len(excluded_sources),"samples_excluded_with_old_eval_sources":len(excluded),
   "old_fit_samples":len(oldfit),"old_fit_reused_in_new_fit":len(set(out["fit"])&oldfit),"old_fit_dropped_to_keep_out_of_fresh_cal_eval":len(dropped),"old_fit_in_fresh_eval":len(set(out["evaluation"])&oldfit),"old_eval_in_fresh_eval":len(set(out["evaluation"])&oldeval),"evaluation_outcomes_read_to_construct_split":False},
  "source_video_disjoint":True}

def config():
 ac=json.loads((AUDIT_ROOT/"resolved_config_preregistered.json").read_text())
 return {"experiment":"partial32 fresh-disjoint frozen nonblank replication","classification":"partial-data replication, not formal confirmatory","created_before_training_or_eval_outcomes":True,"cpu_only":True,"split":split_spec(),
  "frozen":{"target":"full_nonblank_confidence","preview":"31x17 causal pose/hand history + bookkeeping","architecture":ac["predictor"]["model"],"epochs":ac["predictor"]["epochs"],"seed":AUDIT.SEED,"optimizer":"AdamW(lr=1e-3,weight_decay=1e-4)","loss":"SmoothL1 fit-standardized target","policy":"argmax predicted target in offsets1/2/3"},
  "protocol":ac["block_protocol"],"lookahead":{"class":"bounded-lookahead-2","max_extra_video_frames":2},"evaluation_seal":"evaluation IDs known, but dense logits/reference/outcomes forbidden until model manifest is complete",
  "strong_go":"delta WER <= -0.5pp and paired bootstrap CI upper < 0","forbidden":["GPU/CUDA","dev","test","target reselection","hyperparameter tuning","evaluation access before model freeze","git commit"]}

def preregister(root,data):
 if root.exists() or data.exists():raise FileExistsError(root)
 x=config();root.mkdir(parents=True);data.mkdir(parents=True);(data/"workers").mkdir();(root/"resolved_config_preregistered.json").write_text(json.dumps(x,indent=2)+"\n");(data/"resolved_config_preregistered.json").write_text(json.dumps(x,indent=2)+"\n");return x

def loadcfg(root):
 x=json.loads((root/"resolved_config_preregistered.json").read_text());
 if x!=config():raise ValueError("preregister changed")
 return x

def worker(data,index,workers):
 os.environ["CUDA_VISIBLE_DEVICES"]="";torch.set_num_threads(1);cfg=loadcfg(data);ids=[]
 for part in ("fit","calibration"):ids.extend((x,part) for x in cfg["split"]["partitions"][part]["sample_ids"])
 assigned={x:p for pos,(x,p) in enumerate(ids) if pos%workers==index};final=data/"workers"/f"worker-{index:02d}-of-{workers:02d}.jsonl.gz";mark=final.with_suffix(".complete.json")
 if final.exists() and mark.exists() and json.loads(mark.read_text())["sha256"]==sha(final):return json.loads(mark.read_text())|{"resumed":True}
 vocab=json.loads(CHRON.DEFAULT_VOCAB.read_text());blank=vocab.index("<blank>");dense=CHRON.DEFAULT_DENSE_ROOT;indices,n=CHRON.completed_shard_indices(dense);tmp=final.with_name("."+final.name+f".incomplete-{os.getpid()}");done=blocks=0;started=time.perf_counter()
 with gzip.open(tmp,"wt",encoding="utf-8",compresslevel=5) as out:
  for shard in indices:
   results,logits,_=BUILDER.BUILDER.validate_dense_shard(dense,shard,n,verify_hashes=False)
   for name in results:
    if name not in assigned:continue
    value=BLOCK.oracle_sample(name,results[name],logits[name],assigned[name],vocab,blank);p=BUILDER.ORACLE.softmax_rows(np.asarray(logits[name]))
    for row in value["rows"]:
     targets=[float(1-p[c,blank]) for c in row["candidate_starts"]];out.write(json.dumps({"sample_id":name,"source_video_id":source(name),"partition":assigned[name],"block_start":row["block_start"],"candidate_starts":row["candidate_starts"],"features":row["features"],"target_full_nonblank_confidence":targets,"provenance":{"full_logits_label_only":True,"past_teacher_state":True,"evaluation_sample":False}},separators=(",",":"))+"\n");blocks+=1
    done+=1;print(json.dumps({"worker":index,"done":done,"assigned":len(assigned),"elapsed":time.perf_counter()-started}),flush=True)
 if done!=len(assigned):raise RuntimeError("coverage mismatch")
 tmp.replace(final);info={"status":"complete","worker":index,"workers":workers,"samples":done,"blocks":blocks,"seconds":time.perf_counter()-started,"sha256":sha(final),"bytes":final.stat().st_size};mark.write_text(json.dumps(info,indent=2)+"\n");return info

def finalize(data,workers):
 cfg=loadcfg(data);info=[]
 for i in range(workers):
  p=data/"workers"/f"worker-{i:02d}-of-{workers:02d}.jsonl.gz";m=json.loads(p.with_suffix(".complete.json").read_text());
  if sha(p)!=m["sha256"]:raise ValueError("hash mismatch")
  info.append(m)
 if sum(x["samples"] for x in info)!=cfg["split"]["partitions"]["fit"]["samples"]+cfg["split"]["partitions"]["calibration"]["samples"]:raise ValueError("count mismatch")
 x={"status":"complete","created_utc":datetime.now(timezone.utc).isoformat(),"workers":info,"config_sha256":sha(data/"resolved_config_preregistered.json")};(data/"dataset_manifest.json").write_text(json.dumps(x,indent=2)+"\n");return x

def rows(data):
 m=json.loads((data/"dataset_manifest.json").read_text());out={"fit":[],"calibration":[]}
 for w in m["workers"]:
  p=data/"workers"/f"worker-{w['worker']:02d}-of-{w['workers']:02d}.jsonl.gz"
  with gzip.open(p,"rt") as f:
   for line in f: x=json.loads(line);out[x["partition"]].append(x)
 return out

def matrix(rs,sequences):
 x=[];y=[]
 for r in rs:
  for j in range(3):x.append(AUDIT.preview_feature(r,j,sequences));y.append(r["target_full_nonblank_confidence"][j])
 return np.stack(x),np.asarray(y,np.float32)

def train(data,root):
 cfg=loadcfg(root)
 if (root/"evaluation_started.marker").exists():raise RuntimeError("evaluation already started")
 rs=rows(data);seq=OLD.SequenceFeatures(SEQUENCES);xf,yf=matrix(rs["fit"],seq);model,stats,history=POL.frozen_train(xf,yf);xc,yc=matrix(rs["calibration"],seq);xn=(xc-stats["mean"])/stats["scale"]
 with torch.no_grad():score=model(torch.from_numpy(xn).float()).numpy()*stats["target_scale"]+stats["target_mean"]
 offline={"rows":len(yc),"spearman":OLD.spearman(score,yc),"mae":float(np.abs(score-yc).mean())}
 torch.save({"state_dict":model.state_dict(),"stats":stats,"history":history,"config_sha256":sha(root/"resolved_config_preregistered.json")},root/"model.pt")
 manifest={"status":"model_frozen_before_evaluation","created_utc":datetime.now(timezone.utc).isoformat(),"model_sha256":sha(root/"model.pt"),"fit_samples":cfg["split"]["partitions"]["fit"]["samples"],"calibration_samples":cfg["split"]["partitions"]["calibration"]["samples"],"evaluation_outcomes_read":False,"calibration_offline":offline};(root/"training_manifest.json").write_text(json.dumps(manifest,indent=2)+"\n");return manifest

def loadmodel(root):
 m=json.loads((root/"training_manifest.json").read_text());
 if m["status"]!="model_frozen_before_evaluation" or sha(root/"model.pt")!=m["model_sha256"]:raise ValueError("model not frozen")
 c=torch.load(root/"model.pt",map_location="cpu");model=AUDIT.MLP(len(c["stats"]["mean"]));model.load_state_dict(c["state_dict"]);model.eval();return model,c["stats"]

def target_oracle(name,result,logits,vocab,blank):
 p=BUILDER.ORACLE.softmax_rows(np.asarray(logits));ref=BUILDER.REPLAY.clean_phoenix_2014_trans(result["gls_ref"]);selected=[];choices=Counter()
 for s in range(0,len(p),4):
  selected.append(s)
  if s+3>=len(p):continue
  cand=[s+1,s+2,s+3];vals=[1-p[c,blank] for c in cand];j=max(range(3),key=lambda z:(vals[z],z==1,-z));selected.append(cand[j]);choices[j+1]+=1
 return {"sample_id":name,"selected":selected,"counts":BLOCK.decode_counts(p,selected,ref,vocab,blank),"dense_windows":len(p),"coverage":CHRON.coverage_metrics(selected,len(p)),"choice_offsets":dict(choices)}

def evalrun(root):
 cfg=loadcfg(root);model,stats=loadmodel(root);(root/"evaluation_started.marker").write_text(datetime.now(timezone.utc).isoformat()+"\n");ids=set(cfg["split"]["partitions"]["evaluation"]["sample_ids"]);seq=OLD.SequenceFeatures(SEQUENCES);vocab=json.loads(CHRON.DEFAULT_VOCAB.read_text());blank=vocab.index("<blank>");pred=[];uni=[];targ=[];term=[];terminal_informative=0;dense=CHRON.DEFAULT_DENSE_ROOT;indices,n=CHRON.completed_shard_indices(dense)
 for shard in indices:
  results,logits,_=BUILDER.BUILDER.validate_dense_shard(dense,shard,n,verify_hashes=False)
  for name in results:
   if name not in ids:continue
   pred.append(POL.run_sample(name,results[name],logits[name],model,stats,seq,vocab,blank));targ.append(target_oracle(name,results[name],logits[name],vocab,blank));v=BLOCK.oracle_sample(name,results[name],logits[name],"evaluation",vocab,blank);term.append({"sample_id":name,**v["oracle"],"dense_windows":v["dense_windows"]});terminal_informative+=sum(len(set(r["label"]["terminal_errors"]))>1 for r in v["rows"]);uni.append({"sample_id":name,**v["uniform"],"dense_windows":v["dense_windows"]})
 if len(pred)!=len(ids):raise ValueError("eval coverage mismatch")
 by={x["sample_id"]:x for x in uni};match=lambda rs:[by[x["sample_id"]] for x in rs];summ={"fixed_center_uniform":BLOCK.summarize(uni,uni),"predicted_policy":BLOCK.summarize(pred,match(pred)),"full_nonblank_target_oracle":BLOCK.summarize(targ,match(targ)),"terminal_block_oracle":BLOCK.summarize(term,match(term))}
 def choices(rs):
  c=Counter();
  for x in rs:c.update(x.get("choice_offsets",{}))
  return {str(k):v for k,v in sorted(c.items())}
 wait=[]
 for x in pred:wait.extend(x["extra_wait_frames"])
 p=summ["predicted_policy"];ci=p["paired_bootstrap_vs_structured_uniform"]["ci95"];delta=p["delta_wer_pp_vs_structured_uniform"]
 result={"scope":"fresh-disjoint partial32 replication","confirmatory":False,"partial_data_replication":True,"gpu_used":False,"dev_used":False,"test_used":False,"split_audit":cfg["split"],"policies":summ,
  "effective_blocks":{"total_complete_blocks":len(wait),"terminal_informative_blocks":terminal_informative,"predictor_noncenter_blocks":sum(v for k,v in Counter([w for x in pred for w in []]).items()) if False else sum(v for k,v in Counter({int(k):v for k,v in choices(pred).items()}).items() if k!=2)},
  "choice_offsets":{"predicted":choices(pred),"full_nonblank_oracle":choices(targ)},"delay":{"definition":"offset3 availability minus chosen candidate availability","mean_extra_frames":float(np.mean(wait)),"max_extra_frames":int(max(wait)),"class":"bounded-lookahead-2"},
  "decision":{"delta_wer_pp":delta,"ci95":ci,"strong_go":delta<=-.5 and ci[1]<0,"directionally_better":delta<0,"claim":"strong-go" if delta<=-.5 and ci[1]<0 else "not successful"},
  "walltime_statement":"no end-to-end wall-time claim; cached preview excludes real detector/feature cost","limitations":["32/56 non-random partial train shards","target originated on old128","partial-data replication not formal confirmatory","cached detector cost excluded","bounded-lookahead-2"]}
 (root/"metrics.json").write_text(json.dumps(result,indent=2)+"\n");(root/"manifest.json").write_text(json.dumps({"status":"complete","created_utc":datetime.now(timezone.utc).isoformat(),"metrics_sha256":sha(root/"metrics.json"),"model_sha256":sha(root/"model.pt")},indent=2)+"\n");return result

def main():
 p=argparse.ArgumentParser();p.add_argument("mode",choices=("preregister","worker","finalize","train","evaluate"));p.add_argument("--output-root",type=Path,default=OUTPUT_ROOT);p.add_argument("--data-root",type=Path,default=DATA_ROOT);p.add_argument("--worker-index",type=int,default=0);p.add_argument("--workers",type=int,default=1);a=p.parse_args();os.environ["CUDA_VISIBLE_DEVICES"]="";r=a.output_root.resolve();d=a.data_root.resolve()
 if a.mode=="preregister":x=preregister(r,d)
 elif a.mode=="worker":x=worker(d,a.worker_index,a.workers)
 elif a.mode=="finalize":x=finalize(d,a.workers)
 elif a.mode=="train":x=train(d,r)
 else:x=evalrun(r)
 print(json.dumps(x,indent=2))
if __name__=="__main__":main()
