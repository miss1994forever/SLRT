#!/usr/bin/env python3
"""Exploratory deterministic zero-lookahead coverage schedule family."""
import argparse,hashlib,importlib.util,json,os,sys
from datetime import datetime,timezone
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
FRESH_ROOT=ROOT/"results/phoenix-2014t_ISLR/p3_partial32_nonblank_fresh_disjoint_replication_v1_49faacc3"
OUTPUT_ROOT=ROOT/"results/phoenix-2014t_ISLR/p3_partial32_deterministic_coverage_family_exploratory_v1_49faacc3"
FAMILY={"fixed_1":(1,),"fixed_2":(2,),"fixed_3":(3,),"cycle_123":(1,2,3),"cycle_321":(3,2,1),"alternate_13":(1,3),"alternate_12":(1,2),"alternate_23":(2,3)}

def imp(n):
 p=ROOT/f"tools/{n}.py";s=importlib.util.spec_from_file_location(n,p);m=importlib.util.module_from_spec(s);sys.modules[s.name]=m;s.loader.exec_module(m);return m
REP=imp("analyze_phoenix_partial_nonblank_fresh_replication");BLOCK=REP.BLOCK;CHRON,BUILDER=REP.CHRON,REP.BUILDER

def sha(p):
 h=hashlib.sha256()
 with Path(p).open("rb") as f:
  for b in iter(lambda:f.read(1024*1024),b""):h.update(b)
 return h.hexdigest()

def config():
 fresh=json.loads((FRESH_ROOT/"resolved_config_preregistered.json").read_text())
 return {"experiment":"deterministic zero-lookahead coverage schedule family","classification":"exploratory because fresh evaluation was opened previously","created_before_family_calibration":True,"cpu_only":True,"split":fresh["split"],
  "family":{k:list(v) for k,v in FAMILY.items()},"protocol":{"hard_skeleton_offset":0,"one_precommitted_bonus_per_complete_4window_block":True,"tail_topup":False,"unknown_EOS":True,"target_rate":"approximately 50%","max_gap":4,"decoder":"fixed span15"},
  "causality":{"inputs":["completed block index only"],"algorithm_lookahead_frames":0,"future_reference_EOS_visual":False,"controller":"integer modulo and tuple lookup; near-zero but not formally benchmarked"},
  "selection":{"partition":"fresh calibration only","criterion":"minimum aggregate WER","tie_break":["minimum errors","prefer fixed_2","policy name lexical"]},"evaluation":"winner and fixed_2 only on already-opened fresh evaluation",
  "strong_go":"winner eval delta WER <= -0.5pp and paired CI upper < 0","forbidden":["GPU/CUDA","dev","test","family changes after evaluation","git commit"]}

def preregister(root):
 if root.exists():raise FileExistsError(root)
 root.mkdir(parents=True);x=config();(root/"resolved_config_preregistered.json").write_text(json.dumps(x,indent=2)+"\n");return x

def schedule(total,pattern):
 selected=[];block=0
 for s in range(0,total,4):
  selected.append(s)
  if s+3<total:selected.append(s+pattern[block%len(pattern)]);block+=1
 return sorted(selected)

def result(name,result,logits,pattern,vocab,blank):
 p=BUILDER.ORACLE.softmax_rows(np.asarray(logits));ref=BUILDER.REPLAY.clean_phoenix_2014_trans(result["gls_ref"]);selected=schedule(len(p),pattern);return {"sample_id":name,"selected":selected,"counts":BLOCK.decode_counts(p,selected,ref,vocab,blank),"dense_windows":len(p),"coverage":CHRON.coverage_metrics(selected,len(p))}

def selected_offset_counts(rows):
 counts={str(i):0 for i in range(4)}
 for row in rows:
  for start in row["selected"]:counts[str(start%4)]+=1
 return counts

def run(ids,policies):
 ids=set(ids);vocab=json.loads(CHRON.DEFAULT_VOCAB.read_text());blank=vocab.index("<blank>");out={k:[] for k in policies};dense=CHRON.DEFAULT_DENSE_ROOT;indices,n=CHRON.completed_shard_indices(dense)
 for shard in indices:
  results,logits,_=BUILDER.BUILDER.validate_dense_shard(dense,shard,n,verify_hashes=False)
  for name in results:
   if name not in ids:continue
   for key,pattern in policies.items():out[key].append(result(name,results[name],logits[name],pattern,vocab,blank))
 if any(len(x)!=len(ids) for x in out.values()):raise ValueError("coverage mismatch")
 return out

def summarize_all(outputs,center_key="fixed_2"):
 center=outputs[center_key];by={x["sample_id"]:x for x in center};out={}
 for key,rows in outputs.items():
  out[key]=BLOCK.summarize(rows,[by[x["sample_id"]] for x in rows])
  out[key]["selected_offset_counts"] = selected_offset_counts(rows)
 return out

def select(root):
 cfg=json.loads((root/"resolved_config_preregistered.json").read_text());
 if cfg!=config():raise ValueError("preregister changed")
 ids=cfg["split"]["partitions"]["calibration"]["sample_ids"];metrics=summarize_all(run(ids,FAMILY))
 def key(name):
  m=metrics[name]["metrics"];return (m["wer"],m["error"],0 if name=="fixed_2" else 1,name)
 winner=min(FAMILY,key=key);value={"status":"winner_frozen_before_family_evaluation","created_utc":datetime.now(timezone.utc).isoformat(),"winner":winner,"criterion":cfg["selection"],"calibration_samples":len(ids),"all_family":metrics,"config_sha256":sha(root/"resolved_config_preregistered.json")};(root/"selection_manifest.json").write_text(json.dumps(value,indent=2)+"\n");return value

def evaluate(root):
 cfg=json.loads((root/"resolved_config_preregistered.json").read_text());sel=json.loads((root/"selection_manifest.json").read_text());
 if sel["status"]!="winner_frozen_before_family_evaluation" or sel["config_sha256"]!=sha(root/"resolved_config_preregistered.json"):raise ValueError("winner not frozen")
 winner=sel["winner"];policies={"fixed_2":FAMILY["fixed_2"]};policies[winner]=FAMILY[winner];ids=cfg["split"]["partitions"]["evaluation"]["sample_ids"];metrics=summarize_all(run(ids,policies));w=metrics[winner];delta=w["delta_wer_pp_vs_structured_uniform"];ci=w["paired_bootstrap_vs_structured_uniform"]["ci95"]
 value={"scope":"exploratory deterministic coverage family on previously opened fresh evaluation","confirmatory":False,"gpu_used":False,"dev_used":False,"test_used":False,"winner":winner,"calibration_all_family":sel["all_family"],"evaluation":metrics,
  "decision":{"delta_wer_pp":delta,"ci95":ci,"strong_go":delta<=-.5 and ci[1]<0,"claim":"strong-go" if delta<=-.5 and ci[1]<0 else "uniform phase choice only; no success claim"},
  "causality_and_cost":{"algorithm_lookahead_frames":0,"controller":"precommitted pattern lookup from block index; near-zero, not formally benchmarked","unknown_EOS":True,"tail_topup":False},
  "limitations":["evaluation already opened by previous experiment","partial32 non-random shards","exploratory family selection","no real wall-time measurement"]}
 (root/"metrics.json").write_text(json.dumps(value,indent=2)+"\n");(root/"manifest.json").write_text(json.dumps({"status":"complete","created_utc":datetime.now(timezone.utc).isoformat(),"metrics_sha256":sha(root/"metrics.json")},indent=2)+"\n");return value

def main():
 p=argparse.ArgumentParser();p.add_argument("mode",choices=("preregister","select","evaluate"));p.add_argument("--output-root",type=Path,default=OUTPUT_ROOT);a=p.parse_args();os.environ["CUDA_VISIBLE_DEVICES"]="";r=a.output_root.resolve();x=preregister(r) if a.mode=="preregister" else select(r) if a.mode=="select" else evaluate(r);print(json.dumps(x,indent=2))
if __name__=="__main__":main()
