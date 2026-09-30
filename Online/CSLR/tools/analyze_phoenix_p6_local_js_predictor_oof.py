#!/usr/bin/env python3
"""CPU-only P6 local-JS source-disjoint OOF and equal-budget replay."""
import argparse, gzip, hashlib, importlib.util, json, os, sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[1]; BASE=ROOT/"results/phoenix-2014t_ISLR"
DENSE=BASE/"train_dense_stride1_v1_49faacc3"; UROOT=BASE/"p4_csc_l8_heatmap_u_identifiability_v2_49faacc3"
P6=BASE/"p6_local_js_information_gain_oracle_v1_49faacc3"; OUTPUT=BASE/"p6_local_js_predictor_oof_v1_49faacc3"
CONFIG=ROOT/"configs/p6_local_js_predictor_oof_v1_preregistered.json"; DATA=OUTPUT/"dataset.npz"; BOOT=261221

def imp(name):
 p=ROOT/f"tools/{name}.py"; s=importlib.util.spec_from_file_location(name,p); m=importlib.util.module_from_spec(s); sys.modules[s.name]=m; s.loader.exec_module(m); return m
ORACLE=imp("analyze_phoenix_p6_local_js_information_gain_oracle"); RANK=imp("analyze_phoenix_p5_h4_schedule_ranker_oof")
CHRON,BUILDER,BLOCK=ORACLE.CHRON,ORACLE.BUILDER,ORACLE.BLOCK
def now(): return datetime.now(timezone.utc).isoformat()
def sha(path):
 h=hashlib.sha256()
 with Path(path).open("rb") as f:
  for b in iter(lambda:f.read(8*1024*1024),b""): h.update(b)
 return h.hexdigest()
def source_fold(s): return int.from_bytes(hashlib.sha256(f"robust-oof-v1\0{s}".encode()).digest()[:8],"big")%5
def prepare():
 c=json.loads(CONFIG.read_text()); assets={"p6_oracle_metrics_sha256":P6/"metrics.json","p6_oracle_manifest_sha256":P6/"manifest.json","u_frames_sha256":UROOT/"u_frame_features_float32.npy","u_index_sha256":UROOT/"u_frame_feature_index.npz"}
 for k,p in assets.items():
  if sha(p)!=c["fixed_assets"][k]: raise RuntimeError(f"asset mismatch {k}")
 ip=OUTPUT/"preregistered_identity.json"
 if OUTPUT.exists() and any(OUTPUT.iterdir()):
  x=json.loads(ip.read_text())
  if x["config_sha256"]!=sha(CONFIG): raise RuntimeError("config changed")
  return x
 OUTPUT.mkdir(parents=True); x={"created_utc":now(),"config_sha256":sha(CONFIG),"script_sha256_at_prepare":sha(Path(__file__)),"outcomes_produced":False}; ip.write_text(json.dumps(x,indent=2)+"\n"); return x
def shuffle(v,name,index):
 seed=int.from_bytes(hashlib.sha256(f"p6-js-u-shuffle-v1\0{name}\0{index}".encode()).digest()[:8],"big"); return v[np.random.default_rng(seed).permutation(len(v))]
def build():
 prepare()
 if DATA.exists(): raise FileExistsError(DATA)
 with np.load(UROOT/"u_frame_feature_index.npz",allow_pickle=False) as z: names=[str(x) for x in z["video_names"].tolist()]; offs=np.asarray(z["video_offsets"],np.int64)
 ui={n:i for i,n in enumerate(names)}; uf=np.load(UROOT/"u_frame_features_float32.npy",mmap_mode="r")
 ps=RANK.projection(384,64,261131); po=RANK.projection(768,256,261132); basis=RANK.dct_basis()
 vocab=json.loads(CHRON.DEFAULT_VOCAB.read_text()); blank=vocab.index("<blank>"); rng=np.random.default_rng(261130); bucket=rng.integers(0,32,len(vocab)); sign=rng.choice(np.asarray([-1.,1.],np.float32),len(vocab))
 K=[];S=[];O=[];X=[]; y=[]; raw=[]; folds=[]; sources=[]; sample_idx=[]; block_idx=[]; sample_names=[]; smap={}; srcmap={}
 inds,nsh=CHRON.completed_shard_indices(DENSE)
 for sh in inds:
  results,logits,_=BUILDER.BUILDER.validate_dense_shard(DENSE,sh,nsh,verify_hashes=False)
  wanted=ORACLE.P5.FULL.fit_assignments()
  for name in results:
   if name not in wanted: continue
   probs=BUILDER.ORACLE.softmax_rows(np.asarray(logits[name])); fk=RANK.dense_frame_features(probs,blank,bucket,sign)
   sid=smap.setdefault(name,len(smap));
   if sid==len(sample_names): sample_names.append(name)
   source=BUILDER.source_video(name); srci=srcmap.setdefault(source,len(srcmap)); u=ui[name]; local=uf[int(offs[u]):int(offs[u+1])]
   selected=[]; prev=[]
   for bi,start in enumerate(ORACLE.complete_blocks(len(probs))):
    uh=RANK.u_history(local,start); summary=np.concatenate([uh[-1],uh.mean(0),uh.std(0),uh[-1]-uh[0]])@ps; ordered=(basis@uh).reshape(-1)@po; scrambled=(basis@shuffle(uh,name,bi)).reshape(-1)@po
    score=ORACLE.js_score(probs[start],probs[start+2]); K.append(RANK.k_history(fk,selected,start,prev)); S.append(summary); O.append(ordered); X.append(scrambled); raw.append(score); y.append(np.log(score+1e-8)); folds.append(source_fold(source)); sources.append(srci); sample_idx.append(sid); block_idx.append(bi)
    selected.extend([start,start+2]); prev.append(2)
  print(json.dumps({"built_shard":sh,"rows":len(y)}),flush=True)
 arr={"K":np.asarray(K,np.float32),"S":np.asarray(S,np.float32),"O":np.asarray(O,np.float32),"X":np.asarray(X,np.float32),"target":np.asarray(y,np.float32),"raw_js":np.asarray(raw,np.float32),"folds":np.asarray(folds,np.int8),"source_indices":np.asarray(sources,np.int16),"sample_indices":np.asarray(sample_idx,np.int16),"block_indices":np.asarray(block_idx,np.int16),"sample_names":np.asarray(sample_names)}
 if len(y)!=183401 or len(smap)!=6378 or len(srcmap)!=578: raise RuntimeError("scope mismatch")
 np.savez_compressed(DATA,**arr); m={"status":"complete","created_utc":now(),"rows":len(y),"samples":len(smap),"sources":len(srcmap),"shapes":{k:list(v.shape) for k,v in arr.items() if hasattr(v,"shape")},"dataset_sha256":sha(DATA),"outcomes_read":False}; (OUTPUT/"dataset_manifest.json").write_text(json.dumps(m,indent=2)+"\n"); return m
def r2(y,p,folds):
 sse=sst=0.
 for f in range(5):
  e=folds==f; base=float(y[~e].mean()); sse+=float(np.sum((y[e]-p[e])**2)); sst+=float(np.sum((y[e]-base)**2))
 return 1-sse/sst
def predict(x,y,folds):
 p=np.empty(len(y),np.float32)
 for f in range(5):
  t=folds!=f; e=~t; p[e]=RANK.ridge_predict(x[t],y[t,None],x[e],100.).ravel()
 return p
def bootstrap(rows,a,b,reps=10000):
 g=defaultdict(lambda:[0,0])
 for r in rows: g[r["source"]][0]+=r[a]-r[b]; g[r["source"]][1]+=r["ref_len"]
 vals=list(g.values()); er=np.asarray([v[0] for v in vals]); rf=np.asarray([v[1] for v in vals]); rng=np.random.default_rng(BOOT); out=np.empty(reps)
 for i in range(reps):
  q=rng.integers(0,len(vals),len(vals)); out[i]=100*er[q].sum()/rf[q].sum()
 return [float(np.quantile(out,.025)),float(np.quantile(out,.975))]
def run():
 prepare()
 if not DATA.exists(): raise RuntimeError("run --build first")
 if (OUTPUT/"metrics.json").exists(): raise FileExistsError("outcome exists")
 with np.load(DATA,allow_pickle=False) as z:
  k,s,o,x=(np.asarray(z[q],np.float32) for q in ("K","S","O","X")); y=np.asarray(z["target"]); folds=np.asarray(z["folds"]); si=np.asarray(z["sample_indices"]); names=[str(v) for v in z["sample_names"].tolist()]
 mats={"K":k,"KS":np.concatenate([k,s],1),"KO":np.concatenate([k,o],1),"KX":np.concatenate([k,x],1)}; pred={n:predict(v,y,folds) for n,v in mats.items()}; model={n:{"pooled_r2":r2(y,p,folds)} for n,p in pred.items()}
 bysample={i:np.flatnonzero(si==i) for i in range(len(names))}; chosen={}
 for n,p in pred.items(): chosen[n]={i:set(rows[np.lexsort((rows,-p[rows]))[:(len(rows)+1)//2]].tolist()) for i,rows in bysample.items()}
 vocab=json.loads(CHRON.DEFAULT_VOCAB.read_text()); blank=vocab.index("<blank>"); inds,nsh=CHRON.completed_shard_indices(DENSE); rowsout=[]; wanted=ORACLE.P5.FULL.fit_assignments(); name_to_i={n:i for i,n in enumerate(names)}
 for sh in inds:
  results,logits,_=BUILDER.BUILDER.validate_dense_shard(DENSE,sh,nsh,verify_hashes=False)
  for name in results:
   if name not in wanted: continue
   probs=BUILDER.ORACLE.softmax_rows(np.asarray(logits[name])); ref=BUILDER.REPLAY.clean_phoenix_2014_trans(results[name]["gls_ref"]); blocks=ORACLE.complete_blocks(len(probs)); sk=ORACLE.base_skeleton(len(probs)); count=(len(blocks)+1)//2; uni=sorted(sk+[blocks[j]+2 for j in ORACLE.uniform_bonus_indices(blocks,count)]); row={"sample":name,"source":BUILDER.source_video(name)}; uc=BLOCK.decode_counts(probs,uni,ref,vocab,blank); row.update(ref_len=int(uc["ref_len"]),uniform=int(uc["error"])); ii=name_to_i[name]; globalrows=bysample[ii]
   for n in pred:
    local=[j for j,grow in enumerate(globalrows) if int(grow) in chosen[n][ii]]; sel=sorted(sk+[blocks[j]+2 for j in local]); row[n]=int(BLOCK.decode_counts(probs,sel,ref,vocab,blank)["error"])
   rowsout.append(row)
 totals={q:sum(r[q] for r in rowsout) for q in ["uniform",*pred]}; ref=sum(r["ref_len"] for r in rowsout); comparisons={}
 for n in pred: comparisons[f"{n}_vs_uniform"]={"delta_errors":totals[n]-totals["uniform"],"delta_wer_pp":100*(totals[n]-totals["uniform"])/ref,"ci95":bootstrap(rowsout,n,"uniform")}
 for b in ("K","KX"): comparisons[f"KO_vs_{b}"]={"delta_errors":totals["KO"]-totals[b],"ci95":bootstrap(rowsout,"KO",b)}
 g=json.loads(CONFIG.read_text())["gates"]; checks={"KO_uniform_errors":totals["uniform"]-totals["KO"]>=g["KO_error_reduction_vs_uniform_at_least"],"KO_uniform_ci":comparisons["KO_vs_uniform"]["ci95"][1]<0,"KO_K_errors":totals["K"]-totals["KO"]>=g["KO_error_reduction_vs_K_at_least"],"KO_K_ci":comparisons["KO_vs_K"]["ci95"][1]<0,"KO_KX_errors":totals["KX"]-totals["KO"]>=g["KO_error_reduction_vs_KX_at_least"],"KO_KX_ci":comparisons["KO_vs_KX"]["ci95"][1]<0}
 metrics={"status":"complete","created_utc":now(),"gpu_used":False,"train_fit_only":True,"rows":len(y),"models":model,"errors":totals,"wer":{q:100*v/ref for q,v in totals.items()},"comparisons":comparisons,"checks":checks,"gate_passed":all(checks.values()),"decision":"unlock unknown-EOS simulation" if all(checks.values()) else "stop this P6 predictor identity before closed loop or held-out"}; (OUTPUT/"metrics.json").write_text(json.dumps(metrics,indent=2)+"\n"); np.savez_compressed(OUTPUT/"oof_predictions.npz",folds=folds,target=y,**pred); man={"status":"complete","config_sha256":sha(CONFIG),"identity_sha256":sha(OUTPUT/"preregistered_identity.json"),"dataset_sha256":sha(DATA),"metrics_sha256":sha(OUTPUT/"metrics.json"),"oof_sha256":sha(OUTPUT/"oof_predictions.npz"),"gate_passed":metrics["gate_passed"]}; (OUTPUT/"manifest.json").write_text(json.dumps(man,indent=2)+"\n"); return metrics
if __name__=="__main__":
 p=argparse.ArgumentParser(); p.add_argument("--prepare",action="store_true"); p.add_argument("--build",action="store_true"); p.add_argument("--run",action="store_true"); a=p.parse_args()
 if sum((a.prepare,a.build,a.run))!=1:p.error("choose one")
 os.environ["CUDA_VISIBLE_DEVICES"]=""; print(json.dumps(prepare() if a.prepare else build() if a.build else run(),indent=2))
