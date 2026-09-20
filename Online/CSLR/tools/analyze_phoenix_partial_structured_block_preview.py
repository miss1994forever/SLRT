#!/usr/bin/env python3
"""CPU-only causal candidate-preview smoke for structured block ranking."""
import argparse, hashlib, importlib.util, json, os, random, sys, time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
BLOCK_ROOT = ROOT / "results/phoenix-2014t_ISLR/p3_partial32_structured_block_smoke_v1_49faacc3"
BLOCK_DATA = BLOCK_ROOT.with_name(BLOCK_ROOT.name + "_dataset")
OUTPUT_ROOT = ROOT / "results/phoenix-2014t_ISLR/p3_partial32_structured_block_preview_smoke_v1_49faacc3"
SEQUENCES = Path("/tmp/phoenix_p2_train_center_sequences_v1.npz")
SEED, EPOCHS, INFORMATIVE_WEIGHT = 261018, 10, 50.0


def imp(name):
    path=ROOT/f"tools/{name}.py"; spec=importlib.util.spec_from_file_location(name,path)
    module=importlib.util.module_from_spec(spec); sys.modules[spec.name]=module; spec.loader.exec_module(module); return module


BLOCK=imp("analyze_phoenix_partial_structured_block")
OLD=BLOCK.OLD; CHRON,BUILDER=BLOCK.CHRON,BLOCK.BUILDER


def sha256_file(path):
    h=hashlib.sha256()
    with Path(path).open("rb") as f:
        for b in iter(lambda:f.read(1024*1024),b""): h.update(b)
    return h.hexdigest()


def config():
    block_config=json.loads((BLOCK_DATA/"resolved_config_preregistered.json").read_text())
    return {"experiment":"causal candidate preview + structured block ranking smoke",
      "created_before_outcome_computation":True,"cpu_only":True,
      "subsets_and_labels":"frozen structured-block dataset and split","subsets":block_config["subsets"],
      "protocol":block_config["protocol"],
      "preview":{"archive":str(SEQUENCES),"archive_sha256":sha256_file(SEQUENCES),
        "source":"train-only cached per-frame pose/handshape geometry and motion",
        "features":OLD.TEMPORAL_FEATURE_NAMES,"history_frames":31,
        "candidate_start_s_visible_through":"s+8", "decision_at":"block offset3 availability",
        "maximum_additional_wait_video_frames":2,"lookahead_class":"bounded-lookahead-2",
        "real_cost":"per-frame pose/hand detector + feature extraction + TCN/controller; not free",
        "smoke_cost_exclusion":"pose/hand detector already cached; measured controller time excludes detector"},
      "models":{"T0":"bookkeeping + past-selected decoder prefix",
        "Tpreview":"T0 + shared tiny causal TCN over each candidate 31x17 preview",
        "epochs":EPOCHS,"informative_block_weight":INFORMATIVE_WEIGHT,
        "loss":"weighted uniform-target cross entropy over minimum-error candidates",
        "same_seed_optimizer_epochs":True},
      "gate":["Tpreview cumulative terminal reward versus fixed center > 0",
        "Tpreview all-block optimal-set accuracy > T0",
        "Tpreview all-block mean terminal regret < T0"],
      "forbidden":["GPU/CUDA","dev","test","complete ISLR logits as predictor input","reference/future/EOS/terminal oracle output as input","git commit"]}


def preregister(root):
    if root.exists(): raise FileExistsError(root)
    root.mkdir(parents=True); value=config(); (root/"resolved_config_preregistered.json").write_text(json.dumps(value,indent=2)+"\n"); return value


class PreviewArchive(OLD.SequenceFeatures):
    pass


def raw_arrays(rows,sequences,use_preview):
    base=[]; visual=[]; errors=[]; optimal=[]; informative=[]
    for row in rows:
        base.append(np.asarray([f["bookkeeping"]+f["prefix"] for f in row["features"]],dtype=np.float32))
        if use_preview: visual.append(np.stack([sequences.history(row["sample_id"],s) for s in row["candidate_starts"]]))
        e=np.asarray(row["label"]["terminal_errors"],dtype=np.float32); errors.append(e); optimal.append(e==e.min()); informative.append(len(set(e.tolist()))>1)
    return {"base":np.stack(base),"visual":np.stack(visual).astype(np.float16) if use_preview else np.zeros((len(rows),3,31,1),np.float16),
            "errors":np.stack(errors),"optimal":np.stack(optimal).astype(np.float32),"informative":np.asarray(informative,dtype=bool)}


def normalization(data,use_preview):
    bm=data["base"].mean((0,1),dtype=np.float64).astype(np.float32); bs=data["base"].std((0,1),dtype=np.float64).astype(np.float32);bs[bs<1e-6]=1
    if use_preview:
        vm=data["visual"].mean((0,1,2),dtype=np.float64).astype(np.float32);vs=data["visual"].std((0,1,2),dtype=np.float64).astype(np.float32);vs[vs<1e-6]=1
    else: vm=np.zeros(1,np.float32);vs=np.ones(1,np.float32)
    return {"base":(bm,bs),"visual":(vm,vs)}


def normalize(data,stats):
    return {**data,"base":((data["base"]-stats["base"][0])/stats["base"][1]).astype(np.float32),
            "visual":((data["visual"]-stats["visual"][0])/stats["visual"][1]).astype(np.float16)}


class Selector(nn.Module):
    def __init__(self,base_width,visual_width,use_preview):
        super().__init__();self.use_preview=use_preview
        if use_preview:
            self.inp=nn.Conv1d(visual_width,16,1);self.layers=nn.ModuleList([nn.Conv1d(16,16,3,dilation=d) for d in (1,2,4)])
        self.scorer=nn.Sequential(nn.Linear(base_width+(16 if use_preview else 0),48),nn.GELU(),nn.Linear(48,32),nn.GELU(),nn.Linear(32,1))
    def encode(self,x):
        x=self.inp(x.transpose(1,2))
        for layer,d in zip(self.layers,(1,2,4)): x=F.gelu(layer(F.pad(x,(2*d,0))))+x
        return x[:,:,-1]
    def forward(self,base,visual):
        b,k,w=base.shape; parts=[base.reshape(b*k,w)]
        if self.use_preview: parts.append(self.encode(visual.reshape(b*k,visual.shape[2],visual.shape[3])))
        return self.scorer(torch.cat(parts,1)).reshape(b,k)


def train(data,use_preview):
    torch.manual_seed(SEED);model=Selector(data["base"].shape[2],data["visual"].shape[3],use_preview);opt=torch.optim.AdamW(model.parameters(),lr=1e-3,weight_decay=1e-4)
    target=data["optimal"]/data["optimal"].sum(1,keepdims=True);weights=np.where(data["informative"],INFORMATIVE_WEIGHT,1.).astype(np.float32);g=torch.Generator().manual_seed(SEED);history=[]
    for epoch in range(EPOCHS):
        order=torch.randperm(len(target),generator=g);total=0.
        for left in range(0,len(order),512):
            idx=order[left:left+512].numpy(); logits=model(torch.from_numpy(data["base"][idx]).float(),torch.from_numpy(data["visual"][idx]).float());t=torch.from_numpy(target[idx]).float();w=torch.from_numpy(weights[idx]);per=-(t*F.log_softmax(logits,1)).sum(1);loss=(per*w).sum()/w.sum();opt.zero_grad();loss.backward();opt.step();total+=float(loss)*len(idx)
        history.append(total/len(target))
    model.eval();return model,history


def predict(model,data):
    out=[];started=time.perf_counter()
    with torch.no_grad():
        for left in range(0,len(data["base"]),1024): out.append(model(torch.from_numpy(data["base"][left:left+1024]).float(),torch.from_numpy(data["visual"][left:left+1024]).float()).numpy())
    elapsed=time.perf_counter()-started;return np.concatenate(out),elapsed


def subset_metrics(scores,data,mask):
    score=scores[mask];errors=data["errors"][mask];chosen=np.argmax(score,1);picked=errors[np.arange(len(chosen)),chosen];minimum=errors.min(1);center=errors[:,1]
    return {"blocks":int(len(chosen)),"optimal_set_accuracy":float(np.mean(picked==minimum)),
      "mean_terminal_error_regret":float(np.mean(picked-minimum)),"total_terminal_error_regret":int(np.sum(picked-minimum)),
      "mean_true_reward_vs_center":float(np.mean(center-picked)),"cumulative_terminal_reward_vs_center":int(np.sum(center-picked)),
      "choice_counts":{str(k):int(v) for k,v in sorted(zip(*np.unique(chosen,return_counts=True)))}}


def metrics(scores,data,elapsed):
    return {"all":subset_metrics(scores,data,np.ones(len(scores),bool)),"informative":subset_metrics(scores,data,data["informative"]),
            "all_tie_blocks":int((~data["informative"]).sum()),"informative_blocks":int(data["informative"].sum()),
            "cached_controller_seconds":elapsed,"cached_controller_ms_per_block":1000*elapsed/len(scores)}


def feature_for_block(name,p,selected,candidates,blank,sequences,stats,use_preview):
    fake={"sample_id":name,"candidate_starts":candidates,"features":BLOCK.candidate_features(p,selected,candidates,blank),"label":{"terminal_errors":[0,0,0],"optimal_mask":[1,1,1]}}
    raw=raw_arrays([fake],sequences,use_preview);return normalize(raw,stats)


def run_policy(name,result,logits,model,stats,use_preview,sequences,vocab,blank):
    p=BUILDER.ORACLE.softmax_rows(np.asarray(logits));ref=BUILDER.REPLAY.clean_phoenix_2014_trans(result["gls_ref"]);selected=[]
    for s in range(0,len(p),4):
        selected.append(s)
        if s+3>=len(p):continue
        candidates=[s+1,s+2,s+3];d=feature_for_block(name,p,selected,candidates,blank,sequences,stats,use_preview);score,_=predict(model,d);selected.append(candidates[int(np.argmax(score[0]))])
    return {"selected":selected,"counts":BLOCK.decode_counts(p,selected,ref,vocab,blank),"dense_windows":len(p),"coverage":CHRON.coverage_metrics(selected,len(p))}


def analyze(root):
    cfg=json.loads((root/"resolved_config_preregistered.json").read_text());
    if cfg!=config():raise ValueError("preregister changed")
    random.seed(SEED);np.random.seed(SEED);torch.manual_seed(SEED);torch.set_num_threads(min(8,os.cpu_count() or 1));torch.use_deterministic_algorithms(True)
    blocks,samples,_=BLOCK.load_rows(BLOCK_DATA);sequences=PreviewArchive(SEQUENCES);models={};reports={};histories={}
    for label,use_preview in (("T0",False),("Tpreview",True)):
        fit_raw=raw_arrays(blocks["fit"],sequences,use_preview);stats=normalization(fit_raw,use_preview);fit=normalize(fit_raw,stats);model,history=train(fit,use_preview)
        cal=normalize(raw_arrays(blocks["calibration"],sequences,use_preview),stats);score,elapsed=predict(model,cal);reports[label]=metrics(score,cal,elapsed);models[label]=(model,stats,use_preview);histories[label]=history
    a,b=reports["T0"]["all"],reports["Tpreview"]["all"]
    checks={"Tpreview_cumulative_reward_vs_center_gt_0":b["cumulative_terminal_reward_vs_center"]>0,
      "Tpreview_accuracy_gt_T0":b["optimal_set_accuracy"]>a["optimal_set_accuracy"],
      "Tpreview_regret_lt_T0":b["mean_terminal_error_regret"]<a["mean_terminal_error_regret"]};gate={"passed":all(checks.values()),"checks":checks}
    closed=None
    if gate["passed"]:
        ids=set(cfg["subsets"]["calibration"]["sample_ids"]);vocab=json.loads(CHRON.DEFAULT_VOCAB.read_text());blank=vocab.index("<blank>");out={k:[] for k in models};dense=CHRON.DEFAULT_DENSE_ROOT;indices,n=CHRON.completed_shard_indices(dense)
        for shard in indices:
            results,logits,_=BUILDER.BUILDER.validate_dense_shard(dense,shard,n,verify_hashes=False)
            for name in results:
                if name not in ids:continue
                for label,(model,stats,use) in models.items():out[label].append(run_policy(name,results[name],logits[name],model,stats,use,sequences,vocab,blank))
        uniform=[{**x["uniform"],"dense_windows":x["dense_windows"]} for x in samples["calibration"]];closed={k:BLOCK.summarize(v,uniform) for k,v in out.items()}
    prior=json.loads((BLOCK_ROOT/"metrics.json").read_text());result={"scope":"partial32 causal candidate preview structured block smoke","gpu_used":False,"dev_used":False,"test_used":False,
      "preview_audit":cfg["preview"],"offline":reports,"training":histories,"gate":gate,"closed_loop":closed,
      "structured_uniform_reference":prior["structured_uniform"],"structured_block_oracle_reference":prior["structured_block_oracle"],
      "limitations":["32/56 non-random train shards","train calibration repeatedly inspected","fixed hard 128 subset","future/reference/EOS labels",
        "bounded-lookahead-2, not zero-lookahead","cached preview excludes pose/hand detector runtime","no formal wall-time/latency claim"]}
    (root/"metrics.json").write_text(json.dumps(result,indent=2)+"\n")
    for label,(model,stats,use) in models.items():torch.save({"state_dict":model.state_dict(),"stats":stats,"use_preview":use},root/f"{label}.pt")
    (root/"manifest.json").write_text(json.dumps({"status":"complete","created_utc":datetime.now(timezone.utc).isoformat(),"metrics_sha256":sha256_file(root/"metrics.json")},indent=2)+"\n");return result


def main():
    p=argparse.ArgumentParser();p.add_argument("mode",choices=("preregister","analyze"));p.add_argument("--output-root",type=Path,default=OUTPUT_ROOT);a=p.parse_args();os.environ["CUDA_VISIBLE_DEVICES"]="";root=a.output_root.resolve();x=preregister(root) if a.mode=="preregister" else analyze(root);print(json.dumps(x,indent=2))
if __name__=="__main__":main()
