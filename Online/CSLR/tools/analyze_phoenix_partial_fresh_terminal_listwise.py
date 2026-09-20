#!/usr/bin/env python3
"""Fresh-split scale diagnostic for the causal terminal block selector."""
import argparse, gzip, hashlib, importlib.util, json, os, random, sys, time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
FRESH_ROOT = ROOT / "results/phoenix-2014t_ISLR/p3_partial32_nonblank_fresh_disjoint_replication_v1_49faacc3"
OLD_BLOCK_DATA = ROOT / "results/phoenix-2014t_ISLR/p3_partial32_structured_block_smoke_v1_49faacc3_dataset"
OUTPUT_ROOT = ROOT / "results/phoenix-2014t_ISLR/p3_partial32_fresh_terminal_listwise_exploratory_v1_49faacc3"
DATA_ROOT = OUTPUT_ROOT.with_name(OUTPUT_ROOT.name + "_dataset")
SEQUENCES = Path("/tmp/phoenix_p2_train_center_sequences_v1.npz")
SEED, EPOCHS, BATCH, REGRET_WEIGHT = 261021, 20, 128, 0.25


def imp(name):
    path = ROOT / f"tools/{name}.py"; spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec); sys.modules[spec.name] = module; spec.loader.exec_module(module); return module


PREVIEW = imp("analyze_phoenix_partial_structured_block_preview")
BLOCK = PREVIEW.BLOCK; CHRON, BUILDER = BLOCK.CHRON, BLOCK.BUILDER


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""): h.update(chunk)
    return h.hexdigest()


def fresh_split():
    return json.loads((FRESH_ROOT / "resolved_config_preregistered.json").read_text())["split"]


def subset_512(sample_ids):
    ranked = sorted(sample_ids, key=lambda x: (hashlib.sha256(f"{SEED}\0{x}".encode()).hexdigest(), x))
    return ranked[:512]


def config():
    split = fresh_split(); fit_ids = split["partitions"]["fit"]["sample_ids"]
    return {
      "experiment": "fresh-split terminal block listwise scale diagnostic",
      "classification": "exploratory; evaluation partition was opened by earlier experiments",
      "created_before_terminal_dataset_generation": True, "cpu_only": True,
      "split": split,
      "fit_512_scaling_subset": {"rule": "lowest sha256(seed+NUL+sample_id)", "seed": SEED,
        "sample_ids": subset_512(fit_ids), "samples": 512},
      "known_scale_without_terminal_outcomes": {"fit_samples": 2541, "calibration_samples": 459,
        "evaluation_samples": 432, "fit_plus_calibration_complete_blocks": 86430,
        "prior_old_fit_plus_calibration_samples": 640, "prior_old_complete_blocks": 18688,
        "block_scale_factor": 86430/18688},
      "protocol": {"hard_skeleton_offset": 0, "one_bonus_offsets": [1,2,3],
        "decision": "when offset3 is available", "bounded_lookahead_candidate_arrivals": 2,
        "unknown_EOS": True, "tail_topup": False, "target_rate": "approximately 50%",
        "max_gap": 3, "decoder": "fixed span15"},
      "labels": {"target": "terminal edit errors for offsets1/2/3 followed by fixed-center continuation",
        "optimal_set_ties": True, "future_reference_EOS": "label generation only"},
      "inputs": {"T0": "bookkeeping only",
        "Tpreview": "bookkeeping + past-selected decoder prefix + each candidate's 31x17 causal pose/hand preview",
        "preview_visible_through": "candidate start + 8", "forbidden": ["full ISLR logits", "reference", "future", "EOS"]},
      "training": {"architecture": "old preview smoke shared 3-layer causal TCN plus 48/32 MLP",
        "optimizer": "AdamW(lr=1e-3, weight_decay=1e-4)", "epochs": EPOCHS, "batch": BATCH,
        "loss": "informative blocks only: uniform-optimal-set listwise CE + 0.25 expected normalized terminal regret",
        "neutral_blocks": "normalization/state-distribution statistics only", "seed": SEED,
        "models": ["T0_full2541", "Tpreview_hash512", "Tpreview_full2541"]},
      "calibration_gate": ["Tpreview_full2541 cumulative terminal reward versus fixed center > 0",
        "Tpreview_full2541 informative optimal-set accuracy > T0_full2541",
        "Tpreview_full2541 informative mean regret < T0_full2541"],
      "evaluation": "only if all calibration gates pass; frozen Tpreview_full2541 versus fixed center once",
      "strong_go": "eval delta WER <= -0.5pp and paired bootstrap CI upper < 0",
      "forbidden": ["GPU/CUDA", "dev", "test", "eval-dependent tuning", "git commit"]}


def preregister(data_root, output_root):
    if data_root.exists() or output_root.exists(): raise FileExistsError(data_root)
    data_root.mkdir(parents=True); (data_root/"workers").mkdir(); output_root.mkdir(parents=True)
    value=config()
    for root in (data_root, output_root): (root/"resolved_config_preregistered.json").write_text(json.dumps(value,indent=2)+"\n")
    return value


def loadcfg(root):
    value=json.loads((root/"resolved_config_preregistered.json").read_text())
    if value != config(): raise ValueError("preregistered configuration changed")
    return value


def worker(data_root, index, workers):
    os.environ["CUDA_VISIBLE_DEVICES"]=""; torch.set_num_threads(1); cfg=loadcfg(data_root)
    ordered=[(name,part) for part in ("fit","calibration") for name in cfg["split"]["partitions"][part]["sample_ids"]]
    assigned={name:part for pos,(name,part) in enumerate(ordered) if pos%workers==index}
    final=data_root/"workers"/f"worker-{index:02d}-of-{workers:02d}.jsonl.gz"; marker=final.with_suffix(".complete.json")
    if final.exists() and marker.exists() and json.loads(marker.read_text())["sha256"]==sha(final): return json.loads(marker.read_text())|{"resumed":True}
    if final.exists() or marker.exists(): raise FileExistsError(final)
    vocab=json.loads(CHRON.DEFAULT_VOCAB.read_text()); blank=vocab.index("<blank>"); dense=CHRON.DEFAULT_DENSE_ROOT
    indices,n=CHRON.completed_shard_indices(dense); temp=final.with_name("."+final.name+f".incomplete-{os.getpid()}")
    counts=Counter(); started=time.perf_counter()
    with gzip.open(temp,"wt",encoding="utf-8",compresslevel=5) as out:
        for shard in indices:
            results,logits,_=BUILDER.BUILDER.validate_dense_shard(dense,shard,n,verify_hashes=False)
            for name in results:
                if name not in assigned: continue
                value=BLOCK.oracle_sample(name,results[name],logits[name],assigned[name],vocab,blank)
                for row in value["rows"]:
                    out.write(json.dumps(row,separators=(",",":"))+"\n"); counts["blocks"]+=1
                    if len(set(row["label"]["terminal_errors"]))>1: counts["informative"]+=1
                counts[f"{assigned[name]}_samples"]+=1
                if sum(v for k,v in counts.items() if k.endswith("_samples"))%100==0:
                    print(json.dumps({"worker":index,"samples":sum(v for k,v in counts.items() if k.endswith("_samples")),"blocks":counts["blocks"],"seconds":time.perf_counter()-started}),flush=True)
    if sum(v for k,v in counts.items() if k.endswith("_samples"))!=len(assigned): raise RuntimeError("worker coverage mismatch")
    temp.replace(final); info={"status":"complete","worker":index,"workers":workers,"counts":dict(counts),"seconds":time.perf_counter()-started,"sha256":sha(final),"bytes":final.stat().st_size}
    marker.write_text(json.dumps(info,indent=2)+"\n"); return info


def finalize(data_root, workers):
    cfg=loadcfg(data_root); infos=[]; counts=Counter()
    for i in range(workers):
        path=data_root/"workers"/f"worker-{i:02d}-of-{workers:02d}.jsonl.gz"; mark=json.loads(path.with_suffix(".complete.json").read_text())
        if sha(path)!=mark["sha256"]: raise ValueError("worker hash mismatch")
        infos.append(mark); counts.update(mark["counts"])
    expected=cfg["split"]["partitions"]["fit"]["samples"]+cfg["split"]["partitions"]["calibration"]["samples"]
    if counts["fit_samples"]+counts["calibration_samples"]!=expected or counts["blocks"]!=86430: raise ValueError("dataset coverage mismatch")
    value={"status":"complete","created_utc":datetime.now(timezone.utc).isoformat(),"counts":dict(counts),"workers":infos,"config_sha256":sha(data_root/"resolved_config_preregistered.json")}
    (data_root/"dataset_manifest.json").write_text(json.dumps(value,indent=2)+"\n"); return value


def load_rows(data_root):
    manifest=json.loads((data_root/"dataset_manifest.json").read_text()); out={"fit":[],"calibration":[]}
    for info in manifest["workers"]:
        path=data_root/"workers"/f"worker-{info['worker']:02d}-of-{info['workers']:02d}.jsonl.gz"
        if sha(path)!=info["sha256"]: raise ValueError("hash mismatch")
        with gzip.open(path,"rt",encoding="utf-8") as f:
            for line in f:
                row=json.loads(line); out[row["partition"]].append(row)
    return out,manifest


def raw_arrays(rows,sequences,mode):
    full=mode.startswith("Tpreview"); base=[]; visual=[]; errors=[]; optimal=[]; informative=[]; sample=[]
    for row in rows:
        base.append(np.asarray([x["bookkeeping"]+(x["prefix"] if full else []) for x in row["features"]],np.float32))
        if full: visual.append(np.stack([sequences.history(row["sample_id"],s) for s in row["candidate_starts"]]))
        e=np.asarray(row["label"]["terminal_errors"],np.float32); errors.append(e); optimal.append(e==e.min()); informative.append(len(set(e.tolist()))>1); sample.append(row["sample_id"])
    return {"base":np.stack(base),"visual":np.stack(visual).astype(np.float16) if full else np.zeros((len(rows),3,31,1),np.float16),
      "errors":np.stack(errors),"optimal":np.stack(optimal).astype(np.float32),"informative":np.asarray(informative,bool),"sample":np.asarray(sample)}


def stats(data, mask):
    b=data["base"][mask]; bm=b.mean((0,1),dtype=np.float64).astype(np.float32); bs=b.std((0,1),dtype=np.float64).astype(np.float32);bs[bs<1e-6]=1
    if data["visual"].shape[-1]>1:
        v=data["visual"][mask]; vm=v.mean((0,1,2),dtype=np.float64).astype(np.float32);vs=v.std((0,1,2),dtype=np.float64).astype(np.float32);vs[vs<1e-6]=1
    else: vm=np.zeros(1,np.float32);vs=np.ones(1,np.float32)
    return {"base":(bm,bs),"visual":(vm,vs)}


def normalized(data, st):
    return {**data,"base":((data["base"]-st["base"][0])/st["base"][1]).astype(np.float32),
      "visual":((data["visual"]-st["visual"][0])/st["visual"][1]).astype(np.float16)}


def train_one(data, sample_ids, use_preview):
    sample_mask=np.isin(data["sample"],list(sample_ids)); st=stats(data,sample_mask); d=normalized(data,st); train_idx=np.flatnonzero(sample_mask & d["informative"])
    if not len(train_idx): raise ValueError("no informative training blocks")
    torch.manual_seed(SEED); model=PREVIEW.Selector(d["base"].shape[2],d["visual"].shape[3],use_preview); opt=torch.optim.AdamW(model.parameters(),lr=1e-3,weight_decay=1e-4); gen=torch.Generator().manual_seed(SEED); history=[]
    for epoch in range(EPOCHS):
        order=train_idx[torch.randperm(len(train_idx),generator=gen).numpy()]; total=0.
        for left in range(0,len(order),BATCH):
            idx=order[left:left+BATCH]; logits=model(torch.from_numpy(d["base"][idx]).float(),torch.from_numpy(d["visual"][idx]).float()); optimal=torch.from_numpy(d["optimal"][idx]).float(); target=optimal/optimal.sum(1,keepdim=True)
            errors=torch.from_numpy(d["errors"][idx]).float(); regret=errors-errors.min(1,keepdim=True).values; regret=regret/regret.max(1,keepdim=True).values.clamp_min(1.)
            ce=-(target*F.log_softmax(logits,1)).sum(1); expected=(F.softmax(logits,1)*regret).sum(1); loss=(ce+REGRET_WEIGHT*expected).mean();opt.zero_grad();loss.backward();opt.step();total+=float(loss)*len(idx)
        history.append(total/len(train_idx))
    model.eval(); return model,st,history,{"samples":int(len(set(data["sample"][sample_mask].tolist()))),"informative_blocks":int(len(train_idx)),"all_blocks":int(sample_mask.sum())}


def predict(model,data):
    out=[];start=time.perf_counter()
    with torch.no_grad():
        for left in range(0,len(data["base"]),1024):out.append(model(torch.from_numpy(data["base"][left:left+1024]).float(),torch.from_numpy(data["visual"][left:left+1024]).float()).numpy())
    return np.concatenate(out),time.perf_counter()-start


def subset_metrics(score,data,mask):
    e=data["errors"][mask]; chosen=np.argmax(score[mask],1); picked=e[np.arange(len(e)),chosen]; minimum=e.min(1);center=e[:,1]
    return {"blocks":int(len(e)),"optimal_set_accuracy":float(np.mean(picked==minimum)),"mean_terminal_error_regret":float(np.mean(picked-minimum)),
      "total_terminal_error_regret":int(np.sum(picked-minimum)),"cumulative_terminal_reward_vs_center":int(np.sum(center-picked)),
      "mean_true_reward_vs_center":float(np.mean(center-picked)),"choice_counts":{str(k):int(v) for k,v in zip(*np.unique(chosen,return_counts=True))}}


def report(model,raw,st):
    data=normalized(raw,st);score,elapsed=predict(model,data);return {"all":subset_metrics(score,data,np.ones(len(score),bool)),"informative":subset_metrics(score,data,data["informative"]),"cached_controller_seconds":elapsed}


def train_models(data_root, output_root):
    cfg=loadcfg(output_root)
    if (output_root/"evaluation_started.marker").exists(): raise RuntimeError("evaluation already started")
    rows,manifest=load_rows(data_root); seq=PREVIEW.PreviewArchive(SEQUENCES); fit_ids=set(cfg["split"]["partitions"]["fit"]["sample_ids"]); ids512=set(cfg["fit_512_scaling_subset"]["sample_ids"])
    raw_fit={"T0":raw_arrays(rows["fit"],seq,"T0"),"Tpreview":raw_arrays(rows["fit"],seq,"Tpreview")}; raw_cal={"T0":raw_arrays(rows["calibration"],seq,"T0"),"Tpreview":raw_arrays(rows["calibration"],seq,"Tpreview")}
    specs={"T0_full2541":("T0",fit_ids,False),"Tpreview_hash512":("Tpreview",ids512,True),"Tpreview_full2541":("Tpreview",fit_ids,True)}; reports={}; histories={}; scaling={}; hashes={}
    for label,(kind,ids,use) in specs.items():
        model,st,history,scale=train_one(raw_fit[kind],ids,use); reports[label]=report(model,raw_cal[kind],st); histories[label]=history; scaling[label]=scale
        path=output_root/f"{label}.pt";torch.save({"state_dict":model.state_dict(),"stats":st,"use_preview":use,"base_width":raw_fit[kind]["base"].shape[2]},path);hashes[label]=sha(path)
    base=reports["T0_full2541"]["informative"]; final=reports["Tpreview_full2541"]["informative"]
    checks={"cumulative_reward_gt_0":final["cumulative_terminal_reward_vs_center"]>0,"accuracy_gt_T0":final["optimal_set_accuracy"]>base["optimal_set_accuracy"],"regret_lt_T0":final["mean_terminal_error_regret"]<base["mean_terminal_error_regret"]};gate={"passed":all(checks.values()),"checks":checks}
    result={"scope":"fit/cal only terminal listwise scale diagnostic","dataset":manifest,"partition_informative_blocks":{"fit":int(raw_fit["T0"]["informative"].sum()),"calibration":int(raw_cal["T0"]["informative"].sum()),"evaluation_prior_opened_audit":139},"offline_calibration":reports,"training_history":histories,"sample_scaling":scaling,"gate":gate}
    (output_root/"calibration_metrics.json").write_text(json.dumps(result,indent=2)+"\n"); status={"status":"models_frozen_before_new_policy_evaluation","created_utc":datetime.now(timezone.utc).isoformat(),"gate":gate,"model_hashes":hashes,"calibration_metrics_sha256":sha(output_root/"calibration_metrics.json")};(output_root/"training_manifest.json").write_text(json.dumps(status,indent=2)+"\n")
    if not gate["passed"]:
        (output_root/"metrics.json").write_text(json.dumps({**result,"evaluation":None,"decision":"calibration gate failed; new-policy evaluation not read; stop current feature set"},indent=2)+"\n")
    return result


def load_full_model(output_root):
    manifest=json.loads((output_root/"training_manifest.json").read_text());
    if not manifest["gate"]["passed"]: raise RuntimeError("calibration gate failed; evaluation forbidden")
    path=output_root/"Tpreview_full2541.pt"
    if sha(path)!=manifest["model_hashes"]["Tpreview_full2541"]:raise ValueError("model hash mismatch")
    saved=torch.load(path,map_location="cpu");model=PREVIEW.Selector(saved["base_width"],len(saved["stats"]["visual"][0]),True);model.load_state_dict(saved["state_dict"]);model.eval();return model,saved["stats"]


def evaluate(output_root):
    cfg=loadcfg(output_root);model,st=load_full_model(output_root);(output_root/"evaluation_started.marker").write_text(datetime.now(timezone.utc).isoformat()+"\n");ids=set(cfg["split"]["partitions"]["evaluation"]["sample_ids"]);seq=PREVIEW.PreviewArchive(SEQUENCES);vocab=json.loads(CHRON.DEFAULT_VOCAB.read_text());blank=vocab.index("<blank>");pred=[];uniform=[];informative=0;choices=Counter();dense=CHRON.DEFAULT_DENSE_ROOT;indices,n=CHRON.completed_shard_indices(dense)
    for shard in indices:
        results,logits,_=BUILDER.BUILDER.validate_dense_shard(dense,shard,n,verify_hashes=False)
        for name in results:
            if name not in ids:continue
            x=PREVIEW.run_policy(name,results[name],logits[name],model,st,True,seq,vocab,blank);x["sample_id"]=name;pred.append(x);choices.update(s%4 for s in x["selected"] if s%4)
            value=BLOCK.oracle_sample(name,results[name],logits[name],"evaluation",vocab,blank);informative+=sum(len(set(r["label"]["terminal_errors"]))>1 for r in value["rows"]);uniform.append({"sample_id":name,**value["uniform"],"dense_windows":value["dense_windows"]})
    by={x["sample_id"]:x for x in uniform};summary=BLOCK.summarize(pred,[by[x["sample_id"]] for x in pred]);delta=summary["delta_wer_pp_vs_structured_uniform"];ci=summary["paired_bootstrap_vs_structured_uniform"]["ci95"]
    cal=json.loads((output_root/"calibration_metrics.json").read_text());result={**cal,"evaluation":{"samples":len(pred),"informative_blocks":informative,"predicted_policy":summary,"choice_offsets":{str(k):v for k,v in sorted(choices.items())}},"decision":{"strong_go":delta<=-.5 and ci[1]<0,"delta_wer_pp":delta,"ci95":ci,"claim":"strong-go" if delta<=-.5 and ci[1]<0 else "failed; stop direct learned scheduler on current feature set"},"limitations":["partial32 non-random shards","evaluation opened previously","bounded-lookahead-2","cached preview excludes detector wall time"]}
    (output_root/"metrics.json").write_text(json.dumps(result,indent=2)+"\n");(output_root/"manifest.json").write_text(json.dumps({"status":"complete","created_utc":datetime.now(timezone.utc).isoformat(),"metrics_sha256":sha(output_root/"metrics.json")},indent=2)+"\n");return result


def main():
    p=argparse.ArgumentParser();p.add_argument("mode",choices=("preregister","worker","finalize","train","evaluate"));p.add_argument("--data-root",type=Path,default=DATA_ROOT);p.add_argument("--output-root",type=Path,default=OUTPUT_ROOT);p.add_argument("--worker-index",type=int,default=0);p.add_argument("--workers",type=int,default=1);a=p.parse_args();os.environ["CUDA_VISIBLE_DEVICES"]="";d=a.data_root.resolve();o=a.output_root.resolve()
    if a.mode=="preregister":x=preregister(d,o)
    elif a.mode=="worker":x=worker(d,a.worker_index,a.workers)
    elif a.mode=="finalize":x=finalize(d,a.workers)
    elif a.mode=="train":x=train_models(d,o)
    else:x=evaluate(o)
    print(json.dumps(x,indent=2))


if __name__=="__main__":main()
