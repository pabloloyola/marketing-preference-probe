"""Frozen developmental split and resumable, guarded OpenRouter inference.

There are NO network calls without --live. The original 2k pool has already
informed retrieval design, so these are *development* splits, not a pristine
external test. Predictions never send human winner/click labels to the API.
"""
import hashlib
import json
import math
import os
import random
from pathlib import Path

from .api import MODELS, APIError, predict
from .dataset import load_pairs
from .retrieval_v2 import signature

DEFAULT_ROOT = "data/learned_correction"
DEFAULT_OUTPUT = "runs/learned_correction/predictions.jsonl"
PROMPT_VERSION = "headline_zero_v1"

def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def save_jsonl(path, rows):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text("".join(json.dumps(r,ensure_ascii=False)+"\n" for r in rows),encoding="utf8")

def prepare_split(pool="data/retrieval_v2/pool.jsonl",out=DEFAULT_ROOT,
                  train=1400,val=300,test=300,seed=26011):
    if min(train,val,test)<1:raise ValueError("Split sizes must be positive")
    source=Path(pool)
    rows=list(load_pairs(source))
    if train+val+test!=len(rows):
        raise ValueError("Requested %d cases but pool has %d; do not silently truncate the training pool"%
                         (train+val+test,len(rows)))
    ids=[str(r["test_id"]) for r in rows]
    sigs=[signature(r) for r in rows]
    if len(set(ids))!=len(rows) or len(set(sigs))!=len(rows):
        raise ValueError("Duplicate test ID or pair signature within source pool")
    order=list(range(len(rows)))
    random.Random(seed).shuffle(order)
    result={}
    cursor=0
    root=Path(out);root.mkdir(parents=True,exist_ok=True)
    for stage,n in (("train",train),("val",val),("test",test)):
        result[stage]=[rows[i] for i in order[cursor:cursor+n]]
        cursor+=n
    # Refuse overwriting a different split; protects previously scored outputs.
    manifest_path=root/"manifest.json"
    manifest={"source_sha256":sha256(source),"seed":seed,
              "sizes":{k:len(v) for k,v in result.items()},
              "development_only":True,"prompt_version":PROMPT_VERSION}
    for k,v in result.items():
        content="".join(json.dumps(r,ensure_ascii=False)+"\n" for r in v).encode()
        manifest[k+"_sha256"]=hashlib.sha256(content).hexdigest()
    if manifest_path.exists():
        old=json.loads(manifest_path.read_text())
        if old!=manifest:
            raise ValueError("Existing manifest differs. Use a different --out; do not mutate an existing experiment.")
        for k in result:
            if sha256(root/(k+".jsonl"))!=manifest[k+"_sha256"]:
                raise ValueError("Existing split "+k+" has been modified")
        return manifest
    for k,v in result.items():save_jsonl(root/(k+".jsonl"),v)
    manifest_path.write_text(json.dumps(manifest,indent=2)+"\n",encoding="utf8")
    return manifest

def load_split(root=DEFAULT_ROOT):
    root=Path(root)
    m=json.loads((root/"manifest.json").read_text())
    rows={k:list(load_pairs(root/(k+".jsonl"))) for k in ("train","val","test")}
    for k,items in rows.items():
        if len(items)!=m["sizes"][k] or sha256(root/(k+".jsonl"))!=m[k+"_sha256"]:
            raise ValueError("Split drift: "+k)
    if len({str(r["test_id"]) for group in rows.values() for r in group})!=sum(map(len,rows.values())):
        raise ValueError("Test ID overlap between splits")
    if len({signature(r) for group in rows.values() for r in group})!=sum(map(len,rows.values())):
        raise ValueError("Headline-pair overlap between splits")
    return rows,m

def fingerprint(manifest,models):
    payload={"source_sha256":manifest["source_sha256"],
             "split_hashes":[manifest[k+"_sha256"] for k in ("train","val","test")],
             "prompt_version":PROMPT_VERSION,
             "models":{k:MODELS[k][0] for k in models}}
    return hashlib.sha256(json.dumps(payload,sort_keys=True).encode()).hexdigest()

def infer(root=DEFAULT_ROOT,out=DEFAULT_OUTPUT,models=("jev","luna_decisions","luna_chat"),
          stages=("train","val","test"),limit=0,budget=.25,max_calls=100,
          reserve=.005,live=False):
    models=list(models);stages=list(stages)
    if (not models or not stages or len(set(models))!=len(models) or len(set(stages))!=len(stages)
        or any(m not in MODELS for m in models)
        or any(s not in ("train","val","test") for s in stages)
        or limit<0 or budget<=0 or max_calls<1 or reserve<=0):
        raise ValueError("Invalid inference settings")
    data,meta=load_split(root)
    path=Path(out);old=list(load_pairs(path)) if path.exists() else []
    # The data+prompt fingerprint is independent of selected models, allowing
    # model/stage subsets to resume into the SAME output file.
    data_fp=hashlib.sha256(json.dumps({
        "splits":[meta[k+"_sha256"] for k in ("train","val","test")],
        "prompt":PROMPT_VERSION},sort_keys=True).encode()).hexdigest()
    pairs={(stage,r["pair_id"]):r for stage,rows in data.items() for r in rows}
    done=set()
    for r in old:
        if (r.get("data_fingerprint")!=data_fp
            or r.get("prompt_version")!=PROMPT_VERSION
            or r.get("served_model_requested")!=MODELS.get(r.get("model_key"),(None,))[0]
            or (r.get("stage"),r.get("pair_id")) not in pairs):
            raise ValueError("Output mismatch. Existing predictions belong to another split, prompt or model.")
        key=(r["stage"],r["pair_id"],r["model_key"])
        if key in done:raise ValueError("Duplicate prior prediction "+str(key))
        done.add(key)
    tasks=[(stage,row,model) for stage in stages for row in data[stage] for model in models]
    pending=[(s,r,m) for s,r,m in tasks if (s,r["pair_id"],m) not in done]
    todo=pending[:limit] if limit else pending
    print("Planned %d tasks, %d still missing, %d requested this run; old rows %d"%
          (len(tasks),len(pending),len(todo),len(old)))
    if not live:
        print("DRY RUN: no OpenRouter calls. Use --live and explicit --budget/--max-calls.")
        return {"planned":len(tasks),"missing":len(pending),"requested":len(todo)}
    if not os.getenv("OPENROUTER_API_KEY"):
        raise ValueError("OPENROUTER_API_KEY missing; no calls sent")
    spent=sum((float(r["reported_cost_usd"])
               if r.get("reported_cost_usd") is not None else reserve) for r in old)
    successful=0
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open("a",encoding="utf8") as file:
        for stage,row,model in todo:
            # max_calls and budget guard the entire output, including previous calls.
            if len(old)+successful>=max_calls or spent+reserve>budget:
                print("Stopped at cumulative budget/call guardrail")
                break
            try:
                result=predict(model,row["headline_A"],row["headline_B"])
            except APIError as exc:
                print("Stopped on API error: "+str(exc));break
            # Human clicks and winners NEVER enter the request or saved prediction.
            result.update({"stage":stage,"pair_id":row["pair_id"],
                           "test_id":str(row["test_id"]),
                           "prompt_version":PROMPT_VERSION,
                           "data_fingerprint":data_fp,
                           "served_model_requested":MODELS[model][0],
                           "condition":"zero"})
            file.write(json.dumps(result,ensure_ascii=False)+"\n");file.flush()
            successful+=1
            cost=result.get("reported_cost_usd")
            spent+=max(0.,float(cost)) if cost is not None else reserve
            print("%d %s %s %s p_A=%.3f accounted=$%.5f"%
                  (successful,stage,model,row["pair_id"],result["p_A"],spent))
    return {"new_successful":successful,"total":len(old)+successful,
            "accounted_cost_usd":spent,"remaining_requested":len(todo)-successful}
