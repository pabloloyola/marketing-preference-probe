"""Experiment 3: text-only retrieval from disjoint historical training pairs."""
import json
import math
import os
import re
from collections import Counter, defaultdict
from pathlib import Path

from .api import MODELS, APIError, predict
from .dataset import load_pairs
from .experiment2 import score, shrinkage

def words(s):
    return re.findall(r"[a-z0-9]+",s.lower())

class Retriever:
    def __init__(self,train):
        self.train=list(train)
        if not self.train: raise ValueError("Empty training pool")
        n=len(self.train)
        df=Counter(t for r in self.train for t in set(words(r["headline_A"]+" "+r["headline_B"])))
        self.idf={t:math.log((n+1)/(v+1))+1 for t,v in df.items()}
        self.docs=[self.vector(r) for r in self.train]

    def vector(self,r):
        out=[]
        for field in ("headline_A","headline_B"):
            counts=Counter(words(r[field]))
            vec={w:(1+math.log(c))*self.idf[w] for w,c in counts.items() if w in self.idf}
            norm=math.sqrt(sum(v*v for v in vec.values())) or 1.
            out.append({w:v/norm for w,v in vec.items()})
        return out

    def find(self,query,k=5):
        if not 1<=k<=len(self.train): raise ValueError("Invalid k")
        x,y=self.vector(query)
        def dot(a,b): return sum(v*b.get(w,0) for w,v in a.items())
        candidates=[]
        for r,(a,b) in zip(self.train,self.docs):
            if str(r["test_id"])==str(query["test_id"]):continue
            direct=(dot(x,a)+dot(y,b))/2
            reverse=(dot(x,b)+dot(y,a))/2
            candidates.append((max(direct,reverse),reverse>direct,r))
        candidates.sort(key=lambda item:(-item[0],str(item[2]["pair_id"])))
        result=[]
        for sim,flipped,r in candidates[:k]:
            winner=r["winner"]
            if flipped: winner="B" if winner=="A" else "A"
            result.append({"headline_A":r["headline_B"] if flipped else r["headline_A"],
                           "headline_B":r["headline_A"] if flipped else r["headline_B"],
                           "observed_higher_ctr":winner,
                           "similarity":sim,"source_pair_id":r["pair_id"]})
        if len(result)<k:raise ValueError("Insufficient non-overlapping training pairs")
        return result

def prompt_examples(rows):
    return [{key:r[key] for key in ("headline_A","headline_B","observed_higher_ctr")} for r in rows]

def retrieval_baseline(rows):
    weights=[max(.01,x["similarity"]) for x in rows]
    a=sum(w for w,r in zip(weights,rows) if r["observed_higher_ctr"]=="A")
    return (1+a)/(2+sum(weights))

def get_splits(directory):
    return {name:list(load_pairs(Path(directory)/(name+".jsonl"))) for name in ("train","cal","test")}

def run3(directory="data/experiment2",out="runs/experiment3_predictions.jsonl",
         models=None,k=5,limit=0,live=False,budget=.5,max_calls=360):
    models=list(models or MODELS)
    if not set(models)<=set(MODELS) or k<1 or limit<0 or budget<=0 or max_calls<1:
        raise ValueError("Invalid run settings")
    split=get_splits(directory)
    retr=Retriever(split["train"])
    tasks=[(stage,p,model) for stage in ("cal","test")
           for p in (split[stage][:limit] if limit else split[stage])
           for model in models]
    path=Path(out)
    old=list(load_pairs(path)) if path.exists() else []
    if any(r.get("k")!=k for r in old):
        raise ValueError("Existing run has different --k: choose another --out")
    done={(r["stage"],r["pair_id"],r["model_key"]) for r in old if "p_A" in r}
    remaining=sum((s,p["pair_id"],m) not in done for s,p,m in tasks)
    print(f"Planned {len(tasks)} requests; {remaining} remaining; k={k}")
    if not live:
        print("DRY RUN, $0 spent. Use --live to send requests.");return
    if not os.getenv("OPENROUTER_API_KEY"):
        raise ValueError("OPENROUTER_API_KEY not set")
    reserve=.005
    spent=sum(float(r["reported_cost_usd"]) if r.get("reported_cost_usd") is not None else reserve for r in old)
    count=0
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open("a",encoding="utf-8") as stream:
        for stage,p,model in tasks:
            if (stage,p["pair_id"],model) in done:continue
            if spent+reserve>budget or len(old)+count>=max_calls:
                print("Stopped at best-effort budget or call limit");break
            matches=retr.find(p,k)
            try:
                result=predict(model,p["headline_A"],p["headline_B"],prompt_examples(matches))
            except APIError as e:
                print(f"API error, stopping: {e}");break
            result.update({"stage":stage,"pair_id":p["pair_id"],
                           "condition":"retrieved","k":k,
                           "neighbor_ids":[m["source_pair_id"] for m in matches]})
            stream.write(json.dumps(result,ensure_ascii=False)+"\n")
            stream.flush()
            spent+=max(0,float(result["reported_cost_usd"])) if result.get("reported_cost_usd") is not None else reserve
            count+=1
            print(f"{count:3} {stage} {model} p(A)={result['p_A']:.3f} cost USD {spent:.5f}")
    print(f"New requests completed: {count}; cumulative accounted USD {spent:.5f}")

def analyze3(directory="data/experiment2",
             retrieved="runs/experiment3_predictions.jsonl",
             previous="runs/experiment2_predictions.jsonl",
             out="runs/experiment3_report.json",k=5):
    split=get_splits(directory)
    retr=Retriever(split["train"])
    truth={stage:{r["pair_id"]:r for r in split[stage]} for stage in ("cal","test")}
    by=defaultdict(dict)
    for path,conditions in ((retrieved,{"retrieved"}),(previous,{"zero","fewshot"})):
        for r in load_pairs(path):
            if r.get("stage") not in truth or r.get("condition") not in conditions:
                continue
            if r["pair_id"] not in truth[r["stage"]]:continue
            if r["condition"]=="retrieved" and r.get("k")!=k:raise ValueError("k mismatch")
            key=r["model_key"],r["condition"],r["stage"]
            if r["pair_id"] in by[key]:raise ValueError("Duplicate prediction")
            by[key][r["pair_id"]]=r
    report={"split_sizes":{name:len(rows) for name,rows in split.items()},
            "models":{},"baselines":{},"k":k,
            "caveats":["Only compare systems evaluated on identical held-out pair IDs.",
                       "Calibration fits only on the calibration split.",
                       "Historical headline winners are not individual-level customer preferences."]}
    for name,func in (("retrieval_only",lambda p:retrieval_baseline(retr.find(p,k))),
                      ("constant_half",lambda p:.5)):
        preds={stage:[dict(p,p_A=func(p)) for p in split[stage]] for stage in ("cal","test")}
        alpha=shrinkage([p["p_A"] for p in preds["cal"]],
                        [int(p["winner"]=="A") for p in preds["cal"]])
        report["baselines"][name]={"raw_test":score(preds["test"]),"calibration_alpha":alpha,
            "calibrated_test":score([dict(p,p_A=.5+alpha*(p["p_A"]-.5)) for p in preds["test"]])}
    for model,condition,stage in sorted(by):
        if stage!="test":continue
        rs=by[(model,condition,stage)]
        cal=by.get((model,condition,"cal"),{})
        rows=[dict(truth["test"][pid],p_A=float(r["p_A"])) for pid,r in rs.items()]
        metrics={"coverage":f"{len(rows)}/{len(split['test'])}","raw_test":score(rows),
                 "test_cost_usd":sum(float(r.get("reported_cost_usd") or 0) for r in rs.values())}
        if len(cal)>=5:
            alpha=shrinkage([float(r["p_A"]) for r in cal.values()],
                            [int(truth["cal"][pid]["winner"]=="A") for pid in cal])
            metrics["calibration_alpha"]=alpha
            metrics["calibrated_test"]=score([dict(r,p_A=.5+alpha*(r["p_A"]-.5)) for r in rows])
            metrics["cal_coverage"]=f"{len(cal)}/{len(split['cal'])}"
        report["models"][model+"/"+condition]=metrics
    output=Path(out);output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps(report,indent=2),encoding="utf-8")
    return report

def preview3(directory="data/experiment2",k=5):
    split=get_splits(directory)
    q=split["test"][0]
    examples=Retriever(split["train"]).find(q,k)
    print(json.dumps({"target_headlines":[q["headline_A"],q["headline_B"]],
                      "retrieved_training_examples":examples},indent=2,ensure_ascii=False))
