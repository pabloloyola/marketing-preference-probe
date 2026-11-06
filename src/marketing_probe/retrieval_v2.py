"""Retrieval v2: larger leakage-checked corpus, local semantic embeddings and choice features.

The target's click counts and winner are never used in retrieval scoring.
Requires optional 'retrieval' extra only for semantic/hybrid modes.
"""
import hashlib
import json
import math
import os
import re
import tempfile
from collections import Counter
from pathlib import Path
from .dataset import prepare, load_pairs
from .experiment2 import score, shrinkage
from .experiment3 import Retriever, retrieval_baseline, prompt_examples
from .api import MODELS, predict, APIError

MODEL = "Qwen/Qwen3-Embedding-0.6B"
METHODS = ("tfidf", "structure", "semantic", "hybrid")

def signature(pair):
    return tuple(sorted((pair["headline_A"].strip().casefold(),
                         pair["headline_B"].strip().casefold())))

def make_pool(csv_path, split_dir="data/experiment2",
              exploratory="data/pairs.jsonl", output="data/retrieval_v2/pool.jsonl",
              size=2000, seed=3026):
    if size < 5: raise ValueError("size must be >=5")
    protected=[]
    for path in [exploratory]+[str(Path(split_dir)/(n+".jsonl")) for n in ("train","cal","test")]:
        protected.extend(load_pairs(path))
    excluded_tests={str(r["test_id"]) for r in protected}
    excluded_signatures={signature(r) for r in protected}
    # The archive's original helper samples without replacement by experiment ID.
    # Draw a large reproducible candidate sample, then filter protected cases.
    with tempfile.TemporaryDirectory() as td:
        tmp=Path(td)/"candidates.jsonl"
        prepare(csv_path,tmp,n=size+len(excluded_tests)+100,seed=seed)
        candidates=list(load_pairs(tmp))
    rows=[]; seen_tests=set(); seen_signatures=set()
    for r in candidates:
        tid=str(r["test_id"])
        sig=signature(r)
        if (tid in excluded_tests or tid in seen_tests or
                sig in excluded_signatures or sig in seen_signatures):
            continue
        rows.append(r)
        seen_tests.add(tid);seen_signatures.add(sig)
        if len(rows)==size: break
    if len(rows)!=size:
        raise ValueError("Only %s eligible independent pairs; requested %s" % (len(rows),size))
    dest=Path(output);dest.parent.mkdir(parents=True,exist_ok=True)
    dest.write_text("".join(json.dumps(r,ensure_ascii=False)+"\n" for r in rows),encoding="utf8")
    meta={"n":len(rows),"seed":seed,"protected_test_ids":len(excluded_tests),
          "pool_sha256":hashlib.sha256(dest.read_bytes()).hexdigest()}
    dest.with_suffix(".meta.json").write_text(json.dumps(meta,indent=2),encoding="utf8")
    return meta

def features(h):
    text=h.lower()
    words=re.findall(r"[a-z0-9']+",text)
    emotional={"love","death","died","heart","cry","sad","life","fear","hate",
               "hope","amazing","shocking","beautiful","terrible"}
    curiosity={"this","these","what","why","how","then","secret","never","watch",
               "surprising","believe","happened"}
    return [min(len(words),40)/40, min(len(h),200)/200,
            int("?" in h),int("!" in h),int(bool(re.search(r"\d",h))),
            sum(w in emotional for w in words)/max(1,len(words)),
            sum(w in curiosity for w in words)/max(1,len(words)),
            int(bool(re.search(r"\b(you|your)\b",text))),
            int(bool(re.search(r"\b(she|he|her|his)\b",text)))]

def difference(pair):
    a,b=features(pair["headline_A"]),features(pair["headline_B"])
    return [x-y for x,y in zip(a,b)]

def cosine(a,b):
    den=math.sqrt(sum(x*x for x in a)*sum(x*x for x in b))
    return sum(x*y for x,y in zip(a,b))/den if den else 0.

def semantic_cache(pool,queries,cache,model=MODEL,batch_size=16):
    """Cache pairs as normalized A/B vectors; require exact ordered fingerprint."""
    try:
        import numpy as np
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:
        raise RuntimeError("Install embeddings first: uv sync --extra retrieval") from exc
    allrows=pool+queries
    fingerprint=hashlib.sha256(json.dumps(
        [(r["pair_id"],r["headline_A"],r["headline_B"]) for r in allrows],
        ensure_ascii=False).encode()).hexdigest()
    target=Path(cache)
    if target.exists():
        with np.load(target,allow_pickle=False) as loaded:
            if str(loaded["fingerprint"])==fingerprint and str(loaded["model"])==model:
                return loaded["vectors"]
    encoder=SentenceTransformer(model,trust_remote_code=True)
    sentences=[h for r in allrows for h in (r["headline_A"],r["headline_B"])]
    vec=encoder.encode(sentences,batch_size=batch_size,normalize_embeddings=True,
                       show_progress_bar=True,convert_to_numpy=True)
    vec=np.asarray(vec,dtype="float32").reshape(len(allrows),2,-1)
    target.parent.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(target,vectors=vec,fingerprint=fingerprint,model=model)
    return vec

class RetrievalV2:
    def __init__(self,pool,queries=(),method="structure",cache="data/retrieval_v2/embeddings.npz",
                 model=MODEL):
        if method not in METHODS:raise ValueError("Unknown retrieval method")
        self.pool=list(pool)
        self.queries=list(queries)
        self.method=method
        self.lexical=Retriever(self.pool) if method=="tfidf" else None
        self.structure=[difference(p) for p in self.pool]
        self.semantic=None
        if method in ("semantic","hybrid"):
            self.semantic=semantic_cache(self.pool,self.queries,cache,model)
        self.query_ids={r["pair_id"]:i for i,r in enumerate(self.queries)}

    def find(self,q,k=5):
        if self.method=="tfidf":
            return self.lexical.find(q,k)
        if k<1 or k>len(self.pool):raise ValueError("Invalid k")
        if self.semantic is not None:
            if q["pair_id"] not in self.query_ids:
                raise ValueError("Query absent from embedding cache; construct retriever with all cal/test queries")
            qsem=self.semantic[len(self.pool)+self.query_ids[q["pair_id"]]]
        qdiff=difference(q)
        results=[]
        for i,p in enumerate(self.pool):
            if str(p["test_id"])==str(q["test_id"]) or signature(p)==signature(q):
                continue
            structure=cosine(qdiff,self.structure[i])
            if self.semantic is None:
                forward=structure; backward=-structure
            else:
                old=self.semantic[i]
                forward=(float(qsem[0]@old[0])+float(qsem[1]@old[1]))/2
                backward=(float(qsem[0]@old[1])+float(qsem[1]@old[0]))/2
                if self.method=="hybrid":
                    forward=.5*forward+.5*structure
                    backward=.5*backward-.5*structure
            reverse=backward>forward
            results.append((max(forward,backward),reverse,p))
        results.sort(key=lambda v:(-v[0],str(v[2]["pair_id"])))
        out=[]
        for similarity,reverse,p in results[:k]:
            winner=p["winner"]
            if reverse:winner="B" if winner=="A" else "A"
            out.append({"headline_A":p["headline_B"] if reverse else p["headline_A"],
                        "headline_B":p["headline_A"] if reverse else p["headline_B"],
                        "observed_higher_ctr":winner,"source_pair_id":p["pair_id"],
                        "similarity":float(similarity)})
        if len(out)!=k: raise ValueError("Insufficient candidates")
        return out

def load_verified(pool_path,split_dir):
    pool=list(load_pairs(pool_path))
    split={s:list(load_pairs(Path(split_dir)/(s+".jsonl"))) for s in ("train","cal","test")}
    ids={str(r["test_id"]) for s in split.values() for r in s}
    sigs={signature(r) for s in split.values() for r in s}
    if len({str(r["test_id"]) for r in pool})!=len(pool):
        raise ValueError("Duplicate test IDs in retrieval pool")
    if any(str(r["test_id"]) in ids or signature(r) in sigs for r in pool):
        raise ValueError("Retrieval corpus overlaps train/cal/test data")
    return pool,split

def preview(pool_path="data/retrieval_v2/pool.jsonl",split_dir="data/experiment2",
            method="structure",k=5,cache="data/retrieval_v2/embeddings.npz"):
    pool,split=load_verified(pool_path,split_dir)
    queries=split["cal"]+split["test"]
    finder=RetrievalV2(pool,queries,method,cache)
    q=split["test"][0]
    return {"method":method,"pool_size":len(pool),
            "target_headlines":[q["headline_A"],q["headline_B"]],
            "retrieved_training_examples":finder.find(q,k)}

def run(pool_path="data/retrieval_v2/pool.jsonl",split_dir="data/experiment2",
        method="structure",k=5,limit=0,models=None,
        out=None,cache="data/retrieval_v2/embeddings.npz",
        budget=.5,max_calls=360,live=False):
    if method not in METHODS or k<1 or limit<0 or budget<=0 or max_calls<1:
        raise ValueError("Invalid run parameters")
    models=list(models or MODELS)
    if not set(models)<=set(MODELS):raise ValueError("Invalid models")
    path=Path(out or ("runs/retrieval_v2_"+method+".jsonl"))
    pool,split=load_verified(pool_path,split_dir)
    tasks=[(stage,row,model) for stage in ("cal","test")
           for row in (split[stage][:limit] if limit else split[stage]) for model in models]
    past=list(load_pairs(path)) if path.exists() else []
    pool_hash=hashlib.sha256(Path(pool_path).read_bytes()).hexdigest()
    if any(r.get("method")!=method or r.get("k")!=k or
           r.get("pool_hash")!=pool_hash for r in past):
        raise ValueError("Existing output belongs to another method/k/pool; use another --out")
    done={(r["stage"],r["pair_id"],r["model_key"]) for r in past}
    remaining=sum((s,r["pair_id"],m) not in done for s,r,m in tasks)
    print("Planned %d, remaining %d, retrieval pool %d" % (len(tasks),remaining,len(pool)))
    if not live:
        print("DRY RUN; no embeddings loaded or API calls made.");return
    if not os.environ.get("OPENROUTER_API_KEY"):
        raise ValueError("OPENROUTER_API_KEY missing")
    queries=split["cal"]+split["test"]
    finder=RetrievalV2(pool,queries,method,cache)
    reserve=.005
    spent=sum(float(r["reported_cost_usd"]) if r.get("reported_cost_usd") is not None
              else reserve for r in past)
    path.parent.mkdir(parents=True,exist_ok=True)
    count=0
    with path.open("a",encoding="utf8") as f:
        for stage,row,model in tasks:
            if (stage,row["pair_id"],model) in done:continue
            if spent+reserve>budget or len(past)+count>=max_calls:
                print("Best-effort budget or call guard reached");break
            neighbors=finder.find(row,k)
            try:
                res=predict(model,row["headline_A"],row["headline_B"],prompt_examples(neighbors))
            except APIError as e:
                print("API request stopped: %s" % e);break
            res.update({"stage":stage,"pair_id":row["pair_id"],"condition":"retrieval_v2",
                        "method":method,"k":k,"pool_hash":pool_hash,
                        "neighbor_ids":[r["source_pair_id"] for r in neighbors]})
            f.write(json.dumps(res,ensure_ascii=False)+"\n");f.flush()
            count+=1
            spent+=float(res["reported_cost_usd"]) if res.get("reported_cost_usd") is not None else reserve
            print("%3d %s %s %s p_A=%.3f accounted=$%.5f" % (
                count,stage,model,method,res["p_A"],spent))
    print("New successful calls: %d; total accounted $%.5f" % (count,spent))

def analyze(pool_path="data/retrieval_v2/pool.jsonl",split_dir="data/experiment2",
            method="structure",k=5,predictions=None,previous="runs/experiment2_predictions.jsonl",
            out=None,cache="data/retrieval_v2/embeddings.npz"):
    pool,split=load_verified(pool_path,split_dir)
    queries=split["cal"]+split["test"]
    finder=RetrievalV2(pool,queries,method,cache)
    truth={s:{r["pair_id"]:r for r in split[s]} for s in ("cal","test")}
    path=Path(predictions or ("runs/retrieval_v2_"+method+".jsonl"))
    records=list(load_pairs(path)) if path.exists() else []
    pool_hash=hashlib.sha256(Path(pool_path).read_bytes()).hexdigest()
    if any(r.get("method")!=method or r.get("k")!=k or r.get("pool_hash")!=pool_hash for r in records):
        raise ValueError("Prediction metadata mismatch")
    from collections import defaultdict
    allpred=defaultdict(dict)
    for source,conditions in ((records,{"retrieval_v2"}),
                               (list(load_pairs(previous)),{"zero","fewshot"})):
        for r in source:
            stage=r.get("stage")
            if stage not in truth or r.get("condition") not in conditions or r["pair_id"] not in truth[stage]:
                continue
            key=(r["model_key"],r["condition"],stage)
            if r["pair_id"] in allpred[key]:raise ValueError("Duplicate prediction")
            allpred[key][r["pair_id"]]=r
    report={"method":method,"k":k,"pool_size":len(pool),"models":{},
            "baselines":{},"heldout_test_n":len(split["test"])}
    for label,predictor in (("retrieval_only",lambda row:retrieval_baseline(finder.find(row,k))),
                            ("constant_half",lambda row:.5)):
        data={s:[dict(r,p_A=predictor(r)) for r in split[s]] for s in ("cal","test")}
        alpha=shrinkage([r["p_A"] for r in data["cal"]],
                        [int(r["winner"]=="A") for r in data["cal"]])
        report["baselines"][label]={"raw_test":score(data["test"]),
            "calibrated_test":score([dict(r,p_A=.5+alpha*(r["p_A"]-.5)) for r in data["test"]]),
            "calibration_alpha":alpha}
    for model,condition,stage in sorted(allpred):
        if stage!="test":continue
        test=allpred[(model,condition,"test")]
        cal=allpred.get((model,condition,"cal"),{})
        rows=[dict(truth["test"][pid],p_A=float(r["p_A"])) for pid,r in test.items()]
        metrics={"test_coverage":str(len(rows))+"/"+str(len(split["test"])),
                 "raw_test":score(rows),"calibration_coverage":str(len(cal))+"/"+str(len(split["cal"])),
                 "test_cost_usd":sum(float(r.get("reported_cost_usd") or 0) for r in test.values())}
        if len(cal)>=5:
            alpha=shrinkage([float(r["p_A"]) for r in cal.values()],
                            [int(truth["cal"][pid]["winner"]=="A") for pid in cal])
            metrics["calibrated_test"]=score([dict(r,p_A=.5+alpha*(r["p_A"]-.5)) for r in rows])
            metrics["calibration_alpha"]=alpha
        report["models"][model+"/"+condition]=metrics
    target=Path(out or ("runs/retrieval_v2_"+method+"_report.json"))
    target.parent.mkdir(parents=True,exist_ok=True)
    target.write_text(json.dumps(report,indent=2),encoding="utf8")
    return report
