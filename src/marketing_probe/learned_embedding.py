"""Frozen local Qwen embeddings as stronger human-only and hybrid controls.

The same 300 development-test IDs are reused; this is NOT new confirmation.
Embeddings receive only headlines, never labels. Validation selects ridge.
"""
import hashlib
import json
import random
from pathlib import Path
from .learned_data import DEFAULT_ROOT, DEFAULT_OUTPUT, load_split
from .learned_benchmark import (get_numpy,read_predictions,score_column,labels,
                                model_scores,features,evaluate)
from .learned_audit import train_predict,paired_interval

ENCODER="Qwen/Qwen3-Embedding-0.6B"

def load_vectors(data,manifest,cache,batch=16,model=ENCODER):
    np=get_numpy()
    stages=("train","val","test")
    identity=hashlib.sha256(json.dumps({
        "splits":[manifest[k+"_sha256"] for k in stages],
        "model":model},sort_keys=True).encode()).hexdigest()
    path=Path(cache)
    if path.exists():
        with np.load(path,allow_pickle=False) as z:
            if str(z["fingerprint"])!=identity or str(z["model"])!=model:
                raise ValueError("Cached vectors belong to another dataset/model")
            vectors={s:z[s] for s in stages}
            if any(len(vectors[s])!=len(data[s]) for s in stages):
                raise ValueError("Truncated embedding cache")
            return vectors
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:
        raise RuntimeError("Install the retrieval extra for Qwen embeddings") from exc
    encoder=SentenceTransformer(model,trust_remote_code=True)
    texts=[r[h] for s in stages for r in data[s] for h in ("headline_A","headline_B")]
    encoded=encoder.encode(texts,normalize_embeddings=True,batch_size=batch,
        convert_to_numpy=True,show_progress_bar=True)
    encoded=np.asarray(encoded,dtype="float32")
    if not np.isfinite(encoded).all():raise ValueError("Nonfinite embeddings")
    result={};start=0
    for s in stages:
        n=len(data[s]);result[s]=encoded[start:start+2*n].reshape(n,2,-1);start+=2*n
    path.parent.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(path,fingerprint=identity,model=model,**result)
    return result

def project(vectors,dimension=64,seed=26011):
    np=get_numpy()
    d=vectors["train"].shape[-1]
    if not 1<=dimension<=d:raise ValueError("Invalid projection dimension")
    rng=np.random.default_rng(seed)
    transform=rng.standard_normal((d,dimension))/dimension**.5
    return {s:(z[:,0,:]-z[:,1,:])@transform for s,z in vectors.items()}

def run(root=DEFAULT_ROOT,predictions=DEFAULT_OUTPUT,
        out="runs/learned_correction/embedding_baseline",
        cache="data/learned_correction/qwen_headlines.npz",
        sizes=(40,100,250,500,1000,1400),dimension=64,seed=26011,batch=16):
    np=get_numpy()
    data,manifest=load_split(root)
    scored=read_predictions(predictions,data,manifest,("jev","luna_chat","luna_decisions"))
    embed=project(load_vectors(data,manifest,cache,batch),dimension,seed)
    basic={s:features(rows) for s,rows in data.items()}
    model={m:{s:score_column(model_scores(data[s],s,scored,m)) for s in data}
           for m in ("jev","luna_chat","luna_decisions")}
    y={s:labels(rows) for s,rows in data.items()}
    order=list(range(len(data["train"])))
    random.Random(seed).shuffle(order)
    result={"status":"RETROSPECTIVE DEVELOPMENT TEST; previously inspected",
            "encoder":ENCODER,"projection_dimensions":dimension,"seed":seed,
            "data_manifest":manifest,"results":[],"paired_differences":[],
            "limitations":["The development test and upstream retrieval corpus were previously explored",
             "A single random projection seed is not a systematic representation comparison",
             "Local embedding downloads may require network access; no OpenRouter calls",
             "Supervised optimization uses hard observed winners, not count-aware effects",
             "Encoder pretraining contamination of public headlines cannot be excluded"]}
    for n in sizes:
        n=int(n)
        if not 10<=n<=len(data["train"]):raise ValueError("Invalid training size")
        idx=order[:n]
        variants={"handcrafted":basic,"qwen_only":embed,
            "qwen_plus_handcrafted":{s:np.column_stack([embed[s],basic[s]]) for s in data}}
        for m in model:
            variants[m+"_only"]=model[m]
            variants["qwen_plus_"+m]={s:np.column_stack([embed[s],model[m][s]]) for s in data}
        predictions_by_system={}
        for name,groups in variants.items():
            trained=train_predict(groups["train"][idx],y["train"][idx],
                groups["val"],y["val"],groups["test"],kind="linear",seed=seed)
            p=trained["test_prob"]
            predictions_by_system[name]=p
            result["results"].append({"train_n":n,"system":name,
                "validation_brier":trained["validation_brier"],
                "selected_l2":trained["l2"],"test":evaluate(y["test"],p)})
        ref=(predictions_by_system["qwen_only"]-y["test"])**2
        for m in model:
            name="qwen_plus_"+m
            alternative=(predictions_by_system[name]-y["test"])**2
            result["paired_differences"].append({"train_n":n,"system":name,
                "relative_to":"qwen_only",
                **paired_interval(alternative,ref,draws=2000,seed=seed+n)})
    dest=Path(out);dest.mkdir(parents=True,exist_ok=True)
    (dest/"summary.json").write_text(json.dumps(result,indent=2),encoding="utf8")
    lines=["# Frozen Qwen embeddings: supervised versus model-augmented","",
        "**Previously inspected development test. Not independent confirmation.**","",
        "| Train n | System | Validation Brier | Test Brier | Test AUC |",
        "|---:|---|---:|---:|---:|"]
    for r in result["results"]:
        m=r["test"]
        lines.append("| %d | %s | %.4f | %.4f | %s |"%(
            r["train_n"],r["system"],r["validation_brier"],m["brier"],
            "%.3f"%m["auc"] if m["auc"] is not None else "n/a"))
    lines+=["","## Limitations",""]+["- "+v for v in result["limitations"]]
    (dest/"report.md").write_text("\n".join(lines)+"\n",encoding="utf8")
    return result
