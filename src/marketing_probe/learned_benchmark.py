"""AEM-style behavioral correction: label-efficiency benchmark on a DEVELOPMENT split.

Uses small NumPy models: signed structural headline features, model log-odds,
regularized linear logistic heads, and an odd symmetric residual tanh MLP.
All training labels come from the training split; model-selection labels from
validation; test is only evaluated after fitting/selection. No model API calls.
"""
import json
import hashlib
import math
import random
from pathlib import Path
from .dataset import load_pairs
from .retrieval_v2 import difference
from .learned_data import load_split, DEFAULT_ROOT, DEFAULT_OUTPUT, PROMPT_VERSION
from .api import MODELS

SIZES=(40,100,250,500,1000,1400)
RIDGES=(.03,.3)
EPS=1e-5

def get_numpy():
    try:import numpy as np
    except ImportError as exc:raise RuntimeError("Use uv run --extra learning to install NumPy") from exc
    return np

def model_scores(rows,stage,records,model):
    return [records[(stage,str(r["pair_id"]),model)] for r in rows]

def read_predictions(path,data,manifest,models):
    records={}
    valid={(stage,str(r["pair_id"])) for stage,rows in data.items() for r in rows}
    fp=hashlib.sha256(json.dumps({
        "splits":[manifest[k+"_sha256"] for k in ("train","val","test")],
        "prompt":PROMPT_VERSION},sort_keys=True).encode()).hexdigest()
    for r in load_pairs(path):
        key=(r.get("stage"),str(r.get("pair_id")))
        model=r.get("model_key")
        if (key not in valid or r.get("data_fingerprint")!=fp or
            r.get("prompt_version")!=PROMPT_VERSION or
            model not in MODELS or
            r.get("served_model_requested")!=MODELS[model][0]):
            raise ValueError("Invalid/incompatible prediction record")
        p=float(r["p_A"])
        if not (math.isfinite(p) and 0<=p<=1):raise ValueError("Invalid p_A")
        full=(*key,model)
        if full in records:raise ValueError("Duplicate prediction "+str(full))
        records[full]=p
    missing={model:sum((stage,str(r["pair_id"]),model) not in records
                       for stage,rows in data.items() for r in rows)
             for model in models}
    if any(missing.values()):
        raise ValueError("Missing model prediction records: "+str(missing)+
                         ". Run infer-learned again to complete all stages.")
    return records

def features(rows):
    np=get_numpy()
    return np.asarray([difference(r) for r in rows],dtype=float)

def score_column(values):
    np=get_numpy()
    p=np.clip(np.asarray(values,dtype=float),EPS,1-EPS)
    return np.log(p/(1-p))[:,None]

def labels(rows):
    np=get_numpy()
    return np.asarray([int(r["winner"]=="A") for r in rows],dtype=float)

def scale_inputs(xtrain,other):
    np=get_numpy()
    # scale derived ONLY from current training subset
    rms=np.maximum(np.sqrt(np.mean(xtrain*xtrain,axis=0)),.05)
    return xtrain/rms,[arr/rms for arr in other],rms

def sigmoid(z):
    np=get_numpy()
    return 1/(1+np.exp(-np.clip(z,-30,30)))

def brier(y,p):
    np=get_numpy()
    return float(np.mean((p-y)**2))

def evaluate(y,p):
    np=get_numpy()
    p=np.clip(np.asarray(p),EPS,1-EPS)
    return {"n":len(y),"brier":brier(y,p),
            "logloss":float(np.mean(-y*np.log(p)-(1-y)*np.log1p(-p))),
            "accuracy_non_tied":float(np.mean((p>.5)==y)) if np.all(p!=.5) else None,
            "auc":auc(y,p)}

def auc(y,p):
    np=get_numpy()
    positive=np.asarray(p)[np.asarray(y)==1]
    negative=np.asarray(p)[np.asarray(y)==0]
    if not len(positive) or not len(negative):return None
    return float(np.mean((positive[:,None]>negative[None,:])+
                        .5*(positive[:,None]==negative[None,:])))

def fit_linear(x,y,l2=.1,steps=450):
    """Ridge logistic with zero intercept, preserving signed A/B reversal."""
    np=get_numpy()
    w=np.zeros(x.shape[1])
    # Adagrad scales feature updates and remains robust with small samples.
    g2=np.zeros_like(w)
    for _ in range(steps):
        p=sigmoid(x@w)
        grad=x.T@(p-y)/len(y)+l2*w
        g2+=grad*grad
        w-=.17*grad/(np.sqrt(g2)+.05)
    return w

def fit_mlp(x,y,l2=.1,width=12,steps=550,seed=1):
    """Small odd residual MLP: linear score + sum v*tanh(X U).

    No intercept, no hidden biases: f(-X)=-f(X), so sigmoid complements
    after swapping A/B when all features (including model score) reverse.
    """
    np=get_numpy()
    rng=np.random.default_rng(seed)
    d=x.shape[1]
    w=np.zeros(d)
    u=rng.normal(0,.08,size=(d,width))
    v=rng.normal(0,.08,size=width)
    gw2=np.zeros_like(w);gu2=np.zeros_like(u);gv2=np.zeros_like(v)
    for _ in range(steps):
        h=np.tanh(x@u)
        prob=sigmoid(x@w+h@v)
        e=(prob-y)/len(y)
        dw=x.T@e+l2*w
        dv=h.T@e+l2*v
        du=x.T@((e[:,None]*v[None,:])*(1-h*h))+l2*u
        gw2+=dw*dw;gu2+=du*du;gv2+=dv*dv
        w-=.12*dw/(np.sqrt(gw2)+.05)
        u-=.12*du/(np.sqrt(gu2)+.05)
        v-=.12*dv/(np.sqrt(gv2)+.05)
    return (w,u,v)

def predict_mlp(x,weights):
    np=get_numpy()
    w,u,v=weights
    return sigmoid(x@w+np.tanh(x@u)@v)

def choose_head(train,val,test,kind,seed):
    xtr,ytr=train
    xv,yv=val
    xt,yt=test
    best=None
    for ridge in RIDGES:
        if kind=="linear":
            model=fit_linear(xtr,ytr,l2=ridge)
            pv=sigmoid(xv@model);pt=sigmoid(xt@model)
        elif kind=="mlp":
            model=fit_mlp(xtr,ytr,l2=ridge,seed=seed)
            pv=predict_mlp(xv,model);pt=predict_mlp(xt,model)
        else:raise ValueError(kind)
        value=brier(yv,pv)
        if best is None or value<best[0]:
            best=(value,ridge,pt)
    return {"validation_brier":best[0],"selected_l2":best[1],
            "test":evaluate(yt,best[2])}

def learning_curve(root=DEFAULT_ROOT,predictions=DEFAULT_OUTPUT,
                   out="runs/learned_correction/benchmark",
                   models=("jev","luna_decisions","luna_chat"),
                   sizes=SIZES,seed=26011):
    np=get_numpy()
    data,manifest=load_split(root)
    sizes=tuple(sorted(set(int(v) for v in sizes)))
    if not sizes or sizes[0]<10 or sizes[-1]>len(data["train"]):
        raise ValueError("Learning curve sizes must be between 10 and train size")
    models=tuple(models)
    if not models or any(m not in MODELS for m in models):
        raise ValueError("Invalid model names")
    scored=read_predictions(predictions,data,manifest,models)
    x={s:features(data[s]) for s in data}
    y={s:labels(data[s]) for s in data}
    order=list(range(len(data["train"])))
    random.Random(seed).shuffle(order)
    report={"status":"DEVELOPMENT BENCHMARK; source pool previously explored",
            "sizes":list(sizes),"seed":seed,"models":list(models),
            "data_manifest":manifest,
            "methodology":{"selection":"ridge selected only by validation Brier",
                           "test":"reported for each prespecified sample size; do not optimize further on test",
                           "model_score":"clipped logit(p_A), model probabilities collected zero-shot",
                           "features":"9 signed headline feature differences; no test clicks as inputs",
                           "heads":"ridge logistic without intercept; 12-unit odd residual tanh MLP",
                           "label":"empirical higher-CTR headline hard winner"},
            "baselines":{},"results":[]}
    for size in sizes:
        idx=np.asarray(order[:size],dtype=int)
        # Refit standardization on the current training-size subset.
        training=x["train"][idx]
        train_feat,others,_=scale_inputs(training,[x["val"],x["test"]])
        val_feat,test_feat=others
        ytr=y["train"][idx]
        model_features={"human_features":(train_feat,val_feat,test_feat)}
        for model in models:
            scores={s:score_column(model_scores(data[s],s,scored,model)) for s in data}
            trscore=scores["train"][idx]
            # Log-odds is standardized from the training subset as well.
            score_train,score_other,_=scale_inputs(trscore,[scores["val"],scores["test"]])
            model_features[model+"/model_only"]=(score_train,*score_other)
            model_features[model+"/features_plus_score"]=(
                np.concatenate([train_feat,score_train],axis=1),
                np.concatenate([val_feat,score_other[0]],axis=1),
                np.concatenate([test_feat,score_other[1]],axis=1))
        for name,(a,b,c) in model_features.items():
            kinds=("linear","mlp") if name=="human_features" or name.endswith("features_plus_score") else ("linear",)
            for kind in kinds:
                chosen=choose_head((a,ytr),(b,y["val"]),(c,y["test"]),kind,seed)
                report["results"].append({"train_n":size,"system":name,"kind":kind,**chosen})
        valpre=(sum(ytr)+.5)/(len(ytr)+1)
        report["baselines"][str(size)]={
            "half":evaluate(y["test"],np.full(len(y["test"]),.5)),
            "train_prevalence":evaluate(y["test"],np.full(len(y["test"]),valpre))}
    # Model raw score does not depend on training size.
    report["raw_model_baselines"]={
        model:evaluate(y["test"],[scored[("test",str(r["pair_id"]),model)] for r in data["test"]])
        for model in models}
    output=Path(out);output.mkdir(parents=True,exist_ok=True)
    (output/"summary.json").write_text(json.dumps(report,indent=2),encoding="utf8")
    lines=["# Learned behavioral correction — development learning curves","",
           "**Development only.** The 2,000 historical pool previously informed retrieval work.",
           "Validation chooses regularization; test evaluates a fixed design for each training size.",
           "No OpenRouter requests were made by this command.","",
           "| Train n | System | Head | Validation Brier | Test Brier | Test AUC |",
           "|---:|---|---|---:|---:|---:|"]
    for r in report["results"]:
        a=r["test"]["auc"]
        lines.append(f"| {r['train_n']} | {r['system']} | {r['kind']} | {r['validation_brier']:.4f} | {r['test']['brier']:.4f} | {a:.3f} |" if a is not None else
                     f"| {r['train_n']} | {r['system']} | {r['kind']} | {r['validation_brier']:.4f} | {r['test']['brier']:.4f} | n/a |")
    lines+=["","## Interpretation","",
            "- Compare features+model with human-only features at the SAME size and head.",
            "- Compare linear versus nonlinear heads at the SAME size and feature access.",
            "- An improvement only on this development test is not confirmatory.",
            "- The current training objective uses hard winner labels, not count-aware CTR estimation.",
            "- AUC measures discrimination; Brier/log loss include confidence.",
            "- Full model prediction coverage across train, val and test is required before running."]
    (output/"report.md").write_text("\n".join(lines)+"\n",encoding="utf8")
    return report
