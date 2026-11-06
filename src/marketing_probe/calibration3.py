"""Calibration Experiment 3: ranking versus probability quality.

Calibration is fit ONLY on the previously separated cal split; test is never
used to select a mapping. This is RETROSPECTIVE exploration, not confirmation.
No external ML packages and no API calls.
"""
import json
import math
import random
import statistics
from pathlib import Path
from .dataset import load_pairs
from .calibration1 import collect_predictions
from .experiment2 import shrinkage

EPS=1e-5

def clamp(p):
    return max(EPS,min(1-EPS,float(p)))

def logit(p):
    p=clamp(p)
    return math.log(p/(1-p))

def sigmoid(z):
    if z>=0:return 1/(1+math.exp(-min(60,z)))
    e=math.exp(max(-60,z))
    return e/(1+e)

def auc(truth, probabilities):
    """Mann-Whitney AUC with half credit for tied scores; undefined if one class."""
    pos=[p for y,p in zip(truth,probabilities) if y==1]
    neg=[p for y,p in zip(truth,probabilities) if y==0]
    if not pos or not neg:return None
    return sum(1 if a>b else .5 if a==b else 0 for a in pos for b in neg)/(len(pos)*len(neg))

def metrics(truth,probabilities):
    if not probabilities:return None
    p=[clamp(v) for v in probabilities]
    y=list(truth)
    if len(y)!=len(p):raise ValueError("Length mismatch")
    accuracy=sum(int((v>.5)==bool(t)) for v,t in zip(p,y) if v!=.5)
    commitments=sum(v!=.5 for v in p)
    return {"n":len(p),"auc":auc(y,p),
            "brier":statistics.mean((v-t)**2 for v,t in zip(p,y)),
            "logloss":statistics.mean(-t*math.log(v)-(1-t)*math.log1p(-v) for v,t in zip(p,y)),
            "mean_prediction":statistics.mean(p),"observed_A_rate":statistics.mean(y),
            "calibration_in_the_large":statistics.mean(p)-statistics.mean(y),
            "mean_confidence":statistics.mean(max(v,1-v) for v in p),
            "accuracy_non_tied":accuracy/commitments if commitments else None,
            "non_tied_predictions":commitments}

def bins(truth,probabilities,n_bins=5):
    ordered=sorted(zip(probabilities,truth),key=lambda a:a[0])
    count=min(len(ordered),n_bins)
    result=[]
    for j in range(count):
        rows=ordered[j*len(ordered)//count:(j+1)*len(ordered)//count]
        result.append({"n":len(rows),
            "mean_prediction":statistics.mean(r[0] for r in rows),
            "winner_rate":statistics.mean(r[1] for r in rows)})
    return result

def fit_logistic_mapping(cal_p,cal_y,intercept=True,ridge=.12,steps=1400,rate=.035):
    """Regularized logistic mapping on logit(p) using calibration data only.

    Constrain slope >=0 to prohibit post-hoc reversal of a learned ranking.
    If intercept=False this is reversal-symmetric temperature scaling.
    """
    if len(cal_p)<10 or len(cal_p)!=len(cal_y):raise ValueError("At least 10 calibration pairs required")
    x=[logit(p) for p in cal_p]
    w=1.
    a=0.
    for _ in range(steps):
        guesses=[sigmoid(a+w*z) for z in x]
        diff=[v-y for v,y in zip(guesses,cal_y)]
        dw=sum(d*z for d,z in zip(diff,x))/len(x)+ridge*(w-1)
        da=sum(diff)/len(x)+ridge*a if intercept else 0
        w=max(0,w-rate*dw)
        if intercept:a-=rate*da
    return {"intercept":a,"slope":w,"ridge":ridge,
            "transform":"platt_regularized" if intercept else "temperature_symmetric"}

def transformed(probs,model):
    return [sigmoid(model["intercept"]+model["slope"]*logit(p)) for p in probs]

def paired_brier_intervals(y,predictions,baseline="prevalence",draws=1500,seed=2027):
    """Uncertainty for *fixed* predictions, paired on the same test cases."""
    if baseline not in predictions:raise KeyError(baseline)
    rng=random.Random(seed)
    n=len(y)
    ref=predictions[baseline]
    output={}
    for name,p in predictions.items():
        if name==baseline:continue
        differences=[(a-t)**2-(b-t)**2 for a,b,t in zip(p,ref,y)]
        boot=sorted(statistics.mean(differences[j] for j in
             (rng.randrange(n) for _ in range(n))) for _ in range(draws))
        output[name]={"brier_difference_vs_"+baseline:statistics.mean(differences),
                      "ci95":[boot[int(.025*draws)],boot[min(draws-1,int(.975*draws))]]}
    return output

def analyze(directory="data/experiment2",
            prediction_paths=("runs/experiment2_predictions.jsonl",
                              "runs/retrieval_v2_semantic.jsonl",
                              "runs/retrieval_v2_hybrid.jsonl"),
            out="runs/calibration_3",bootstrap_draws=1500,seed=2027):
    root=Path(directory)
    splits={stage:list(load_pairs(root/(stage+".jsonl"))) for stage in ("cal","test")}
    for stage,rows in splits.items():
        ids=[str(r["pair_id"]) for r in rows]
        if len(ids)!=len(set(ids)):raise ValueError("Duplicate IDs in "+stage)
    truth={(stage,str(r["pair_id"])):r for stage,rows in splits.items() for r in rows}
    if {str(r["test_id"]) for r in splits["cal"]}&{str(r["test_id"]) for r in splits["test"]}:
        raise ValueError("Calibration/test experiment overlap")
    saved=collect_predictions(prediction_paths,truth)
    yc=[int(r["winner"]=="A") for r in splits["cal"]]
    yt=[int(r["winner"]=="A") for r in splits["test"]]
    prevalence=(sum(yc)+.5)/(len(yc)+1)  # smoothed calibration-only winner rate
    baseline={"half":[.5]*len(yt),"prevalence":[prevalence]*len(yt)}
    result={"status":"RETROSPECTIVE EXPLORATORY; test cases previously inspected",
            "cal_size":len(yc),"test_size":len(yt),
            "prevalence_fitted_on_cal":prevalence,"systems":{},
            "baselines":{name:metrics(yt,p) for name,p in baseline.items()},
            "warnings":[
                "All calibrator parameters fitted using cal only; test was used to inspect methods previously and is not a new confirmation set.",
                "AUC assesses ranking, not calibration; monotone calibration should preserve AUC apart from numerical ties.",
                "Platt intercept breaks exact A/B swap complementarity; symmetric temperature fixes intercept at zero.",
                "Raw model output probabilities may represent API choice distributions rather than epistemic human-winner probabilities.",
                "Only 40 calibration pairs; logistic calibration parameters are regularized and noisy.",
                "Calibration targets observed hard winners, not true latent CTR advantage or individual choice."]}
    for name,records in sorted(saved.items()):
        cal=[records.get(("cal",str(r["pair_id"]))) for r in splits["cal"]]
        test=[records.get(("test",str(r["pair_id"]))) for r in splits["test"]]
        if any(p is None for p in cal+test):
            result["systems"][name]={"status":"skipped_incomplete",
                "cal_coverage":sum(p is not None for p in cal),
                "test_coverage":sum(p is not None for p in test)}
            continue
        alpha=shrinkage(cal,yc)
        temp=fit_logistic_mapping(cal,yc,intercept=False)
        platt=fit_logistic_mapping(cal,yc,intercept=True)
        variants={
            "raw":list(test),
            "shrinkage":[.5+alpha*(p-.5) for p in test],
            "temperature":transformed(test,temp),
            "platt":transformed(test,platt),
            **baseline}
        stats={method:metrics(yt,p) for method,p in variants.items()}
        result["systems"][name]={
            "status":"complete","calibration":{"alpha":alpha,
                "temperature":temp,"platt":platt},
            "metrics":stats,
            "reliability_bins":{k:bins(yt,v) for k,v in variants.items()},
            "paired_brier_vs_prevalence":paired_brier_intervals(
                yt,variants,draws=bootstrap_draws,seed=seed)}
    outpath=Path(out);outpath.mkdir(parents=True,exist_ok=True)
    (outpath/"summary.json").write_text(json.dumps(result,indent=2),encoding="utf8")
    lines=["# Calibration Experiment 3 — discrimination versus calibration","",
           "**EXPLORATORY: TEST SET PREVIOUSLY INSPECTED.** Calibrator fitting uses calibration data only.",
           f"Calibration cases: {len(yc)}; evaluation cases: {len(yt)}.",
           f"Smoothed calibration-only A-winner rate: {prevalence:.3f}.",
           "","## Probability quality and AUC","",
           "| System | Mapping | AUC | Brier | Log loss | Mean p(A) |",
           "|---|---|---:|---:|---:|---:|"]
    for name,item in result["systems"].items():
        if item["status"]!="complete":
            lines.append(f"| {name} | incomplete: skipped | — | — | — | — |")
            continue
        for variant,m in item["metrics"].items():
            auc_value="n/a" if m["auc"] is None else f"{m['auc']:.3f}"
            lines.append(f"| {name} | {variant} | {auc_value} | {m['brier']:.4f} | {m['logloss']:.4f} | {m['mean_prediction']:.3f} |")
    lines+=["","## Interpretation","",
            "- AUC unchanged after a strictly monotone mapping means calibration cannot recover a bad ranking.",
            "- Constant predictions have AUC=0.5 when both winner classes occur; their classification accuracy is not a meaningful choice recommendation.",
            "- Brier differences and paired bootstrap intervals are in summary.json.",
            "- No calibration approach is selected as a final winner based on this previously examined test set.",
            "","## Caveats",""]+["- "+w for w in result["warnings"]]
    (outpath/"report.md").write_text("\n".join(lines)+"\n",encoding="utf8")
    return result
