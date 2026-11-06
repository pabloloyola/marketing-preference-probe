"""Post-hoc robustness audit of a fixed development test.

The previously observed 300-row test set is reused; all results exploratory.
No API calls or external embeddings. Does not select models on the test set.
"""
import json
import math
import random
import statistics
from pathlib import Path
from .learned_data import load_split, DEFAULT_ROOT, DEFAULT_OUTPUT
from .learned_benchmark import (read_predictions, model_scores, score_column,
    features, labels, scale_inputs, fit_linear, fit_mlp, sigmoid, predict_mlp, evaluate,
    choose_head, get_numpy)

def paired_interval(a,b,draws=3000,seed=26011):
    """Paired bootstrap over *test IDs* for Brier loss difference A minus B."""
    if len(a)!=len(b) or not len(a):raise ValueError("Invalid paired vectors")
    rng=random.Random(seed)
    deltas=[float(x-y) for x,y in zip(a,b)]
    n=len(deltas)
    samples=sorted(sum(deltas[rng.randrange(n)] for _ in range(n))/n for _ in range(draws))
    return {"difference":statistics.mean(deltas),
            "ci95":[samples[int(.025*draws)],samples[min(draws-1,int(.975*draws))]],
            "n":n}

def train_predict(xtrain,ytrain,xval,yval,xtest,kind="linear",seed=1):
    np=get_numpy()
    fit,others,_=scale_inputs(xtrain,[xval,xtest])
    validation,test=others
    best=None
    for ridge in (.003,.03,.3):
        model=fit_linear(fit,ytrain,l2=ridge) if kind=="linear" else fit_mlp(fit,ytrain,l2=ridge,seed=seed)
        pv=sigmoid(validation@model) if kind=="linear" else predict_mlp(validation,model)
        pt=sigmoid(test@model) if kind=="linear" else predict_mlp(test,model)
        v=float(np.mean((pv-yval)**2))
        if best is None or v<best["validation_brier"]:
            if kind=="mlp":
                w,u,z=model
                nonlinear=np.tanh(test@u)@z
                linear=test@w
                diagnostic={"hidden_weight_norm":float(np.linalg.norm(u)),
                    "nonlinear_output_std":float(np.std(nonlinear)),
                    "linear_output_std":float(np.std(linear)),
                    "nonlinear_variance_fraction":float(np.var(nonlinear)/
                        (np.var(nonlinear)+np.var(linear)+1e-12))}
            else:diagnostic={}
            best={"validation_brier":v,"l2":ridge,"test_prob":pt,
                  "nonlinear_diagnostic":diagnostic}
    return best

def nonlinear_positive_control(seed=26011,n=1000):
    """Known odd nonlinear labeling task; no test outcomes from Upworthy.

    Two independent features with boundary y=1 iff x0*x1*x2 > 0.
    No linear decision boundary can represent this pattern. We diagnose
    MLP results, but DO NOT silently assert that its training succeeds.
    """
    np=get_numpy()
    rng=np.random.default_rng(seed)
    x=rng.choice([-1.,1.],size=(n,3)).astype(float)
    y=(np.prod(x,axis=1)>0).astype(float)
    order=rng.permutation(n);x=x[order];y=y[order]
    split=int(n*.6);val=int(n*.8)
    trained={}
    for kind in ("linear","mlp"):
        prediction=train_predict(x[:split],y[:split],x[split:val],y[split:val],x[val:],kind,seed)
        m=evaluate(y[val:],prediction["test_prob"])
        trained[kind]={"metrics":m,"validation_brier":prediction["validation_brier"],
                       "nonlinear_diagnostic":prediction["nonlinear_diagnostic"]}
    return {"task":"three-way parity sign(x0*x1*x2), odd under swapping",
            "linear_cannot_express_boundary":True,"results":trained,
            "mlp_improvement":trained["linear"]["metrics"]["brier"]-
                              trained["mlp"]["metrics"]["brier"]}

def audit(root=DEFAULT_ROOT,predictions=DEFAULT_OUTPUT,
          out="runs/learned_correction/audit",size=500,
          repeats=10,draws=2000,seed=26011):
    np=get_numpy()
    data,meta=load_split(root)
    records=read_predictions(predictions,data,meta,("jev","luna_decisions","luna_chat"))
    if not 20<=size<=len(data["train"]) or repeats<2 or draws<100:
        raise ValueError("Invalid audit configuration")
    y={k:labels(v) for k,v in data.items()}
    x={k:features(v) for k,v in data.items()}
    score={m:{k:score_column(model_scores(data[k],k,records,m)) for k in data}
           for m in ("jev","luna_decisions","luna_chat")}
    rng=random.Random(seed)
    repetitions=[]
    for repeat in range(repeats):
        ids=rng.sample(range(len(data["train"])),size)
        tr=x["train"][ids];validation=x["val"];test=x["test"]
        train_labels=y["train"][ids]
        sys={"human":(tr,validation,test)}
        for m in score:
            s=score[m]
            sys[m+"_only"]=(s["train"][ids],s["val"],s["test"])
            sys[m+"_plus_features"]=(
                np.column_stack([tr,s["train"][ids]]),
                np.column_stack([validation,s["val"]]),
                np.column_stack([test,s["test"]]))
        forecasts={}
        head_status={}
        for system,featureset in sys.items():
            head=train_predict(*featureset[:1],train_labels,
                               featureset[1],y["val"],featureset[2],kind="linear",
                               seed=seed+repeat)
            forecasts[system]=head["test_prob"]
            head_status[system]=head["l2"]
        l_j=(forecasts["jev_only"]-y["test"])**2
        l_h=(forecasts["human"]-y["test"])**2
        l_jh=(forecasts["jev_plus_features"]-y["test"])**2
        repetitions.append({"repeat":repeat,"human_brier":float(np.mean(l_h)),
            "jev_only_brier":float(np.mean(l_j)),
            "jev_plus_features_brier":float(np.mean(l_jh)),
            "jev_minus_human":paired_interval(l_j,l_h,draws,seed+repeat),
            "jev_features_minus_human":paired_interval(l_jh,l_h,draws,seed+repeat+500),
            "chosen_ridge":head_status})
    # Pairwise intervals at fixed training subset are already informative, but
    # multiple repeats share the SAME development test and are not independent.
    positive=nonlinear_positive_control(seed)
    # Quantify the currently used MLP on a real, fixed subset.
    idx=list(range(size))
    fit=x["train"][idx];yv=y["train"][idx]
    diagnostics={}
    for name,featureset in {"human":(fit,x["val"],x["test"]),
        "jev_plus_features":(np.column_stack([fit,score["jev"]["train"][idx]]),
          np.column_stack([x["val"],score["jev"]["val"]]),
          np.column_stack([x["test"],score["jev"]["test"]]))}.items():
        selected=train_predict(featureset[0],yv,featureset[1],y["val"],featureset[2],
                               "mlp",seed)
        diagnostics[name]={"test":evaluate(y["test"],selected["test_prob"]),
             "selected_l2":selected["l2"],**selected["nonlinear_diagnostic"]}
    report={"status":"RETROSPECTIVE DEVELOPMENT; previously inspected test",
            "train_size":size,"repeated_subsets":repeats,"paired_bootstrap_draws":draws,
            "positive_control":positive,"mlp_real_diagnostics":diagnostics,
            "repetitions":repetitions,
            "caveats":[
                "Repeated subsets share one test set; do not treat repeats as independent new external evaluations.",
                "Test set was previously examined and cannot validate a confirmatory model selection.",
                "Positive control tests the actual implemented optimization, not a theoretical MLP capacity guarantee.",
                "The parity positive control is intentionally symmetric and cannot be represented by a linear head.",
                "Only fixed handcrafted headline features are audited here; richer embeddings are a separate benchmark."]}
    target=Path(out);target.mkdir(parents=True,exist_ok=True)
    (target/"summary.json").write_text(json.dumps(report,indent=2),encoding="utf8")
    lines=["# Learned correction — robustness and MLP audit","",
           "**Retrospective developmental analysis.**","",
           "## Synthetic nonlinear positive control",""]
    for name,r in positive["results"].items():
        lines.append("- "+name+": Brier %.4f; AUC %.3f"%(r["metrics"]["brier"],r["metrics"]["auc"]))
    lines+=["","MLP gain in Brier (linear minus MLP): %.5f."%positive["mlp_improvement"],
            "","## MLP nonlinear contribution on real data",""]
    for name,r in diagnostics.items():
        lines.append("- %s: nonlinear output SD %.7f; variance fraction %.7f; Brier %.4f"%
                     (name,r["nonlinear_output_std"],r["nonlinear_variance_fraction"],r["test"]["brier"]))
    lines+=["","## Resampled training subsets (same fixed development test)","",
            "| Repetition | Human Brier | JEV only Brier | JEV + features Brier | JEV-only minus human [95% paired CI] |",
            "|---:|---:|---:|---:|---|"]
    for r in repetitions:
        d=r["jev_minus_human"]
        lines.append("| %d | %.4f | %.4f | %.4f | %+.4f [%.4f, %.4f] |"%
                     (r["repeat"],r["human_brier"],r["jev_only_brier"],r["jev_plus_features_brier"],
                      d["difference"],*d["ci95"]))
    lines+=["","## Caveats",""]+["- "+v for v in report["caveats"]]
    (target/"report.md").write_text("\n".join(lines)+"\n",encoding="utf8")
    return report
