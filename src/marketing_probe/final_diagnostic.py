"""Final offline behavioral-validity diagnostic.

Fresh test IDs and headline-pair signatures are disjoint from all prior
experiments and the 2k retrieval/training corpus. No OpenRouter requests.
"""
import hashlib
import json
import math
import random
import statistics
import tempfile
from pathlib import Path

from .dataset import load_pairs, prepare
from .experiment2 import score, shrinkage
from .retrieval_v2 import RetrievalV2, difference, retrieval_baseline, signature

def read(path):
    return list(load_pairs(path))

def write(path, rows):
    dest=Path(path)
    dest.parent.mkdir(parents=True,exist_ok=True)
    dest.write_text("".join(json.dumps(r,ensure_ascii=False)+"\n" for r in rows),encoding="utf8")

def protected_paths(pool_path,split_dir,exploratory_path):
    d=Path(split_dir)
    return [pool_path,exploratory_path,d/"train.jsonl",d/"cal.jsonl",d/"test.jsonl"]

def prepare_final(csv_path, pool_path="data/retrieval_v2/pool.jsonl",
                  split_dir="data/experiment2", exploratory_path="data/pairs.jsonl",
                  output_dir="data/final_holdout", cal_n=100,test_n=300,seed=5091):
    if cal_n<20 or test_n<30:
        raise ValueError("Use at least 20 calibration and 30 test pairs")
    previous=[r for path in protected_paths(pool_path,split_dir,exploratory_path)
              for r in read(path)]
    exclude_ids={str(r["test_id"]) for r in previous}
    exclude_signatures={signature(r) for r in previous}
    desired=cal_n+test_n
    with tempfile.TemporaryDirectory() as td:
        tmp=Path(td)/"all.jsonl"
        # Existing prepare() picks at most one pair per randomized test.
        prepare(csv_path,tmp,n=desired+len(exclude_ids)+100,seed=seed)
        pool=read(tmp)
    selected=[]; seen=set(); seen_sigs=set()
    for r in pool:
        tid=str(r["test_id"])
        sig=signature(r)
        if tid in exclude_ids or tid in seen or sig in exclude_signatures or sig in seen_sigs:
            continue
        selected.append(r); seen.add(tid);seen_sigs.add(sig)
        if len(selected)==desired:break
    if len(selected)<desired:
        raise ValueError("Not enough fresh eligible pairs: found %d of %d" % (len(selected),desired))
    root=Path(output_dir)
    write(root/"cal.jsonl",selected[:cal_n])
    write(root/"test.jsonl",selected[cal_n:])
    meta={"cal":cal_n,"test":test_n,"seed":seed,"excluded_test_ids":len(exclude_ids),
          "retrieval_pool_sha256":hashlib.sha256(Path(pool_path).read_bytes()).hexdigest(),
          "cal_sha256":hashlib.sha256((root/"cal.jsonl").read_bytes()).hexdigest(),
          "test_sha256":hashlib.sha256((root/"test.jsonl").read_bytes()).hexdigest()}
    (root/"manifest.json").write_text(json.dumps(meta,indent=2),encoding="utf8")
    return meta

def validated(pool_path, holdout_dir,split_dir,exploratory_path):
    pool=read(pool_path)
    root=Path(holdout_dir)
    cal=read(root/"cal.jsonl")
    test=read(root/"test.jsonl")
    groups=[pool,cal,test,read(exploratory_path)]+[
        read(Path(split_dir)/(s+".jsonl")) for s in ("train","cal","test")]
    ids=[{str(r["test_id"]) for r in group} for group in groups]
    signatures=[{signature(r) for r in group} for group in groups]
    if any(len(i)!=len(group) or len(j)!=len(group)
           for i,j,group in zip(ids,signatures,groups)):
        raise ValueError("Duplicate IDs or headline pairs within dataset")
    if any(ids[i]&ids[j] or signatures[i]&signatures[j]
           for i in range(len(groups)) for j in range(i+1,len(groups))):
        raise ValueError("Data leakage: overlapping test IDs or headline pairs")
    manifest=json.loads((root/"manifest.json").read_text())
    checks={"retrieval_pool_sha256":Path(pool_path),
            "cal_sha256":root/"cal.jsonl","test_sha256":root/"test.jsonl"}
    for name,path in checks.items():
        if manifest.get(name)!=hashlib.sha256(path.read_bytes()).hexdigest():
            raise ValueError("Dataset changed after creating final manifest")
    return pool,cal,test,manifest

def sigmoid(z):
    return 1/(1+math.exp(-max(-30,min(30,z))))

class ChoiceLogit:
    """Regularized logistic regression on signed A minus B features.

    No intercept: reversing A/B reverses log-odds, avoiding an ordering bias.
    No neural or API inference. Feature scaling learned from training pool only.
    """
    def __init__(self,train,steps=900,lr=.2,reg=.3):
        if not train:raise ValueError("No training rows")
        xs=[difference(r) for r in train]
        dim=len(xs[0])
        self.scales=[max(.025,math.sqrt(sum(x[j]*x[j] for x in xs)/len(xs)))
                     for j in range(dim)]
        xs=[[x[j]/self.scales[j] for j in range(dim)] for x in xs]
        ys=[int(r["winner"]=="A") for r in train]
        self.w=[0.]*dim
        for _ in range(steps):
            grad=[0.]*dim
            for x,y in zip(xs,ys):
                err=sigmoid(sum(w*v for w,v in zip(self.w,x)))-y
                for j,v in enumerate(x):grad[j]+=err*v
            for j in range(dim):
                self.w[j]-=lr*(grad[j]/len(xs)+reg*self.w[j])
    def predict(self,pair):
        x=difference(pair)
        return sigmoid(sum(w*v/scale for w,v,scale in zip(self.w,x,self.scales)))

def evaluate(pairs,probabilities):
    rows=[dict(row,p_A=p) for row,p in zip(pairs,probabilities)]
    return score(rows)

def bootstrap(pairs,systems,draws=3000,seed=2027):
    """Paired bootstrap for comparison between fixed predictions on test rows."""
    rng=random.Random(seed)
    n=len(pairs)
    y=[int(r["winner"]=="A") for r in pairs]
    correct={name:[int((p>=.5)==bool(t)) for p,t in zip(pred,y)]
             for name,pred in systems.items()}
    brier={name:[(p-t)**2 for p,t in zip(pred,y)]
           for name,pred in systems.items()}
    names=sorted(systems)
    results={}
    for i,a in enumerate(names):
        for b in names[i+1:]:
            key=a+" minus "+b
            diffa=[correct[a][j]-correct[b][j] for j in range(n)]
            diffb=[brier[a][j]-brier[b][j] for j in range(n)]
            indices=[rng.choices(range(n),k=n) for _ in range(draws)]
            acc=sorted(sum(diffa[j] for j in ids)/n for ids in indices)
            br=sorted(sum(diffb[j] for j in ids)/n for ids in indices)
            results[key]={"accuracy_diff":statistics.mean(diffa),
                          "accuracy_ci95":[acc[int(.025*draws)],acc[int(.975*draws)]],
                          "brier_diff":statistics.mean(diffb),
                          "brier_ci95":[br[int(.025*draws)],br[int(.975*draws)]]}
    return results

def reliability(row):
    """Descriptive Wald z for observed CTR difference; noisy at low CTR."""
    a=float(row["ctr_A"]);b=float(row["ctr_B"])
    na=int(row["impressions_A"]);nb=int(row["impressions_B"])
    se=math.sqrt(a*(1-a)/na+b*(1-b)/nb)
    return abs(a-b)/se if se>0 else 0.

def report_final(pool_path="data/retrieval_v2/pool.jsonl",
                 holdout_dir="data/final_holdout",split_dir="data/experiment2",
                 exploratory_path="data/pairs.jsonl",
                 cache="data/final_holdout/embeddings.npz",
                 output_dir="runs/final_diagnostic",k=5):
    pool,cal,test,meta=validated(pool_path,holdout_dir,split_dir,exploratory_path)
    # Both methods fixed before held-out analysis; calibration uses separate fresh cal.
    predictor=ChoiceLogit(pool)
    finder=RetrievalV2(pool,cal+test,method="hybrid",cache=cache)
    def preds(rows):
        return {"constant_half":[.5]*len(rows),
                "choice_logit":[predictor.predict(r) for r in rows],
                "hybrid_retrieval_only":[retrieval_baseline(finder.find(r,k)) for r in rows]}
    calibration=preds(cal)
    test_predictions=preds(test)
    alpha={name:shrinkage(p,[int(r["winner"]=="A") for r in cal])
           for name,p in calibration.items()}
    corrected={name:[.5+alpha[name]*(x-.5) for x in probabilities]
               for name,probabilities in test_predictions.items()}
    metrics={name:{"test":evaluate(test,values),
                   "calibration_alpha":alpha[name],
                   "calibrated_test":evaluate(test,corrected[name])}
             for name,values in test_predictions.items()}
    bins={"ambiguous_z_lt_1":[], "intermediate_z_1_to_2":[],
          "clearer_z_ge_2":[]}
    for i,p in enumerate(test):
        z=reliability(p)
        label="ambiguous_z_lt_1" if z<1 else "intermediate_z_1_to_2" if z<2 else "clearer_z_ge_2"
        bins[label].append(i)
    by_reliability={}
    for label,ids in bins.items():
        if not ids:
            by_reliability[label]={"n":0};continue
        by_reliability[label]={"n":len(ids)}
        for name,values in test_predictions.items():
            sub=[test[i] for i in ids]
            by_reliability[label][name]=evaluate(sub,[values[i] for i in ids])
    results={"manifest":meta,"train_size":len(pool),"cal_size":len(cal),
             "test_size":len(test),"k":k,"metrics":metrics,
             "paired_uncertainty":bootstrap(test,test_predictions),
             "ctr_reliability":by_reliability,
             "limitations":["Test outcomes were not used for fitting or calibration; do not select and tune models on this same final test.",
                            "Historical Upworthy population differs from target marketing populations.",
                            "CTR z-score is a descriptive approximation, not a causal or individual preference measure.",
                            "Choice features are handcrafted proxies; the regularization strength was fixed in advance.",
                            "Public data contamination remains possible."]}
    root=Path(output_dir);root.mkdir(parents=True,exist_ok=True)
    (root/"summary.json").write_text(json.dumps(results,indent=2),encoding="utf8")
    lines=["# Final offline diagnostic","",
           "No OpenRouter requests were made. Fresh held-out A/B test IDs and exact headline pairs",
           "were excluded from the training/retrieval corpus and all previous experiments.","",
           "| System | Accuracy | Brier | Log loss | Calibrated Brier |",
           "|---|---:|---:|---:|---:|"]
    for name,item in metrics.items():
        r=item["test"];c=item["calibrated_test"]
        lines.append("| %s | %.3f | %.4f | %.4f | %.4f |"%(name,r["accuracy"],r["brier"],r["logloss"],c["brier"]))
    lines+=["","## Paired uncertainty",""]
    for label,entry in results["paired_uncertainty"].items():
        lo,hi=entry["accuracy_ci95"]
        lines.append("- %s: accuracy difference %+.3f; 95%% paired bootstrap [%.3f, %.3f]"%
                     (label,entry["accuracy_diff"],lo,hi))
    lines+=["","## Accuracy by observed CTR reliability",""]
    for label,item in by_reliability.items():
        lines.append("- %s (n=%d): %s"%(label,item["n"],
            ", ".join("%s %.3f"%(name,item[name]["accuracy"])
                      for name in test_predictions) if item["n"] else "No tests"))
    lines+=["","## Caveats",""]+["- "+s for s in results["limitations"]]
    (root/"report.md").write_text("\n".join(lines)+"\n",encoding="utf8")
    return results
