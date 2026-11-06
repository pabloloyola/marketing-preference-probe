"""Calibration Experiment 1: observation uncertainty versus model probability.

Purely OFFLINE descriptive diagnostic. The posterior is conditioned on the
same held-out click counts and MUST NOT be used as independent ground truth.
"""
import csv
import hashlib
import json
import math
import random
import statistics
from collections import defaultdict
from pathlib import Path

from .dataset import load_pairs
from .experiment2 import shrinkage

def counts(pair):
    values={}
    for arm in ("A","B"):
        n=int(pair["impressions_"+arm]); c=int(pair["clicks_"+arm])
        if not 0<=c<=n or n<=0:
            raise ValueError("Invalid clicks/impressions for "+str(pair.get("pair_id")))
        values[arm]=(c,n)
    return values

def observations(pair, prior=0.5, draws=4000, seed=2027, repeat_n=1000):
    """Independent Beta(prior, prior) arm priors and posterior Monte Carlo.

    A/B labels are not used. Posterior probability of *latent* superiority is
    conditional on the observed click counts. A new-trial win probability
    includes binomial sampling noise, at specified per-arm sample size.
    """
    if prior<=0 or draws<100 or repeat_n<1:raise ValueError("Invalid posterior settings")
    c=counts(pair)
    rng=random.Random(seed)
    wins=0
    repeat=0.
    deltas=[]
    for _ in range(draws):
        pa=rng.betavariate(prior+c["A"][0],prior+c["A"][1]-c["A"][0])
        pb=rng.betavariate(prior+c["B"][0],prior+c["B"][1]-c["B"][0])
        wins+=pa>pb
        deltas.append(pa-pb)
        # Approximate future CTR-difference distribution by normal binomial
        # variance; avoids 2 * draws * repeat_n Bernoulli samples.
        se=math.sqrt(pa*(1-pa)/repeat_n+pb*(1-pb)/repeat_n)
        z=(pa-pb)/se if se>0 else (1e8 if pa>pb else -1e8)
        repeat+=0.5*(1+math.erf(z/math.sqrt(2)))
    deltas.sort()
    lo=deltas[int(.025*draws)]
    hi=deltas[min(draws-1,int(.975*draws))]
    return {"p_latent_A_superior":wins/draws,
            "p_repeat_A_wins_approx":repeat/draws,
            "mean_ctr_difference":statistics.mean(deltas),
            "ctr_difference_95ci":[lo,hi],
            "prior_beta_shape":prior,"repeat_impressions_per_arm":repeat_n}

def calibration_bins(records,bins=6):
    """Equal-count bins, descriptive reliability against observed winners."""
    if not records:return []
    ordered=sorted(records,key=lambda r:r["p_A"])
    groups=[]
    for j in range(min(bins,len(ordered))):
        lo=j*len(ordered)//min(bins,len(ordered))
        hi=(j+1)*len(ordered)//min(bins,len(ordered))
        g=ordered[lo:hi]
        if not g:continue
        groups.append({"n":len(g),"mean_prediction":statistics.mean(r["p_A"] for r in g),
                       "observed_A_winner_frequency":statistics.mean(int(r["winner"]=="A") for r in g),
                       "mean_observation_uncertainty":statistics.mean(
                           2*min(r["posterior"]["p_latent_A_superior"],1-r["posterior"]["p_latent_A_superior"])
                           for r in g)})
    return groups

def diagnostics(records):
    if not records:return None
    y=[int(r["winner"]=="A") for r in records]
    p=[min(1-1e-8,max(1e-8,float(r["p_A"]))) for r in records]
    return {"n":len(p),"mean_forecast":statistics.mean(p),
            "observed_A_winner_rate":statistics.mean(y),
            "calibration_in_the_large":statistics.mean(p)-statistics.mean(y),
            "brier":statistics.mean((a-b)**2 for a,b in zip(p,y)),
            "logloss":statistics.mean(-b*math.log(a)-(1-b)*math.log1p(-a) for a,b in zip(p,y)),
            "mean_absolute_confidence_from_half":statistics.mean(abs(x-.5) for x in p),
            "accuracy_no_ties":sum(int((a>.5)==bool(b)) for a,b in zip(p,y) if a!=.5)/
                    max(1,sum(a!=.5 for a in p)),
            "non_tied_forecasts":sum(a!=.5 for a in p)}

def collect_predictions(prediction_paths, truth):
    grouped=defaultdict(dict)
    for filename in prediction_paths:
        path=Path(filename)
        if not path.exists():raise FileNotFoundError(str(path))
        for r in load_pairs(path):
            if r.get("stage") not in ("cal","test"):continue
            key=(r["stage"],r.get("pair_id"))
            if key not in truth:continue
            p=float(r["p_A"])
            if not 0<=p<=1:raise ValueError("Invalid prediction probability")
            label=r["model_key"]+"/"+r["condition"]
            if r.get("condition")=="retrieval_v2":
                label+="/"+str(r.get("method"))
            record=(r["stage"],str(r["pair_id"]))
            if record in grouped[label]:
                raise ValueError("Duplicate predictions for "+label+" "+str(record))
            grouped[label][record]=p
    return grouped

def analyze(directory="data/experiment2",
            prediction_paths=("runs/experiment2_predictions.jsonl",
                              "runs/retrieval_v2_semantic.jsonl",
                              "runs/retrieval_v2_hybrid.jsonl"),
            output="runs/calibration_1",
            priors=(0.5,2.0),draws=4000,repeat_n=1000,seed=2027):
    folder=Path(directory)
    splits={s:list(load_pairs(folder/(s+".jsonl"))) for s in ("cal","test")}
    truth={(stage,str(r["pair_id"])):r for stage,rows in splits.items() for r in rows}
    if len(truth)!=sum(len(r) for r in splits.values()):
        raise ValueError("Duplicate stage/pair id")
    post={}
    for j,((stage,pid),r) in enumerate(truth.items()):
        post[(stage,pid)]={}
        for k,prior in enumerate(priors):
            post[(stage,pid)][str(prior)]=observations(
                r,prior=prior,draws=draws,repeat_n=repeat_n,
                seed=seed+1009*j+17*k)
    predict=collect_predictions(prediction_paths,truth)
    report={"analysis_status":"EXPLORATORY, PREVIOUSLY INSPECTED DATA",
            "target":"observed CTR winner; posterior is conditioned on click counts, not a truth label",
            "posterior_assumptions":{"arm_prior":"Independent Beta(a,a) for each arm",
                "prior_shapes":list(priors),"monte_carlo_draws":draws,
                "repeat_impressions_per_arm":repeat_n,
                "repeat_winner_method":"Gaussian approximation to new binomial CTR difference"},
            "split_sizes":{s:len(v) for s,v in splits.items()},
            "systems":{},"outcome_uncertainty":{},
            "warnings":[
                "Posterior probabilities are conditioned on held-out clicks and are not forecast labels.",
                "This is retrospective method diagnosis, not new out-of-sample validation.",
                "Hard winner labels include finite-sample uncertainty.",
                "No API calls; model prediction files must exist locally.",
                "Current fitted calibration uses only shrinkage, not learned reliability slope/intercept.",
                "A/B probabilities from decision and chat endpoints may not be semantically equivalent."]}
    for stage,rows in splits.items():
        report["outcome_uncertainty"][stage]={}
        for prior in priors:
            probabilities=[post[(stage,str(r["pair_id"]))][str(prior)]["p_latent_A_superior"] for r in rows]
            ambiguity=[2*min(p,1-p) for p in probabilities]
            report["outcome_uncertainty"][stage][str(prior)]={
                "n":len(rows),
                "mean_binary_ambiguity":statistics.mean(ambiguity),
                "fraction_posterior_between_0_2_and_0_8":sum(.2<=p<=.8 for p in probabilities)/len(rows),
                "fraction_posterior_between_0_4_and_0_6":sum(.4<=p<=.6 for p in probabilities)/len(rows),
                "mean_empirical_ctr_gap_pp":statistics.mean(
                    abs(r["ctr_A"]-r["ctr_B"])*100 for r in rows)}
    for label,vals in sorted(predict.items()):
        cal=[(r,vals[("cal",str(r["pair_id"]))]) for r in splits["cal"]
             if ("cal",str(r["pair_id"])) in vals]
        test=[(r,vals[("test",str(r["pair_id"]))]) for r in splits["test"]
              if ("test",str(r["pair_id"])) in vals]
        if not test:continue
        baseprior=str(priors[0])
        def attach(items,alpha=1):
            return [dict(r,p_A=.5+alpha*(p-.5),
                posterior=post[(stage,str(r["pair_id"]))][baseprior])
                for r,p in items]
        entry={"test_coverage":str(len(test))+"/"+str(len(splits["test"])),
               "cal_coverage":str(len(cal))+"/"+str(len(splits["cal"])),
               "raw":diagnostics(attach(test)),
               "bins_raw":calibration_bins(attach(test)),
               "point_forecast_versus_posterior_mean":statistics.mean(
                    abs(p-post[("test",str(r["pair_id"]))][baseprior]["p_latent_A_superior"])
                    for r,p in test)}
        if len(cal)>=5:
            alpha=shrinkage([p for _,p in cal],[int(r["winner"]=="A") for r,_ in cal])
            entry["shrinkage_alpha"]=alpha
            entry["shrunk"]=diagnostics(attach(test,alpha))
            entry["bins_shrunk"]=calibration_bins(attach(test,alpha))
        report["systems"][label]=entry
    root=Path(output);root.mkdir(parents=True,exist_ok=True)
    (root/"summary.json").write_text(json.dumps(report,indent=2),encoding="utf8")
    with (root/"observation_uncertainty.csv").open("w",newline="",encoding="utf8") as f:
        columns=["stage","pair_id","prior","observed_winner",
                 "p_latent_A_superior","p_repeat_A_wins_approx",
                 "mean_ctr_difference","ctr_difference_ci_low","ctr_difference_ci_high"]
        writer=csv.DictWriter(f,fieldnames=columns);writer.writeheader()
        for stage,pid in truth:
            row=truth[(stage,pid)]
            for prior in priors:
                p=post[(stage,pid)][str(prior)]
                writer.writerow({"stage":stage,"pair_id":pid,"prior":prior,
                    "observed_winner":row["winner"],
                    "p_latent_A_superior":p["p_latent_A_superior"],
                    "p_repeat_A_wins_approx":p["p_repeat_A_wins_approx"],
                    "mean_ctr_difference":p["mean_ctr_difference"],
                    "ctr_difference_ci_low":p["ctr_difference_95ci"][0],
                    "ctr_difference_ci_high":p["ctr_difference_95ci"][1]})
    lines=["# Calibration experiment 1 — observation uncertainty and model confidence",
           "","**Exploratory/retrospective**; prior test labels were already inspected.",
           "All posterior quantities condition on the observed clicks; they are *not* independent labels.",
           "","## Observation uncertainty", ""]
    for stage,values in report["outcome_uncertainty"].items():
        for prior,item in values.items():
            lines.append("- %s, Beta(%s,%s): %.1f%% latent-superiority posteriors in [0.2,0.8], %.1f%% in [0.4,0.6]"%
                (stage,prior,prior,100*item["fraction_posterior_between_0_2_and_0_8"],
                 100*item["fraction_posterior_between_0_4_and_0_6"]))
    lines+=["","## Model forecast diagnostics","",
        "| System | Test coverage | Raw Brier | Shrunk Brier | Raw mean p(A) | Mean absolute confidence |",
        "|---|---:|---:|---:|---:|---:|"]
    for label,item in report["systems"].items():
        a=item["raw"]; b=item.get("shrunk") or {}
        lines.append("| %s | %s | %.4f | %s | %.3f | %.3f |"%
           (label,item["test_coverage"],a["brier"],
            ("%.4f"%b["brier"]) if b else "n/a",
            a["mean_forecast"],a["mean_absolute_confidence_from_half"]))
    lines+=["","## Interpretation guardrails",""]+["- "+w for w in report["warnings"]]
    lines+=["","## Outputs","","- summary.json: per-model diagnostics and equal-count reliability bins",
           "- observation_uncertainty.csv: posterior probabilities and CTR difference intervals under both priors",
           "- These files do not constitute a new confirmatory benchmark."]
    (root/"report.md").write_text("\n".join(lines)+"\n",encoding="utf8")
    return report
