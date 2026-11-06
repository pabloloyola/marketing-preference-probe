"""Calibration Experiment 2: retrospective error-by-ambiguity and synthetic controls.

All analyses offline. Posterior ambiguity is computed from observed outcomes
and MUST NOT be used as a model input or a new forecast ground-truth label.
"""
import csv
import json
import math
import random
import statistics
from collections import defaultdict
from pathlib import Path

from .calibration1 import observations, collect_predictions
from .dataset import load_pairs

def summarize_cases(rows):
    if not rows:
        return {"n": 0}
    n=len(rows)
    wrong=[r for r in rows if r["predicted"] != r["winner"]]
    def mean(key, population=rows):
        return statistics.mean(r[key] for r in population) if population else None
    return {"n":n, "mistakes":len(wrong), "accuracy":1-len(wrong)/n,
            "mean_confidence":mean("confidence"),
            "mean_confidence_on_errors":mean("confidence",wrong),
            "mean_brier":mean("brier"),
            "mean_model_posterior_disagreement":mean("posterior_disagreement"),
            "mean_posterior_ambiguity":mean("ambiguity"),
            "confident_errors_at_0_8":sum(r["confidence"]>=.8 and
                r["predicted"]!=r["winner"] for r in rows)}

def strata_for(posterior):
    ambiguity=2*min(posterior,1-posterior)
    return ("high_ambiguity" if ambiguity>=.6 else
            "intermediate_ambiguity" if ambiguity>=.2 else "low_ambiguity")

def model_groups(pairs, forecasts, posteriors):
    grouped=defaultdict(list)
    for pair in pairs:
        pid=str(pair["pair_id"])
        if pid not in forecasts:
            continue
        p=float(forecasts[pid])
        posterior=posteriors[pid]
        label=pair["winner"]
        predicted="A" if p>.5 else ("B" if p<.5 else None)
        # Exact 0.5 predictions abstain; do not label them wrong by a tie convention.
        if predicted is None:
            continue
        correct=int(predicted==label)
        row={"pair_id":pid,"predicted":predicted,"winner":label,
             "p_A":p,"confidence":max(p,1-p),
             "brier":(p-int(label=="A"))**2,
             "ambiguity":2*min(posterior,1-posterior),
             "posterior_disagreement":abs(p-posterior),
             "correct":correct}
        grouped[strata_for(posterior)].append(row)
        grouped["all_non_tied"].append(row)
    return {name:summarize_cases(rows) for name,rows in grouped.items()}

def synthetic_control(n=900, draws=500, seed=2027, prior=.5,
                      impressions=(100,1000,10000), effects=(0.,.002,.01,.04)):
    """Known latent CTR positive controls under a declared synthetic DGP.

    Population CTR_B varies, effect size is signed at random, and observed
    clicks are binomial draws. Posterior coverage uses independent Beta priors.
    This tests observation-model behavior, NOT forecasting by the actual LLMs.
    """
    if n<20 or draws<100 or prior<=0 or not impressions or not effects:
        raise ValueError("Invalid simulation settings")
    rng=random.Random(seed)
    cells=defaultdict(list)
    for k in range(n):
        sample_n=impressions[k%len(impressions)]
        effect=effects[(k//len(impressions))%len(effects)]
        q_b=rng.uniform(.015,.14)
        delta=effect*(1 if rng.random()<.5 else -1)
        q_a=q_b+delta
        if not 0<q_a<1:
            raise ValueError("Synthetic CTR out of range")
        # Built-in binomial simulation, no external dependency.
        ca=sum(rng.random()<q_a for _ in range(sample_n))
        cb=sum(rng.random()<q_b for _ in range(sample_n))
        pair={"pair_id":str(k),"clicks_A":ca,"impressions_A":sample_n,
              "clicks_B":cb,"impressions_B":sample_n}
        result=observations(pair,prior=prior,draws=draws,
                            seed=seed+20000+k,repeat_n=sample_n)
        interval=result["ctr_difference_95ci"]
        true_delta=q_a-q_b
        cell=(sample_n,effect)
        cells[cell].append({
            "covered":int(interval[0]<=true_delta<=interval[1]),
            "latent_prob":result["p_latent_A_superior"],
            "true_A_superior":int(true_delta>0),
            "observed_A_superior":int(ca/sample_n>cb/sample_n),
            "interval_width":interval[1]-interval[0],
            "true_delta":true_delta
        })
    output={}
    for (size,effect),rows in sorted(cells.items()):
        output[f"n={size};abs_delta={effect}"]={
            "n":len(rows),
            "ctr_difference_95ci_coverage":statistics.mean(x["covered"] for x in rows),
            "mean_95ci_width":statistics.mean(x["interval_width"] for x in rows),
            "observed_win_matches_latent_superiority":statistics.mean(
                x["true_A_superior"]==x["observed_A_superior"] for x in rows),
            "mean_posterior_superiority_when_A_really_better":(
                statistics.mean(x["latent_prob"] for x in rows if x["true_A_superior"])
                if any(x["true_A_superior"] for x in rows) else None),
        }
    return {"description":"Synthetic beta-binomial posterior recovery; simulated CTRs are known",
            "n":n,"draws_per_posterior":draws,"seed":seed,
            "prior_shape":prior,"impression_sizes":list(impressions),
            "absolute_effect_sizes":list(effects),"cells":output,
            "limitation":"Equal underlying CTRs (delta=0) have no strict A-winner; observed-win agreement uses strict comparison and is not a classifier accuracy claim."}

def analyze(directory="data/experiment2",
            prediction_paths=("runs/experiment2_predictions.jsonl",
                              "runs/retrieval_v2_semantic.jsonl",
                              "runs/retrieval_v2_hybrid.jsonl"),
            out="runs/calibration_2",prior=.5,
            draws=1500,synthetic_n=900,synthetic_draws=500,seed=2027):
    if draws<100 or synthetic_draws<100:raise ValueError("At least 100 draws required")
    tests=list(load_pairs(Path(directory)/"test.jsonl"))
    ids=[str(p["pair_id"]) for p in tests]
    if len(set(ids))!=len(ids):raise ValueError("Duplicate pair IDs")
    truth={("test",str(p["pair_id"])):p for p in tests}
    by=collect_predictions(prediction_paths,truth)
    post={str(p["pair_id"]):observations(
        p,prior=prior,draws=draws,seed=seed+i*1009)["p_latent_A_superior"]
          for i,p in enumerate(tests)}
    systems={}
    for name,values in sorted(by.items()):
        pred={pid:p for (stage,pid),p in values.items() if stage=="test"}
        systems[name]={"coverage":f"{len(pred)}/{len(tests)}",
                       "groups":model_groups(tests,pred,post)}
    positive=synthetic_control(n=synthetic_n,draws=synthetic_draws,seed=seed)
    report={"status":"Exploratory retrospective analysis plus synthetic positive control",
            "observed_tests":len(tests),"prior":prior,"draws":draws,
            "models":systems,"synthetic_control":positive,
            "caveats":[
                "Ambiguity strata are computed from held-out clicks: they are post-outcome and descriptive, not prospective subgroups.",
                "Posterior agreement is not independent forecast validation; it uses clicks from the same cases.",
                "Observed hard-winner Brier scores remain sensitive to sampling noise.",
                "Synthetic positive controls validate the Bayesian observation code under simplified binomial assumptions, not the model predictions.",
                "A/B swapping is not tested on original model calls; additional inference would be needed to audit order effects.",
                "Tests have been used for exploratory research; do not treat this as a confirmatory evaluation."
            ]}
    output=Path(out);output.mkdir(parents=True,exist_ok=True)
    (output/"summary.json").write_text(json.dumps(report,indent=2),encoding="utf8")
    with (output/"error_by_ambiguity.csv").open("w",newline="",encoding="utf8") as f:
        fields=["system","group","n","mistakes","accuracy","mean_confidence",
                "mean_confidence_on_errors","mean_brier",
                "mean_model_posterior_disagreement","mean_posterior_ambiguity",
                "confident_errors_at_0_8"]
        writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader()
        for system,data in systems.items():
            for group,values in data["groups"].items():
                writer.writerow({"system":system,"group":group,**values})
    with (output/"synthetic_coverage.csv").open("w",newline="",encoding="utf8") as f:
        fields=["cell","n","ctr_difference_95ci_coverage","mean_95ci_width",
                "observed_win_matches_latent_superiority",
                "mean_posterior_superiority_when_A_really_better"]
        writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader()
        for name,data in positive["cells"].items():
            writer.writerow({"cell":name,**data})
    lines=["# Calibration Experiment 2 — observation noise and predictive errors","",
           "**Retrospective diagnostic; not an independent model validation.**","",
           "## Per-model mistakes by observation ambiguity","",
           "| Model | Group | n | Accuracy | Mean confidence | Brier | High-confidence mistakes |",
           "|---|---|---:|---:|---:|---:|---:|"]
    for name,entry in systems.items():
        for group in ("low_ambiguity","intermediate_ambiguity","high_ambiguity"):
            d=entry["groups"].get(group)
            if not d:continue
            lines.append(f"| {name} | {group} | {d['n']} | {d['accuracy']:.3f} | {d['mean_confidence']:.3f} | {d['mean_brier']:.4f} | {d['confident_errors_at_0_8']} |")
    lines+=["","## Synthetic positive-control coverage","",
            "| Known CTR setting | n | 95% posterior interval coverage | Mean width |",
            "|---|---:|---:|---:|"]
    for name,d in positive["cells"].items():
        lines.append(f"| {name} | {d['n']} | {d['ctr_difference_95ci_coverage']:.3f} | {d['mean_95ci_width']:.4f} |")
    lines+=["","## Caveats",""]+["- "+c for c in report["caveats"]]
    (output/"report.md").write_text("\n".join(lines)+"\n",encoding="utf8")
    return report
