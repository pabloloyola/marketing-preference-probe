"""Offline diagnostics for A/B headline predictions; standard library only."""
import csv
import html
import json
import math
import random
import statistics as stats
from collections import defaultdict
from pathlib import Path

def read(path):
    with open(path, encoding="utf-8") as stream:
        for line in stream:
            if line.strip():
                yield json.loads(line)

def mean(values):
    return stats.mean(values)

def bootstrap(groups, other=None, draws=2000, seed=2026):
    """Pair-cluster bootstrap; paired when other is provided."""
    ids=sorted(set(groups) & (set(other) if other is not None else set(groups)))
    if not ids:
        return None
    differences=[mean(groups[k])-(mean(other[k]) if other is not None else 0) for k in ids]
    rng=random.Random(seed)
    results=sorted(mean(rng.choices(differences,k=len(ids))) for _ in range(draws))
    return [mean(differences),results[int(.025*draws)],results[int(.975*draws)],len(ids)]

def svg_bar(path, title, data, max_value=1):
    height=80+54*len(data)
    lines=[f'<svg xmlns="http://www.w3.org/2000/svg" width="790" height="{height}">',
           '<rect width="100%" height="100%" fill="white"/>',
           f'<text x="20" y="30" font-family="sans-serif" font-size="19">{html.escape(title)}</text>']
    for i,(name,value) in enumerate(data):
        y=55+i*54
        lines.extend([f'<text x="20" y="{y+18}" font-family="sans-serif" font-size="13">{html.escape(name)}</text>',
                      f'<rect x="230" y="{y}" width="440" height="26" fill="#eee"/>',
                      f'<rect x="230" y="{y}" width="{440*min(1,max(0,value/max_value)):.1f}" height="26" fill="#527db8"/>',
                      f'<text x="687" y="{y+18}" font-family="sans-serif" font-size="13">{value:.4f}</text>'])
    Path(path).write_text("\n".join(lines+['</svg>']),encoding="utf-8")

def csv_out(path, rows, columns):
    with open(path,"w",newline="",encoding="utf-8") as stream:
        w=csv.DictWriter(stream,fieldnames=columns)
        w.writeheader()
        w.writerows(rows)

def write_report(pairs_path, predictions_path, output_dir):
    pairs={p["pair_id"]:p for p in read(pairs_path)}
    records=list(read(predictions_path))
    models=defaultdict(list)
    for rec in records:
        if rec.get("pair_id") not in pairs or "p_A" not in rec:
            continue
        pair=pairs[rec["pair_id"]]
        y=int(pair["winner"]=="A")
        p=max(1e-9,min(1-1e-9,float(rec["p_A"])))
        models[rec["model_key"]].append({
            "id":rec["pair_id"],"p":p,"correct":int((p>=.5)==bool(y)),
            "brier":(p-y)**2,"logloss":-y*math.log(p)-(1-y)*math.log(1-p),
            "confidence":max(p,1-p),"margin":abs(float(pair["ctr_A"])-float(pair["ctr_B"])),
            "pair":pair,"latency":rec.get("latency_s"),"cost":rec.get("reported_cost_usd")})
    if not models:
        raise ValueError("No matching model predictions; check --pairs and --predictions")
    dest=Path(output_dir)
    dest.mkdir(parents=True,exist_ok=True)
    summary={}
    grouped={}
    confidence=[]
    margin=[]
    mistakes=[]
    for model,rs in sorted(models.items()):
        g=defaultdict(list)
        for r in rs:
            g[r["id"]].append(r["correct"])
        grouped[model]=g
        ci=bootstrap(g)
        latency=[float(r["latency"]) for r in rs if r["latency"] is not None]
        costs=[float(r["cost"]) for r in rs if r["cost"] is not None]
        summary[model]={"n":len(rs),"unique_pairs":len(g),
                        "accuracy":mean([r["correct"] for r in rs]),
                        "accuracy_ci95":ci[1:3],
                        "brier":mean([r["brier"] for r in rs]),
                        "logloss":mean([r["logloss"] for r in rs]),
                        "mean_latency_s":mean(latency) if latency else None,
                        "reported_cost_usd":sum(costs),"cost_coverage":f"{len(costs)}/{len(rs)}"}
        for i in range(5):
            lo=.5+.1*i
            sub=[r for r in rs if lo<=r["confidence"]<lo+.1 or (i==4 and r["confidence"]==1)]
            if sub:
                confidence.append({"model":model,"bin_start":round(lo,2),"n":len(sub),
                  "mean_confidence":mean([r["confidence"] for r in sub]),
                  "observed_accuracy":mean([r["correct"] for r in sub])})
        for label,lo,hi in [("0-0.1pp",0,.001),("0.1-0.5pp",.001,.005),
                             ("0.5-1pp",.005,.01),(">1pp",.01,float("inf"))]:
            sub=[r for r in rs if lo<=r["margin"]<hi]
            if sub:
                margin.append({"model":model,"ctr_margin":label,"n":len(sub),
                  "accuracy":mean([r["correct"] for r in sub]),
                  "brier":mean([r["brier"] for r in sub])})
        incorrect=sorted((r for r in rs if not r["correct"]),key=lambda r:-r["confidence"])
        for r in incorrect[:5]:
            pair=r["pair"]
            mistakes.append({"model":model,"pair_id":r["id"],
              "confidence":round(r["confidence"],4),"predicted":"A" if r["p"]>=.5 else "B",
              "observed":pair["winner"],"ctr_A":pair["ctr_A"],"ctr_B":pair["ctr_B"],
              "headline_A":pair["headline_A"],"headline_B":pair["headline_B"]})
    names=sorted(summary)
    comparisons={}
    for i,name in enumerate(names):
        for other in names[i+1:]:
            comparisons[f"{name} minus {other}"]=bootstrap(grouped[name],grouped[other])
    svg_bar(dest/"accuracy.svg","Winner accuracy",[(n,summary[n]["accuracy"]) for n in names])
    svg_bar(dest/"brier.svg","Brier score (lower better)",
            [("constant 0.5",.25)]+[(n,summary[n]["brier"]) for n in names],max_value=.75)
    csv_out(dest/"confidence_bins.csv",confidence,
            ["model","bin_start","n","mean_confidence","observed_accuracy"])
    csv_out(dest/"ctr_margin_bins.csv",margin,["model","ctr_margin","n","accuracy","brier"])
    (dest/"confident_errors.json").write_text(json.dumps(mistakes,indent=2,ensure_ascii=False),encoding="utf-8")
    all_results={"pairs_in_dataset":len(pairs),"models":summary,
                 "paired_accuracy_differences":comparisons,
                 "constant_half_baseline":{"brier":.25,"logloss":math.log(2)}}
    (dest/"summary.json").write_text(json.dumps(all_results,indent=2),encoding="utf-8")
    lines=["# Marketing Preference Probe — Analysis v2","",
           "Offline report: no new API calls or charges.","",
           "## Summary","",
           "| Model | Predictions | Accuracy (95% pair bootstrap interval) | Brier | Log loss | Latency s | Cost USD |",
           "|---|---:|---:|---:|---:|---:|---:|"]
    for n in names:
        s=summary[n]
        lines.append(f'| {n} | {s["n"]} | {s["accuracy"]:.3f} [{s["accuracy_ci95"][0]:.3f}, {s["accuracy_ci95"][1]:.3f}] | {s["brier"]:.3f} | {s["logloss"]:.3f} | {s["mean_latency_s"]:.3f} | {s["reported_cost_usd"]:.6f} |')
    lines+=["","![Winner accuracy](accuracy.svg)","","![Brier score](brier.svg)","",
            "## Simple baseline","",
            "A constant P(A wins)=0.5 has Brier 0.25 and log loss 0.6931. Its winner accuracy depends on arbitrary tie-breaking.",
            "","## Paired comparisons",""]
    for name,values in comparisons.items():
        lines.append(f"- {name}: accuracy difference {values[0]:+.3f}, 95% bootstrap interval [{values[1]:+.3f}, {values[2]:+.3f}], common pairs {values[3]}")
    lines+=["","## Diagnostic files","",
            "- confidence_bins.csv: confidence vs correctness by model (count per bin).",
            "- ctr_margin_bins.csv: results grouped by observed CTR gap in percentage points.",
            "- confident_errors.json: up to five high-confidence wrong choices per model.",
            "","## Caveats","",
            "- Predicting which A/B arm had higher CTR is **not** estimating individual click probabilities or a population preference share.",
            "- Winner labels are noisy for near-tied CTRs; margin bins are descriptive, not hypothesis tests.",
            "- Calibration bins need adequate sample sizes and do not establish calibration to human CTR.",
            "- The bootstrap resamples test IDs; this does not establish external validity or avoid public-data leakage.",
            "- Multiple predictions for the same pair are grouped for uncertainty, not counted as independent experiments.",
            "- Do not use these evaluation pairs to tune prompts and then report scores as held-out performance.",
            ""]
    (dest/"report.md").write_text("\n".join(lines),encoding="utf-8")
    return all_results
