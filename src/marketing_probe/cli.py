import argparse
import json
import math
import os
import statistics
from pathlib import Path
from .api import MODELS, APIError, predict
from .dataset import prepare, load_pairs


def append_jsonl(path, obj):
    with Path(path).open("a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")


def make_parser():
    p=argparse.ArgumentParser(description="Marketing preference probe: real A/B labels, model predictions, cost guard")
    sub=p.add_subparsers(dest="cmd", required=True)
    d=sub.add_parser("prepare", help="Prepare headline-only pairs from downloaded Upworthy exploratory CSV")
    d.add_argument("--csv", required=True); d.add_argument("--output", default="data/pairs.jsonl")
    d.add_argument("--n", type=int, default=100); d.add_argument("--seed", type=int, default=42)
    d.add_argument("--min-impressions", type=int, default=1000)
    s=sub.add_parser("smoke", help="One request per model; requires --live")
    s.add_argument("--live", action="store_true")
    s.add_argument("--models", nargs="+", choices=MODELS, default=list(MODELS))
    r=sub.add_parser("run", help="Run benchmark. Defaults to no API calls.")
    r.add_argument("--pairs", default="data/pairs.jsonl"); r.add_argument("--out", default="runs/predictions.jsonl")
    r.add_argument("--models", nargs="+", choices=MODELS, default=list(MODELS))
    r.add_argument("--reps", type=int, default=1); r.add_argument("--limit", type=int, default=10)
    r.add_argument("--budget", type=float, default=1.0)
    r.add_argument("--reserve-per-call", type=float, default=.003, help="Guardrail reserve when actual cost is missing or not yet known")
    r.add_argument("--max-calls", type=int, default=300)
    r.add_argument("--live", action="store_true", help="Required to send paid requests")
    a=sub.add_parser("analyze"); a.add_argument("--pairs", default="data/pairs.jsonl"); a.add_argument("--predictions", default="runs/predictions.jsonl")
    return p


def analyze(args):
    pairs={p["pair_id"]:p for p in load_pairs(args.pairs)}
    rows=[]
    with open(args.predictions, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rec=json.loads(line)
                if rec.get("pair_id") in pairs and "p_A" in rec:
                    rows.append(rec)
    by_model={}
    for row in rows:
        by_model.setdefault(row["model_key"], []).append(row)
    print(f"Pairs in dataset: {len(pairs)}")
    for model, recs in by_model.items():
        metrics=[]
        for r in recs:
            pair=pairs[r["pair_id"]]
            truth=1.0 if pair["winner"]=="A" else 0.0
            probability=max(1e-6,min(1-1e-6,float(r["p_A"])))
            metrics.append((int((probability>=.5)==bool(truth)), (probability-truth)**2,
                            -truth*math.log(probability)-(1-truth)*math.log(1-probability)))
        costs=[float(r["reported_cost_usd"]) for r in recs if r.get("reported_cost_usd") is not None]
        print(json.dumps({"model": model, "predictions":len(recs), "unique_pairs":len({r['pair_id'] for r in recs}),
            "winner_accuracy":round(statistics.mean(x[0] for x in metrics),4),
            "brier_winner":round(statistics.mean(x[1] for x in metrics),4),
            "log_loss_winner":round(statistics.mean(x[2] for x in metrics),4),
            "mean_latency_s":round(statistics.mean(r["latency_s"] for r in recs),4),
            "reported_cost_usd":round(sum(costs),7), "cost_coverage":f"{len(costs)}/{len(recs)}"}, indent=2))
    print("WARNING: These are winner-probability metrics, not calibration to human click-share. Near ties have noisy labels. Multiple repetitions are not independent experiments.")


def main():
    args=make_parser().parse_args()
    if args.cmd=="prepare":
        count, eligible=prepare(args.csv,args.output,args.n,args.seed,args.min_impressions)
        print(f"Wrote {count} REAL pairs from {eligible} eligible within-test comparisons to {args.output}")
    elif args.cmd=="smoke":
        if not args.live:
            print("DRY RUN: would test", ", ".join(args.models), "; add --live to spend money")
            return
        for m in args.models:
            try:
                res=predict(m,"A New Study Changes Everything We Thought About Sleep","Why Some People Wake Up Before Their Alarm")
                print(json.dumps(res,indent=2))
            except APIError as e:
                print(f"ERROR {m}: {e}")
    elif args.cmd=="analyze":
        analyze(args)
    elif args.cmd=="run":
        if args.limit<1 or args.reps<1 or args.max_calls<1 or args.budget<=0 or args.reserve_per_call<=0:
            raise SystemExit("Invalid run limits")
        pairs=list(load_pairs(args.pairs))[:args.limit]
        plan=[(p,m,rep) for p in pairs for m in args.models for rep in range(args.reps)]
        print(f"Planned: {len(plan)} calls, {len(pairs)} pairs; budget ${args.budget:.2f}; max calls {args.max_calls}")
        if not args.live:
            print("DRY RUN: no requests sent. Add --live to call OpenRouter.")
            return
        if not os.getenv("OPENROUTER_API_KEY"):
            raise SystemExit("OPENROUTER_API_KEY not set; no requests sent")
        out=Path(args.out); out.parent.mkdir(parents=True,exist_ok=True)
        already=set(); spent=0.0; reserved_unknown=0.0; prior_calls=0
        if out.exists():
            for line in out.read_text().splitlines():
                if not line.strip(): continue
                old=json.loads(line)
                if "p_A" in old:
                    already.add((old["pair_id"],old["model_key"],old["rep"]))
                    prior_calls+=1
                    cost=old.get("reported_cost_usd")
                    if cost is None: reserved_unknown+=args.reserve_per_call
                    else: spent+=max(0,float(cost))
        calls=0
        for p,m,rep in plan:
            key=(p["pair_id"],m,rep)
            if key in already: continue
            if spent+reserved_unknown+args.reserve_per_call > args.budget or calls+prior_calls>=args.max_calls:
                print("Stopped at configured spending/call guardrail")
                break
            try:
                result=predict(m,p["headline_A"],p["headline_B"])
            except APIError as e:
                print(f"Request stopped after {calls} successful calls: {e}")
                break
            result.update({"pair_id":p["pair_id"],"rep":rep})
            append_jsonl(out,result)
            calls+=1
            if result.get("reported_cost_usd") is None:
                reserved_unknown+=args.reserve_per_call
            else:
                spent+=max(0,float(result["reported_cost_usd"]))
            print(f"{calls:3} {m:15} {p['pair_id']} p(A)={result['p_A']:.3f} latency={result['latency_s']:.2f}s cumulative=${spent+reserved_unknown:.5f}")
        print(f"Finished: new successful calls={calls}, reported_cost=${spent:.5f}, unknown-cost reserve=${reserved_unknown:.5f}; file={out}")

if __name__=="__main__":
    main()
