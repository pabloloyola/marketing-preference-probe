import argparse
import json
import math
import os
import statistics
from pathlib import Path
from .api import MODELS, APIError, predict
from .dataset import prepare, load_pairs
from .report import write_report
from .experiment2 import prepare2, run2, analyze2
from .experiment3 import run3, analyze3, preview3
from .retrieval_v2 import make_pool, preview as preview_v2, run as run_v2, analyze as analyze_v2, METHODS
from .final_diagnostic import prepare_final, report_final
from .calibration1 import analyze as analyze_calibration1


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
    v=sub.add_parser('report', help='Offline analysis v2: plots, baselines and bootstrap intervals')
    v.add_argument('--pairs', default='data/pairs.jsonl')
    v.add_argument('--predictions', default='runs/predictions.jsonl')
    v.add_argument('--out', default='runs/analysis_v2')
    e=sub.add_parser('prepare2', help='Disjoint train/cal/test A/B splits')
    e.add_argument('--csv', required=True)
    e.add_argument('--exclude', default='data/pairs.jsonl')
    e.add_argument('--out', default='data/experiment2')
    e.add_argument('--train', type=int, default=80)
    e.add_argument('--cal', type=int, default=40)
    e.add_argument('--test', type=int, default=80)
    e.add_argument('--seed', type=int, default=2027)
    e=sub.add_parser('run2', help='Zero-shot vs behavior-grounded models; dry-run default')
    e.add_argument('--data', default='data/experiment2')
    e.add_argument('--out', default='runs/experiment2_predictions.jsonl')
    e.add_argument('--models', nargs='+', choices=MODELS, default=list(MODELS))
    e.add_argument('--conditions', nargs='+', choices=['zero','fewshot'], default=['zero','fewshot'])
    e.add_argument('--stages', nargs='+', choices=['cal','test'], default=['cal','test'])
    e.add_argument('--examples', type=int, default=5)
    e.add_argument('--limit', type=int, default=0)
    e.add_argument('--budget', type=float, default=.50)
    e.add_argument('--max-calls', type=int, default=600)
    e.add_argument('--live', action='store_true')
    e=sub.add_parser('analyze2', help='Compare grounded, ungrounded and calibrated results offline')
    e.add_argument('--data', default='data/experiment2')
    e.add_argument('--predictions', default='runs/experiment2_predictions.jsonl')
    e.add_argument('--out', default='runs/experiment2_report.json')
    e=sub.add_parser('preview3', help='Preview five retrieved examples; offline')
    e.add_argument('--data', default='data/experiment2')
    e.add_argument('--k', type=int, default=5)
    e=sub.add_parser('run3', help='Run retrieval-grounded models; dry-run default')
    e.add_argument('--data', default='data/experiment2')
    e.add_argument('--out', default='runs/experiment3_predictions.jsonl')
    e.add_argument('--models', nargs='+', choices=MODELS, default=list(MODELS))
    e.add_argument('--k', type=int, default=5)
    e.add_argument('--limit', type=int, default=0)
    e.add_argument('--budget', type=float, default=.5)
    e.add_argument('--max-calls', type=int, default=360)
    e.add_argument('--live', action='store_true')
    e=sub.add_parser('analyze3', help='Compare retrieval against Experiment 2')
    e.add_argument('--data', default='data/experiment2')
    e.add_argument('--retrieved', default='runs/experiment3_predictions.jsonl')
    e.add_argument('--previous', default='runs/experiment2_predictions.jsonl')
    e.add_argument('--out', default='runs/experiment3_report.json')
    e.add_argument('--k', type=int, default=5)
    e=sub.add_parser('prepare-retrieval',help='Expand corpus without test/calibration leakage')
    e.add_argument('--csv',required=True)
    e.add_argument('--data',default='data/experiment2')
    e.add_argument('--exploratory',default='data/pairs.jsonl')
    e.add_argument('--pool',default='data/retrieval_v2/pool.jsonl')
    e.add_argument('--size',type=int,default=2000)
    e.add_argument('--seed',type=int,default=3026)
    e=sub.add_parser('preview-retrieval',help='Inspect nearest comparisons; may download local embedding model')
    e.add_argument('--pool',default='data/retrieval_v2/pool.jsonl')
    e.add_argument('--data',default='data/experiment2')
    e.add_argument('--method',choices=METHODS,default='structure')
    e.add_argument('--cache',default='data/retrieval_v2/embeddings.npz')
    e.add_argument('--k',type=int,default=5)
    e=sub.add_parser('run-retrieval',help='Compare retrieval methods on held-out cases; dry-run default')
    e.add_argument('--pool',default='data/retrieval_v2/pool.jsonl')
    e.add_argument('--data',default='data/experiment2')
    e.add_argument('--method',choices=METHODS,default='structure')
    e.add_argument('--cache',default='data/retrieval_v2/embeddings.npz')
    e.add_argument('--k',type=int,default=5)
    e.add_argument('--limit',type=int,default=0)
    e.add_argument('--models',nargs='+',choices=MODELS,default=list(MODELS))
    e.add_argument('--out',default=None)
    e.add_argument('--budget',type=float,default=.50)
    e.add_argument('--max-calls',type=int,default=360)
    e.add_argument('--live',action='store_true')
    e=sub.add_parser('analyze-retrieval',help='Offline report vs Experiment 2')
    e.add_argument('--pool',default='data/retrieval_v2/pool.jsonl')
    e.add_argument('--data',default='data/experiment2')
    e.add_argument('--method',choices=METHODS,default='structure')
    e.add_argument('--cache',default='data/retrieval_v2/embeddings.npz')
    e.add_argument('--k',type=int,default=5)
    e.add_argument('--predictions',default=None)
    e.add_argument('--previous',default='runs/experiment2_predictions.jsonl')
    e.add_argument('--out',default=None)
    e=sub.add_parser('prepare-final',help='Prepare fresh disjoint held-out diagnostics')
    e.add_argument('--csv',required=True)
    e.add_argument('--pool',default='data/retrieval_v2/pool.jsonl')
    e.add_argument('--data',default='data/experiment2')
    e.add_argument('--exploratory',default='data/pairs.jsonl')
    e.add_argument('--out',default='data/final_holdout')
    e.add_argument('--cal',type=int,default=40)
    e.add_argument('--test',type=int,default=120)
    e.add_argument('--seed',type=int,default=5091)
    e=sub.add_parser('analyze-final',help='Fully offline retrieval vs choice logit on fresh tests')
    e.add_argument('--pool',default='data/retrieval_v2/pool.jsonl')
    e.add_argument('--holdout',default='data/final_holdout')
    e.add_argument('--data',default='data/experiment2')
    e.add_argument('--exploratory',default='data/pairs.jsonl')
    e.add_argument('--cache',default='data/final_holdout/embeddings.npz')
    e.add_argument('--out',default='runs/final_diagnostic')
    e.add_argument('--k',type=int,default=5)
    e=sub.add_parser('calibration1',help='Offline CTR observation uncertainty and model confidence')
    e.add_argument('--data',default='data/experiment2')
    e.add_argument('--predictions',nargs='+',default=[
        'runs/experiment2_predictions.jsonl',
        'runs/retrieval_v2_semantic.jsonl',
        'runs/retrieval_v2_hybrid.jsonl'])
    e.add_argument('--out',default='runs/calibration_1')
    e.add_argument('--draws',type=int,default=4000)
    e.add_argument('--repeat-n',type=int,default=1000)
    e.add_argument('--seed',type=int,default=2027)
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
    elif args.cmd=='report':
        write_report(args.pairs, args.predictions, args.out)
        print(f'Offline report written to {args.out}/report.md (no API calls)')
    elif args.cmd=='prepare2':
        print(json.dumps(prepare2(args.csv,args.exclude,args.out,args.train,args.cal,args.test,args.seed),indent=2))
    elif args.cmd=='run2':
        run2(args.data,args.out,args.models,args.conditions,args.stages,args.examples,args.limit,args.live,args.budget,args.max_calls)
    elif args.cmd=='analyze2':
        result=analyze2(args.data,args.predictions,args.out)
        print(json.dumps(result,indent=2))
    elif args.cmd=='preview3':
        preview3(args.data,args.k)
    elif args.cmd=='run3':
        run3(args.data,args.out,args.models,args.k,args.limit,args.live,args.budget,args.max_calls)
    elif args.cmd=='analyze3':
        print(json.dumps(analyze3(args.data,args.retrieved,args.previous,args.out,args.k),indent=2))
    elif args.cmd=='prepare-retrieval':
        print(json.dumps(make_pool(args.csv,args.data,args.exploratory,args.pool,args.size,args.seed),indent=2))
    elif args.cmd=='preview-retrieval':
        print(json.dumps(preview_v2(args.pool,args.data,args.method,args.k,args.cache),indent=2,ensure_ascii=False))
    elif args.cmd=='run-retrieval':
        run_v2(args.pool,args.data,args.method,args.k,args.limit,args.models,args.out,args.cache,args.budget,args.max_calls,args.live)
    elif args.cmd=='analyze-retrieval':
        print(json.dumps(analyze_v2(args.pool,args.data,args.method,args.k,args.predictions,args.previous,args.out,args.cache),indent=2))
    elif args.cmd=='prepare-final':
        print(json.dumps(prepare_final(args.csv,args.pool,args.data,args.exploratory,args.out,args.cal,args.test,args.seed),indent=2))
    elif args.cmd=='analyze-final':
        results=report_final(args.pool,args.holdout,args.data,args.exploratory,args.cache,args.out,args.k)
        print('Created '+args.out+'/report.md and summary.json (no paid API calls)')
        print(json.dumps(results['metrics'],indent=2))
    elif args.cmd=='calibration1':
        result=analyze_calibration1(args.data,args.predictions,args.out,draws=args.draws,
                                    repeat_n=args.repeat_n,seed=args.seed)
        print('Offline outputs written to '+args.out)
        print(json.dumps({'split_sizes':result['split_sizes'],
                          'outcome_uncertainty':result['outcome_uncertainty'],
                          'systems':{k:{'coverage':v['test_coverage'],
                                        'raw_brier':v['raw']['brier'],
                                        'shrunk_brier':v.get('shrunk',{}).get('brier')}
                                     for k,v in result['systems'].items()}},indent=2))
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
