"""Disjoint behavioral examples, calibration and test experiment (offline + OpenRouter)."""
import json, math, os, random, tempfile
from pathlib import Path
from collections import defaultdict
from .api import MODELS, APIError, predict
from .dataset import prepare, load_pairs

def save(path, rows):
    p=Path(path); p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(''.join(json.dumps(x,ensure_ascii=False)+'\n' for x in rows),encoding='utf-8')

def prepare2(csv, exclude='data/pairs.jsonl', out='data/experiment2',
             n_train=80,n_cal=40,n_test=80,seed=2027):
    original=list(load_pairs(exclude))
    used={str(p['test_id']) for p in original}
    if not used: raise ValueError('Exploratory pairs required for exclusion')
    n=n_train+n_cal+n_test
    if min(n_train,n_cal,n_test)<1: raise ValueError('Each split must be nonempty')
    with tempfile.TemporaryDirectory() as td:
        tmp=Path(td)/'candidates.jsonl'
        prepare(csv,tmp,n=n+len(used)+50,seed=seed)
        available=[x for x in load_pairs(tmp) if str(x['test_id']) not in used]
    if len(available)<n: raise ValueError('Not enough tests after excluding exploratory benchmark')
    available=available[:n]
    splits={'train':available[:n_train],
            'cal':available[n_train:n_train+n_cal],
            'test':available[n_train+n_cal:]}
    ids=[{str(x['test_id']) for x in rows} for rows in splits.values()]
    if any(ids[i]&ids[j] for i in range(3) for j in range(i+1,3)):
        raise ValueError('Overlap between splits')
    if any(x&used for x in ids): raise ValueError('Exploratory contamination')
    for name,rows in splits.items(): save(Path(out)/(name+'.jsonl'),rows)
    metadata={'train':n_train,'cal':n_cal,'test':n_test,'seed':seed,
              'excluded_exploratory_tests':len(used)}
    Path(out,'split_metadata.json').write_text(json.dumps(metadata,indent=2))
    return metadata

def examples_from(train,k):
    if not 1<=k<=len(train): raise ValueError('Invalid number of examples')
    groups={v:[r for r in train if r['winner']==v] for v in ('A','B')}
    rng=random.Random(113)
    for rows in groups.values(): rng.shuffle(rows)
    result=[]
    for i in range(k):
        desired='A' if i%2==0 else 'B'
        choices=groups[desired] or groups['B' if desired=='A' else 'A']
        r=choices.pop()
        result.append({'headline_A':r['headline_A'],'headline_B':r['headline_B'],
                       'observed_higher_ctr':r['winner']})
    return result

def feature(row):
    def values(s):
        w=s.lower().split()
        return [len(s)/100,len(w)/20,int('?' in s),int('!' in s),
                int(any(c.isdigit() for c in s)),int('you' in w)]
    a,b=values(row['headline_A']),values(row['headline_B'])
    return [1.]+[x-y for x,y in zip(a,b)]

def sigmoid(z): return 1/(1+math.exp(-max(-35,min(35,z))))

def train_logit(train):
    w=[0.]*len(feature(train[0]))
    for _ in range(1200):
        grad=[0.]*len(w)
        for r in train:
            x=feature(r); error=sigmoid(sum(a*b for a,b in zip(w,x)))-int(r['winner']=='A')
            for j,v in enumerate(x): grad[j]+=error*v
        for j in range(len(w)): w[j]-=.08*(grad[j]/len(train)+(.08*w[j] if j else 0))
    return w

def score(pairs):
    if not pairs:return None
    y=[int(r['winner']=='A') for r in pairs]
    p=[max(1e-8,min(1-1e-8,float(r['p_A']))) for r in pairs]
    return {'n':len(p),
            'accuracy':sum(int((a>=.5)==bool(b)) for a,b in zip(p,y))/len(p),
            'brier':sum((a-b)**2 for a,b in zip(p,y))/len(p),
            'logloss':sum(-b*math.log(a)-(1-b)*math.log(1-a) for a,b in zip(p,y))/len(p)}

def shrinkage(pred, truth):
    if len(pred)<5: raise ValueError('Calibration needs >=5 examples')
    d=[p-.5 for p in pred]
    den=sum(x*x for x in d)
    return max(0.,min(1.,sum(x*(y-.5) for x,y in zip(d,truth))/den)) if den else 0.

def run2(directory='data/experiment2',output='runs/experiment2_predictions.jsonl',
         models=None,conditions=('zero','fewshot'),stages=('cal','test'),examples=5,
         limit=0,live=False,budget=.50,max_calls=600,reserve_per_call=.005):
    models=list(models or MODELS)
    if not set(models)<=set(MODELS) or not set(conditions)<={'zero','fewshot'} or not set(stages)<={'cal','test'}:
        raise ValueError('Invalid selection')
    if budget<=0 or max_calls<=0 or reserve_per_call<=0 or limit<0: raise ValueError('Invalid limits')
    demos=examples_from(list(load_pairs(Path(directory)/'train.jsonl')),examples)
    tasks=[]
    for stage in stages:
        rows=list(load_pairs(Path(directory)/(stage+'.jsonl')))
        for row in (rows[:limit] if limit else rows):
            for model in models:
                for condition in conditions: tasks.append((stage,row,model,condition))
    p=Path(output)
    old=list(load_pairs(p)) if p.exists() else []
    done={(r['stage'],r['pair_id'],r['model_key'],r['condition']) for r in old if 'p_A' in r}
    remaining=sum((s,r['pair_id'],m,c) not in done for s,r,m,c in tasks)
    print(f'Planned {len(tasks)} tasks; {remaining} remaining; budget ${budget:.2f}; max calls {max_calls}')
    if not live:
        print('DRY RUN: no API requests. Supply --live to run.'); return
    if not os.getenv('OPENROUTER_API_KEY'): raise ValueError('OPENROUTER_API_KEY not set')
    spent=sum(float(r['reported_cost_usd']) for r in old if r.get('reported_cost_usd') is not None)
    unknown=reserve_per_call*sum(r.get('reported_cost_usd') is None for r in old)
    count=0; p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('a',encoding='utf-8') as file:
        for stage,row,model,condition in tasks:
            key=stage,row['pair_id'],model,condition
            if key in done: continue
            if spent+unknown+reserve_per_call>budget or len(old)+count>=max_calls:
                print('Stopped at best-effort spending/call guardrail');break
            try:
                res=predict(model,row['headline_A'],row['headline_B'],
                            examples=demos if condition=='fewshot' else None)
            except APIError as exc:
                print(f'API error; stopping: {exc}');break
            res.update({'stage':stage,'pair_id':row['pair_id'],'condition':condition,
                        'examples':examples if condition=='fewshot' else 0})
            file.write(json.dumps(res,ensure_ascii=False)+'\n'); file.flush()
            count+=1
            if res.get('reported_cost_usd') is None:unknown+=reserve_per_call
            else:spent+=max(0,float(res['reported_cost_usd']))
            print(f'{count:3} {stage:4} {model:16} {condition:7} p(A)={res["p_A"]:.2f} accounted=${spent+unknown:.5f}')
    print(f'New successful calls={count}; recorded/buffered cost ${spent+unknown:.5f}')

def analyze2(directory='data/experiment2',predictions='runs/experiment2_predictions.jsonl',
             output='runs/experiment2_report.json'):
    root=Path(directory)
    train=list(load_pairs(root/'train.jsonl'))
    truth={stage:{r['pair_id']:r for r in load_pairs(root/(stage+'.jsonl'))}
           for stage in ('cal','test')}
    grouped=defaultdict(dict)
    for r in load_pairs(predictions):
        if r['stage'] not in truth or r['pair_id'] not in truth[r['stage']]:continue
        key=r['model_key'],r['condition'],r['stage']
        if r['pair_id'] in grouped[key]:raise ValueError(f'Duplicate prediction: {key}, {r["pair_id"]}')
        grouped[key][r['pair_id']]=r
    test=truth['test']; w=train_logit(train)
    report={'split_sizes':{'train':len(train),'cal':len(truth['cal']),'test':len(test)},
            'systems':{
                'constant_half':{'test':score([dict(r,p_A=.5) for r in test.values()])},
                'logistic_features':{'test':score([dict(r,p_A=sigmoid(sum(a*b for a,b in zip(w,feature(r))))) for r in test.values()])}},
            'caveats':['Only compare conditions with the same complete test coverage.',
                       'Calibration trained exclusively on cal split.',
                       'Public historical data may have appeared in pretraining.',
                       'Winner probability is not CTR or individual consumer preference.']}
    for (model,condition,stage),observed in sorted(grouped.items()):
        if stage!='test':continue
        name=model+'/'+condition
        rows=[{'winner':test[i]['winner'],'p_A':r['p_A']} for i,r in observed.items()]
        item={'raw_test':score(rows),'test_coverage':f'{len(rows)}/{len(test)}',
              'test_cost_usd':sum(float(r['reported_cost_usd']) for r in observed.values()
                                  if r.get('reported_cost_usd') is not None)}
        cal=grouped.get((model,condition,'cal'),{})
        if len(cal)>=5:
            alpha=shrinkage([float(r['p_A']) for r in cal.values()],
                            [int(truth['cal'][i]['winner']=='A') for i in cal])
            item['calibration_alpha']=alpha
            item['cal_coverage']=f'{len(cal)}/{len(truth["cal"])}'
            item['calibrated_test']=score([dict(r,p_A=.5+alpha*(float(r['p_A'])-.5)) for r in rows])
        report['systems'][name]=item
    p=Path(output);p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(json.dumps(report,indent=2),encoding='utf-8')
    return report
