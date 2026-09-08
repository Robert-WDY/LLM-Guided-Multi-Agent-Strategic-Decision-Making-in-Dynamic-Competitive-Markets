"""Public, one-step-lag response model. Never observes simultaneous candidate intents."""
import hashlib
import json
from pathlib import Path

MODEL_PATH=Path(__file__).resolve().parents[3]/'configs/reliable-response-v1.json'

def bucket(x):return -1 if x < -.01 else 1 if x > .01 else 0

def frame(state):
    return dict(round=state.round,prices={c.company_id:c.commercial.price_cents for c in state.companies},
                shortage=state.market.lost_after_stockout_orders/max(1,state.market.realized_demand_orders))

def features(previous,current,peer,service=0):
    others=[a for a in current['prices'] if a!=peer]
    movement=sum(current['prices'][a]/previous['prices'][a]-1 for a in others)/max(1,len(others))
    return (bucket(movement),int(current['shortage']>.1),int(service>0))

def fit(rows):
    groups={};global_counts=[1.,2.,1.];global_sums=[-.05,0.,.05]
    for r in rows:
        label=bucket(r['change'])+1;key=','.join(map(str,r['features']))
        g=groups.setdefault(key,dict(counts=[1.,2.,1.],sums=[-.05,0.,.05],n=0))
        g['counts'][label]+=1;g['sums'][label]+=r['change'];g['n']+=1
        global_counts[label]+=1;global_sums[label]+=r['change']
    model=dict(version='lagged-public-response-v1',groups=groups,global_counts=global_counts,global_sums=global_sums,n=len(rows),
               scope='Histogram conditional on settled competitors price movement, public shortage and own service. Association in randomized synthetic training; not identified real-world causality.')
    model['sha256']=hashlib.sha256(json.dumps(model,sort_keys=True).encode()).hexdigest();return model

def verify(model):
    v=dict(model);digest=v.pop('sha256')
    if hashlib.sha256(json.dumps(v,sort_keys=True).encode()).hexdigest()!=digest:raise ValueError('response model checksum mismatch')
    return model

def load_model():
    if not MODEL_PATH.exists():return fit([])
    return verify(json.loads(MODEL_PATH.read_text(encoding='utf-8')))

def predict(model,x,mode='conditional'):
    g=model['groups'].get(','.join(map(str,x))) if mode=='conditional' else None
    counts=g['counts'] if g else model['global_counts'];sums=g['sums'] if g else model['global_sums']
    return dict(probabilities=[c/sum(counts) for c in counts],changes=[s/c for s,c in zip(sums,counts)],
                observations=g['n'] if g else model['n'],conditional_match=bool(g))

def sample(prediction,seed,scenario,round_number,peer):
    u=int(hashlib.sha256(f'{seed}:{scenario}:{round_number}:{peer}'.encode()).hexdigest()[:12],16)/16**12
    p=prediction['probabilities'];k=0 if u<p[0] else 1 if u<p[0]+p[1] else 2
    return prediction['changes'][k]
