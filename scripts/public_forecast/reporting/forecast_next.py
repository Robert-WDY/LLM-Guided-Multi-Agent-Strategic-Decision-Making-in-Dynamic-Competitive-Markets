"""Forecast one settlement from a JSON stdin or file containing the firm's observations."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from collect import OUT,ROOT,read
from models import bundle,diagnosis,forecasts,flags


def next_round(observation):
    m=observation['market']
    if m not in ('econ','native'):
        raise ValueError('market must be econ or native')
    source=OUT/f'{m}-forecast.joblib'
    sealed=read(OUT/'freeze.json')['files'][source.relative_to(ROOT).as_posix()]
    if hashlib.sha256(source.read_bytes()).hexdigest()!=sealed:raise ValueError('forecast artifact mismatch')
    model=bundle(m);context='full' if m=='native' else 'basic'
    h=observation['settlements']
    if len(h)<6:raise ValueError('six settled rounds required')
    a=np.asarray(h,dtype=float)
    if a.ndim!=2 or a.shape[1]!=9 or not np.isfinite(a).all() or np.any(a[:,:2]<=0):
        raise ValueError('invalid settled observations')
    plan=[(observation['request'],observation['floor'],observation['ceiling'])]
    if not np.isfinite(plan).all() or plan[0][1]>plan[0][2]:raise ValueError('invalid price terms')
    cause=None
    if len(h)>6:
        pred,_,base,pr=diagnosis(model,h,context,plan)
        cause=dict(probabilities=dict(zip(('stable','demand','rival'),pr[-1].tolist())),
                   alert=('no_alert','demand','rival')[flags(pr[-1:],model['alert'][context])[0]])
    else:
        pred,_,base=forecasts(model['fc'],h,context,plan)
    return dict(round=len(h)+1,sales=pred[-1,0],rival_quote=pred[-1,1],
                carry_sales=base[-1,0],carry_quote=base[-1,1],last_settled_diagnosis=cause)


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('observations',nargs='?',type=Path)
    args=parser.parse_args()
    data=json.loads(args.observations.read_text('utf8')) if args.observations else json.load(sys.stdin)
    print(json.dumps(next_round(data),ensure_ascii=False,allow_nan=False))

