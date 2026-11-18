"""Independent weak and simultaneous changes. The classifier never receives the intervention."""
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace
import numpy as np
from collect import world,limits,settle,decision,draw,OUT,RAW,save,read
from models import bundle,diagnosis,flags
from game_theory_agent.market.protocols import state_hash


def one(task):
    m,s=task;result=[]
    for case in ('weak_demand','joint','placebo'):
        w=world(m,s,'rival' if case=='joint' else 'stable')
        h=[];old=None
        for t in range(1,61):
            lo,hi=limits(w)
            d=decision(w,h,old,None,s,lo,hi,[])
            if t==w.when and case!='placebo':
                sign=1 if w.up else -1
                magnitude=.15 if case=='weak_demand' else .45
                if m=='econ':w.kw=dict(w.kw,a=(2+sign*magnitude,)*2)
                else:
                    st=w.state
                    st=replace(st,market=replace(st.market,base_demand_orders=round(st.market.base_demand_orders*(1+sign*magnitude))))
                    w.state=replace(st,state_hash=state_hash(st.to_dict()))
                    w.env.load_state(w.state)
            h.append(settle(w,d['price'],lo,hi));old=d
        obj=bundle(m);ctx='full' if m=='native' else 'basic'
        *_,pr=diagnosis(obj,h,ctx)
        j=w.when-7+3
        result.append(dict(seed=s,market=m,case=case,when=w.when,probs=pr[j].tolist(),
                    label=int(np.argmax(pr[j])),alert=int(flags(pr[j:j+1],obj['alert'][ctx])[0])))
    return result


if __name__=='__main__':
    assert (OUT/'freeze.json').exists()
    with ProcessPoolExecutor(max_workers=8) as ex:
        rs=[r for group in ex.map(one,[(m,s) for m in ('econ','native') for s in range(78200,78232)]) for r in group]
    save(OUT/'stress.json',rs)

