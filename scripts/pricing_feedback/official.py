"""New seeds on unmodified EconEvals Basic pricing instances and official last-50 metric."""
from concurrent.futures import ProcessPoolExecutor
import numpy as np
from study import OUT,RAW,read,save,decide
from analyze import cluster
from pricing import instance,market


def one(seed):
    args=instance(seed,1)
    pars=dict(a0=args.a0,a=args.a_tuple,c=tuple(args.costs.values()),mu=args.mu,sigma=args.sigma,group_idxs=args.group_idxs)
    schedule=list(zip(args.alpha_list,args.multiplier_list))
    opt=market.get_monopoly_prices_varying_alphas(**pars,alpha_list=args.alpha_list,multiplier_list=args.multiplier_list)
    direct=[market.get_monopoly_prices(**pars,alpha=a,multiplier=m)[0] for a,m in schedule]
    factor=np.random.RandomState(seed).uniform(1.5,2.5)
    scores={}
    trajectories={}
    for arm in ('fixed','feedback','last10'):
        hist=[];prev=None;den=0;price=0.
        for t,(alpha,multiplier) in enumerate(schedule):
            kw=dict(pars,alpha=alpha,multiplier=multiplier)
            hint=round(max(direct[t//10*10:t//10*10+10])*factor,2)
            if t==0:ref=hint*.48
            if arm=='last10':
                price=ref if t==0 else max(.01,min(hint,max(hist[-12:],key=lambda x:x[3])[0]+hint*(.06 if t%2 else -.06)))
            else:
                # The upstream hint is advisory. Commit the requested price for its full term.
                prev=decide(t+1,100-t,ref,.01,max(hint,ref,prev['price'] if prev else ref),hist,prev,mode=arm)
                price=prev['price']
            pi=round(market.get_profits(p=[price],**kw)[0],2)
            q=round(market.get_quantities(p=[price],**{k:v for k,v in kw.items() if k!='c'})[0],2)
            if t>=50:den+=market.get_profits(p=opt[t],**kw)[0]
            hist.append([price,ref,q,pi])
        scores[arm]=sum(v[3] for v in hist[50:])/den
        trajectories[arm]=hist
    save(RAW/f'official-{seed}.json.gz',trajectories)
    return dict(seed=seed,change=args.env_type.value,scores=scores)


if __name__=='__main__':
    with ProcessPoolExecutor(max_workers=4) as ex:rows=list(ex.map(one,range(56100,56124)))
    groups={}
    for name in sorted({r['change'] for r in rows}):
        g=[r['scores'] for r in rows if r['change']==name]
        groups[name]=dict(means={a:np.mean([v[a] for v in g]) for a in g[0]},
                          feedback_minus_fixed=cluster([v['feedback']-v['fixed'] for v in g]),
                          feedback_minus_previous=cluster([v['feedback']-v['last10'] for v in g]))
    save(OUT/'official.json',dict(episodes=3*len(rows),units='official last-50 OPT ratio',results=groups,seeds=rows))

