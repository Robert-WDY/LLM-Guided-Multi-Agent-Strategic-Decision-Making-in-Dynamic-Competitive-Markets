"""Paired five-step economic effects of real drafts and independently requested final prices."""
import sys
import numpy as np
from common import RAW, OUT, read, save
from provider import decision, MODELS
from duopoly import ALPHAS,FAMILIES,Opponent,advise,branch_value,environment,simple


def run(provider):
    fit=read(OUT/'freeze.json')['fit']
    rows=[]
    for seed in range(1200,1206):
        baseline=read(RAW/f'holdout-{seed}.json.gz')
        for j,family in enumerate(FAMILIES):
            source=next(x for x in baseline if x['family']==family and x['arm']=='B')
            t=10+10*((seed+j)%3)
            h=source['history'][:t]
            kw,c,n,m,hi=environment(ALPHAS[seed%3])
            op=Opponent(seed,family,hi)
            visible=[[round(v,3) for v in r] for r in h[-18:]]
            prompt=(f"Choose your current selling price in [0.01,{hi}]. Maximize your discounted own profit. "
                    "Simultaneous pricing. Other seller can react only after settlement. "
                    f"Unit cost 1. Public hint upper price {hi}. History [your price,rival price,your quantity,your profit]: {visible}. "
                    "Return only the numeric price.")
            ident=f"window:{seed}:{family}"
            draft=decision(provider,ident+":draft",prompt,.01,hi)
            prices={'A':draft,'B':simple(h,hi)}
            for arm,mode in [('C','generic'),('D','conditional')]:
                a=advise(h,hi,draft,mode,seed*1000+t,fit[arm]['budget'])
                prices[arm]=a['price']
            rec=prices['D']
            final=decision(provider,ident+":final",
                prompt+f" Your draft was {draft}. The advisor's final price is {rec:.8f}. You may keep your draft, accept the suggestion, or set another price.",.01,hi)
            prices['LLM_final']=final
            values={k:branch_value(t,h,p,draft,op,kw,c)[1] for k,p in prices.items()}
            rows.append(dict(seed=seed,family=family,t=t,prices=prices,profit=values))
            save(RAW/f'windows-{provider}.json',rows)
    stats={}
    for l,r in [('B','A'),('C','A'),('D','A'),('D','B'),('D','C'),('LLM_final','A')]:
        cluster=[np.mean([x['profit'][l]-x['profit'][r] for x in rows if x['seed']==s]) for s in range(1200,1206)]
        b=np.random.default_rng(71).choice(cluster,(5000,len(cluster))).mean(axis=1)
        stats[f'{l}-{r}']={'mean':float(np.mean(cluster)),'ci':np.quantile(b,[.025,.975]).tolist()}
    offered=[x for x in rows if abs(x['prices']['D']-x['prices']['A'])>.0001]
    accepted=sum(abs(x['prices']['D']-x['prices']['LLM_final'])<.011 for x in offered)
    save(OUT/f'windows-{provider}.json',dict(model=MODELS[provider],windows=len(rows),seeds=6,offered=len(offered),accepted=accepted,
        final_changed=sum(abs(x['prices']['A']-x['prices']['LLM_final'])>.011 for x in rows),comparisons=stats))
    print(provider,len(rows),'done',flush=True)


if __name__=='__main__':run(sys.argv[1])

