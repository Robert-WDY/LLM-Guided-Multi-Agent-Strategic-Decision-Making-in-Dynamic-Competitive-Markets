"""Exploratory closed-loop LLM-draft comparison, with matched observations cached."""
import hashlib
import sys
from common import OUT,RAW,read,save
from duopoly import ALPHAS,Opponent,environment,settle,startup,simple,advise
from provider import decision


def run(provider):
    cal=read(OUT/'freeze.json')['fit'];rows=[]
    for seed in (3300,3301):
        for family in ('matching','switching'):
            kw,c,_,_,hi=environment(ALPHAS[seed%3])
            for arm in ('A','B','C','D','Dopt'):
                op=Opponent(seed,family,hi)
                h=[];decisions=[];last=peer=.55*hi
                for t in range(50):
                    rival=op.price(t,last,peer)
                    p=startup(t,hi,seed) if t<10 else last
                    if t and t%10==0:
                        if arm=='B':
                            p=simple(h,hi)
                            decisions.append(dict(t=t,executed=p))
                        else:
                            visible=[[round(v,4) for v in r] for r in h[-15:]]
                            msg=(f"Set your price for the next 10 periods in a synthetic duopoly. Unit cost 1. "
                                 f"Maximize discounted profit; valid range 0.01 to {hi}. Recent "
                                 f"[your price,peer price,your quantity,your profit]: {visible}. Reply only with the price as a number.")
                            key=hashlib.sha256(msg.encode()).hexdigest()[:20]
                            draft=decision(provider,'pilot:draft:'+key,msg,.01,hi)
                            rec=draft
                            if arm!='A':
                                mode='generic' if arm=='C' else 'conditional'
                                rec=advise(h,hi,draft,mode,seed*1000+t,cal[arm[0]]['budget'])['price']
                            p=rec
                            if arm=='Dopt' and abs(rec-draft)>1e-7:
                                p=decision(provider,'pilot:adopt:'+key,msg+f" Your draft was {draft}; the suggested price is {rec:.6f}. Make your final choice.",.01,hi)
                            decisions.append(dict(t=t,draft=draft,rec=rec,executed=p))
                    q,pi=settle(kw,c,(p,rival));h.append((p,rival,q[0],pi[0]));last,peer=p,rival
                rows.append(dict(seed=seed,family=family,arm=arm,profit=sum(.99**t*r[3] for t,r in enumerate(h)),history=h,decisions=decisions))
                save(RAW/f'pilot-{provider}.json.gz',rows)
    save(OUT/f'pilot-{provider}.json',[{k:v for k,v in r.items() if k!='history'} for r in rows])


if __name__=='__main__':run(sys.argv[1])

