"""Transfer of the project's draft-relative search, public response and paired validation."""
from functools import lru_cache
import math
import numpy as np
from common import market
from game_theory_agent.game_theory import reliable_response as response
from game_theory_agent.game_theory.reliable_advisor import lower_bound

FAMILIES = ("stationary", "matching", "undercut", "switching")
ALPHAS = (1., 3.2, 10.)
DISCOUNT = .99
FRACTIONS = (1., .95, 1.05)


@lru_cache
def environment(alpha, companies=2):
    kw = dict(a0=0., a=(2.,)*companies, mu=.25, alpha=(alpha,)*companies,
              multiplier=100., sigma=0., group_idxs=(1,)*companies)
    cost=(1.,)*companies
    monopoly=market.get_monopoly_prices(**kw,c=cost)[0]
    nash=market.get_nash_prices(**kw,c=cost)[0]
    return kw, cost, float(nash),float(monopoly),round(monopoly*1.5,2)


def settle(kw,cost,p):
    q=market.get_quantities(p=p,**kw)
    profits=market.get_profits(p=p,c=cost,**kw)
    return [round(x,2) for x in q],[round(x,2) for x in profits]


class Opponent:
    """Evaluator-owned. Same tape across branches; no simultaneous focal action."""
    def __init__(self,seed,family,hi):
        rng=np.random.default_rng(seed)
        self.base=hi*rng.uniform(.45,.69)
        self.shocks=rng.normal(0,hi*.012,305)
        self.offset=rng.uniform(.014,.045)*hi
        self.family=family
        self.hi=hi

    def price(self,t,last,peer):
        family=self.family
        if family=="switching":
            family=("matching","undercut","stationary")[(t//17)%3]
        target=self.base
        if family=="matching":target=.74*last+.26*self.base
        if family=="undercut":target=.88*last+.12*self.base-self.offset
        return float(np.clip(.2*peer+.8*target+self.shocks[t],.26*self.hi,.9*self.hi))


def startup(t,hi,seed):
    seq=(.40,.62,.52,.72,.47,.66,.58,.45,.67,.56)
    return hi*seq[(t+seed)%len(seq)]


def simple(hist,hi):
    best=max(hist[-10:],key=lambda row:row[3])[0]
    step=.025*hi*(1 if (len(hist)//10)%2 else -1)
    return float(np.clip(best+step,.12*hi,.96*hi))


def fitted(history,hi):
    data=np.array(history[-50:])
    p,peer,q,profit=data.T
    # Learn both demand and the unit margin in upstream currency using settled data.
    x=np.stack((np.ones(len(p)),p/hi,peer/hi),axis=1)
    penalty=np.diag([.00001,.018,.018])
    beta=np.linalg.solve(x.T@x+penalty,x.T@np.log(np.maximum(q,.01)))
    valid=q>.05
    if valid.sum()<2:
        margins=np.array([0.,0.])
    else:
        z=x[valid,:2]
        margins=np.linalg.solve(z.T@z+np.diag([.00001,.0001]),z.T@(profit[valid]/q[valid]))
    samples=[dict(features=(response.bucket(p[i-1]/p[i-2]-1),0,0),change=peer[i]/peer[i-1]-1)
             for i in range(2,len(p))]
    return beta,margins,response.fit(samples)


def advise(history,hi,draft,mode,seed,budgets=None):
    """Same history, fitter and scenario tape in generic and conditional modes."""
    beta,margins,model=fitted(history,hi)
    cand=np.clip(draft*np.array(FRACTIONS),.01*hi,hi)
    rng=np.random.default_rng(seed)
    # 3 search and 8 independent validation scenarios; evaluate all alike.
    uniforms=rng.random((5,11))
    peer=np.full((3,11),history[-1][1])
    last=np.full((3,1),history[-1][0])
    movement=np.full((3,1),response.bucket(history[-1][0]/history[-2][0]-1))
    predictions=[response.predict(model,(b,0,0),mode) for b in (-1,0,1)]
    probs=np.array([r["probabilities"] for r in predictions])
    changes=np.array([r["changes"] for r in predictions])
    rewards=[]
    for t in range(5):
        idx=movement+1
        u=uniforms[t][None,:]
        label=(u>=probs[idx,0]).astype(int)+(u>=probs[idx,0]+probs[idx,1]).astype(int)
        peer=np.clip(peer*(1+changes[idx,label]),hi*.01,hi)
        own=cand[:,None] if t==0 else np.full((3,1),draft)
        x=own/hi
        qty=np.exp(np.clip(beta[0]+beta[1]*x+beta[2]*peer/hi,-6,6))
        rewards.append(.95**t*qty*(margins[0]+margins[1]*x))
        delta=own/last-1
        movement=np.where(delta<-.01,-1,np.where(delta>.01,1,0))
        last=own
    accumulated=np.cumsum(rewards,axis=0)
    pick=1+int(np.argmax(accumulated[2,1:,:3].mean(axis=1)))
    gains=accumulated[[2,4],pick,3:]-accumulated[[2,4],0,3:]
    bounds=[lower_bound(list(g)) for g in gains]
    basic_pass=bool(np.all(gains>=0) and min(bounds)>0)
    proposed=float(cand[pick])
    passed=basic_pass and (budgets is None or all(x>y for x,y in zip(bounds,budgets)))
    return dict(price=proposed if passed else draft,proposed=proposed,raw_best=proposed if accumulated[2,pick,:3].mean()>accumulated[2,0,:3].mean() else draft,
                bounds=bounds,base_pass=basic_pass,steps=3*11*5)


def branch_value(t,history,first,draft,peer_policy,kw,cost):
    last,peer=history[-1][:2]
    profit=0.
    values=[]
    for j in range(5):
        rival=peer_policy.price(t+j,last,peer)
        own=first if j==0 else draft
        profit+=.95**j*market.get_profits(p=(own,rival),c=cost,**kw)[0]
        if j in (2,4):values.append(profit)
        last,peer=own,rival
    return values


def episode(seed,family,arm,budgets=None,calibration=False):
    alpha=ALPHAS[seed%3]
    kw,cost,nash,mono,hi=environment(alpha)
    op=Opponent(seed,family,hi)
    history=[]; decisions=[]; errors=[];surplus=0.;total=0.
    last=peer=hi*.55
    for t in range(300):
        own=startup(t,hi,seed) if t<10 else last
        rival=op.price(t,last,peer)
        if t and t%10==0:
            draft=own; steps=0
            if arm=="B":own=simple(history,hi)
            if arm[0] in ("C","D"):
                a=advise(history,hi,draft,"generic" if arm[0]=="C" else "conditional",seed*1000+t,budgets)
                steps=a["steps"]
                own=a["raw_best"] if arm.endswith("raw") else a["price"]
                if calibration and a["base_pass"]:
                    y=branch_value(t,history,a["proposed"],draft,op,kw,cost)
                    z=branch_value(t,history,draft,draft,op,kw,cost)
                    errors.append([b-(i-j) for b,i,j in zip(a["bounds"],y,z)])
            decisions.append((t,draft,own,steps))
        q,pi=settle(kw,cost,(own,rival))
        total+=DISCOUNT**t*pi[0]
        # Log-sum inclusive value in base (price / alpha) units, 100 consumers.
        surplus+=100*.25*math.log(1+math.exp((2-own/alpha)/.25)+math.exp((2-rival/alpha)/.25))
        history.append((own,rival,q[0],pi[0]))
        last,peer=own,rival
    index=(np.mean([.5*(x[0]+x[1]) for x in history[-50:]])-nash)/(mono-nash)
    return dict(seed=seed,family=family,arm=arm,profit=total,cs=surplus/300,
                index=float(np.clip(index,0,1)),interventions=sum(abs(x[1]-x[2])>1e-9 for x in decisions),
                opportunities=len(decisions),steps=sum(x[3] for x in decisions),errors=errors,
                decisions=decisions,history=history)

