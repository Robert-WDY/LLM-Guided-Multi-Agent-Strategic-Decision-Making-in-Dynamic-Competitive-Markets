"""Original instance generation, settlement, public price hints and last-50 scoring."""
import sys
import numpy as np
from common import RAW, OUT, market, save, read
from econ_evals.experiments.pricing.generate_instance import generate_instance


def instance(seed, n):
    rng = np.random.RandomState(seed)
    alphas = rng.uniform(1, 10, n).tolist()
    ids = [f"Product_{i+1}" for i in range(n)]
    costs = dict(zip(ids, rng.uniform(1, 2, n)))
    tastes = tuple(rng.uniform(2, 3, n))
    length = rng.randint(10, 20)
    return generate_instance(100, "v1", seed, "external", ["linear_shifts","periodic_shifts"][seed%2],
        n, 0, .5, 1, 100, .2, .25, ids, costs, tastes, alphas, length, rng)


def run(seed, n, policy):
    args = instance(seed,n)
    pars = dict(a0=args.a0,a=args.a_tuple,c=tuple(args.costs.values()),mu=args.mu,sigma=args.sigma,group_idxs=args.group_idxs)
    opt = market.get_monopoly_prices_varying_alphas(**pars,alpha_list=args.alpha_list,multiplier_list=args.multiplier_list)
    factor = np.random.RandomState(seed).uniform(1.5,2.5)
    # Original hint uses a fresh solver each period, distinct from scoring OPT.
    direct = [market.get_monopoly_prices(**pars,alpha=a,multiplier=m) for a,m in zip(args.alpha_list,args.multiplier_list)]
    hist = []
    for t,(a,m) in enumerate(zip(args.alpha_list,args.multiplier_list)):
        hi = round(max(map(max,direct[t//10*10:t//10*10+10]))*factor,2)
        if policy in ("doubao","deepseek"):
            from provider import decision
            assert n==1
            prompt = (f"Set a price to maximize your own profit, by exploration and learning. Attempt {t}, "
                      f"unit cost {pars['c'][0]:.5f}; suggested range 0 to {hi}. History of recent settled "
                      "attempts [price,quantity,profit]: "+str([(r["p"][0],r["q"][0],r["profit"][0]) for r in hist[-20:]])+
                      ". Reply only with your numeric price.")
            p=[decision(policy,f"basic:{seed}:{t}",prompt,0,hi)]
        else:
            # Simple local feedback policy. Regular exploration of individual product prices.
            p=[hi*.48]*n if not hist else list(hist[-1]["p"])
            j=t%n
            if t:
                best=max(hist[-12:],key=lambda x:sum(x["profit"]))
                p=list(best["p"])
                p[j]=min(hi,max(.01,p[j]+hi*(.06 if (t//n)%2 else -.06)))
        kw=dict(pars,alpha=a,multiplier=m)
        profits=market.get_profits(p=p,**kw)
        q=market.get_quantities(p=p,**{k:v for k,v in kw.items() if k!="c"})
        optimal=sum(market.get_profits(p=opt[t],**kw))
        hist.append(dict(t=t,p=[round(x,2) for x in p],q=[round(x,2) for x in q],
                         profit=[round(x,2) for x in profits],opt=optimal,hint=hi))
    ratio=sum(sum(r["profit"]) for r in hist[50:])/sum(r["opt"] for r in hist[50:])
    result=dict(seed=seed,products=n,policy=policy,ratio=ratio,periods=len(hist),env=args.env_type.value)
    save(RAW/f"pricing-{policy}-{n}-{seed}.json.gz",hist)
    save(OUT/f"pricing-{policy}-{n}-{seed}.json",result)
    print(result,flush=True)


if __name__=="__main__":
    if sys.argv[1]=="offline":
        for n in [1,4,10]:
            for seed in [0,1]:run(seed,n,"feedback")
    else:
        run(int(sys.argv[2]),1,sys.argv[1])

