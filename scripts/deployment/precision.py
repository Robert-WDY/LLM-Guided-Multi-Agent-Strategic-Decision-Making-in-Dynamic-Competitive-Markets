"""Precision and stratification sensitivity for the locked primary threshold."""
from math import ceil

import numpy as np
from scipy.stats import beta,norm

from results import OUT,THRESH,read,save


def stats(m,rows):
    values=[];kinds=[];clipped=0
    by_decision={"released":[],"withheld":[]}
    executable={"released":[],"withheld":[]}
    for r in rows:
        p=np.array(r["profit"])
        a=r["decision"]["inferred"]
        values.append((p[a]-p[0])@(.99**np.arange(p.shape[1]))/(100 if m=="native" else 1))
        kinds.append(r["condition"])
        group="released" if a else "withheld"
        candidate=r["decision"]["offer"]
        assert not a or a==candidate
        v=(p[candidate]-p[0])@(.99**np.arange(p.shape[1]))/(100 if m=="native" else 1)
        by_decision[group].append(v)
        if "settled" in r and r["active"] and abs(r["settled"][candidate][0][0]-r["settled"][0][0][0])>1e-8:
            executable[group].append(v)
        if "prefix_calls" in r and r["active"]:
            e=r["prefix_calls"][-1]
            assert e[0]==r["public"]["round"]
            clipped+=e[2]!=e[3]
    z=np.array(values);groups=[z[np.asarray(kinds)==k] for k in sorted(set(kinds))]
    assert len(set(map(len,groups)))==1
    rng=np.random.default_rng(830811);boots=[]
    for _ in range(96):
        b=sum(rng.choice(g,(64,len(g))).mean(1) for g in groups)/len(groups)
        boots.extend(b)
    sd0=19.29 if m=="econ" else 432134.
    critical=norm.ppf(.9875)+norm.ppf(.80)
    def counts(v):
        x=np.asarray(v)
        return dict(n=len(x),positive=int((x>1e-8).sum()),negative=int((x< -1e-8).sum()),
                    zero=int((np.abs(x)<=1e-8).sum()),mean=float(x.mean()) if len(x) else None,
                    harm_lower=(float(beta.ppf(.0125,(x< -1e-8).sum(),len(x)-(x< -1e-8).sum()+1))
                                if (x< -1e-8).any() else 0.) if len(x) else None)
    return dict(offer_outcomes={k:counts(v) for k,v in by_decision.items()},
                feasible_price_changes={k:counts(v) for k,v in executable.items()} if "settled" in rows[0] else None,
                locked_sd=sd0,observed_sd=float(z.std(ddof=1)),
                planned_n_required=int(ceil((critical*sd0/THRESH[m])**2)),
                stratified_bounds=np.quantile(boots,[.0125,.9875]).tolist(),clipped_at_opportunity=clipped,
                by_condition={k:dict(mean=float(g.mean()),n=len(g)) for k,g in zip(sorted(set(kinds)),groups)})


if __name__=="__main__":
    result={}
    for study in ("independent","numeric"):
        for market in THRESH:
            r=read(OUT/f"{study}-{market}.json.gz")
            for model in ("qwen","smol"):
                result["-".join((study,market,model))]=stats(market,[v for v in r if v["model"]==model])
    save(OUT/"precision.json",result)

