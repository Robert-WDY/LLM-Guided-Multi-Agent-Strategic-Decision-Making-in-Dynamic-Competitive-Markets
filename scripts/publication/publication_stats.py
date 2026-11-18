"""Recompute confirmatory contrasts from published prospective actions and settlement payoffs."""
from collections import defaultdict
import gzip
import json
from pathlib import Path

import numpy as np

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/"experiment-results/publication-v1"
ARMS=("keep","down","up")


def read(path):
    data=path.read_bytes()
    return json.loads(gzip.decompress(data) if path.suffix==".gz" else data)


def estimate(x):
    x=np.asarray(x,dtype=float)
    indices=np.random.default_rng(42621).integers(len(x),size=(12000,len(x)))
    boot=x[indices].mean(axis=1)
    return dict(mean=float(x.mean()),ci95=np.quantile(boot,[.025,.975]).tolist(),
                ci9875=np.quantile(boot,[.00625,.99375]).tolist(),seeds=len(x))


def weighted(y,actions,q):
    # P(kind)=1/2; P(signal=kind | kind)=q. Enumerate both signals exactly.
    return .5*sum((q if k==s else 1-q)*y[k,actions[s]] for k in range(2) for s in range(2))


def optimal(y,q):
    return float((q*y+(1-q)*y[::-1]).max(axis=1).mean())


def paired(market,discount):
    rows=read(OUT/f"test-{market}.json.gz")
    assert [r["seed"] for r in rows]==list(range(103000,103192))
    stats=defaultdict(list)
    counts=defaultdict(int)
    for r in rows:
        p=np.array(r["profit"],dtype=float)
        y=(p-p[:,:1])@(discount**np.arange(p.shape[-1]))/(100 if market=="native" else 1)
        assert y.shape==(2,3) and np.all(y[:,0]==0)
        negative=np.repeat(y[:1,:],2,axis=0)
        assert optimal(negative,1)==optimal(negative,.5)
        winning=np.isclose(y,y.max(axis=1)[:,None],rtol=0,atol=1.e-8)
        counts["incompatible_winners"]+=int(not np.any(winning[0]&winning[1]))
        counts["opposite_sign"]+=int(np.any((y[0,1:]*y[1,1:]) < -1.e-8))
        counts["some_gain_no_robust_gain"]+=int(y.max()>1e-8 and y.min(axis=0).max()<=1.e-8)
        decision=r["prospective"]
        oracle={}
        actual={}
        for q in (.5,.75,1.):
            key=str(q)
            oracle[q]=optimal(y,q)
            d=decision["signal"][key]
            a=d["act"];g=d["gated"]
            actual[q]=weighted(y,a,q)
            stats["oracle_"+key].append(oracle[q])
            stats["policy_"+key].append(actual[q])
            stats["gated_"+key].append(weighted(y,g,q))
            # Ranking chooses between active candidates. Include the rejected direction even on abstentions.
            scores=np.array(decision["scores"])
            offer=1+(q*scores[:,1:]+(1-q)*scores[::-1,1:]).argmax(axis=1)
            counts["active_"+key]+=int(any(g))
            coverage=weighted(np.tile([0.,1.,1.],(2,1)),g,q)
            stats["coverage_"+key].append(coverage)
            loss=(y < -1.e-8).astype(float)
            stats["loss_mass_"+key].append(weighted(loss,g,q))
            stats["offered_"+key].append(weighted(y,offer,q))
            stats["missed_win_mass_"+key].append(.5*sum((q if k==s else 1-q)*(g[s]==0 and y[k,offer[s]]>1e-8)
                                                        for k in range(2) for s in range(2)))
        assert oracle[1.]+1e-6>=oracle[.75]>=oracle[.5]-1e-6
        stats["oracle_gap"].append(oracle[1.]-oracle[.5])
        stats["policy_gap"].append(actual[1.]-actual[.5])
        stats["blind_estimation_gap"].append(oracle[.5]-actual[.5])
        stats["signal_estimation_gap"].append(oracle[1.]-actual[1.])
        for idx,arm in enumerate(ARMS[1:],1):
            stats["fixed_"+arm].append(y[:,idx].mean())
        for mode,d in decision["prior"].items():
            arm=ARMS.index(d["arm"]) if d["use"] else 0
            stats["legacy_"+mode].append(y[:,arm].mean())
    out={k:estimate(v) for k,v in stats.items()}
    out["counts"]=dict(counts)
    for key in ("0.5","0.75","1.0"):
        cov=np.array(stats["coverage_"+key])
        losses=np.array(stats["loss_mass_"+key])
        out["loss_given_accepted_"+key]=float(losses.sum()/cov.sum()) if cov.sum() else None
        # Expected gain from assigning the same overall coverage independently of the offered gain.
        z=np.array(stats["gated_"+key])
        offered=np.array(stats["offered_"+key])
        idx=np.random.default_rng(42621).integers(len(z),size=(12000,len(z)))
        boot=z[idx].mean(1)-cov[idx].mean(1)*offered[idx].mean(1)
        out["gate_minus_random_"+key]=dict(mean=float(z.mean()-cov.mean()*offered.mean()),
                                            ci95=np.quantile(boot,[.025,.975]).tolist())
    return out


def real_sample(market,archive):
    rows=read(OUT/("archive-econ.json.gz" if archive else "fresh-native.json.gz"))
    by_seed=defaultdict(list)
    n_accept=n_loss=missed=diff=0
    nf=0
    for r in rows:
        if archive:
            d=r["decisions"];p=np.asarray(r["profit"])
            nf+=int(bool(r["fallback"]))
        else:
            d=r["decision"];p=np.array([r["paths"][arm]["settled"] for arm in ARMS])[:,:,3]
            nf+=int(bool(r["prefix_fallback"] or any(h["fallback"] for h in r["paths"].values())))
        y=(p-p[:1])@(.99**np.arange(p.shape[1]))/(100 if market=="native" else 1)
        c,g=d["conditional"],d["generic"]
        v=y[ARMS.index(c["arm"])]
        n_accept+=int(c["use"]);n_loss+=int(c["use"] and v<0)
        missed+=int(not c["use"] and v>0)
        diff+=int(c["arm"]!=g["arm"])
        actual=v*c["use"]
        by_seed[r["seed"]].append([actual,actual-y[ARMS.index(g["arm"])]*g["use"],actual-y[1],actual-y[2],y.max()])
    averaged=np.array([np.mean(v,axis=0) for v in by_seed.values()])
    results={k:estimate(v) for k,v in zip(("keep","generic","fixed_down","fixed_up","oracle"),averaged.T)}
    results.update(n=len(rows),n_accepted=n_accept,n_lost=n_loss,n_missed=missed,n_differing_directions=diff,
                   n_cases_with_fallback=nf,paired_sd=float(averaged[:,0].std(ddof=1)))
    return results


def main():
    result={m:{str(d):paired(m,d) for d in (.99,1.)} for m in ("econ","native")}
    result["archive"]=real_sample("econ",True)
    result["fresh"]=real_sample("native",False)
    (OUT/"results.json").write_bytes(json.dumps(result,indent=2,allow_nan=False).encode())


if __name__=="__main__":
    main()
