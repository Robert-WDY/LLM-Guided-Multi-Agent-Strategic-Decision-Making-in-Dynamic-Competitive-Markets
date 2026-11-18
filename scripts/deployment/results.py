"""Analyze independent market seeds, separately by model; no inference service is required."""
import argparse
import gzip
import json
from pathlib import Path

import numpy as np
from scipy.stats import beta

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/"experiment-results/independent-deployment-v1"
DATES=np.arange(19,48,4)
THRESH={"econ":5.,"native":20000.}


def read(path):
    data=path.read_bytes()
    return json.loads(gzip.decompress(data) if path.suffix==".gz" else data)


def save(path,obj):
    path.write_bytes(json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False).encode())


def bootstrap(values,tail=False):
    x=np.array(values,float)
    rng=np.random.default_rng(180619)
    means=[];tails=[]
    k=max(1,int(np.ceil(.05*len(x))))
    for _ in range(128):
        v=rng.choice(x,(64,len(x)),replace=True)
        means.extend(v.mean(axis=1))
        if tail:
            tails.extend(np.partition(v,k-1,axis=1)[:,:k].mean(axis=1))
    q=np.quantile(means,[.0125,.025,.975,.9875])
    result=dict(mean=float(x.mean()),ci95=q[1:3].tolist(),bounds=q[[0,3]].tolist(),n=len(x))
    if tail:
        result["worst5"]=dict(mean=float(np.partition(x,k-1)[:k].mean()),ci95=np.quantile(tails,[.025,.975]).tolist(),k=k)
    return result


def summarize(rows,m,discount):
    n=len(rows)
    y=[]
    for r in rows:
        p=np.array(r["profit"])
        assert p.shape==(3,61-r["public"]["round"])
        y.append((p-p[:1])@(discount**np.arange(p.shape[1]))/(100 if m=="native" else 1))
    y=np.array(y);ix=np.arange(n)
    picks={k:np.array([r["decision"][k] for r in rows]) for k in ("inferred","uniform","direct","offer")}
    policy=y[ix,picks["inferred"]]
    use=picks["inferred"]!=0
    accepted=int(use.sum())
    losses=int((policy< -1.e-8).sum())
    upper=float(beta.ppf(.9875,losses+1,accepted-losses)) if losses<accepted else (1. if accepted else None)
    res=bootstrap(policy,True)
    return dict(keep=res,vs_uniform=bootstrap(policy-y[ix,picks["uniform"]]),vs_direct=bootstrap(policy-y[ix,picks["direct"]]),
                vs_up=bootstrap(policy-y[:,2]),vs_down=bootstrap(policy-y[:,1]),active=sum(r["active"] for r in rows),
                accepted=accepted,lost=losses,missed=int(((~use)&(y[ix,picks["offer"]]>1e-8)).sum()),
                risk_upper=upper,threshold=THRESH[m],precision_half_width=(res["ci95"][1]-res["ci95"][0])/2,
                deployment_pass=bool(res["bounds"][0]>THRESH[m] and accepted>=.1*n and upper is not None and upper<=.1),
                target_excluded=bool(res["bounds"][1]<THRESH[m]),identical_prior=int((picks["inferred"]==picks["uniform"]).sum()))


def independent(tag):
    results={}
    for m in ("econ","native"):
        rows=read(OUT/f"{tag}-{m}.json.gz")
        for name in ("qwen","smol"):
            group=[r for r in rows if r["model"]==name]
            seeds=[r["seed"] for r in group]
            assert len(seeds)==len(set(seeds))
            assert len(group)==(4608 if tag=="independent" and m=="native" else 192)
            item={str(d):summarize(group,m,d) for d in (.99,1.)}
            if tag=="numeric":
                events=[e for r in group for e in r["prefix_calls"]]
                follow=[e for r in group for branch in r["branch_calls"] for e in branch]
                def quality(es):
                    return dict(uses=len(es),unique_requests=len({e[1] for e in es}),raw_outside=int(sum(not e[5]<=e[2]<=e[6] for e in es)),
                                differs_from_reference=int(sum(abs(e[3]-e[4])>1e-8 for e in es)))
                item["prefix_quality"]=quality(events);item["continuation_quality"]=quality(follow)
            else:
                from collections import Counter
                model=read(OUT/f"llm-{name}.json.gz")["entries"]
                keys=[key for r in group for key in r["prefix_keys"]+sum(r["branch_keys"],[])]
                item["menu"]=dict(uses=len(keys),distinct_prompts=len(set(keys)),
                                  choices=dict(Counter(model[key]["letter"] for key in keys)))
            results[m+"-"+name]=item
    return results


def blind():
    out={}
    for m in THRESH:
        rows=read(OUT/f"blind-test-{m}.json.gz")
        assert [r["seed"] for r in rows]==list(range(131000,131512))
        units=100 if m=="native" else 1
        vals=[];waits=[];chosen=[];adjust=[];types=[];onsets=[];early_exit=0;favorable_missed=0
        for r in rows:
            d=r["decisions"]
            tindex=next(j for j,v in enumerate(d) if v["action"]!=3)
            a=d[tindex]["action"]
            y=np.zeros((len(DATES),3))
            for j,t in enumerate(DATES):
                actual=np.asarray(r["paths"][j])-np.asarray(r["base"][t-1:])
                y[j,1:]=actual@(.99**np.arange(t-1,60))/units
            v=y[tindex,a]
            immediate=y[0,d[0]["immediate"]]
            fixed=y[1,d[1]["immediate"]]
            # Forced delay costs are path-dependent opportunity costs, not a per-day fee.
            vals.append([v,immediate,fixed,v-immediate,v-fixed,fixed-immediate,y.max(),y.max()-v])
            waits.append(tindex*4);chosen.append(a);adjust.append(DATES[tindex]);types.append(r["condition"]);onsets.append(r["onset"])
            early_exit+=int(a==0 and tindex<len(DATES)-1)
            favorable_missed+=int(a==0 and y.max()>1e-8)
        v=np.asarray(vals);action=np.asarray(chosen)!=0;at=np.asarray(adjust)
        pre=sum(bool(a and o is not None and t<o) for a,o,t in zip(action,onsets,at))
        delays=[int(t-o) for a,o,t in zip(action,onsets,at) if a and o is not None and t>=o]
        st=np.asarray(types)=="stable"
        out[m]=dict(effects={k:bootstrap(x,k=="dynamic") for k,x in zip(
            ("dynamic","immediate","fixed_wait","dynamic_minus_immediate","dynamic_minus_fixed_wait","fixed_wait_minus_immediate",
             "oracle","oracle_minus_dynamic"),v.T)},intervened=int(action.sum()),immediate_interventions=int((action & (at==19)).sum()),
             wait_first=int((np.asarray(waits)>0).sum()),early_permanent_abstention=early_exit,
             never_intervened=int((~action).sum()),missed_profitable_state=favorable_missed,
             waiting_rounds_mean=float(np.mean(waits)),
             waited_foregone_immediate=float(np.maximum(-v[np.array(waits)>0,3],0).sum()/len(rows)),
             waited_gain_relative_immediate=float(np.maximum(v[np.array(waits)>0,3],0).sum()/len(rows)),
             changed_pre_event=pre,changed_after_event=len(delays),median_delay_after=float(np.median(delays)) if delays else None,
             no_change_intervened=int((st&action).sum()),no_change_total=int(st.sum()),
             by_kind={kind:bootstrap(v[np.asarray(types)==kind,3]) for kind in sorted(set(types))})
    return out


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("stage",choices=("independent","numeric","blind"))
    arg=parser.parse_args().stage
    save(OUT/f"{arg}-results.json",blind() if arg=="blind" else independent(arg))

