"""Default full-candidate native empirical advisor on independent market populations."""
from concurrent.futures import ProcessPoolExecutor
from collections import Counter
import numpy as np

from native_advisor import terminal
from trials import OUT, RAW, read, save
from report import ci, selection
from reliable_v19_common import ReliableRequest, rebind
from evaluate_reliable_v20 import snapshot
from game_theory_agent.game_theory import empirical_advisor as empirical


def one(seed):
    i=(seed-(98000 if seed>=98000 else 97000))%54
    path=RAW/f"empirical-{seed}.json.gz"
    if path.exists():
        return read(path)
    regime=("normal","recession","tight_supply")[i%3]
    count=(2,5,10)[i//3%3]
    c,s,mem,draft,history=snapshot(regime,count,seed,4+i//9)
    raw=draft.to_dict()
    req=ReliableRequest(draft_action=raw, mode="research", goal="profit", public_history=history, seed=17)
    if "company_A" not in s.strategic_market.active_company_ids:
        # Retain the scheduled case in intention-to-treat; an exited firm cannot intervene.
        gate=dict(finalist="exited",legacy="agent_draft",empirical="agent_draft",reason="企业在干预前已退出",
                  score=0.,steps=0,pool=0)
        save(RAW/f"sealed-empirical-{seed}.json",gate)
        keep=terminal(c,s,mem,seed,raw)["profit"]
        row=dict(seed=seed,count=count,regime=regime,t=s.round,gate=gate,
                 paths={name:keep for name in ("keep","up","finalist")})
        save(path,row)
        return row
    result=empirical.advise(c,s,req)
    assert result["search"]["complete"] and result["draft_action"]==raw
    winner=max((r for r in result["rankings"] if r["candidate"]["id"]!="agent_draft"),
               key=lambda r:(r["safe"],r["score"]))["candidate"]
    price_up=rebind(c,s,"company_A",dict(raw,price_cents=draft.price_cents*105//100)).to_dict()
    gate=dict(finalist=winner["id"],legacy=result["proposed_id"],empirical=result["recommended_id"],
              reason=result["rationale"],score=min(v["lower_bound"] for v in result["validation"]),
              steps=result["search"]["used_steps"],pool=len(result["rankings"]))
    seal=RAW/f"sealed-empirical-{seed}.json"
    if seal.exists():
        assert read(seal)==gate
    else:
        save(seal,gate)
    paths={k:terminal(c,s,mem,seed,v)["profit"] for k,v in
           (("keep",raw),("up",price_up),("finalist",winner["action"]))}
    row=dict(seed=seed,count=count,regime=regime,t=s.round,gate=gate,paths=paths)
    save(path,row)
    return row


def summary(rows):
    out={}
    for discount in (.95,.99):
        values={}
        p=np.array
        for mode in ("legacy","empirical"):
            gains=[];mask=[];scores=[];up=[];short=[]
            for r in rows:
                h=r["paths"]
                y=p(h["finalist"])-h["keep"]
                weights=discount**np.arange(len(y))
                gains.append(float(y@weights))
                up.append(float((p(h["up"])-h["keep"])@weights))
                short.append(float(y[:5]@weights[:5]))
                mask.append(r["gate"][mode]!="agent_draft")
                scores.append(r["gate"]["score"])
            accepted=p(mask)
            values[mode]=dict(**selection(gains,mask,scores),minus_fixed_up=ci(p(gains)*accepted-up),
                         early_gain_terminal_loss=int(np.sum(accepted & (p(short)>0) & (p(gains)<0))),
                         first_five_keep=ci(p(short)*accepted))
        values["fixed_up_keep"]=ci(up)
        out[str(discount)]=values
    out["finalists"]=dict(Counter(r["gate"]["finalist"] for r in rows))
    out["reasons"]=dict(Counter(r["gate"]["reason"] for r in rows))
    return out


if __name__=="__main__":
    import argparse
    parser=argparse.ArgumentParser()
    parser.add_argument("--replicate",action="store_true")
    rep=parser.parse_args().replicate
    seeds=range(98000,98108) if rep else range(97000,97054)
    label="empirical-replication" if rep else "empirical"
    with ProcessPoolExecutor(max_workers=8) as ex:
        rows=list(ex.map(one,seeds))
    save(OUT/f"{label}-paths.json.gz",rows)
    save(OUT/f"{label}.json",summary(rows))
