"""Remove the rival's causal response while preserving its entire baseline price path."""
from concurrent.futures import ProcessPoolExecutor
from types import SimpleNamespace
import numpy as np

from trials import RAW, OUT, CONDITIONS, draw, world, limits, decision, settle, save, read
from public_value_selector import choose, verify
from report import ci


def one(seed):
    condition = CONDITIONS[(seed-96000)%4]
    w = world("econ", seed, condition)
    probe = 23+4*(draw(seed, "opportunity")%6)
    h = []
    term = None
    for t in range(1, probe+1):
        lo,hi = limits(w)
        term = decision(w,h,term,None,seed,lo,hi,[])
        if t < probe:
            h.append(settle(w,term["price"],lo,hi))
    choice = choose("econ", {"prices":[r[:2] for r in h],"draft":term["draft"],"round":probe})
    def branch(mult,tape=None):
        b = w.clone()
        if tape is not None:
            # With the observed peer tape the transition needs only the common demand shock.
            b.change = "demand" if condition == "demand" else "stable"
            b.op = SimpleNamespace(price=lambda time,last,peer: tape[time-(probe-1)])
        rows = list(h)
        d = dict(term,price=float(np.clip(term["draft"]*mult,b.low,b.high)))
        for t in range(probe,61):
            lo,hi = limits(b)
            if t>probe:
                d = decision(b,rows,d,None,seed,lo,hi,[])
            rows.append(settle(b,d["price"],lo,hi))
        return np.asarray(rows[probe-1:])
    keep=branch(1.)
    tape=keep[:,1]
    assert np.array_equal(keep,branch(1.,tape))
    differences={}
    for name,mult in (("up",1.04),("down",.96)):
        live,fixed=branch(mult),branch(mult,tape)
        if condition in ("stable","demand"):
            assert np.array_equal(live,fixed)
        assert np.array_equal(fixed[:,1],tape)
        discount=.99**np.arange(len(keep))
        differences[name]=dict(live=float((live[:,3]-keep[:,3])@discount),
                               fixed=float((fixed[:,3]-keep[:,3])@discount),
                               peer_moved=bool(np.any(live[:,1]!=fixed[:,1])))
    return dict(seed=seed,condition=condition,decision=choice,outcomes=differences)


def summary(rows):
    result={}
    for condition in CONDITIONS:
        part=[r for r in rows if r["condition"]==condition]
        mechanisms=[]
        absolute=[]
        edges=[]
        moved=0
        for r in part:
            d=r["outcomes"]
            effect=[v["live"]-v["fixed"] for v in d.values()]
            absolute.append(float(np.mean(np.abs(effect))))
            moved+=any(v["peer_moved"] for v in d.values())
            c,g=[r["decision"][m] for m in ("conditional","generic")]
            def value(v,kind):
                return d[v["arm"]][kind] if v["use"] else 0.
            mechanisms.append(value(c,"live")-value(c,"fixed"))
            edges.append(value(c,"live")-value(g,"live"))
        result[condition]=dict(n=len(part),peer_moved=moved,absolute_response_value=ci(absolute),
                               conditional_policy_response_value=ci(mechanisms),conditional_minus_generic=ci(edges))
    return result


if __name__=="__main__":
    verify()
    with ProcessPoolExecutor(max_workers=6) as ex:
        rows=list(ex.map(one,range(96000,96080)))
    save(OUT/"rival-replay-path-effects.json.gz", rows)
    save(OUT/"rival-replay.json",summary(rows))
