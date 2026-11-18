"""Verify sample flow/ledger and optionally regenerate selected terminal paths offline."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import tempfile
from unittest.mock import patch

import numpy as np

from results import OUT,ROOT,read


def check():
    req=read(OUT/"numeric-requests.json.gz")
    extra=read(OUT/"numeric-stopping-requests.json.gz")
    assert set(req).isdisjoint(extra)
    for key,v in (req|extra).items():
        assert key==hashlib.sha256(json.dumps(v["body"],separators=(",",":")).encode()).hexdigest()
        assert np.isfinite(v["price"])
    for m,first_n,first_a in (("econ",121000,140000),("native",122000,141000)):
        for tag,start,n in (("independent",first_n,192 if m=="econ" else 4608),("numeric",first_a,192)):
            rows=read(OUT/f"{tag}-{m}.json.gz")
            result=read(OUT/f"{tag}-results.json")
            for name in ("qwen","smol"):
                group=[r for r in rows if r["model"]==name]
                assert [r["seed"] for r in group]==list(range(start,start+n))
                assert sorted(Counter(r["condition"] for r in group).values())==[n//4]*4
                y=[]
                for r in group:
                    t=r["public"]["round"]
                    v=np.asarray(r["profit"])
                    assert len(r["public"]["prices"])==t-1 and v.shape==(3,61-t) and np.isfinite(v).all()
                    y.append((v[r["decision"]["inferred"]]-v[0])@(.99**np.arange(61-t))/(100 if m=="native" else 1))
                    if tag=="numeric":
                        for event in r["prefix_calls"]+sum(r["branch_calls"],[]):
                            assert req[event[1]]["price"]==event[2] and req[event[1]]["body"]["model"]==name
                z=result[m+"-"+name]["0.99"]
                np.testing.assert_allclose(np.mean(y),z["keep"]["mean"],atol=1e-8)
                assert z["lost"]==sum(a< -1e-8 for a in y)
        rows=read(OUT/f"blind-test-{m}.json.gz")
        assert [r["seed"] for r in rows]==list(range(131000,131512))
        for r in rows:
            assert (r["onset"] is None)==(r["condition"]=="stable")
            assert r["onset"] is None or 8<=r["onset"]<=53
            assert all(len(p)==61-t for t,arm in zip(range(19,48,4),r["paths"]) for p in arm)
            assert all(np.isfinite(r["base"])) and len(r["base"])==60
    budget=read(OUT/"budget.json")
    assert budget["commercial_increment_micro_cny"]==0 and budget["remaining_micro_cny"]==155084
    print("Independent seeds, balanced conditions, all request hashes, cash budget and primary profits verified.")


def replay():
    import blind
    import blind_transfer
    import independent
    import numeric_stopping
    import numeric
    import adoption
    from markets import contract,observation,source
    from local_prices import SOURCES,messages
    for name,(_,revision,digest,_) in SOURCES.items():
        model=read(OUT/f"llm-{name}.json.gz")
        assert model["revision"]==revision and model["sha256"]==digest and len(model["entries"])==162
        for k,v in model["entries"].items():
            body=dict(model=name,messages=messages(k),temperature=0,seed=1741,max_tokens=3,grammar='root ::= [ABC]',stream=False)
            assert "ABC"[v["action"]]==v["letter"] and v["prompt"]==messages(k)
            assert hashlib.sha256(json.dumps(body,ensure_ascii=False,separators=(",",":")).encode()).hexdigest()==v["request_sha256"]
    reversal=read(OUT/"menu-reversal.json")
    for name in SOURCES:
        result=read(OUT/f"menu-reversal-{name}.json.gz")
        original=read(OUT/f"llm-{name}.json.gz")["entries"]
        assert result.keys()==original.keys()
        assert reversal[name]["same_price"]==sum(v["action"]==original[k]["action"] for k,v in result.items())
        assert reversal[name]["same_letter"]==sum(v["letter"]==original[k]["letter"] for k,v in result.items())
    assert read(OUT/"stopping-freeze.json")==blind.fingerprint()
    saved=read(OUT/"numeric-requests.json.gz")|read(OUT/"numeric-stopping-requests.json.gz")
    stopping=read(OUT/"numeric-stopping.json.gz")
    assert numeric_stopping.report(stopping)==read(OUT/"numeric-stopping-results.json")
    assert all(k in saved for r in stopping for k in r["requests"])
    lookup=lambda *args: hashlib.sha256(json.dumps(numeric.build_request(*args),separators=(",",":")).encode()).hexdigest()
    def offline(*args):
        key=lookup(*args)
        return key,saved[key]["price"]
    with tempfile.TemporaryDirectory() as temp:
        loc=Path(temp)
        for m in ("econ","native"):
            rows=read(OUT/f"numeric-{m}.json.gz")
            with patch.object(numeric,"RAW",loc),patch.object(numeric,"number_request",offline):
                for row in (*rows[:2],*rows[-2:]):
                    numeric.trial((m,row["model"],row["seed"]))
                    assert row==read(loc/f"numeric-{m}-{row['model']}-{row['seed']}.json.gz")
            with patch.object(numeric_stopping,"RAW",loc),patch.object(numeric,"number_request",offline):
                selected=[r for r in stopping if r["market"]==m]
                for r in rows[:2]:
                    generated=numeric_stopping.one((m,r))
                    assert generated==next(v for v in selected if v["model"]==r["model"] and v["seed"]==r["seed"])
            rows=read(OUT/f"independent-{m}.json.gz")
            with patch.object(independent,"RAW",loc):
                for ix in (0,len(rows)-2):
                    seed=rows[ix]["seed"]
                    independent.work((m,seed,1))
                    assert rows[ix:ix+2]==read(loc/f"independent-{m}-{seed}.json.gz")
            rows=read(OUT/f"blind-test-{m}.json.gz")
            with patch.object(blind,"RAW",loc):
                for r in (rows[0],rows[-1]):
                    assert blind.sample(("test",m,r["seed"]))==r
            # Remove/scramble hidden future events and compare every pre-change public decision.
            checked=0
            for r in rows:
                if r["onset"] is None or r["onset"]<24:continue
                w=source(m,r["seed"],True);w.change="stable";w.when=999
                h=[];d=None
                for t in range(1,r["onset"]+1):
                    d=contract(w,h,d)
                    if t in blind.DATES:
                        j=blind.DATES.index(t)
                        assert observation(h,d,t)==r["x"][j]
                        x=np.asarray(r["x"][j]).copy()
                        pred=blind.models(m)[t].predict([x])[0]
                        assert blind.action(pred[None,:])[0]==r["decisions"][j]["action"]
                        checked+=1
                    h.append(w.execute(d["price"]))
                if checked>=75:break
            assert checked>=75
    assert blind_transfer.report(read(OUT/"blind-transfer.json.gz"))==read(OUT/"blind-transfer-results.json")
    assert adoption.report(read(OUT/"adoption.json.gz"))==read(OUT/"adoption-results.json")
    print("Independent weight identifiers, frozen stopping policies, hidden-label prefix invariance and sampled terminal replays verified offline.")


if __name__=="__main__":
    parser=argparse.ArgumentParser();parser.add_argument("--replay",action="store_true")
    args=parser.parse_args();check()
    if args.replay:replay()

