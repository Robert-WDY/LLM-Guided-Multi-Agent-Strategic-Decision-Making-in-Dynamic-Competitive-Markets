"""Use the frozen public-price advisor on archived EconEvals triplets and new native drafts."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import numpy as np

from paired import RAW, ROOT, OUT, read, save
import trials
from public_value_selector import choose, verify
from paid import connect, initialize

BASE_COST=49_329_416


def archived():
    old=ROOT/"runs/public-forecast-v1"
    files=sorted(old.glob("direction-econ-*.json.gz"))
    assert len(files)==80
    compact=[]
    for f in files:
        r=read(f)[0]
        v=r["opportunity"]
        t=v["t"]
        public=dict(prices=[x[:2] for x in r["history"][:t-1]],draft=v["draft"],round=t)
        suffix=[r["history"][t-1:],r["branches"]["down"],r["branches"]["up"]]
        for h,m in zip(suffix,(1.,.96,1.04)):
            assert len(h)==61-t and np.isclose(h[0][6],np.clip(public["draft"]*m,h[0][7],h[0][8]))
        compact.append(dict(public=public,provider=r["provider"],seed=r["seed"],condition=r["condition"],
                     decisions=choose("econ",public),profit=[np.asarray(h)[:,3].tolist() for h in suffix],
                     fallback=r["fallbacks"]))
    assert len({(v["provider"],v["seed"]) for v in compact})==80
    save(OUT/"archive-econ.json.gz",compact)


def fresh():
    initialize()
    ledger=RAW/"budget-start.json"
    with connect() as db:
        ident,initial=db.execute("select max(id),sum(cost) from calls").fetchone()
    if not ledger.exists():
        if initial!=BASE_COST:
            raise RuntimeError("unexpected initial shared ledger")
        save(ledger,dict(last_id=ident,initial=initial))
    start=read(ledger)
    trials.RAW=RAW
    trials.SPLITS={**trials.SPLITS,"real":(105000,8)}
    with ThreadPoolExecutor(max_workers=4) as pool:
        list(pool.map(trials.run,[("real","native",s) for s in range(105000,105008)]))
    rows=[read(RAW/f"real-native-{s}.json.gz") for s in range(105000,105008)]
    for r in rows:
        assert r["decision"]==read(RAW/f"sealed-native-{r['seed']}.json")
    save(OUT/"fresh-native.json.gz",rows)
    with connect() as db:
        end=db.execute("select sum(cost) from calls").fetchone()[0]
        counts=db.execute("select provider,status,count(*),sum(cost) from calls where id>? group by 1,2",
                          (start["last_id"],)).fetchall()
    assert end<=50_000_000 and not any(row[1]=="reserved" for row in counts)
    save(OUT/"budget.json",dict(initial_micro_cny=start["initial"],final_micro_cny=end,
          remaining_micro_cny=50_000_000-end,columns=["provider","status","requests","micro_cny"],rows=counts))


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("phase",choices=("archive","fresh"))
    phase=parser.parse_args().phase
    verify()
    (fresh if phase=="fresh" else archived)()
