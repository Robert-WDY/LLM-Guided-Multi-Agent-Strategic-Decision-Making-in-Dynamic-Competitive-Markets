"""Assess the delayed policy's rejected alternatives with clusters at the seed level."""
import numpy as np

from delayed import OUT, models, xrows, payoffs, read, save


def one(m):
    rows=read(OUT/f"delayed-test-{m}.json.gz")
    cases=[c for r in rows for c in r["cases"]]
    _,value,_=models(m)
    offers=[xrows(c["public"]) for c in cases]
    table=value.predict([a+[k] for pair in offers for k in range(4) for a in pair]).reshape(-1,4,2)
    prob=np.array([c["selected"]["prob"] for c in cases])
    weighted=(prob[:,:,None]*table).sum(1)
    alt=1+weighted.argmax(1)
    use=weighted.max(1)>0
    assert np.array_equal(np.where(use,alt,0),[c["selected"]["inferred"] for c in cases])
    y=np.array([payoffs(c,m)[a] for c,a in zip(cases,alt)])
    clusters=lambda x:x.reshape(-1,4).mean(1)
    cover,accepted,all_gains,losses=map(clusters,(use,use*y,y,use*(y < -1e-8)))
    idx=np.random.default_rng(46930).integers(len(rows),size=(12000,len(rows)))
    diff=accepted[idx].mean(1)-cover[idx].mean(1)*all_gains[idx].mean(1)
    risk=losses[idx].mean(1)/cover[idx].mean(1)
    return dict(accepted=int(use.sum()),losses=int((use & (y< -1e-8)).sum()),rejected=int((~use).sum()),
                missed=int((~use & (y>1e-8)).sum()),risk_upper95=float(np.quantile(risk,.95)),
                gate_minus_random=[float(accepted.mean()-cover.mean()*all_gains.mean()),*np.quantile(diff,[.025,.975]).tolist()])


if __name__=="__main__":
    save(OUT/"selection.json",{m:one(m) for m in ("econ","native")})
