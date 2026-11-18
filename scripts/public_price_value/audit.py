"""Replay every confirmatory price path from exported requests without a provider connection."""
from concurrent.futures import ProcessPoolExecutor
import numpy as np

from trials import OUT, world, limits, settle, read, save
from public_value_selector import choose, verify, features
from report import summarize, native
from rival_replay import summary as rival_summary
from empirical_terminal import summary as empirical_summary


def replay(r):
    source=world(r["market"],r["seed"],r["condition"])
    for posted in r["public"]["prices"]:
        # Native settlements may clip requests; replaying their executed prices suffices for state.
        a=source.execute(posted[0])
        np.testing.assert_allclose(a[:2],posted,rtol=0,atol=1e-8)
    assert choose(r["market"],r["public"])==r["decision"]
    assert source.t==r["public"]["round"]
    active=source.active()
    for h in r["paths"].values():
        w=source.clone()
        for row in h["settled"]:
            actual=settle(w,row[6],*limits(w))
            np.testing.assert_allclose(row,actual,rtol=0,atol=1e-8)
        assert w.t==61
    return active


def main():
    verify()
    checks={}
    summary=read(OUT/"summary.json")
    with ProcessPoolExecutor(max_workers=8) as pool:
        for phase in ("test","real"):
            for market in ("econ","native"):
                tag=f"{phase}-{market}"
                rows=read(OUT/f"{tag}-paths.json.gz")
                actives=list(pool.map(replay,rows))
                assert summary[tag]==summarize(rows)
                try:
                    features(dict(rows[0]["public"], rival_cash=99), "conditional")
                except ValueError:
                    pass
                else:
                    raise AssertionError("nonpublic fields accepted")
                checks[tag]=dict(seeds=len(rows),active=sum(actives),all_three_paths_replayed=len(rows),
                                 continuation_rounds=sum(3*(61-r["public"]["round"]) for r in rows))
                if phase=="real":
                    clean=[r for r in rows if not r["prefix_fallback"] and not any(p["fallback"] for p in r["paths"].values())]
                    checks[tag]["without_any_fallback_n"]=len(clean)
                    checks[tag]["without_any_fallback_gate_keep"]=summarize(clean)["conditional"]["gate_keep"]
    assert summary["native-advisor"]==native()
    assert read(OUT/"rival-replay.json")==rival_summary(read(OUT/"rival-replay-path-effects.json.gz"))
    for label,first,n in (("empirical",97000,54),("empirical-replication",98000,108)):
        rows=read(OUT/f"{label}-paths.json.gz")
        assert [r["seed"] for r in rows]==list(range(first,first+n))
        assert all(len(p)==21-r["t"] for r in rows for p in r["paths"].values())
        assert read(OUT/f"{label}.json")==empirical_summary(rows)
    from undiscounted import results
    assert read(OUT/"undiscounted.json")==results()
    save(OUT/"replay-audit.json",checks)


if __name__=="__main__":
    main()
