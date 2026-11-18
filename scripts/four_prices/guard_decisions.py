"""Diagnose the reasons for abstention against matched single-contract terminal labels."""

from collections import defaultdict
import numpy as np
from guard_study import OUTPUT, guard, jobs, path
from renewal_study import files, tasks
from run import read, save
from saturation_report import audit, ci


def label(r):
    t = r["probe"]
    j = r["masks"]["1"][0]
    p = r["paths"]["once"]
    decision = next(d for d in p["decisions"] if d["t"] == t)
    h = []
    for rows in p["rows"][: t - 1]:
        peers = [v[0] for i, v in enumerate(rows) if i != j and v[4]]
        h.append([rows[j][0], float(np.median(peers)) if peers else rows[j][0]])
    market = r["market"]
    result = guard().explain(
        "native" if market == "native" else "econ",
        h,
        decision["draft"][j],
        t,
        decision["selected"][j],
    )
    actual = next(
        d["selected"][j] for d in r["paths"]["safe_once"]["decisions"] if d["t"] == t
    )
    assert actual == result["action"]
    a = np.array(p["rows"])
    b = np.array(r["paths"]["keep"]["rows"])
    y = audit(a, b, 1, j, t, 20000 if market == "native" else 5)
    return dict(
        seed=r["seed"], market=market, group=r["group"], t=t, seat=j, **result, **y
    )


def report(rows):
    groups = defaultdict(list)
    for r in rows:
        groups[r["reason"]].append(r)
    return {
        reason: dict(
            n=len(sample),
            effective=sum(v["used"] for v in sample),
            profitable=sum(v["gain"] > 1e-7 for v in sample),
            loss=sum(v["lost"] for v in sample),
            terminal_profit=ci([v["gain"] for v in sample]),
            worst=min(v["gain"] for v in sample),
            best=max(v["gain"] for v in sample),
        )
        for reason, sample in groups.items()
    }


if __name__ == "__main__":
    groups = defaultdict(list)
    for task in jobs():
        interface, m = task[:2]
        if interface != "legacy":
            groups[f"{interface}-{m}"].append(label(read(path(task))))
    for task in tasks():
        m = task[0]
        groups["menu-" + m].append(label(read(files(task)[0])))
    save(OUTPUT / "abstention.json", {k: report(v) for k, v in groups.items()})
    save(OUTPUT / "decisions.json.gz", groups)
