"""Preserve heterogeneity of held-out managers, demand and supply conditions."""

import numpy as np
from prices import OUT, MARKETS
from run import read, save
from saturation_report import ci, counts


def sections(rows):
    v = [r["installed"]["1"]["inf"]["gain"] for r in rows]
    d = [r["installed"]["1"]["dir"]["gain"] for r in rows]
    net = [r["installed"]["4"]["inf"]["gain"] for r in rows]
    return dict(
        n=len(rows),
        one=ci(v),
        four=ci(net),
        relative_direct=ci(np.subtract(v, d)),
        once=ci([r["once"]["gain"] for r in rows]),
        risk=counts([r["once"] for r in rows]),
    )


if __name__ == "__main__":
    rows = read(OUT / "seed-metrics.json.gz")
    data = {}
    for m in MARKETS:
        market = [r for r in rows if r["market"] == m]
        for k in (1, 2, 4):
            gaps = [
                r["installed"][str(k)]["inf"]["gap"] - r["sampled"][1]
                for r in market
                if r["sampled"][1] is not None and r["installed"][str(k)]["inf"]["gap"] is not None
            ]
            data[f"{m}-gap-change-{k}"] = ci(gaps)
        for j in range(4):
            group = [
                r for r in market if r["group"] == "random" and r["roles"][r["seats"]["1"][0]] == j
            ]
            if group:
                data[f"{m}-random-role-{j}"] = sections(group)
        if m == "native":
            for j, condition in enumerate(("normal", "recession", "tight_supply")):
                data[f"{m}-{condition}"] = sections([r for r in market if r["seed"] % 3 == j])
    save(OUT / "subgroups.json", data)
