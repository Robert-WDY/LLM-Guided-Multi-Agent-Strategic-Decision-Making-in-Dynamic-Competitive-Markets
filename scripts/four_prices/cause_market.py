"""Independent pricing equations, native ledgers, clone invariance, and a median alias witness."""

import numpy as np
from cause_trials import OUT
from guard_study import OUTPUT as PRIOR
from prices import number
from run import read, save, effective
from world import World


def demand(w, prices):
    kw = w.kw
    quality = kw["a"][0]
    if w.t >= 15 + number(w.seed, "demand") % 25 and w.seed % 4 in (1, 2):
        quality = 2.35 if w.seed % 4 == 1 else 1.65
    utility = np.exp((quality - prices / kw["alpha"][0]) / kw["mu"])
    q = kw["multiplier"] * utility / (np.exp(kw["a0"] / kw["mu"]) + utility.sum())
    return q, q * (prices / kw["alpha"][0] - w.cost)


def witness():
    w = World("econ", 180000)
    grid = np.linspace(w.low, w.high, 21).round(2)
    for med in grid[6:-4]:
        for focal in grid[4:-4]:
            opts = {}
            for left in grid[grid <= med]:
                for right in grid[grid >= med]:
                    peers = [left, med, right]
                    vals = [
                        round(float(demand(w, np.round([p, *peers], 2))[1][0]), 2)
                        for p in (focal, 0.96 * focal, 1.04 * focal)
                    ]
                    opts[int(np.argmax(vals))] = dict(rivals=peers, profits=vals)
                    if 1 in opts and 2 in opts:
                        for scenario in (opts[1], opts[2]):
                            exact = [
                                w.clone().settle([p, *scenario["rivals"]])[0][0][2]
                                for p in (focal, 0.96 * focal, 1.04 * focal)
                            ]
                            assert exact == scenario["profits"]
                        return dict(
                            draft=focal, median=med, cut=opts[1], raise_=opts[2]
                        )
    raise AssertionError("No opposite best action found")


def audit_record(r, mode):
    p = r["paths"][mode]
    w = World(r["market"], r["seed"])
    terms = {d["t"]: d["requested"] for d in p["decisions"]}
    counts = dict(ledger=0, constrained=0, exited=0, clone=0)
    for t in range(1, 61):
        if t in terms:
            targets = terms[t]
        legal = [effective(w, j, x) for j, x in enumerate(targets)]
        if w.market == "econ":
            pr = np.round(np.clip(targets, w.low, w.high), 2)
            q, net = demand(w, pr)
        if t in (19, 27, 43):
            branch = w.clone()
            copied = branch.settle(targets)
        rows, market = w.settle(targets)
        assert rows == p["rows"][t - 1] and market == p["market"][t - 1]
        if t in (19, 27, 43):
            assert copied == (rows, market)
            counts["clone"] += 1
        for j, row in enumerate(rows):
            if row[4]:
                assert np.isclose(row[0], legal[j])
            counts["constrained"] += int(
                row[4]
                and abs(targets[j] - row[0])
                > (0.51 if w.market == "native" else 0.0051)
            )
            counts["exited"] += int(not row[4])
            if w.market == "econ":
                assert (
                    abs(row[1] - q[j]) <= 0.005001 and abs(row[2] - net[j]) <= 0.005001
                )
            else:
                f = w.state.companies[j].financial
                expected = (
                    f.round_revenue_cents
                    - f.round_variable_cost_cents
                    - f.round_fixed_spend_cents
                    - f.round_incident_cost_cents
                    - f.round_operating_cost_cents
                    + (f.round_government_support_cents or 0)
                    - (f.round_regulatory_fine_cents or 0)
                )
                assert expected == round(100 * row[2])
                if not row[4]:
                    assert row[1] == row[2] == 0
            counts["ledger"] += 1
    return counts


if __name__ == "__main__":
    results = {}
    for m, first in [("econ", 195000), ("native", 196000)]:
        total = {}
        for i in range(36):
            r = read(PRIOR / "menu" / f"{first + i}.json.gz")
            for arm in ("keep", "safe_inf1", "safe_inf4"):
                for k, n in audit_record(r, arm).items():
                    total[k] = total.get(k, 0) + n
        results[m] = total
    s = read(PRIOR / "tail.json")["native-4"]
    results["tail"] = dict(
        seed=s["seed"],
        seat=s["seat"],
        total=s["natural"]["gain"],
        withdraw=s["contracts"]["27"],
    )
    results["median_alias"] = witness()
    save(OUT / "market-audit.json", results)
