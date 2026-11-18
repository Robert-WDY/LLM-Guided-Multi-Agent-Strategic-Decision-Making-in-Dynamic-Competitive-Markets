"""Fixed-volume ranking, matched risk thresholds, and timing of terminal tail losses."""

import numpy as np
from cause_trials import MARKETS
from cause_value import metrics
from prices import number
from renewal_risk import estimates, decisions
from renewal_trials import tasks, archive, OUT
from run import read, save
from saturation_report import ci


def comparisons(rows, predictions, calibration, hurdle):
    enabled = np.array([s["enabled"] for s in predictions])
    eligible = np.where(enabled)[0]
    count = (len(eligible) + 3) // 4
    rankings = {}
    shared = {}
    for name in ("renewal", "equal_budget", "one_stage", "random"):
        if name == "random":
            risk = [number("renewal-rank", r["seed"]) for r in rows]
        else:
            values = np.asarray([s["scores"][name] for s in predictions])
            _, risk = estimates(values)
            risk = risk[:, 0]
            for source in ("renewal", "equal_budget"):
                cutoff = calibration[source]["chosen"]["cutoff"]
                actions = decisions(values, enabled, cutoff, hurdle)
                key = f"{name}_using_{source}"
                for r, a in zip(rows, actions):
                    r["actions"][key] = int(a)
                shared[key] = metrics(rows, key, hurdle)
        used = set(sorted(eligible, key=lambda j: risk[j])[:count])
        for i, r in enumerate(rows):
            r["actions"][name + "_rank_cut"] = int(i in used)
        rankings[name] = metrics(rows, name + "_rank_cut", hurdle)

    def pair(a, b):
        return ci(
            [r["gains"][r["actions"][a]] - r["gains"][r["actions"][b]] for r in rows]
        )

    return dict(
        quota=count,
        eligible=len(eligible),
        fixed_quota=rankings,
        same_threshold=shared,
        fixed_quota_diff=pair("renewal_rank_cut", "equal_budget_rank_cut"),
        same_cutoff_diff={
            k: pair("renewal_using_" + k, "equal_budget_using_" + k)
            for k in ("renewal", "equal_budget")
        },
    )


def abstention(rows, predictions):
    eligible = [
        (r, s)
        for r, s in zip(rows, predictions)
        if s["enabled"] and any(r["changed"][1:])
    ]
    rejected = [r for r, s in eligible if s["choices"]["renewal"] == 0]

    def should_keep(r):
        return max(r["gains"][1:]) <= 1e-7

    total = sum(should_keep(r) for r, s in eligible)
    correct = sum(should_keep(r) for r in rejected)
    mistake = sum(not should_keep(r) for r in rejected)
    best = [max(r["gains"]) for r in rows]
    regret = [
        b - r["gains"][s["choices"]["renewal"]]
        for b, r, s in zip(best, rows, predictions)
    ]
    return dict(
        evaluable=len(eligible),
        actual_best_to_keep=total,
        rejected=len(rejected),
        correct_rejections=correct,
        missed_profitable_actions=mistake,
        profitable_move_value_missed=sum(max(r["gains"]) for r in rejected),
        harmful_to_act_and_not_rejected=total - correct,
        hindsight_oracle=ci(best),
        regret_to_oracle=ci(regret),
    )


def tail(record, arm, snap):
    t, seat = record["probe"], record["masks"]["1"][0]
    base, path = (
        record["paths"][name]
        for name in ("keep", ("audit_cut", "audit_raise")[arm - 1])
    )
    a, b = np.array(path["rows"]), np.array(base["rows"])
    profit = (a[t - 1 :, seat, 2] - b[t - 1 :, seat, 2]) * 0.99 ** np.arange(61 - t)
    periods = [float(x.sum()) for x in (profit[:4], profit[4:12], profit[12:])]
    _, risk = estimates(snap["scores"]["renewal"])
    renewals = []
    forecast = np.asarray(snap["path"]).reshape(2, 2, 2)
    for step, time in enumerate((t + 4, t + 8)):
        active = [bool(v[time - 1, seat, 4]) for v in (a, b)]
        quote = [
            next(d["requested"][seat] for d in p["decisions"] if d["t"] == time)
            if flag
            else None
            for p, flag in zip((path, base), active)
        ]
        renewals.append(
            dict(
                t=time,
                requested_arm_keep=quote,
                observed_log_effect=float(np.log(quote[0] / quote[1]))
                if all(active)
                else None,
                predicted_log_effect=float(forecast[arm - 1, step, 0]),
            )
        )
    return dict(
        seed=record["seed"],
        t=t,
        seat=seat,
        arm=arm,
        group=record["group"],
        predicted_conditional_risk=float(risk[0, arm - 1]),
        periods=periods,
        total=float(profit.sum()),
        renewals=renewals,
    )


def analyze(market, calibration):
    rows = read(OUT / f"decisions-{market}.json.gz")
    predictions = read(OUT / f"predictions-{market}.json.gz")
    files = {t: archive(t) for t in tasks() if t[0] == market}
    by_seed = {int(p.name.split(".")[0]): p for p in files.values()}
    assert [r["seed"] for r in rows] == [s["seed"] for s in predictions]
    report = comparisons(
        rows, predictions, calibration, 20000 if market == "native" else 5
    )
    tails = {}
    for name in ("worst_accepted", "worst_rejected_cut"):
        cases = []
        for r, s in zip(rows, predictions):
            chosen = r["actions"]["renewal"]
            arm = chosen if name == "worst_accepted" else (1 if chosen == 0 else 0)
            if arm and r["changed"][arm] and r["gains"][arm] < 0:
                cases.append((r["gains"][arm], r["seed"], arm, s))
        if cases:
            actual, seed, arm, snap = min(cases, key=lambda v: v[0])
            v = tail(read(by_seed[seed]), arm, snap)
            assert np.isclose(sum(v["periods"]), actual)
            tails[name] = v
        else:
            tails[name] = None
    return dict(**report, tails=tails, abstention=abstention(rows, predictions))


if __name__ == "__main__":
    lock = read(OUT / "freeze.json")
    save(OUT / "diagnostics.json", {m: analyze(m, lock[m]["models"]) for m in MARKETS})
