"""Summaries on independent market seeds; all losses retain their original units."""

from collections import defaultdict
from itertools import combinations
from scipy.stats import beta
import numpy as np
from guard_study import OUTPUT, jobs, path, read, save
from saturation_report import audit, ci, counts, discounted, prices, tail


def opportunity_risk(rows):
    risk = counts(rows)
    used, lost = risk["used"], risk["lost"]
    risk["risk_lower"] = (
        float(beta.ppf(0.0125, lost, used - lost + 1))
        if lost
        else (0.0 if used else None)
    )
    return risk


def gate(effect, risk, threshold):
    return bool(
        effect[1] > threshold
        and risk["coverage"] >= 0.1
        and risk["risk_upper"] is not None
        and risk["risk_upper"] <= 0.1
    )


def case(a, b, seat, start, seed):
    d = a[:, seat, :] - b[:, seat, :]
    flow = d[start - 1 :, 2] * 0.99 ** np.arange(61 - start)
    return dict(
        seed=seed,
        seat=seat,
        start=start,
        gain=float(flow.sum()),
        first_contract=float(flow[:4].sum()),
        later=float(flow[4:].sum()),
        units=float(d[start - 1 :, 1].sum()),
        extra_exit=int(-d[-1, 5]),
        initial_prices=[float(z[start - 1, seat, 0]) for z in (a, b)],
    )


def matched_prices(treated, original):
    """Compare the identical active firm-pair and date in the two worlds."""
    price_differences = []
    tie_differences = []
    for i, j in combinations(range(4), 2):
        active = (
            treated[18:, i, 4]
            * treated[18:, j, 4]
            * original[18:, i, 4]
            * original[18:, j, 4]
        ).astype(bool)

        def distances(rows):
            a, b = rows[18:, i, 0][active], rows[18:, j, 0][active]
            return 2 * abs(a - b) / (a + b), (a == b).astype(float)

        treated_gap, treated_tie = distances(treated)
        base_gap, base_tie = distances(original)
        price_differences.extend(treated_gap - base_gap)
        tie_differences.extend(treated_tie - base_tie)
    return dict(
        pairs=len(tie_differences),
        gap=float(np.mean(price_differences)) if price_differences else None,
        tie=float(np.mean(tie_differences)) if tie_differences else None,
    )


def one(r, interface):
    m, s, t = r["market"], r["seed"], r["probe"]
    threshold = 20000 if m == "native" else 5
    mats = {v: np.asarray(p["rows"], float) for v, p in r["paths"].items()}
    base = mats["keep"]
    assert r["live"] and all(
        a.shape == (60, 4, 6) and np.isfinite(a).all() for a in mats.values()
    )
    record = dict(
        seed=s,
        group=r["group"],
        interface=interface,
        market=m,
        calls=len(r["tape"]),
        exit=int(4 - base[-1, :, 5].sum()),
        same=prices(base)[0],
        paths=len(mats),
    )
    if interface != "menu":
        record["clips"] = sum(
            not v["low"] <= float(v["raw"]) <= v["high"] for v in r["tape"].values()
        )
    if interface == "legacy":
        return record
    j = r["masks"]["1"][0]
    single = {}
    for name in ("once", "safe_once"):
        single[name] = audit(mats[name], base, 1, j, t, threshold)
    if single["safe_once"]["used"]:
        assert np.array_equal(mats["safe_once"], mats["once"])
    record["single"] = single
    record["install"] = {}
    for k in (1, 2, 4):
        mask = r["masks"][str(k)]
        j = r["focal"][str(k)]
        arms = {}
        for method in ("inf", "dir", "safe_inf", "safe_dir"):
            a = mats[method + str(k)]
            gains = discounted(a[:, :, 2] - base[:, :, 2])[mask]
            p = prices(a)
            worst = mask[int(np.argmin(gains))]
            arms[method] = dict(
                gain=float(gains.mean()),
                individuals=list(gains),
                worst=case(a, base, worst, 19, s),
                same=p[0],
                gap=p[1],
                matched=matched_prices(a, base),
                extra_exits=int((base[-1, mask, 5] - a[-1, mask, 5]).sum()),
            )
        a = mats["safe_inf" + str(k)]
        b = mats["safe_skip" + str(k)]
        assert np.array_equal(a[: t - 1], b[: t - 1])
        record["install"][k] = dict(
            **arms,
            probe=audit(a, b, k, j, t, threshold),
            probe_case=case(a, b, j, t, s),
        )
    return record


def summarize(rows):
    threshold = 20000 if rows[0]["market"] == "native" else 5
    out = dict(
        n=len(rows),
        calls=sum(r["calls"] for r in rows),
        paths=sum(r["paths"] for r in rows),
        baseline_same=ci([r["same"] for r in rows]),
        exits=sum(r["exit"] for r in rows),
        clips=sum(r.get("clips", 0) for r in rows),
    )
    if rows[0]["interface"] == "legacy":
        return out
    once = [r["single"]["safe_once"] for r in rows]
    rejected = [
        r["single"]["once"]["gain"]
        for r in rows
        if r["single"]["once"]["used"] and not r["single"]["safe_once"]["used"]
    ]
    once_risk = opportunity_risk(once)
    effect = ci([v["gain"] for v in once])
    out["once"] = dict(
        **once_risk,
        effect=effect,
        passed=gate(effect, once_risk, threshold),
        rejected=len(rejected),
        rejected_would_lose=sum(v < 0 for v in rejected),
        rejected_gain=ci(rejected),
        accepted_gain=ci([v["gain"] for v in once if v["used"]]),
    )
    out["install"] = {}
    for k in (1, 2, 4):
        paths = [r["install"][k] for r in rows]
        methods = {}
        for method in ("inf", "dir", "safe_inf", "safe_dir"):
            a = [p[method] for p in paths]
            individual = [x for b in a for x in b["individuals"]]
            methods[method] = dict(
                effect=ci([b["gain"] for b in a]),
                individual_tail=tail(individual),
                worst=min((b["worst"] for b in a), key=lambda v: v["gain"]),
                extra_exits=sum(b["extra_exits"] for b in a),
            )
        gain = np.asarray([p["safe_inf"]["gain"] for p in paths])
        risk = opportunity_risk([p["probe"] for p in paths])
        per = ci(gain / 10)
        out["install"][k] = dict(
            **methods,
            per=per,
            risk=risk,
            passed=gate(per, risk, threshold),
            vs_previous=ci(gain - [p["inf"]["gain"] for p in paths]),
            vs_direct=ci(gain - [p["dir"]["gain"] for p in paths]),
            vs_safe_direct=ci(gain - [p["safe_dir"]["gain"] for p in paths]),
            wins_over_safe_direct=sum(
                int(x > p["safe_dir"]["gain"] + 1e-7) for x, p in zip(gain, paths)
            ),
            losses_to_safe_direct=sum(
                int(x < p["safe_dir"]["gain"] - 1e-7) for x, p in zip(gain, paths)
            ),
            matched_gap=ci([p["safe_inf"]["matched"]["gap"] for p in paths]),
            matched_tie=ci([p["safe_inf"]["matched"]["tie"] for p in paths]),
            coactive_pairs=sum(p["safe_inf"]["matched"]["pairs"] for p in paths),
            same_change=ci(
                [
                    p["safe_inf"]["same"] - r["same"]
                    for p, r in zip(paths, rows)
                    if p["safe_inf"]["same"] is not None and r["same"] is not None
                ]
            ),
            groups={
                g: ci(
                    [
                        p["safe_inf"]["gain"]
                        for r, p in zip(rows, paths)
                        if r["group"] == g
                    ]
                )
                for g in sorted({r["group"] for r in rows})
            },
            worst_probe=min((p["probe_case"] for p in paths), key=lambda v: v["gain"]),
        )
    return out


if __name__ == "__main__":
    metrics = defaultdict(list)
    for task in jobs():
        interface, market = task[:2]
        metrics[f"{interface}-{market}"].append(one(read(path(task)), interface))
    save(OUTPUT / "metrics.json.gz", metrics)
    save(OUTPUT / "summary.json", {k: summarize(v) for k, v in metrics.items()})
