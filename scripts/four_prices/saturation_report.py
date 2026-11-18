"""Seed-level, paired saturation effects, single-opportunity gates and tail accounting."""

import itertools
from functools import lru_cache

import numpy as np
from scipy.stats import beta
from prices import OUT, RAW, MARKETS, GROUPS
from run import read, save, jobs, design, REVIEW

PAIRS = list(itertools.combinations(range(4), 2))


def discounted(a, start=19):
    return (0.99 ** np.arange(61 - start)) @ a[start - 1 :]


def prices(r):
    a = r[18:, :, 0]
    active = r[18:, :, 4] > 0
    valid = np.column_stack([active[:, i] & active[:, j] for i, j in PAIRS])
    gaps = np.column_stack([np.abs(a[:, i] - a[:, j]) for i, j in PAIRS])
    return (
        float((gaps[valid] < 1.0e-8).mean()) if valid.any() else None,
        float(gaps[valid].mean()) if valid.any() else None,
        float(a[active].mean()) if active.any() else None,
    )


def audit(r, b, k, idx, t, threshold):
    y = discounted(r[:, :, 2] - b[:, :, 2], t)[idx]
    used = bool(np.any(np.abs(r[t - 1 : t + 3, idx, 0] - b[t - 1 : t + 3, idx, 0]) > 1.0e-8))
    if not used:
        assert abs(y) < 1.0e-7
    return dict(gain=float(y), used=int(used), lost=int(y < -1.0e-7), large=int(y < -threshold))


def counts(x):
    n = len(x)
    used = sum(r["used"] for r in x)
    lost = sum(r["lost"] for r in x)
    upper = (
        float(beta.ppf(0.9875, lost + 1, used - lost)) if lost < used else (1.0 if used else None)
    )
    return dict(
        n=n,
        used=used,
        lost=lost,
        large=sum(r["large"] for r in x),
        coverage=used / n,
        risk_upper=upper,
    )


def one(record):
    m = record["market"]
    seed = record["seed"]
    probe = record["probe"]
    arrays = {v: np.asarray(p["rows"], float) for v, p in record["paths"].items()}
    masks = {int(k): v for k, v in record["masks"].items()}
    focus = {int(k): v for k, v in record["focal"].items()}
    base = arrays["keep"]
    baseprice = prices(base)
    greedy = prices(arrays["greedy"])
    th = 20000 if m == "native" else 5
    item = dict(
        seed=seed,
        market=m,
        group=record["group"],
        live=record["live"],
        sampled=baseprice,
        greedy=greedy,
        roles=record["roles"],
        requests=len(record["tape"]),
        seats=masks,
    )
    installed = {}
    comparator = {}
    for k, mask in masks.items():
        installed[k] = {}
        comparator[k] = {}
        for name in ("inf", "dir"):
            ident = name + str(k)
            b = arrays[ident]
            gain = discounted(b[:, :, 2] - base[:, :, 2])
            chosen = np.array(mask)
            others = [i for i in range(4) if i not in mask]
            sim, gap, mean = prices(b)
            v = dict(
                gain=float(gain[chosen].mean()),
                sole_seat_gain=float(gain[masks[1][0]]),
                untreated=float(gain[others].mean()) if others else None,
                all_firms=float(gain.mean()),
                same=sim,
                gap=gap,
                price=mean,
                same_delta=(
                    sim - baseprice[0] if sim is not None and baseprice[0] is not None else None
                ),
                price_delta_pct=(
                    100 * (mean / baseprice[2] - 1) if mean is not None and baseprice[2] else None
                ),
                individual={i: float(gain[i]) for i in mask},
            )
            market = np.asarray(record["paths"][ident]["market"], float)
            old = np.asarray(record["paths"]["keep"]["market"], float)
            v["welfare"] = (
                float(discounted(market[:, 0] - old[:, 0])) if m != "appointments" else None
            )
            v["unserved"] = float((market - old)[18:, 1].sum())
            installed[k][name] = v
        audited = focus[k]
        skip = arrays["skip" + str(k)]
        assert np.array_equal(arrays["inf" + str(k)][: probe - 1], skip[: probe - 1])
        comparator[k] = audit(arrays["inf" + str(k)], skip, k, audited, probe, th)
    item["installed"] = installed
    item["audit"] = comparator
    item["once"] = audit(arrays["once"], base, 1, masks[1][0], probe, th)
    item["once_direct"] = audit(arrays["once_direct"], base, 1, masks[1][0], probe, th)
    return item


@lru_cache(None)
def weights(n):
    return np.random.default_rng(188111).multinomial(n, np.full(n, 1 / n), size=4096) / n


def ci(x):
    x = np.array([v for v in x if v is not None], float)
    if not len(x):
        return None
    return [float(x.mean()), *map(float, np.quantile(weights(len(x)) @ x, [0.0125, 0.9875]))]


def tail(x):
    y = np.sort(x)
    return dict(
        min=float(y[0]),
        es5=float(y[: max(1, int(np.ceil(len(y) * 0.05)))].mean()),
        losses=int((y < -1.0e-7).sum()),
    )


def summary(rows, th):
    out = {}
    for k in (1, 2, 4):
        inf = [r["installed"][k]["inf"] for r in rows]
        direct = [r["installed"][k]["dir"] for r in rows]
        gains = np.array([a["gain"] for a in inf])
        risks = counts([r["audit"][k] for r in rows])
        per = ci(gains / len(REVIEW))
        duel = ci([a["gain"] - b["gain"] for a, b in zip(inf, direct)])
        out[k] = dict(
            inferred=ci(gains),
            direct=ci([a["gain"] for a in direct]),
            advantage=duel,
            per_opportunity=per,
            lower_tail=tail(gains),
            worst_individual=min(v for a in inf for v in a["individual"].values()),
            **{
                key: ci([a[key] for a in inf])
                for key in (
                    "untreated",
                    "all_firms",
                    "same",
                    "gap",
                    "same_delta",
                    "price_delta_pct",
                    "welfare",
                    "unserved",
                )
            },
            audit=risks,
            pass_gate=bool(
                per[1] > th
                and risks["coverage"] >= 0.1
                and risks["risk_upper"] is not None
                and risks["risk_upper"] <= 0.1
            ),
        )
    once = counts([r["once"] for r in rows])
    effect = ci([r["once"]["gain"] for r in rows])
    return dict(
        n=len(rows),
        online=sum(r["live"] for r in rows),
        arms=out,
        all_minus_one_same_seat=ci(
            [
                r["installed"][4]["inf"]["sole_seat_gain"] - r["installed"][1]["inf"]["gain"]
                for r in rows
            ]
        ),
        all_minus_one=ci(
            [r["installed"][4]["inf"]["gain"] - r["installed"][1]["inf"]["gain"] for r in rows]
        ),
        baseline_same=ci([r["sampled"][0] for r in rows]),
        greedy_same=ci([r["greedy"][0] for r in rows]),
        stochastic_minus_greedy=ci(
            [
                r["sampled"][0] - r["greedy"][0]
                for r in rows
                if None not in (r["sampled"][0], r["greedy"][0])
            ]
        ),
        once=dict(
            **once,
            effect=effect,
            tail=tail([r["once"]["gain"] for r in rows]),
            advantage=ci([r["once"]["gain"] - r["once_direct"]["gain"] for r in rows]),
            pass_gate=bool(
                effect[1] > th
                and once["coverage"] >= 0.1
                and once["risk_upper"] is not None
                and once["risk_upper"] <= 0.1
            ),
        ),
    )


def decompose(row, which, baseline, seat, start):
    a = np.array(row["paths"][which]["rows"])
    b = np.array(row["paths"][baseline]["rows"])
    weights_ = 0.99 ** np.arange(61 - start)
    x = a[start - 1 :, seat]
    y = b[start - 1 :, seat]
    diff = x[:, 2] - y[:, 2]
    rev = x[:, 3] - y[:, 3]
    cost = rev - diff
    periods = [
        dict(
            round=start + i,
            delta=float(weights_[i] * d),
            p=x[i, 0],
            base_p=y[i, 0],
            q=x[i, 1],
            base_q=y[i, 1],
            profit=x[i, 2],
            base_profit=y[i, 2],
        )
        for i, d in enumerate(diff)
    ]
    # Exact additive split, using realized revenue per realized unit and an explicit rounding residual.
    alpha = (1.0, 3.2, 10.0)[row["seed"] % 3]
    unit = 100 if row["market"] == "native" else alpha
    volume = (x[:, 1] - y[:, 1]) * y[:, 0] / unit
    price = (x[:, 0] - y[:, 0]) * x[:, 1] / unit
    cash = dict(
        price=float(weights_ @ price),
        volume=float(weights_ @ volume),
        other_revenue=float(weights_ @ (rev - price - volume)),
        cost_effect=float(-weights_ @ cost),
    )
    result = float(weights_ @ diff)
    assert abs(result - sum(cash.values())) < 1.0e-5
    return dict(
        seed=row["seed"],
        group=row["group"],
        seat=seat,
        policy=which,
        baseline=baseline,
        start=start,
        gain=result,
        first_four=float(weights_[:4] @ diff[:4]),
        rest=float(weights_[4:] @ diff[4:]),
        parts=cash,
        worst_rounds=sorted(periods, key=lambda v: v["delta"])[:4],
        exit=[next((i + start for i, v in enumerate(s[:, 5]) if not v), None) for s in (x, y)],
    )


def aggregate():
    rows = []
    cases = {}
    for m, g, i in jobs():
        seed = design(m, g, i)[0]
        row = read(RAW / f"{seed}.json.gz")
        rows.append(one(row))
    report = {}
    for m in MARKETS:
        r = [v for v in rows if v["market"] == m]
        th = 20000 if m == "native" else 5
        report[m] = {
            name: summary(group, th)
            for name, group in [
                ("all", r),
                ("backend", [v for v in r if v["live"]]),
                *[(g, [v for v in r if v["group"] == g]) for g in GROUPS],
            ]
        }
        for k in (1, 2, 4):
            bad = sorted(
                (value, v["seed"], j)
                for v in r
                for j, value in v["installed"][k]["inf"]["individual"].items()
            )[:3]
            cases[f"{m}-rolling-{k}"] = [
                decompose(read(RAW / f"{s}.json.gz"), f"inf{k}", "keep", j, 19) for _, s, j in bad
            ]
            bad = sorted((v["audit"][k]["gain"], v["seed"]) for v in r)[:3]
            found = []
            for _, s in bad:
                row = read(RAW / f"{s}.json.gz")
                found.append(
                    decompose(row, f"inf{k}", f"skip{k}", row["focal"][str(k)], row["probe"])
                )
            cases[f"{m}-single-switch-{k}"] = found
    save(OUT / "summary.json", report)
    save(OUT / "tail.json", cases)
    save(OUT / "seed-metrics.json.gz", rows)
    # Separate archives make individual retained market trajectories directly recoverable.
    for m in MARKETS:
        for g in GROUPS:
            ids = [r["seed"] for r in rows if r["market"] == m and r["group"] == g]
            save(OUT / f"{m}-{g}.json.gz", [read(RAW / f"{s}.json.gz") for s in ids])


if __name__ == "__main__":
    aggregate()
