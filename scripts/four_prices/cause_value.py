"""Train on complete three-action labels and isolate the value of forecasting other sellers."""

import argparse
import numpy as np
from sklearn.ensemble import ExtraTreesRegressor
from game_theory_agent.game_theory.quote_panel import (
    rivals,
    seller_features,
    public_state,
)
from game_theory_agent.game_theory.public_price_guard import PublicPriceGuard
from cause_trials import OUT, MARKETS, archive, jobs
from guard_study import guard, OUTPUT as PRIOR
from independent import frozen_actions
from run import REVIEW, read, save
from saturation_report import discounted, counts, ci

LABELS = ("keep", "audit_cut", "audit_raise")
DEPLOYABLE = (
    "legacy",
    "legacy_simple",
    "legacy_uniform",
    "value_guard",
    "frequency_guard",
    "direct",
    "persistence",
    "forecast",
)


def public_panel(rows):
    return np.where(rows[:, :, 4] > 0, rows[:, :, 0], np.nan)


def moves(history, seat, future):
    peers = rivals(history, seat)
    valid = [bool(np.isfinite(future[j])) for j in peers]
    gains = [
        float(np.log(future[j] / history[-1, j])) if ok else 0.0
        for j, ok in zip(peers, valid)
    ]
    return gains + [0.0] * (3 - len(peers)), valid + [False] * (3 - len(peers))


def data(record):
    t, j = record["probe"], record["masks"]["1"][0]
    paths = record["paths"]
    a = np.array([paths[name]["rows"] for name in LABELS])
    assert np.array_equal(a[0, : t - 1], a[1, : t - 1]) and np.array_equal(
        a[0, : t - 1], a[2, : t - 1]
    )
    others = [i for i in range(4) if i != j]
    assert all(
        np.array_equal(a[0, t - 1, others, 0], arm[t - 1, others, 0]) for arm in a
    )
    panel = public_panel(a[0])
    h = panel[: t - 1]
    dec = next(d for d in paths["keep"]["decisions"] if d["t"] == t)
    draft = dec["draft"][j]
    public = []
    for row, settled in zip(h, a[0]):
        peers = row[others]
        peer = np.nanmedian(peers) if np.isfinite(peers).any() else settled[j, 0]
        public.append([settled[j, 0], peer])
    gains = [float(discounted(arm[:, :, 2] - a[0, :, :, 2], t)[j]) for arm in a]
    changed = [
        bool(np.any(arm[t - 1 : t + 3, j, 0] != a[0, t - 1 : t + 3, j, 0])) for arm in a
    ]
    assert all(abs(y) < 1e-6 or flag for y, flag in zip(gains, changed))
    reaction = []
    for arm in a[1:]:
        future = public_panel(arm)[t + 3, others]
        base = panel[t + 3, others]
        valid = np.isfinite(future) & np.isfinite(base)
        reaction.extend(np.abs(np.log(future[valid] / base[valid])).tolist())
    return dict(
        seed=record["seed"],
        t=t,
        seat=j,
        draft=draft,
        panel=panel,
        h=h,
        state=public_state(h, j, t, draft),
        actual=moves(h, j, panel[t - 1])[0],
        gains=gains,
        changed=changed,
        enabled=bool(np.isfinite(h[-1, j]) and rivals(h, j)),
        legacy=dict(prices=public, round=t, draft=draft),
        reaction=reaction,
    )


def dataset(phase, market):
    return [data(read(archive(task))) for task in jobs(phase) if task[1] == market]


def estimator():
    return ExtraTreesRegressor(
        n_estimators=96,
        min_samples_leaf=8,
        max_features=1.0,
        random_state=217341,
        n_jobs=1,
    )


def fit(rows):
    features = []
    targets = []
    for row in rows:
        p = row["panel"]
        for t in REVIEW:
            for j in range(4):
                end = p[[t - 1, t + 3], j]
                if np.isfinite(p[t - 2, j]) and np.isfinite(end).all():
                    features.append(seller_features(p[: t - 1], j, t))
                    targets.append(np.log(end / p[t - 2, j]))
    forecaster = estimator().fit(features, targets)
    y = [r["gains"][1:] for r in rows]
    direct = estimator().fit([r["state"] for r in rows], y)
    value = estimator().fit([r["state"] + r["actual"] for r in rows], y)
    return forecaster, direct, value


def run(market, rows, models):
    forecaster, direct, value = models
    points = [r["legacy"] for r in rows]
    old = frozen_actions(market, points)
    library = read(PRIOR / "bank.json.gz")
    strict = PublicPriceGuard(library)
    X = np.asarray([r["state"] for r in rows])
    actual = np.asarray([r["actual"] for r in rows])
    next_price = np.zeros_like(actual)
    further = np.zeros_like(actual)
    forecast_x = []
    coords = []
    for i, r in enumerate(rows):
        for k, j in enumerate(rivals(r["h"], r["seat"])):
            coords.append((i, k))
            forecast_x.append(seller_features(r["h"], j, r["t"]))
    if coords:
        for (i, k), v in zip(coords, forecaster.predict(forecast_x)):
            next_price[i, k], further[i, k] = v

    def select(pred):
        return np.column_stack([np.zeros(len(rows)), pred]).argmax(axis=1)

    choices = {"direct": select(direct.predict(X))}
    for key, opponents in [
        ("persistence", np.zeros_like(actual)),
        ("forecast", next_price),
        ("known_next", actual),
    ]:
        choices[key] = select(value.predict(np.column_stack((X, opponents))))
    output = []
    for i, r in enumerate(rows):
        j, t, h = r["seat"], r["t"], r["h"]
        p = points[i]
        a = old[i]["inferred"]
        use = {k: int(v[i]) if r["enabled"] else 0 for k, v in choices.items()}
        use.update(
            legacy=a,
            legacy_simple=old[i]["direct"],
            legacy_uniform=old[i]["uniform"],
            value_guard=guard().assess(market, p["prices"], p["draft"], t, a),
            frequency_guard=strict.assess(market, p["prices"], p["draft"], t, a),
            best_terminal=int(np.argmax(r["gains"])),
        )
        directional = [
            strict.assess(market, p["prices"], p["draft"], t, k) for k in (1, 2)
        ]
        errs = {}
        for label, k, pred in [("next", t - 1, next_price), ("later", t + 3, further)]:
            y, flags = moves(h, j, r["panel"][k])
            flags = np.array(flags)
            yp = np.asarray(y)[flags]
            errs[label] = dict(
                n=int(flags.sum()),
                frozen=float(abs(yp).mean()) if len(yp) else None,
                forecast=float(abs(pred[i][flags] - yp).mean()) if len(yp) else None,
            )
        output.append(
            dict(
                seed=r["seed"],
                actions=use,
                gains=r["gains"],
                changed=r["changed"],
                errors=errs,
                directional=directional,
                reaction=r["reaction"],
            )
        )
    return output


def metrics(rows, policy, threshold):
    ys = []
    risk = []
    for r in rows:
        arm = r["actions"][policy]
        gain = r["gains"][arm]
        ys.append(gain)
        risk.append(
            dict(
                used=int(r["changed"][arm]),
                lost=int(gain < -1e-7),
                large=int(gain < -threshold),
            )
        )
    risk = counts(risk)
    effect = ci(ys)
    return dict(
        effect=effect,
        **risk,
        worst=min(ys),
        passed=bool(
            effect[1] > threshold
            and risk["coverage"] >= 0.1
            and risk["risk_upper"] is not None
            and risk["risk_upper"] <= 0.1
        ),
    )


def summarize(rows, threshold):
    policies = {k: metrics(rows, k, threshold) for k in rows[0]["actions"]}
    contrasts = {}
    for a, b in [
        ("forecast", "persistence"),
        ("forecast", "direct"),
        ("known_next", "persistence"),
        ("known_next", "forecast"),
        ("legacy", "legacy_uniform"),
        ("legacy", "legacy_simple"),
        ("frequency_guard", "value_guard"),
    ]:
        contrasts[a + "-" + b] = dict(
            effect=ci(
                [
                    r["gains"][r["actions"][a]] - r["gains"][r["actions"][b]]
                    for r in rows
                ]
            ),
            distinct=sum(r["actions"][a] != r["actions"][b] for r in rows),
        )
    errors = {}
    for k in ("next", "later"):
        e = [r["errors"][k] for r in rows if r["errors"][k]["n"]]
        errors[k] = dict(
            persistence=ci([x["frozen"] for x in e]),
            forecast=ci([x["forecast"] for x in e]),
            difference=ci([x["forecast"] - x["frozen"] for x in e]),
            seeds=len(e),
        )
    response = [float(np.mean(r["reaction"])) for r in rows if r["reaction"]]
    return dict(
        policies=policies, contrasts=contrasts, errors=errors, reaction=ci(response)
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("phase", choices=("calibrate", "test"))
    phase = parser.parse_args().phase
    output = {}
    certificate = {}
    for m in MARKETS:
        model = fit(dataset("fit", m))
        rows = run(m, dataset(phase, m), model)
        threshold = 20000 if m == "native" else 5
        if phase == "test":
            certified = read(OUT / "certificates.json")[m]
            for r in rows:
                for name in DEPLOYABLE:
                    r["actions"]["certified_" + name] = (
                        r["actions"][name] if certified[name] else 0
                    )
        summary = summarize(rows, threshold)
        if phase == "calibrate":
            certificate[m] = {k: summary["policies"][k]["passed"] for k in DEPLOYABLE}
            certificate[m]["external"] = {}
            for index, direction in enumerate(("cut", "raise")):
                candidates = [
                    {**r, "actions": {"offered": r["directional"][index]}} for r in rows
                ]
                certificate[m]["external"][direction] = metrics(
                    candidates, "offered", threshold
                )
        save(OUT / f"{phase}-{m}-decisions.json.gz", rows)
        output[m] = summary
    save(OUT / f"{phase}-summary.json", output)
    if phase == "calibrate":
        save(OUT / "certificates.json", certificate)
