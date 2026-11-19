"""Allocate the same count of terminal labels to more markets or more continuations."""

import argparse

import numpy as np

from cause_trials import MARKETS
from cause_value import metrics
from distribution_trials import OUT, new_jobs, new_path, repeats
from game_theory_agent.game_theory.renewal_review import PublicRenewalReview, estimates
from prices import GROUPS
from renewal_risk import forest, opportunity, public_proposal, targets
from renewal_risk_study import digest, historical
from renewal_trials import OUT as RENEWAL, archive as earlier
from run import read, save
from saturation_report import ci, weights

METHODS = ("wide", "repeated")


def training(market):
    base = historical("fit", market)
    single = [earlier((market, g, i)) for i in range(32) for g in GROUPS]
    grouped = [repeats((market, g, i), False)[0] for i in range(8) for g in GROUPS]
    repeated = [f for group in grouped for f in group]
    assert len(base) == 120 and len(single) == len(repeated) == 96
    assert {group[0] for group in grouped}.issubset(single)
    models = {}
    for key, paths in zip(METHODS, (single, repeated)):
        records = [read(f) for f in base + paths]
        x = [opportunity(r)[0] for r in records]
        y = [targets(r)[1] for r in records]
        models[key] = forest(x, y)
    return (
        models,
        grouped,
        {k: digest(base + p) for k, p in zip(METHODS, (single, repeated))},
    )


def candidates(values, enabled, hurdle):
    score, risk = estimates(values)
    action = score.argmax(axis=1)
    chosen_risk = risk[np.arange(len(risk)), action]
    eligible = enabled & (score.max(axis=1) > hurdle)
    return action + 1, chosen_risk, eligible


def sorted_eligible(risk, eligible):
    ids = np.flatnonzero(eligible)
    return ids[np.argsort(risk[ids], kind="stable")]


def freeze(models, market, hurdle):
    files = historical("calibrate", market)
    prior = [opportunity(read(p)) for p in files]
    x, enabled = map(np.asarray, zip(*prior))
    q = int(np.ceil(enabled.sum() / 4))
    cutoffs = {}
    for name, model in models.items():
        _, risk, e = candidates(model.predict(x), enabled, hurdle)
        ids = sorted_eligible(risk, e)
        take = min(q, len(ids))
        cutoffs[name] = float(risk[ids[take - 1]]) if take else -1.0
    return dict(
        cutoffs=cutoffs,
        calibration_digest=digest(files),
        historical_eligible=int(enabled.sum()),
    )


def predict(records, models, market, hurdle, cutoffs):
    x, enabled = map(np.asarray, zip(*(opportunity(r) for r in records)))
    raw = {name: model.predict(x) for name, model in models.items()}
    rows = [
        dict(
            seed=r["seed"],
            enabled=bool(flag),
            actions=dict(keep=0, cut=int(flag)),
            scores={},
        )
        for r, flag in zip(records, enabled)
    ]
    choices = [candidates(raw[key], enabled, hurdle) for key in METHODS]
    quota = min(int(np.ceil(enabled.sum() / 4)), *(int(a[2].sum()) for a in choices))
    reviewer = PublicRenewalReview(RENEWAL / f"public-{market}.npz")
    for r, src in zip(rows, records):
        result = reviewer.review(*public_proposal(src))
        r["actions"]["previous"] = ("keep", "cut", "raise").index(result["candidate"])
    for key, (actions, risk, eligible) in zip(METHODS, choices):
        chosen = (eligible & (risk <= cutoffs[key])) * actions
        selected = np.zeros(len(rows), dtype=bool)
        selected[sorted_eligible(risk, eligible)[:quota]] = True
        for i, row in enumerate(rows):
            row["actions"][key] = int(chosen[i])
            row["actions"][key + "_matched"] = int(actions[i] * selected[i])
            row["scores"][key] = raw[key][i].tolist()
    return rows


def evaluate(records, scores, hurdle):
    data = np.asarray([targets(r)[1] for r in records]).reshape(-1, 2, 4)
    assert [r["seed"] for r in records] == [s["seed"] for s in scores]
    rows = [
        dict(
            **r, gains=[0.0, *map(float, y[:, 0])], changed=[False, *map(bool, y[:, 1])]
        )
        for r, y in zip(scores, data)
    ]
    summary = {key: metrics(rows, key, hurdle) for key in scores[0]["actions"]}
    active = data[:, :, 1].astype(bool)
    differences = []
    for key in METHODS:
        _, risk = estimates([s["scores"][key] for s in scores])
        error = np.where(active, (risk - data[:, :, 2]) ** 2, 0.0)
        differences.append(error.sum(axis=1))
        summary[key]["risk_brier"] = float(error.sum() / active.sum())
    resamples = weights(len(rows))
    denom = resamples @ active.sum(axis=1)
    delta = (resamples @ (differences[1] - differences[0])) / denom

    def effect(a, b):
        return ci(
            [r["gains"][r["actions"][a]] - r["gains"][r["actions"][b]] for r in rows]
        )

    abstention = {}
    for key in (*METHODS, "previous"):
        eligible = [r for r in rows if r["enabled"] and any(r["changed"])]
        holds = [r for r in eligible if r["actions"][key] == 0]
        abstention[key] = dict(
            eligible=len(eligible),
            held=len(holds),
            correctly_held=sum(max(r["gains"]) <= 1e-7 for r in holds),
            missed_better_move=sum(max(r["gains"]) > 1e-7 for r in holds),
        )
    return rows, dict(
        policies=summary,
        selection=abstention,
        matched_gain_diff=effect("repeated_matched", "wide_matched"),
        frozen_gain_diff=effect("repeated", "wide"),
        brier_diff=[
            float((differences[1] - differences[0]).sum() / active.sum()),
            *np.quantile(delta, [0.0125, 0.9875]).tolist(),
        ],
    )


def variance(grouped):
    result = {}
    data = np.asarray([[targets(read(f))[1] for f in paths] for paths in grouped])
    for label, column in (("cut", 0), ("raise", 4)):
        y = data[data[:, 0, column + 1].astype(bool), :, column]
        n, k = y.shape
        mu = y.mean(axis=1)
        within = y.var(axis=1, ddof=1)
        # Variance of the observed per-market mean includes 1/k within-market variance.
        resamples = weights(n)
        noise = resamples @ within
        between = (resamples @ (mu**2) - (resamples @ mu) ** 2) * n / (
            n - 1
        ) - noise / k

        def share(var, heterogeneity):
            return var / (var + np.maximum(0.0, heterogeneity))

        signal = mu.var(ddof=1) - within.mean() / k
        others = (k * mu[:, None] - y) / (k - 1)
        err3 = np.mean((y - others) ** 2, axis=1)
        npairs = k * (k - 1)
        err1 = np.sum((y[:, :, None] - y[:, None, :]) ** 2, axis=(1, 2)) / npairs
        first_mae = np.sum(np.abs(y[:, :, None] - y[:, None, :]), axis=(1, 2)) / npairs
        three_mae = np.mean(np.abs(y - others), axis=1)
        result[label] = dict(
            n=n,
            noise_share=float(share(within.mean(), signal)),
            noise_share_interval=np.quantile(
                share(noise, between), [0.0125, 0.9875]
            ).tolist(),
            first_rmse=float(err1.mean() ** 0.5),
            three_rmse=float(err3.mean() ** 0.5),
            first_absolute_error=ci(first_mae),
            three_absolute_error=ci(three_mae),
            absolute_error_diff=ci(three_mae - first_mae),
            both_won_and_lost=int(
                ((y.max(axis=1) > 1e-7) & (y.min(axis=1) < -1e-7)).sum()
            ),
        )
    return result


def datasets(market):
    yield "smol-" + market, [new_path(t) for t in new_jobs("smol") if t[1] == market]
    if market == "econ":
        yield "qwen-econ", [new_path(t) for t in new_jobs("qwen")]
    yield "archived-" + market, historical("test", market)


def run(phase):
    config = {} if phase == "freeze" else read(OUT / "freeze.json")
    report = {}
    diagnostics = {}
    for m in MARKETS:
        models, groups, hashes = training(m)
        hurdle = 20000 if m == "native" else 5
        if phase == "freeze":
            config[m] = dict(**freeze(models, m, hurdle), training_hashes=hashes)
            continue
        frozen = config[m]
        assert hashes == frozen["training_hashes"]
        assert digest(historical("calibrate", m)) == frozen["calibration_digest"]
        if phase == "measure":
            diagnostics[m] = variance(groups)
        for name, files in datasets(m):
            records = [read(f) for f in files]
            assert len({r["seed"] for r in records}) == len(records)
            scores = predict(records, models, m, hurdle, frozen["cutoffs"])
            filename = OUT / f"{name}-predictions.json.gz"
            if phase == "predict":
                save(filename, scores)
            else:
                assert scores == read(filename)
                rows, report[name] = evaluate(records, scores, hurdle)
                save(OUT / f"{name}-decisions.json.gz", rows)
    if phase == "freeze":
        save(OUT / "freeze.json", config)
    if phase == "measure":
        save(OUT / "results.json", dict(cohorts=report, within_market=diagnostics))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("phase", choices=("freeze", "predict", "measure"))
    run(parser.parse_args().phase)
