"""Freeze with historical markets, score new public prefixes, then open paired outcomes."""

import argparse
from hashlib import sha256

import numpy as np
from cause_trials import MARKETS, archive as old_archive, jobs
from cause_value import metrics
from prices import GROUPS
from game_theory_agent.game_theory.renewal_review import decisions, estimates
from renewal_risk import (
    CUTOFFS,
    METHODS,
    RenewalRisk,
    opportunity,
    targets,
)
from renewal_trials import OUT, archive, tasks
from run import read, save
from saturation_report import ci


def historical(phase, market):
    return [old_archive(task) for task in jobs(phase) if task[1] == market]


def digest(paths):
    h = sha256()
    for p in paths:
        h.update(p.name.encode())
        h.update(sha256(p.read_bytes()).digest())
    return h.hexdigest()


def arrays(records):
    paths, outcomes, masks = zip(*(targets(r) for r in records))
    return np.asarray(paths), np.asarray(outcomes), np.asarray(masks)


def inputs(records):
    x, enabled = zip(*(opportunity(r) for r in records))
    return np.asarray(x), np.asarray(enabled)


def fit(market):
    files = historical("fit", market) + historical("calibrate", market)
    records = [read(p) for p in files]
    assert len(set(r["seed"] for r in records)) == len(records) == 216
    x, _ = inputs(records)
    path, value, _ = arrays(records)
    return RenewalRisk(x, path, value)


def labels(records, value):
    by_arm = value.reshape(-1, 2, 4)
    return [
        dict(
            seed=r["seed"],
            group=r["group"],
            gains=[0.0, *map(float, v[:, 0])],
            changed=[False, *map(bool, v[:, 1])],
            actions={},
        )
        for r, v in zip(records, by_arm)
    ]


def evaluate(rows, name, choices, hurdle):
    for r, a in zip(rows, choices, strict=True):
        r["actions"][name] = int(a)
    return metrics(rows, name, hurdle)


def freeze(market, model, hurdle):
    files = historical("test", market)
    records = [read(p) for p in files]
    x, enabled = inputs(records)
    path, value, _ = arrays(records)
    values, _ = model.predict(x)
    values["known_renewal"] = model.diagnostic(x, path)
    rows = labels(records, value)
    result = {}
    for name, pred in values.items():
        candidates = []
        for c in CUTOFFS:
            acts = decisions(pred, enabled, c, hurdle)
            r = evaluate(rows, name, acts, hurdle)
            candidates.append(dict(cutoff=c, **r))
        eligible = [r for r in candidates if r["coverage"] >= 0.10]
        chosen = (
            min(
                eligible,
                key=lambda r: (
                    not r["passed"],
                    r["risk_upper"],
                    -r["effect"][0],
                    r["cutoff"],
                ),
            )
            if eligible
            else candidates[-1]
        )
        result[name] = dict(chosen=chosen, grid=candidates)
    return result


def audit(rows, name, hurdle):
    def gain(r):
        return r["gains"][r["actions"][name]]

    kept = [r for r in rows if r["actions"][name] == 0]
    # Explicit comparator for each rejected opportunity: what a fixed cut would earn.
    omitted = [r["gains"][1] for r in kept]
    return dict(
        **metrics(rows, name, hurdle),
        rejected=len(kept),
        rejected_cut_wins=sum(g > 1e-7 for g in omitted),
        rejected_cut_losses=sum(g < -1e-7 for g in omitted),
        rejected_cut_gains=sum(max(g, 0) for g in omitted),
        rejected_cut_downside=sum(max(-g, 0) for g in omitted),
        paired_vs_cut=ci(
            [gain(r) - r["gains"][r["actions"]["fixed_cut"]] for r in rows]
        ),
        paired_vs_equal_budget=ci(
            [gain(r) - r["gains"][r["actions"]["equal_budget"]] for r in rows]
        ),
        by_personality={
            g: metrics([r for r in rows if r["group"] == g], name, hurdle)
            for g in GROUPS
        },
    )


def trajectory_errors(actual, pred, mean_path, mask):
    out = {}
    for slot in range(4):
        j = 2 * slot
        valid = mask[:, slot]
        delta = actual[valid, j]
        err = abs(pred[valid, j] - delta)
        status = actual[:, j + 1]
        out[f"{('cut', 'raise')[slot // 2]}_renewal_{1 + slot % 2}"] = dict(
            continuing=int(valid.sum()),
            nonzero_price_effects=int((abs(delta) > 1e-9).sum()),
            price_forecast_mae=ci(err),
            price_no_change_mae=ci(abs(delta)),
            price_historical_mean_mae=ci(abs(delta - mean_path[j])),
            paired_error_vs_zero=ci(err - abs(delta)),
            paired_error_vs_mean=ci(err - abs(delta - mean_path[j])),
            availability_rmse=float(np.mean((pred[:, j + 1] - status) ** 2) ** 0.5),
            availability_no_change_rmse=float(np.mean(status**2) ** 0.5),
        )
    return out


def snapshot(records, model, calibration, hurdle):
    x, enabled = inputs(records)
    vals, paths = model.predict(x)
    result = [
        dict(seed=r["seed"], enabled=bool(e), path=p.tolist(), choices={}, scores={})
        for r, e, p in zip(records, enabled, paths)
    ]
    for name, v in vals.items():
        choices = decisions(v, enabled, calibration[name]["chosen"]["cutoff"], hurdle)
        for r, a, scores in zip(result, choices, v):
            r["choices"][name] = int(a)
            r["scores"][name] = scores.tolist()
    return result


def confirm(records, model, calibration, snap, hurdle):
    x, enabled = inputs(records)
    assert snap == snapshot(records, model, calibration, hurdle)
    path, value, mask = arrays(records)
    rows = labels(records, value)
    for r, s in zip(rows, snap):
        r["actions"] = dict(s["choices"], keep=0, fixed_cut=int(s["enabled"]))
    known = model.diagnostic(x, path)
    evaluate(
        rows,
        "known_renewal",
        decisions(
            known, enabled, calibration["known_renewal"]["chosen"]["cutoff"], hurdle
        ),
        hurdle,
    )
    reports = {
        name: audit(rows, name, hurdle) for name in ("keep", "fixed_cut", *METHODS)
    }
    brier = {}
    changed = value.reshape(-1, 2, 4)[:, :, 1].astype(bool)
    lost = value.reshape(-1, 2, 4)[:, :, 2]
    predictions = {k: [r["scores"][k] for r in snap] for k in snap[0]["scores"]}
    predictions["known_renewal"] = known
    for name, v in predictions.items():
        _, risk = estimates(v)
        brier[name] = float(np.mean((risk[changed] - lost[changed]) ** 2))
    return rows, dict(
        policies=reports,
        risk_brier_changed_arms=brier,
        trajectory=trajectory_errors(
            path, np.array([s["path"] for s in snap]), model.mean_path, mask
        ),
        incremental_changed_choices=sum(
            r["actions"]["renewal"] != r["actions"]["equal_budget"] for r in rows
        ),
    )


def run(phase):
    lock_file = OUT / "freeze.json"
    lock = {} if phase == "freeze" else read(lock_file)
    analyses = {}
    for market in MARKETS:
        hurdle = 20000 if market == "native" else 5
        model = fit(market)
        fingerprint = digest(
            [
                p
                for part in ("fit", "calibrate", "test")
                for p in historical(part, market)
            ]
        )
        if phase == "freeze":
            lock[market] = dict(
                historical_digest=fingerprint, models=freeze(market, model, hurdle)
            )
            continue
        assert fingerprint == lock[market]["historical_digest"]
        records = [read(archive(t)) for t in tasks() if t[0] == market]
        assert len({r["seed"] for r in records}) == len(records) == 120
        calibration = lock[market]["models"]
        filename = OUT / f"predictions-{market}.json.gz"
        if phase == "predict":
            save(filename, snapshot(records, model, calibration, hurdle))
        else:
            rows, analyses[market] = confirm(
                records, model, calibration, read(filename), hurdle
            )
            save(OUT / f"decisions-{market}.json.gz", rows)
    if phase == "freeze":
        save(lock_file, lock)
    if phase == "confirm":
        save(OUT / "confirmation.json", analyses)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("phase", choices=("freeze", "predict", "confirm"))
    run(parser.parse_args().phase)
