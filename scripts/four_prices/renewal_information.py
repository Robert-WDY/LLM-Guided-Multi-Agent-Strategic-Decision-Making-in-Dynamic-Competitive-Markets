"""Value of disclosing the manager's settled private sales and profit, offline only."""

import argparse

import numpy as np
from cause_trials import MARKETS
from game_theory_agent.game_theory.renewal_review import decisions
from renewal_risk import CUTOFFS, RenewalRisk
from renewal_risk_study import (
    arrays,
    inputs,
    labels,
    evaluate,
    historical,
    trajectory_errors,
)
from renewal_trials import OUT, archive, tasks
from run import read, save
from saturation_report import ci


def private_inputs(records):
    x, enabled = inputs(records)
    ledger = []
    for r in records:
        j, t = r["masks"]["1"][0], r["probe"]
        # Two completed four-round contracts, oldest first.
        rows = np.asarray(r["paths"]["keep"]["rows"][t - 9 : t - 1])
        assert rows.shape == (8, 4, 6)
        v = rows[:, j, 1:3].reshape(2, 4, 2).sum(axis=1)
        ledger.append(np.arcsinh(v).ravel())
    return np.column_stack((x, ledger)), enabled


def fit_private(m):
    records = [read(p) for phase in ("fit", "calibrate") for p in historical(phase, m)]
    p, v, _ = arrays(records)
    x, _ = private_inputs(records)
    return RenewalRisk(x, p, v)


def process(phase):
    lockfile = OUT / "private-freeze.json"
    config = {} if phase == "freeze" else read(lockfile)
    summary = {}
    for m in MARKETS:
        hurdle = 20000 if m == "native" else 5
        model = fit_private(m)
        files = (
            historical("test", m)
            if phase == "freeze"
            else [archive(t) for t in tasks() if t[0] == m]
        )
        records = [read(p) for p in files]
        y, v, masks = arrays(records)
        x, enabled = private_inputs(records)
        rows = labels(records, v)
        vals, pred = model.predict(x)
        if phase == "freeze":
            grid = []
            for cutoff in CUTOFFS:
                a = decisions(vals["renewal"], enabled, cutoff, hurdle)
                grid.append(dict(cutoff=cutoff, **evaluate(rows, "private", a, hurdle)))
            eligible = [g for g in grid if g["coverage"] >= 0.10]
            config[m] = (
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
                else grid[-1]
            )
        else:
            public = read(OUT / f"predictions-{m}.json.gz")
            assert [r["seed"] for r in rows] == [r["seed"] for r in public]
            a = decisions(vals["renewal"], enabled, config[m]["cutoff"], hurdle)
            result = evaluate(rows, "private", a, hurdle)
            result["extra_over_public"] = ci(
                [
                    r["gains"][k] - r["gains"][s["choices"]["renewal"]]
                    for k, r, s in zip(a, rows, public)
                ]
            )
            result["trajectory"] = trajectory_errors(y, pred, model.mean_path, masks)
            result["quote_error_minus_public"] = {}
            for n in range(4):
                mask = masks[:, n]
                idx = 2 * n
                before = np.array([r["path"][idx] for r in public])
                diff = np.abs(pred[mask, idx] - y[mask, idx]) - np.abs(
                    before[mask] - y[mask, idx]
                )
                result["quote_error_minus_public"][str(n)] = ci(diff)
            summary[m] = result
    save(
        lockfile if phase == "freeze" else OUT / "private-confirmation.json",
        config if phase == "freeze" else summary,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("phase", choices=("freeze", "confirm"))
    process(parser.parse_args().phase)
