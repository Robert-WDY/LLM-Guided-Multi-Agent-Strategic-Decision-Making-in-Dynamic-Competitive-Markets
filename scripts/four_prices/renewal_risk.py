"""Estimate renewal trajectories and terminal losses using settled public prices."""

import numpy as np
from sklearn.compose import TransformedTargetRegressor
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler

from cause_value import estimator, public_panel
from game_theory_agent.game_theory.quote_panel import public_state, rivals

METHODS = ("renewal", "equal_budget", "one_stage", "known_renewal")
CUTOFFS = (0.05, 0.10, 0.15, 0.20, 0.30, 0.40, 1.0)


def decision_input(history, seat, t, draft):
    state = public_state(history, seat, t, draft)
    enabled = bool(np.isfinite(history[-1, seat]) and rivals(history, seat))
    return state, enabled


def opportunity(record):
    t, j = record["probe"], record["masks"]["1"][0]
    keep = record["paths"]["keep"]
    rows = np.asarray(keep["rows"][: t - 1])
    draft = next(d["draft"][j] for d in keep["decisions"] if d["t"] == t)
    return decision_input(public_panel(rows), j, t, draft)


def targets(record):
    """Future branch data: labels for fitting, diagnostics and settlement only."""
    t, j = record["probe"], record["masks"]["1"][0]

    def branch(name):
        path = record["paths"][name]
        rows = np.asarray(path["rows"])
        assert rows.shape == (60, 4, 6)
        requests = {d["t"]: d["requested"][j] for d in path["decisions"]}
        return rows, requests

    base, base_requests = branch("keep")
    trajectory, comparable, economic = [], [], []
    discount = 0.99 ** np.arange(61 - t)
    for name in ("audit_cut", "audit_raise"):
        arm, requests = branch(name)
        assert np.array_equal(arm[: t - 1], base[: t - 1])
        gain = float(discount @ (arm[t - 1 :, j, 2] - base[t - 1 :, j, 2]))
        changed = np.any(arm[t - 1 : t + 3, j, 0] != base[t - 1 : t + 3, j, 0])
        assert changed or abs(gain) < 1e-7
        economic.extend((gain, float(changed), float(gain < -1e-7), max(-gain, 0.0)))
        for renewal in (t + 4, t + 8):
            active, old_active = arm[renewal - 1, j, 4], base[renewal - 1, j, 4]
            both = bool(active and old_active)
            move = np.log(requests[renewal] / base_requests[renewal]) if both else 0.0
            trajectory.extend((move, active - old_active))
            comparable.append(both)
    return trajectory, economic, comparable


def forest(x, y):
    # Equal output scales keep a large currency unit from dominating the splits.
    return TransformedTargetRegressor(
        regressor=estimator(), transformer=StandardScaler()
    ).fit(x, y)


class RenewalRisk:
    def __init__(self, x, path, value):
        self.aux, self.final = {}, {}
        self.mean_path = path.mean(axis=0)
        folds = list(KFold(4, shuffle=True, random_state=31673).split(x))
        for name, target in (("renewal", path), ("equal_budget", value)):
            oof = np.zeros_like(target)
            for f, h in folds:
                oof[h] = forest(x[f], target[f]).predict(x[h])
            self.aux[name] = forest(x, target)
            self.final[name] = forest(np.column_stack((x, oof)), value)
        self.final["one_stage"] = forest(x, value)
        self.final["known_renewal"] = forest(np.column_stack((x, path)), value)

    def predict(self, x):
        values = {}
        aux = {name: model.predict(x) for name, model in self.aux.items()}
        for name, forecast in aux.items():
            values[name] = self.final[name].predict(np.column_stack((x, forecast)))
        values["one_stage"] = self.final["one_stage"].predict(x)
        return values, aux["renewal"]

    def diagnostic(self, x, actual_path):
        return self.final["known_renewal"].predict(np.column_stack((x, actual_path)))
