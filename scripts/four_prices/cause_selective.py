"""Public-only estimate of loss frequency trained on both actions, before independent testing."""

import numpy as np
from cause_trials import OUT, MARKETS
from cause_value import dataset, estimator, metrics
from run import save, read
from saturation_report import ci


def train(rows):
    models = []
    for a in (1, 2):
        eligible = [r for r in rows if r["changed"][a]]
        models.append(
            estimator().fit(
                [r["state"] for r in eligible],
                [int(r["gains"][a] < -1e-7) for r in eligible],
            )
        )
    return models


def apply(rows, loss_models, value_model, m):
    X = np.asarray([r["state"] for r in rows])
    means = value_model.predict(X)
    prob = np.column_stack([f.predict(X) for f in loss_models])
    threshold = 20000 if m == "native" else 5
    allowed = (means > threshold) & (prob <= 0.1)
    arm = np.column_stack([np.zeros(len(X)), np.where(allowed, means, -np.inf)]).argmax(
        1
    )
    audit = []
    for r, k, pred in zip(rows, arm, prob):
        choice = int(k) if r["enabled"] else 0
        audit.append(
            dict(
                seed=r["seed"],
                gains=r["gains"],
                changed=r["changed"],
                actions={"selective": choice},
                predicted_loss=pred.tolist(),
            )
        )
    return audit


if __name__ == "__main__":
    import sys

    phase = sys.argv[1]
    assert phase in ("calibrate", "test")
    out = {}
    for m in MARKETS:
        r = dataset("fit", m)
        losses = train(r)
        value = estimator().fit(
            [row["state"] for row in r], [row["gains"][1:] for row in r]
        )
        a = apply(dataset(phase, m), losses, value, m)
        key = "selective"
        res = metrics(a, key, 20000 if m == "native" else 5)
        old = read(OUT / f"{phase}-{m}-decisions.json.gz")
        assert [v["seed"] for v in old] == [v["seed"] for v in a]
        res["vs_direct"] = ci(
            [
                r["gains"][r["actions"][key]] - o["gains"][o["actions"]["direct"]]
                for r, o in zip(a, old)
            ]
        )
        out[m] = res
        save(OUT / f"selective-{phase}-{m}.json.gz", a)
    save(OUT / f"selective-{phase}.json", out)
