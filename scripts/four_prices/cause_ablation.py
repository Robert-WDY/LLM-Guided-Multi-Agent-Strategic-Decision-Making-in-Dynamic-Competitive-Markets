"""Equal-architecture ablation of action coverage and selection of labelled training examples."""

import numpy as np
from cause_trials import OUT, MARKETS
from cause_value import dataset, estimator, read, save, metrics
from independent import frozen_actions
from saturation_report import ci


def fit_values(rows, selected):
    X = np.asarray([r["state"] for r in rows])
    y = np.array([r["gains"] for r in rows])
    out = {}
    sizes = {}
    for name in ("full", "selected", "random"):
        models = []
        for arm in (1, 2):
            ix = np.where(selected == arm)[0]
            sizes[str(arm)] = len(ix)
            if name == "full":
                ix = np.arange(len(rows))
            elif name == "random":
                ix = np.random.default_rng(211042 + arm).choice(
                    len(rows), len(ix), replace=False
                )
            models.append(estimator().fit(X[ix], y[ix, arm]) if len(ix) else None)
        out[name] = models
    return out, sizes


if __name__ == "__main__":
    out = {}
    for m in MARKETS:
        train = dataset("fit", m)
        test = dataset("test", m)
        selected = np.array(
            [r["inferred"] for r in frozen_actions(m, [r["legacy"] for r in train])]
        )
        models, counts = fit_values(train, selected)
        old = read(OUT / f"test-{m}-decisions.json.gz")
        X = np.array([r["state"] for r in test])
        for name, arms in models.items():
            predictions = [
                a.predict(X) if a is not None else np.zeros(len(test)) for a in arms
            ]
            policy = np.column_stack([np.zeros(len(test)), *predictions]).argmax(1)
            for r, d, arm in zip(test, old, policy):
                assert r["seed"] == d["seed"]
                d["actions"][name] = int(arm) if r["enabled"] else 0
        for r in old:
            r["actions"].update(keep=0, cut=1, raise_=2)

        def gain(r, k):
            return r["gains"][r["actions"][k]]

        keys = [
            "keep",
            "cut",
            "raise_",
            "full",
            "selected",
            "random",
            "forecast",
            "direct",
        ]
        out[m] = dict(
            fit_selected=counts,
            policies={k: metrics(old, k, 20000 if m == "native" else 5) for k in keys},
            choices={
                k: [sum(r["actions"][k] == a for r in old) for a in (0, 1, 2)]
                for k in keys
            },
            contrast={
                a + "-" + b: ci([gain(r, a) - gain(r, b) for r in old])
                for a, b in [
                    ("full", "selected"),
                    ("random", "selected"),
                    ("full", "random"),
                    ("forecast", "cut"),
                    ("direct", "cut"),
                ]
            },
        )
    save(OUT / "ablation.json", out)
