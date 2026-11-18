"""Export frozen in-domain models and verify prediction parity using training inputs only."""

import numpy as np
from cause_trials import OUT, MARKETS
from cause_value import fit, dataset
from game_theory_agent.game_theory.panel_advisor import PanelAdvisor


def pack(name, regressor):
    buffers = {k: [] for k in ("feature", "threshold", "left", "right", "value")}
    roots = []
    offset = 0
    for estimator in regressor.estimators_:
        tree = estimator.tree_
        roots.append(offset)
        for k, x in (
            ("feature", tree.feature),
            ("threshold", tree.threshold),
            ("left", tree.children_left),
            ("right", tree.children_right),
            ("value", tree.value[:, :, 0]),
        ):
            if k in ("left", "right"):
                x = np.where(x < 0, -1, x + offset)
            buffers[k].append(x)
        offset += tree.node_count
    return {
        name + "_roots": np.asarray(roots),
        **{name + "_" + k: np.concatenate(items) for k, items in buffers.items()},
    }


if __name__ == "__main__":
    for m in MARKETS:
        rows = dataset("fit", m)
        models = fit(rows)
        data = {}
        for key, model in zip(("quotes", "direct", "value"), models):
            data.update(pack(key, model))
        f = OUT / ("panel-" + m + ".npz")
        np.savez_compressed(f, **data)
        reader = PanelAdvisor(f)
        from game_theory_agent.game_theory.quote_panel import seller_features

        for row in rows:
            for attr, reference, x in zip(
                ("quotes", "direct", "value"),
                models,
                (
                    seller_features(row["h"], row["seat"], row["t"])
                    if row["enabled"]
                    else [0.0] * 10,
                    row["state"],
                    row["state"] + row["actual"],
                ),
            ):
                np.testing.assert_allclose(
                    getattr(reader, attr).predict(x),
                    reference.predict([x])[0],
                    rtol=1.0e-11,
                    atol=1.0e-6,
                )
