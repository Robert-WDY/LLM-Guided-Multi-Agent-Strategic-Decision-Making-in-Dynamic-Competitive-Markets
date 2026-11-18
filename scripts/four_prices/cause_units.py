"""Prospective one-sentence units ablation for direct native numeric quotes."""

from concurrent.futures import ProcessPoolExecutor
from functools import partial
import json
import numpy as np
from cause_trials import OUT
from prices import GROUPS, number
from guard_study import design
from run import read, save, simulation
from legal_quote import feasible_quote
from saturation_report import ci, discounted


def profile(m, g, i):
    _, masks, focal, _, _, _ = design("legal", m, g, i % 4)
    seed = 231000 + 8 * GROUPS.index(g) + i
    roles = (
        [0] * 4
        if g == "balanced"
        else [4] * 4
        if g == "unseen"
        else [number("unit-role", seed, j) % 5 for j in range(4)]
    )
    return seed, masks, focal, roles, 27, True


def trial(task):
    model, g, i = task
    data = {}
    for label in ("old", "labelled"):
        seed = profile("native", g, i)[0]
        f = OUT / "units" / f"{model}-{label}-{seed}.json.gz"
        f.parent.mkdir(exist_ok=True, parents=True)

        def play(tape):
            return simulation(
                "native",
                g,
                i,
                run_only=["keep"],
                study_design=profile,
                offline=tape,
                numeric_quote=partial(
                    feasible_quote,
                    model=model,
                    port=19943 if model == "smol" else 19944,
                    disclose_units=label == "labelled",
                ),
            )

        if not f.exists():
            save(f, play(None))
        r = read(f)
        assert all(
            ("100 cents = 1 yuan" in v["body"]["messages"][1]["content"])
            == (label == "labelled")
            for v in r["tape"].values()
        )
        if i == 0:
            assert json.loads(json.dumps(play(r["tape"]))) == r
        a = np.asarray(r["paths"]["keep"]["rows"])
        data[label] = dict(
            profit=float(discounted(a[:, :, 2], 1).mean()),
            survivors=int(a[-1, :, 5].sum()),
            requests=len(r["tape"]),
        )
    return model, data


if __name__ == "__main__":
    groups = {"smol": [], "qwen": []}
    with ProcessPoolExecutor(8) as pool:
        for m, r in pool.map(
            trial, [(m, g, i) for i in range(8) for g in GROUPS for m in groups]
        ):
            groups[m].append(r)
    out = {}
    for m, rows in groups.items():
        out[m] = {
            k: dict(
                old=ci([r["old"][k] for r in rows]),
                labelled=ci([r["labelled"][k] for r in rows]),
                difference=ci([r["labelled"][k] - r["old"][k] for r in rows]),
            )
            for k in ("profit", "survivors")
        }
        out[m]["requests"] = sum(v["requests"] for r in rows for v in r.values())
    save(OUT / "units.json", out)
