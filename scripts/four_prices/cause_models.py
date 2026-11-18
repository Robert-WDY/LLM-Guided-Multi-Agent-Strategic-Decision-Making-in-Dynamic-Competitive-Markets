"""Cross the focal pricing model with the three rival pricing models, holding the world fixed."""

from concurrent.futures import ProcessPoolExecutor
from functools import partial
import json
import numpy as np
from cause_trials import OUT
from guard_study import OUTPUT as PRIOR, design, jobs, path
from legal_quote import feasible_quote
from run import simulation, read, save
from saturation_report import ci, discounted


def mixed(task, focal_model, tape, replay):
    _, m, g, i = task
    seat = design("legal", m, g, i)[1][1][0]

    def submit(m, role, h, reference, low, high, seed, j, t, cache, offline):
        own = (j == seat) == (focal_model == "qwen")
        return feasible_quote(
            m,
            role,
            h,
            reference,
            low,
            high,
            seed,
            j,
            t,
            cache,
            replay,
            model="qwen" if own else "smol",
            port=19944 if own else 19943,
        )

    return simulation(
        m,
        g,
        i,
        offline=tape,
        study_design=partial(design, "legal"),
        numeric_quote=submit,
        run_only=["keep"],
    )


def metric(r, seat):
    a = np.array(r["paths"]["keep"]["rows"])
    return dict(
        profit=float(discounted(a[:, :, 2], 1)[seat]),
        survived=float(a[-1, seat, 5]),
        volume=float(a[:, seat, 1].sum()),
    )


def pair(task):
    _, m, g, i = task
    s = read(path(task))
    q = read(PRIOR / "qwen" / f"{s['seed']}.json.gz")
    seat = s["masks"]["1"][0]
    rows = {"SS": metric(s, seat), "QQ": metric(q, seat)}
    tape = {**s["tape"], **q["tape"]}
    start = len(tape)
    for focal, code in [("smol", "SQ"), ("qwen", "QS")]:
        f = OUT / "cross" / f"{s['seed']}-{code}.json.gz"
        f.parent.mkdir(exist_ok=True, parents=True)
        if not f.exists():
            r = mixed(task, focal, tape, False)
            used = {
                key
                for d in r["paths"]["keep"]["decisions"]
                for key in d["model_actions"]
                if key
            }
            r["tape"] = {k: r["tape"][k] for k in sorted(used)}
            save(f, r)
        r = read(f)
        for d in r["paths"]["keep"]["decisions"]:
            for j, tag in enumerate(d["model_actions"]):
                if tag is not None:
                    assert r["tape"][tag]["body"]["model"] == (
                        "qwen" if code[int(j != seat)] == "Q" else "smol"
                    )
        assert json.loads(json.dumps(mixed(task, focal, r["tape"], True))) == r
        rows[code] = metric(r, seat)
        tape.update(r["tape"])
    return m, rows, len(tape) - start


if __name__ == "__main__":
    by = {"econ": [], "native": []}
    calls = 0
    with ProcessPoolExecutor(6) as pool:
        for m, row, n in pool.map(pair, [t for t in jobs() if t[0] == "legal"]):
            by[m].append(row)
            calls += n
    out = {}
    for m, rows in by.items():
        out[m] = {}
        for k in ("survived", "profit", "volume"):
            out[m][k] = {key: ci([r[key][k] for r in rows]) for key in rows[0]}
            for a, b in [("QS", "SS"), ("QQ", "SQ"), ("SQ", "SS"), ("QQ", "QS")]:
                out[m][k][a + "-" + b] = ci([r[a][k] - r[b][k] for r in rows])
    save(OUT / "model-cross.json", dict(markets=out, extra_requests=calls))
