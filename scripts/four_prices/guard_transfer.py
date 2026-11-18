"""Change only the manager model on the locked feasible-price scenarios."""

from concurrent.futures import ProcessPoolExecutor
from functools import partial
import json
from guard_study import ARMS, OUTPUT, design, guard, jobs, path
from legal_quote import feasible_quote
from guard_report import one, summarize
from guard_verify import verify_raw
from run import read, save, simulation
from saturation_report import ci


def paired(task):
    _, m, g, i = task
    ref = read(path(task))
    f = OUTPUT / "qwen" / (str(ref["seed"]) + ".json.gz")
    f.parent.mkdir(exist_ok=True)

    def simulate(tape):
        return simulation(
            m,
            g,
            i,
            offline=tape,
            numeric_quote=partial(
                feasible_quote, model="qwen", port=19944, disclose_units=False
            ),
            study_design=partial(design, "legal"),
            price_guard=guard(),
            run_only=ARMS,
        )

    if not f.exists():
        save(f, simulate(None))
    r = read(f)
    assert (
        r["roles"] == ref["roles"]
        and r["masks"] == ref["masks"]
        and r["seed"] == ref["seed"]
    )
    verify_raw(r, "legal")
    assert all(x["body"]["model"] == "qwen" for x in r["tape"].values())
    if g == "unseen" and i == 0:
        assert json.loads(json.dumps(simulate(r["tape"]))) == r
    return m, one(r, "legal"), one(ref, "legal")


if __name__ == "__main__":
    rows = {"econ": [], "native": []}
    with ProcessPoolExecutor(6) as pool:
        for m, new, old in pool.map(paired, [t for t in jobs() if t[0] == "legal"]):
            rows[m].append((new, old))
    out = {}
    for m, pairs in rows.items():
        out[m] = summarize([a for a, b in pairs])
        out[m]["model_difference"] = {
            k: ci(
                [
                    a["install"][k]["safe_inf"]["gain"]
                    - b["install"][k]["safe_inf"]["gain"]
                    for a, b in pairs
                ]
            )
            for k in (1, 2, 4)
        }
    save(OUTPUT / "transfer.json", out)
