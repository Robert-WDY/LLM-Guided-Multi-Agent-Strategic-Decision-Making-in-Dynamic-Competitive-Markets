"""Intervention-specific tail verification. These cases were selected after observing losses."""

from concurrent.futures import ProcessPoolExecutor
import numpy as np
from prices import OUT, RAW
from run import REVIEW, design, jobs, read, save, simulation
from saturation_report import decompose, discounted


def check(task):
    m, k, bad = task
    s, j = bad["seed"], bad["seat"]
    rec = read(RAW / f"{s}.json.gz")
    g = rec["group"]
    i = next(
        i for market, group, i in jobs() if market == m and group == g and design(m, g, i)[0] == s
    )
    base = rec["paths"]["keep"]
    target = {d["t"]: d["requested"] for d in base["decisions"]}
    r = simulation(m, g, i, offline=rec["tape"], diagnostic=(k, j, target), allow_new=True)
    raw = r["paths"]
    assert raw["natural"]["rows"] == rec["paths"][f"inf{k}"]["rows"]
    r["paths"]["keep"] = base
    natural = np.asarray(raw["natural"]["rows"])
    fixed = np.asarray(raw["fixed_rivals"]["rows"])
    peers = [i for i in range(4) if i != j]
    mismatches = int(
        (np.abs(fixed[:, peers, 0] - np.array(base["rows"])[:, peers, 0]) > 1.0e-8).sum()
    )
    contributions = {}
    for t in REVIEW:
        skip = np.asarray(raw[f"omit-{t}"]["rows"])
        assert np.array_equal(skip[: t - 1], natural[: t - 1])
        contributions[t] = float(discounted(natural[:, :, 2] - skip[:, :, 2])[j])
    save(OUT / f"tail-replay-{m}-{k}.json.gz", r)
    return dict(
        seed=s,
        seat=j,
        realized=bad["gain"],
        fixed_rivals=decompose(r, "fixed_rivals", "keep", j, 19)["gain"],
        rival_price_mismatches=mismatches,
        one_at_a_time=contributions,
    )


if __name__ == "__main__":
    tail = read(OUT / "tail.json")
    tasks = [
        (m, k, tail[f"{m}-rolling-{k}"][0])
        for m in ("econ", "native", "appointments")
        for k in (1, 2, 4)
    ]
    with ProcessPoolExecutor(6) as pool:
        data = dict(zip([f"{m}-{k}" for m, k, _ in tasks], pool.map(check, tasks)))
    save(OUT / "tail-replay.json", data)
