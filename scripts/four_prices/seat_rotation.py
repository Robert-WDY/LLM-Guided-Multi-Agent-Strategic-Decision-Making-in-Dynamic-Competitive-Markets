"""Exhaustive assignments reusing exactly the same market, firm identities and shocks."""

from concurrent.futures import ProcessPoolExecutor
import time
import numpy as np
from prices import MARKETS, GROUPS, OUT, RAW
from run import PAIRS, design, read, save, simulation
from saturation_report import discounted


def job(task):
    m, g, i = task
    s = design(m, g, i)[0]
    path = RAW / f"{s}.json.gz"
    while not path.exists():
        time.sleep(2)
    base = read(path)
    tape = dict(base["tape"])
    path = OUT / f"rotation-{s}.json.gz"
    if path.exists():
        return read(path)["summary"]
    data = {}
    stats = {}
    for a, pair in enumerate(PAIRS):
        masks = {1: (a % 4,), 2: pair, 4: (0, 1, 2, 3)}
        modes = ["inf2", "dir2"] + (["inf1", "dir1"] if a < 4 else [])
        trial = simulation(m, g, i, offline=tape, allow_new=True, allocation=masks, run_only=modes)
        tape = trial["tape"]
        for mode, p in trial["paths"].items():
            installed = masks[int(mode[-1])]
            key = mode + "-" + "".join(map(str, installed))
            rows = np.asarray(p["rows"])
            keep = np.array(base["paths"]["keep"]["rows"])
            gain = discounted(rows[:, :, 2] - keep[:, :, 2])
            data[key] = p
            stats[key] = float(gain[list(installed)].mean())
            if set(installed) == set(base["masks"][mode[-1]]):
                assert p["rows"] == base["paths"][mode]["rows"]
    result = dict(market=m, group=g, seed=s, live=base["live"], effects=stats)
    save(
        path,
        dict(
            paths=data,
            extra_tape={k: v for k, v in tape.items() if k not in base["tape"]},
            summary=result,
        ),
    )
    return result


if __name__ == "__main__":
    jobs = [
        (m, g, i) for m in MARKETS for g in GROUPS for i in (0, 23 if m == "appointments" else 47)
    ]
    with ProcessPoolExecutor(4) as pool:
        rows = list(pool.map(job, jobs))
    save(OUT / "seat-rotation.json", rows)
