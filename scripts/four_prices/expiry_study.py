"""New matched worlds for preserving the manager's pre-advice anchor on renewal."""

from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
import json
from guard_study import ARMS, OUTPUT, design, guard
from guard_report import one, summarize
from guard_verify import verify_raw
from prices import GROUPS, number
from run import read, save, simulation
from saturation_report import ci


def profile(m, g, i):
    _, masks, _, _, _, _ = design("menu", m, g, i)
    seed = (209000 if m == "native" else 208000) + 12 * GROUPS.index(g) + i
    roles = (
        [0] * 4
        if g == "balanced"
        else [4] * 4
        if g == "unseen"
        else [number("repair-role", seed, j) % 5 for j in range(4)]
    )
    return (
        seed,
        masks,
        {k: v[number("audit-seat", seed, k) % k] for k, v in masks.items()},
        roles,
        23 + 4 * (number(seed, "audit-time") % 6),
        True,
    )


def run(task, tape, expire, online):
    return simulation(
        *task,
        offline=tape,
        allow_new=online,
        run_only=ARMS,
        price_guard=guard(),
        study_design=profile,
        expire_advice=expire,
    )


def pair(task):
    m, g, i = task
    seed = profile(*task)[0]
    tape = None
    effects = []
    for expire in (False, True):
        folder = OUTPUT / ("expiry" if expire else "expiry-control")
        folder.mkdir(exist_ok=True)
        f = folder / f"{seed}.json.gz"
        if not f.exists():
            save(f, run(task, tape, expire, True))
        r = read(f)
        assert r["seed"] == seed and verify_raw(r, "menu") == len(ARMS)
        if tape is not None:
            assert all(r["tape"][k] == v for k, v in tape.items())
        tape = r["tape"]
        if i == 0 and g == "unseen":
            assert json.loads(json.dumps(run(task, tape, expire, False))) == r
        effects.append(one(r, "menu"))
        if not expire:
            base = r["paths"]["keep"]["rows"]
        else:
            assert r["paths"]["keep"]["rows"] == base
    return m, effects, len(tape)


if __name__ == "__main__":
    tasks = [(m, g, i) for i in range(12) for g in GROUPS for m in ("econ", "native")]
    rows = defaultdict(list)
    calls = 0
    with ProcessPoolExecutor(8) as pool:
        for n, (m, pairrows, requests) in enumerate(pool.map(pair, tasks), 1):
            calls += requests
            rows[m].append(pairrows)
            if n % 12 == 0:
                print(n, flush=True)
    output = {}
    for m, pairs in rows.items():
        output[m] = {
            "expire": summarize([b for a, b in pairs]),
            "control": summarize([a for a, b in pairs]),
        }
        output[m]["improvement"] = {
            method + str(k): ci(
                [
                    b["install"][k][method]["gain"] - a["install"][k][method]["gain"]
                    for a, b in pairs
                ]
            )
            for method in ("inf", "safe_inf", "dir", "safe_dir")
            for k in (1, 2, 4)
        }
    save(
        OUTPUT / "expiry.json",
        dict(
            markets=output,
            backend_requests=calls,
            distinct_seeds=len(tasks),
            terminal_paths=len(tasks) * len(ARMS) * 2,
            full_replays=4,
        ),
    )
