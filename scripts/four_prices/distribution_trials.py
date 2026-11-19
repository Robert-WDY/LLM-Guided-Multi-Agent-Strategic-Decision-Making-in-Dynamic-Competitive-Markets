"""Collect the frozen matched-budget training continuations and independent markets."""

import argparse
import os
from concurrent.futures import ProcessPoolExecutor

from cause_trials import ARMS, MARKETS
from guard_verify import replay, verify_raw
from prices import GROUPS, ROOT, number
from renewal_draws import is_focal_future
from renewal_risk import opportunity
from renewal_trials import OUT as RENEWAL, archive as earlier, profile as old_profile
from run import read, save, simulation

OUT = ROOT / "experiment-results/distribution-v1"


def profile(market, group, i):
    _, masks, focal, _, _, _ = old_profile(market, group, i)
    seed = 260000 + 1000 * MARKETS.index(market) + 16 * GROUPS.index(group) + i
    roles = (
        [0] * 4
        if group == "balanced"
        else [4] * 4
        if group == "unseen"
        else [number("distribution-role", seed, j) % 5 for j in range(4)]
    )
    return seed, masks, focal, roles, 23 + 4 * (number(seed, "audit-time") % 6), True


def new_jobs(model):
    markets = MARKETS if model == "smol" else ("econ",)
    return [(model, m, g, i) for i in range(16) for g in GROUPS for m in markets]


def new_path(task):
    model, *args = task
    return OUT / model / f"{profile(*args)[0]}.json.gz"


def collect_new(task, collect):
    model, *args = task
    file = new_path(task)
    if not file.exists() and collect:
        file.parent.mkdir(parents=True, exist_ok=True)
        save(file, simulation(*args, study_design=profile, run_only=ARMS))
    record = read(file)
    seed, masks, _, roles, time, _ = profile(*args)
    assert (record["seed"], record["roles"], record["probe"], record["masks"]["1"]) == (
        seed,
        roles,
        time,
        masks[1],
    )
    verify_raw(record, "menu")
    assert set(record["paths"]) == set(ARMS)
    for arm, label in enumerate(ARMS):
        picked = [
            (d["t"], j, a)
            for d in record["paths"][label]["decisions"]
            for j, a in enumerate(d["selected"])
            if a
        ]
        assert picked == ([(time, masks[1][0], arm)] if arm else [])
    if task[-1] == 0:
        replay(
            record,
            simulation(
                *args, offline=record["tape"], study_design=profile, run_only=ARMS
            ),
        )
    return len(record["tape"])


def repeats(task, collect):
    market, group, i = task
    original = read(earlier(task))
    seed = original["seed"]
    seat, time = original["masks"]["1"][0], original["probe"]
    common = {
        k: v for k, v in original["tape"].items() if not is_focal_future(k, seat, time)
    }
    files = [earlier(task)]
    extra = set()
    for rep in (1, 2, 3):
        reuse = i < 4
        folder = RENEWAL / "draws" if reuse else OUT / "repeated-training"
        file = folder / f"{seed}-{rep}.json.gz"
        draw = number(
            "renewal-innovation" if reuse else "distribution-repetition", seed, rep
        )
        if not file.exists() and collect and not reuse:
            file.parent.mkdir(parents=True, exist_ok=True)
            save(
                file,
                simulation(
                    *task,
                    offline=common,
                    allow_new=True,
                    study_design=old_profile,
                    run_only=ARMS,
                    future_focal_draw=draw,
                ),
            )
        r = read(file)
        assert r["seed"] == seed and r["future_focal_draw"] == draw
        verify_raw(r, "menu")
        assert opportunity(r) == opportunity(original)
        assert all(
            r["paths"][a]["rows"][: time + 3]
            == original["paths"][a]["rows"][: time + 3]
            for a in ARMS
        )
        assert all(r["tape"].get(k) == v for k, v in common.items())
        for k, v in r["tape"].items():
            if k not in common:
                extra.add(v["sha"])
            if not is_focal_future(k, seat, time):
                common[k] = v
        if i == 4 and rep == 1:
            replay(
                r,
                simulation(
                    *task,
                    offline=r["tape"],
                    study_design=old_profile,
                    run_only=ARMS,
                    future_focal_draw=draw,
                ),
            )
        files.append(file)
    return files, len(extra)


def execute(item):
    task, collect = item
    if len(task) == 3:
        return repeats(task, collect)[1]
    return collect_new(task, collect)


def run(model, collect):
    os.environ["FOUR_PRICES_MENU_PORT"] = "19943" if model == "smol" else "19944"
    work = new_jobs(model)
    if model == "smol":
        work = [(m, g, i) for i in range(8) for g in GROUPS for m in MARKETS] + work
    with ProcessPoolExecutor(8) as pool:
        counts = []
        for n, count in enumerate(pool.map(execute, [(t, collect) for t in work]), 1):
            counts.append(count)
            if n % 12 == 0:
                print(model, n, len(work), flush=True)
    save(
        OUT / f"{model}-checks.json",
        dict(
            tasks=len(work),
            request_records=sum(counts),
            tests=len(new_jobs(model)),
            repeated_cases=len(work) - len(new_jobs(model)),
        ),
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("model", choices=("smol", "qwen"))
    parser.add_argument("--collect", action="store_true")
    args = parser.parse_args()
    run(args.model, args.collect)
