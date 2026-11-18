"""Locked four-firm confirmation. Every new stochastic quote uses a local model."""

import argparse
from concurrent.futures import ProcessPoolExecutor
from functools import lru_cache, partial

from fit_guard import OUTPUT
from game_theory_agent.game_theory.public_price_guard import PublicPriceGuard
from legal_quote import feasible_quote
from open_price import quote
from prices import GROUPS, MARKETS, number
from run import PAIRS, read, save, simulation

ARMS = ["keep", "once", "safe_once"] + [
    name + str(k)
    for k in (1, 2, 4)
    for name in ("inf", "dir", "safe_inf", "safe_dir", "safe_skip")
]


def design(interface, m, g, i):
    index = (12 if interface == "menu" else 4) * GROUPS.index(g) + i
    seed = (195000 if interface == "menu" else 198000) + 1000 * MARKETS.index(m) + index
    perm = sorted(range(4), key=lambda j: number("repair-seats", m, index // 12, j))
    masks = {
        1: [perm[index % 4]],
        2: [perm[k] for k in PAIRS[index % 6]],
        4: list(range(4)),
    }
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
        {k: mask[number("audit-seat", seed, k) % k] for k, mask in masks.items()},
        roles,
        23 + 4 * (number(seed, "audit-time") % 6),
        True,
    )


@lru_cache(maxsize=1)
def guard():
    return PublicPriceGuard(read(OUTPUT / "bank.json.gz"), historical_value_only=True)


def jobs():
    for interface in ("menu", "legal", "legacy"):
        for i in range(12 if interface == "menu" else 4):
            for g in GROUPS:
                for m in MARKETS if interface == "menu" else MARKETS[:2]:
                    yield interface, m, g, i


def simulate(task, tape=None):
    interface, m, g, i = task
    return simulation(
        m,
        g,
        i,
        offline=tape,
        numeric_quote={
            "menu": None,
            "legal": partial(feasible_quote, disclose_units=False),
            "legacy": quote,
        }[interface],
        study_design=lambda m, g, i: design(interface, m, g, i),
        price_guard=guard(),
        run_only=ARMS if interface != "legacy" else ["keep"],
    )


def path(task):
    interface, m, g, i = task
    return OUTPUT / interface / (str(design(interface, m, g, i)[0]) + ".json.gz")


def trial(task):
    f = path(task)
    f.parent.mkdir(exist_ok=True)
    if not f.exists():
        save(f, simulate(task))
    return f.name


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--workers", type=int, default=8)
    p.add_argument(
        "--interface", choices=["all", "menu", "legal", "legacy"], default="all"
    )
    args = p.parse_args()
    tasks = [task for task in jobs() if args.interface in ("all", task[0])]
    with ProcessPoolExecutor(args.workers) as pool:
        for i, name in enumerate(pool.map(trial, tasks), 1):
            if i % 12 == 0:
                print(i, name, flush=True)
