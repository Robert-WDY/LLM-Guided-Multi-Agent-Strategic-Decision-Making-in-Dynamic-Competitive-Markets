"""CPU confirmation of a single four-round contract under three paired actions."""

from concurrent.futures import ProcessPoolExecutor

from cause_trials import ARMS, MARKETS
from guard_study import design
from prices import GROUPS, ROOT, number
from run import read, save, simulation

OUT = ROOT / "experiment-results/renewal-risk-v1"


def profile(market, group, i):
    seed = 250000 + 1000 * MARKETS.index(market) + 40 * GROUPS.index(group) + i
    _, masks, _, _, _, _ = design("menu", market, group, i % 12)
    roles = (
        [0] * 4
        if group == "balanced"
        else [4] * 4
        if group == "unseen"
        else [number("renewal-role", seed, j) % 5 for j in range(4)]
    )
    t = 23 + 4 * (number(seed, "audit-time") % 6)
    return seed, masks, {k: v[0] for k, v in masks.items()}, roles, t, True


def tasks():
    return [(m, g, i) for i in range(40) for g in GROUPS for m in MARKETS]


def archive(task):
    return OUT / "confirm" / f"{profile(*task)[0]}.json.gz"


def simulate(task, tape=None):
    return simulation(*task, offline=tape, study_design=profile, run_only=ARMS)


def collect(task):
    dest = archive(task)
    if dest.exists():
        if read(dest)["seed"] != profile(*task)[0]:
            raise ValueError(f"Wrong archive {dest}")
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    save(dest, simulate(task))


if __name__ == "__main__":
    with ProcessPoolExecutor(8) as pool:
        for n, _ in enumerate(pool.map(collect, tasks()), 1):
            if n % 24 == 0:
                print(n, flush=True)
