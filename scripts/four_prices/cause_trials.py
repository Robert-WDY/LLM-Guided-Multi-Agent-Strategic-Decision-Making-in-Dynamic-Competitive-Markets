"""Complete all three candidate contracts at the same randomly scheduled opportunity."""

from concurrent.futures import ProcessPoolExecutor
from functools import partial
import argparse
from prices import ROOT, GROUPS, number
from run import save, simulation
from guard_study import design

OUT = ROOT / "experiment-results/price-causes-v1"
SIZES = {"fit": 120, "calibrate": 96, "test": 96}
MARKETS = ("econ", "native")
ARMS = ["keep", "audit_cut", "audit_raise"]


def profile(phase, market, group, i):
    n = SIZES[phase]
    seed = (
        {"fit": 222000, "calibrate": 224000, "test": 226000}[phase]
        + 1000 * MARKETS.index(market)
        + n // 3 * GROUPS.index(group)
        + i
    )
    _, masks, _, _, _, _ = design("menu", market, group, i % 12)
    roles = (
        [0] * 4
        if group == "balanced"
        else [4] * 4
        if group == "unseen"
        else [number("causes-role", seed, j) % 5 for j in range(4)]
    )
    probe = 23 + 4 * (number(seed, "audit-time") % 6)
    return seed, masks, {k: v[0] for k, v in masks.items()}, roles, probe, True


def jobs(phase):
    return [
        (phase, m, g, i)
        for i in range(SIZES[phase] // 3)
        for g in GROUPS
        for m in MARKETS
    ]


def archive(task):
    phase, m, g, i = task
    return OUT / phase / (str(profile(*task)[0]) + ".json.gz")


def simulate(task, tape=None):
    phase, m, g, i = task
    return simulation(
        m, g, i, offline=tape, study_design=partial(profile, phase), run_only=ARMS
    )


def trial(task):
    f = archive(task)
    f.parent.mkdir(exist_ok=True, parents=True)
    if not f.exists():
        save(f, simulate(task))
    return f.name


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("phase", choices=SIZES)
    phase = parser.parse_args().phase
    with ProcessPoolExecutor(8) as pool:
        for idx, _ in enumerate(pool.map(trial, jobs(phase)), 1):
            if idx % 48 == 0:
                print(idx, flush=True)
