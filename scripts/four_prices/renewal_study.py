"""Independent confirmation plus paired manager-side rollback of an unsuccessful trial."""

from concurrent.futures import ProcessPoolExecutor
from guard_study import ARMS, OUTPUT, design, guard
from prices import GROUPS, number
from run import simulation, read, save


def profile(m, g, i):
    # Reuse balanced assignment; select fresh independent manager and market seeds.
    seat_block = i % 12
    _, masks, _, _, _, _ = design("menu", m, g, seat_block)
    seed = (204000 if m == "native" else 203000) + GROUPS.index(g) * 24 + i
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


def tasks():
    return [(m, g, i) for i in range(24) for g in GROUPS for m in ("econ", "native")]


def simulate(task, old=None, allow_new=False, rollback=False):
    m, g, i = task
    return simulation(
        m,
        g,
        i,
        study_design=profile,
        run_only=ARMS,
        price_guard=guard(),
        feedback_rollback=rollback,
        offline=old,
        allow_new=allow_new,
    )


def files(task):
    seed = profile(*task)[0]
    return [OUTPUT / v / f"{seed}.json.gz" for v in ("confirm", "renewal")]


def trial(task):
    a, b = files(task)
    for p in (a, b):
        p.parent.mkdir(exist_ok=True)
    if not a.exists():
        save(a, simulate(task))
    if not b.exists():
        original = read(a)
        updated = simulate(task, original["tape"], allow_new=True, rollback=True)
        assert updated["paths"]["keep"]["rows"] == original["paths"]["keep"]["rows"]
        save(b, updated)
    return b.name


if __name__ == "__main__":
    with ProcessPoolExecutor(8) as pool:
        for n, name in enumerate(pool.map(trial, tasks()), 1):
            if n % 12 == 0:
                print(n, name, flush=True)
