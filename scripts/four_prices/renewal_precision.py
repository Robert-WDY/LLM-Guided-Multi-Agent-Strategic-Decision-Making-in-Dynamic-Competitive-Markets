"""Prospective independent retail precision cohort. Archived replays require no server."""

import argparse
from concurrent.futures import ProcessPoolExecutor
from functools import lru_cache

import numpy as np
from cause_value import metrics
from game_theory_agent.game_theory.renewal_review import PublicRenewalReview
from guard_study import design
from guard_verify import replay, verify_raw
from prices import GROUPS, number
from renewal_trials import OUT
from run import read, save, simulation
from saturation_report import discounted


@lru_cache(1)
def reviewer():
    return PublicRenewalReview(OUT / "public-econ.npz")


def profile(m, group, i):
    assert m == "econ"
    seed = 254000 + 120 * GROUPS.index(group) + i
    _, masks, _, _, _, _ = design("menu", m, group, i % 12)
    roles = (
        [0] * 4
        if group == "balanced"
        else [4] * 4
        if group == "unseen"
        else [number("renewal-role", seed, j) % 5 for j in range(4)]
    )
    return (
        seed,
        masks,
        {k: v[0] for k, v in masks.items()},
        roles,
        23 + 4 * (number(seed, "audit-time") % 6),
        True,
    )


def case(item):
    group, i, collect = item
    seed, masks, _, roles, t, _ = profile("econ", group, i)
    folder = OUT / "precision"
    raw = folder / f"{seed}.json.gz"
    prefix = OUT / "precision-pretrade" / f"{seed}.json.gz"

    def action(panel, j, time, draft):
        assert j == masks[1][0] and t == time
        v = dict(
            market="econ",
            seed=seed,
            seat=j,
            round=t,
            draft=draft,
            panel=panel,
            review=reviewer().review(panel, j, time, draft),
        )
        if collect and not prefix.exists():
            prefix.parent.mkdir(parents=True, exist_ok=True)
            save(prefix, v)
        assert v == read(prefix)
        return ("keep", "cut", "raise").index(v["review"]["candidate"])

    def play(tape):
        return simulation(
            "econ",
            group,
            i,
            offline=tape,
            study_design=profile,
            run_only=("keep", "public_probe"),
            probe_policy=action,
        )

    if not raw.exists() and collect:
        folder.mkdir(parents=True, exist_ok=True)
        save(raw, play(None))
    r = read(raw)
    assert r["seed"] == seed and r["roles"] == roles and r["probe"] == t
    verify_raw(r, "menu")
    if i < 2:
        replay(r, play(r["tape"]))
    v = read(prefix)
    assert action(v["panel"], v["seat"], t, v["draft"]) == (
        "keep",
        "cut",
        "raise",
    ).index(v["review"]["candidate"])
    chosen = ("keep", "cut", "raise").index(v["review"]["candidate"])
    j = masks[1][0]
    base, actual = (np.asarray(r["paths"][a]["rows"]) for a in ("keep", "public_probe"))
    assert np.array_equal(actual[: t - 1], base[: t - 1])
    selected = [
        (d["t"], seat, k)
        for d in r["paths"]["public_probe"]["decisions"]
        for seat, k in enumerate(d["selected"])
        if k
    ]
    assert selected == ([(t, j, chosen)] if chosen else [])
    delta = float(discounted(actual[:, :, 2] - base[:, :, 2], t)[j])
    changed = bool(np.any(actual[t - 1 : t + 3, j, 0] != base[t - 1 : t + 3, j, 0]))
    if not changed:
        assert np.array_equal(actual, base)
    return dict(
        seed=seed,
        group=group,
        actual_action=chosen,
        gains=[0.0, delta],
        changed=[False, changed],
        actions={"review": int(bool(chosen))},
        requests=len(r["tape"]),
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--collect", action="store_true")
    collect = parser.parse_args().collect
    with ProcessPoolExecutor(8) as pool:
        rows = []
        for r in pool.map(case, [(g, i, collect) for i in range(120) for g in GROUPS]):
            rows.append(r)
            if len(rows) % 60 == 0:
                print(len(rows), flush=True)
    assert len({r["seed"] for r in rows}) == 360
    summary = metrics(rows, "review", 5)
    summary["by_personality"] = {
        g: metrics([r for r in rows if r["group"] == g], "review", 5) for g in GROUPS
    }
    summary["requests"] = sum(r["requests"] for r in rows)
    save(OUT / "precision-results.json", summary)
    save(OUT / "precision-decisions.json.gz", rows)
