"""Isolate attribution of an externally changed price in the manager's feedback controller."""

from concurrent.futures import ProcessPoolExecutor
import numpy as np
from cause_trials import OUT, ARMS
from guard_study import design
from prices import GROUPS, number
from run import read, save, simulation
from saturation_report import ci, counts, discounted


def profile(m, g, i):
    _, masks, _, _, _, _ = design("menu", m, g, i)
    seed = 233000 + GROUPS.index(g) * 8 + i
    roles = (
        [0] * 4
        if g == "balanced"
        else [4] * 4
        if g == "unseen"
        else [number("credit-role", seed, j) % 5 for j in range(4)]
    )
    return (
        seed,
        masks,
        {k: v[0] for k, v in masks.items()},
        roles,
        23 + 4 * (number(seed, "audit-time") % 6),
        True,
    )


def case(task):
    g, i = task
    seed, masks, _, _, t, _ = profile("native", g, i)
    j = masks[1][0]
    tape = {}
    values = {}
    activity = {}
    for mode in ("original", "anchor_only", "source"):
        f = OUT / "credit" / f"{seed}-{mode}.json.gz"
        f.parent.mkdir(parents=True, exist_ok=True)

        def play(cache, online):
            return simulation(
                "native",
                g,
                i,
                offline=cache,
                allow_new=online,
                run_only=ARMS,
                study_design=profile,
                expire_advice=mode == "anchor_only",
                track_feedback_source=mode == "source",
            )

        if not f.exists():
            save(f, play(tape, True))
        r = read(f)
        assert all(k in r["tape"] and v == r["tape"][k] for k, v in tape.items())
        tape = r["tape"]
        if i == 0:
            assert (
                r["paths"]["keep"]["rows"] == play(tape, False)["paths"]["keep"]["rows"]
            )
            import json

            assert json.loads(json.dumps(play(tape, False))) == r
        base = np.array(r["paths"]["keep"]["rows"])
        if mode == "original":
            initial = base
        assert np.array_equal(initial, base)
        values[mode] = {}
        for arm in ARMS[1:]:
            a = np.array(r["paths"][arm]["rows"])
            d = discounted(a[:, :, 2] - base[:, :, 2], t)[j]
            activity[arm] = bool(
                np.any(a[t - 1 : t + 3, j, 0] != base[t - 1 : t + 3, j, 0])
            )
            if mode == "original":
                activity[arm + "_prefix"] = a[: t + 3].tolist()
            else:
                assert np.array_equal(a[: t + 3], activity[arm + "_prefix"])
            values[mode][arm] = float(d)
    return dict(
        seed=seed,
        values=values,
        changed={arm: activity[arm] for arm in ARMS[1:]},
        requests=len(tape),
    )


if __name__ == "__main__":
    with ProcessPoolExecutor(8) as pool:
        rows = list(pool.map(case, [(g, i) for g in GROUPS for i in range(8)]))
    out = {}
    for arm in ARMS[1:]:
        cases = {}
        for mode in ("original", "anchor_only", "source"):
            gain = [r["values"][mode][arm] for r in rows]
            risk = counts(
                [
                    dict(
                        used=int(r["changed"][arm]),
                        lost=int(y < -1e-7),
                        large=int(y < -20000),
                    )
                    for r, y in zip(rows, gain)
                ]
            )
            cases[mode] = dict(effect=ci(gain), worst=min(gain), **risk)
        for comparator in ("original", "anchor_only"):
            cases["source_minus_" + comparator] = ci(
                [
                    r["values"]["source"][arm] - r["values"][comparator][arm]
                    for r in rows
                ]
            )
        out[arm] = cases
    save(
        OUT / "credit-results.json",
        dict(arms=out, seeds=rows, requests=sum(r["requests"] for r in rows)),
    )
