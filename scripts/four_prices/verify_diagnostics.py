"""Independently reconstruct the income effects of the archived forks and seat permutations."""

from itertools import combinations
from unittest.mock import patch
import numpy as np
from prices import GROUPS, MARKETS, OUT
from run import REVIEW, design, jobs, read, simulation
from saturation_report import discounted


def run():
    tail = read(OUT / "tail-replay.json")
    seats = read(OUT / "seat-rotation.json")
    wanted = {row["seed"] for row in seats} | {row["seed"] for row in tail.values()}
    primary = {
        r["seed"]: r
        for m in MARKETS
        for g in GROUPS
        for r in read(OUT / f"{m}-{g}.json.gz")
        if r["seed"] in wanted
    }
    ranks = {design(*task)[0]: task for task in jobs()}
    for m in MARKETS:
        for k in (1, 2, 4):
            data = read(OUT / f"tail-replay-{m}-{k}.json.gz")
            result = tail[f"{m}-{k}"]
            r = primary[data["seed"]]
            j = result["seat"]
            p = data["paths"]
            a = np.asarray(p["natural"]["rows"])
            b = np.array(r["paths"]["keep"]["rows"])
            assert np.array_equal(a, r["paths"][f"inf{k}"]["rows"])
            np.testing.assert_allclose(discounted(a[:, :, 2] - b[:, :, 2])[j], result["realized"])
            fixed = np.asarray(p["fixed_rivals"]["rows"])
            np.testing.assert_allclose(
                discounted(fixed[:, :, 2] - b[:, :, 2])[j], result["fixed_rivals"]
            )
            for t in REVIEW:
                omit = np.asarray(p[f"omit-{t}"]["rows"])
                assert np.array_equal(a[: t - 1], omit[: t - 1])
                np.testing.assert_allclose(
                    discounted(a[:, :, 2] - omit[:, :, 2])[j], result["one_at_a_time"][str(t)]
                )
            if k == 4:
                targets = {d["t"]: d["requested"] for d in r["paths"]["keep"]["decisions"]}
                with patch("prices.request", side_effect=AssertionError("unexpected new request")):
                    rebuilt = simulation(
                        *ranks[r["seed"]], offline=data["tape"], diagnostic=(k, j, targets)
                    )
                for name, policy in rebuilt["paths"].items():
                    np.testing.assert_array_equal(policy["rows"], p[name]["rows"])
    for summary in seats:
        q = read(OUT / f"rotation-{summary['seed']}.json.gz")
        base = np.array(primary[summary["seed"]]["paths"]["keep"]["rows"])
        for n in (1, 2):
            for roster in combinations(range(4), n):
                for method in ("inf", "dir"):
                    name = f"{method}{n}-" + "".join(map(str, roster))
                    a = np.asarray(q["paths"][name]["rows"])
                    assert np.array_equal(a[:18], base[:18])
                    np.testing.assert_allclose(
                        discounted(a[:, :, 2] - base[:, :, 2])[list(roster)].mean(),
                        summary["effects"][name],
                    )
    print(
        "9 worst cases, 90 single-contract withdrawals and 360 same-market rotated policy paths verified. Rebuilt 3 full tail audits offline."
    )


if __name__ == "__main__":
    run()
