"""Measure how other firms' installations change the very same focal firm's return."""

from collections import defaultdict
import numpy as np
from guard_study import OUTPUT, jobs, path
from renewal_study import tasks, files
from saturation_report import ci, discounted
from run import read, save


def measure(r):
    j = r["masks"]["1"][0]

    def profit(arm):
        return discounted(np.array(r["paths"][arm]["rows"])[:, :, 2])

    base = profit("keep")
    solo = profit("safe_inf1") - base
    all_firms = profit("safe_inf4") - base
    others = [i for i in range(4) if i != j]
    return dict(
        focal_all_minus_solo=float(all_firms[j] - solo[j]),
        rivals_when_solo=float(solo[others].mean()),
        aggregate_when_solo=float(solo.sum()),
        aggregate_when_all=float(all_firms.sum()),
    )


if __name__ == "__main__":
    by_market = defaultdict(list)
    for task in jobs():
        if task[0] == "menu":
            by_market[task[1]].append(measure(read(path(task))))
    for task in tasks():
        by_market[task[0]].append(measure(read(files(task)[0])))
    save(
        OUTPUT / "spillovers.json",
        {
            m: {key: ci([r[key] for r in rows]) for key in rows[0]}
            for m, rows in by_market.items()
        },
    )
