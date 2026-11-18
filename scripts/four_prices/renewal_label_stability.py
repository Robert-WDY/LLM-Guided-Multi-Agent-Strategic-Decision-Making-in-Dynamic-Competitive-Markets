"""How often can the very same public prefix receive opposing hindsight abstention labels?"""

import numpy as np
from renewal_trials import OUT
from run import read, save
from saturation_report import ci


def analyze(rows):
    changes = 0
    varying_best = 0
    chosen_flips = 0
    accepted = 0
    cases = []
    evpi = []
    for r in rows:
        if not any(r["changed"]):
            continue
        y = np.column_stack((np.zeros(4), np.asarray(r["gains"])))
        best_gain = y.max(axis=1)
        should_keep = best_gain <= 1e-7
        changes += int(should_keep.any() and not should_keep.all())
        varying_best += int(len(set(y.argmax(axis=1))) > 1)
        # Both oracle comparisons look at the four replicated futures, solely to price uncertainty.
        advantage = float(best_gain.mean() - y.mean(axis=0).max())
        evpi.append(advantage)
        offered = read(OUT / "pretrade" / f"{r['seed']}.json")["review"]["candidate"]
        k = ("keep", "cut", "raise").index(offered)
        gains = y[:, k]
        if k and r["changed"][k - 1]:
            accepted += 1
            chosen_flips += int(gains.min() < -1e-7 and gains.max() > 1e-7)
        cases.append(
            dict(
                seed=r["seed"],
                abstain_for_draw=should_keep.tolist(),
                best_for_draw=y.argmax(axis=1).tolist(),
                chosen=offered,
                chosen_gains=gains.tolist(),
                hindsight_advantage=advantage,
            )
        )
    return dict(
        eligible=len(cases),
        abstention_label_flips=changes,
        best_action_flips=varying_best,
        actually_accepted=accepted,
        accepted_gain_flips=chosen_flips,
        value_of_knowing_the_future_draw=ci(evpi),
        cases=cases,
    )


if __name__ == "__main__":
    rows = read(OUT / "innovations.json")["cases"]
    save(
        OUT / "label-stability.json",
        {m: analyze([r for r in rows if r["market"] == m]) for m in ("econ", "native")},
    )
