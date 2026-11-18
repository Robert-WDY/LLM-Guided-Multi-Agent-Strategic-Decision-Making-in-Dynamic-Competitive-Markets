"""Exercise the production review entry on the exact pre-decision information of known losses."""

import numpy as np
from cause_trials import OUT
from guard_study import OUTPUT as PRIOR, guard
from review_price import review, ACTIONS
from run import read, save


if __name__ == "__main__":
    summary = read(PRIOR / "confirmation.json")["pooled-native"]["install"]
    tails = read(PRIOR / "tail.json")
    results = []
    for k in (1, 2, 4):
        if k == 2:
            fix = summary["2"]["worst_probe"]
            seed, seat, t = fix["seed"], fix["seat"], fix["start"]
            avoid = -fix["gain"]
        else:
            fix = tails[f"native-{k}"]
            seed, seat = fix["seed"], fix["seat"]
            t = int(min(fix["contracts"], key=lambda n: fix["contracts"][n]["total"]))
            avoid = -fix["contracts"][str(t)]["total"]
        folder = "menu" if seed < 200000 else "confirm"
        p = read(PRIOR / folder / f"{seed}.json.gz")["paths"][f"safe_inf{k}"]
        d = next(d for d in p["decisions"] if d["t"] == t)
        a = d["selected"][seat]
        assert a
        h = []
        for rows in p["rows"][: t - 1]:
            peers = [r[0] for j, r in enumerate(rows) if j != seat and r[4]]
            h.append(
                [rows[seat][0], float(np.median(peers)) if peers else rows[seat][0]]
            )
        req = dict(
            market="native",
            round=t,
            prices=h,
            draft=d["draft"][seat],
            proposal=ACTIONS[a],
        )
        old = guard().explain("native", h, req["draft"], t, a)
        new = review(req)
        assert old["action"] == a
        results.append(
            dict(
                seed=seed,
                seat=seat,
                t=t,
                installed=k,
                old_action=ACTIONS[a],
                new=new,
                single_veto_gain=avoid,
            )
        )
    save(OUT / "reproduced-cases.json", results)
