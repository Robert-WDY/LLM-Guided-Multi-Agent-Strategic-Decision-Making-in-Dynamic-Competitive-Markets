"""Compare like-for-like median forecasts on unchanged sets of listed rivals."""

import numpy as np
from cause_trials import OUT, MARKETS
from cause_value import dataset
from game_theory_agent.game_theory.panel_advisor import PanelAdvisor
from public_value_selector import features
from saturation_report import ci
from run import save


if __name__ == "__main__":
    out = {}
    for m in MARKETS:
        advisor = PanelAdvisor(OUT / f"panel-{m}.npz")
        errors = []
        clocks = []
        for r in dataset("test", m):
            h = r["h"]
            seat = r["seat"]
            t = r["t"]
            v = h[-1].copy()
            v[seat] = np.nan
            mask = np.isfinite(v)
            actual = r["panel"][t - 1].copy()
            actual[seat] = np.nan
            if not r["enabled"] or not np.array_equal(mask, np.isfinite(actual)):
                continue
            truth = np.median(actual[mask])
            output = advisor.review(h, seat, t, r["draft"])
            panel = np.median([q["this_contract"] for q in output["forecasts"]])
            legacy = features(r["legacy"], "conditional")[0][-4] * h[-1, seat]
            last = np.median(v[mask])
            errors.append([abs(np.log(x / truth)) for x in (last, legacy, panel)])
            peers = np.array(r["legacy"]["prices"])[:, 1]
            daily = peers[7:] / peers[6:-1]
            prices = peers[np.arange(6, len(peers), 4)]
            clock = prices[1:] / prices[:-1]
            clocks.append(
                [
                    float(np.mean(abs(daily - 1) < 1e-9)),
                    float(np.mean(abs(clock - 1) < 1e-9)),
                ]
            )
        e = np.array(errors)
        out[m] = dict(
            seeds=len(errors),
            persist=ci(e[:, 0]),
            legacy_quote_component=ci(e[:, 1]),
            panel=ci(e[:, 2]),
            panel_minus_persist=ci(e[:, 2] - e[:, 0]),
            component_minus_persist=ci(e[:, 1] - e[:, 0]),
            unchanged_neighbour_rounds=ci(np.array(clocks)[:, 0]),
            unchanged_at_renewal=ci(np.array(clocks)[:, 1]),
        )
    save(OUT / "median-forecasts.json", out)
