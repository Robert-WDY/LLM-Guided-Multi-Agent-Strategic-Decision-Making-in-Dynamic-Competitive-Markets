"""Verify the public interface, model replies, treatment timing and full archived replays."""

from collections import Counter
import numpy as np
from game_theory_agent.game_theory.renewal_review import PublicRenewalReview
from cause_trials import ARMS, MARKETS
from cause_value import public_panel
from guard_verify import verify_raw, replay
from renewal_trials import OUT, archive, tasks, simulate, profile
from renewal_risk import opportunity
from run import read, save


def verify():
    count = Counter()
    seats = {m: Counter() for m in MARKETS}
    for m in MARKETS:
        model = PublicRenewalReview(OUT / f"public-{m}.npz")
        predictions = {r["seed"]: r for r in read(OUT / f"predictions-{m}.json.gz")}
        for task in tasks():
            if task[0] != m:
                continue
            r = read(archive(task))
            seed, masks, _, roles, t, _ = profile(*task)
            j = masks[1][0]
            seats[m][j] += 1
            assert (
                seed == r["seed"]
                and roles == r["roles"]
                and t == r["probe"]
                and r["masks"]["1"] == [j]
            )
            count["terminal_paths"] += verify_raw(r, "menu")
            count["model_requests"] += len(r["tape"])
            count["markets"] += 1
            keep = r["paths"]["keep"]
            for k, name in enumerate(ARMS):
                interventions = [
                    (d["t"], i, a)
                    for d in r["paths"][name]["decisions"]
                    for i, a in enumerate(d["selected"])
                    if a
                ]
                assert interventions == ([(t, j, k)] if k else [])
            if task[2] == 0:
                replay(r, simulate(task, r["tape"]))
                count["fully_replayed_markets"] += 1
            panel = public_panel(np.asarray(keep["rows"][: t - 1]))
            draft = next(d["draft"][j] for d in keep["decisions"] if d["t"] == t)
            redacted = np.zeros((t - 1, 4, 6))
            redacted[:, :, 0] = np.nan_to_num(panel)
            redacted[:, :, 4] = np.isfinite(panel)
            drafts = [None] * 4
            drafts[j] = draft
            minimal = dict(
                probe=t,
                masks={"1": [j]},
                paths={
                    "keep": dict(rows=redacted, decisions=[dict(t=t, draft=drafts)])
                },
            )
            assert opportunity(r) == opportunity(minimal)
            s = predictions.pop(seed)
            live = model.review(panel, j, t, draft)
            assert (
                live["candidate"] == ("keep", "cut", "raise")[s["choices"]["renewal"]]
            )
            assert live["requested"] == draft and live["disposition"] == "keep"
            assert bool(live["alternatives"]) == s["enabled"]
            for arm, v in enumerate(live["alternatives"].values()):
                prediction = np.asarray(s["path"]).reshape(2, 2, 2)[arm]
                np.testing.assert_allclose(
                    v["predicted_log_quote_difference"], prediction[:, 0], atol=1e-10
                )
                np.testing.assert_allclose(
                    v["predicted_listing_probability_difference"],
                    prediction[:, 1],
                    atol=1e-10,
                )
                y = s["scores"]["renewal"][4 * arm : 4 * arm + 4]
                np.testing.assert_allclose(
                    [v["estimated_terminal_gain"], v["expected_downside"]],
                    [y[0], y[3]],
                    atol=1e-7,
                )
            count["public_prefix_parity"] += 1
        assert not predictions
    save(OUT / "checks.json", dict(**count, focal_seat_counts=seats))


if __name__ == "__main__":
    verify()
