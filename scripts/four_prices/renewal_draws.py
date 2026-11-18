"""Change only the focal manager's future sampling stream, with full paired settlements."""

import argparse
from concurrent.futures import ProcessPoolExecutor

import numpy as np
from cause_trials import ARMS, MARKETS
from game_theory_agent.game_theory.renewal_review import PublicRenewalReview
from guard_verify import replay, verify_raw
from prices import number
from renewal_risk import opportunity, targets, verify_pretrade
from renewal_trials import OUT, archive, profile, tasks
from run import read, save, simulation
from saturation_report import ci


def is_focal_future(tag, seat, t):
    j, time, _ = tag.split(":")
    return int(j) == seat and int(time) > t


def case(item):
    task, collect = item
    market, group, i = task
    original = read(archive(task))
    seat, t, seed = original["masks"]["1"][0], original["probe"], original["seed"]
    common = {
        k: v for k, v in original["tape"].items() if not is_focal_future(k, seat, t)
    }
    _, values, _ = targets(original)
    gains = [[values[0], values[4]]]
    changed = [bool(values[1]), bool(values[5])]
    additional = set()
    reviewer = PublicRenewalReview(OUT / f"public-{market}.npz")
    preview = OUT / "pretrade" / f"{seed}.json"

    def observe(listings, focal_seat, time, draft):
        assert focal_seat == seat and t == time and len(listings) == time - 1
        record = dict(
            seed=seed,
            market=market,
            panel=listings,
            seat=seat,
            round=time,
            draft=draft,
            review=reviewer.review(listings, seat, time, draft),
        )
        if collect and not preview.exists():
            preview.parent.mkdir(parents=True, exist_ok=True)
            save(preview, record)
        assert read(preview) == record

    for rep in (1, 2, 3):
        draw = number("renewal-innovation", seed, rep)
        dst = OUT / "draws" / f"{seed}-{rep}.json.gz"
        if not dst.exists() and collect:
            dst.parent.mkdir(parents=True, exist_ok=True)
            result = simulation(
                *task,
                offline=common,
                allow_new=True,
                study_design=profile,
                run_only=ARMS,
                future_focal_draw=draw,
                observe_proposal=observe,
            )
            save(dst, result)
        assert preview.is_file()
        r = read(dst)
        assert r["future_focal_draw"] == draw and r["seed"] == seed
        assert all(k in r["tape"] and r["tape"][k] == v for k, v in common.items())
        verify_raw(r, "menu")
        for k, v in r["tape"].items():
            if k not in common:
                additional.add(v["sha"])
            if not is_focal_future(k, seat, t):
                common[k] = v
        assert opportunity(original) == opportunity(r)
        for arm in ARMS:
            assert (
                r["paths"][arm]["rows"][: t + 3]
                == original["paths"][arm]["rows"][: t + 3]
            )
        assert all(
            all(
                d["model_actions"] == old["model_actions"]
                for d, old in zip(
                    r["paths"][arm]["decisions"], original["paths"][arm]["decisions"]
                )
                if d["t"] <= t
            )
            for arm in ARMS
        )
        if i == 0 and rep == 1:
            replay(
                r,
                simulation(
                    *task,
                    offline=r["tape"],
                    study_design=profile,
                    run_only=ARMS,
                    future_focal_draw=draw,
                    observe_proposal=observe,
                ),
            )
        _, v, _ = targets(r)
        assert changed == [bool(v[1]), bool(v[5])]
        gains.append([v[0], v[4]])
    preview_record = read(preview)
    verify_pretrade(original, preview_record)
    observe(preview_record["panel"], seat, t, preview_record["draft"])
    return dict(
        seed=seed,
        market=market,
        group=group,
        changed=changed,
        gains=gains,
        additional_requests=len(additional),
    )


def summarize(rows, market):
    result = {}
    for idx, arm in enumerate(("cut", "raise")):
        eligible = [r for r in rows if r["market"] == market and r["changed"][idx]]
        gains = np.asarray([r["gains"] for r in eligible])[:, :, idx]
        low, high = gains.min(axis=1), gains.max(axis=1)
        both = (low < -1e-7) & (high > 1e-7)
        result[arm] = dict(
            executed=len(gains),
            flips=int(both.sum()),
            min_gain=float(low.min()),
            max_gain=float(high.max()),
            within_market_range=ci(high - low),
            mean_across_draws=ci(gains.mean(axis=1)),
        )
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--collect", action="store_true")
    flag = parser.parse_args().collect
    with ProcessPoolExecutor(8) as pool:
        rows = list(pool.map(case, [(t, flag) for t in tasks() if t[2] < 4]))
    save(
        OUT / "innovations.json",
        dict(
            cases=rows,
            markets={m: summarize(rows, m) for m in MARKETS},
            additional_requests=sum(r["additional_requests"] for r in rows),
        ),
    )
