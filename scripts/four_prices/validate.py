"""Verify sample rotation, stochastic independence, causal forks and full terminal replay offline."""

from collections import Counter
import hashlib
import json
from unittest.mock import patch

import numpy as np
from prices import BASE_OUT, MARKETS, GROUPS, OUT, MODEL_SHA, build_request, number, table
from run import REVIEW, design, jobs, read, simulation
from saturation_report import one, summary


def check(replay=True):
    report = read(OUT / "summary.json")
    sample = read(BASE_OUT / "sampling-check.json")
    assert len(sample["0.0"]["counts"]) == 1 and len(sample["0.85"]["counts"]) > 1
    assert read(BASE_OUT / "model.json")["model_sha256"] == MODEL_SHA
    assert all(abs(sum(item["p"]) - 1) < 2.0e-5 for item in table().values())
    assets = read(BASE_OUT / "frozen-assets.json")
    for name, digest in assets["advisers"].items():
        assert (
            hashlib.sha256((BASE_OUT.parent / "publication-v1" / name).read_bytes()).hexdigest()
            == digest
        )
    total = 0
    requests = 0
    max_diff = 0.0
    roles = {}
    for m in MARKETS:
        pool = []
        for g in GROUPS:
            group = read(OUT / f"{m}-{g}.json.gz")
            masks1 = Counter()
            masks2 = Counter()
            seats = Counter()
            for i, r in enumerate(group):
                seed, masks, focal, persona, probe, live = design(m, g, i)
                assert r["seed"] == seed and r["roles"] == persona and r["live"] == live
                assert (len(r["tape"]) > 0) == live
                masks1.update([masks[1]])
                masks2.update([tuple(sorted(masks[2]))])
                seats.update(persona)
                for tag, event in r["tape"].items():
                    seat, t, k = tag.split(":", 2)
                    market, *ints = k.split("/")
                    key = (market, *map(int, ints))
                    rnd = number("model", seed, int(seat), int(t)) % 2147483646 + 1
                    payload = json.dumps(build_request(key, rnd), separators=(",", ":")).encode()
                    assert hashlib.sha256(payload).hexdigest() == event["sha"] and event[
                        "action"
                    ] in range(9)
                    max_diff = max(max_diff, max(abs(np.asarray(table()[k]["p"]) - event["p"])))
                q = np.array(r["paths"]["keep"]["rows"])
                for name, p in r["paths"].items():
                    y = np.array(p["rows"])
                    assert y.shape == (60, 4, 6) and np.isfinite(y).all()
                    if name != "greedy":
                        assert np.array_equal(q[:18], y[:18])
                    if name in ("once", "once_direct"):
                        assert np.array_equal(q[: probe - 1], y[: probe - 1])
                    assert len(p["decisions"]) == 17
                    for d in p["decisions"]:
                        if d["t"] not in REVIEW or name in ("greedy", "keep"):
                            assert not any(d["selected"])
                pool.append(one(r))
                requests += len(r["tape"])
                total += 1
            assert len(masks1) == 4 and len(set(masks1.values())) == 1
            assert len(masks2) == 6 and len(set(masks2.values())) == 1
            roles[m + "-" + g] = dict(seats)
            if replay:
                with patch(
                    "prices.request", side_effect=AssertionError("unexpected backend access")
                ):
                    for i in (0, len(group) - 1):
                        got = simulation(m, g, i, offline=group[i]["tape"])
                        assert json.loads(json.dumps(got)) == group[i]
        th = 20000 if m == "native" else 5
        order = {design(*task)[0]: ix for ix, task in enumerate(jobs())}
        pool.sort(key=lambda r: order[r["seed"]])
        recomputed = summary(pool, th)
        for k in (1, 2, 4):
            for field in ("inferred", "direct", "advantage", "per_opportunity"):
                assert np.allclose(
                    recomputed["arms"][k][field], report[m]["all"]["arms"][str(k)][field]
                )
    assert total == 360 and requests > 0
    return dict(
        seeds=total,
        terminal_policy_paths=total * 13,
        backend_decodes=requests,
        max_backend_probability_drift=max_diff,
        role_seat_counts=roles,
        offline_regenerated_market_seeds=18 if replay else 0,
    )


if __name__ == "__main__":
    result = check()
    (OUT / "checks.json").write_text(json.dumps(result, indent=2), encoding="utf8")
    print(json.dumps({k: v for k, v in result.items() if k != "role_seat_counts"}))
