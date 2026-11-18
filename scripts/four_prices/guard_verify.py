"""Validate raw decision clocks, allocations, response provenance and representative full replays."""

from collections import Counter, defaultdict
from hashlib import sha256
import json
from guard_study import OUTPUT, jobs, path, simulate
from fit_guard import BASE_OUT, examples
from renewal_study import tasks, files, simulate as renew
from prices import GROUPS, build_request, number
from run import read, save
from saturation_report import ci
from guard_report import one, summarize


def verify_raw(r, interface):
    assert len(r["roles"]) == 4 and r["live"]
    for tag, v in r["tape"].items():
        if interface in ("legal", "legacy"):
            digest = sha256(
                json.dumps(v["body"], separators=(",", ":")).encode()
            ).hexdigest()
            assert digest == tag
            if interface == "legal":
                assert v["low"] <= float(v["raw"]) <= v["high"]
        else:
            seat, t, k = tag.split(":")
            market, role, rel, sales, profit = k.split("/")
            assert r["roles"][int(seat)] == int(role)
            seed = number("model", r["seed"], int(seat), int(t)) % 2147483646 + 1
            body = build_request(
                (market, int(role), int(rel), int(sales), int(profit)), seed
            )
            assert (
                sha256(json.dumps(body, separators=(",", ":")).encode()).hexdigest()
                == v["sha"]
            )
            assert 0 <= v["action"] <= 8 and abs(sum(v["p"]) - 1) < 2e-5
    for name, p in r["paths"].items():
        assert len(p["rows"]) == 60 and len(p["market"]) == 60
        assert [d["t"] for d in p["decisions"]] == [1, 3, 5, *range(7, 60, 4)]
        for d in p["decisions"]:
            assert len(d["selected"]) == 4
            if name[-1].isnumeric():
                assert all(
                    not a or j in r["masks"][name[-1]]
                    for j, a in enumerate(d["selected"])
                )
            for j, id_ in enumerate(d["model_actions"]):
                if id_ is None:
                    continue
                if interface in ("legal", "legacy"):
                    assert id_ in r["tape"]
                    if interface == "legal":
                        assert float(r["tape"][id_]["raw"]) == d["draft"][j]
    return len(r["paths"])


def replay(r, new):
    # Normalize integer dictionary keys as serialized on disk.
    assert json.loads(json.dumps(new)) == r


def main():
    counters = Counter()
    primary = {}
    sides = {}
    banks = read(OUTPUT / "bank.json.gz")
    for m in ("native", "econ"):
        old = [
            row
            for g in GROUPS
            for r in read(BASE_OUT / "live" / f"{m}-{g}.json.gz")
            for row in examples(r)
        ]
        assert old == banks[m]
        counters["verified_training_examples"] += len(old)
    training_seeds = {r["seed"] for bank in banks.values() for r in bank}
    allocations = defaultdict(list)
    cohort_seeds = defaultdict(set)

    def record_allocation(r, cohort):
        market, group = r["market"], r["group"]
        if group == "unseen":
            assert r["roles"] == [4] * 4
        if group == "balanced":
            assert r["roles"] == [0] * 4
        assert r["seed"] not in cohort_seeds[cohort]
        cohort_seeds[cohort].add(r["seed"])
        allocations[cohort, market].append(r["masks"])

    for task in jobs():
        interface, m, g, i = task
        r = read(path(task))
        record_allocation(r, interface)
        assert r["seed"] not in training_seeds
        counters["paths"] += verify_raw(r, interface)
        counters["backend_requests"] += len(r["tape"])
        if i == 0 and g == "unseen":
            replay(r, simulate(task, r["tape"]))
            counters["full_replays"] += 1
        if interface == "menu":
            primary.setdefault(m, []).append(one(r, "menu"))
    for task in tasks():
        m, g, i = task
        a, b = map(read, files(task))
        record_allocation(a, "confirm")
        assert a["seed"] not in training_seeds
        assert (
            a["seed"] == b["seed"]
            and a["paths"]["keep"]["rows"] == b["paths"]["keep"]["rows"]
        )
        assert all(k in b["tape"] and b["tape"][k] == v for k, v in a["tape"].items())
        counters["backend_requests"] += len(b["tape"])
        for r, stage in [(a, "confirm"), (b, "renewal")]:
            counters["paths"] += verify_raw(r, "menu")
            if i == 0 and g == "unseen":
                replay(r, renew(task, r["tape"], rollback=stage == "renewal"))
                counters["full_replays"] += 1
            row = one(r, "menu")
            row["rollback_counts"] = {
                arm: sum(v > 0 for d in p["decisions"] for v in d.get("rollback", []))
                for arm, p in r["paths"].items()
            }
            sides.setdefault(f"{stage}-{m}", []).append(row)
            if stage == "confirm":
                primary[m].append(row)
    assert cohort_seeds["legacy"] == cohort_seeds["legal"]
    assert not (
        cohort_seeds["confirm"] & (cohort_seeds["menu"] | cohort_seeds["legal"])
    )
    assert not (cohort_seeds["menu"] & cohort_seeds["legal"])
    for masks in allocations.values():
        one_counts = Counter(tuple(m["1"]) for m in masks)
        pair_counts = Counter(tuple(sorted(m["2"])) for m in masks)
        assert len(one_counts) == 4 and len(set(one_counts.values())) == 1
        assert len(pair_counts) == 6 and len(set(pair_counts.values())) == 1
    counters["independent_seeds"] = len(set.union(*cohort_seeds.values()))
    output = {k: summarize(v) for k, v in sides.items()}
    for m, rows in primary.items():
        if m == "appointments":
            continue
        assert len(set(r["seed"] for r in rows)) == len(rows) == 108
        output["pooled-" + m] = summarize(rows)
        a = sides["confirm-" + m]
        b = sides["renewal-" + m]
        pairs = {}
        for k in (1, 2, 4):
            for method in ("inf", "safe_inf", "dir", "safe_dir"):
                change = [
                    y["install"][k][method]["gain"] - x["install"][k][method]["gain"]
                    for x, y in zip(a, b)
                ]
                pairs[method + str(k)] = dict(
                    vs_no_rollback=ci(change),
                    triggers=sum(r["rollback_counts"][method + str(k)] for r in b),
                )
        output["rollback-effect-" + m] = pairs
    save(OUTPUT / "confirmation.json", output)
    save(OUTPUT / "checks.json", dict(counters))
    print(dict(counters), flush=True)


if __name__ == "__main__":
    main()
