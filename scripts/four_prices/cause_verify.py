"""Audit raw paired paths, public forecast deployment parity, backend strata and replays."""

import json
from collections import Counter
from cause_trials import OUT, MARKETS, ARMS, archive, jobs, simulate
from cause_value import data, summarize, read, save
from guard_verify import verify_raw
from game_theory_agent.game_theory.panel_advisor import PanelAdvisor


def check():
    devices = read(OUT / "hardware-split.json")
    totals = Counter()
    all_seeds = set()
    for phase in ("fit", "calibrate", "test"):
        rows = {m: [] for m in MARKETS}
        for task in jobs(phase):
            _, m, g, i = task
            f = archive(task)
            r = read(f)
            assert r["seed"] not in all_seeds
            all_seeds.add(r["seed"])
            # Earliest scope can only change one specified firm's contract.
            for name in ARMS:
                acts = [
                    (v["t"], j, a)
                    for v in r["paths"][name]["decisions"]
                    for j, a in enumerate(v["selected"])
                    if a
                ]
                assert acts == (
                    []
                    if name == "keep"
                    else [
                        (
                            r["probe"],
                            r["masks"]["1"][0],
                            1 if name == "audit_cut" else 2,
                        )
                    ]
                )
            totals["paths"] += verify_raw(r, "menu")
            totals["backend_requests"] += len(r["tape"])
            if i == 0 and g == "unseen":
                assert json.loads(json.dumps(simulate(task, r["tape"]))) == r
                totals["replayed_markets"] += 1
            if phase == "test":
                rows[m].append((f.name, r))
        if phase != "test":
            continue
        output = {}
        overlaps = {}
        for m in MARKETS:
            values = read(OUT / f"test-{m}-decisions.json.gz")
            by_seed = {r["seed"]: r for r in values}
            advisor = PanelAdvisor(OUT / f"panel-{m}.npz")
            group = {"cpu": [], "gpu": []}
            for name, r in rows[m]:
                d = data(r)
                v = by_seed[r["seed"]]
                live = advisor.review(d["h"], d["seat"], d["t"], d["draft"])
                assert ["keep", "cut", "raise"].index(live["candidate"]) == v[
                    "actions"
                ]["forecast"]
                group["gpu" if name in devices["test_gpu"] else "cpu"].append(v)
            output[m] = {
                k: summarize(sample, 20000 if m == "native" else 5)
                for k, sample in group.items()
            }
            overlaps[m] = {k: len(v) for k, v in group.items()}
        save(OUT / "backend-summary.json", output)
    totals["seeds"] = len(all_seeds)
    save(OUT / "checks.json", dict(**totals, hardware_counts=overlaps))


if __name__ == "__main__":
    check()
