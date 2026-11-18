"""Paired CPU/GPU executions collected before/after the backend switch; do not count twice."""

from collections import defaultdict
from cause_trials import OUT
from cause_value import data
from run import read, save
from saturation_report import ci


if __name__ == "__main__":
    groups = defaultdict(list)
    for f in sorted((OUT / "backend-overlap").glob("*.json.gz")):
        gpu = read(f)
        cpu = read(OUT / "test" / f.name)
        assert gpu["seed"] == cpu["seed"] and gpu["masks"] == cpu["masks"]
        same_keys = set(gpu["tape"]) & set(cpu["tape"])
        assert all(gpu["tape"][k]["sha"] == cpu["tape"][k]["sha"] for k in same_keys)
        a, b = data(gpu), data(cpu)
        groups[gpu["market"]].append(
            dict(
                seed=gpu["seed"],
                same_requests=len(same_keys),
                different_draws=sum(
                    gpu["tape"][k]["action"] != cpu["tape"][k]["action"]
                    for k in same_keys
                ),
                gpu_requests=len(gpu["tape"]),
                cpu_requests=len(cpu["tape"]),
                cut=b["gains"][1] - a["gains"][1],
                raise_=b["gains"][2] - a["gains"][2],
            )
        )
    report = {
        m: dict(
            n=len(v),
            same_requests=sum(r["same_requests"] for r in v),
            different_draws=sum(r["different_draws"] for r in v),
            cut_diff=ci([r["cut"] for r in v]),
            raise_diff=ci([r["raise_"] for r in v]),
        )
        for m, v in groups.items()
    }
    save(OUT / "hardware-overlap.json", dict(summary=report, seeds=dict(groups)))
