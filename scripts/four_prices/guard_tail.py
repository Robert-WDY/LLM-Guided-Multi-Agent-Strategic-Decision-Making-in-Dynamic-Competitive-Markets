"""Post hoc removal of one contract at a time in the greatest observed guarded losses."""

import json
from functools import partial
from concurrent.futures import ProcessPoolExecutor
import numpy as np
from guard_study import OUTPUT, design, guard
from renewal_study import profile
from run import REVIEW, read, save, simulation
from world import World
from saturation_report import decompose


def native_orders(seed, seat, trajectory):
    """Ex-post simulator instrumentation; never available to the public price guard."""
    world = World("native", seed)
    targets = {d["t"]: d["requested"] for d in trajectory["decisions"]}
    periods = []
    for t in range(1, 61):
        if t in targets:
            contract = targets[t]
        rows, _ = world.settle(contract)
        assert rows == trajectory["rows"][t - 1]
        c = world.state.companies[seat].commercial
        if t >= 19:
            periods.append(
                dict(
                    potential=c.potential_demand_orders,
                    sales=c.sales_orders,
                    attempted_unfulfilled=c.attempted_unfulfilled_orders,
                    lost=c.orders_lost_after_redistribution,
                    redistributed=c.orders_received_from_redistribution,
                )
            )
    return {key: sum(p[key] for p in periods) for key in periods[0]}


def diagnose(task):
    market, k, bad = task
    seed, j = bad["seed"], bad["seat"]
    second = seed >= 203000
    folder = "confirm" if second else "menu"
    raw = read(OUTPUT / folder / f"{seed}.json.gz")
    fn = profile if second else partial(design, "menu")
    group = raw["group"]
    i = next(i for i in range(24 if second else 12) if fn(market, group, i)[0] == seed)
    baseline = raw["paths"]["keep"]
    targets = {d["t"]: d["requested"] for d in baseline["decisions"]}
    f = OUTPUT / "tail" / f"{market}-{k}.json.gz"
    f.parent.mkdir(exist_ok=True)

    def replay(tape, live):
        return simulation(
            market,
            group,
            i,
            offline=tape,
            allow_new=live,
            study_design=fn,
            price_guard=guard(),
            diagnostic_rule="safe_inf",
            diagnostic=(k, j, targets),
        )

    if not f.exists():
        save(f, replay(raw["tape"], True))
    r = read(f)
    assert r["paths"]["natural"]["rows"] == raw["paths"]["safe_inf" + str(k)]["rows"]
    assert json.loads(json.dumps(replay(r["tape"], False)["paths"])) == r["paths"]
    r["paths"]["keep"] = baseline
    nat = np.array(r["paths"]["natural"]["rows"])
    one = {}
    for t in REVIEW:
        omit = np.array(r["paths"][f"omit-{t}"]["rows"])
        assert np.array_equal(nat[: t - 1], omit[: t - 1])
        flow = (nat[t - 1 :, j, 2] - omit[t - 1 :, j, 2]) * 0.99 ** np.arange(61 - t)
        one[t] = dict(
            total=float(flow.sum()),
            first=float(flow[:4].sum()),
            later=float(flow[4:].sum()),
        )
    peers = [i for i in range(4) if i != j]
    fixed = np.array(r["paths"]["fixed_rivals"]["rows"])
    return dict(
        seed=seed,
        seat=j,
        group=group,
        calls=len(r["tape"]) - len(raw["tape"]),
        natural=decompose(r, "natural", "keep", j, 19),
        fixed=decompose(r, "fixed_rivals", "keep", j, 19),
        peer_price_mismatch=int(
            (
                abs(fixed[:, peers, 0] - np.array(baseline["rows"])[:, peers, 0]) > 1e-8
            ).sum()
        ),
        contracts=one,
        orders=(
            {v: native_orders(seed, j, r["paths"][v]) for v in ("natural", "keep")}
            if market == "native"
            else None
        ),
    )


if __name__ == "__main__":
    first = read(OUTPUT / "summary.json")
    second = read(OUTPUT / "confirmation.json")
    tasks = []
    for m in ("native", "econ", "appointments"):
        result = first["menu-" + m] if m == "appointments" else second["pooled-" + m]
        for k in (1, 4):
            tasks.append((m, k, result["install"][str(k)]["safe_inf"]["worst"]))
    with ProcessPoolExecutor(6) as pool:
        out = dict(zip([f"{m}-{k}" for m, k, _ in tasks], pool.map(diagnose, tasks)))
    save(OUTPUT / "tail.json", out)
