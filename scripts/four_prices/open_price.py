"""Check whether stochastic pricing and installation effects persist with digits instead of a menu."""

from concurrent.futures import ProcessPoolExecutor
from hashlib import sha256
import json
from urllib.request import ProxyHandler, Request, build_opener
import numpy as np
from prices import BASE_OUT, GROUPS, ROOT, ROLES, number
from run import PAIRS, read, save, simulation
from saturation_report import discounted, prices, ci

OUTPUT = BASE_OUT / "open-price"
OUTPUT.mkdir(exist_ok=True)


def profile(m, g, i):
    n = 4 * GROUPS.index(g) + i
    seed = (183000 if m == "econ" else 184000) + n
    perm = sorted(range(4), key=lambda j: number("open-seat", m, j))
    masks = {1: (perm[n % 4],), 2: tuple(perm[j] for j in PAIRS[n % 6]), 4: (0, 1, 2, 3)}
    roles = (
        [0] * 4
        if g == "balanced"
        else [1] * 4 if g == "unseen" else [number(seed, j, "persona") % 4 for j in range(4)]
    )
    return (
        seed,
        masks,
        {k: v[0] for k, v in masks.items()},
        roles,
        23 + 4 * (number(seed, "audit-time") % 6),
        True,
    )


def quote(m, role, h, reference, low, high, seed, j, t, tape, offline):
    context = [[round(v, 3) for v in row] for row in h[-4:]]
    msgs = [
        dict(
            role="system",
            content=f"You independently set prices at one of four competing firms. {ROLES[role]}",
        ),
        dict(
            role="user",
            content=f"Market {m}. Set your own numeric selling price for the next {min(4,61-t)} rounds. "
            f"Remaining rounds: {61-t}. Feasible [{low:.3f}, {high:.3f}]. Feedback reference: {reference:.3f}. "
            f"Settled rows [your price, rival median price, own sales, own profit]: {context}. Return only the number.",
        ),
    ]
    body = dict(
        model="smol",
        messages=msgs,
        seed=number("model", seed, j, t) % 2147483646 + 1,
        temperature=0.85,
        top_p=1.0,
        min_p=0.0,
        top_k=0,
        max_tokens=16,
        grammar='root ::= ("0" | [1-9] [0-9]{0,6}) ("." [0-9]{1,3})?',
    )
    data = json.dumps(body, separators=(",", ":")).encode()
    key = sha256(data).hexdigest()
    if key not in tape:
        if offline:
            raise ValueError("Missing archived numeric response")
        req = Request(
            "http://127.0.0.1:19943/v1/chat/completions",
            data=data,
            headers={"Content-Type": "application/json"},
        )
        with build_opener(ProxyHandler({})).open(req, timeout=180) as res:
            ans = json.load(res)
        raw = ans["choices"][0]["message"]["content"].strip()
        assert ans["model"] == "smol" and 0 <= float(raw) < 1e7
        tape[key] = dict(body=body, raw=raw, low=low, high=high)
    return float(np.clip(float(tape[key]["raw"]), low, high)), key


def trial(task):
    m, g, i = task
    s = profile(m, g, i)[0]
    f = OUTPUT / f"{s}.json.gz"
    if not f.exists():
        save(
            f,
            simulation(
                m,
                g,
                i,
                numeric_quote=quote,
                study_design=profile,
                run_only=["keep", "inf1", "dir1", "inf2", "dir2", "inf4", "dir4"],
            ),
        )
    return s


def summarize(rows):
    arms = {}
    base_prices = []
    clips = 0
    calls = 0
    unique = set()
    for r in rows:
        base_prices.append(prices(np.array(r["paths"]["keep"]["rows"]))[0])
        for key, v in r["tape"].items():
            assert sha256(json.dumps(v["body"], separators=(",", ":")).encode()).hexdigest() == key
            x = float(v["raw"])
            unique.add(x)
            clips += int(x < v["low"] or x > v["high"])
            calls += 1
    for k in (1, 2, 4):
        treatments = {}
        same = []
        all_firms = []
        for method in ("inf", "dir"):
            v = []
            for r in rows:
                a = np.array(r["paths"][method + str(k)]["rows"])
                base = np.array(r["paths"]["keep"]["rows"])
                gain = discounted(a[:, :, 2] - base[:, :, 2])[r["masks"][str(k)]]
                v.append(float(gain.mean()))
                if method == "inf":
                    same.append(prices(a)[0])
                    all_firms.extend(gain.tolist())
            treatments[method] = v
        arms[k] = dict(
            inferred=ci(treatments["inf"]),
            direct=ci(treatments["dir"]),
            advantage=ci(np.subtract(treatments["inf"], treatments["dir"])),
            same=ci(same),
            same_delta=ci(
                [a - b for a, b in zip(same, base_prices) if a is not None and b is not None]
            ),
            comparable_markets=sum(
                a is not None and b is not None for a, b in zip(same, base_prices)
            ),
            worst=min(all_firms),
        )
    return dict(
        seeds=len(rows),
        baseline_same=ci(base_prices),
        calls=calls,
        clipped=clips,
        distinct_numbers=len(unique),
        baseline_comparable_markets=sum(x is not None for x in base_prices),
        end_active={
            v: sum(int(np.asarray(r["paths"][v]["rows"])[-1, :, 5].sum()) for r in rows)
            for v in ("keep", "inf4")
        },
        arms=arms,
    )


if __name__ == "__main__":
    jobs = [(m, g, i) for i in range(4) for g in GROUPS for m in ("econ", "native")]
    with ProcessPoolExecutor(6) as pool:
        list(pool.map(trial, jobs))
    results = {}
    for m in ("econ", "native"):
        rows = [read(OUTPUT / f"{profile(*t)[0]}.json.gz") for t in jobs if t[0] == m]
        results[m] = summarize(rows)
        for g, i in ((GROUPS[0], 0), (GROUPS[-1], 3)):
            r = next(v for v in rows if v["seed"] == profile(m, g, i)[0])
            copy = simulation(
                m,
                g,
                i,
                offline=r["tape"],
                numeric_quote=quote,
                study_design=profile,
                run_only=["keep", "inf1", "dir1", "inf2", "dir2", "inf4", "dir4"],
            )
            assert json.loads(json.dumps(copy)) == r
    save(OUTPUT / "summary.json", results)
