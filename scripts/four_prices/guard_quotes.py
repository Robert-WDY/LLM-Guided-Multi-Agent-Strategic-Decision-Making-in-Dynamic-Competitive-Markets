"""Compare feasible quotes and check independent draws with the exact same prompt."""

from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from itertools import combinations
import json
from urllib.request import ProxyHandler, Request, build_opener

from guard_study import OUTPUT, jobs, path
from quote_collisions import analyze
from run import read, save
from prices import number
from saturation_report import ci, prices
import numpy as np


def repeat(market, template, instance):
    body = dict(
        template, seed=number("legal-repeats", market, instance) % 2147483646 + 1
    )
    req = Request(
        "http://127.0.0.1:19943/v1/chat/completions",
        json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    with build_opener(ProxyHandler({})).open(req, timeout=180) as result:
        reply = json.load(result)
    assert reply["model"] == "smol"
    return dict(
        seed=body["seed"], raw=reply["choices"][0]["message"]["content"].strip()
    )


def main():
    result = {}
    for m in ("native", "econ"):
        arms = {}
        for name in ("legal", "legacy"):
            arms[name] = [read(path(task)) for task in jobs() if task[:2] == (name, m)]
        a, b = arms["legal"], arms["legacy"]
        assert [r["seed"] for r in a] == [r["seed"] for r in b]
        same = lambda r: prices(np.array(r["paths"]["keep"]["rows"]))[0]
        pair = [
            same(x) - same(y) for x, y in zip(a, b) if None not in (same(x), same(y))
        ]
        keep = a[0]["paths"]["keep"]
        key = next(d for d in keep["decisions"] if d["t"] == 7)["model_actions"][0]
        sample = a[0]["tape"][key]
        cache = OUTPUT / (m + "-repeat.json.gz")
        if not cache.exists():
            with ThreadPoolExecutor(8) as pool:
                draws = list(
                    pool.map(lambda i: repeat(m, sample["body"], i), range(128))
                )
            save(
                cache,
                dict(
                    template=sample["body"],
                    draws=draws,
                    low=sample["low"],
                    high=sample["high"],
                ),
            )
        saved = read(cache)
        assert saved["template"] == sample["body"]
        draws = saved["draws"]
        amounts = [float(d["raw"]) for d in draws]
        assert len(set(d["seed"] for d in draws)) == 128 and all(
            sample["low"] <= v <= sample["high"] for v in amounts
        )
        pairs = [
            x == y
            for i in range(0, 128, 4)
            for x, y in combinations(amounts[i : i + 4], 2)
        ]
        result[m] = dict(
            interfaces={k: analyze(v) for k, v in arms.items()},
            paired_same_change=ci(pair),
            comparable_pairs=len(pair),
            repetitions=128,
            distinct=len(set(amounts)),
            pair_ties=sum(pairs),
            pairs=len(pairs),
            top=Counter(amounts).most_common(3),
        )
    save(OUTPUT / "quotes.json", result)


if __name__ == "__main__":
    main()
