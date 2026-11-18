"""Direct repeated identical prompts, changing only the sampling seed and temperature."""

from concurrent.futures import ThreadPoolExecutor
from collections import Counter
import itertools
import numpy as np
from prices import OUT, number, request, label, table
import json


def one(task):
    t, s = task
    return request(("native", 0, 0, 0, 0), number("sampling-check", s) % 2147483646 + 1, t)[
        "action"
    ]


if __name__ == "__main__":
    tasks = [(t, s) for t in (0.0, 0.85) for s in range(256)]
    with ThreadPoolExecutor(max_workers=4) as ex:
        draws = list(ex.map(one, tasks))
    results = {}
    for ix, t in enumerate((0.0, 0.85)):
        a = draws[ix * 256 : (ix + 1) * 256]
        same = np.mean(
            [x == y for j in range(0, 256, 4) for x, y in itertools.combinations(a[j : j + 4], 2)]
        )
        results[str(t)] = dict(counts=dict(Counter(a)), pair_match=float(same), draws=a)
    p = np.asarray(table()[label(("native", 0, 0, 0, 0))]["p"])
    results["expected_stochastic_match"] = float((p @ p) / (p.sum() ** 2))
    (OUT / "sampling-check.json").write_text(json.dumps(results), encoding="utf8")
