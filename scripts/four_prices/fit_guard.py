"""Extract terminal consequences of one changed contract in historical markets."""

from collections import Counter
import numpy as np
from prices import BASE_OUT, ROOT, GROUPS
from run import read, save
from saturation_report import discounted
from game_theory_agent.game_theory.public_price_guard import public_features

OUTPUT = ROOT / "experiment-results/price-guard-v1"


def examples(record):
    t = record["probe"]
    seen = set()
    for arm, control, j in [
        ("once", "keep", record["masks"]["1"][0]),
        ("once_direct", "keep", record["masks"]["1"][0]),
        *[(f"inf{k}", f"skip{k}", record["focal"][str(k)]) for k in (1, 2, 4)],
    ]:
        path = record["paths"][arm]
        a = np.asarray(path["rows"])
        b = np.asarray(record["paths"][control]["rows"])
        assert np.array_equal(a[: t - 1], b[: t - 1])
        if np.array_equal(a[t - 1 : t + 3, j, 0], b[t - 1 : t + 3, j, 0]):
            continue
        h = []
        for rows in a[: t - 1]:
            peers = [r[0] for seat, r in enumerate(rows) if seat != j and r[4]]
            h.append([rows[j, 0], np.median(peers) if peers else rows[j, 0]])
        d = next(d for d in path["decisions"] if d["t"] == t)
        features = public_features(h, d["draft"][j], t)
        gain = float(discounted(a[:, :, 2] - b[:, :, 2], t)[j])
        identity = (j, *features, gain)
        if identity in seen:
            continue
        seen.add(identity)
        yield dict(
            seed=record["seed"],
            action=d["selected"][j],
            features=features,
            gain=gain,
            sales_delta=float((a[t - 1 :, j, 1] - b[t - 1 :, j, 1]).sum()),
            exit_delta=int(b[-1, j, 5]) - int(a[-1, j, 5]),
        )


if __name__ == "__main__":
    OUTPUT.mkdir(exist_ok=True)
    banks = {}
    for market in ("native", "econ"):
        rows = [
            v
            for group in GROUPS
            for r in read(BASE_OUT / "live" / f"{market}-{group}.json.gz")
            for v in examples(r)
        ]
        banks[market] = rows
        print(market, len(rows), Counter(r["action"] for r in rows), flush=True)
    save(OUTPUT / "bank.json.gz", banks)
