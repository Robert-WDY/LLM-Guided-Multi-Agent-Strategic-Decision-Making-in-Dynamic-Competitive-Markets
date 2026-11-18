"""Three counterfactual settlements from an identical history, without exposing labels to advice."""
import argparse
import hashlib
import sys
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts/public_forecast"))
from collect import decision, draw, limits, read, save, settle, world
from paid import initialize

OUT = ROOT / "experiment-results/public-price-value-v1"
RAW = ROOT / "runs/public-price-value-v1"
CONDITIONS = ("stable", "demand", "rival", "reverse")
SPLITS = {"train": (91000, 96), "cal": (92000, 40), "test": (93000, 80), "real": (94000, 12)}
for directory in (OUT, RAW):
    directory.mkdir(parents=True, exist_ok=True)


def run(task):
    phase, market, seed = task
    path = RAW / f"{phase}-{market}-{seed}.json.gz"
    if path.exists():
        return
    index = seed - SPLITS[phase][0]
    provider = ("deepseek", "doubao")[(index + index // 4) % 2] if phase == "real" else None
    condition = CONDITIONS[index % 4]
    original = world(market, seed, condition)
    t0 = 23 + 4 * (draw(seed, "opportunity") % 6)
    rows, failures = [], []
    last = None
    for t in range(1, t0 + 1):
        lo, hi = limits(original)
        draft = decision(original, rows, last, provider, seed, lo, hi, failures)
        if t == t0:
            break
        rows.append(settle(original, draft["price"], lo, hi))
        last = draft
    assert draft["start"] == t0 and draft["end"] == t0 + 3
    public = {"prices": [v[:2] for v in rows], "draft": draft["draft"], "round": t0}
    prefix_fallbacks = list(failures)
    if phase in ("test", "real"):
        from public_value_selector import choose
        choice = choose(market, public)
        seal = RAW / f"sealed-{market}-{seed}.json"
        if seal.exists():
            assert read(seal) == choice
        else:
            save(seal, choice)
    else:
        choice = None
    paths = {}
    for arm, multiplier in (("keep", 1.), ("up", 1.04), ("down", .96)):
        w = original.clone()
        h = list(rows)
        contract = dict(draft, price=draft["draft"] * multiplier)
        # All arms use the same published market-wide request bounds, then native settlement constraints.
        contract["price"] = float(np.clip(contract["price"], w.low, w.high))
        fallback = []
        for t in range(t0, 61):
            lo, hi = limits(w)
            if t > t0:
                contract = decision(w, h, contract, provider, seed, lo, hi, fallback)
            h.append(settle(w, contract["price"], lo, hi))
        paths[arm] = {"settled": h[t0 - 1:], "fallback": fallback}
    save(path, dict(seed=seed, market=market, condition=condition, provider=provider, public=public,
                    decision=choice, prefix_fallback=prefix_fallbacks, paths=paths))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("phase", choices=SPLITS)
    p.add_argument("--workers", type=int, default=8)
    args = p.parse_args()
    start, n = SPLITS[args.phase]
    real = args.phase == "real"
    if real:
        initialize()
    if args.phase in ("test", "real"):
        from public_value_selector import verify
        verify()
    with (ThreadPoolExecutor if real else ProcessPoolExecutor)(max_workers=args.workers) as ex:
        list(ex.map(run, [(args.phase, m, s) for s in range(start, start + n) for m in ("econ", "native")]))
    print(args.phase, 2*n, flush=True)


if __name__ == "__main__":
    main()
