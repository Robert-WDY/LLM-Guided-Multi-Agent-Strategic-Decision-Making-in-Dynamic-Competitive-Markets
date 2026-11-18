"""Rebuild statistics from exported settlement rows; no model calls or fitting."""
import argparse
import gzip
import json
from pathlib import Path

import numpy as np
from scipy.stats import beta, rankdata

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "experiment-results/public-price-value-v1"
RAW = ROOT / "runs/public-price-value-v1"
ARMS = ("keep", "down", "up")


def read(path):
    data = path.read_bytes()
    return json.loads(gzip.decompress(data) if path.suffix == ".gz" else data)


def save(path, value):
    data = json.dumps(value, ensure_ascii=False, allow_nan=False,
                      indent=None if path.suffix == ".gz" else 2).encode()
    path.write_bytes(gzip.compress(data, mtime=0) if path.suffix == ".gz" else data)


def ci(x):
    x = np.asarray(x, dtype=float)
    samples = np.random.default_rng(27191).choice(x, (10000, len(x))).mean(axis=1)
    return [float(x.mean()), *np.quantile(samples, [.025, .975]).tolist()]


def selection(y, use, scores):
    y, use = np.asarray(y), np.asarray(use, dtype=bool)
    n = len(y)
    k = int(use.sum())
    lost = int((y[use] < -1.e-8).sum())
    hit = y > 1.e-8
    rate = k/n
    # Compare with uniformly random selection of k opportunities, preserving the candidate direction.
    rng = np.random.default_rng(27191)
    idx = rng.integers(n, size=(10000, n))
    gap = (use*y)[idx].mean(axis=1) - use[idx].mean(axis=1)*y[idx].mean(axis=1)
    auc = None
    if hit.any() and not hit.all():
        auc = float((rankdata(scores)[hit].sum()-hit.sum()*(hit.sum()+1)/2)/(hit.sum()*(~hit).sum()))
    return dict(n=n, accepted=k, lost=lost, missed_winners=int(hit[~use].sum()),
                rejected=int((~use).sum()), wins_accepted=int(hit[use].sum()), auc=auc,
                loss_upper=float(beta.ppf(.95,lost+1,k-lost)) if k>lost else (1. if k else None),
                gate_keep=ci(use*y), force_keep=ci(y), random_select_gain=[
                    float((use*y).mean()-rate*y.mean()), *np.quantile(gap, [.025, .975]).tolist()])


def summarize(rows):
    outcomes = []
    for r in rows:
        t = r["public"]["round"]
        assert set(r["public"]) == {"prices","draft","round"} and len(r["public"]["prices"]) == t-1
        h = np.array([r["paths"][a]["settled"] for a in ARMS])
        assert h.shape == (3, 61-t, 9) and np.isfinite(h).all()
        assert np.allclose(h[:, :4, 6], h[:, 0, 6, None])
        assert np.allclose(h[:, 0, 1], h[0,0,1]), "rival saw a simultaneous bid"
        scale = 100 if r["market"]=="native" else 1
        diff = (h[:,:,3]-h[0,:,3]) @ (.99**np.arange(61-t)) / scale
        rec = dict(values=diff.tolist(), condition=r["condition"], same_first_price=bool(np.ptp(h[:,0,0])<1e-8))
        for m in ("conditional", "generic"):
            v = r["decision"][m]
            rec[m] = (float(diff[ARMS.index(v["arm"])]), v["use"], v["score"], v["arm"])
        outcomes.append(rec)
    stats = {}
    for m in ("conditional", "generic"):
        stats[m] = selection(*zip(*(v[m][:3] for v in outcomes)))
    def gain(o,m):
        return o[m][0]*o[m][1]
    gains = [gain(o,"conditional")-gain(o,"generic") for o in outcomes]
    stats.update(gated_conditional_minus_generic=ci(gains),
                 force_conditional_minus_generic=ci([o["conditional"][0]-o["generic"][0] for o in outcomes]),
                 direction_disagrees=sum(o["conditional"][3]!=o["generic"][3] for o in outcomes),
                 oracle_ceiling=ci([max(o["values"]) for o in outcomes]),
                 both_directions_equal_keep_first_price=sum(o["same_first_price"] for o in outcomes),
                 conditions={c:ci([gain(o,"conditional") for o in outcomes if o["condition"]==c]) for c in sorted({r["condition"] for r in rows})},
                 fallback_prefix=sum(len(r["prefix_fallback"]) for r in rows),
                 fallback_at_draft=sum(r["public"]["round"] in r["prefix_fallback"] for r in rows),
                 fallback_continuations=sum(len(p["fallback"]) for r in rows for p in r["paths"].values()))
    return stats


def native():
    rows = read(OUT/"native-advisor-paths.json.gz")
    all_values = {}
    results = {}
    for m in ("conditional", "generic"):
        y, gates, scores = [], [], []
        delta4 = []
        for r in rows:
            d = r["proposals"][m]
            diff = np.asarray(r["paths"][d["id"]]["profit"]) - r["paths"]["agent_draft"]["profit"]
            y.append(float(diff @ (.99**np.arange(len(diff)))))
            delta4.append(float(diff[:4] @ (.99**np.arange(4))))
            gates.append(d["accepted"])
            scores.append(min(d["gate"]))
        all_values[m] = np.array(y)*gates
        results[m] = dict(**selection(y,gates,scores), first_four_keep=ci(np.array(delta4)*gates),
                          steps_minmax=[min(r["proposals"][m]["steps"] for r in rows),max(r["proposals"][m]["steps"] for r in rows)])
    results["gated_conditional_minus_generic"] = ci(all_values["conditional"]-all_values["generic"])
    return results


def main(export):
    result = {}
    for phase, start, n in (("test",93000,80),("real",94000,12)):
        for market in ("econ","native"):
            path = OUT/f"{phase}-{market}-paths.json.gz"
            if export:
                rows = [read(RAW/f"{phase}-{market}-{s}.json.gz") for s in range(start,start+n)]
                for r in rows:
                    assert r["decision"] == read(RAW/f"sealed-{market}-{r['seed']}.json")
                save(path,rows)
            rows = read(path)
            assert [r["seed"] for r in rows] == list(range(start,start+n))
            result[phase+"-"+market] = summarize(rows)
    result["native-advisor"] = native()
    save(OUT/"summary.json",result)


if __name__ == "__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--export", action="store_true")
    main(parser.parse_args().export)
