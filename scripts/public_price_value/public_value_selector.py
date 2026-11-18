"""Frozen public-price terminal valuation, with a lagged opponent-response ablation."""
import hashlib
from functools import lru_cache
from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import ExtraTreesRegressor

from trials import ROOT, RAW, OUT, SPLITS, read, save
from game_theory_agent.game_theory import reliable_response as response

ARMS = ("down", "up")
MODES = ("generic", "conditional")


def features(public, mode):
    if set(public) != {"prices", "draft", "round"}:
        raise ValueError("public contract requires posted prices, draft and clock only")
    h = np.asarray(public["prices"], dtype=float)
    if h.shape != (public["round"] - 1, 2) or not np.isfinite(h).all() or np.min(h) <= 0:
        raise ValueError("invalid posted-price history")
    p, r = h[-24:].T
    last = p[-1]
    draft = public["draft"]
    samples = [{"features": (response.bucket(p[i-1]/p[i-2]-1), 0, 0),
                "change": r[i]/r[i-1]-1} for i in range(2, len(r))]
    model = response.fit(samples)
    common = [public["round"], last, r[-1]/last, draft/last, p.mean()/last,
              r.mean()/last, p.std()/last, r.std()/last,
              *(p[-4:]/last), *(r[-4:]/last)]
    rows = []
    for multiplier in (.96, 1.04):
        candidate = draft * multiplier
        peer = r[-1]
        # Simultaneous period cannot react to the pending bid. The previous move is public.
        forecasts = []
        for move in (p[-1]/p[-2]-1, candidate/last-1, 0., 0.):
            prediction = response.predict(model, (response.bucket(move), 0, 0), mode)
            peer *= 1 + np.dot(prediction["changes"], prediction["probabilities"])
            forecasts.append(peer/last)
        rows.append([*common, multiplier, candidate/last, *forecasts])
    return rows


def values(row):
    t = row["public"]["round"]
    p = np.array([row["paths"][a]["settled"] for a in ("keep", *ARMS)])
    assert p.shape == (3, 61-t, 9)
    unit = 100 if row["market"] == "native" else 1
    return (p[1:, :, 3] - p[0, :, 3]) @ (.99**np.arange(61-t)) / unit


def cases(phase, market):
    start, n = SPLITS[phase]
    return [read(RAW / f"{phase}-{market}-{s}.json.gz") for s in range(start, start+n)]


@lru_cache(maxsize=4)
def load(market, mode):
    return joblib.load(OUT / f"{market}-{mode}.joblib")


def choose(market, public):
    result = {}
    for mode in MODES:
        model, threshold = load(market, mode)
        scores = model.predict(features(public, mode))
        i = int(scores.argmax())
        result[mode] = dict(arm=ARMS[i], score=float(scores[i]), use=bool(scores[i] > threshold))
    return result


def checksum(path):
    data = path.read_bytes()
    if path.suffix in (".md", ".py"):
        data = data.replace(b"\r\n", b"\n")
    return hashlib.sha256(data).hexdigest()


def frozen_files():
    paths = [Path(__file__), Path(__file__).with_name("trials.py"),
             ROOT/"experiment-specs/public-price-value-v1/PLAN.md",
             ROOT/"src/game_theory_agent/game_theory/reliable_response.py",
             *sorted(OUT.glob("*.joblib"))]
    return {p.relative_to(ROOT).as_posix(): checksum(p) for p in paths}


def verify():
    if read(OUT/"freeze.json") != frozen_files():
        raise ValueError("frozen estimator or design changed")


def train():
    if (OUT/"freeze.json").exists():
        verify()
        return
    for market in ("econ", "native"):
        training, calibration = cases("train", market), cases("cal", market)
        for mode in MODES:
            x = [v for row in training for v in features(row["public"], mode)]
            y = np.concatenate([values(row) for row in training])
            model = ExtraTreesRegressor(n_estimators=96, min_samples_leaf=8, random_state=5511, n_jobs=1)
            model.fit(x, y)
            calibration_x = [v for row in calibration for v in features(row["public"], mode)]
            best_scores = model.predict(calibration_x).reshape(-1, 2).max(axis=1)
            threshold = max(0., float(np.median(best_scores)))
            joblib.dump((model, threshold), OUT / f"{market}-{mode}.joblib", compress=3)
    save(OUT/"freeze.json", frozen_files())


if __name__ == "__main__":
    train()
