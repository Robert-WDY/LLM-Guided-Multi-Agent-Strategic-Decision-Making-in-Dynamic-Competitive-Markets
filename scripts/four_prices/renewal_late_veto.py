"""Can a monetary warning about later contracts improve frozen retail proposals?"""

import argparse
from hashlib import sha256
import time

import numpy as np
from cause_value import metrics
from export_panel import pack
from game_theory_agent.game_theory.panel_advisor import Forest
from game_theory_agent.game_theory.quote_panel import public_state
from game_theory_agent.game_theory.renewal_review import PublicRenewalReview
from prices import GROUPS, number
from renewal_precision import profile
from renewal_risk import forest, opportunity
from renewal_risk_study import historical
from renewal_trials import OUT
from run import read, save
from saturation_report import ci

LABELS = ("late", "terminal")
ARMS = ("keep", "cut", "raise")


def seeds():
    return [profile("econ", g, i)[0] for i in range(24, 120) for g in GROUPS]


def losses(r):
    t, j = r["probe"], r["masks"]["1"][0]
    p = r["paths"]
    b = np.asarray(p["keep"]["rows"])[t - 1 :, j, 2]
    out = {k: [] for k in LABELS}
    for a in ("audit_cut", "audit_raise"):
        y = (np.asarray(p[a]["rows"])[t - 1 :, j, 2] - b) * 0.99 ** np.arange(len(b))
        net = float(y.sum())
        out["late"].append(max(-float(y[12:].sum()), 0) if net < -1e-7 else 0.0)
        out["terminal"].append(max(-net, 0.0))
    return out


def freeze(pristine):
    if pristine:
        assert not any(
            (OUT / "precision-pretrade" / f"{s}.json.gz").exists() for s in seeds()
        )
    train = [
        read(p) for phase in ("fit", "calibrate") for p in historical(phase, "econ")
    ]
    x = np.asarray([opportunity(r)[0] for r in train])
    y = [losses(r) for r in train]
    calibrate = [read(p) for p in historical("test", "econ")]
    base = PublicRenewalReview(OUT / "public-econ.npz")
    selected = []
    for r in calibrate:
        t, j = r["probe"], r["masks"]["1"][0]
        p = r["paths"]["keep"]
        d = next(v["draft"][j] for v in p["decisions"] if v["t"] == t)
        h = [[c[0] if c[4] else None for c in row] for row in p["rows"][: t - 1]]
        a = ARMS.index(base.review(h, j, t, d)["candidate"])
        if a:
            selected.append((opportunity(r)[0], a - 1))
    cfg = {}
    data = {}
    for key in LABELS:
        model = forest(x, np.asarray([v[key] for v in y]))
        vals = model.predict([v[0] for v in selected])
        scores = [v[a] for v, (_, a) in zip(vals, selected)]
        cfg[key] = dict(limit=float(np.quantile(scores, 0.75)), n=len(scores))
        data.update(pack(key, model.regressor_))
        data[key + "_scale"] = model.transformer_.scale_
        data[key + "_mean"] = model.transformer_.mean_
    np.savez_compressed(OUT / "late-veto.npz", **data)
    save(OUT / "late-veto-freeze.json", cfg)


def observe(live):
    cfg = read(OUT / "late-veto-freeze.json")
    with np.load(OUT / "late-veto.npz", allow_pickle=False) as d:
        models = {k: (Forest(d, k), d[k + "_scale"], d[k + "_mean"]) for k in LABELS}
    dest = OUT / "late-pretrade"
    dest.mkdir(exist_ok=True)
    pending = set(seeds())
    while pending:
        ready = 0
        for seed in sorted(pending):
            prefix = OUT / "precision-pretrade" / f"{seed}.json.gz"
            if not prefix.exists():
                continue
            raw = OUT / "precision" / prefix.name
            f = dest / prefix.name
            first = not f.exists()
            if live and first and raw.exists():
                raise RuntimeError(f"Missed prospective scoring for {seed}")
            v = read(prefix)
            assert v["seed"] == seed
            a = ARMS.index(v["review"]["candidate"])
            x = public_state(
                np.asarray(v["panel"], float), v["seat"], v["round"], v["draft"]
            )
            score = {
                k: float((model.predict(x) * scale + mean)[a - 1]) if a else None
                for k, (model, scale, mean) in models.items()
            }
            choices = {k: int(bool(a and score[k] <= cfg[k]["limit"])) for k in LABELS}
            choices.update(
                base=int(bool(a)),
                random=int(bool(a and number("late-veto", seed) % 4 != 0)),
                keep=0,
            )
            record = dict(
                seed=seed,
                prefix_sha256=sha256(prefix.read_bytes()).hexdigest(),
                base_arm=a,
                score=score,
                actions=choices,
            )
            if first:
                save(f, record)
            assert read(f) == record
            if live and first and raw.exists():
                raise RuntimeError(f"Settlement raced prospective scoring for {seed}")
            pending.remove(seed)
            ready += 1
        if pending:
            if not live:
                raise FileNotFoundError(
                    "Missing public prefixes in confirmation archive"
                )
            if not ready:
                time.sleep(1)


def measure():
    outcomes = {r["seed"]: r for r in read(OUT / "precision-decisions.json.gz")}
    rows = []
    active = []
    for seed in seeds():
        p = read(OUT / "late-pretrade" / f"{seed}.json.gz")
        r = outcomes[seed]
        assert r["actual_action"] == p["base_arm"]
        rows.append(
            dict(
                seed=seed, gains=r["gains"], changed=r["changed"], actions=p["actions"]
            )
        )
        if p["base_arm"]:
            active.append(p)

    def profit(r, k):
        return r["gains"][r["actions"][k]]

    summary = {}
    for key in ("base", "late", "terminal", "random", "keep"):
        avoided = [
            r["gains"][1]
            for r in rows
            if r["actions"]["base"] and not r["actions"][key]
        ]
        summary[key] = dict(
            **metrics(rows, key, 5),
            vetoed=len(avoided),
            canceled_wins=sum(x > 1e-7 for x in avoided),
            canceled_losses=sum(x < -1e-7 for x in avoided),
            avoided_loss_amount=sum(max(0, -x) for x in avoided),
            missed_win_amount=sum(max(0, x) for x in avoided),
            difference_from_base=ci([profit(r, key) - profit(r, "base") for r in rows]),
            difference_from_terminal=ci(
                [profit(r, key) - profit(r, "terminal") for r in rows]
            ),
        )
    assert len(rows) == len(seeds()) == 288 and len(outcomes) == 360
    for key, result in summary.items():
        assert all(r["actions"][key] <= r["actions"]["base"] for r in rows)
        recovered = result["avoided_loss_amount"] - result["missed_win_amount"]
        assert np.isclose(recovered / len(rows), result["difference_from_base"][0])
    n = summary["late"]["vetoed"]
    for key in LABELS:
        # Exactly the same number of vetoes, breaking score ties by the preassigned seed.
        removed = {
            p["seed"]
            for p in sorted(active, key=lambda p: (-p["score"][key], p["seed"]))[:n]
        }
        for r in rows:
            r["actions"][key + "_matched"] = int(
                bool(r["actions"]["base"] and r["seed"] not in removed)
            )
    contrast = ci(
        [profit(r, "late_matched") - profit(r, "terminal_matched") for r in rows]
    )
    save(
        OUT / "late-veto-results.json",
        dict(
            policies=summary, matched_veto_count=n, matched_late_minus_terminal=contrast
        ),
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "phase", choices=("freeze", "seal", "observe", "live", "measure")
    )
    phase = parser.parse_args().phase
    if phase in ("freeze", "seal"):
        freeze(phase == "seal")
    elif phase in ("observe", "live"):
        observe(phase == "live")
    else:
        measure()
