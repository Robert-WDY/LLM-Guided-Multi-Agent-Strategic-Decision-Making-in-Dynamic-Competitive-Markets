"""Paired joint policies. Fork only when joint posted price contracts diverge."""

import argparse
from concurrent.futures import ProcessPoolExecutor
import gzip
import itertools
import json
from pathlib import Path

import numpy as np

from prices import ALL_LIVE, FACTORS, GROUPS, MARKETS, OUT, RAW, key, label, number, sample
from world import World, apply_native_price, company_action
from independent import frozen_actions
from game_theory_agent.game_theory.cycle_pricing import decide, failed_trial_adjustment
from common import read, save

REVIEW = tuple(range(19, 56, 4))
PAIRS = tuple(itertools.combinations(range(4), 2))


def design(m, g, i):
    n = 24 if m == "appointments" else 48
    seed = 180000 + 1000 * MARKETS.index(m) + n * GROUPS.index(g) + i
    block = i // 12
    perm = sorted(range(4), key=lambda j: number(m, g, block, j, "seat"))
    sole = perm[i % 4]
    two = tuple(perm[j] for j in PAIRS[i % 6])
    roles = (
        [0] * 4
        if g == "balanced"
        else [1] * 4 if g == "unseen" else [number(seed, j, "persona") % 4 for j in range(4)]
    )
    masks = {1: (sole,), 2: two, 4: tuple(range(4))}
    focal = {k: v[number(seed, k, "audited-seat") % len(v)] for k, v in masks.items()}
    return (
        seed,
        masks,
        focal,
        roles,
        23 + 4 * (number(seed, "audit-time") % 6),
        (ALL_LIVE or i >= n - 12),
    )


def variants():
    return ["keep", "greedy", "once", "once_direct"] + [
        f"{a}{k}" for k in (1, 2, 4) for a in ("inf", "dir", "skip")
    ]


def arms(v, t, choices, masks, focal, probe):
    use = [0] * 4
    if v in ("keep", "greedy") or t not in REVIEW:
        return use
    if v.startswith("once"):
        if t == probe:
            j = masks[1][0]
            use[j] = choices[j]["direct" if v == "once_direct" else "inferred"]
        return use
    k = int(v[-1])
    for j in masks[k]:
        if not (v.startswith("skip") and t == probe and j == focal[k]):
            use[j] = choices[j]["direct" if v.startswith("dir") else "inferred"]
    return use


def effective(w, j, draft):
    if w.market != "native":
        return float(np.round(np.clip(draft, w.low, w.high), 2))
    cid = w.state.company_ids[j]
    a = company_action(w.config, w.state, cid, "balanced")
    if cid in w.state.strategic_market.active_company_ids:
        a = apply_native_price(w.config, w.state, cid, a, draft)
    return a.price_cents


def simulation(
    m,
    g,
    i,
    offline=None,
    diagnostic=None,
    allow_new=False,
    allocation=None,
    run_only=None,
    numeric_quote=None,
    study_design=design,
    price_guard=None,
    feedback_rollback=False,
    expire_advice=False,
    diagnostic_rule="inf",
):
    if expire_advice and feedback_rollback:
        raise ValueError("Select one renewal policy")
    seed, masks, focal, roles, probe, live = study_design(m, g, i)
    tape = {} if offline is None else dict(offline)
    if allocation is not None:
        masks = allocation
    profiles = variants() if run_only is None else run_only
    if diagnostic is not None:
        coverage, seat, base_targets = diagnostic
        profiles = ["natural", "fixed_rivals", *[f"omit-{t}" for t in REVIEW]]
    logs = {v: [] for v in profiles}
    paths = {}
    world = World(m, seed)
    nodes = [
        (world, [[] for _ in range(4)], None, [], [], [v for v in profiles if v != "greedy"]),
        (world.clone(), [[] for _ in range(4)], None, [], [], ["greedy"]),
    ]
    if "greedy" not in profiles:
        nodes = nodes[:1]
    ends = list(range(1, 7, 2)) + list(range(7, 60, 4))
    for t in ends:
        prepared = []
        points = []
        for w, h, old, records, market, vs in nodes:
            assert w.t == t
            terms = []
            indices = []
            for j in range(4):
                term = decide(t, 61 - t, w.ref, w.low, w.high, h[j], old[j] if old else None)
                if feedback_rollback and old:
                    rollback = failed_trial_adjustment(h[j], old[j].get("trial_delta", 0))
                    term["rollback"] = rollback
                    term["draft"] = float(np.clip(term["draft"] - rollback, w.low, w.high))
                active = (
                    w.state is None
                    or w.state.company_ids[j] in w.state.strategic_market.active_company_ids
                )
                ix = None
                if t > 6 and active and numeric_quote is not None:
                    low, high = effective(w, j, w.low), effective(w, j, w.high)
                    term["draft"], ix = numeric_quote(
                        m,
                        roles[j],
                        h[j],
                        term["draft"],
                        low,
                        high,
                        seed,
                        j,
                        t,
                        tape,
                        offline is not None,
                    )
                elif t > 6 and active:
                    k = key(m, roles[j], h[j])
                    if offline is not None and not allow_new and live and vs != ["greedy"]:
                        assert f"{j}:{t}:{label(k)}" in tape
                    ix = sample(k, seed, j, t, live, vs == ["greedy"], tape)
                    term["draft"] = float(np.clip(term["draft"] * FACTORS[ix], w.low, w.high))
                terms.append(term)
                indices.append(ix)
            offset = len(points)
            if t in REVIEW and vs != ["greedy"]:
                points.extend(
                    dict(prices=[row[:2] for row in h[j]], round=t, draft=terms[j]["draft"])
                    for j in range(4)
                )
            prepared.append((w, h, terms, indices, records, market, vs, offset))
        scores = frozen_actions("native" if m == "native" else "econ", points) if points else []
        nodes = []
        for w, h, terms, indices, records, market, vs, offset in prepared:
            drafts = [term["draft"] for term in terms]
            legal = [effective(w, j, draft) for j, draft in enumerate(drafts)]
            brackets = {}
            for v in vs:
                rule = v if diagnostic is None else f"{diagnostic_rule}{coverage}"
                guarded = rule.startswith("safe_")
                rule = rule.removeprefix("safe_")
                use = arms(rule, t, scores[offset : offset + 4], masks, focal, probe)
                if guarded:
                    if price_guard is None:
                        raise ValueError("A frozen public price guard is required")
                    use = [
                        price_guard.assess("native" if m == "native" else "econ",
                                           [row[:2] for row in h[j]], drafts[j], t, action)
                        for j, action in enumerate(use)
                    ]
                if feedback_rollback:
                    use = [0 if term.get("rollback", 0) else a for term, a in zip(terms, use)]
                if diagnostic is not None and v == f"omit-{t}":
                    use[seat] = 0
                targets = tuple(
                    float(np.clip(d * (1, 0.96, 1.04)[a], w.low, w.high))
                    for d, a in zip(drafts, use)
                )
                if diagnostic is not None and v == "fixed_rivals":
                    targets = tuple(
                        targets[j] if j == seat else base_targets[t][j] for j in range(4)
                    )
                logs[v].append(
                    dict(
                        t=t,
                        draft=drafts,
                        original=legal,
                        selected=use,
                        requested=targets,
                        model_actions=indices,
                        **({"rollback": [x.get("rollback", 0) for x in terms]} if feedback_rollback else {}),
                    )
                )
                brackets.setdefault(targets, []).append(v)
            children = [
                (targets, subset, w if i == len(brackets) - 1 else w.clone())
                for i, (targets, subset) in enumerate(brackets.items())
            ]
            for targets, subset, b in children:
                next_terms = terms
                if feedback_rollback:
                    next_terms = [dict(term, trial_delta=effective(b,j,targets[j])-legal[j])
                                  for j,term in enumerate(terms)]
                elif expire_advice:
                    next_terms = [
                        dict(term, renewal_anchor=legal[j])
                        if effective(b, j, targets[j]) != legal[j] else term
                        for j, term in enumerate(terms)
                    ]
                h2 = [list(a) for a in h]
                rec = list(records)
                mr = list(market)
                while b.t <= terms[0]["end"]:
                    rows, context = b.settle(targets)
                    p = [r[0] for r in rows]
                    for j, row in enumerate(rows):
                        rivals = [p[k] for k in range(4) if k != j and rows[k][4]]
                        peer = float(np.median(rivals)) if rivals else p[j]
                        h2[j].append([p[j], peer, row[1], row[2]])
                    rec.append(rows)
                    mr.append(context)
                if b.t == 61:
                    for v in subset:
                        paths[v] = dict(rows=rec, market=mr, decisions=logs[v])
                else:
                    nodes.append((b, h2, next_terms, rec, mr, subset))
    assert set(paths) == set(profiles) and all(len(v["rows"]) == 60 for v in paths.values())
    return dict(
        seed=seed,
        market=m,
        group=g,
        roles=roles,
        masks=masks,
        focal=focal,
        probe=probe,
        live=live,
        tape=tape,
        paths=paths,
    )


def job(task):
    m, g, i = task
    seed = design(m, g, i)[0]
    f = RAW / f"{seed}.json.gz"
    if not f.exists():
        base = RAW.parent / "four-prices-v1" / f.name
        if ALL_LIVE and i >= (24 if m == "appointments" else 48) - 12:
            import shutil

            assert read(base)["live"]
            shutil.copyfile(base, f)
        else:
            save(f, simulation(m, g, i))
    return seed


def jobs():
    return [
        (m, g, i)
        for i in range(48)
        for m in MARKETS
        for g in GROUPS
        if i < (24 if m == "appointments" else 48)
    ]


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--workers", type=int, default=8)
    args = p.parse_args()
    with ProcessPoolExecutor(args.workers) as pool:
        for n, _ in enumerate(pool.map(job, jobs()), 1):
            if not n % 24:
                print(n, flush=True)
