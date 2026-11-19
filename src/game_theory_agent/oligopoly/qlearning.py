"""Tabular Q-learning baseline of Calvano et al. (2020), memory of one period.

Price grid: m equally spaced points on
[p_N - xi (p_M - p_N), p_M + xi (p_M - p_N)], state = last period's price
profile, epsilon-greedy with epsilon_t = exp(-beta t). A session stops when the
greedy strategy of every firm has been unchanged for ``stable`` periods.
"""

from __future__ import annotations

import math
import random
from typing import Any

import numpy as np

from game_theory_agent.oligopoly.economics import LogitMarket


def run_session(
    market: LogitMarket,
    *,
    seed: int,
    m: int = 15,
    xi: float = 0.1,
    alpha: float = 0.15,
    beta: float = 4e-6,
    delta: float = 0.95,
    stable: int = 100_000,
    max_steps: int = 3_000_000,
) -> dict[str, Any]:
    if market.n != 2:
        raise ValueError("this baseline implements the two-firm case")
    rng = random.Random(seed)
    pn, pm = market.nash_price, market.monopoly_price
    grid = np.linspace(pn - xi * (pm - pn), pm + xi * (pm - pn), m)
    payoff = np.zeros((m, m, 2))
    for a in range(m):
        for b in range(m):
            payoff[a, b] = market.outcome([grid[a], grid[b]])["profits"]
    # Q initialised at the discounted payoff of the rival randomising uniformly.
    q = [np.zeros((m * m, m)) for _ in range(2)]
    for i in range(2):
        own = payoff[:, :, 0].mean(axis=1) if i == 0 else payoff[:, :, 1].mean(axis=0)
        q[i][:] = own / (1 - delta)
    state = rng.randrange(m * m)
    greedy = [q[i].argmax(axis=1) for i in range(2)]
    unchanged = 0
    t = 0
    for t in range(1, max_steps + 1):
        eps = math.exp(-beta * t)
        acts = []
        for i in range(2):
            if rng.random() < eps:
                acts.append(rng.randrange(m))
            else:
                acts.append(int(greedy[i][state]))
        nxt = acts[0] * m + acts[1]
        changed = False
        for i in range(2):
            r = payoff[acts[0], acts[1], i]
            row = q[i][state]
            row[acts[i]] = (1 - alpha) * row[acts[i]] + alpha * (r + delta * q[i][nxt].max())
            g = int(row.argmax())
            if g != greedy[i][state]:
                greedy[i][state] = g
                changed = True
        unchanged = 0 if changed else unchanged + 1
        state = nxt
        if unchanged >= stable:
            break
    # Play the learned greedy strategies from the final state until a cycle.
    seen: dict[int, int] = {}
    path = []
    s = state
    while s not in seen:
        seen[s] = len(path)
        a0, a1 = int(greedy[0][s]), int(greedy[1][s])
        path.append((a0, a1))
        s = a0 * m + a1
    cycle = path[seen[s] :]
    profits = np.mean([payoff[a, b].mean() for a, b in cycle])
    prices = np.mean([(grid[a] + grid[b]) / 2 for a, b in cycle])
    return {
        "seed": seed,
        "steps": t,
        "converged": unchanged >= stable,
        "cycle_length": len(cycle),
        "mean_price": float(prices),
        "price_index": market.price_index(float(prices)),
        "delta": market.collusion_index(float(profits)),
    }
