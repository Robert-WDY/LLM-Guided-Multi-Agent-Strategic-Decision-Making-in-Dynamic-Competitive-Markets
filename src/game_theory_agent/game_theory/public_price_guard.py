"""Screen a proposed price using comparable completed pricing experiments."""

from statistics import median
from math import comb, log
from functools import lru_cache
import numpy as np


@lru_cache(maxsize=2048)
def loss_upper(losses, n):
    """One-sided 98.75% Clopper-Pearson bound for distinct sampled markets."""
    if losses == n:
        return 1.0
    lo, hi = 0.0, 1.0
    for _ in range(52):
        p = (lo + hi) / 2
        cdf = sum(comb(n, k) * p**k * (1 - p) ** (n - k) for k in range(losses + 1))
        if cdf > 0.0125:
            lo = p
        else:
            hi = p
    return hi


def public_features(prices, draft, round_number):
    """Rows are pairs of own posted price and the public median rival quote."""
    own, peer = zip(*prices[-8:])
    latest = median(peer[-4:])
    previous = median(peer[:4])
    return [
        log(draft / latest),
        log(draft / own[-1]),
        log(latest / previous),
        log(own[-1] / own[0]),
        (round_number - 35) / 100,
    ]


class PublicPriceGuard:
    """Training outcomes are historical paired interventions; live input is price-only."""

    def __init__(self, examples, *, historical_value_only=False):
        self.historical_value_only = historical_value_only
        self.banks = {}
        for market, rows in examples.items():
            for action in (1, 2):
                subset = [r for r in rows if r["action"] == action]
                if subset:
                    self.banks[market, action] = (
                        np.asarray([r["features"] for r in subset]),
                        subset,
                    )

    def assess(self, market, prices, draft, round_number, action):
        return self.explain(market, prices, draft, round_number, action)["action"]

    def explain(self, market, prices, draft, round_number, action):
        if not action:
            return dict(action=0, reason="original_plan")
        if len(prices) < 8 or (market, action) not in self.banks:
            return dict(action=0, reason="missing_history")
        x = np.asarray(public_features(prices, draft, round_number))
        points, examples = self.banks[market, action]
        distance = np.max(np.abs(points - x), axis=1)
        neighbors = []
        seeds = set()
        for index in np.argsort(distance, kind="stable"):
            if distance[index] > 0.16:
                break
            row = examples[index]
            if row["seed"] not in seeds:
                seeds.add(row["seed"])
                neighbors.append(row)
            if len(neighbors) == (16 if self.historical_value_only else 80):
                break
        if len(neighbors) < 16:
            return dict(action=0, reason="insufficient_comparables", n=len(neighbors))
        gains = np.array([r["gain"] for r in neighbors])
        forecast = gains.mean() + 0.25 * np.minimum(gains, 0).mean()
        extra_exits = sum(r["exit_delta"] for r in neighbors)
        volume_loss = sum(r["sales_delta"] < 0 for r in neighbors)
        evidence = {}
        if not self.historical_value_only:
            losses = int((gains < -1e-7).sum())
            evidence = dict(
                losses=losses, risk_upper=loss_upper(losses, len(neighbors))
            )
        reason = "accepted"
        if forecast <= 0:
            reason = "terminal_profit"
        elif extra_exits > 0:
            reason = "exit_risk"
        elif action == 2 and volume_loss >= 0.75 * len(neighbors):
            reason = "volume_risk"
        elif evidence and evidence["risk_upper"] > 0.1:
            reason = "loss_frequency"
        return dict(
            **evidence,
            action=action if reason == "accepted" else 0,
            reason=reason,
            n=len(neighbors),
            risk_adjusted_gain=float(forecast),
            historical_mean_gain=float(gains.mean()),
            historical_seeds=[row["seed"] for row in neighbors],
            extra_exits=extra_exits,
            volume_losses=volume_loss,
        )
