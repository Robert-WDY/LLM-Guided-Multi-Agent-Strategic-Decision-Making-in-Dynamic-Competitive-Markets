"""Screen a proposed price using comparable completed pricing experiments."""

from statistics import median
from math import log
import numpy as np


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

    def __init__(self, examples):
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
            if len(neighbors) == 16:
                break
        if len(neighbors) < 16:
            return dict(action=0, reason="insufficient_comparables", n=len(neighbors))
        gains = np.array([r["gain"] for r in neighbors])
        forecast = gains.mean() + 0.25 * np.minimum(gains, 0).mean()
        extra_exits = sum(r["exit_delta"] for r in neighbors)
        volume_loss = sum(r["sales_delta"] < 0 for r in neighbors)
        reason = "accepted"
        if forecast <= 0:
            reason = "terminal_profit"
        elif extra_exits > 0:
            reason = "exit_risk"
        elif action == 2 and volume_loss >= 12:
            reason = "volume_risk"
        return dict(
            action=action if reason == "accepted" else 0,
            reason=reason,
            n=len(neighbors),
            risk_adjusted_gain=float(forecast),
            historical_mean_gain=float(gains.mean()),
            historical_seeds=[row["seed"] for row in neighbors],
            extra_exits=extra_exits,
            volume_losses=volume_loss,
        )
