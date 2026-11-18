"""Screen a proposed price using comparable completed pricing experiments."""

from statistics import median
from math import log
import numpy as np


def public_features(prices, draft, round_number):
    """Rows are pairs of own posted price and the public median rival quote."""
    own, peer = zip(*prices[-8:])
    latest = median(peer[-4:])
    previous = median(peer[:4])
    return [log(draft / latest), log(draft / own[-1]), log(latest / previous),
            log(own[-1] / own[0]), (round_number - 35) / 100]


class PublicPriceGuard:
    """Training outcomes are historical paired interventions; live input is price-only."""

    def __init__(self, examples):
        self.banks = {}
        for market, rows in examples.items():
            for action in (1, 2):
                subset = [r for r in rows if r["action"] == action]
                if subset:
                    self.banks[market, action] = (
                        np.asarray([r["features"] for r in subset]), subset
                    )

    def assess(self, market, prices, draft, round_number, action):
        if not action or len(prices) < 8 or (market, action) not in self.banks:
            return 0
        x = np.asarray(public_features(prices, draft, round_number))
        points, examples = self.banks[market, action]
        distance = np.max(np.abs(points - x), axis=1)
        neighbors = []
        seeds = set()
        for index in np.argsort(distance, kind="stable"):
            if distance[index] > .16:
                break
            row = examples[index]
            if row["seed"] not in seeds:
                seeds.add(row["seed"])
                neighbors.append(row)
            if len(neighbors) == 16:
                break
        if len(neighbors) < 16:
            return 0
        gains = np.array([r["gain"] for r in neighbors])
        forecast = gains.mean() + .25 * np.minimum(gains, 0).mean()
        extra_exits = sum(r["exit_delta"] for r in neighbors)
        volume_loss = sum(r["sales_delta"] < 0 for r in neighbors)
        if forecast <= 0 or extra_exits > 0 or (action == 2 and volume_loss >= 12):
            return 0
        return action

