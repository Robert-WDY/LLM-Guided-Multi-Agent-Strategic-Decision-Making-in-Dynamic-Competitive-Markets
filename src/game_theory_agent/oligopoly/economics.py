"""Logit-demand Bertrand oligopoly (Calvano, Calzolari, Denicolo & Pastorello 2020).

Demand for firm i with n differentiated products and an outside good:

    q_i = exp((a_i - p_i) / mu) / (sum_j exp((a_j - p_j) / mu) + exp(a0 / mu))

Profit is (p_i - c_i) * q_i. Consumer surplus of the unit mass of consumers is
the logit inclusive value mu * log(sum_j exp((a_j - p_j)/mu) + exp(a0/mu)), up to
an additive constant that cancels in all comparisons we make.

``scale`` multiplies every monetary quantity (prices, costs, profits, surplus)
and leaves quantities unchanged. Agents only ever see scaled numbers, so a
randomised scale prevents a model from recalling the textbook Calvano prices.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property

import numpy as np
from scipy.optimize import brentq, minimize


@dataclass(frozen=True)
class LogitMarket:
    n: int = 2
    a: float = 2.0
    c: float = 1.0
    a0: float = 0.0
    mu: float = 0.25
    scale: float = 1.0

    def __post_init__(self) -> None:
        if self.n < 2:
            raise ValueError("need at least two firms")
        if self.mu <= 0 or self.scale <= 0:
            raise ValueError("mu and scale must be positive")

    # ---- primitives in unscaled units -------------------------------------
    def _shares(self, p: np.ndarray) -> np.ndarray:
        u = (self.a - p) / self.mu
        top = max(float(u.max()), self.a0 / self.mu)
        e = np.exp(u - top)
        return e / (e.sum() + np.exp(self.a0 / self.mu - top))

    def _profits(self, p: np.ndarray) -> np.ndarray:
        return (p - self.c) * self._shares(p)

    def _cs(self, p: np.ndarray) -> float:
        u = np.append((self.a - p) / self.mu, self.a0 / self.mu)
        top = float(u.max())
        return float(self.mu * (top + np.log(np.exp(u - top).sum())))

    # ---- public API in scaled units ---------------------------------------
    def outcome(self, prices: list[float] | np.ndarray) -> dict[str, list[float] | float]:
        p = np.asarray(prices, dtype=float) / self.scale
        if p.shape != (self.n,):
            raise ValueError(f"expected {self.n} prices")
        q = self._shares(p)
        pi = (p - self.c) * q
        cs = self._cs(p)
        return {
            "quantities": q.tolist(),
            "profits": (pi * self.scale).tolist(),
            "consumer_surplus": cs * self.scale,
            "total_welfare": (cs + float(pi.sum())) * self.scale,
        }

    def best_response(self, rival_prices: list[float] | np.ndarray) -> float:
        """Static best response of a firm to the other n-1 prices (scaled)."""

        rivals = np.asarray(rival_prices, dtype=float) / self.scale
        if rivals.shape != (self.n - 1,):
            raise ValueError(f"expected {self.n - 1} rival prices")
        other = np.exp((self.a - rivals) / self.mu).sum() + np.exp(self.a0 / self.mu)

        # FOC of (p-c) e^{(a-p)/mu} / (e^{(a-p)/mu} + other): p - c = mu / (1 - q)
        def foc(p: float) -> float:
            own = np.exp((self.a - p) / self.mu)
            q = own / (own + other)
            return p - self.c - self.mu / (1.0 - q)

        return float(brentq(foc, self.c, self.c + 50 * self.mu + 10)) * self.scale

    @cached_property
    def nash_price(self) -> float:
        """Symmetric one-shot Nash price (scaled)."""

        p = self.c + self.mu
        for _ in range(500):
            new = self.best_response([p * self.scale] * (self.n - 1)) / self.scale
            if abs(new - p) < 1e-12:
                break
            p = new
        return p * self.scale

    @cached_property
    def monopoly_price(self) -> float:
        """Symmetric joint-profit-maximising price (scaled)."""

        res = minimize(
            lambda x: -float(self._profits(np.full(self.n, x[0])).sum()),
            x0=[self.c + 2 * self.mu],
            method="Nelder-Mead",
            options={"xatol": 1e-12, "fatol": 1e-14, "maxiter": 5000},
        )
        return float(res.x[0]) * self.scale

    @cached_property
    def nash_profit(self) -> float:
        return float(self.outcome([self.nash_price] * self.n)["profits"][0])

    @cached_property
    def monopoly_profit(self) -> float:
        return float(self.outcome([self.monopoly_price] * self.n)["profits"][0])

    def collusion_index(self, mean_profit_per_firm: float) -> float:
        """Delta = (pi - pi_N) / (pi_M - pi_N); 0 at Nash, 1 at joint monopoly."""

        return (mean_profit_per_firm - self.nash_profit) / (
            self.monopoly_profit - self.nash_profit
        )

    def price_index(self, mean_price: float) -> float:
        """Same normalisation on prices: 0 at Nash price, 1 at monopoly price."""

        return (mean_price - self.nash_price) / (self.monopoly_price - self.nash_price)

    def benchmarks(self) -> dict[str, float]:
        nash = self.outcome([self.nash_price] * self.n)
        mono = self.outcome([self.monopoly_price] * self.n)
        return {
            "nash_price": self.nash_price,
            "monopoly_price": self.monopoly_price,
            "nash_profit": self.nash_profit,
            "monopoly_profit": self.monopoly_profit,
            "nash_welfare": float(nash["total_welfare"]),
            "monopoly_welfare": float(mono["total_welfare"]),
            "nash_consumer_surplus": float(nash["consumer_surplus"]),
            "monopoly_consumer_surplus": float(mono["consumer_surplus"]),
            "marginal_cost": self.c * self.scale,
        }
