"""Adapters for the native market, the author's four-firm demand, and a held-out service."""

from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "scripts/deployment"), str(ROOT / "scripts/pricing_feedback")]
from study import (
    ALPHAS,
    MarketEnv,
    apply_native_price,
    company_action,
    environment,
    make,
    settle,
    step,
)
from prices import number


# Per firm: executed price, units, profit, revenue, active at start, active at end.
class World:
    def __init__(self, m, seed):
        self.market = m
        self.seed = seed
        self.state = None
        self.t = 1
        if m == "native":
            self.config, self.env, self.state = make(
                ("normal", "recession", "tight_supply")[seed % 3], 4, seed, 60
            )
            b = self.config.mapping("action", "bounds")["price_cents"]
            self.low, self.high = b["min"], b["max"]
            self.ref = self.state.companies[0].commercial.price_cents
        else:
            self.kw, self.cost, _, _, self.high = environment(ALPHAS[seed % 3], 4)
            self.low, self.ref = 0.15 * self.high, 0.54 * self.high
        self.scale = 100 if m == "native" else 1

    def clone(self):
        b = object.__new__(World)
        b.__dict__.update(self.__dict__)
        if self.state is not None:
            b.env = MarketEnv(self.config)
            b.env.load_state(self.state)
        return b

    def settle(self, prices):
        if self.market == "native":
            s = self.state
            active = s.strategic_market.active_company_ids
            acts = {cid: company_action(self.config, s, cid, "balanced") for cid in s.company_ids}
            for i, cid in enumerate(s.company_ids):
                if cid in active:
                    acts[cid] = apply_native_price(self.config, s, cid, acts[cid], prices[i])
            self.state = step(self.env, s, acts)
            after = self.state
            rows = [
                [
                    x.commercial.price_cents,
                    x.commercial.sales_orders,
                    x.financial.round_profit_cents / 100,
                    x.financial.round_revenue_cents / 100,
                    int(x.company_id in active),
                    int(x.company_id in after.strategic_market.active_company_ids),
                ]
                for x in after.companies
            ]
            welfare = after.welfare_accounting.round_consumer_surplus_cents / 100
            stockout = after.market.lost_after_stockout_orders
        else:
            p = np.round(np.clip(prices, self.low, self.high), 2)
            alpha = self.kw["alpha"][0]
            kw = dict(self.kw)
            when = 15 + number(self.seed, "demand") % 25
            if self.t >= when and self.seed % 4 in (1, 2):
                kw["a"] = ((2.35 if self.seed % 4 == 1 else 1.65),) * 4
            qty, prof = settle(kw, self.cost, tuple(p))
            offered = np.asarray(qty)
            if self.market == "appointments":
                capacities = np.array(
                    [12 + number(self.seed, self.t // 4, i, "slots") % 18 for i in range(4)]
                )
                qty = np.minimum(offered, capacities)
                prof = qty * (p / alpha - 1.0) - capacities * 0.18
            revenue = np.asarray(qty) * p / alpha
            rows = [
                [float(p[i]), float(qty[i]), float(prof[i]), float(revenue[i]), 1, 1]
                for i in range(4)
            ]
            # Unconstrained logit surplus only applies in the EconEvals adapter.
            welfare = (
                100
                * kw["mu"]
                * np.log1p(np.exp((np.asarray(kw["a"]) - p / alpha) / kw["mu"]).sum())
                if self.market == "econ"
                else None
            )
            stockout = float(np.sum(offered - np.asarray(qty)))
        self.t += 1
        return rows, [welfare, stockout]
