"""Public-capacity procurement heuristics, with explicit information boundaries."""
from dataclasses import asdict, dataclass, replace
from typing import Literal

from .models import CompanyAction, MarketState
from .protocols import sha256_hash

PPM = 1_000_000
VERSION = "public-capacity-procurement-v1.0.0"


@dataclass(frozen=True)
class PublicSupplierQuote:
    supplier_id: str
    unit_price_cents: int
    available_capacity_orders: int


@dataclass(frozen=True)
class ProcurementView:
    round: int
    own_requested_orders: int
    active_buyer_count: int
    quotes: tuple[PublicSupplierQuote, PublicSupplierQuote]
    previous_primary_share_ppm: int


@dataclass(frozen=True)
class ProcurementPlan:
    policy_version: str
    mode: str
    view: ProcurementView
    primary_share_ppm: int
    expected_fulfilled_orders: int
    expected_input_cost_cents: int
    retained_previous: bool

    def to_dict(self):
        result = asdict(self)
        result["information_boundary"] = "Own requirements and previous procurement; public quotes, available capacity and active buyer count only. Equal-share capacity is a heuristic, not a guarantee."
        result["plan_hash"] = sha256_hash(result)
        return result


def procurement_view(state: MarketState, company_id: str) -> ProcurementView:
    if state.supply_chain is None or len(state.supply_chain.suppliers) != 2:
        raise ValueError("public capacity policy requires exactly two suppliers")
    quotes = tuple(PublicSupplierQuote(s.supplier_id, s.unit_price_cents, s.available_capacity_orders)
                   for s in sorted(state.supply_chain.suppliers, key=lambda s: s.supplier_id))
    previous = 500_000
    for outcome in state.supply_chain.last_procurement_outcomes:
        if outcome.company_id != company_id:
            continue
        selected_share = outcome.primary_supplier_share_ppm if outcome.backup_supplier_id else PPM
        previous = selected_share if outcome.primary_supplier_id == quotes[0].supplier_id else PPM - selected_share
    active = len(state.strategic_market.active_company_ids) if state.strategic_market else len(state.company_ids)
    return ProcurementView(state.round, state.company(company_id).operations.base_capacity_orders,
                           active, quotes, previous)


def select_procurement(view: ProcurementView, mode: Literal["capacity", "gradual"] = "capacity") -> ProcurementPlan:
    if mode not in {"capacity", "gradual"}:
        raise ValueError("unknown procurement mode")
    if (view.active_buyer_count < 1 or view.own_requested_orders < 0
        or not 0 <= view.previous_primary_share_ppm <= PPM or len(view.quotes) != 2
        or view.quotes[0].supplier_id == view.quotes[1].supplier_id
        or any(q.unit_price_cents < 0 or q.available_capacity_orders < 0 for q in view.quotes)):
        raise ValueError("invalid procurement view")
    previous = view.previous_primary_share_ppm
    step = PPM if mode == "capacity" else 250_000
    choices = sorted({previous, *range(0, PPM + 1, 50_000)})
    choices = [share for share in choices if abs(share - previous) <= step]

    def evaluate(share):
        requested = (view.own_requested_orders * share + PPM // 2) // PPM
        filled = [min(requested, view.quotes[0].available_capacity_orders // view.active_buyer_count),
                  min(view.own_requested_orders - requested, view.quotes[1].available_capacity_orders // view.active_buyer_count)]
        return sum(filled), sum(f * q.unit_price_cents for f, q in zip(filled, view.quotes))

    # Prioritize expected supply continuity, then procurement expense. The
    # previous allocation and canonical share break ties deterministically.
    best = min(choices, key=lambda share: (-evaluate(share)[0], evaluate(share)[1], abs(share - previous), share))
    filled, cost = evaluate(best)
    old_filled, old_cost = evaluate(previous)
    # A 1% band avoids reacting to negligible modelled changes. Compare unit
    # expense by cross multiplication when expected delivered volumes differ.
    retained = (old_filled * 100 >= filled * 99 and
                (filled == 0 or (old_filled > 0 and old_cost * filled * 100 <= cost * old_filled * 101)))
    if retained:
        best, filled, cost = previous, old_filled, old_cost
    return ProcurementPlan(VERSION, mode, view, best, filled, cost, retained)


def apply_procurement(action: CompanyAction, plan: ProcurementPlan) -> CompanyAction:
    first, second = plan.view.quotes
    return replace(action, primary_supplier_id=first.supplier_id,
                   backup_supplier_id=second.supplier_id,
                   primary_supplier_share_ppm=plan.primary_share_ppm,
                   strategy_summary=action.strategy_summary + f"; procurement:{plan.mode}:{plan.to_dict()['plan_hash']}")
