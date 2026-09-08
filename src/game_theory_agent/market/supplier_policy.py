"""Independent, lagged supplier pricing. No model, buyer secrets or future RNG."""
from dataclasses import replace
from typing import Mapping

from .models import SupplierQuoteDecision, SupplierState, SupplyChainState

PPM = 1_000_000
POLICY_VERSION = "bounded-utilization-pricing-v1.0.0"


def pricing_enabled(config: Mapping) -> bool:
    policy = config.get("autonomous_pricing")
    return isinstance(policy, Mapping) and policy.get("enabled") is True


def price_bounds(supplier: SupplierState, config: Mapping) -> tuple[int, int]:
    policy = config["autonomous_pricing"]
    base = config["suppliers"][supplier.supplier_id]["unit_price_cents"]
    floor = (supplier.unit_cost_cents * (PPM + policy["min_markup_ppm"]) + PPM - 1) // PPM
    ceiling = base * policy["max_base_price_ppm"] // PPM
    return floor, ceiling


def decide_quote(*, supplier: SupplierState, config: Mapping, observed_round: int) -> SupplierQuoteDecision:
    """Only own settled utilization and fixed policy parameters enter the decision."""
    policy = config["autonomous_pricing"]
    capacity, sales = supplier.available_capacity_orders, supplier.round_sales_orders
    utilization = sales * PPM // capacity if capacity else 0
    multiplier, reason = PPM, "target_band"
    if capacity == 0:
        reason = "unavailable"
    elif utilization >= policy["raise_threshold_ppm"]:
        multiplier, reason = PPM + policy["step_ppm"], "high_utilization"
    elif utilization <= policy["lower_threshold_ppm"]:
        multiplier, reason = PPM - policy["step_ppm"], "low_utilization"
    floor, ceiling = price_bounds(supplier, config)
    quote = max(floor, min(ceiling, (supplier.unit_price_cents * multiplier + PPM // 2) // PPM))
    return SupplierQuoteDecision(
        policy_version=POLICY_VERSION, observed_round=observed_round,
        applies_round=observed_round + 1, previous_price_cents=supplier.unit_price_cents,
        quoted_price_cents=quote, observed_sales_orders=sales,
        observed_capacity_orders=capacity, utilization_ppm=utilization, reason=reason,
    )


def advance_supplier_quotes(*, settled: SupplyChainState, config: Mapping, observed_round: int) -> SupplyChainState:
    if not pricing_enabled(config):
        return settled
    # Each decision reads the same settled snapshot. Iteration order cannot
    # expose another supplier's newly chosen quote.
    decisions = {s.supplier_id: decide_quote(supplier=s, config=config, observed_round=observed_round)
                 for s in settled.suppliers}
    return replace(settled, suppliers=tuple(
        replace(s, unit_price_cents=decisions[s.supplier_id].quoted_price_cents,
                quote_decision=decisions[s.supplier_id]) for s in settled.suppliers
    ))
