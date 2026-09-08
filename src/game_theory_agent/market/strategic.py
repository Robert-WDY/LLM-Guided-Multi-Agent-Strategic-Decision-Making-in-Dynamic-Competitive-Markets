"""Versioned strategic-market overlay for lifecycle and market-power research."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import replace

from game_theory_agent.market.models import (
    CompanyAction,
    CompanyLifecycleState,
    CompanyOperatingStatus,
    CompanyState,
    ConcentrationRegime,
    MarketSnapshot,
    StrategicMarketState,
    ThresholdCooperationProjectState,
    ThresholdProjectStatus,
)


PPM = 1_000_000


def _clip(value: int, low: int = 0, high: int = PPM) -> int:
    return min(max(value, low), high)


def _ratio(numerator: int, denominator: int) -> int:
    if denominator <= 0:
        return 0
    return (numerator + denominator // 2) // denominator


def _ppm_mul(left: int, right: int) -> int:
    return _ratio(left * right, PPM)


def _normalized_shares(
    companies: Sequence[CompanyState], active_ids: set[str]
) -> dict[str, int]:
    raw = {
        company.company_id: max(0, company.commercial.market_share_ppm)
        for company in companies
        if company.company_id in active_ids
    }
    total = sum(raw.values())
    if not raw:
        return {}
    if total <= 0:
        ids = sorted(raw)
        base, remainder = divmod(PPM, len(ids))
        return {
            company_id: base + (1 if index < remainder else 0)
            for index, company_id in enumerate(ids)
        }
    exact = {
        company_id: value * PPM / total for company_id, value in raw.items()
    }
    result = {company_id: int(value) for company_id, value in exact.items()}
    remainder = PPM - sum(result.values())
    order = sorted(
        result,
        key=lambda company_id: (
            -(exact[company_id] - result[company_id]),
            company_id,
        ),
    )
    for company_id in order[:remainder]:
        result[company_id] += 1
    return result


def _structure(
    companies: Sequence[CompanyState],
    lifecycle: Sequence[CompanyLifecycleState],
    config: Mapping[str, object],
) -> tuple[int, int, ConcentrationRegime, str | None, int]:
    active_ids = {
        item.company_id
        for item in lifecycle
        if item.status is not CompanyOperatingStatus.EXITED
    }
    count = len(active_ids)
    shares = _normalized_shares(companies, active_ids)
    if not shares:
        return 0, 0, ConcentrationRegime.NO_ACTIVE_MARKET, None, 0
    hhi = _ratio(sum(value * value for value in shares.values()), PPM)
    dominant_id, dominant_share = min(
        shares.items(), key=lambda item: (-item[1], item[0])
    )
    if count == 1:
        regime = ConcentrationRegime.MONOPOLY
    elif dominant_share >= int(config["dominance_share_ppm"]):
        regime = ConcentrationRegime.DOMINANT
    elif hhi >= int(config["hhi_concentrated_ppm"]):
        regime = ConcentrationRegime.CONCENTRATED
    elif hhi >= int(config["hhi_moderate_ppm"]):
        regime = ConcentrationRegime.MODERATE
    else:
        regime = ConcentrationRegime.COMPETITIVE
    return count, hhi, regime, dominant_id, dominant_share


def initial_strategic_market_state(
    *,
    company_ids: Sequence[str],
    companies: Sequence[CompanyState],
    config: Mapping[str, object],
    threshold_project_enabled: bool = False,
    mutual_aid_enabled: bool = False,
    price_coordination_enabled: bool = False,
) -> StrategicMarketState:
    lifecycle = tuple(
        CompanyLifecycleState(company_id=company_id)
        for company_id in company_ids
    )
    count, hhi, regime, dominant_id, dominant_share = _structure(
        companies, lifecycle, config
    )
    project_cfg = config.get("threshold_project")
    project = None
    if threshold_project_enabled:
        if not isinstance(project_cfg, Mapping):
            raise ValueError("threshold project configuration is missing")
        project = ThresholdCooperationProjectState(
            project_id=str(project_cfg["project_id"]),
            status=ThresholdProjectStatus.ACTIVE,
            required_total_contribution_cents=int(
                project_cfg["required_total_contribution_cents"]
            ),
            accumulated_total_contribution_cents=0,
            contribution_by_company_cents=tuple(
                (company_id, 0) for company_id in company_ids
            ),
            deadline_round=int(project_cfg["deadline_round"]),
            failure_refund_rate_ppm=int(
                project_cfg["failure_refund_rate_ppm"]
            ),
            last_contribution_by_company_cents=tuple(
                (company_id, 0) for company_id in company_ids
            ),
            last_refund_by_company_cents=tuple(
                (company_id, 0) for company_id in company_ids
            ),
        )
    return StrategicMarketState(
        protocol_version=str(config["protocol_version"]),
        company_lifecycle=lifecycle,
        active_company_count=count,
        hhi_ppm=hhi,
        concentration_regime=regime,
        dominant_company_id=dominant_id,
        dominant_share_ppm=dominant_share,
        threshold_project=project,
        mutual_aid_enabled=mutual_aid_enabled,
        price_coordination_enabled=price_coordination_enabled,
        coordination_credibility_by_company_ppm=(
            tuple(
                (company_id, int(config["price_coordination"]["initial_credibility_ppm"]))
                for company_id in company_ids
            )
            if price_coordination_enabled
            else ()
        ),
    )


def settle_threshold_project(
    *,
    previous: ThresholdCooperationProjectState | None,
    settled_round: int,
    company_ids: Sequence[str],
    actions: Mapping[str, CompanyAction],
    config: Mapping[str, object],
) -> tuple[
    ThresholdCooperationProjectState | None,
    dict[str, int],
    bool,
]:
    """Settle one provision-point contribution round.

    Returns the next state, failure refunds, and whether success occurred now.
    Contributions after success/failure are forbidden by validation and ignored
    defensively here.
    """

    if previous is None:
        return None, {company_id: 0 for company_id in company_ids}, False
    zeros = {company_id: 0 for company_id in company_ids}
    if previous.status is not ThresholdProjectStatus.ACTIVE:
        return replace(
            previous,
            last_contribution_by_company_cents=tuple(sorted(zeros.items())),
            last_refund_by_company_cents=tuple(sorted(zeros.items())),
        ), zeros, False

    project_cfg = config.get("threshold_project")
    if not isinstance(project_cfg, Mapping):
        raise ValueError("threshold project configuration is missing")
    round_contributions = {
        company_id: max(
            0,
            int(actions[company_id].threshold_project_contribution_cents or 0),
        )
        for company_id in company_ids
    }
    cumulative = dict(previous.contribution_by_company_cents)
    for company_id, value in round_contributions.items():
        cumulative[company_id] = cumulative.get(company_id, 0) + value
    total = sum(cumulative.values())
    succeeded = total >= previous.required_total_contribution_cents
    failed = not succeeded and settled_round >= previous.deadline_round
    refunds = dict(zeros)
    if succeeded:
        status = ThresholdProjectStatus.SUCCEEDED
        success_round = settled_round
        failure_round = None
        protection = int(project_cfg["public_protection_bonus_ppm"])
        supply_reduction = int(project_cfg["supply_cost_reduction_ppm"])
    elif failed:
        status = ThresholdProjectStatus.FAILED
        success_round = None
        failure_round = settled_round
        protection = 0
        supply_reduction = 0
        refunds = {
            company_id: _ppm_mul(
                cumulative[company_id], previous.failure_refund_rate_ppm
            )
            for company_id in company_ids
        }
    else:
        status = ThresholdProjectStatus.ACTIVE
        success_round = None
        failure_round = None
        protection = 0
        supply_reduction = 0
    return (
        ThresholdCooperationProjectState(
            project_id=previous.project_id,
            status=status,
            required_total_contribution_cents=(
                previous.required_total_contribution_cents
            ),
            accumulated_total_contribution_cents=total,
            contribution_by_company_cents=tuple(sorted(cumulative.items())),
            deadline_round=previous.deadline_round,
            success_round=success_round,
            failure_round=failure_round,
            failure_refund_rate_ppm=previous.failure_refund_rate_ppm,
            public_protection_bonus_ppm=protection,
            supply_cost_reduction_ppm=supply_reduction,
            last_contribution_by_company_cents=tuple(
                sorted(round_contributions.items())
            ),
            last_refund_by_company_cents=tuple(sorted(refunds.items())),
        ),
        refunds,
        succeeded,
    )


def apply_threshold_project_refunds(
    companies: Sequence[CompanyState], refunds: Mapping[str, int]
) -> tuple[CompanyState, ...]:
    result = []
    for company in companies:
        refund = max(0, int(refunds.get(company.company_id, 0)))
        if refund == 0:
            result.append(company)
            continue
        result.append(
            replace(
                company,
                financial=replace(
                    company.financial,
                    cash_balance_cents=(
                        company.financial.cash_balance_cents + refund
                    ),
                    round_profit_cents=(
                        company.financial.round_profit_cents + refund
                    ),
                    cumulative_profit_cents=(
                        company.financial.cumulative_profit_cents + refund
                    ),
                ),
            )
        )
    return tuple(result)


def advance_strategic_market_state(
    *,
    previous: StrategicMarketState,
    settled_round: int,
    companies: Sequence[CompanyState],
    actions: Mapping[str, CompanyAction],
    market: MarketSnapshot,
    config: Mapping[str, object],
    fulfillment_cost_per_order_cents: int,
) -> tuple[StrategicMarketState, tuple[str, ...]]:
    """Advance lifecycle and public market-power metrics after settlement."""

    by_company = {company.company_id: company for company in companies}
    distress_cash = int(config["distress_cash_threshold_cents"])
    immediate_exit_cash = int(config["immediate_exit_cash_threshold_cents"])
    rounds_to_exit = int(config["distress_rounds_to_exit"])
    lifecycle: list[CompanyLifecycleState] = []
    newly_exited: list[str] = []
    for prior in previous.company_lifecycle:
        company = by_company[prior.company_id]
        if prior.status is CompanyOperatingStatus.EXITED:
            lifecycle.append(prior)
            continue
        exhausted = company.financial.cash_balance_cents <= immediate_exit_cash
        distressed = (
            company.financial.cash_balance_cents <= distress_cash
            and company.financial.round_profit_cents < 0
        )
        streak = prior.distress_streak + 1 if distressed else 0
        exits = exhausted or streak >= rounds_to_exit
        if exits:
            reason = "cash_exhausted" if exhausted else "persistent_distress"
            lifecycle.append(
                CompanyLifecycleState(
                    company_id=prior.company_id,
                    status=CompanyOperatingStatus.EXITED,
                    distress_streak=streak,
                    first_distress_round=(
                        prior.first_distress_round
                        if prior.first_distress_round is not None
                        else settled_round
                    ),
                    exit_round=settled_round,
                    exit_reason=reason,
                )
            )
            newly_exited.append(prior.company_id)
        elif distressed:
            lifecycle.append(
                CompanyLifecycleState(
                    company_id=prior.company_id,
                    status=CompanyOperatingStatus.DISTRESSED,
                    distress_streak=streak,
                    first_distress_round=(
                        prior.first_distress_round
                        if prior.first_distress_round is not None
                        else settled_round
                    ),
                )
            )
        else:
            lifecycle.append(CompanyLifecycleState(company_id=prior.company_id))

    lifecycle_tuple = tuple(lifecycle)
    count, hhi, regime, dominant_id, dominant_share = _structure(
        companies, lifecycle_tuple, config
    )
    previously_active = set(previous.active_company_ids)
    predatory: list[str] = []
    predatory_intensities: list[int] = []
    consumer_surplus = 0
    producer_welfare = 0
    for company in companies:
        if company.company_id not in previously_active:
            continue
        action = actions[company.company_id]
        sustainable_cost = (
            company.operations.actual_unit_cost_cents
            + fulfillment_cost_per_order_cents
        )
        if action.price_cents < sustainable_cost:
            predatory.append(company.company_id)
            predatory_intensities.append(
                _clip(
                    _ratio(
                        (sustainable_cost - action.price_cents) * PPM,
                        max(1, sustainable_cost),
                    )
                )
            )
        consumer_surplus += company.commercial.sales_orders * max(
            0, market.price_anchor_cents - action.price_cents
        )
        producer_welfare += company.financial.round_profit_cents

    price_war_intensity = (
        sum(predatory_intensities) // len(predatory_intensities)
        if predatory_intensities
        else 0
    )
    moderate = int(config["hhi_moderate_ppm"])
    concentration_signal = _clip(
        _ratio(max(0, hhi - moderate) * PPM, max(1, PPM - moderate))
    )
    markup_signal = _clip(
        _ratio(
            max(0, market.average_paid_price_cents - market.price_anchor_cents)
            * PPM,
            max(1, market.price_anchor_cents),
        )
    )
    market_power_markup = _ppm_mul(markup_signal, dominant_share)
    pressure = _clip(
        _ppm_mul(
            previous.regulatory_pressure_ppm,
            int(config["regulatory_pressure_retention_ppm"]),
        )
        + _ppm_mul(
            concentration_signal,
            int(config["concentration_pressure_weight_ppm"]),
        )
        + _ppm_mul(
            price_war_intensity,
            int(config["predatory_pressure_weight_ppm"]),
        )
        + _ppm_mul(
            market_power_markup,
            int(config["monopoly_markup_pressure_weight_ppm"]),
        )
    )
    stockout_cost = (
        market.lost_after_stockout_orders
        * int(config["stockout_social_cost_per_order_cents"])
    )
    social_welfare = consumer_surplus + producer_welfare - stockout_cost
    return (
        StrategicMarketState(
            protocol_version=previous.protocol_version,
            company_lifecycle=lifecycle_tuple,
            active_company_count=count,
            hhi_ppm=hhi,
            concentration_regime=regime,
            dominant_company_id=dominant_id,
            dominant_share_ppm=dominant_share,
            predatory_pricing_company_ids=tuple(sorted(predatory)),
            price_war_intensity_ppm=price_war_intensity,
            market_power_markup_ppm=market_power_markup,
            regulatory_pressure_ppm=pressure,
            consumer_surplus_proxy_cents=consumer_surplus,
            producer_welfare_cents=producer_welfare,
            social_welfare_proxy_cents=social_welfare,
            cumulative_consumer_surplus_proxy_cents=(
                previous.cumulative_consumer_surplus_proxy_cents
                + consumer_surplus
            ),
            cumulative_producer_welfare_cents=(
                previous.cumulative_producer_welfare_cents + producer_welfare
            ),
            cumulative_social_welfare_proxy_cents=(
                previous.cumulative_social_welfare_proxy_cents + social_welfare
            ),
            threshold_project=previous.threshold_project,
            mutual_aid_enabled=previous.mutual_aid_enabled,
            last_mutual_aid_transfers=(),
            price_coordination_enabled=previous.price_coordination_enabled,
            coordination_credibility_by_company_ppm=(
                previous.coordination_credibility_by_company_ppm
            ),
            last_price_coordination_outcomes=(),
        ),
        tuple(sorted(newly_exited)),
    )


def liquidate_newly_exited(
    companies: Sequence[CompanyState],
    newly_exited: Sequence[str],
    config: Mapping[str, object],
) -> tuple[CompanyState, ...]:
    exited = set(newly_exited)
    recovery = int(config["liquidation_recovery_ppm"])
    result: list[CompanyState] = []
    for company in companies:
        if company.company_id not in exited:
            result.append(company)
            continue
        proceeds = _ppm_mul(
            company.financial.capacity_book_value_cents, recovery
        )
        result.append(
            replace(
                company,
                financial=replace(
                    company.financial,
                    cash_balance_cents=(
                        company.financial.cash_balance_cents + proceeds
                    ),
                    capacity_book_value_cents=0,
                ),
                operations=replace(
                    company.operations,
                    base_capacity_orders=0,
                    effective_capacity_orders=0,
                    financial_capacity_orders=0,
                    capacity_utilization_ppm=0,
                ),
                risk=replace(company.risk, active_incident=None),
            )
        )
    return tuple(result)


__all__ = [
    "apply_threshold_project_refunds",
    "advance_strategic_market_state",
    "initial_strategic_market_state",
    "liquidate_newly_exited",
    "settle_threshold_project",
]
