"""Public-observation forecast construction and Agent-visible rollout advice."""

from __future__ import annotations

from dataclasses import fields, replace
from typing import Any, Literal, Mapping

from game_theory_agent.agents.observation import ObservationBuilder
from game_theory_agent.agents.personas import PersonaProfile
from game_theory_agent.belief import BeliefState, compute_belief_hash
from game_theory_agent.gameplay import build_rule_action, build_terminal_rankings
from game_theory_agent.market import (
    BrandState,
    CommercialState,
    CompanyHistory,
    CompanyAction,
    IncidentResponse,
    IncidentResponseMode,
    CompanyState,
    MarketConfig,
    MarketEnv,
    MarketEvent,
    MarketState,
    Persona,
    RiskSignal,
    SharedResilienceState,
    StrategicMarketState,
)
from game_theory_agent.market.protocols import sha256_hash, state_hash
from game_theory_agent.market.models import SupplyChainState, WelfareAccountingState, GovernmentState, ConsumerDecisionAudit
from game_theory_agent.opponent import (
    OpponentModelState,
    compute_opponent_model_hash,
)

from .public_contracts import (
    PublicForecastStateRecord,
    PublicRolloutCandidateSummary,
    PublicStrategicAdvice,
    compute_public_advice_hash,
)
from .contracts import (
    CandidateEconomicAction,
    StrategicActionCandidate,
    StrategicReliabilityPlan,
    compute_reliability_plan_hash,
)
from .rollout import AuthoritativeMarketRolloutEvaluator, generate_candidate_actions
from .pareto_planner import (
    PROMOTED_PARETO_SPEC,
    PROMOTION_EVIDENCE_SHA256,
    select_pareto_decision,
)
from .reliable_planner import (
    FINAL_MARKET_GATE_POLICY,
    build_abstention_gate,
    build_final_market_gate,
    build_reliability_gate,
)
from .marginal_investment import (
    MarginalInvestmentPlan,
    build_marginal_investment_plan,
)
from .paired_gate import PAIRED_MARKET_GATE_POLICY, build_paired_market_gate


def _clip(value: int, low: int, high: int) -> int:
    return min(max(value, low), high)


def _public_decision_payload(
    *,
    config: MarketConfig,
    observation: Mapping[str, Any],
    company_id: str,
    persona_profile: PersonaProfile,
    belief_state: BeliefState | None,
    opponent_model: OpponentModelState | None,
) -> dict[str, Any]:
    """Return the complete and deliberately narrow legal planning input.

    The authoritative ``state_hash`` and outer ``observation_hash`` are omitted:
    both bind an observation to TrueState but encode hidden-state changes.  They
    must not select forecast random scenarios or alter strategic rankings.
    """

    return {
        "protocol": "public-strategic-decision-input-v1.0.0",
        "config_sha256": config.config_sha256,
        "company_id": company_id,
        "information_mode": observation.get("information_mode"),
        "visibility_policy_version": observation.get(
            "visibility_policy_version"
        ),
        "public_state": observation.get("public_state"),
        "private_state": observation.get("private_state"),
        "belief_state": (
            belief_state.model_dump(mode="json")
            if belief_state is not None
            else None
        ),
        "opponent_model_state": (
            opponent_model.model_dump(mode="json")
            if opponent_model is not None
            else None
        ),
        "persona_profile": persona_profile.model_dump(mode="json"),
    }


def _public_scenario_payload(
    *,
    config: MarketConfig,
    observation: Mapping[str, Any],
    company_id: str,
) -> dict[str, Any]:
    """Bind rollout randomness to observed economics, not the treatment.

    Belief, opponent-model and Persona inputs may change rankings.  They must
    not also select a different random market forecast when treatments are
    compared on the same legal observation.
    """

    return {
        "protocol": "public-strategic-common-scenario-v1.0.0",
        "config_sha256": config.config_sha256,
        "company_id": company_id,
        "information_mode": observation.get("information_mode"),
        "visibility_policy_version": observation.get(
            "visibility_policy_version"
        ),
        "public_state": observation.get("public_state"),
        "private_state": observation.get("private_state"),
    }


def _event_from_public(
    config: MarketConfig, payload: Mapping[str, Any]
) -> MarketEvent:
    event_type = str(payload["event_type"])
    severity = str(payload["severity"])
    impact = config.mapping(
        "events", "definitions", event_type, "severity", severity
    )
    return MarketEvent(
        event_id=str(payload["event_id"]),
        event_type=event_type,
        severity=severity,
        started_round=int(payload["started_round"]),
        remaining_rounds=int(payload["remaining_rounds"]),
        demand_multiplier_ppm=int(impact["demand_multiplier_ppm"]),
        supply_cost_multiplier_ppm=int(impact["supply_cost_multiplier_ppm"]),
        capacity_multiplier_ppm=int(impact["capacity_multiplier_ppm"]),
        advertising_multiplier_ppm=int(impact["advertising_multiplier_ppm"]),
        service_penalty_ppm=int(impact["service_penalty_ppm"]),
        reputation_penalty_ppm=int(impact["reputation_penalty_ppm"]),
    )


def _estimated_opponent(
    base: CompanyState,
    public: Mapping[str, Any],
    *,
    mutual_aid_received_orders: int | None = None,
    mutual_aid_provided_orders: int | None = None,
    regulatory_fine_cents: int | None = None,
) -> CompanyState:
    sales = int(public["sales_orders"])
    capacity = max(1, base.operations.effective_capacity_orders, sales)
    commercial = CommercialState(
        price_cents=int(public["price_cents"]),
        market_share_ppm=int(public["market_share_ppm"]),
        potential_demand_orders=sales,
        sales_orders=sales,
        mutual_aid_fulfilled_orders=mutual_aid_received_orders,
        mutual_aid_provided_orders=mutual_aid_provided_orders,
    )
    brand = BrandState(
        brand_awareness_ppm=base.brand.brand_awareness_ppm,
        service_quality_ppm=base.brand.service_quality_ppm,
        reputation_ppm=int(public["reputation_ppm"]),
    )
    operations = replace(
        base.operations,
        base_capacity_orders=max(base.operations.base_capacity_orders, sales),
        effective_capacity_orders=capacity,
        capacity_utilization_ppm=min(1_000_000, sales * 1_000_000 // capacity),
    )
    return replace(
        base,
        persona=Persona.NONE,
        financial=replace(
            base.financial,
            round_regulatory_fine_cents=regulatory_fine_cents,
        ),
        commercial=commercial,
        operations=operations,
        brand=brand,
        history=CompanyHistory(recent_market_share_ppm=(commercial.market_share_ppm,)),
    )


def build_public_forecast_state(
    *,
    config: MarketConfig,
    observation: Mapping[str, Any],
    company_id: str,
    persona_profile: PersonaProfile,
    belief_state: BeliefState | Mapping[str, Any] | None = None,
    opponent_model: OpponentModelState | Mapping[str, Any] | None = None,
) -> tuple[MarketState, PublicForecastStateRecord]:
    """Construct a simulation state without reading authoritative TrueState."""

    if observation.get("information_mode") != "public":
        raise ValueError("public rollout requires a public-information observation")
    public_state = observation.get("public_state")
    private_state = observation.get("private_state")
    if not isinstance(public_state, Mapping) or not isinstance(
        private_state, Mapping
    ):
        raise ValueError("public rollout requires public_state and private_state")
    if private_state.get("company_id") != company_id:
        raise ValueError("private state belongs to another company")
    parsed_belief = (
        belief_state
        if isinstance(belief_state, BeliefState)
        else BeliefState.model_validate(
            belief_state if belief_state is not None else observation.get("belief_state")
        )
        if (belief_state is not None or observation.get("belief_state") is not None)
        else None
    )
    parsed_model = (
        opponent_model
        if isinstance(opponent_model, OpponentModelState)
        else OpponentModelState.model_validate(
            opponent_model
            if opponent_model is not None
            else observation.get("opponent_model_state")
        )
        if (
            opponent_model is not None
            or observation.get("opponent_model_state") is not None
        )
        else None
    )
    episode_id = str(public_state["episode_id"])
    round_number = int(public_state["round"])
    state_version = int(public_state["state_version"])
    if round_number != state_version + 1:
        raise ValueError("forecast requires the current pre-decision state")
    if parsed_belief is not None and (
        parsed_belief.episode_id != episode_id
        or parsed_belief.observer_company_id != company_id
        or parsed_belief.prediction_target_round != round_number
        or parsed_belief.state_version != state_version
    ):
        raise ValueError("belief state is not bound to this public observation")
    if parsed_model is not None and (
        parsed_model.episode_id != episode_id
        or parsed_model.observer_company_id != company_id
        or parsed_model.prediction_target_round != round_number
        or parsed_model.state_version != state_version
    ):
        raise ValueError("opponent model is not bound to this public observation")

    public_companies = public_state.get("companies")
    if not isinstance(public_companies, list) or len(public_companies) < 2:
        raise ValueError("public forecast requires at least two public companies")
    company_ids = tuple(str(item["company_id"]) for item in public_companies)
    if len(set(company_ids)) != len(company_ids) or company_id not in company_ids:
        raise ValueError("public company identities are invalid")
    own_company = CompanyState.from_dict(private_state["company"])
    own_public = next(
        item for item in public_companies if item["company_id"] == company_id
    )
    if ObservationBuilder.public_company(own_company) != dict(own_public):
        raise ValueError("own public and private company views disagree")

    decision_payload = _public_decision_payload(
        config=config,
        observation=observation,
        company_id=company_id,
        persona_profile=persona_profile,
        belief_state=parsed_belief,
        opponent_model=parsed_model,
    )
    input_hash = sha256_hash(decision_payload)
    scenario_hash = sha256_hash(
        _public_scenario_payload(
            config=config,
            observation=observation,
            company_id=company_id,
        )
    )
    forecast_seed = int(scenario_hash.rsplit(":", 1)[-1][-16:], 16)
    rounds_remaining = int(public_state["rounds_remaining"])
    max_rounds = state_version + rounds_remaining
    market_public = public_state.get("market")
    if not isinstance(market_public, Mapping):
        raise ValueError("public market state is missing")
    strategic_public = market_public.get("strategic_market")
    cooperation_modes: list[str] = []
    if public_state.get("shared_resilience") is not None:
        cooperation_modes.append("shared_resilience_v1")
    if isinstance(strategic_public, Mapping):
        if strategic_public.get("threshold_project") is not None:
            cooperation_modes.append("threshold_project_v1")
        if bool(strategic_public.get("mutual_aid", {}).get("enabled", False)):
            cooperation_modes.append("mutual_aid_v1")
        if bool(
            strategic_public.get("price_coordination", {}).get("enabled", False)
        ):
            cooperation_modes.append("price_coordination_v1")
    base = MarketEnv(config).reset(
        company_ids,
        episode_id=episode_id,
        episode_seed=forecast_seed,
        market_model=str(market_public["market_model_id"]),
        max_rounds=max_rounds,
        cooperation_modes=tuple(cooperation_modes),
    )
    market_fields = {item.name for item in fields(base.market)}
    projected_market = replace(
        base.market,
        **{
            key: value
            for key, value in market_public.items()
            if key in market_fields
        },
    )
    strategic_market = (
        StrategicMarketState.from_dict(strategic_public)
        if isinstance(strategic_public, Mapping)
        else None
    )
    mutual_received = {item: 0 for item in company_ids}
    mutual_provided = {item: 0 for item in company_ids}
    regulatory_fines = {item: 0 for item in company_ids}
    if strategic_market is not None:
        for transfer in strategic_market.last_mutual_aid_transfers:
            mutual_received[transfer.recipient_company_id] += transfer.fulfilled_orders
            mutual_provided[transfer.donor_company_id] += transfer.fulfilled_orders
        for outcome in strategic_market.last_price_coordination_outcomes:
            for item, value in outcome.fine_by_company_cents:
                regulatory_fines[item] += value
    public_by_id = {
        str(item["company_id"]): item for item in public_companies
    }
    base_by_id = {item.company_id: item for item in base.companies}
    companies = tuple(
        own_company
        if item == company_id
        else _estimated_opponent(
            base_by_id[item],
            public_by_id[item],
            mutual_aid_received_orders=(
                mutual_received[item]
                if strategic_market is not None
                and strategic_market.mutual_aid_enabled
                and state_version > 0
                else None
            ),
            mutual_aid_provided_orders=(
                mutual_provided[item]
                if strategic_market is not None
                and strategic_market.mutual_aid_enabled
                and state_version > 0
                else None
            ),
            regulatory_fine_cents=(
                regulatory_fines[item]
                if strategic_market is not None
                and strategic_market.price_coordination_enabled
                and state_version > 0
                else None
            ),
        )
        for item in company_ids
    )
    events = tuple(
        _event_from_public(config, item)
        for item in public_state.get("active_market_events", [])
    )
    signals = tuple(
        RiskSignal.from_dict(item) for item in public_state.get("risk_signals", [])
    )
    shared = (
        SharedResilienceState.from_dict(public_state["shared_resilience"])
        if public_state.get("shared_resilience") is not None
        else None
    )
    projected_government = GovernmentState.from_dict(market_public["government"]) if market_public.get("government") else None
    projected_consumers=tuple(ConsumerDecisionAudit.from_dict(d) for d in market_public.get("consumer_decisions",()))
    if config.data.get("supply_chain",{}).get("transaction_accounting"):
        from .forecast_accounting import forecast_accounting
        projected_supply, companies = forecast_accounting(config=config, public_supply=market_public["supply_chain"],
            companies=companies,company_id=company_id,strategic=strategic_market,state_version=state_version,terminal=bool(public_state["terminal"]),remaining_rounds=rounds_remaining)
        if projected_government and projected_government.last_decision:
            support=dict(projected_government.last_decision.support_by_company_cents)
            if projected_government.last_decision.strategic_policy:
                support={cid:min(cap,projected_government.round_matched_support_cents or 0) for cid,cap in support.items()}
            companies=tuple(replace(c,financial=replace(c.financial,round_government_support_cents=support.get(c.company_id,0))) if c.company_id!=company_id else c for c in companies)
        if projected_consumers:
            # Only aggregate refunds are public. Allocate the unobserved
            # remainder by opponent sales as an explicit historical prior.
            residual_refunds=sum(d.refund_cents for d in projected_consumers)-own_company.financial.round_incident_cost_cents
            weights={c.company_id:c.commercial.sales_orders for c in companies if c.company_id!=company_id}
            total=sum(weights.values()); refund_est={cid:residual_refunds*q//total if total else 0 for cid,q in weights.items()}
            remaining=residual_refunds-sum(refund_est.values())
            for cid in sorted(refund_est)[:remaining]:refund_est[cid]+=1
            estimated=[]
            for c in companies:
                if c.company_id!=company_id:
                    revenue=sum(dict(d.purchases_by_company).get(c.company_id,0)*dict(d.posted_prices_cents)[c.company_id] for d in projected_consumers)
                    revenue+=sum(t.total_transfer_fee_cents for t in strategic_market.last_mutual_aid_transfers if t.donor_company_id==c.company_id) if strategic_market else 0
                    c=replace(c,financial=replace(c.financial,round_revenue_cents=revenue,round_incident_cost_cents=refund_est[c.company_id]))
                estimated.append(c)
            companies=tuple(estimated)
    else:
        projected_supply=SupplyChainState.from_dict(market_public["supply_chain"]) if market_public.get("supply_chain") is not None else base.supply_chain
    unsealed = replace(
        base,
        round=round_number,
        rounds_remaining=rounds_remaining,
        state_version=state_version,
        terminal=bool(public_state["terminal"]),
        max_rounds=max_rounds,
        market=projected_market,
        risk_signals=signals,
        active_market_events=events,
        companies=companies,
        shared_resilience=shared,
        strategic_market=strategic_market,
        supply_chain=projected_supply,
        government=projected_government,
        consumer_decisions=projected_consumers,
        welfare_accounting=(WelfareAccountingState.from_dict(market_public["welfare_accounting"]) if market_public.get("welfare_accounting") is not None else base.welfare_accounting),
        last_joint_action=(),
        terminal_enterprise_values_cents=(),
        state_hash="",
    )
    forecast = replace(unsealed, state_hash=state_hash(unsealed.to_dict()))
    record = PublicForecastStateRecord(
        episode_id=episode_id,
        round=round_number,
        state_version=state_version,
        company_id=company_id,
        public_decision_input_hash=input_hash,
        forecast_state_hash=forecast.state_hash,
        forecast_seed=forecast_seed,
        uses_belief_state=parsed_belief is not None,
        uses_public_opponent_model=parsed_model is not None,
        assumptions=[
            "opponent cash, cost, capacity, brand assets, resilience and incidents use neutral config estimates",
            "public event type and severity map to versioned config effects",
            "forecast randomness is derived from legal decision inputs, not TrueState hash",
            "belief, opponent model and Persona treatments share common random forecast scenarios on the same observation",
            "the observing company's complete private state is used only for its own planning",
            *(["supplier costs and initial cash use declared config priors; invoices use public deliveries/quotes; hidden historical buyer budgets equal invoices; processing costs use config priors; aggregate refunds minus own known refunds are allocated by opponent sales"] if config.data.get("supply_chain",{}).get("transaction_accounting") else []),
            *(["private supplier requested quantities use the minimum excess consistent with realized investment, not the true hidden order book"] if config.data.get("supply_chain",{}).get("track_supplier_demand") else []),
            *(["v14 disclosed settled contracts, inventory and creditor reports are public; supplier cash/cost use declared initial/unit-cost priors; rejected bids and private order book are not reconstructed as facts"] if config.data.get("supply_chain",{}).get("strategic_policy") else []),
        ],
    )
    return forecast, record


def generate_public_overlay_candidates(
    config: MarketConfig,
    state: MarketState,
    company_id: str,
) -> tuple[StrategicActionCandidate, ...]:
    """Overlay one strategic change on a feasible deterministic operating plan.

    The Stage 6 oracle's original candidates intentionally isolate dimensions
    by setting other optional spending to zero.  Executing such a candidate as
    an online policy would silently erase normal advertising and service work.
    Public advice therefore uses the legal Rule policy as an operational floor
    and changes only the named strategic dimension.
    """

    baseline = build_rule_action(config, state, company_id)
    templates = generate_candidate_actions(config, state, company_id)
    env = MarketEnv(config)
    env.load_state(state)
    cash = state.company(company_id).financial.cash_balance_cents
    candidates: list[StrategicActionCandidate] = []
    seen: set[str] = set()

    for template in templates:
        source = template.action
        price = baseline.price_cents
        advertising = baseline.advertising_budget_cents
        service = baseline.service_budget_cents
        capacity = baseline.capacity_investment_cents
        resilience = baseline.resilience_budget_cents
        shared = baseline.shared_resilience_contribution_cents
        incident_response = baseline.incident_response

        if template.label in {
            "price_cut_small",
            "price_cut_large",
            "price_increase",
        }:
            price = source.price_cents
        elif template.label == "increase_advertising":
            advertising += source.advertising_budget_cents
        elif template.label == "increase_service":
            service += source.service_budget_cents
        elif template.label == "increase_capacity":
            capacity += source.capacity_investment_cents
        elif template.label == "increase_resilience":
            resilience += source.resilience_budget_cents
        elif template.label == "shared_resilience_contribution":
            shared = (shared or 0) + (
                source.shared_resilience_contribution_cents or 0
            )
        elif template.label == "repair_incident":
            incident_response = baseline.incident_response.__class__(
                IncidentResponseMode(source.incident_response_mode),
                source.repair_budget_cents,
            )

        spends = {
            "advertising": advertising,
            "service": service,
            "capacity": capacity,
            "resilience": resilience,
            "shared": shared or 0,
            "repair": incident_response.repair_budget_cents,
        }
        total = sum(spends.values())
        if total > cash and template.label != "maintain":
            changed = {
                "increase_advertising": "advertising",
                "increase_service": "service",
                "increase_capacity": "capacity",
                "increase_resilience": "resilience",
                "shared_resilience_contribution": "shared",
                "repair_incident": "repair",
            }.get(template.label)
            if changed is not None:
                spends[changed] = max(0, spends[changed] - (total - cash))
                advertising = spends["advertising"]
                service = spends["service"]
                capacity = spends["capacity"]
                resilience = spends["resilience"]
                shared = (
                    spends["shared"]
                    if baseline.shared_resilience_contribution_cents is not None
                    else None
                )
                if changed == "repair":
                    incident_response = incident_response.__class__(
                        incident_response.mode, spends["repair"]
                    )

        action = CompanyAction(
            action_id=(
                f"public-overlay:{state.episode_id}:{state.round}:"
                f"{company_id}:{template.candidate_id}"
            ),
            episode_id=state.episode_id,
            agent_id=company_id,
            round=state.round,
            state_version=state.state_version,
            price_cents=price,
            advertising_budget_cents=advertising,
            service_budget_cents=service,
            capacity_investment_cents=capacity,
            resilience_budget_cents=resilience,
            shared_resilience_contribution_cents=shared,
            incident_response=incident_response,
            primary_supplier_id=source.primary_supplier_id if template.label.startswith("sourcing_") else baseline.primary_supplier_id,
            backup_supplier_id=source.backup_supplier_id if template.label.startswith("sourcing_") else baseline.backup_supplier_id,
            primary_supplier_share_ppm=source.primary_supplier_share_ppm if template.label.startswith("sourcing_") else baseline.primary_supplier_share_ppm,
            procurement_quantity_orders=baseline.procurement_quantity_orders,
            strategy_summary=(
                f"公开运营基线 + 战略增量：{template.candidate_id}"
            ),
        )
        validated = env.validate_action(action, company_id)
        if not validated.valid or validated.action is None:
            continue
        normalized = validated.action
        payload = CandidateEconomicAction(
            price_cents=normalized.price_cents,
            primary_supplier_id=normalized.primary_supplier_id,
            backup_supplier_id=normalized.backup_supplier_id,
            primary_supplier_share_ppm=normalized.primary_supplier_share_ppm,
            procurement_quantity_orders=normalized.procurement_quantity_orders,
            advertising_budget_cents=normalized.advertising_budget_cents,
            service_budget_cents=normalized.service_budget_cents,
            capacity_investment_cents=normalized.capacity_investment_cents,
            resilience_budget_cents=normalized.resilience_budget_cents,
            shared_resilience_contribution_cents=(
                normalized.shared_resilience_contribution_cents
            ),
            incident_response_mode=normalized.incident_response.mode.value,
            repair_budget_cents=normalized.incident_response.repair_budget_cents,
        )
        key = sha256_hash(payload.model_dump(mode="json"))
        if key in seen:
            continue
        seen.add(key)
        candidates.append(
            StrategicActionCandidate(
                candidate_id=template.candidate_id,
                label=template.label,
                action=payload,
                changed_dimensions=template.changed_dimensions,
                rationale=(
                    "在合法的公开运营基线上进行单一战略调整。"
                    + template.rationale
                ),
            )
        )
    if not candidates or candidates[0].candidate_id != "maintain":
        raise ValueError("public overlay candidate baseline is missing")
    return tuple(candidates)


def generate_public_reliable_candidates(
    config: MarketConfig,
    state: MarketState,
    company_id: str,
    decision_support: Mapping[str, Any],
) -> tuple[StrategicActionCandidate, ...]:
    """Add conservative operating alternatives without hidden state.

    Stage 6.6 forensics found that the Rule overlay baseline could force
    advertising, service, and capacity spending into every candidate.  These
    alternatives are derived only from the company's own visible economics.
    """

    candidates = list(generate_public_overlay_candidates(config, state, company_id))
    env = MarketEnv(config)
    env.load_state(state)
    company = state.company(company_id)
    current_price = company.commercial.price_cents
    break_even = int(decision_support.get("estimated_break_even_price_cents", 0))
    recovery_price = max(current_price, ((break_even + 99) // 100) * 100)
    standard_risk_budget = min(
        500_000,
        max(0, company.financial.cash_balance_cents // 20),
    )
    shared = 0 if state.shared_resilience is not None else None
    specs = [
        (
            "status_quo",
            "status_quo",
            current_price,
            0,
            ["price", "fixed_spend"],
            "保持当前价格并停止新的可选固定投入。",
        ),
        (
            "profit_recovery",
            "profit_recovery",
            recovery_price,
            0,
            ["price", "fixed_spend"],
            "将价格提高到公开测算的百元取整保本线，并停止可选投入。",
        ),
        (
            "risk_buffer",
            "risk_buffer",
            current_price,
            standard_risk_budget,
            ["resilience", "fixed_spend"],
            "保持当前价格，仅保留有限抗冲击投入。",
        ),
    ]
    seen = {
        sha256_hash(item.action.model_dump(mode="json")) for item in candidates
    }
    for candidate_id, label, price, resilience, dimensions, rationale in specs:
        raw = CompanyAction(
            action_id=(
                f"public-reliable:{state.episode_id}:{state.round}:"
                f"{company_id}:{candidate_id}"
            ),
            episode_id=state.episode_id,
            agent_id=company_id,
            round=state.round,
            state_version=state.state_version,
            price_cents=price,
            resilience_budget_cents=resilience,
            shared_resilience_contribution_cents=shared,
            strategy_summary=f"Stage 6.6 可靠性候选：{candidate_id}",
        )
        validated = env.validate_action(raw, company_id)
        if not validated.valid or validated.action is None:
            continue
        action = validated.action
        payload = CandidateEconomicAction(
            price_cents=action.price_cents,
            advertising_budget_cents=action.advertising_budget_cents,
            service_budget_cents=action.service_budget_cents,
            capacity_investment_cents=action.capacity_investment_cents,
            resilience_budget_cents=action.resilience_budget_cents,
            shared_resilience_contribution_cents=(
                action.shared_resilience_contribution_cents
            ),
            incident_response_mode=action.incident_response.mode.value,
            repair_budget_cents=action.incident_response.repair_budget_cents,
        )
        key = sha256_hash(payload.model_dump(mode="json"))
        if key in seen:
            continue
        seen.add(key)
        candidates.append(
            StrategicActionCandidate(
                candidate_id=candidate_id,
                label=label,
                action=payload,
                changed_dimensions=dimensions,
                rationale=rationale,
            )
        )
    return tuple(candidates)


def generate_public_marginal_candidates(
    config: MarketConfig,
    state: MarketState,
    company_id: str,
    decision_support: Mapping[str, Any],
    marginal_plan: MarginalInvestmentPlan,
) -> tuple[StrategicActionCandidate, ...]:
    """Build v6 candidates around the screened operating baseline.

    ``status_quo`` no longer means zero discretionary spending.  It means the
    current price plus only those operating investments whose standalone and
    combined public-forecast marginal returns clear the reliability floors.
    """

    candidates = list(generate_public_overlay_candidates(config, state, company_id))
    env = MarketEnv(config)
    env.load_state(state)
    company = state.company(company_id)
    screened = marginal_plan.selected_action
    current_price = company.commercial.price_cents
    break_even = int(decision_support.get("estimated_break_even_price_cents", 0))
    recovery_price = max(current_price, ((break_even + 99) // 100) * 100)
    current_spend = sum(
        (
            screened.advertising_budget_cents,
            screened.service_budget_cents,
            screened.capacity_investment_cents,
            screened.resilience_budget_cents,
            screened.shared_resilience_contribution_cents or 0,
            screened.repair_budget_cents,
        )
    )
    risk_increment = min(
        500_000,
        max(0, company.financial.cash_balance_cents - current_spend),
        max(0, company.financial.cash_balance_cents // 20),
    )
    specs = (
        (
            "status_quo",
            "status_quo",
            screened,
            "保持当前价格，并只保留通过逐项及组合边际收益门槛的经营投入。",
        ),
        (
            "profit_recovery",
            "profit_recovery",
            screened.model_copy(update={"price_cents": recovery_price}),
            "在边际筛选经营基线上，将价格提高到公开测算的保本线。",
        ),
        (
            "risk_buffer",
            "risk_buffer",
            screened.model_copy(
                update={
                    "resilience_budget_cents": (
                        screened.resilience_budget_cents + risk_increment
                    )
                }
            ),
            "在边际筛选经营基线上增加不超过剩余现金的抗冲击投入。",
        ),
    )
    seen = {
        sha256_hash(item.action.model_dump(mode="json")) for item in candidates
    }
    for candidate_id, label, proposed, rationale in specs:
        raw = CompanyAction(
            action_id=(
                f"public-marginal:{state.episode_id}:{state.round}:"
                f"{company_id}:{candidate_id}"
            ),
            episode_id=state.episode_id,
            agent_id=company_id,
            round=state.round,
            state_version=state.state_version,
            price_cents=proposed.price_cents,
            advertising_budget_cents=proposed.advertising_budget_cents,
            service_budget_cents=proposed.service_budget_cents,
            capacity_investment_cents=proposed.capacity_investment_cents,
            resilience_budget_cents=proposed.resilience_budget_cents,
            shared_resilience_contribution_cents=(
                proposed.shared_resilience_contribution_cents
            ),
            incident_response=IncidentResponse(
                IncidentResponseMode(proposed.incident_response_mode),
                proposed.repair_budget_cents,
            ),
            strategy_summary=f"Stage 6.7 边际筛选候选：{candidate_id}",
        )
        validated = env.validate_action(raw, company_id)
        if not validated.valid or validated.action is None:
            continue
        action = validated.action
        payload = CandidateEconomicAction(
            price_cents=action.price_cents,
            advertising_budget_cents=action.advertising_budget_cents,
            service_budget_cents=action.service_budget_cents,
            capacity_investment_cents=action.capacity_investment_cents,
            resilience_budget_cents=action.resilience_budget_cents,
            shared_resilience_contribution_cents=(
                action.shared_resilience_contribution_cents
            ),
            incident_response_mode=action.incident_response.mode.value,
            repair_budget_cents=action.incident_response.repair_budget_cents,
        )
        key = sha256_hash(payload.model_dump(mode="json"))
        # Keep status_quo even when it happens to equal another baseline: the
        # distinct ID carries the new screened-baseline semantics.
        if key in seen and candidate_id != "status_quo":
            continue
        seen.add(key)
        candidates.append(
            StrategicActionCandidate(
                candidate_id=candidate_id,
                label=label,
                action=payload,
                changed_dimensions=["marginally_screened_portfolio"],
                rationale=rationale,
            )
        )
    if not any(item.candidate_id == "status_quo" for item in candidates):
        raise ValueError("v6 screened status_quo candidate is missing")
    return tuple(candidates)


def generate_final_market_candidates(
    config: MarketConfig,
    state: MarketState,
    company_id: str,
    decision_support: Mapping[str, Any],
    marginal_plan: MarginalInvestmentPlan,
) -> tuple[StrategicActionCandidate, ...]:
    """Add bounded final-market strategic choices to the safe v7 set.

    Every candidate changes one strategic relationship on top of the screened
    operating portfolio.  Bilateral actions are evaluated against explicit
    accept/reject or honor/undercut response scenarios; a declaration alone
    never creates a transfer or an agreement in ``MarketEnv``.
    """

    candidates = list(
        generate_public_marginal_candidates(
            config,
            state,
            company_id,
            decision_support,
            marginal_plan,
        )
    )
    strategic = state.strategic_market
    if strategic is None:
        return tuple(candidates)
    env = MarketEnv(config)
    env.load_state(state)
    constraints = env.get_action_constraints(company_id, state.state_version)
    company = state.company(company_id)
    base = marginal_plan.selected_action
    price_bounds = constraints["bounds"]["price_cents"]
    remaining_cash = max(
        0,
        company.financial.cash_balance_cents
        - sum(
            (
                base.advertising_budget_cents,
                base.service_budget_cents,
                base.capacity_investment_cents,
                base.resilience_budget_cents,
                base.shared_resilience_contribution_cents or 0,
                base.repair_budget_cents,
            )
        ),
    )
    specs: list[tuple[str, str, dict[str, Any], list[str], str]] = []

    project = strategic.threshold_project
    if (
        project is not None
        and constraints["threshold_project_contribution_enabled"]
    ):
        project_bounds = constraints["bounds"][
            "threshold_project_contribution_cents"
        ]
        gap = max(
            0,
            project.required_total_contribution_cents
            - project.accumulated_total_contribution_cents,
        )
        contribution = min(
            gap,
            remaining_cash,
            int(project_bounds["max"]),
        )
        if contribution > 0:
            specs.append(
                (
                    "threshold_project_contribution",
                    "threshold_project_contribution",
                    {"threshold_project_contribution_cents": contribution},
                    ["threshold_project_contribution"],
                    "为有截止期和成功门槛的公共项目出资；只有总额达标才产生公共收益。",
                )
            )

    partners = list(constraints.get("mutual_aid_eligible_partners", ()))
    if constraints.get("mutual_aid_enabled") and partners:
        partner = min(partners)
        demand_reference = (
            company.commercial.potential_demand_orders
            or company.commercial.sales_orders
            or state.market.base_demand_orders
            * max(1, company.commercial.market_share_ppm)
            // 1_000_000
        )
        capacity = company.operations.effective_capacity_orders
        max_orders = int(
            constraints["bounds"]["mutual_aid_capacity_request_orders"]["max"]
        )
        request = min(max_orders, max(0, demand_reference - capacity))
        offer = min(max_orders, max(0, capacity - demand_reference))
        if request > 0:
            specs.append(
                (
                    "mutual_aid_request",
                    "mutual_aid_request",
                    {
                        "mutual_aid_partner_company_id": partner,
                        "mutual_aid_capacity_offer_orders": 0,
                        "mutual_aid_capacity_request_orders": request,
                    },
                    ["mutual_aid_request"],
                    "向一个公开可见的伙伴请求应急履约；仅在对方匹配报价且确有余量时成交。",
                )
            )
        if offer > 0:
            specs.append(
                (
                    "mutual_aid_offer",
                    "mutual_aid_offer",
                    {
                        "mutual_aid_partner_company_id": partner,
                        "mutual_aid_capacity_offer_orders": offer,
                        "mutual_aid_capacity_request_orders": 0,
                    },
                    ["mutual_aid_offer"],
                    "向一个公开可见的伙伴出售闲置履约能力；仅在对方匹配请求时成交。",
                )
            )

    coordination_partners = list(
        constraints.get("price_coordination_eligible_partners", ())
    )
    if constraints.get("price_coordination_enabled") and coordination_partners:
        credibility = dict(strategic.coordination_credibility_by_company_ppm)
        partner = min(
            coordination_partners,
            key=lambda item: (-credibility.get(item, 0), item),
        )
        target = _clip(
            company.commercial.price_cents + 1_000,
            int(price_bounds["min"]),
            int(price_bounds["max"]),
        )
        undercut = max(int(price_bounds["min"]), target - 500)
        specs.extend(
            (
                (
                    "price_coordination_honor",
                    "price_coordination_honor",
                    {
                        "price_cents": target,
                        "price_coordination_partner_company_id": partner,
                        "price_coordination_target_cents": target,
                    },
                    ["price", "price_coordination_target"],
                    "提出并遵守目标价格；可能提高利润，也会损害消费者并产生可审计监管风险。",
                ),
                (
                    "price_coordination_undercut",
                    "price_coordination_undercut",
                    {
                        "price_cents": undercut,
                        "price_coordination_partner_company_id": partner,
                        "price_coordination_target_cents": target,
                    },
                    ["price", "price_coordination_target"],
                    "提出目标价格后低价背叛，换取短期份额但损失公开协调信誉。",
                ),
            )
        )

    seen = {sha256_hash(item.action.model_dump(mode="json")) for item in candidates}
    for candidate_id, label, changes, dimensions, rationale in specs:
        raw = CompanyAction(
            action_id=(
                f"strategic-v9:{state.episode_id}:{state.round}:"
                f"{company_id}:{candidate_id}"
            ),
            episode_id=state.episode_id,
            agent_id=company_id,
            round=state.round,
            state_version=state.state_version,
            price_cents=int(changes.get("price_cents", base.price_cents)),
            advertising_budget_cents=base.advertising_budget_cents,
            service_budget_cents=base.service_budget_cents,
            capacity_investment_cents=base.capacity_investment_cents,
            resilience_budget_cents=base.resilience_budget_cents,
            shared_resilience_contribution_cents=(
                base.shared_resilience_contribution_cents
            ),
            threshold_project_contribution_cents=changes.get(
                "threshold_project_contribution_cents"
            ),
            mutual_aid_partner_company_id=changes.get(
                "mutual_aid_partner_company_id"
            ),
            mutual_aid_capacity_offer_orders=changes.get(
                "mutual_aid_capacity_offer_orders"
            ),
            mutual_aid_capacity_request_orders=changes.get(
                "mutual_aid_capacity_request_orders"
            ),
            price_coordination_partner_company_id=changes.get(
                "price_coordination_partner_company_id"
            ),
            price_coordination_target_cents=changes.get(
                "price_coordination_target_cents"
            ),
            incident_response=IncidentResponse(
                IncidentResponseMode(base.incident_response_mode),
                base.repair_budget_cents,
            ),
            strategy_summary=f"最终市场建议候选：{candidate_id}",
        )
        validated = env.validate_action(raw, company_id)
        if not validated.valid or validated.action is None:
            continue
        normalized = validated.action
        payload = CandidateEconomicAction(
            price_cents=normalized.price_cents,
            advertising_budget_cents=normalized.advertising_budget_cents,
            service_budget_cents=normalized.service_budget_cents,
            capacity_investment_cents=normalized.capacity_investment_cents,
            resilience_budget_cents=normalized.resilience_budget_cents,
            shared_resilience_contribution_cents=(
                normalized.shared_resilience_contribution_cents
            ),
            threshold_project_contribution_cents=(
                normalized.threshold_project_contribution_cents
            ),
            mutual_aid_partner_company_id=(
                normalized.mutual_aid_partner_company_id
            ),
            mutual_aid_capacity_offer_orders=(
                normalized.mutual_aid_capacity_offer_orders
            ),
            mutual_aid_capacity_request_orders=(
                normalized.mutual_aid_capacity_request_orders
            ),
            price_coordination_partner_company_id=(
                normalized.price_coordination_partner_company_id
            ),
            price_coordination_target_cents=(
                normalized.price_coordination_target_cents
            ),
            incident_response_mode=normalized.incident_response.mode.value,
            repair_budget_cents=(
                normalized.incident_response.repair_budget_cents
            ),
        )
        key = sha256_hash(payload.model_dump(mode="json"))
        if key in seen:
            continue
        seen.add(key)
        candidates.append(
            StrategicActionCandidate(
                candidate_id=candidate_id,
                label=label,
                action=payload,
                changed_dimensions=dimensions,
                rationale=rationale,
            )
        )
    return tuple(candidates)


def _reliability_fallback_candidate_id(
    *,
    candidates: tuple[StrategicActionCandidate, ...],
    observation: Mapping[str, Any],
    persona_profile: PersonaProfile,
) -> str:
    available = {item.candidate_id for item in candidates}
    phase = str(
        observation.get("decision_support", {}).get("strategic_phase", "growth")
    )
    active_events = observation.get("active_market_events", [])
    if (
        active_events
        and persona_profile.traits_ppm.risk_aversion >= 500_000
        and "risk_buffer" in available
    ):
        return "risk_buffer"
    if phase in {"profit_recovery", "liquidity_crisis"} and (
        "profit_recovery" in available
    ):
        return "profit_recovery"
    if "status_quo" in available:
        return "status_quo"
    return "maintain"


def _without_research_only_candidates(
    plan: StrategicReliabilityPlan,
) -> StrategicReliabilityPlan:
    """Keep strategic counterfactuals visible without recommending collusion."""

    blocked = {
        "price_coordination_honor",
        "price_coordination_undercut",
    }
    evaluations = [
        item
        for item in plan.evaluations
        if item.candidate.candidate_id not in blocked
    ]
    ranked = max(
        evaluations,
        key=lambda item: (
            item.certainty_equivalent_value_cents,
            item.expected_enterprise_value_cents,
            item.worst_case_risk_adjusted_value_cents,
            item.candidate.candidate_id,
        ),
    )
    baseline = next(
        item for item in evaluations if item.candidate.candidate_id == "maintain"
    )
    gain = (
        ranked.certainty_equivalent_value_cents
        - baseline.certainty_equivalent_value_cents
    )
    payload = plan.model_dump(mode="json")
    payload.update(
        {
            "evaluations": [item.model_dump(mode="json") for item in evaluations],
            "recommended_candidate_id": ranked.candidate.candidate_id,
            "expected_gain_over_baseline_cents": gain,
            "baseline_regret_cents": max(0, gain),
            "plan_hash": "pending",
        }
    )
    payload["plan_hash"] = compute_reliability_plan_hash(payload)
    return StrategicReliabilityPlan.model_validate(payload)


class PublicMarketRolloutAdvisor:
    """Finite-horizon Advisor safe for a public-information Agent context."""

    def __init__(self, config: MarketConfig, *, gate_policy: str | None = None) -> None:
        self.config = config
        self.gate_policy = gate_policy or str(config.data.get("public_advisor_gate", "legacy"))
        if self.gate_policy not in {"legacy", PAIRED_MARKET_GATE_POLICY["policy_version"]}:
            raise ValueError("unsupported public advisor gate policy")
        self._rollout = AuthoritativeMarketRolloutEvaluator(config)

    def advise(
        self,
        *,
        observation: Mapping[str, Any],
        company_id: str,
        persona_profile: PersonaProfile,
        belief_state: BeliefState | Mapping[str, Any] | None = None,
        opponent_model: OpponentModelState | Mapping[str, Any] | None = None,
        horizon_rounds: int = 3,
        scenario_count: int = 5,
        advisor_mode: Literal[
            "public_rollout_v3",
            "pareto_rollout_v4",
            "pareto_reliable_v5",
            "pareto_reliable_v6",
            "pareto_reliable_v7",
            "strategic_market_v9",
        ] = "public_rollout_v3",
    ) -> PublicStrategicAdvice:
        parsed_belief = (
            belief_state
            if isinstance(belief_state, BeliefState)
            else BeliefState.model_validate(
                belief_state
                if belief_state is not None
                else observation.get("belief_state")
            )
            if (belief_state is not None or observation.get("belief_state") is not None)
            else None
        )
        parsed_model = (
            opponent_model
            if isinstance(opponent_model, OpponentModelState)
            else OpponentModelState.model_validate(
                opponent_model
                if opponent_model is not None
                else observation.get("opponent_model_state")
            )
            if (
                opponent_model is not None
                or observation.get("opponent_model_state") is not None
            )
            else None
        )
        forecast, record = build_public_forecast_state(
            config=self.config,
            observation=observation,
            company_id=company_id,
            persona_profile=persona_profile,
            belief_state=parsed_belief,
            opponent_model=parsed_model,
        )
        marginal_plan = (
            build_marginal_investment_plan(
                config=self.config,
                state=forecast,
                company_id=company_id,
                persona_profile=persona_profile,
                public_decision_input_hash=record.public_decision_input_hash,
                opponent_model=parsed_model,
                belief_state=parsed_belief,
                horizon_rounds=horizon_rounds,
                scenario_count=scenario_count,
            )
            if advisor_mode in {
                "pareto_reliable_v6",
                "pareto_reliable_v7",
                "strategic_market_v9",
            }
            else None
        )
        candidates = (
            generate_final_market_candidates(
                self.config,
                forecast,
                company_id,
                observation.get("decision_support", {}),
                marginal_plan,
            )
            if advisor_mode == "strategic_market_v9"
            else generate_public_marginal_candidates(
                self.config,
                forecast,
                company_id,
                observation.get("decision_support", {}),
                marginal_plan,
            )
            if marginal_plan is not None
            else
            generate_public_reliable_candidates(
                self.config,
                forecast,
                company_id,
                observation.get("decision_support", {}),
            )
            if advisor_mode == "pareto_reliable_v5"
            else generate_public_overlay_candidates(
                self.config, forecast, company_id
            )
        )
        oracle = self._rollout.evaluate(
            state=forecast,
            company_id=company_id,
            persona_profile=persona_profile,
            opponent_model=parsed_model,
            belief_state=parsed_belief,
            horizon_rounds=horizon_rounds,
            scenario_count=scenario_count,
            candidates=candidates,
        )
        summaries = [
            PublicRolloutCandidateSummary(
                candidate=item.candidate,
                expected_enterprise_value_delta_cents=(
                    item.expected_enterprise_value_delta_cents
                ),
                expected_cumulative_profit_delta_cents=(
                    item.expected_cumulative_profit_delta_cents
                ),
                expected_loss_cents=item.expected_loss_cents,
                worst_case_risk_adjusted_value_cents=(
                    item.worst_case_risk_adjusted_value_cents
                ),
                expected_persona_utility_ppm=item.expected_persona_utility_ppm,
                certainty_equivalent_value_cents=(
                    item.certainty_equivalent_value_cents
                ),
            )
            for item in oracle.evaluations
        ]
        pareto_decision = None
        reliability_gate = None
        selection_oracle = (
            _without_research_only_candidates(oracle)
            if advisor_mode == "strategic_market_v9"
            else oracle
        )
        if advisor_mode in {
            "pareto_rollout_v4",
            "pareto_reliable_v5",
            "pareto_reliable_v6",
            "pareto_reliable_v7",
            "strategic_market_v9",
        }:
            values = {
                str(item["company_id"]): int(item["value_cents"])
                for item in build_terminal_rankings(
                    forecast, self.config
                )["composite"]
            }
            own_value = values[company_id]
            best_opponent_value = max(
                value for key, value in values.items() if key != company_id
            )
            current_rank = 1 + sum(
                value > own_value for value in values.values()
            )
            pareto_decision = select_pareto_decision(
                selection_oracle,
                PROMOTED_PARETO_SPEC,
                current_rank=current_rank,
                current_competitive_margin_cents=(
                    own_value - best_opponent_value
                ),
                rounds_remaining=forecast.rounds_remaining,
            )
            selected_id = pareto_decision.recommended_candidate_id
            if advisor_mode in {
                "pareto_reliable_v5",
                "pareto_reliable_v6",
                "pareto_reliable_v7",
                "strategic_market_v9",
            }:
                if parsed_model is None:
                    raise ValueError("reliable Pareto advice requires opponent model")
                fallback_id = _reliability_fallback_candidate_id(
                    candidates=candidates,
                    observation=observation,
                    persona_profile=persona_profile,
                )
                reliability_gate = (
                    (build_paired_market_gate if self.gate_policy == PAIRED_MARKET_GATE_POLICY["policy_version"] else build_final_market_gate)(
                        plan=selection_oracle,
                        decision=pareto_decision,
                        diagnostic_fallback_candidate_id=fallback_id,
                        public_decision_input_hash=(
                            record.public_decision_input_hash
                        ),
                        observation=observation,
                        opponent_model=parsed_model,
                        marginal_investment_plan_hash=marginal_plan.plan_hash,
                    )
                    if advisor_mode == "strategic_market_v9"
                    else build_abstention_gate(
                        plan=selection_oracle,
                        decision=pareto_decision,
                        diagnostic_fallback_candidate_id=fallback_id,
                        public_decision_input_hash=(
                            record.public_decision_input_hash
                        ),
                        observation=observation,
                        opponent_model=parsed_model,
                        marginal_investment_plan_hash=marginal_plan.plan_hash,
                    )
                    if advisor_mode == "pareto_reliable_v7"
                    else build_reliability_gate(
                        plan=selection_oracle,
                        decision=pareto_decision,
                        fallback_candidate_id=fallback_id,
                        public_decision_input_hash=(
                            record.public_decision_input_hash
                        ),
                        observation=observation,
                        opponent_model=parsed_model,
                        marginal_investment_plan_hash=(
                            marginal_plan.plan_hash
                            if marginal_plan is not None
                            else None
                        ),
                    )
                )
                selected_id = reliability_gate.effective_candidate_id
        else:
            selected_id = oracle.recommended_candidate_id
        selected = (
            next(
                item
                for item in summaries
                if item.candidate.candidate_id == selected_id
            )
            if selected_id is not None
            else None
        )
        baseline = next(
            item for item in summaries if item.candidate.candidate_id == "maintain"
        )
        expected_gain = (
            0
            if selected is None
            else selected.certainty_equivalent_value_cents
            - baseline.certainty_equivalent_value_cents
        )
        is_pareto = advisor_mode in {
            "pareto_rollout_v4",
            "pareto_reliable_v5",
            "pareto_reliable_v6",
            "pareto_reliable_v7",
            "strategic_market_v9",
        }
        is_reliable = advisor_mode in {
            "pareto_reliable_v5",
            "pareto_reliable_v6",
            "pareto_reliable_v7",
            "strategic_market_v9",
        }
        is_marginal = advisor_mode in {
            "pareto_reliable_v6",
            "pareto_reliable_v7",
            "strategic_market_v9",
        }
        is_abstention = advisor_mode in {"pareto_reliable_v7", "strategic_market_v9"}
        is_strategic_market = advisor_mode == "strategic_market_v9"
        payload: dict[str, Any] = {
            "advice_schema_version": (
                "public-strategic-market-advice-v9.0.0"
                if is_strategic_market
                else "public-pareto-abstention-advice-v7.0.0"
                if is_abstention
                else "public-pareto-marginal-advice-v6.0.0"
                if is_marginal
                else "public-pareto-reliable-advice-v5.0.0"
                if is_reliable
                else "public-pareto-advice-v4.0.0"
                if is_pareto
                else "public-strategic-advice-v3.0.0"
            ),
            "advisor_mode": advisor_mode,
            "advisor_model_version": (
                "public-final-strategic-market-rollout-v1.0.0"
                if is_strategic_market
                else "public-pareto-abstention-market-rollout-v3.0.0"
                if is_abstention
                else "public-pareto-marginal-market-rollout-v2.0.0"
                if is_marginal
                else "public-pareto-reliable-market-rollout-v1.0.0"
                if is_reliable
                else "public-pareto-market-rollout-v1.0.0"
                if is_pareto
                else "public-observation-market-rollout-v1.0.0"
            ),
            "episode_id": forecast.episode_id,
            "round": forecast.round,
            "state_version": forecast.state_version,
            "company_id": company_id,
            "persona_id": persona_profile.persona_id,
            "persona_profile_hash": persona_profile.profile_hash,
            "public_decision_input_hash": record.public_decision_input_hash,
            "forecast_state_hash": record.forecast_state_hash,
            "belief_hash": (
                compute_belief_hash(parsed_belief)
                if parsed_belief is not None
                else None
            ),
            "opponent_model_hash": (
                compute_opponent_model_hash(parsed_model)
                if parsed_model is not None
                else None
            ),
            "horizon_rounds": oracle.horizon_rounds,
            "scenario_count": oracle.scenario_count,
            "candidate_actions": [item.model_dump(mode="json") for item in summaries],
            "recommended_candidate_id": (
                selected.candidate.candidate_id if selected is not None else None
            ),
            "recommended_action": (
                selected.candidate.action.model_dump(mode="json")
                if selected is not None
                else None
            ),
            "expected_gain_over_baseline_cents": expected_gain,
            "baseline_regret_cents": max(0, expected_gain),
            "recommendation_reason": (
                "公开预测证据不足，可靠性门禁已弃权；系统没有生成替代动作，Agent 应保留自己的合法经营判断。"
                if is_abstention
                and reliability_gate is not None
                and reliability_gate.should_abstain
                else "公开预测证据不足，可靠性门禁已放弃强建议并回退到可审计的安全经营候选。"
                if reliability_gate is not None and reliability_gate.should_abstain
                else "在合法公开信息重建的预测市场中，该候选通过了价值、竞争、最坏情景与人格效用约束，并位于安全 Pareto 前沿。"
                if is_pareto
                else "在合法公开信息重建的预测市场中，该候选的人格风险调整长期价值最高。"
            ),
            "recommendation_is_non_binding": True,
            "approximate_best_response": True,
            "claims_nash_equilibrium": False,
            "uses_authoritative_hidden_market_state": False,
            "uses_hidden_opponent_state": False,
            "hidden_opponent_fields_are_estimated": True,
            "allowed_in_public_agent_context": True,
            "pareto_decision": (
                pareto_decision.model_dump(mode="json")
                if pareto_decision is not None
                else None
            ),
            "selection_situation": (
                pareto_decision.situation.situation
                if pareto_decision is not None
                else None
            ),
            "promotion_evidence_sha256": (
                None
                if is_strategic_market
                else PROMOTION_EVIDENCE_SHA256
                if is_pareto
                else None
            ),
            "planner_recommended_candidate_id": (
                pareto_decision.recommended_candidate_id
                if reliability_gate is not None and pareto_decision is not None
                else None
            ),
            "safe_candidate_ids": (
                reliability_gate.safe_candidate_ids
                if reliability_gate is not None
                else None
            ),
            "excluded_candidates": (
                [
                    item.model_dump(mode="json")
                    for item in reliability_gate.excluded_candidates
                ]
                if reliability_gate is not None
                else None
            ),
            "reliability_gate": (
                reliability_gate.model_dump(mode="json")
                if reliability_gate is not None
                else None
            ),
            "investment_marginal_plan": (
                marginal_plan.model_dump(mode="json")
                if marginal_plan is not None
                else None
            ),
            "execution_disposition": (
                reliability_gate.execution_disposition
                if is_abstention and reliability_gate is not None
                else None
            ),
            "withheld_candidate_id": (
                reliability_gate.withheld_candidate_id
                if is_abstention and reliability_gate is not None
                else None
            ),
            "research_only_candidate_ids": (
                [
                    item.candidate.candidate_id
                    for item in summaries
                    if item.candidate.candidate_id
                    in {
                        "price_coordination_honor",
                        "price_coordination_undercut",
                    }
                ]
                if is_strategic_market
                else None
            ),
            "final_market_gate_policy": (
                (PAIRED_MARKET_GATE_POLICY if self.gate_policy == PAIRED_MARKET_GATE_POLICY["policy_version"] else FINAL_MARKET_GATE_POLICY) if is_strategic_market else None
            ),
            "limitations": [
                *record.assumptions,
                "finite candidate set and bounded horizon do not establish equilibrium",
                "forecast values are estimates; authoritative outcomes may differ",
                *(
                    [
                        "promotion evidence is deterministic synthetic holdout validation, not real-market calibration"
                    ]
                    if is_pareto
                    else []
                ),
                *(
                    [
                        "Stage 6.6 reliability gating uses public opponent confidence, forecast dispersion and own decision support only"
                    ]
                    if is_reliable
                    else []
                ),
                *(
                    [
                        "Stage 6.7 status_quo retains only investments that pass standalone and portfolio marginal-return floors"
                    ]
                    if is_marginal
                    else []
                ),
                *(
                    [
                        "Stage 6.9 fail-closed abstention emits no executable fallback; the Agent retains its independently generated legal action"
                    ]
                    if is_abstention
                    else []
                ),
                *(
                    [
                        "Stage 8 v9 evaluates threshold cooperation, bilateral mutual aid and price coordination against explicit opponent-response scenarios",
                        "price coordination can trigger fines and consumer harm; it is modeled for research and is not normative business advice",
                    ]
                    if is_strategic_market
                    else []
                ),
            ],
            "advice_hash": "pending",
        }
        payload["advice_hash"] = compute_public_advice_hash(payload)
        return PublicStrategicAdvice.model_validate(payload)
