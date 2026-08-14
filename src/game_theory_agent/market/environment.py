"""Stateful, replayable Engineering MVP v4 grocery market environment."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import replace
from typing import Any

from game_theory_agent.market.config import MarketConfig
from game_theory_agent.market.exceptions import (
    ActionValidationError,
    EpisodeCompleteError,
    IdempotencyConflictError,
    JointActionError,
    StateInvariantError,
    StateVersionConflictError,
)
from game_theory_agent.market.models import (
    BrandState,
    CommercialState,
    CompanyAction,
    CompanyHistory,
    CompanyIncident,
    CompanyState,
    FinancialState,
    MarketEvent,
    MarketSnapshot,
    MarketState,
    OperationsState,
    Persona,
    RiskSignal,
    RiskState,
    StepResult,
)
from game_theory_agent.market.protocols import ComponentRng, sha256_hash, state_hash
from game_theory_agent.market.validation import ActionValidator, ValidationResult


PPM = 1_000_000


def _clip(value: int, low: int, high: int) -> int:
    return min(max(value, low), high)


def _round_ratio(numerator: int, denominator: int) -> int:
    """Integer division with round-half-to-even."""

    if denominator <= 0:
        raise ValueError("denominator must be positive")
    sign = -1 if numerator < 0 else 1
    quotient, remainder = divmod(abs(numerator), denominator)
    doubled = remainder * 2
    if doubled > denominator or (doubled == denominator and quotient % 2 == 1):
        quotient += 1
    return sign * quotient


def _ppm_mul(*values: int) -> int:
    result = PPM
    for value in values:
        result = _round_ratio(result * value, PPM)
    return result


def _sat_ppm(budget_cents: int, scale_cents: int) -> int:
    if budget_cents <= 0:
        return 0
    return _round_ratio(budget_cents * PPM, budget_cents + scale_cents)


def _allocate_integer(total: int, weights: Mapping[str, int | float]) -> dict[str, int]:
    """Largest remainder with entity-id tie breaking."""

    if total < 0 or not weights:
        raise ValueError("invalid allocation request")
    positive = {str(key): max(0.0, float(value)) for key, value in weights.items()}
    denominator = sum(positive.values())
    if denominator <= 0:
        first = sorted(positive)[0]
        return {key: (total if key == first else 0) for key in positive}
    exact = {key: total * value / denominator for key, value in positive.items()}
    allocated = {key: math.floor(value) for key, value in exact.items()}
    remaining = total - sum(allocated.values())
    order = sorted(
        exact,
        key=lambda key: (-(exact[key] - allocated[key]), key),
    )
    for key in order[:remaining]:
        allocated[key] += 1
    return allocated


class MarketEnv:
    """The backend's single source of truth for a dynamic stochastic market."""

    def __init__(self, config: MarketConfig) -> None:
        self.config = config
        self.validator = ActionValidator(config)
        self._state: MarketState | None = None
        self._step_cache: dict[str, tuple[str, StepResult]] = {}
        self._action_registry: dict[str, str] = {}

    def reset(
        self,
        company_ids: Sequence[str] | None = None,
        *,
        episode_id: str = "episode-0001",
        episode_seed: int = 42,
        personas: Mapping[str, Persona | str] | None = None,
        market_model: str = "random",
        max_rounds: int | None = None,
    ) -> MarketState:
        ids = tuple(
            company_ids
            or tuple(
                f"company_{chr(65 + index)}"
                for index in range(self.config.integer("market", "default_agents"))
            )
        )
        self._validate_company_ids(ids)
        if not episode_id.strip():
            raise StateInvariantError("episode_id must be non-empty")
        if not 0 <= episode_seed < (1 << 64):
            raise StateInvariantError("episode_seed must fit uint64")
        selected_rounds = (
            self.config.integer("episode_options", "default_rounds")
            if max_rounds is None
            else max_rounds
        )
        round_options = tuple(self.config.get("episode_options", "round_options"))
        if selected_rounds not in round_options:
            raise StateInvariantError(
                f"max_rounds must be one of {list(round_options)}"
            )

        model_profiles = self.config.mapping("market_models", "profiles")
        if market_model == "random":
            model_rng = ComponentRng(
                self.config.rng_protocol_version,
                episode_seed,
                0,
                "market_model_selection",
                "market",
                0,
            )
            selected_model = model_rng.weighted_choice(
                self.config.mapping("market_models", "selection_weights_ppm")
            )
            demand_jitter = round(
                (model_rng.uniform() * 2 - 1)
                * self.config.integer(
                    "market_models", "random_demand_jitter_ppm"
                )
            )
            price_jitter = round(
                (model_rng.uniform() * 2 - 1)
                * self.config.integer(
                    "market_models", "random_price_anchor_jitter_cents"
                )
            )
        else:
            selected_model = market_model
            demand_jitter = 0
            price_jitter = 0
        if selected_model not in model_profiles:
            raise StateInvariantError(
                f"unknown market model: {selected_model}; expected random or one of "
                f"{sorted(model_profiles)}"
            )
        model_profile = model_profiles[selected_model]
        utility_multipliers = model_profile["utility_multipliers_ppm"]
        demand_bias = _clip(
            int(model_profile["demand_bias_ppm"]) + demand_jitter,
            900_000,
            1_100_000,
        )
        price_anchor = max(
            1,
            int(model_profile["price_anchor_cents"]) + price_jitter,
        )

        initial = self.config.mapping("company_initial")
        shares = _allocate_integer(PPM, {company_id: 1 for company_id in ids})
        parsed_personas = personas or {}
        companies = tuple(
            CompanyState(
                company_id=company_id,
                persona=Persona(parsed_personas.get(company_id, Persona.NONE)),
                financial=FinancialState(
                    cash_balance_cents=int(initial["cash_balance_cents"]),
                    capacity_book_value_cents=int(initial["capacity_book_value_cents"]),
                ),
                commercial=CommercialState(
                    price_cents=int(initial["initial_price_cents"]),
                    market_share_ppm=shares[company_id],
                ),
                operations=OperationsState(
                    base_capacity_orders=int(initial["base_capacity_orders"]),
                    effective_capacity_orders=int(initial["base_capacity_orders"]),
                    financial_capacity_orders=self.config.integer(
                        "capacity", "infinite_financial_capacity_orders"
                    ),
                    capacity_utilization_ppm=0,
                    base_unit_cost_cents=int(initial["base_unit_cost_cents"]),
                    actual_unit_cost_cents=int(initial["base_unit_cost_cents"]),
                ),
                brand=BrandState(
                    brand_awareness_ppm=int(initial["brand_awareness_ppm"]),
                    service_quality_ppm=int(initial["service_quality_ppm"]),
                    reputation_ppm=int(initial["reputation_ppm"]),
                ),
                risk=RiskState(resilience_ppm=int(initial["resilience_ppm"])),
            )
            for company_id in ids
        )
        segments = tuple(
            sorted(
                (name, int(weight))
                for name, weight in model_profile["segment_weights_ppm"].items()
            )
        )
        signals, _ = self._generate_signals(
            episode_id=episode_id,
            episode_seed=episode_seed,
            settled_round=0,
            active_events=(),
            existing_signals=(),
            max_rounds=selected_rounds,
        )
        base_demand = self.config.base_demand_orders
        initial_sentiment = self.config.integer("market", "initial_sentiment_ppm")
        initial_supply = self.config.integer("market", "initial_supply_cost_index_ppm")
        state = MarketState(
            episode_id=episode_id,
            episode_seed=episode_seed,
            round=1,
            rounds_remaining=selected_rounds,
            state_version=0,
            terminal=False,
            max_rounds=selected_rounds,
            market=MarketSnapshot(
                base_demand_orders=base_demand,
                realized_demand_orders=0,
                no_purchase_orders=0,
                lost_after_stockout_orders=0,
                market_sentiment_ppm=initial_sentiment,
                base_supply_cost_index_ppm=initial_supply,
                actual_supply_cost_index_ppm=initial_supply,
                average_paid_price_cents=0,
                market_model_id=selected_model,
                market_model_label=str(model_profile["label"]),
                market_model_description=str(model_profile["description"]),
                demand_bias_ppm=demand_bias,
                price_anchor_cents=price_anchor,
                price_band_cents=int(model_profile["price_band_cents"]),
                utility_price_multiplier_ppm=int(utility_multipliers["price"]),
                utility_awareness_multiplier_ppm=int(utility_multipliers["awareness"]),
                utility_service_multiplier_ppm=int(utility_multipliers["service"]),
                utility_reputation_multiplier_ppm=int(utility_multipliers["reputation"]),
                utility_prior_stockout_multiplier_ppm=int(utility_multipliers["prior_stockout"]),
            ),
            consumer_segments=segments,
            risk_signals=signals,
            active_market_events=(),
            companies=companies,
        )
        state = replace(state, state_hash=state_hash(state.to_dict()))
        self.assert_invariants(state)
        self._state = state
        self._step_cache.clear()
        self._action_registry.clear()
        return state

    def load_state(self, state: MarketState) -> None:
        """Install a validated snapshot for replay or counterfactual evaluation."""

        self.assert_invariants(state)
        self._state = state
        self._step_cache.clear()
        self._action_registry.clear()

    def get_state(self) -> MarketState:
        if self._state is None:
            raise StateInvariantError("environment has not been reset")
        return self._state

    def get_observation(
        self, agent_id: str, information_mode: str = "perfect"
    ) -> dict[str, Any]:
        state = self.get_state()
        if agent_id not in state.company_ids:
            raise KeyError(agent_id)
        if information_mode != "perfect":
            raise ValueError("Engineering MVP only supports perfect information")
        return state.to_dict()

    def get_action_constraints(
        self, agent_id: str, state_version: int
    ) -> dict[str, Any]:
        state = self.get_state()
        if state_version != state.state_version:
            raise StateVersionConflictError("STATE_VERSION_CONFLICT")
        company = state.company(agent_id)
        incident = company.risk.active_incident
        operating = self.config.mapping("operating_costs")
        return {
            "schema_version": self.config.text("schema_versions", "action"),
            "cash_available_cents": company.financial.cash_balance_cents,
            "bounds": self.config.to_dict()["action"]["bounds"],
            "capacity_investment_enabled": state.rounds_remaining > 1,
            "resilience_investment_enabled": state.rounds_remaining > 1,
            "active_incident": incident.to_dict() if incident else None,
            "max_useful_repair_budget_cents": (
                incident.remaining_repair_cents if incident else 0
            ),
            "mandatory_operating_costs": {
                "fixed_overhead_cents": int(operating["fixed_overhead_cents"]),
                "fulfillment_cost_per_order_cents": int(
                    operating["fulfillment_cost_per_order_cents"]
                ),
                "description": str(operating["description"]),
            },
            "constraints": ["total_fixed_spend <= cash_at_round_start"],
        }

    def validate_action(self, raw_action: Any, company_id: str) -> ValidationResult:
        return self.validator.validate(
            raw_action, state=self.get_state(), company_id=company_id
        )

    def step(
        self,
        step_id: str,
        joint_action: Mapping[str, CompanyAction | Mapping[str, Any]],
    ) -> StepResult:
        state = self.get_state()
        raw_joint_hash = sha256_hash(
            {
                str(company_id): (
                    action.to_dict()
                    if isinstance(action, CompanyAction)
                    else dict(action)
                )
                for company_id, action in sorted(joint_action.items())
            }
        )
        cached = self._step_cache.get(step_id)
        if cached:
            cached_hash, result = cached
            if cached_hash != raw_joint_hash:
                raise IdempotencyConflictError("IDEMPOTENCY_CONFLICT")
            return result

        if state.terminal:
            raise EpisodeCompleteError("episode is already terminal")
        expected_step_id = f"{state.episode_id}:{state.round}:{state.state_version}"
        if step_id != expected_step_id:
            raise StateVersionConflictError(
                f"step_id must be {expected_step_id}; got {step_id}"
            )
        actions = self._validate_joint_action(state, joint_action)
        joint_hash = sha256_hash(
            {
                company_id: actions[company_id].to_dict()
                for company_id in state.company_ids
            }
        )
        pending_action_ids: dict[str, str] = {}
        for action in actions.values():
            payload_hash = sha256_hash(action.to_dict())
            prior = self._action_registry.get(action.action_id)
            if prior is not None:
                if prior != payload_hash:
                    raise IdempotencyConflictError("IDEMPOTENCY_CONFLICT")
                raise IdempotencyConflictError("action_id has already been executed")
            pending_action_ids[action.action_id] = payload_hash

        result = self._transition(state, step_id, actions, joint_hash)
        self.assert_invariants(result.state_after)
        self._state = result.state_after
        self._step_cache[step_id] = (raw_joint_hash, result)
        self._action_registry.update(pending_action_ids)
        return result

    def _transition(
        self,
        state: MarketState,
        step_id: str,
        actions: Mapping[str, CompanyAction],
        joint_hash: str,
    ) -> StepResult:
        random_summary: dict[str, int] = {}
        market_cfg = self.config.mapping("market")
        action_cfg = self.config.mapping("action")
        choice_cfg = self.config.mapping("consumer_choice")
        update_cfg = self.config.mapping("state_updates")
        capacity_cfg = self.config.mapping("capacity")
        event_cfg = self.config.mapping("events")
        incident_cfg = self.config.mapping("incidents")
        operating_cfg = self.config.mapping("operating_costs")

        scales = action_cfg["saturation_scales_cents"]
        ad_inputs = {
            company_id: _sat_ppm(
                action.advertising_budget_cents, int(scales["advertising"])
            )
            for company_id, action in actions.items()
        }
        service_inputs = {
            company_id: _sat_ppm(action.service_budget_cents, int(scales["service"]))
            for company_id, action in actions.items()
        }
        resilience_inputs = {
            company_id: _sat_ppm(
                action.resilience_budget_cents, int(scales["resilience"])
            )
            for company_id, action in actions.items()
        }
        average_offered_price = _round_ratio(
            sum(action.price_cents for action in actions.values()), len(actions)
        )

        event_demand_multiplier = PPM
        for event in state.active_market_events:
            event_demand_multiplier = _ppm_mul(
                event_demand_multiplier, event.demand_multiplier_ppm
            )

        demand_rng = self._rng(state, "demand_noise", summary=random_summary)
        demand_noise_ppm = int(
            round(demand_rng.normal_approx() * int(market_cfg["demand_noise_std_ppm"]))
        )
        realized_demand = max(
            0,
            _round_ratio(
                state.market.base_demand_orders
                * state.market.market_sentiment_ppm
                * event_demand_multiplier
                * state.market.demand_bias_ppm
                * (PPM + demand_noise_ppm),
                PPM * PPM * PPM * PPM,
            ),
        )

        segment_definitions = choice_cfg["segments"]
        segment_demand = _allocate_integer(
            realized_demand,
            dict(state.consumer_segments),
        )

        company_runtime: dict[str, dict[str, int | CompanyIncident | None]] = {}
        utilities: dict[str, dict[str, float]] = {}
        max_event_reduction = int(event_cfg["resilience_max_reduction_ppm"])

        for company in state.companies:
            company_id = company.company_id
            action = actions[company_id]
            resilience = company.risk.resilience_ppm
            supply_multiplier = PPM
            capacity_multiplier = PPM
            advertising_multiplier = PPM
            market_service_penalty = 0
            market_reputation_penalty = 0
            for event in state.active_market_events:
                supply_multiplier = _ppm_mul(
                    supply_multiplier,
                    self._protected_cost_multiplier(
                        event.supply_cost_multiplier_ppm,
                        resilience,
                        max_event_reduction,
                    ),
                )
                capacity_multiplier = _ppm_mul(
                    capacity_multiplier,
                    self._protected_loss_multiplier(
                        event.capacity_multiplier_ppm, resilience, max_event_reduction
                    ),
                )
                advertising_multiplier = _ppm_mul(
                    advertising_multiplier,
                    self._protected_loss_multiplier(
                        event.advertising_multiplier_ppm,
                        resilience,
                        max_event_reduction,
                    ),
                )
                reduction = PPM - _ppm_mul(max_event_reduction, resilience)
                market_service_penalty += _ppm_mul(event.service_penalty_ppm, reduction)
                market_reputation_penalty += _ppm_mul(
                    event.reputation_penalty_ppm, reduction
                )

            incident_after_repair, incident_factors = self._repair_incident(
                company.risk.active_incident,
                action.incident_response.repair_budget_cents,
                int(incident_cfg["max_repair_mitigation_ppm"]),
            )
            effective_ad = _ppm_mul(
                ad_inputs[company_id],
                advertising_multiplier,
                incident_factors["advertising_multiplier_ppm"],
            )
            choice_awareness = _clip(
                _ppm_mul(
                    int(choice_cfg["prior_awareness_weight_ppm"]),
                    company.brand.brand_awareness_ppm,
                )
                + _ppm_mul(
                    int(choice_cfg["current_awareness_weight_ppm"]), effective_ad
                ),
                0,
                PPM,
            )
            choice_service = _clip(
                _ppm_mul(
                    int(choice_cfg["prior_service_weight_ppm"]),
                    company.brand.service_quality_ppm,
                )
                + _ppm_mul(
                    int(choice_cfg["current_service_weight_ppm"]),
                    service_inputs[company_id],
                )
                - market_service_penalty
                - incident_factors["service_penalty_ppm"],
                0,
                PPM,
            )

            actual_unit_cost = _round_ratio(
                company.operations.base_unit_cost_cents
                * state.market.base_supply_cost_index_ppm
                * supply_multiplier,
                PPM * PPM,
            )
            operational_rng = self._rng(
                state,
                "operational_capacity_noise",
                company_id,
                summary=random_summary,
            )
            operational_noise = _clip(
                PPM
                + int(
                    round(
                        operational_rng.normal_approx()
                        * int(capacity_cfg["operational_noise_std_ppm"])
                    )
                ),
                int(capacity_cfg["operational_noise_min_ppm"]),
                int(capacity_cfg["operational_noise_max_ppm"]),
            )
            effective_capacity = max(
                0,
                (
                    company.operations.base_capacity_orders
                    * capacity_multiplier
                    * incident_factors["capacity_multiplier_ppm"]
                    * operational_noise
                )
                // (PPM * PPM * PPM),
            )
            refund_per_order = _round_ratio(
                action.price_cents * incident_factors["refund_rate_ppm"], PPM
            )
            fulfillment_cost_per_order = int(
                operating_cfg["fulfillment_cost_per_order_cents"]
            )
            contribution = (
                action.price_cents
                - actual_unit_cost
                - refund_per_order
                - fulfillment_cost_per_order
            )
            available_after_action = max(
                0, company.financial.cash_balance_cents - action.fixed_spend_cents
            )
            operating_overhead = min(
                int(operating_cfg["fixed_overhead_cents"]), available_after_action
            )
            cash_after_fixed = available_after_action - operating_overhead
            if contribution >= 0:
                financial_capacity = int(
                    capacity_cfg["infinite_financial_capacity_orders"]
                )
            else:
                financial_capacity = cash_after_fixed // -contribution
            fulfillment_cap = min(effective_capacity, financial_capacity)

            company_runtime[company_id] = {
                "effective_ad_ppm": effective_ad,
                "choice_awareness_ppm": choice_awareness,
                "choice_service_ppm": choice_service,
                "market_service_penalty_ppm": market_service_penalty,
                "market_reputation_penalty_ppm": market_reputation_penalty,
                "incident_after_repair": incident_after_repair,
                "incident_service_penalty_ppm": incident_factors["service_penalty_ppm"],
                "incident_reputation_penalty_ppm": incident_factors[
                    "reputation_penalty_ppm"
                ],
                "refund_rate_ppm": incident_factors["refund_rate_ppm"],
                "operating_overhead_cents": operating_overhead,
                "fulfillment_cost_per_order_cents": fulfillment_cost_per_order,
                "actual_unit_cost_cents": actual_unit_cost,
                "effective_capacity_orders": effective_capacity,
                "financial_capacity_orders": financial_capacity,
                "fulfillment_cap_orders": fulfillment_cap,
            }

            relative_price_signal = _clip(
                _round_ratio(
                    (average_offered_price - action.price_cents) * PPM,
                    int(choice_cfg["price_scale_cents"]),
                ),
                int(choice_cfg["relative_price_signal_min_ppm"]),
                int(choice_cfg["relative_price_signal_max_ppm"]),
            )
            utilities[company_id] = {}
            for segment_name, segment in segment_definitions.items():
                coefficients = segment["coefficients_ppm"]
                utility_noise_rng = self._rng(
                    state,
                    "consumer_utility_noise",
                    f"{company_id}|{segment_name}",
                    summary=random_summary,
                )
                noise_ppm = int(
                    round(
                        utility_noise_rng.normal_approx()
                        * int(choice_cfg["utility_noise_std_ppm"])
                    )
                )
                utility_ppm = (
                    _ppm_mul(
                        int(coefficients["price"]),
                        state.market.utility_price_multiplier_ppm,
                        relative_price_signal,
                    )
                    + _ppm_mul(
                        int(coefficients["awareness"]),
                        state.market.utility_awareness_multiplier_ppm,
                        choice_awareness,
                    )
                    + _ppm_mul(
                        int(coefficients["service"]),
                        state.market.utility_service_multiplier_ppm,
                        choice_service,
                    )
                    + _ppm_mul(
                        int(coefficients["reputation"]),
                        state.market.utility_reputation_multiplier_ppm,
                        company.brand.reputation_ppm,
                    )
                    + _ppm_mul(
                        int(coefficients["prior_stockout"]),
                        state.market.utility_prior_stockout_multiplier_ppm,
                        company.brand.last_attempted_unfulfilled_rate_ppm,
                    )
                    + noise_ppm
                )
                utilities[company_id][segment_name] = utility_ppm / PPM

        initial_assignments: dict[str, dict[str, int]] = {
            company_id: {segment: 0 for segment in segment_definitions}
            for company_id in state.company_ids
        }
        no_purchase_orders = 0
        temperature = int(choice_cfg["temperature_ppm"]) / PPM
        for segment_name, demand in segment_demand.items():
            values = {
                company_id: utilities[company_id][segment_name]
                for company_id in state.company_ids
            }
            values["outside"] = (
                int(segment_definitions[segment_name]["outside_utility_ppm"]) / PPM
            )
            probabilities = self._softmax(values, temperature)
            allocation = _allocate_integer(demand, probabilities)
            no_purchase_orders += allocation["outside"]
            for company_id in state.company_ids:
                initial_assignments[company_id][segment_name] = allocation[company_id]

        initial_fulfilled: dict[str, dict[str, int]] = {}
        attempted: dict[str, dict[str, int]] = {}
        remaining_caps: dict[str, int] = {}
        for company_id in state.company_ids:
            assigned = initial_assignments[company_id]
            total_assigned = sum(assigned.values())
            cap = int(company_runtime[company_id]["fulfillment_cap_orders"])
            fulfill_total = min(total_assigned, cap)
            fulfilled = (
                _allocate_integer(fulfill_total, assigned)
                if total_assigned
                else {segment: 0 for segment in segment_definitions}
            )
            initial_fulfilled[company_id] = fulfilled
            attempted[company_id] = {
                segment: assigned[segment] - fulfilled[segment]
                for segment in segment_definitions
            }
            remaining_caps[company_id] = cap - fulfill_total

        received = {company_id: 0 for company_id in state.company_ids}
        own_lost = {company_id: 0 for company_id in state.company_ids}
        lost_after_stockout = 0
        for origin in sorted(state.company_ids):
            for segment_name in sorted(segment_definitions):
                quantity = attempted[origin][segment_name]
                if quantity <= 0:
                    continue
                candidates = [
                    company_id
                    for company_id in state.company_ids
                    if company_id != origin and remaining_caps[company_id] > 0
                ]
                values = {
                    company_id: utilities[company_id][segment_name]
                    for company_id in candidates
                }
                values["outside"] = (
                    int(segment_definitions[segment_name]["outside_utility_ppm"]) / PPM
                )
                allocation = _allocate_integer(
                    quantity, self._softmax(values, temperature)
                )
                recovered = 0
                for company_id in candidates:
                    accepted = min(allocation[company_id], remaining_caps[company_id])
                    received[company_id] += accepted
                    remaining_caps[company_id] -= accepted
                    recovered += accepted
                batch_lost = quantity - recovered
                own_lost[origin] += batch_lost
                lost_after_stockout += batch_lost

        sales = {
            company_id: sum(initial_fulfilled[company_id].values())
            + received[company_id]
            for company_id in state.company_ids
        }
        total_sales = sum(sales.values())
        shares = (
            _allocate_integer(PPM, sales)
            if total_sales > 0
            else {company_id: 0 for company_id in state.company_ids}
        )

        next_companies: list[CompanyState] = []
        history_window = int(self.config.get("company_initial", "history_window"))
        for company in state.companies:
            company_id = company.company_id
            action = actions[company_id]
            runtime = company_runtime[company_id]
            company_sales = sales[company_id]
            actual_unit_cost = int(runtime["actual_unit_cost_cents"])
            revenue = action.price_cents * company_sales
            variable_cost = actual_unit_cost * company_sales
            refund_per_order = _round_ratio(
                action.price_cents * int(runtime["refund_rate_ppm"]), PPM
            )
            refund_cost = refund_per_order * company_sales
            operating_cost = int(runtime["operating_overhead_cents"]) + (
                int(runtime["fulfillment_cost_per_order_cents"]) * company_sales
            )
            round_profit = (
                revenue
                - variable_cost
                - action.fixed_spend_cents
                - refund_cost
                - operating_cost
            )
            next_cash = company.financial.cash_balance_cents + round_profit

            assigned_total = sum(initial_assignments[company_id].values())
            attempted_total = sum(attempted[company_id].values())
            unfulfilled_rate = (
                _round_ratio(attempted_total * PPM, assigned_total)
                if assigned_total
                else 0
            )
            fulfillment_rate = PPM - unfulfilled_rate
            awareness_next = _clip(
                _ppm_mul(
                    int(update_cfg["awareness_retention_ppm"]),
                    company.brand.brand_awareness_ppm,
                )
                + _ppm_mul(
                    int(update_cfg["awareness_input_weight_ppm"]),
                    int(runtime["effective_ad_ppm"]),
                ),
                0,
                PPM,
            )
            service_next = _clip(
                _ppm_mul(
                    int(update_cfg["service_retention_ppm"]),
                    company.brand.service_quality_ppm,
                )
                + _ppm_mul(
                    int(update_cfg["service_input_weight_ppm"]),
                    service_inputs[company_id],
                )
                - int(runtime["market_service_penalty_ppm"])
                - int(runtime["incident_service_penalty_ppm"]),
                0,
                PPM,
            )
            reputation_next = _clip(
                _ppm_mul(
                    int(update_cfg["reputation_retention_ppm"]),
                    company.brand.reputation_ppm,
                )
                + _ppm_mul(
                    int(update_cfg["reputation_service_weight_ppm"]),
                    int(runtime["choice_service_ppm"]),
                )
                + _ppm_mul(
                    int(update_cfg["reputation_fulfillment_weight_ppm"]),
                    fulfillment_rate,
                )
                - int(runtime["market_reputation_penalty_ppm"])
                - int(runtime["incident_reputation_penalty_ppm"]),
                0,
                PPM,
            )
            base_capacity_next = (
                company.operations.base_capacity_orders
                * (PPM - int(capacity_cfg["capacity_depreciation_ppm"]))
            ) // PPM + action.capacity_investment_cents // int(
                capacity_cfg["investment_unit_cost_cents"]
            )
            book_value_next = (
                _round_ratio(
                    company.financial.capacity_book_value_cents
                    * (PPM - int(capacity_cfg["book_value_depreciation_ppm"])),
                    PPM,
                )
                + action.capacity_investment_cents
            )
            resilience_next = _clip(
                _ppm_mul(
                    int(update_cfg["resilience_retention_ppm"]),
                    company.risk.resilience_ppm,
                )
                + _ppm_mul(
                    int(update_cfg["resilience_input_weight_ppm"]),
                    resilience_inputs[company_id],
                ),
                0,
                PPM,
            )

            current_incident = runtime["incident_after_repair"]
            carry_incident = None
            if (
                isinstance(current_incident, CompanyIncident)
                and current_incident.remaining_rounds > 1
            ):
                carry_incident = replace(
                    current_incident,
                    remaining_rounds=current_incident.remaining_rounds - 1,
                )
            next_company = CompanyState(
                company_id=company_id,
                persona=company.persona,
                financial=FinancialState(
                    cash_balance_cents=next_cash,
                    round_revenue_cents=revenue,
                    round_variable_cost_cents=variable_cost,
                    round_fixed_spend_cents=action.fixed_spend_cents,
                    round_incident_cost_cents=refund_cost,
                    round_operating_cost_cents=operating_cost,
                    round_profit_cents=round_profit,
                    cumulative_profit_cents=(
                        company.financial.cumulative_profit_cents + round_profit
                    ),
                    capacity_book_value_cents=book_value_next,
                ),
                commercial=CommercialState(
                    price_cents=action.price_cents,
                    market_share_ppm=shares[company_id],
                    potential_demand_orders=assigned_total,
                    sales_orders=company_sales,
                    attempted_unfulfilled_orders=attempted_total,
                    orders_received_from_redistribution=received[company_id],
                    orders_lost_after_redistribution=own_lost[company_id],
                ),
                operations=OperationsState(
                    base_capacity_orders=base_capacity_next,
                    effective_capacity_orders=int(runtime["effective_capacity_orders"]),
                    financial_capacity_orders=int(runtime["financial_capacity_orders"]),
                    capacity_utilization_ppm=(
                        _round_ratio(
                            company_sales * PPM,
                            int(runtime["effective_capacity_orders"]),
                        )
                        if int(runtime["effective_capacity_orders"])
                        else 0
                    ),
                    base_unit_cost_cents=company.operations.base_unit_cost_cents,
                    actual_unit_cost_cents=actual_unit_cost,
                ),
                brand=BrandState(
                    brand_awareness_ppm=awareness_next,
                    service_quality_ppm=service_next,
                    reputation_ppm=reputation_next,
                    last_attempted_unfulfilled_rate_ppm=unfulfilled_rate,
                ),
                risk=RiskState(
                    resilience_ppm=resilience_next,
                    active_incident=carry_incident,
                ),
                history=CompanyHistory(
                    last_action_id=action.action_id,
                    last_action=action,
                    recent_profit_cents=(
                        company.history.recent_profit_cents + (round_profit,)
                    )[-history_window:],
                    recent_market_share_ppm=(
                        company.history.recent_market_share_ppm + (shares[company_id],)
                    )[-history_window:],
                ),
            )
            next_companies.append(next_company)

        sentiment_rng = self._rng(state, "sentiment_noise", summary=random_summary)
        sentiment_noise = int(
            round(
                sentiment_rng.normal_approx()
                * int(market_cfg["sentiment_noise_std_ppm"])
            )
        )
        next_sentiment = _clip(
            _ppm_mul(
                PPM - int(market_cfg["sentiment_reversion_ppm"]),
                state.market.market_sentiment_ppm,
            )
            + _ppm_mul(
                int(market_cfg["sentiment_reversion_ppm"]),
                int(market_cfg["sentiment_mean_ppm"]),
            )
            + sentiment_noise,
            int(market_cfg["sentiment_min_ppm"]),
            int(market_cfg["sentiment_max_ppm"]),
        )
        supply_rng = self._rng(state, "supply_cost_noise", summary=random_summary)
        supply_noise = int(
            round(
                supply_rng.normal_approx()
                * int(market_cfg["supply_cost_noise_std_ppm"])
            )
        )
        next_base_supply = _clip(
            _ppm_mul(
                PPM - int(market_cfg["supply_cost_reversion_ppm"]),
                state.market.base_supply_cost_index_ppm,
            )
            + _ppm_mul(
                int(market_cfg["supply_cost_reversion_ppm"]),
                int(market_cfg["supply_cost_mean_ppm"]),
            )
            + supply_noise,
            int(market_cfg["supply_cost_min_ppm"]),
            int(market_cfg["supply_cost_max_ppm"]),
        )

        terminal = state.round >= state.max_rounds
        if terminal:
            next_events: tuple[MarketEvent, ...] = ()
            next_signals: tuple[RiskSignal, ...] = ()
        else:
            next_events, next_signals = self._advance_events(state, random_summary)
            next_signals, generated_summary = self._generate_signals(
                episode_id=state.episode_id,
                episode_seed=state.episode_seed,
                settled_round=state.round,
                active_events=next_events,
                existing_signals=next_signals,
                max_rounds=state.max_rounds,
            )
            random_summary.update(generated_summary)

        if not terminal:
            next_companies = [
                replace(
                    company,
                    risk=replace(
                        company.risk,
                        active_incident=self._maybe_generate_incident(
                            state,
                            company,
                            company.risk.active_incident,
                            random_summary,
                        ),
                    ),
                )
                for company in next_companies
            ]

        next_actual_supply = next_base_supply
        for event in next_events:
            next_actual_supply = _ppm_mul(
                next_actual_supply, event.supply_cost_multiplier_ppm
            )
        average_paid_price = (
            _round_ratio(
                sum(actions[cid].price_cents * sales[cid] for cid in state.company_ids),
                total_sales,
            )
            if total_sales
            else 0
        )
        terminal_values: tuple[tuple[str, int], ...] = ()
        if terminal:
            terminal_values = tuple(
                (company.company_id, self._terminal_value(company))
                for company in next_companies
            )
        next_state = MarketState(
            episode_id=state.episode_id,
            episode_seed=state.episode_seed,
            round=state.round + 1,
            rounds_remaining=max(0, state.rounds_remaining - 1),
            state_version=state.state_version + 1,
            terminal=terminal,
            max_rounds=state.max_rounds,
            market=MarketSnapshot(
                base_demand_orders=state.market.base_demand_orders,
                realized_demand_orders=realized_demand,
                no_purchase_orders=no_purchase_orders,
                lost_after_stockout_orders=lost_after_stockout,
                market_sentiment_ppm=next_sentiment,
                base_supply_cost_index_ppm=next_base_supply,
                actual_supply_cost_index_ppm=next_actual_supply,
                average_paid_price_cents=average_paid_price,
                market_model_id=state.market.market_model_id,
                market_model_label=state.market.market_model_label,
                market_model_description=state.market.market_model_description,
                demand_bias_ppm=state.market.demand_bias_ppm,
                price_anchor_cents=state.market.price_anchor_cents,
                price_band_cents=state.market.price_band_cents,
                utility_price_multiplier_ppm=state.market.utility_price_multiplier_ppm,
                utility_awareness_multiplier_ppm=state.market.utility_awareness_multiplier_ppm,
                utility_service_multiplier_ppm=state.market.utility_service_multiplier_ppm,
                utility_reputation_multiplier_ppm=state.market.utility_reputation_multiplier_ppm,
                utility_prior_stockout_multiplier_ppm=state.market.utility_prior_stockout_multiplier_ppm,
            ),
            consumer_segments=state.consumer_segments,
            risk_signals=next_signals,
            active_market_events=next_events,
            companies=tuple(next_companies),
            last_joint_action=tuple(
                actions[company_id] for company_id in state.company_ids
            ),
            terminal_enterprise_values_cents=terminal_values,
        )
        next_state = replace(next_state, state_hash=state_hash(next_state.to_dict()))
        return StepResult(
            step_id=step_id,
            settled_round=state.round,
            state_before_hash=state.state_hash,
            state_after=next_state,
            joint_action_hash=joint_hash,
            random_draw_summary=tuple(sorted(random_summary.items())),
            invariant_results=("all_passed",),
        )

    def assert_invariants(self, state: MarketState) -> None:
        failures: list[str] = []
        if not self.config.min_agents <= len(state.companies) <= self.config.max_agents:
            failures.append("company count is outside configured bounds")
        if len(set(state.company_ids)) != len(state.company_ids):
            failures.append("company ids must be unique")
        if state.round != state.state_version + 1:
            failures.append("round/state_version relation is invalid")
        if state.rounds_remaining != max(0, state.max_rounds - state.state_version):
            failures.append("rounds_remaining is inconsistent")
        if state.terminal != (state.state_version >= state.max_rounds):
            failures.append("terminal flag is inconsistent")
        if state.state_hash != state_hash(state.to_dict()):
            failures.append("state hash is inconsistent")
        if len(state.active_market_events) > self.config.integer(
            "events", "max_active_events"
        ):
            failures.append("too many active market events")
        event_types = [event.event_type for event in state.active_market_events]
        if len(event_types) != len(set(event_types)):
            failures.append("same market event type cannot overlap")

        segment_weight_sum = sum(weight for _, weight in state.consumer_segments)
        if segment_weight_sum != PPM:
            failures.append("consumer segment weights must sum to 1000000")
        share_sum = sum(
            company.commercial.market_share_ppm for company in state.companies
        )
        if state.market.realized_demand_orders and share_sum not in (0, PPM):
            failures.append("market shares must sum to 1000000 when there are sales")
        if state.state_version > 0:
            closure = (
                state.market.no_purchase_orders
                + state.market.lost_after_stockout_orders
                + sum(company.commercial.sales_orders for company in state.companies)
            )
            if closure != state.market.realized_demand_orders:
                failures.append("demand allocation does not close")

        for company in state.companies:
            if company.financial.cash_balance_cents < 0:
                failures.append(f"{company.company_id} cash is negative")
            if (
                company.commercial.sales_orders
                > company.operations.effective_capacity_orders
            ):
                failures.append(f"{company.company_id} sales exceed effective capacity")
            if (
                company.commercial.sales_orders
                > company.operations.financial_capacity_orders
            ):
                failures.append(f"{company.company_id} sales exceed financial capacity")
            for label, value in (
                ("share", company.commercial.market_share_ppm),
                ("awareness", company.brand.brand_awareness_ppm),
                ("service", company.brand.service_quality_ppm),
                ("reputation", company.brand.reputation_ppm),
                ("resilience", company.risk.resilience_ppm),
                ("stockout", company.brand.last_attempted_unfulfilled_rate_ppm),
                ("utilization", company.operations.capacity_utilization_ppm),
            ):
                if not 0 <= value <= PPM:
                    failures.append(
                        f"{company.company_id} {label} is outside [0, 1000000]"
                    )
        if failures:
            raise StateInvariantError("; ".join(failures))

    def _validate_joint_action(
        self,
        state: MarketState,
        joint_action: Mapping[str, CompanyAction | Mapping[str, Any]],
    ) -> dict[str, CompanyAction]:
        if not isinstance(joint_action, Mapping):
            raise JointActionError("joint_action must be a mapping")
        missing = set(state.company_ids) - set(joint_action)
        unknown = set(joint_action) - set(state.company_ids)
        if missing or unknown:
            raise JointActionError(
                f"joint action mismatch; missing={sorted(missing)}, unknown={sorted(unknown)}"
            )
        actions: dict[str, CompanyAction] = {}
        errors: list[str] = []
        for company_id in state.company_ids:
            result = self.validator.validate(
                joint_action[company_id], state=state, company_id=company_id
            )
            if result.valid and result.action:
                actions[company_id] = result.action
            else:
                errors.append(f"{company_id}: {'; '.join(result.errors)}")
        action_ids = [action.action_id for action in actions.values()]
        if len(action_ids) != len(set(action_ids)):
            errors.append("action_id must be unique within a joint action")
        if errors:
            raise ActionValidationError(" | ".join(errors))
        return actions

    def _rng(
        self,
        state: MarketState,
        component: str,
        entity_id: str = "",
        draw_index: int = 0,
        *,
        summary: dict[str, int],
    ) -> ComponentRng:
        rng = ComponentRng(
            self.config.rng_protocol_version,
            state.episode_seed,
            state.round,
            component,
            entity_id,
            draw_index,
        )
        key = f"{component}:{entity_id}:{draw_index}"
        summary[key] = rng.sub_seed
        return rng

    @staticmethod
    def _softmax(values: Mapping[str, float], temperature: float) -> dict[str, float]:
        scaled = {key: value / temperature for key, value in values.items()}
        maximum = max(scaled.values())
        exponentials = {key: math.exp(value - maximum) for key, value in scaled.items()}
        denominator = sum(exponentials.values())
        return {key: value / denominator for key, value in exponentials.items()}

    @staticmethod
    def _protected_loss_multiplier(
        multiplier: int, resilience: int, max_reduction: int
    ) -> int:
        if multiplier >= PPM:
            return multiplier
        loss = PPM - multiplier
        protected_loss = _ppm_mul(loss, PPM - _ppm_mul(max_reduction, resilience))
        return PPM - protected_loss

    @staticmethod
    def _protected_cost_multiplier(
        multiplier: int, resilience: int, max_reduction: int
    ) -> int:
        if multiplier <= PPM:
            return multiplier
        increase = multiplier - PPM
        protected_increase = _ppm_mul(
            increase, PPM - _ppm_mul(max_reduction, resilience)
        )
        return PPM + protected_increase

    @staticmethod
    def _repair_incident(
        incident: CompanyIncident | None,
        repair_budget: int,
        max_mitigation: int,
    ) -> tuple[CompanyIncident | None, dict[str, int]]:
        neutral = {
            "capacity_multiplier_ppm": PPM,
            "advertising_multiplier_ppm": PPM,
            "service_penalty_ppm": 0,
            "reputation_penalty_ppm": 0,
            "refund_rate_ppm": 0,
        }
        if incident is None:
            return None, neutral
        accumulated = min(
            incident.repair_required_cents,
            incident.accumulated_repair_cents + repair_budget,
        )
        fully_repaired = accumulated >= incident.repair_required_cents
        updated = (
            None
            if fully_repaired
            else replace(incident, accumulated_repair_cents=accumulated)
        )
        progress = _round_ratio(accumulated * PPM, incident.repair_required_cents)
        impact_factor = PPM - _ppm_mul(max_mitigation, progress)
        # Repair happens during the operating round. Even a full repair can only
        # mitigate part of the already-started disruption; it clears carry-over.
        return updated, {
            "capacity_multiplier_ppm": PPM
            - _ppm_mul(PPM - incident.capacity_multiplier_ppm, impact_factor),
            "advertising_multiplier_ppm": PPM
            - _ppm_mul(PPM - incident.advertising_multiplier_ppm, impact_factor),
            "service_penalty_ppm": _ppm_mul(
                incident.service_penalty_ppm, impact_factor
            ),
            "reputation_penalty_ppm": _ppm_mul(
                incident.reputation_penalty_ppm, impact_factor
            ),
            "refund_rate_ppm": _ppm_mul(incident.refund_rate_ppm, impact_factor),
        }

    def _advance_events(
        self,
        state: MarketState,
        summary: dict[str, int],
    ) -> tuple[tuple[MarketEvent, ...], tuple[RiskSignal, ...]]:
        carried = [
            replace(event, remaining_rounds=event.remaining_rounds - 1)
            for event in state.active_market_events
            if event.remaining_rounds > 1
        ]
        future_signals: list[RiskSignal] = []
        due_signals: list[RiskSignal] = []
        for signal in state.risk_signals:
            if signal.target_round == state.round + 1:
                due_signals.append(signal)
            elif signal.target_round > state.round + 1:
                future_signals.append(signal)
        max_active = self.config.integer("events", "max_active_events")
        active_types = {event.event_type for event in carried}
        event_definitions = self.config.mapping("events", "definitions")
        for signal in sorted(due_signals, key=lambda item: item.signal_id):
            if len(carried) >= max_active or signal.event_type in active_types:
                continue
            rng = self._rng(
                state,
                "event_realization",
                signal.signal_id,
                summary=summary,
            )
            if rng.uniform() >= signal.estimated_probability_ppm / PPM:
                continue
            definition = event_definitions[signal.event_type]["severity"][
                signal.severity
            ]
            duration_rng = self._rng(
                state,
                "event_realization",
                signal.signal_id,
                1,
                summary=summary,
            )
            duration = int(
                duration_rng.weighted_choice(
                    {
                        str(k): int(v)
                        for k, v in definition["duration_weights_ppm"].items()
                    }
                )
            )
            carried.append(
                MarketEvent(
                    event_id=f"{state.episode_id}:event:{signal.signal_id}",
                    event_type=signal.event_type,
                    severity=signal.severity,
                    started_round=state.round + 1,
                    remaining_rounds=duration,
                    demand_multiplier_ppm=int(definition["demand_multiplier_ppm"]),
                    supply_cost_multiplier_ppm=int(
                        definition["supply_cost_multiplier_ppm"]
                    ),
                    capacity_multiplier_ppm=int(definition["capacity_multiplier_ppm"]),
                    advertising_multiplier_ppm=int(
                        definition["advertising_multiplier_ppm"]
                    ),
                    service_penalty_ppm=int(definition["service_penalty_ppm"]),
                    reputation_penalty_ppm=int(definition["reputation_penalty_ppm"]),
                )
            )
            active_types.add(signal.event_type)
        return (
            tuple(sorted(carried, key=lambda item: item.event_id)),
            tuple(sorted(future_signals, key=lambda item: item.signal_id)),
        )

    def _generate_signals(
        self,
        *,
        episode_id: str,
        episode_seed: int,
        settled_round: int,
        active_events: Sequence[MarketEvent],
        existing_signals: Sequence[RiskSignal],
        max_rounds: int,
    ) -> tuple[tuple[RiskSignal, ...], dict[str, int]]:
        summary: dict[str, int] = {}
        result = list(existing_signals)
        blocked_types = {event.event_type for event in active_events} | {
            signal.event_type for signal in existing_signals
        }
        definitions = self.config.mapping("events", "definitions")
        for event_type in sorted(definitions):
            if event_type in blocked_types:
                continue
            definition = definitions[event_type]
            lead = int(definition["lead_time_rounds"])
            target_round = settled_round + 1 + lead
            if target_round > max_rounds:
                continue
            generation_rng = ComponentRng(
                self.config.rng_protocol_version,
                episode_seed,
                settled_round,
                "risk_signal_generation",
                event_type,
                0,
            )
            summary[f"risk_signal_generation:{event_type}:0"] = generation_rng.sub_seed
            if (
                generation_rng.uniform()
                >= int(definition["signal_generation_probability_ppm"]) / PPM
            ):
                continue
            severity_rng = ComponentRng(
                self.config.rng_protocol_version,
                episode_seed,
                settled_round,
                "risk_signal_generation",
                event_type,
                1,
            )
            summary[f"risk_signal_generation:{event_type}:1"] = severity_rng.sub_seed
            severity = severity_rng.weighted_choice(
                {str(k): int(v) for k, v in definition["severity_weights_ppm"].items()}
            )
            severity_definition = definition["severity"][severity]
            result.append(
                RiskSignal(
                    signal_id=(
                        f"{episode_id}:signal:{event_type}:{target_round}:{settled_round}"
                    ),
                    event_type=event_type,
                    target_round=target_round,
                    estimated_probability_ppm=int(
                        severity_definition["estimated_probability_ppm"]
                    ),
                    severity=severity,
                    lead_time_rounds=lead,
                )
            )
        return tuple(sorted(result, key=lambda item: item.signal_id)), summary

    def _maybe_generate_incident(
        self,
        prior_state: MarketState,
        company: CompanyState,
        active_incident: CompanyIncident | None,
        summary: dict[str, int],
    ) -> CompanyIncident | None:
        if active_incident is not None:
            return active_incident
        cfg = self.config.mapping("incidents")
        probability = _ppm_mul(
            int(cfg["base_probability_ppm"]),
            PPM
            - _ppm_mul(
                int(cfg["probability_reduction_ppm"]), company.risk.resilience_ppm
            ),
        )
        rng = self._rng(
            prior_state,
            "incident_generation",
            company.company_id,
            0,
            summary=summary,
        )
        if rng.uniform() >= probability / PPM:
            return None
        type_rng = self._rng(
            prior_state,
            "incident_generation",
            company.company_id,
            1,
            summary=summary,
        )
        incident_type = type_rng.weighted_choice(
            {str(k): int(v) for k, v in cfg["type_weights_ppm"].items()}
        )
        severity_rng = self._rng(
            prior_state,
            "incident_generation",
            company.company_id,
            2,
            summary=summary,
        )
        severity = severity_rng.weighted_choice(
            {str(k): int(v) for k, v in cfg["severity_weights_ppm"].items()}
        )
        definition = cfg["definitions"][incident_type]["severity"][severity]
        impact_factor = PPM - _ppm_mul(
            int(cfg["severity_reduction_ppm"]), company.risk.resilience_ppm
        )
        return CompanyIncident(
            incident_id=(
                f"{prior_state.episode_id}:incident:{company.company_id}:"
                f"{prior_state.round + 1}:{incident_type}"
            ),
            incident_type=incident_type,
            severity=severity,
            started_round=prior_state.round + 1,
            remaining_rounds=int(definition["duration_rounds"]),
            repair_required_cents=int(definition["repair_required_cents"]),
            accumulated_repair_cents=0,
            capacity_multiplier_ppm=PPM
            - _ppm_mul(PPM - int(definition["capacity_multiplier_ppm"]), impact_factor),
            advertising_multiplier_ppm=PPM
            - _ppm_mul(
                PPM - int(definition["advertising_multiplier_ppm"]), impact_factor
            ),
            service_penalty_ppm=_ppm_mul(
                int(definition["service_penalty_ppm"]), impact_factor
            ),
            reputation_penalty_ppm=_ppm_mul(
                int(definition["reputation_penalty_ppm"]), impact_factor
            ),
            refund_rate_ppm=_ppm_mul(int(definition["refund_rate_ppm"]), impact_factor),
        )

    def _terminal_value(self, company: CompanyState) -> int:
        cfg = self.config.mapping("terminal")
        return (
            company.financial.cash_balance_cents
            + _ppm_mul(
                company.financial.capacity_book_value_cents,
                int(cfg["capacity_salvage_rate_ppm"]),
            )
            + _ppm_mul(
                int(cfg["awareness_value_max_cents"]),
                company.brand.brand_awareness_ppm,
            )
            + _ppm_mul(
                int(cfg["service_value_max_cents"]), company.brand.service_quality_ppm
            )
            + _ppm_mul(
                int(cfg["reputation_value_max_cents"]), company.brand.reputation_ppm
            )
            + _ppm_mul(
                int(cfg["resilience_value_max_cents"]), company.risk.resilience_ppm
            )
        )

    def _validate_company_ids(self, ids: Sequence[str]) -> None:
        if not self.config.min_agents <= len(ids) <= self.config.max_agents:
            raise StateInvariantError(
                f"environment supports {self.config.min_agents} to {self.config.max_agents} companies"
            )
        if any(
            not isinstance(company_id, str) or not company_id.strip()
            for company_id in ids
        ):
            raise StateInvariantError("company ids must be non-empty strings")
        if len(set(ids)) != len(ids):
            raise StateInvariantError("company ids must be unique")
