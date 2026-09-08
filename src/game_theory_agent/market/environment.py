"""Stateful, replayable Engineering MVP v4 grocery market environment."""

from __future__ import annotations
from game_theory_agent.market.transaction_accounting import enabled as cash_accounting, close_materials, accounting_failures
from game_theory_agent.market.autonomous_market import decide_government, settle_government, advance_investment, actor_failures
from game_theory_agent.market.models import GovernmentState, ConsumerDecisionAudit

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
    CompanyOperatingStatus,
    CompanyState,
    FinancialState,
    MarketEvent,
    MarketSnapshot,
    MarketState,
    MutualAidTransfer,
    OperationsState,
    Persona,
    PriceCoordinationOutcome,
    PriceCoordinationStatus,
    RiskSignal,
    RiskState,
    SharedResilienceState,
    StepResult,
    SupplyChainState,
    WelfareAccountingState,
)
from game_theory_agent.market.protocols import ComponentRng, sha256_hash, state_hash
from game_theory_agent.market.strategic import (
    apply_threshold_project_refunds,
    advance_strategic_market_state,
    initial_strategic_market_state,
    liquidate_newly_exited,
    settle_threshold_project,
)
from game_theory_agent.market.supply_chain import (
    advance_supplier_availability,
    initial_supply_chain_state,
    settle_procurement,
)
from game_theory_agent.market.supplier_policy import advance_supplier_quotes, decide_quote, price_bounds, pricing_enabled
from game_theory_agent.market.validation import ActionValidator, ValidationResult
from game_theory_agent.market.welfare import (
    advance_welfare_state,
    initial_welfare_state,
)


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
        cooperation_mode: str = "off",
        cooperation_modes: Sequence[str] | None = None,
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
        supported_cooperation_modes = {
            "shared_resilience_v1",
            "threshold_project_v1",
            "mutual_aid_v1",
            "price_coordination_v1",
        }
        if cooperation_modes is not None:
            if cooperation_mode != "off":
                raise StateInvariantError(
                    "use cooperation_mode or cooperation_modes, not both"
                )
            selected_cooperation_modes = set(cooperation_modes)
        elif cooperation_mode == "off":
            selected_cooperation_modes = set()
        elif cooperation_mode == "combined_v1":
            selected_cooperation_modes = set(supported_cooperation_modes)
        else:
            selected_cooperation_modes = {cooperation_mode}
        if not selected_cooperation_modes <= supported_cooperation_modes:
            raise StateInvariantError("unsupported cooperation_mode")
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
        strategic_cfg = self.config.data.get("strategic_market")
        if (
            (
                "threshold_project_v1" in selected_cooperation_modes
                or "mutual_aid_v1" in selected_cooperation_modes
                or "price_coordination_v1" in selected_cooperation_modes
            )
            and not isinstance(strategic_cfg, Mapping)
        ):
            raise StateInvariantError(
                "strategic cooperation requires a strategic market config"
            )
        strategic_market = (
            initial_strategic_market_state(
                company_ids=ids,
                companies=companies,
                config=strategic_cfg,
                threshold_project_enabled=(
                    "threshold_project_v1" in selected_cooperation_modes
                ),
                mutual_aid_enabled=(
                    "mutual_aid_v1" in selected_cooperation_modes
                ),
                price_coordination_enabled=(
                    "price_coordination_v1" in selected_cooperation_modes
                ),
            )
            if isinstance(strategic_cfg, Mapping)
            else None
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
            shared_resilience=(
                SharedResilienceState(
                    industry_resilience_ppm=self.config.integer(
                        "shared_resilience",
                        "initial_industry_resilience_ppm",
                    ),
                    last_contribution_by_company_cents=tuple(
                        (company_id, 0) for company_id in sorted(ids)
                    ),
                )
                if "shared_resilience_v1" in selected_cooperation_modes
                else None
            ),
            strategic_market=strategic_market,
            supply_chain=(
                initial_supply_chain_state(
                    self.config.mapping("supply_chain")
                )
                if isinstance(self.config.data.get("supply_chain"), Mapping)
                else None
            ),
            welfare_accounting=(
                initial_welfare_state(
                    self.config.mapping("welfare_accounting")
                )
                if isinstance(
                    self.config.data.get("welfare_accounting"), Mapping
                )
                else None
            ),
            government=(GovernmentState(self.config.data["autonomous_market"]["government"]["initial_cash_cents"])
                        if self.config.data.get("autonomous_market") else None),
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

    def get_action_constraints(
        self, agent_id: str, state_version: int
    ) -> dict[str, Any]:
        state = self.get_state()
        if state_version != state.state_version:
            raise StateVersionConflictError("STATE_VERSION_CONFLICT")
        company = state.company(agent_id)
        operating_status = (
            state.strategic_market.lifecycle(agent_id).status.value
            if state.strategic_market is not None
            else "operating"
        )
        incident = company.risk.active_incident
        operating = self.config.mapping("operating_costs")
        bounds = self.config.to_dict()["action"]["bounds"]
        if state.shared_resilience is None:
            bounds.pop("shared_resilience_contribution_cents", None)
        threshold_project = (
            state.strategic_market.threshold_project
            if state.strategic_market is not None
            else None
        )
        if threshold_project is None:
            bounds.pop("threshold_project_contribution_cents", None)
        mutual_aid_enabled = bool(
            state.strategic_market is not None
            and state.strategic_market.mutual_aid_enabled
        )
        if not mutual_aid_enabled:
            bounds.pop("mutual_aid_capacity_offer_orders", None)
            bounds.pop("mutual_aid_capacity_request_orders", None)
        price_coordination_enabled = bool(
            state.strategic_market is not None
            and state.strategic_market.price_coordination_enabled
        )
        supply_chain_enabled = state.supply_chain is not None
        if not supply_chain_enabled:
            bounds.pop("primary_supplier_share_ppm", None)
        if self.config.data.get("autonomous_market"):
            bounds["procurement_quantity_orders"] = {"min":0,"max":company.operations.base_capacity_orders}
        return {
            **({"cash_material_accounting":True,"procurement_quantity_enabled":bool(self.config.data.get("autonomous_market"))} if cash_accounting(self.config.data.get("supply_chain")) else {}),
            "schema_version": self.config.text("schema_versions", "action"),
            "cash_available_cents": company.financial.cash_balance_cents,
            "operating_status": operating_status,
            "commercial_action_enabled": operating_status != "exited",
            "bounds": bounds,
            "supply_contract_enabled": bool(self.config.data.get("supply_chain",{}).get("strategic_policy")),
            "capacity_investment_enabled": state.rounds_remaining > 1,
            "resilience_investment_enabled": state.rounds_remaining > 1,
            "shared_resilience_contribution_enabled": (
                state.shared_resilience is not None
                and state.rounds_remaining > 1
            ),
            "threshold_project_contribution_enabled": (
                threshold_project is not None
                and threshold_project.status.value == "active"
                and state.round <= threshold_project.deadline_round
                and state.rounds_remaining > 1
            ),
            "mutual_aid_enabled": mutual_aid_enabled,
            "mutual_aid_eligible_partners": [
                company_id
                for company_id in (
                    state.strategic_market.active_company_ids
                    if state.strategic_market is not None
                    else ()
                )
                if company_id != agent_id
            ],
            "mutual_aid_fee_per_order_cents": (
                int(self.config.mapping("strategic_market", "mutual_aid")["fee_per_order_cents"])
                if mutual_aid_enabled
                else None
            ),
            "price_coordination_enabled": price_coordination_enabled,
            "price_coordination_eligible_partners": [
                company_id
                for company_id in (
                    state.strategic_market.active_company_ids
                    if state.strategic_market is not None
                    else ()
                )
                if company_id != agent_id
            ],
            "price_coordination_adherence_tolerance_cents": (
                int(
                    self.config.mapping(
                        "strategic_market", "price_coordination"
                    )["adherence_tolerance_cents"]
                )
                if price_coordination_enabled
                else None
            ),
            "supply_chain_enabled": supply_chain_enabled,
            "eligible_suppliers": (
                [{k:v for k,v in supplier.to_dict().items() if supplier.account is None or k not in {"account","investment_decision","unit_cost_cents","round_requested_orders"}} for supplier in state.supply_chain.suppliers]
                if state.supply_chain is not None
                else []
            ),
            "cooperation_mode": (
                "combined_v1"
                if sum(
                    (
                        state.shared_resilience is not None,
                        threshold_project is not None,
                        mutual_aid_enabled,
                        price_coordination_enabled,
                    )
                )
                > 1
                else (
                    "shared_resilience_v1"
                    if state.shared_resilience is not None
                    else (
                        "threshold_project_v1"
                        if threshold_project is not None
                        else (
                            "mutual_aid_v1" if mutual_aid_enabled else "off"
                            if not price_coordination_enabled
                            else "price_coordination_v1"
                        )
                    )
                )
            ),
            "cooperation_modes": [
                mode
                for mode, enabled in (
                    (
                        "shared_resilience_v1",
                        state.shared_resilience is not None,
                    ),
                    ("threshold_project_v1", threshold_project is not None),
                    ("mutual_aid_v1", mutual_aid_enabled),
                    (
                        "price_coordination_v1",
                        price_coordination_enabled,
                    ),
                )
                if enabled
            ],
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
        *, actor_choices: Mapping[str,str] | None = None,
    ) -> StepResult:
        state = self.get_state()
        actor_choices=dict(actor_choices or {})
        if actor_choices:
            allowed={sid:{"balanced","inventory","reliability","no_credit"} for sid in state.supply_chain.supplier_ids} if self.config.data.get("supply_chain",{}).get("strategic_policy") else {}
            if self.config.data.get("autonomous_market",{}).get("government",{}).get("strategic_policy"):
                from .government_strategy import OPTIONS
                allowed["government"] = set(OPTIONS)
            if self.config.data.get("four_actor_policies"):
                from .actor_policies import CONSUMER_OPTIONS
                allowed["consumers"] = set(CONSUMER_OPTIONS)
            if any(k not in allowed or v not in allowed[k] for k,v in actor_choices.items()):
                raise ValueError("unknown strategic actor or policy option")
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
        if actor_choices: raw_joint_hash=sha256_hash({"company_actions":raw_joint_hash,"actor_choices":actor_choices})
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

        result = self._transition(state, step_id, actions, joint_hash,actor_choices)
        self.assert_invariants(result.state_after)
        self._state = result.state_after
        self._step_cache[step_id] = (raw_joint_hash, result)
        self._action_registry.update(pending_action_ids)
        return result

    def counterfactual_without_public_resilience(
        self,
        state: MarketState,
        joint_action: Mapping[str, CompanyAction | Mapping[str, Any]],
        *, actor_choices: Mapping[str,str] | None = None,
    ) -> StepResult:
        """Settle the same round with inherited public stock set to zero.

        This is a read-only shadow evaluation.  It uses the same episode,
        round, actions and component RNG, and it keeps current contributions
        in the action because they only create protection for the next round.
        The returned transition is never installed into this environment.
        """

        if state.shared_resilience is None:
            raise StateInvariantError(
                "public-resilience counterfactual requires cooperation state"
            )
        actions = self._validate_joint_action(state, joint_action)
        zero_shared = replace(
            state.shared_resilience,
            industry_resilience_ppm=0,
        )
        shadow_state = replace(
            state,
            shared_resilience=zero_shared,
            state_hash="",
        )
        shadow_state = replace(
            shadow_state,
            state_hash=state_hash(shadow_state.to_dict()),
        )
        shadow = MarketEnv(self.config)
        shadow.load_state(shadow_state)
        return shadow.step(
            f"{state.episode_id}:{state.round}:{state.state_version}",
            actions,
            actor_choices=actor_choices,
        )

    def _transition(
        self,
        state: MarketState,
        step_id: str,
        actions: Mapping[str, CompanyAction],
        joint_hash: str,
        actor_choices: Mapping[str,str] | None = None,
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
        shared_cfg = self.config.mapping("shared_resilience")
        strategic_cfg = self.config.data.get("strategic_market")
        supply_chain_cfg = self.config.data.get("supply_chain")
        welfare_cfg = self.config.data.get("welfare_accounting")
        autonomous_cfg = self.config.data.get("autonomous_market")
        government_action = decide_government(state, autonomous_cfg["government"], (actor_choices or {}).get("government")) if autonomous_cfg else None
        from .government_strategy import matched_support, rebates, unpack, learn
        government_support = matched_support(government_action, actions) if government_action else {}
        active_company_ids = (
            set(state.strategic_market.active_company_ids)
            if state.strategic_market is not None
            else set(state.company_ids)
        )
        settled_supply_chain: SupplyChainState | None = None
        procurement_capacity_by_company: dict[str, int] = {}
        procurement_price_by_company: dict[str, int] = {}
        if state.supply_chain is not None:
            if not isinstance(supply_chain_cfg, Mapping):
                raise StateInvariantError("supply chain state requires config")
            (
                settled_supply_chain,
                procurement_capacity_by_company,
                procurement_price_by_company,
            ) = settle_procurement(
                previous=state.supply_chain,
                companies=tuple(
                    company
                    for company in state.companies
                    if company.company_id in active_company_ids
                ),
                actions=actions,
                config=supply_chain_cfg,
                fixed_overhead_cents=int(operating_cfg["fixed_overhead_cents"]),
                rounds_remaining=state.rounds_remaining,
                actor_choices=actor_choices,
            )
        material_payments = ({o.company_id: o.material.payment_cents for o in settled_supply_chain.last_procurement_outcomes}
                             if cash_accounting(supply_chain_cfg) else {})
        current_industry_resilience = (
            state.shared_resilience.industry_resilience_ppm
            if state.shared_resilience is not None
            else 0
        )
        current_public_protection = _ppm_mul(
            int(shared_cfg["public_protection_weight_ppm"]),
            current_industry_resilience,
        )
        current_threshold_project = (
            state.strategic_market.threshold_project
            if state.strategic_market is not None
            else None
        )
        if current_threshold_project is not None:
            current_public_protection = _clip(
                current_public_protection
                + current_threshold_project.public_protection_bonus_ppm,
                0,
                PPM,
            )
        project_supply_multiplier = (
            PPM - current_threshold_project.supply_cost_reduction_ppm
            if current_threshold_project is not None
            else PPM
        )

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
        average_offered_price = (
            _round_ratio(
                sum(actions[company_id].price_cents for company_id in active_company_ids),
                len(active_company_ids),
            )
            if active_company_ids
            else state.market.price_anchor_cents
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
        # v7 may split each preference segment into explicit willingness-to-pay
        # cohorts.  Older configs create exactly one identically named group,
        # preserving their RNG keys and transition hashes.
        consumer_groups: dict[str, dict[str, int | str | None]] = {}
        for segment_name, demand in segment_demand.items():
            segment = segment_definitions[segment_name]
            wtp_distribution = segment.get("wtp_distribution")
            if isinstance(wtp_distribution, Mapping):
                weights = {
                    str(key): int(value)
                    for key, value in wtp_distribution["weights_ppm"].items()
                }
                offsets = {
                    str(key): int(value)
                    for key, value in wtp_distribution["offsets_cents"].items()
                }
                cohort_demand = _allocate_integer(demand, weights)
                for cohort_name in sorted(weights):
                    group_name = f"{segment_name}::{cohort_name}"
                    consumer_groups[group_name] = {
                        "segment_name": segment_name,
                        "demand_orders": cohort_demand[cohort_name],
                        "wtp_cents": max(
                            0,
                            state.market.price_anchor_cents
                            + offsets[cohort_name],
                        ),
                    }
            else:
                consumer_groups[segment_name] = {
                    "segment_name": segment_name,
                    "demand_orders": demand,
                    "wtp_cents": None,
                }

        company_runtime: dict[str, dict[str, int | CompanyIncident | None]] = {}
        utilities: dict[str, dict[str, float]] = {}
        max_event_reduction = int(event_cfg["resilience_max_reduction_ppm"])

        for company in state.companies:
            company_id = company.company_id
            action = actions[company_id]
            if company_id not in active_company_ids:
                continue
            resilience = company.risk.resilience_ppm
            effective_resilience = PPM - _ppm_mul(
                PPM - resilience,
                PPM - current_public_protection,
            )
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
                        effective_resilience,
                        max_event_reduction,
                    ),
                )
                capacity_multiplier = _ppm_mul(
                    capacity_multiplier,
                    self._protected_loss_multiplier(
                        event.capacity_multiplier_ppm,
                        effective_resilience,
                        max_event_reduction,
                    ),
                )
                advertising_multiplier = _ppm_mul(
                    advertising_multiplier,
                    self._protected_loss_multiplier(
                        event.advertising_multiplier_ppm,
                        effective_resilience,
                        max_event_reduction,
                    ),
                )
                reduction = PPM - _ppm_mul(
                    max_event_reduction, effective_resilience
                )
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
            actual_unit_cost = _ppm_mul(
                actual_unit_cost, project_supply_multiplier
            )
            if state.supply_chain is not None:
                assert isinstance(supply_chain_cfg, Mapping)
                baseline_input_price = int(
                    supply_chain_cfg["baseline_input_price_cents"]
                )
                input_share = int(
                    supply_chain_cfg["downstream_input_cost_share_ppm"]
                )
                supplier_price = procurement_price_by_company[company_id]
                supplier_price_index = _round_ratio(
                    supplier_price * PPM,
                    baseline_input_price,
                )
                supplier_cost_multiplier = (
                    PPM - input_share
                    + _ppm_mul(input_share, supplier_price_index)
                )
                if cash_accounting(supply_chain_cfg):
                    supplier_cost_multiplier = PPM - input_share
                actual_unit_cost = _ppm_mul(
                    actual_unit_cost, supplier_cost_multiplier
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
            if state.supply_chain is not None:
                effective_capacity = min(
                    effective_capacity,
                    procurement_capacity_by_company[company_id],
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
                0, company.financial.cash_balance_cents - action.fixed_spend_cents - material_payments.get(company_id, 0)
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

            seller_relative_price_signal = _clip(
                _round_ratio(
                    (average_offered_price - action.price_cents) * PPM,
                    int(choice_cfg["price_scale_cents"]),
                ),
                int(choice_cfg["relative_price_signal_min_ppm"]),
                int(choice_cfg["relative_price_signal_max_ppm"]),
            )
            utilities[company_id] = {}
            for group_name, group in consumer_groups.items():
                segment_name = str(group["segment_name"])
                segment = segment_definitions[segment_name]
                coefficients = segment["coefficients_ppm"]
                if self.config.data.get("four_actor_policies"):
                    from .actor_policies import consumer_coefficients
                    coefficients = consumer_coefficients(coefficients,(actor_choices or {}).get("consumers","balanced"))
                price_signal = seller_relative_price_signal
                if isinstance(strategic_cfg, Mapping):
                    reference_price = (
                        int(group["wtp_cents"])
                        if group["wtp_cents"] is not None
                        else state.market.price_anchor_cents
                    )
                    absolute_price_signal = _clip(
                        _round_ratio(
                            (reference_price - action.price_cents) * PPM,
                            int(choice_cfg["price_scale_cents"]),
                        ),
                        int(choice_cfg["relative_price_signal_min_ppm"]),
                        int(choice_cfg["relative_price_signal_max_ppm"]),
                    )
                    price_signal = (
                        _ppm_mul(
                            seller_relative_price_signal,
                            int(strategic_cfg["relative_price_weight_ppm"]),
                        )
                        + _ppm_mul(
                            absolute_price_signal,
                            int(strategic_cfg["absolute_price_weight_ppm"]),
                        )
                    )
                utility_noise_rng = self._rng(
                    state,
                    "consumer_utility_noise",
                    f"{company_id}|{group_name}",
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
                        price_signal,
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
                utilities[company_id][group_name] = utility_ppm / PPM

        initial_assignments: dict[str, dict[str, int]] = {
            company_id: {group_name: 0 for group_name in consumer_groups}
            for company_id in state.company_ids
        }
        no_purchase_orders = 0
        no_purchase_by_group = {}
        temperature = int(choice_cfg["temperature_ppm"]) / PPM
        for group_name, group in consumer_groups.items():
            demand = int(group["demand_orders"])
            segment_name = str(group["segment_name"])
            values = {
                company_id: utilities[company_id][group_name]
                for company_id in active_company_ids
                if (
                    group["wtp_cents"] is None
                    or actions[company_id].price_cents
                    <= int(group["wtp_cents"])
                )
            }
            values["outside"] = (
                int(segment_definitions[segment_name]["outside_utility_ppm"]) / PPM
            )
            if (actor_choices or {}).get("consumers")=="cautious":
                values["outside"] += 0.5
            probabilities = self._softmax(values, temperature)
            allocation = _allocate_integer(demand, probabilities)
            no_purchase_orders += allocation["outside"]
            no_purchase_by_group[group_name] = allocation["outside"]
            for company_id in state.company_ids:
                initial_assignments[company_id][group_name] = allocation.get(
                    company_id, 0
                )

        initial_fulfilled: dict[str, dict[str, int]] = {}
        attempted: dict[str, dict[str, int]] = {}
        remaining_caps: dict[str, int] = {}
        for company_id in state.company_ids:
            assigned = initial_assignments[company_id]
            total_assigned = sum(assigned.values())
            cap = (
                int(company_runtime[company_id]["fulfillment_cap_orders"])
                if company_id in company_runtime
                else 0
            )
            fulfill_total = min(total_assigned, cap)
            fulfilled = (
                _allocate_integer(fulfill_total, assigned)
                if total_assigned
                else {group_name: 0 for group_name in consumer_groups}
            )
            initial_fulfilled[company_id] = fulfilled
            attempted[company_id] = {
                segment: assigned[segment] - fulfilled[segment]
                for segment in consumer_groups
            }
            remaining_caps[company_id] = cap - fulfill_total

        attempted_before_mutual_aid = {
            company_id: dict(by_segment)
            for company_id, by_segment in attempted.items()
        }
        mutual_aid_received = {
            company_id: 0 for company_id in state.company_ids
        }
        mutual_aid_received_by_group = {
            company_id: {group_name: 0 for group_name in consumer_groups}
            for company_id in state.company_ids
        }
        mutual_aid_provided = {
            company_id: 0 for company_id in state.company_ids
        }
        mutual_aid_transfers: list[MutualAidTransfer] = []
        mutual_aid_enabled = bool(
            state.strategic_market is not None
            and state.strategic_market.mutual_aid_enabled
        )
        mutual_aid_fee_per_order = 0
        if mutual_aid_enabled:
            if not isinstance(strategic_cfg, Mapping):
                raise StateInvariantError("mutual aid requires strategic config")
            mutual_aid_cfg = strategic_cfg.get("mutual_aid")
            if not isinstance(mutual_aid_cfg, Mapping):
                raise StateInvariantError("mutual aid config is missing")
            mutual_aid_fee_per_order = int(
                mutual_aid_cfg["fee_per_order_cents"]
            )
            for donor_id in sorted(active_company_ids):
                donor_action = actions[donor_id]
                recipient_id = donor_action.mutual_aid_partner_company_id
                offered = int(
                    donor_action.mutual_aid_capacity_offer_orders or 0
                )
                if recipient_id is None or offered <= 0:
                    continue
                if recipient_id not in active_company_ids:
                    continue
                if cash_accounting(supply_chain_cfg):
                    donor_runtime = company_runtime[donor_id]
                    recipient_runtime = company_runtime[recipient_id]
                    # Both incremental legs must be self-financing. Existing
                    # own-order financial caps cannot secure a different fee.
                    donor_margin = mutual_aid_fee_per_order - donor_runtime["actual_unit_cost_cents"] - donor_runtime["fulfillment_cost_per_order_cents"]
                    recipient_margin = recipient_action_price = actions[recipient_id].price_cents
                    recipient_margin -= mutual_aid_fee_per_order + _round_ratio(recipient_action_price * recipient_runtime["refund_rate_ppm"], PPM)
                    if donor_margin < 0 or recipient_margin < 0:
                        continue
                recipient_action = actions[recipient_id]
                requested = int(
                    recipient_action.mutual_aid_capacity_request_orders or 0
                )
                if (
                    recipient_action.mutual_aid_partner_company_id != donor_id
                    or requested <= 0
                ):
                    continue
                unmet = sum(attempted[recipient_id].values())
                fulfilled_orders = min(
                    offered,
                    requested,
                    remaining_caps[donor_id],
                    unmet,
                )
                if fulfilled_orders <= 0:
                    continue
                fulfilled_by_segment = _allocate_integer(
                    fulfilled_orders, attempted[recipient_id]
                )
                for segment_name, quantity in fulfilled_by_segment.items():
                    attempted[recipient_id][segment_name] -= quantity
                    mutual_aid_received_by_group[recipient_id][
                        segment_name
                    ] += quantity
                remaining_caps[donor_id] -= fulfilled_orders
                mutual_aid_received[recipient_id] += fulfilled_orders
                mutual_aid_provided[donor_id] += fulfilled_orders
                mutual_aid_transfers.append(
                    MutualAidTransfer(
                        donor_company_id=donor_id,
                        recipient_company_id=recipient_id,
                        fulfilled_orders=fulfilled_orders,
                        fee_per_order_cents=mutual_aid_fee_per_order,
                        total_transfer_fee_cents=(
                            fulfilled_orders * mutual_aid_fee_per_order
                        ),
                    )
                )

        received = {company_id: 0 for company_id in state.company_ids}
        received_by_group = {
            company_id: {group_name: 0 for group_name in consumer_groups}
            for company_id in state.company_ids
        }
        own_lost = {company_id: 0 for company_id in state.company_ids}
        lost_after_stockout = 0
        for origin in sorted(state.company_ids):
            for group_name in sorted(consumer_groups):
                quantity = attempted[origin][group_name]
                if quantity <= 0:
                    continue
                segment_name = str(consumer_groups[group_name]["segment_name"])
                candidates = [
                    company_id
                    for company_id in active_company_ids
                    if company_id != origin
                    and remaining_caps[company_id] > 0
                    and (
                        consumer_groups[group_name]["wtp_cents"] is None
                        or actions[company_id].price_cents
                        <= int(consumer_groups[group_name]["wtp_cents"])
                    )
                ]
                values = {
                    company_id: utilities[company_id][group_name]
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
                    received_by_group[company_id][group_name] += accepted
                    remaining_caps[company_id] -= accepted
                    recovered += accepted
                batch_lost = quantity - recovered
                own_lost[origin] += batch_lost
                lost_after_stockout += batch_lost

        sales = {
            company_id: sum(initial_fulfilled[company_id].values())
            + received[company_id]
            + mutual_aid_received[company_id]
            for company_id in state.company_ids
        }
        sales_by_company_group = {
            company_id: {
                group_name: (
                    initial_fulfilled[company_id][group_name]
                    + received_by_group[company_id][group_name]
                    + mutual_aid_received_by_group[company_id][group_name]
                )
                for group_name in consumer_groups
            }
            for company_id in state.company_ids
        }
        exact_consumer_surplus = sum(
            quantity
            * max(
                0,
                int(consumer_groups[group_name]["wtp_cents"])
                - actions[company_id].price_cents,
            )
            for company_id, by_group in sales_by_company_group.items()
            for group_name, quantity in by_group.items()
            if consumer_groups[group_name]["wtp_cents"] is not None
        )
        total_sales = sum(sales.values())
        consumer_decisions = ()
        if autonomous_cfg:
            consumer_decisions = tuple(ConsumerDecisionAudit(
                group_id=group_id, settled_round=state.round, demand_orders=int(group["demand_orders"]),
                unit_budget_cents=int(group["wtp_cents"]), voluntary_no_purchase_orders=no_purchase_by_group[group_id],
                stockout_orders=int(group["demand_orders"])-no_purchase_by_group[group_id]-sum(sales_by_company_group[cid][group_id] for cid in state.company_ids),
                purchases_by_company=tuple((cid,sales_by_company_group[cid][group_id]) for cid in state.company_ids),
                posted_prices_cents=tuple((cid,actions[cid].price_cents) for cid in state.company_ids),
                spending_cents=sum(actions[cid].price_cents*sales_by_company_group[cid][group_id] for cid in state.company_ids),
                refund_cents=sum(_round_ratio(actions[cid].price_cents*company_runtime[cid]["refund_rate_ppm"],PPM)*sales_by_company_group[cid][group_id] for cid in active_company_ids),
            ) for group_id,group in sorted(consumer_groups.items()))
            exact_consumer_surplus += sum(d.refund_cents for d in consumer_decisions)
            consumer_decisions = rebates(government_action, consumer_decisions)
            exact_consumer_surplus += sum(d.government_rebate_cents or 0 for d in consumer_decisions)
        shares = (
            _allocate_integer(PPM, sales)
            if total_sales > 0
            else {company_id: 0 for company_id in state.company_ids}
        )

        price_coordination_enabled = bool(
            state.strategic_market is not None
            and state.strategic_market.price_coordination_enabled
        )
        coordination_specs: list[dict[str, Any]] = []
        regulatory_fine_assessed = {
            company_id: 0 for company_id in state.company_ids
        }
        if price_coordination_enabled:
            if not isinstance(strategic_cfg, Mapping):
                raise StateInvariantError(
                    "price coordination requires strategic config"
                )
            coordination_cfg = strategic_cfg.get("price_coordination")
            if not isinstance(coordination_cfg, Mapping):
                raise StateInvariantError(
                    "price coordination config is missing"
                )
            tolerance = int(coordination_cfg["adherence_tolerance_cents"])
            for company_a_id in sorted(active_company_ids):
                action_a = actions[company_a_id]
                company_b_id = action_a.price_coordination_partner_company_id
                target = action_a.price_coordination_target_cents
                if (
                    company_b_id is None
                    or company_b_id not in active_company_ids
                    or company_a_id >= company_b_id
                    or target is None
                ):
                    continue
                action_b = actions[company_b_id]
                if (
                    action_b.price_coordination_partner_company_id
                    != company_a_id
                    or action_b.price_coordination_target_cents != target
                ):
                    continue
                adhered_a = abs(action_a.price_cents - target) <= tolerance
                adhered_b = abs(action_b.price_cents - target) <= tolerance
                if adhered_a and adhered_b:
                    status = PriceCoordinationStatus.HONORED
                elif (
                    not adhered_a
                    and adhered_b
                    and action_a.price_cents < target - tolerance
                ):
                    status = PriceCoordinationStatus.UNDERCUT_BY_A
                elif (
                    adhered_a
                    and not adhered_b
                    and action_b.price_cents < target - tolerance
                ):
                    status = PriceCoordinationStatus.UNDERCUT_BY_B
                else:
                    status = PriceCoordinationStatus.MUTUAL_DEVIATION
                markup_signal = _clip(
                    _round_ratio(
                        max(0, target - state.market.price_anchor_cents)
                        * PPM,
                        max(1, state.market.price_anchor_cents),
                    ),
                    0,
                    PPM,
                )
                detection_probability = _clip(
                    int(coordination_cfg["base_detection_probability_ppm"])
                    + _ppm_mul(
                        markup_signal,
                        int(coordination_cfg["markup_detection_weight_ppm"]),
                    )
                    + _ppm_mul(
                        state.strategic_market.regulatory_pressure_ppm,
                        int(
                            coordination_cfg[
                                "regulatory_pressure_detection_weight_ppm"
                            ]
                        ),
                    ),
                    0,
                    PPM,
                )
                detection_rng = self._rng(
                    state,
                    "price_coordination_detection",
                    f"{company_a_id}|{company_b_id}",
                    summary=random_summary,
                )
                if government_action is not None:
                    detection_probability = min(PPM, detection_probability + government_action.detection_boost_ppm) if len(coordination_specs) < government_action.inspection_cases else 0
                detected = detection_rng.uniform() < (
                    detection_probability / PPM
                )
                if detected:
                    for company_id in (company_a_id, company_b_id):
                        consumer_revenue = actions[company_id].price_cents * sales[
                            company_id
                        ]
                        regulatory_fine_assessed[company_id] += int(
                            coordination_cfg["base_fine_cents"]
                        ) + _ppm_mul(
                            consumer_revenue,
                            int(coordination_cfg["fine_revenue_share_ppm"]),
                        )
                coordination_specs.append(
                    {
                        "company_a_id": company_a_id,
                        "company_b_id": company_b_id,
                        "target_price_cents": target,
                        "company_a_actual_price_cents": action_a.price_cents,
                        "company_b_actual_price_cents": action_b.price_cents,
                        "company_a_adhered": adhered_a,
                        "company_b_adhered": adhered_b,
                        "status": status,
                        "detection_probability_ppm": detection_probability,
                        "detected": detected,
                    }
                )

        if government_action and government_action.strategic_policy:
            multiplier = unpack(government_action.strategic_policy)["fine_multiplier_ppm"]
            regulatory_fine_assessed = {cid:value*multiplier//PPM for cid,value in regulatory_fine_assessed.items()}
        next_companies: list[CompanyState] = []
        history_window = int(self.config.get("company_initial", "history_window"))
        for company in state.companies:
            company_id = company.company_id
            action = actions[company_id]
            if company_id not in active_company_ids:
                next_companies.append(
                    replace(
                        company,
                        financial=replace(
                            company.financial,
                            round_revenue_cents=0,
                            round_variable_cost_cents=0,
                            round_fixed_spend_cents=0,
                            round_incident_cost_cents=0,
                            round_operating_cost_cents=0,
                            round_profit_cents=0,
                            round_material_payment_cents=(0 if cash_accounting(supply_chain_cfg) else None),
                            round_government_support_cents=(0 if autonomous_cfg else None),
                            round_regulatory_fine_cents=(
                                0 if price_coordination_enabled else None
                            ),
                        ),
                        commercial=replace(
                            company.commercial,
                            market_share_ppm=0,
                            potential_demand_orders=0,
                            sales_orders=0,
                            attempted_unfulfilled_orders=0,
                            orders_received_from_redistribution=0,
                            orders_lost_after_redistribution=0,
                            mutual_aid_fulfilled_orders=(
                                0 if mutual_aid_enabled else None
                            ),
                            mutual_aid_provided_orders=(
                                0 if mutual_aid_enabled else None
                            ),
                        ),
                        operations=replace(
                            company.operations,
                            effective_capacity_orders=0,
                            financial_capacity_orders=0,
                            capacity_utilization_ppm=0,
                        ),
                        risk=replace(company.risk, active_incident=None),
                        history=replace(
                            company.history,
                            last_action_id=action.action_id,
                            last_action=action,
                            recent_profit_cents=(
                                company.history.recent_profit_cents + (0,)
                            )[-int(self.config.get("company_initial", "history_window")):],
                            recent_market_share_ppm=(
                                company.history.recent_market_share_ppm + (0,)
                            )[-int(self.config.get("company_initial", "history_window")):],
                        ),
                    )
                )
                continue
            runtime = company_runtime[company_id]
            company_sales = sales[company_id]
            actual_unit_cost = int(runtime["actual_unit_cost_cents"])
            internally_fulfilled_orders = (
                company_sales
                - mutual_aid_received[company_id]
                + mutual_aid_provided[company_id]
            )
            transfer_fee_income = (
                mutual_aid_provided[company_id] * mutual_aid_fee_per_order
            )
            transfer_fee_expense = (
                mutual_aid_received[company_id] * mutual_aid_fee_per_order
            )
            revenue = (
                action.price_cents * company_sales + transfer_fee_income
            )
            variable_cost = (
                actual_unit_cost * internally_fulfilled_orders
                + transfer_fee_expense
                + material_payments.get(company_id, 0)
            )
            refund_per_order = _round_ratio(
                action.price_cents * int(runtime["refund_rate_ppm"]), PPM
            )
            refund_cost = refund_per_order * company_sales
            operating_cost = int(runtime["operating_overhead_cents"]) + (
                int(runtime["fulfillment_cost_per_order_cents"])
                * internally_fulfilled_orders
            )
            pre_fine_profit = (
                revenue
                - variable_cost
                - action.fixed_spend_cents
                - refund_cost
                - operating_cost
                + government_support.get(company_id, 0)
            )
            regulatory_fine = min(
                regulatory_fine_assessed[company_id],
                max(
                    0,
                    company.financial.cash_balance_cents + pre_fine_profit,
                ),
            )
            round_profit = pre_fine_profit - regulatory_fine
            next_cash = company.financial.cash_balance_cents + round_profit

            assigned_total = sum(initial_assignments[company_id].values())
            attempted_total = sum(
                attempted_before_mutual_aid[company_id].values()
            )
            unfulfilled_after_mutual_aid = max(
                0,
                attempted_total - mutual_aid_received[company_id],
            )
            unfulfilled_rate = (
                _round_ratio(
                    unfulfilled_after_mutual_aid * PPM,
                    assigned_total,
                )
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
                    round_material_payment_cents=(material_payments.get(company_id, 0) if cash_accounting(supply_chain_cfg) else None),
                    round_government_support_cents=(government_support.get(company_id, 0) if autonomous_cfg else None),
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
                    round_regulatory_fine_cents=(
                        regulatory_fine
                        if price_coordination_enabled
                        else None
                    ),
                ),
                commercial=CommercialState(
                    price_cents=action.price_cents,
                    market_share_ppm=shares[company_id],
                    potential_demand_orders=assigned_total,
                    sales_orders=company_sales,
                    attempted_unfulfilled_orders=attempted_total,
                    orders_received_from_redistribution=received[company_id],
                    orders_lost_after_redistribution=own_lost[company_id],
                    mutual_aid_fulfilled_orders=(
                        mutual_aid_received[company_id]
                        if mutual_aid_enabled
                        else None
                    ),
                    mutual_aid_provided_orders=(
                        mutual_aid_provided[company_id]
                        if mutual_aid_enabled
                        else None
                    ),
                ),
                operations=OperationsState(
                    base_capacity_orders=base_capacity_next,
                    effective_capacity_orders=int(runtime["effective_capacity_orders"]),
                    financial_capacity_orders=int(runtime["financial_capacity_orders"]),
                    capacity_utilization_ppm=(
                        _round_ratio(
                            internally_fulfilled_orders * PPM,
                            int(runtime["effective_capacity_orders"]),
                        )
                        if int(runtime["effective_capacity_orders"])
                        else 0
                    ),
                    base_unit_cost_cents=company.operations.base_unit_cost_cents,
                    actual_unit_cost_cents=actual_unit_cost,
                    procurement_requested_orders=(
                        company.operations.base_capacity_orders
                        if state.supply_chain is not None
                        else None
                    ),
                    procurement_fulfilled_orders=(
                        procurement_capacity_by_company[company_id]
                        if state.supply_chain is not None
                        else None
                    ),
                    procurement_unit_input_price_cents=(
                        procurement_price_by_company[company_id]
                        if state.supply_chain is not None
                        else None
                    ),
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

        actual_fines = {
            company.company_id: int(
                company.financial.round_regulatory_fine_cents or 0
            )
            for company in next_companies
        }
        price_coordination_outcomes = tuple(
            PriceCoordinationOutcome(
                **spec,
                fine_by_company_cents=tuple(
                    (
                        company_id,
                        actual_fines[company_id],
                    )
                    for company_id in (
                        spec["company_a_id"],
                        spec["company_b_id"],
                    )
                ),
            )
            for spec in coordination_specs
        )

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

        contributions = {
            company_id: int(
                actions[company_id].shared_resilience_contribution_cents or 0
            )
            for company_id in state.company_ids
        }
        total_contribution = sum(contributions.values())
        next_shared_resilience: SharedResilienceState | None = None
        next_public_protection = 0
        if state.shared_resilience is not None:
            next_industry_resilience = _clip(
                _ppm_mul(
                    int(shared_cfg["retention_ppm"]),
                    current_industry_resilience,
                )
                + _ppm_mul(
                    int(shared_cfg["contribution_input_weight_ppm"]),
                    _sat_ppm(
                        total_contribution,
                        int(shared_cfg["contribution_scale_cents"]),
                    ),
                ),
                0,
                PPM,
            )
            next_shared_resilience = SharedResilienceState(
                industry_resilience_ppm=next_industry_resilience,
                last_total_contribution_cents=total_contribution,
                last_contribution_by_company_cents=tuple(
                    sorted(contributions.items())
                ),
            )
            next_public_protection = _ppm_mul(
                int(shared_cfg["public_protection_weight_ppm"]),
                next_industry_resilience,
            )

        next_threshold_project, project_refunds, _ = settle_threshold_project(
            previous=current_threshold_project,
            settled_round=state.round,
            company_ids=state.company_ids,
            actions=actions,
            config=strategic_cfg if isinstance(strategic_cfg, Mapping) else {},
        )
        next_companies = list(
            apply_threshold_project_refunds(next_companies, project_refunds)
        )
        if next_threshold_project is not None:
            next_public_protection = _clip(
                next_public_protection
                + next_threshold_project.public_protection_bonus_ppm,
                0,
                PPM,
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

        if cash_accounting(supply_chain_cfg):
            settled_supply_chain = close_materials(settled_supply_chain, {
                cid: sales[cid] - mutual_aid_received[cid] + mutual_aid_provided[cid] for cid in state.company_ids})
        next_supply_chain = settled_supply_chain
        if not terminal and settled_supply_chain is not None:
            assert isinstance(supply_chain_cfg, Mapping)
            quoted_supply_chain = advance_supplier_quotes(
                settled=settled_supply_chain, config=supply_chain_cfg,
                observed_round=state.round,
            )
            if autonomous_cfg:
                quoted_supply_chain = advance_investment(quoted_supply_chain, autonomous_cfg["supplier_investment"], state.round, False, state.rounds_remaining-1)
            supplier_draws: dict[str, float] = {}
            for supplier in settled_supply_chain.suppliers:
                supplier_draws[supplier.supplier_id] = self._rng(
                    state,
                    "supplier_disruption",
                    supplier.supplier_id,
                    summary=random_summary,
                ).uniform()
            next_supply_chain = advance_supplier_availability(
                settled=quoted_supply_chain,
                uniform_draw_by_supplier=supplier_draws,
                config=supply_chain_cfg,
            )
        elif terminal and settled_supply_chain is not None and autonomous_cfg:
            next_supply_chain = advance_investment(settled_supply_chain, autonomous_cfg["supplier_investment"], state.round, True, 0)

        if not terminal:
            next_companies = [
                (
                    replace(
                        company,
                        risk=replace(
                            company.risk,
                            active_incident=self._maybe_generate_incident(
                                state,
                                company,
                                company.risk.active_incident,
                                random_summary,
                                public_protection_ppm=next_public_protection,
                            ),
                        ),
                    )
                    if company.company_id in active_company_ids
                    else company
                )
                for company in next_companies
            ]

        next_actual_supply = next_base_supply
        for event in next_events:
            next_actual_supply = _ppm_mul(
                next_actual_supply, event.supply_cost_multiplier_ppm
            )
        if next_threshold_project is not None:
            next_actual_supply = _ppm_mul(
                next_actual_supply,
                PPM - next_threshold_project.supply_cost_reduction_ppm,
            )
        average_paid_price = (
            _round_ratio(
                sum(actions[cid].price_cents * sales[cid] for cid in state.company_ids),
                total_sales,
            )
            if total_sales
            else 0
        )
        next_market = MarketSnapshot(
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
        )
        next_strategic_market = None
        newly_exited: tuple[str, ...] = ()
        if state.strategic_market is not None:
            if not isinstance(strategic_cfg, Mapping):
                raise StateInvariantError("v6 strategic state requires config")
            next_strategic_market, newly_exited = advance_strategic_market_state(
                previous=state.strategic_market,
                settled_round=state.round,
                companies=next_companies,
                actions=actions,
                market=next_market,
                config=strategic_cfg,
                fulfillment_cost_per_order_cents=int(
                    operating_cfg["fulfillment_cost_per_order_cents"]
                ),
            )
            next_strategic_market = replace(
                next_strategic_market,
                threshold_project=next_threshold_project,
                last_mutual_aid_transfers=tuple(mutual_aid_transfers),
            )
            if price_coordination_enabled:
                coordination_cfg = strategic_cfg.get("price_coordination")
                if not isinstance(coordination_cfg, Mapping):
                    raise StateInvariantError(
                        "price coordination config is missing"
                    )
                update_weight = int(
                    coordination_cfg["credibility_update_weight_ppm"]
                )
                credibility = dict(
                    state.strategic_market.coordination_credibility_by_company_ppm
                )
                for outcome in price_coordination_outcomes:
                    for company_id, adhered in (
                        (outcome.company_a_id, outcome.company_a_adhered),
                        (outcome.company_b_id, outcome.company_b_adhered),
                    ):
                        credibility[company_id] = _clip(
                            _ppm_mul(
                                credibility[company_id],
                                PPM - update_weight,
                            )
                            + _ppm_mul(
                                PPM if adhered else 0,
                                update_weight,
                            ),
                            0,
                            PPM,
                        )
                coordination_signal = max(
                    (
                        outcome.detection_probability_ppm
                        for outcome in price_coordination_outcomes
                    ),
                    default=0,
                )
                coordination_pressure = _ppm_mul(
                    coordination_signal,
                    int(
                        coordination_cfg[
                            "coordination_pressure_weight_ppm"
                        ]
                    ),
                )
                next_strategic_market = replace(
                    next_strategic_market,
                    regulatory_pressure_ppm=_clip(
                        next_strategic_market.regulatory_pressure_ppm
                        + coordination_pressure,
                        0,
                        PPM,
                    ),
                    coordination_credibility_by_company_ppm=tuple(
                        (company_id, credibility[company_id])
                        for company_id in state.company_ids
                    ),
                    last_price_coordination_outcomes=(
                        price_coordination_outcomes
                    ),
                )
            next_companies = list(
                liquidate_newly_exited(
                    next_companies, newly_exited, strategic_cfg
                )
            )
        next_welfare: WelfareAccountingState | None = None
        consumer_rebates = sum(d.government_rebate_cents or 0 for d in consumer_decisions)
        next_government = settle_government(state.government, government_action, sum(actual_fines.values()),
                                           sum(government_support.values()), consumer_rebates) if government_action else None
        if state.welfare_accounting is not None:
            if not isinstance(welfare_cfg, Mapping):
                raise StateInvariantError("welfare state requires config")
            next_welfare = advance_welfare_state(
                previous=state.welfare_accounting,
                companies=next_companies,
                supply_chain=next_supply_chain,
                consumer_surplus_cents=exact_consumer_surplus,
                government_fine_revenue_cents=sum(actual_fines.values()),
                enforcement_case_count=sum(
                    1 for outcome in price_coordination_outcomes if outcome.detected
                ),
                lost_after_stockout_orders=lost_after_stockout,
                voluntary_no_purchase_orders=no_purchase_orders,
                newly_exited_company_count=len(newly_exited),
                config=welfare_cfg,
                actual_enforcement_cost_cents=government_action.inspection_cost_cents if government_action else None,
                government_support_cents=sum(government_support.values())+consumer_rebates,
            )
        if next_government and next_welfare:
            next_government = learn(next_government, next_welfare.round_total_economic_welfare_cents)
        terminal_values: tuple[tuple[str, int], ...] = ()
        if terminal:
            terminal_values = tuple(
                (
                    company.company_id,
                    (
                        company.financial.cash_balance_cents
                        if next_strategic_market is not None
                        and next_strategic_market.lifecycle(
                            company.company_id
                        ).status
                        is CompanyOperatingStatus.EXITED
                        else self._terminal_value(company)
                    ),
                )
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
            market=next_market,
            consumer_segments=state.consumer_segments,
            risk_signals=next_signals,
            active_market_events=next_events,
            companies=tuple(next_companies),
            shared_resilience=next_shared_resilience,
            strategic_market=next_strategic_market,
            supply_chain=next_supply_chain,
            welfare_accounting=next_welfare,
            government=next_government,
            consumer_decisions=consumer_decisions,
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
            actor_choices=tuple(sorted((actor_choices or {}).items())),
        )

    def assert_invariants(self, state: MarketState) -> None:
        failures: list[str] = []
        if self.config.data.get("autonomous_market"):
            failures.extend(actor_failures(state, self.config.data))
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

        shared = state.shared_resilience
        if shared is not None:
            if not 0 <= shared.industry_resilience_ppm <= PPM:
                failures.append(
                    "industry resilience is outside [0, 1000000]"
                )
            by_company = dict(shared.last_contribution_by_company_cents)
            if set(by_company) != set(state.company_ids):
                failures.append(
                    "shared resilience contribution attribution is incomplete"
                )
            if any(value < 0 for value in by_company.values()):
                failures.append("shared resilience contribution is negative")
            if sum(by_company.values()) != shared.last_total_contribution_cents:
                failures.append(
                    "shared resilience contribution total is inconsistent"
                )

        supply_chain_cfg = self.config.data.get("supply_chain")
        supply_chain = state.supply_chain
        if supply_chain is None and isinstance(supply_chain_cfg, Mapping):
            failures.append("supply chain state is missing")
        if supply_chain is not None and not isinstance(supply_chain_cfg, Mapping):
            failures.append("supply chain state exists without config")
        if supply_chain is not None:
            if cash_accounting(supply_chain_cfg):
                failures.extend(accounting_failures(state, supply_chain_cfg))
            supplier_ids = supply_chain.supplier_ids
            if len(supplier_ids) != len(set(supplier_ids)):
                failures.append("supplier ids must be unique")
            sold_by_supplier = {supplier_id: 0 for supplier_id in supplier_ids}
            outcome_company_ids: set[str] = set()
            for outcome in supply_chain.last_procurement_outcomes:
                if outcome.company_id not in state.company_ids:
                    failures.append("procurement outcome company is unknown")
                if outcome.company_id in outcome_company_ids:
                    failures.append("duplicate procurement outcome")
                outcome_company_ids.add(outcome.company_id)
                if outcome.primary_supplier_id not in sold_by_supplier:
                    failures.append("procurement primary supplier is unknown")
                if (
                    outcome.backup_supplier_id is not None
                    and outcome.backup_supplier_id not in sold_by_supplier
                ):
                    failures.append("procurement backup supplier is unknown")
                if not 0 <= outcome.primary_supplier_share_ppm <= PPM:
                    failures.append("procurement share is outside bounds")
                allocations = dict(outcome.supplier_allocation_orders)
                if any(
                    supplier_id not in sold_by_supplier
                    for supplier_id in allocations
                ):
                    failures.append("procurement allocation supplier is unknown")
                if any(value < 0 for value in allocations.values()):
                    failures.append("procurement allocation is negative")
                if sum(allocations.values()) != outcome.fulfilled_orders:
                    failures.append("procurement fulfilled total conflicts")
                if outcome.fulfilled_orders + outcome.unfulfilled_orders != (
                    outcome.requested_orders
                ):
                    failures.append("procurement request does not close")
                for supplier_id, value in allocations.items():
                    if supplier_id in sold_by_supplier:
                        sold_by_supplier[supplier_id] += value
            upstream_surplus = 0
            for supplier in supply_chain.suppliers:
                from game_theory_agent.market.supplier_strategy import decode as decode_supplier
                supplier_ledger=decode_supplier(supplier)
                inventory_bound=supplier_ledger.get("inventory_orders",0)
                if state.terminal and supplier_ledger.get("audit"):
                    inventory_bound=max(inventory_bound,supplier_ledger["audit"]["opening_inventory_orders"])
                if not 0 <= supplier.reliability_ppm <= PPM:
                    failures.append("supplier reliability is outside bounds")
                if not 0 <= supplier.available_capacity_orders <= (
                    supplier.base_capacity_orders+inventory_bound
                ):
                    failures.append("supplier available capacity is invalid")
                if supplier.round_sales_orders != sold_by_supplier[supplier.supplier_id]:
                    failures.append("supplier sales attribution conflicts")
                settlement_price = supplier.unit_price_cents
                if pricing_enabled(supply_chain_cfg):
                    floor, ceiling = price_bounds(supplier, supply_chain_cfg)
                    if not floor <= supplier.unit_price_cents <= ceiling:
                        failures.append("supplier quote outside policy bounds")
                    if state.state_version > 0:
                        if supplier.last_settled_unit_price_cents is None:
                            failures.append("supplier settlement price missing")
                        else:
                            settlement_price = supplier.last_settled_unit_price_cents
                        if not floor <= settlement_price <= ceiling:
                            failures.append("supplier settlement price outside policy bounds")
                    decision = supplier.quote_decision
                    if decision is None and state.state_version > 0 and not state.terminal:
                        failures.append("supplier quote audit missing")
                    if decision is not None:
                        audited = replace(supplier, unit_price_cents=decision.previous_price_cents,
                                          round_sales_orders=decision.observed_sales_orders,
                                          available_capacity_orders=decision.observed_capacity_orders)
                        expected = decide_quote(supplier=audited, config=supply_chain_cfg, observed_round=decision.observed_round)
                        if (expected != decision or decision.quoted_price_cents != supplier.unit_price_cents
                            or not 0 <= decision.observed_sales_orders <= decision.observed_capacity_orders <= supplier.base_capacity_orders+((supply_chain_cfg.get("strategic_policy") or {}).get("max_inventory_orders",0) if state.terminal else (supplier_ledger.get("audit") or {}).get("opening_inventory_orders",0))):
                            failures.append("supplier quote audit conflicts")
                        if not state.terminal and (decision.applies_round != state.round
                            or decision.previous_price_cents != settlement_price
                            or decision.observed_sales_orders != supplier.round_sales_orders):
                            failures.append("supplier quote timing conflicts")
                elif supplier.quote_decision is not None or supplier.last_settled_unit_price_cents is not None:
                    failures.append("supplier autonomous fields without enabled policy")
                expected_profit = supplier.round_sales_orders * (settlement_price - supplier.unit_cost_cents) - ((supplier.account.investment_cents or 0) if supplier.account else 0)
                if not supplier.strategic_ledger and supplier.round_profit_cents != expected_profit:
                    failures.append("supplier round profit conflicts")
                upstream_surplus += supplier.round_profit_cents
                upstream_surplus += (supplier_ledger.get("audit") or {}).get("lender_profit_cents",0)
            if upstream_surplus != (
                supply_chain.round_upstream_producer_surplus_cents
            ):
                failures.append("upstream producer surplus conflicts")

        welfare_cfg = self.config.data.get("welfare_accounting")
        welfare = state.welfare_accounting
        if welfare is None and isinstance(welfare_cfg, Mapping):
            failures.append("welfare accounting state is missing")
        if welfare is not None and not isinstance(welfare_cfg, Mapping):
            failures.append("welfare accounting exists without config")
        if welfare is not None and state.state_version > 0:
            if welfare.round_government_net_budget_cents != (
                welfare.round_government_fine_revenue_cents
                - welfare.round_government_enforcement_cost_cents
                - ((state.government.round_matched_support_cents or 0)+(state.government.round_consumer_rebate_cents or 0)
                   if state.government and state.government.last_decision and state.government.last_decision.strategic_policy
                   else sum(v for _,v in state.government.last_decision.support_by_company_cents) if state.government and state.government.last_decision else 0)
            ):
                failures.append("government welfare ledger does not close")
            if welfare.round_service_continuity_orders != sum(
                item.commercial.sales_orders for item in state.companies
            ):
                failures.append("service continuity conflicts with sales")
            if welfare.round_voluntary_no_purchase_orders != (
                state.market.no_purchase_orders
            ):
                failures.append("voluntary no-purchase attribution conflicts")
            expected_welfare = (
                welfare.round_consumer_surplus_cents
                + welfare.round_downstream_producer_surplus_cents
                + welfare.round_upstream_producer_surplus_cents
                + welfare.round_government_net_budget_cents
                - welfare.round_stockout_externality_cents
                - welfare.round_business_exit_externality_cents
            )
            if welfare.round_total_economic_welfare_cents != expected_welfare:
                failures.append("total economic welfare does not close")
            if supply_chain is not None and (
                welfare.round_upstream_producer_surplus_cents
                != supply_chain.round_upstream_producer_surplus_cents
            ):
                failures.append("welfare upstream surplus conflicts")

        strategic = state.strategic_market
        strategic_cfg = self.config.data.get("strategic_market")
        if strategic is None and isinstance(strategic_cfg, Mapping):
            failures.append("v6 strategic market state is missing")
        if strategic is not None:
            lifecycle_ids = tuple(
                item.company_id for item in strategic.company_lifecycle
            )
            if lifecycle_ids != state.company_ids:
                failures.append("strategic lifecycle company order is inconsistent")
            if strategic.active_company_count != len(
                strategic.active_company_ids
            ):
                failures.append("strategic active company count is inconsistent")
            if not 0 <= strategic.hhi_ppm <= PPM:
                failures.append("strategic HHI is outside [0, 1000000]")
            if not 0 <= strategic.dominant_share_ppm <= PPM:
                failures.append("dominant share is outside [0, 1000000]")
            if not 0 <= strategic.regulatory_pressure_ppm <= PPM:
                failures.append("regulatory pressure is outside [0, 1000000]")
            if not 0 <= strategic.price_war_intensity_ppm <= PPM:
                failures.append("price war intensity is outside [0, 1000000]")
            if not 0 <= strategic.market_power_markup_ppm <= PPM:
                failures.append("market-power markup is outside [0, 1000000]")
            for lifecycle in strategic.company_lifecycle:
                if lifecycle.distress_streak < 0:
                    failures.append(
                        f"{lifecycle.company_id} distress streak is negative"
                    )
                if (
                    lifecycle.status is CompanyOperatingStatus.EXITED
                    and lifecycle.exit_round is None
                ):
                    failures.append(
                        f"{lifecycle.company_id} exited without exit round"
                    )
                company = state.company(lifecycle.company_id)
                if (
                    lifecycle.status is CompanyOperatingStatus.EXITED
                    and lifecycle.exit_round is not None
                    and lifecycle.exit_round < state.round - 1
                    and (
                        company.commercial.sales_orders != 0
                        or company.commercial.market_share_ppm != 0
                    )
                ):
                    failures.append(
                        f"{lifecycle.company_id} exited company still trades"
                    )
            project = strategic.threshold_project
            if project is not None:
                contributed = dict(project.contribution_by_company_cents)
                last = dict(project.last_contribution_by_company_cents)
                refunds = dict(project.last_refund_by_company_cents)
                if set(contributed) != set(state.company_ids):
                    failures.append("threshold project attribution is incomplete")
                if set(last) != set(state.company_ids):
                    failures.append("threshold project last contribution is incomplete")
                if set(refunds) != set(state.company_ids):
                    failures.append("threshold project refund attribution is incomplete")
                if any(value < 0 for value in contributed.values()):
                    failures.append("threshold project contribution is negative")
                if any(value < 0 for value in refunds.values()):
                    failures.append("threshold project refund is negative")
                if sum(contributed.values()) != (
                    project.accumulated_total_contribution_cents
                ):
                    failures.append("threshold project contribution total conflicts")
                if project.status.value == "succeeded":
                    if project.success_round is None or project.failure_round is not None:
                        failures.append("threshold project success markers conflict")
                    if project.accumulated_total_contribution_cents < (
                        project.required_total_contribution_cents
                    ):
                        failures.append("threshold project succeeded below threshold")
                if project.status.value == "failed":
                    if project.failure_round is None or project.success_round is not None:
                        failures.append("threshold project failure markers conflict")
            if not strategic.mutual_aid_enabled and (
                strategic.last_mutual_aid_transfers
            ):
                failures.append("mutual aid transfers exist while disabled")
            received_by_company = {
                company_id: 0 for company_id in state.company_ids
            }
            provided_by_company = {
                company_id: 0 for company_id in state.company_ids
            }
            for transfer in strategic.last_mutual_aid_transfers:
                if (
                    transfer.donor_company_id not in received_by_company
                    or transfer.recipient_company_id not in received_by_company
                ):
                    failures.append("mutual aid transfer company is unknown")
                    continue
                if transfer.donor_company_id == transfer.recipient_company_id:
                    failures.append("mutual aid transfer cannot target self")
                if transfer.fulfilled_orders <= 0:
                    failures.append("mutual aid transfer must be positive")
                if transfer.fee_per_order_cents < 0:
                    failures.append("mutual aid transfer fee is negative")
                if transfer.total_transfer_fee_cents != (
                    transfer.fulfilled_orders * transfer.fee_per_order_cents
                ):
                    failures.append("mutual aid transfer fee total conflicts")
                received_by_company[transfer.recipient_company_id] += (
                    transfer.fulfilled_orders
                )
                provided_by_company[transfer.donor_company_id] += (
                    transfer.fulfilled_orders
                )
            if strategic.mutual_aid_enabled and state.state_version > 0:
                for company in state.companies:
                    if company.commercial.mutual_aid_fulfilled_orders != (
                        received_by_company[company.company_id]
                    ):
                        failures.append(
                            f"{company.company_id} mutual aid received conflicts"
                        )
                    if company.commercial.mutual_aid_provided_orders != (
                        provided_by_company[company.company_id]
                    ):
                        failures.append(
                            f"{company.company_id} mutual aid provided conflicts"
                        )
            coordination_credibility = dict(
                strategic.coordination_credibility_by_company_ppm
            )
            if strategic.price_coordination_enabled:
                if set(coordination_credibility) != set(state.company_ids):
                    failures.append(
                        "price coordination credibility attribution is incomplete"
                    )
                if any(
                    value < 0 or value > PPM
                    for value in coordination_credibility.values()
                ):
                    failures.append(
                        "price coordination credibility is outside bounds"
                    )
            elif (
                coordination_credibility
                or strategic.last_price_coordination_outcomes
            ):
                failures.append(
                    "price coordination state exists while disabled"
                )
            outcome_fines = {
                company_id: 0 for company_id in state.company_ids
            }
            seen_coordination_companies: set[str] = set()
            for outcome in strategic.last_price_coordination_outcomes:
                participants = (outcome.company_a_id, outcome.company_b_id)
                if (
                    outcome.company_a_id not in outcome_fines
                    or outcome.company_b_id not in outcome_fines
                    or outcome.company_a_id >= outcome.company_b_id
                ):
                    failures.append(
                        "price coordination participant attribution conflicts"
                    )
                    continue
                if seen_coordination_companies.intersection(participants):
                    failures.append(
                        "company appears in multiple price coordination outcomes"
                    )
                seen_coordination_companies.update(participants)
                if not 0 <= outcome.detection_probability_ppm <= PPM:
                    failures.append(
                        "price coordination detection probability is outside bounds"
                    )
                fines = dict(outcome.fine_by_company_cents)
                if set(fines) != set(participants):
                    failures.append(
                        "price coordination fine attribution is incomplete"
                    )
                if any(value < 0 for value in fines.values()):
                    failures.append("price coordination fine is negative")
                if not outcome.detected and any(fines.values()):
                    failures.append(
                        "undetected price coordination has a regulatory fine"
                    )
                for company_id, value in fines.items():
                    outcome_fines[company_id] += value
            if strategic.price_coordination_enabled and state.state_version > 0:
                for company in state.companies:
                    if company.financial.round_regulatory_fine_cents != (
                        outcome_fines[company.company_id]
                    ):
                        failures.append(
                            f"{company.company_id} regulatory fine conflicts"
                        )

        for company in state.companies:
            if company.financial.cash_balance_cents < 0:
                failures.append(f"{company.company_id} cash is negative")
            if (company.financial.round_regulatory_fine_cents or 0) < 0:
                failures.append(f"{company.company_id} regulatory fine is negative")
            outsourced = company.commercial.mutual_aid_fulfilled_orders or 0
            provided = company.commercial.mutual_aid_provided_orders or 0
            internal_throughput = (
                company.commercial.sales_orders - outsourced + provided
            )
            just_exited = bool(
                strategic is not None
                and strategic.lifecycle(company.company_id).status
                is CompanyOperatingStatus.EXITED
                and strategic.lifecycle(company.company_id).exit_round
                == state.round - 1
            )
            if (
                not just_exited
                and internal_throughput
                > company.operations.effective_capacity_orders
            ):
                failures.append(f"{company.company_id} sales exceed effective capacity")
            if (
                not just_exited
                and internal_throughput
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
        *,
        public_protection_ppm: int = 0,
    ) -> CompanyIncident | None:
        if active_incident is not None:
            return active_incident
        cfg = self.config.mapping("incidents")
        effective_resilience = PPM - _ppm_mul(
            PPM - company.risk.resilience_ppm,
            PPM - public_protection_ppm,
        )
        probability = _ppm_mul(
            int(cfg["base_probability_ppm"]),
            PPM
            - _ppm_mul(
                int(cfg["probability_reduction_ppm"]), effective_resilience
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
            int(cfg["severity_reduction_ppm"]), effective_resilience
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
