"""Authoritative multi-round counterfactual rollouts for Stage 6.

This module is an evaluation oracle, not an incomplete-information Advisor.
It uses the real ``MarketEnv`` transition function to measure whether a
candidate would improve long-run enterprise value, cumulative profit and
risk-adjusted value under a deterministic set of opponent-response scenarios.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Any, Mapping, Sequence

from game_theory_agent.agents.personas import (
    PPM,
    PersonaProfile,
    PersonaUtilityEvaluator,
    PersonaUtilityTracker,
)
from game_theory_agent.belief import BeliefState
from game_theory_agent.gameplay import build_rule_action, build_terminal_rankings
from game_theory_agent.market import (
    CompanyAction,
    IncidentResponse,
    IncidentResponseMode,
    MarketConfig,
    MarketEnv,
    MarketState,
)
from game_theory_agent.market.protocols import sha256_hash, state_hash
from game_theory_agent.opponent import OpponentModelState

from .contracts import (
    CandidateEconomicAction,
    RolloutCandidateEvaluation,
    RolloutScenarioOutcome,
    StrategicActionCandidate,
    StrategicReliabilityPlan,
    compute_reliability_plan_hash,
)


def _mean(values: Sequence[int]) -> int:
    if not values:
        raise ValueError("mean requires at least one value")
    total = sum(values)
    sign = -1 if total < 0 else 1
    return sign * ((abs(total) + len(values) // 2) // len(values))


def _clip(value: int, low: int, high: int) -> int:
    return min(max(value, low), high)


def _enterprise_value(state: MarketState, company_id: str, config: MarketConfig) -> int:
    rankings = build_terminal_rankings(state, config)["composite"]
    return next(
        int(row["value_cents"])
        for row in rankings
        if row["company_id"] == company_id
    )


def _competitive_labels(
    state: MarketState, company_id: str, config: MarketConfig
) -> tuple[int, int, int]:
    rankings = build_terminal_rankings(state, config)["composite"]
    focal = next(item for item in rankings if item["company_id"] == company_id)
    best_opponent = max(
        int(item["value_cents"])
        for item in rankings
        if item["company_id"] != company_id
    )
    own_value = int(focal["value_cents"])
    return best_opponent, own_value - best_opponent, int(focal["rank"])


def _scenario_seed(state: MarketState, scenario_index: int) -> int:
    if scenario_index == 0:
        return state.episode_seed
    digest = sha256_hash(
        {
            "protocol": "strategic-rollout-scenario-seed-v1",
            "episode_id": state.episode_id,
            "episode_seed": state.episode_seed,
            "round": state.round,
            "scenario_index": scenario_index,
        }
    )
    return int(digest.rsplit(":", 1)[-1][-16:], 16)


def _state_for_scenario(state: MarketState, scenario_index: int) -> MarketState:
    seed = _scenario_seed(state, scenario_index)
    if seed == state.episode_seed:
        return state
    changed = replace(state, episode_seed=seed, state_hash="")
    return replace(changed, state_hash=state_hash(changed.to_dict()))


def _candidate_action_payload(action: CompanyAction) -> CandidateEconomicAction:
    return CandidateEconomicAction(
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


def generate_candidate_actions(
    config: MarketConfig,
    state: MarketState,
    company_id: str,
) -> tuple[StrategicActionCandidate, ...]:
    """Generate a small legal action set instead of searching continuous space."""

    company = state.company(company_id)
    env = MarketEnv(config)
    env.load_state(state)
    constraints = env.get_action_constraints(company_id, state.state_version)
    bounds = constraints["bounds"]
    current_price = company.commercial.price_cents
    price_min = int(bounds["price_cents"]["min"])
    price_max = int(bounds["price_cents"]["max"])
    cash = company.financial.cash_balance_cents
    standard_budget = min(500_000, max(0, cash // 20))
    capacity_budget = min(750_000, max(0, cash // 16))
    shared_value: int | None = 0 if state.shared_resilience is not None else None

    specs: list[tuple[str, CompanyAction, list[str], str]] = []

    def action(
        candidate_id: str,
        *,
        price: int = current_price,
        advertising: int = 0,
        service: int = 0,
        capacity: int = 0,
        resilience: int = 0,
        shared: int | None = shared_value,
        incident_response: IncidentResponse = IncidentResponse(),
    ) -> CompanyAction:
        return CompanyAction(
            action_id=(
                f"strategic-candidate:{state.episode_id}:{state.round}:"
                f"{company_id}:{candidate_id}"
            ),
            episode_id=state.episode_id,
            agent_id=company_id,
            round=state.round,
            state_version=state.state_version,
            price_cents=_clip(price, price_min, price_max),
            advertising_budget_cents=advertising,
            service_budget_cents=service,
            capacity_investment_cents=capacity,
            resilience_budget_cents=resilience,
            shared_resilience_contribution_cents=shared,
            incident_response=incident_response,
            strategy_summary=f"Stage 6 candidate: {candidate_id}",
        )

    specs.extend(
        [
            (
                "maintain",
                action("maintain"),
                ["price"],
                "保持当前报价并避免新增可选固定投入。",
            ),
            (
                "price_cut_small",
                action("price_cut_small", price=current_price - 500),
                ["price"],
                "小幅降价，检验份额提升能否覆盖毛利损失。",
            ),
            (
                "price_cut_large",
                action("price_cut_large", price=current_price - 1_000),
                ["price"],
                "较大幅度降价，用于显式测量价格战下行风险。",
            ),
            (
                "price_increase",
                action("price_increase", price=current_price + 500),
                ["price"],
                "提高报价，检验单位利润与需求损失之间的权衡。",
            ),
            (
                "increase_advertising",
                action("increase_advertising", advertising=standard_budget),
                ["advertising"],
                "用品牌认知投入替代单纯价格竞争。",
            ),
            (
                "increase_service",
                action("increase_service", service=standard_budget),
                ["service"],
                "提高当期服务并积累后续声誉资产。",
            ),
        ]
    )
    if constraints["capacity_investment_enabled"]:
        specs.append(
            (
                "increase_capacity",
                action("increase_capacity", capacity=capacity_budget),
                ["capacity"],
                "支付当期成本换取后续履约能力。",
            )
        )
    if constraints["resilience_investment_enabled"]:
        specs.append(
            (
                "increase_resilience",
                action("increase_resilience", resilience=standard_budget),
                ["resilience"],
                "支付当期成本降低后续事故和市场冲击损失。",
            )
        )
    if constraints["shared_resilience_contribution_enabled"]:
        specs.append(
            (
                "shared_resilience_contribution",
                action(
                    "shared_resilience_contribution",
                    shared=standard_budget,
                ),
                ["shared_resilience"],
                "为下一轮行业公共保护贡献，并保留搭便车对照。",
            )
        )
    incident = company.risk.active_incident
    if incident is not None:
        useful = min(
            int(constraints["max_useful_repair_budget_cents"]),
            max(0, cash // 4),
        )
        if useful > 0:
            mode = (
                IncidentResponseMode.FULL_REPAIR
                if useful >= incident.remaining_repair_cents
                else IncidentResponseMode.PARTIAL_REPAIR
            )
            specs.append(
                (
                    "repair_incident",
                    action(
                        "repair_incident",
                        incident_response=IncidentResponse(mode, useful),
                    ),
                    ["repair"],
                    "比较立即维修成本与持续运营损失。",
                )
            )

    candidates: list[StrategicActionCandidate] = []
    seen_actions: set[str] = set()
    for candidate_id, raw_action, dimensions, rationale in specs:
        validated = env.validate_action(raw_action, company_id)
        if not validated.valid or validated.action is None:
            continue
        normalized = validated.action
        action_key = sha256_hash(_candidate_action_payload(normalized).model_dump(mode="json"))
        if action_key in seen_actions:
            continue
        seen_actions.add(action_key)
        candidates.append(
            StrategicActionCandidate(
                candidate_id=candidate_id,
                label=candidate_id,  # type: ignore[arg-type]
                action=_candidate_action_payload(normalized),
                changed_dimensions=dimensions,
                rationale=rationale,
            )
        )
    if len(candidates) < 2 or candidates[0].candidate_id != "maintain":
        raise ValueError("candidate generator did not produce a valid baseline")
    return tuple(candidates)


def _candidate_to_action(
    candidate: StrategicActionCandidate,
    state: MarketState,
    company_id: str,
) -> CompanyAction:
    payload = candidate.action
    return CompanyAction(
        action_id=(
            f"strategic-rollout:{state.episode_id}:{state.round}:"
            f"{company_id}:{candidate.candidate_id}"
        ),
        episode_id=state.episode_id,
        agent_id=company_id,
        round=state.round,
        state_version=state.state_version,
        price_cents=payload.price_cents,
        advertising_budget_cents=payload.advertising_budget_cents,
        service_budget_cents=payload.service_budget_cents,
        capacity_investment_cents=payload.capacity_investment_cents,
        resilience_budget_cents=payload.resilience_budget_cents,
        shared_resilience_contribution_cents=(
            payload.shared_resilience_contribution_cents
        ),
        incident_response=IncidentResponse(
            IncidentResponseMode(payload.incident_response_mode),
            payload.repair_budget_cents,
        ),
        strategy_summary=f"Stage 6 rollout: {candidate.candidate_id}",
    )


def _sample_direction(
    *,
    scenario_index: int,
    opponent_id: str,
    cut_ppm: int,
    raise_ppm: int,
) -> str:
    digest = sha256_hash(
        {
            "protocol": "opponent-response-quantile-v1",
            "scenario_index": scenario_index,
            "opponent_id": opponent_id,
        }
    )
    draw = int(digest.rsplit(":", 1)[-1][-12:], 16) % PPM
    if draw < cut_ppm:
        return "price_cut"
    if draw < cut_ppm + raise_ppm:
        return "price_raise"
    return "maintain"


def _opponent_response_action(
    *,
    config: MarketConfig,
    state: MarketState,
    opponent_id: str,
    focal_candidate: StrategicActionCandidate,
    scenario_index: int,
    opponent_model: OpponentModelState | None,
    belief_state: BeliefState | None,
) -> CompanyAction:
    baseline = build_rule_action(config, state, opponent_id)
    bounds = config.mapping("action", "bounds", "price_cents")
    cut = 250_000
    raise_ppm = 150_000
    if opponent_model is not None and opponent_id in opponent_model.opponent_models:
        profile = opponent_model.opponent_models[opponent_id]
        strategy = profile.strategy_distribution
        cut = _clip(
            profile.behavior_profile.price_aggressiveness_ppm // 2
            + strategy.growth_ppm // 3,
            50_000,
            850_000,
        )
        raise_ppm = _clip(
            strategy.profit_ppm // 3,
            25_000,
            450_000,
        )
    if belief_state is not None and opponent_id in belief_state.opponent_beliefs:
        distribution = belief_state.opponent_beliefs[
            opponent_id
        ].next_price_direction
        if opponent_model is None:
            cut = distribution.price_cut_ppm
            raise_ppm = distribution.price_raise_ppm
        else:
            # Blend the short-run public action forecast with the slower public
            # strategy model.  Neither source contains private opponent state.
            cut = (cut + distribution.price_cut_ppm) // 2
            raise_ppm = (raise_ppm + distribution.price_raise_ppm) // 2
    if focal_candidate.label in {"price_cut_small", "price_cut_large"}:
        cut = _clip(
            cut + (180_000 if focal_candidate.label == "price_cut_large" else 90_000),
            50_000,
            900_000,
        )
    if cut + raise_ppm > 950_000:
        raise_ppm = 950_000 - cut
    direction = _sample_direction(
        scenario_index=scenario_index,
        opponent_id=opponent_id,
        cut_ppm=cut,
        raise_ppm=raise_ppm,
    )
    price = baseline.price_cents
    if direction == "price_cut":
        price -= 500
    elif direction == "price_raise":
        price += 500
    price = _clip(price, int(bounds["min"]), int(bounds["max"]))
    return replace(
        baseline,
        action_id=f"{baseline.action_id}:response-{scenario_index}",
        price_cents=price,
        strategy_summary=(
            f"{baseline.strategy_summary}; sampled public-model response={direction}"
        ),
    )


class AuthoritativeMarketRolloutEvaluator:
    """Evaluate bounded candidates with true MarketEnv outcomes.

    Output is an oracle label for Stage 6 evaluation.  It must never be copied
    into a public-information Agent observation because the rollout starts
    from the authoritative hidden market state.
    """

    def __init__(self, config: MarketConfig) -> None:
        self.config = config

    def evaluate(
        self,
        *,
        state: MarketState,
        company_id: str,
        persona_profile: PersonaProfile,
        opponent_model: OpponentModelState | Mapping[str, Any] | None = None,
        belief_state: BeliefState | Mapping[str, Any] | None = None,
        horizon_rounds: int = 5,
        scenario_count: int = 10,
        candidates: Sequence[StrategicActionCandidate] | None = None,
    ) -> StrategicReliabilityPlan:
        if state.terminal:
            raise ValueError("cannot evaluate a terminal market state")
        if company_id not in state.company_ids:
            raise KeyError(company_id)
        horizon = min(max(1, horizon_rounds), state.rounds_remaining)
        scenarios = min(max(1, scenario_count), 100)
        parsed_model = (
            opponent_model
            if isinstance(opponent_model, OpponentModelState)
            else (
                OpponentModelState.model_validate(opponent_model)
                if opponent_model is not None
                else None
            )
        )
        parsed_belief = (
            belief_state
            if isinstance(belief_state, BeliefState)
            else (
                BeliefState.model_validate(belief_state)
                if belief_state is not None
                else None
            )
        )
        selected = tuple(candidates or generate_candidate_actions(
            self.config, state, company_id
        ))
        persona_evaluator = PersonaUtilityEvaluator(
            persona_profile,
            profit_scale_cents=self.config.integer(
                "persona_utilities", "profit_scale_cents"
            ),
            share_growth_scale_ppm=self.config.integer(
                "persona_utilities", "share_growth_scale_ppm"
            ),
        )
        initial_ev = _enterprise_value(state, company_id, self.config)
        initial_profit = state.company(company_id).financial.cumulative_profit_cents
        evaluations: list[RolloutCandidateEvaluation] = []

        for candidate in selected:
            outcomes: list[RolloutScenarioOutcome] = []
            for scenario_index in range(scenarios):
                scenario_state = _state_for_scenario(state, scenario_index)
                env = MarketEnv(self.config)
                env.load_state(scenario_state)
                tracker = PersonaUtilityTracker(persona_evaluator)
                incident_loss = 0
                unserved_loss = 0
                discounted_utilities: list[int] = []
                for offset in range(horizon):
                    before = env.get_state()
                    actions = {
                        item: build_rule_action(self.config, before, item)
                        for item in before.company_ids
                    }
                    if offset == 0:
                        actions[company_id] = _candidate_to_action(
                            candidate, before, company_id
                        )
                        for opponent_id in before.company_ids:
                            if opponent_id == company_id:
                                continue
                            actions[opponent_id] = _opponent_response_action(
                                config=self.config,
                                state=before,
                                opponent_id=opponent_id,
                                focal_candidate=candidate,
                                scenario_index=scenario_index,
                                opponent_model=parsed_model,
                                belief_state=parsed_belief,
                            )
                    result = env.step(
                        f"{before.episode_id}:{before.round}:{before.state_version}",
                        actions,
                    )
                    assessment = tracker.record(before, result.state_after, company_id)
                    incident_loss += assessment.realized_incident_loss_cents
                    unserved_loss += assessment.realized_unserved_contribution_loss_cents
                    discounted_utilities.append(
                        assessment.discounted_round_utility_ppm
                    )
                    if result.done:
                        break
                final = env.get_state()
                final_company = final.company(company_id)
                enterprise_value = _enterprise_value(
                    final, company_id, self.config
                )
                best_opponent_ev, competitive_margin, final_rank = (
                    _competitive_labels(final, company_id, self.config)
                )
                total_loss = incident_loss + unserved_loss
                risk_adjusted = enterprise_value - total_loss
                mean_utility = _mean(discounted_utilities)
                persona_bonus = (
                    mean_utility * persona_evaluator.profit_scale_cents // PPM
                )
                outcomes.append(
                    RolloutScenarioOutcome(
                        scenario_index=scenario_index,
                        scenario_seed=scenario_state.episode_seed,
                        final_state_hash=final.state_hash,
                        enterprise_value_cents=enterprise_value,
                        enterprise_value_delta_cents=enterprise_value - initial_ev,
                        best_opponent_enterprise_value_cents=best_opponent_ev,
                        competitive_margin_cents=competitive_margin,
                        final_rank=final_rank,
                        cumulative_profit_delta_cents=(
                            final_company.financial.cumulative_profit_cents
                            - initial_profit
                        ),
                        expected_loss_cents=total_loss,
                        risk_adjusted_value_cents=risk_adjusted,
                        final_market_share_ppm=(
                            final_company.commercial.market_share_ppm
                        ),
                        mean_discounted_persona_utility_ppm=mean_utility,
                        persona_aligned_value_cents=risk_adjusted + persona_bonus,
                    )
                )
            expected_aligned = _mean(
                [item.persona_aligned_value_cents for item in outcomes]
            )
            worst_aligned = min(
                item.persona_aligned_value_cents for item in outcomes
            )
            risk_aversion = persona_profile.traits_ppm.risk_aversion
            certainty_equivalent = expected_aligned - (
                risk_aversion * (expected_aligned - worst_aligned) // PPM
            )
            evaluations.append(
                RolloutCandidateEvaluation(
                    candidate=candidate,
                    outcomes=outcomes,
                    expected_enterprise_value_cents=_mean(
                        [item.enterprise_value_cents for item in outcomes]
                    ),
                    expected_enterprise_value_delta_cents=_mean(
                        [item.enterprise_value_delta_cents for item in outcomes]
                    ),
                    expected_competitive_margin_cents=_mean(
                        [
                            int(item.competitive_margin_cents)
                            for item in outcomes
                        ]
                    ),
                    worst_case_competitive_margin_cents=min(
                        int(item.competitive_margin_cents) for item in outcomes
                    ),
                    expected_final_rank_milli=_mean(
                        [int(item.final_rank) * 1_000 for item in outcomes]
                    ),
                    expected_cumulative_profit_delta_cents=_mean(
                        [item.cumulative_profit_delta_cents for item in outcomes]
                    ),
                    expected_loss_cents=_mean(
                        [item.expected_loss_cents for item in outcomes]
                    ),
                    expected_risk_adjusted_value_cents=_mean(
                        [item.risk_adjusted_value_cents for item in outcomes]
                    ),
                    worst_case_risk_adjusted_value_cents=min(
                        item.risk_adjusted_value_cents for item in outcomes
                    ),
                    expected_persona_utility_ppm=_mean(
                        [
                            item.mean_discounted_persona_utility_ppm
                            for item in outcomes
                        ]
                    ),
                    expected_persona_aligned_value_cents=expected_aligned,
                    worst_case_persona_aligned_value_cents=worst_aligned,
                    certainty_equivalent_value_cents=certainty_equivalent,
                )
            )

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
        payload: dict[str, Any] = {
            "plan_schema_version": "strategic-reliability-plan-v1.0.0",
            "evaluator_version": "authoritative-market-rollout-v1.0.0",
            "episode_id": state.episode_id,
            "round": state.round,
            "state_version": state.state_version,
            "state_hash": state.state_hash,
            "company_id": company_id,
            "persona_id": persona_profile.persona_id,
            "persona_profile_hash": persona_profile.profile_hash,
            "horizon_rounds": horizon,
            "scenario_count": scenarios,
            "evaluations": [item.model_dump(mode="json") for item in evaluations],
            "recommended_candidate_id": ranked.candidate.candidate_id,
            "baseline_candidate_id": baseline.candidate.candidate_id,
            "expected_gain_over_baseline_cents": gain,
            "baseline_regret_cents": max(0, gain),
            "recommendation_is_non_binding": True,
            "uses_authoritative_hidden_market_state": True,
            "allowed_in_public_agent_context": False,
            "claims_nash_equilibrium": False,
            "limitations": [
                "authoritative oracle output is for evaluation, not public Agent input",
                "finite candidate set rather than continuous action optimization",
                "opponent response scenarios are deterministic approximations",
                "rollout horizon is bounded and does not claim equilibrium",
            ],
            "plan_hash": "pending",
        }
        payload["plan_hash"] = compute_reliability_plan_hash(payload)
        return StrategicReliabilityPlan.model_validate(payload)
