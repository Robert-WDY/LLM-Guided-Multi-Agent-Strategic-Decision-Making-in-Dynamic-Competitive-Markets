"""Deterministic approximate Bayesian response over public price beliefs."""

from __future__ import annotations

from typing import Any, Mapping

from game_theory_agent.advisor.contracts import (
    BayesianCandidateEvaluation,
    GameTheoryAdvice,
    compute_advice_hash,
    PredictedOpponentResponse,
    StrategicCandidateEvaluation,
    StrategicGameTheoryAdvice,
    compute_strategic_advice_hash,
)
from game_theory_agent.belief import BeliefState, compute_belief_hash
from game_theory_agent.opponent import (
    OpponentModelState,
    compute_opponent_model_hash,
)
from game_theory_agent.utility_inference import (
    UtilityInferenceState,
    compute_utility_inference_hash,
)


class BayesianGameAdvisor:
    """Score bounded own-price candidates under opponent direction beliefs.

    This is deliberately an approximate advisor, not a market solver.  It
    marginalizes over the public direction distribution, exposes its proxy
    assumptions, and cannot submit an action.
    """

    def advise(
        self,
        *,
        belief_state: BeliefState | Mapping[str, Any],
        own_company: Mapping[str, Any],
        action_constraints: Mapping[str, Any],
    ) -> GameTheoryAdvice:
        belief = (
            belief_state
            if isinstance(belief_state, BeliefState)
            else BeliefState.model_validate(belief_state)
        )
        current = int(own_company["commercial"]["price_cents"])
        unit_cost = int(own_company["operations"]["actual_unit_cost_cents"])
        price_bounds = action_constraints["bounds"]["price_cents"]
        minimum = int(price_bounds["min"])
        maximum = int(price_bounds["max"])
        candidates = sorted(
            {
                minimum,
                max(minimum, current - 500),
                current,
                min(maximum, current + 500),
                maximum,
            }
        )
        pressures: list[int] = []
        downside_pressures: list[int] = []
        for opponent in belief.opponent_beliefs.values():
            distribution = opponent.next_price_direction
            # A likely opponent cut is positive competitive pressure; a raise
            # relieves pressure.  Values are exact integer expectations.
            pressures.append(
                distribution.price_cut_ppm - distribution.price_raise_ppm
            )
            downside_pressures.append(distribution.price_cut_ppm)
        expected_pressure = (
            sum(pressures) // len(pressures) if pressures else 0
        )
        downside_pressure = (
            sum(downside_pressures) // len(downside_pressures)
            if downside_pressures
            else 0
        )
        evaluations: list[BayesianCandidateEvaluation] = []
        for candidate in candidates:
            margin = candidate - unit_cost
            own_price_response = (current - candidate) * 800
            demand_index = max(
                100_000,
                min(1_900_000, 1_000_000 + own_price_response - expected_pressure // 2),
            )
            downside_index = max(
                100_000,
                min(1_900_000, 1_000_000 + own_price_response - downside_pressure),
            )
            evaluations.append(
                BayesianCandidateEvaluation(
                    price_cents=candidate,
                    expected_margin_index_cents=margin,
                    expected_competitive_pressure_ppm=expected_pressure,
                    expected_payoff_proxy=margin * demand_index,
                    downside_payoff_proxy=margin * downside_index,
                )
            )
        best = max(
            evaluations,
            key=lambda item: (
                item.expected_payoff_proxy,
                item.downside_payoff_proxy,
                -abs(item.price_cents - current),
                -item.price_cents,
            ),
        )
        payload: dict[str, Any] = {
            "advice_schema_version": "bayesian-price-advice-v1.0.0",
            "advisor_mode": "bayesian_price_v1",
            "advisor_model_version": (
                "independent-direction-payoff-proxy-v1.0.0"
            ),
            "episode_id": belief.episode_id,
            "round": belief.prediction_target_round,
            "state_version": belief.state_version,
            "company_id": belief.observer_company_id,
            "belief_hash": compute_belief_hash(belief),
            "current_price_cents": current,
            "unit_cost_cents": unit_cost,
            "candidates": [item.model_dump(mode="json") for item in evaluations],
            "recommended_price_cents": best.price_cents,
            "recommendation_is_non_binding": True,
            "uses_hidden_opponent_state": False,
            "limitations": [
                "uses an independent opponent direction approximation",
                "payoff is a transparent proxy rather than MarketEnv profit",
                "does not infer opponent cash, costs, persona, or utility",
            ],
            "advice_hash": "pending",
        }
        payload["advice_hash"] = compute_advice_hash(payload)
        return GameTheoryAdvice.model_validate(payload)


class BayesianStrategyAdvisor:
    """Finite-action approximate Bayesian best response.

    The response model uses only public opponent strategy/utility beliefs.  It
    estimates how likely each opponent is to answer an own price move, then
    ranks own candidates with an auditable expected-utility proxy.
    """

    def advise(
        self,
        *,
        belief_state: BeliefState | Mapping[str, Any],
        opponent_model: OpponentModelState | Mapping[str, Any],
        utility_inference: UtilityInferenceState | Mapping[str, Any],
        own_company: Mapping[str, Any],
        action_constraints: Mapping[str, Any],
    ) -> StrategicGameTheoryAdvice:
        belief = (
            belief_state
            if isinstance(belief_state, BeliefState)
            else BeliefState.model_validate(belief_state)
        )
        model = (
            opponent_model
            if isinstance(opponent_model, OpponentModelState)
            else OpponentModelState.model_validate(opponent_model)
        )
        utility = (
            utility_inference
            if isinstance(utility_inference, UtilityInferenceState)
            else UtilityInferenceState.model_validate(utility_inference)
        )
        if (
            belief.observer_company_id != model.observer_company_id
            or model.observer_company_id != utility.observer_company_id
        ):
            raise ValueError("advisor inputs belong to different observers")
        if (
            belief.episode_id != model.episode_id
            or model.episode_id != utility.episode_id
        ):
            raise ValueError("advisor inputs belong to different episodes")
        model_hash = compute_opponent_model_hash(model)
        utility_hash = compute_utility_inference_hash(utility)
        if utility.opponent_model_hash != model_hash:
            raise ValueError("utility inference is not bound to opponent model")

        current = int(own_company["commercial"]["price_cents"])
        current_share = int(own_company["commercial"]["market_share_ppm"])
        unit_cost = int(own_company["operations"]["actual_unit_cost_cents"])
        price_bounds = action_constraints["bounds"]["price_cents"]
        minimum = int(price_bounds["min"])
        maximum = int(price_bounds["max"])
        candidate_specs = (
            ("aggressive_price_cut", max(minimum, current - 1_000)),
            ("price_cut", max(minimum, current - 500)),
            ("maintain", current),
            ("price_raise", min(maximum, current + 500)),
        )
        evaluations: list[StrategicCandidateEvaluation] = []
        for label, candidate in candidate_specs:
            responses: list[PredictedOpponentResponse] = []
            response_cuts: list[int] = []
            for company_id, profile in sorted(model.opponent_models.items()):
                inferred = utility.opponent_utilities[company_id]
                strategy = profile.strategy_distribution
                public_action = belief.opponent_beliefs[company_id].next_price_direction
                own_cut = max(0, current - candidate)
                own_raise = max(0, candidate - current)
                retaliation = (
                    profile.behavior_profile.price_aggressiveness_ppm * 2 // 5
                    + strategy.growth_ppm // 4
                    + inferred.market_share.mean_ppm // 5
                    + inferred.growth.mean_ppm // 5
                    + public_action.price_cut_ppm // 5
                    + min(250_000, own_cut * 250)
                    - min(120_000, own_raise * 120)
                )
                cut = max(25_000, min(900_000, retaliation))
                raise_ppm = max(
                    25_000,
                    min(
                        500_000,
                        strategy.profit_ppm // 3
                        + public_action.price_raise_ppm // 3
                        + min(150_000, own_raise * 150),
                    ),
                )
                if cut + raise_ppm > 950_000:
                    raise_ppm = 950_000 - cut
                maintain = 1_000_000 - cut - raise_ppm
                responses.append(
                    PredictedOpponentResponse(
                        opponent_company_id=company_id,
                        price_cut_ppm=cut,
                        maintain_ppm=maintain,
                        price_raise_ppm=raise_ppm,
                    )
                )
                response_cuts.append(cut)
            retaliation_ppm = (
                sum(response_cuts) // len(response_cuts)
                if response_cuts
                else 0
            )
            own_price_gain = (current - candidate) * 260
            expected_share = max(
                0,
                min(
                    1_000_000,
                    current_share
                    + own_price_gain
                    - retaliation_ppm // 5,
                ),
            )
            demand_index = max(
                100_000,
                min(
                    1_800_000,
                    1_000_000
                    + own_price_gain
                    - retaliation_ppm // 3,
                ),
            )
            margin = candidate - unit_cost
            expected_profit = margin * demand_index
            risk = max(
                0,
                min(
                    1_000_000,
                    retaliation_ppm
                    + (200_000 if candidate < current - 500 else 0),
                ),
            )
            expected_utility = (
                expected_profit // 100
                + expected_share * 120
                - risk * 80
            )
            worst_case = (
                margin * max(100_000, demand_index - risk) // 100
                + max(0, expected_share - risk // 3) * 100
                - risk * 100
            )
            evaluations.append(
                StrategicCandidateEvaluation(
                    action_label=label,  # type: ignore[arg-type]
                    price_cents=candidate,
                    predicted_opponent_responses=responses,
                    expected_profit_proxy=expected_profit,
                    expected_market_share_ppm=expected_share,
                    strategic_risk_ppm=risk,
                    expected_utility_proxy=expected_utility,
                    worst_case_utility_proxy=worst_case,
                )
            )
        best = max(
            evaluations,
            key=lambda item: (
                item.expected_utility_proxy,
                item.worst_case_utility_proxy,
                -item.strategic_risk_ppm,
                -abs(item.price_cents - current),
            ),
        )
        payload: dict[str, Any] = {
            "advice_schema_version": "bayesian-strategy-advice-v2.0.0",
            "advisor_mode": "bayesian_strategy_v2",
            "advisor_model_version": "expected-strategic-response-v2.0.0",
            "episode_id": belief.episode_id,
            "round": belief.prediction_target_round,
            "state_version": belief.state_version,
            "company_id": belief.observer_company_id,
            "belief_hash": compute_belief_hash(belief),
            "opponent_model_hash": model_hash,
            "utility_inference_hash": utility_hash,
            "current_price_cents": current,
            "unit_cost_cents": unit_cost,
            "candidate_actions": [
                item.model_dump(mode="json") for item in evaluations
            ],
            "recommended_action": best.action_label,
            "recommended_price_cents": best.price_cents,
            "recommendation_reason": (
                "marginalized public opponent strategy and inferred utility "
                "responses; selected highest expected utility proxy"
            ),
            "recommendation_is_non_binding": True,
            "approximate_best_response": True,
            "claims_nash_equilibrium": False,
            "uses_hidden_opponent_state": False,
            "limitations": [
                "finite price actions only",
                "opponent responses are conditionally independent approximations",
                "utility and payoff values are auditable proxies",
                "does not solve a Bayesian Nash equilibrium",
            ],
            "advice_hash": "pending",
        }
        payload["advice_hash"] = compute_strategic_advice_hash(payload)
        return StrategicGameTheoryAdvice.model_validate(payload)
