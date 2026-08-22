"""Deterministic first-party Agent for local integration tests and demos."""

from __future__ import annotations

import json
from typing import Any

from game_theory_agent.agents.contracts import (
    AgentDecision,
    AgentRequestedAction,
    CommunicationContext,
    DecisionContext,
    ExpectedOutcome,
    IncidentIntent,
    MessageResponse,
    ModelGeneration,
    StrategyPlan,
    SuccessCriteria,
)
from game_theory_agent.interaction.contracts import (
    CommunicationSubmission,
    MessageDraft,
)
from game_theory_agent.cooperation import (
    ProposalResponseDraft,
    SharedResilienceProposalDraft,
)


def _bounded(value: int, field: str, bounds: dict[str, Any]) -> int:
    limit = bounds[field]
    return min(max(value, int(limit["min"])), int(limit["max"]))


class MockModelClient:
    """A transparent heuristic that exercises the same contract as an LLM."""

    def __init__(
        self,
        model_name: str = "mock-balanced-v1",
        *,
        communication_submission: CommunicationSubmission | None = None,
        honor_requested_price: bool = False,
        cooperation_proposal_receiver: str | None = None,
        cooperation_proposal_round: int | None = None,
        cooperation_proposal_target_round: int | None = None,
        cooperation_proposal_amount_cents: int = 1_000_000,
        cooperation_response: str | None = None,
        shared_resilience_contribution_cents: int = 0,
        shared_resilience_contribution_rounds: tuple[int, ...] | None = None,
        honor_shared_resilience_commitments: bool = False,
        minimum_proposer_credibility_ppm: int = 0,
        belief_price_response_cents: int = 0,
        belief_response_threshold_ppm: int = 450_000,
        honor_game_theory_advice: bool = False,
    ) -> None:
        self.model_name = model_name
        self.communication_submission = (
            communication_submission or CommunicationSubmission()
        )
        self.honor_requested_price = honor_requested_price
        self.cooperation_proposal_receiver = cooperation_proposal_receiver
        self.cooperation_proposal_round = cooperation_proposal_round
        self.cooperation_proposal_target_round = cooperation_proposal_target_round
        self.cooperation_proposal_amount_cents = cooperation_proposal_amount_cents
        if cooperation_response not in {None, "accept", "reject"}:
            raise ValueError("cooperation_response must be accept, reject, or None")
        self.cooperation_response = cooperation_response
        self.shared_resilience_contribution_cents = max(
            0, int(shared_resilience_contribution_cents)
        )
        self.shared_resilience_contribution_rounds = (
            None
            if shared_resilience_contribution_rounds is None
            else frozenset(shared_resilience_contribution_rounds)
        )
        self.honor_shared_resilience_commitments = (
            honor_shared_resilience_commitments
        )
        self.minimum_proposer_credibility_ppm = min(
            1_000_000, max(0, int(minimum_proposer_credibility_ppm))
        )
        self.belief_price_response_cents = max(
            0, int(belief_price_response_cents)
        )
        self.belief_response_threshold_ppm = min(
            1_000_000, max(0, int(belief_response_threshold_ppm))
        )
        self.honor_game_theory_advice = bool(honor_game_theory_advice)

    async def generate_communication(
        self, context: CommunicationContext
    ) -> ModelGeneration:
        submission = CommunicationSubmission()
        if context.communication_mode != "off":
            messages = list(self.communication_submission.messages)
            cooperation = context.cooperation or {}
            if (
                cooperation.get("mode") == "shared_resilience_v1"
                and self.cooperation_proposal_receiver is not None
                and self.cooperation_proposal_round == context.round
            ):
                messages = [item for item in messages if item.channel != "private"]
                target_round = self.cooperation_proposal_target_round
                if target_round is None:
                    target_round = context.round + 1
                messages.append(
                    MessageDraft(
                        channel="private",
                        recipients=[self.cooperation_proposal_receiver],
                        speech_act="proposal",
                        content="请在目标轮共同投入行业韧性。",
                        cooperation_proposal=SharedResilienceProposalDraft(
                            target_round=target_round,
                            requested_contribution_cents=(
                                self.cooperation_proposal_amount_cents
                            ),
                        ),
                    )
                )
            pending = list(cooperation.get("pending_proposals_received", []))
            if (
                cooperation.get("mode") == "shared_resilience_v1"
                and self.cooperation_response is not None
                and pending
            ):
                proposal = pending[0]
                response = self.cooperation_response
                proposer_id = proposal["sender_company_id"]
                proposer_credibility = int(
                    cooperation.get("public_credibility", {})
                    .get(proposer_id, {})
                    .get("credibility_ppm", 500_000)
                )
                if (
                    response == "accept"
                    and proposer_credibility
                    < self.minimum_proposer_credibility_ppm
                ):
                    response = "reject"
                messages = [item for item in messages if item.channel != "private"]
                messages.append(
                    MessageDraft(
                        channel="private",
                        recipients=[proposal["sender_company_id"]],
                        speech_act="response",
                        content=(
                            "接受该韧性贡献提议。"
                            if response == "accept"
                            else "拒绝该韧性贡献提议。"
                        ),
                        cooperation_response=ProposalResponseDraft(
                            proposal_id=proposal["proposal_id"],
                            response=response,
                        ),
                    )
                )
            submission = CommunicationSubmission(messages=messages)
        parsed = submission.model_dump(mode="json")
        return ModelGeneration(
            model_name=self.model_name,
            prompt_version="mock-communication-policy-v1.1.0",
            parsed_output=parsed,
            raw_response=json.dumps(parsed, ensure_ascii=False, sort_keys=True),
        )

    async def generate_decision(self, context: DecisionContext) -> ModelGeneration:
        company = context.current_company
        constraints = context.action_constraints
        bounds = constraints["bounds"]
        commercial = company["commercial"]
        operations = company["operations"]

        utilization = int(operations["capacity_utilization_ppm"])
        plan_constraints = (context.current_plan or {}).get("constraints", {})
        price = _bounded(
            max(
                int(commercial["price_cents"]),
                int(plan_constraints.get("minimum_price_cents", 0)),
            ),
            "price_cents",
            bounds,
        )
        if context.belief_state and self.belief_price_response_cents:
            opponent_beliefs = context.belief_state.get(
                "opponent_beliefs", {}
            )
            cut_probability = max(
                (
                    int(item["next_price_direction"]["price_cut_ppm"])
                    for item in opponent_beliefs.values()
                ),
                default=0,
            )
            if cut_probability >= self.belief_response_threshold_ppm:
                price = _bounded(
                    price - self.belief_price_response_cents,
                    "price_cents",
                    bounds,
                )
        if self.honor_game_theory_advice and context.game_theory_advice:
            recommended_price = context.game_theory_advice.get(
                "recommended_price_cents"
            )
            if recommended_price is not None:
                price = _bounded(
                    int(recommended_price), "price_cents", bounds
                )
        message_responses: list[MessageResponse] = []
        communication_view = context.communication_view
        if communication_view is not None:
            for message in communication_view.visible_messages:
                if message.sender_company_id == context.company_id:
                    continue
                requested_price = (
                    message.requested_peer_action.price_cents
                    if message.requested_peer_action is not None
                    else None
                )
                if self.honor_requested_price and requested_price is not None:
                    price = _bounded(requested_price, "price_cents", bounds)
                    message_responses.append(
                        MessageResponse(
                            message_id=message.message_id,
                            disposition="accepted",
                            rationale=(
                                "deterministic test policy accepted the visible "
                                "requested price"
                            ),
                        )
                    )
                else:
                    message_responses.append(
                        MessageResponse(
                            message_id=message.message_id,
                            disposition="ignored",
                            rationale=(
                                "deterministic baseline does not act on this "
                                "visible message"
                            ),
                        )
                    )
        advertising = _bounded(700_000, "advertising_budget_cents", bounds)
        service = _bounded(700_000, "service_budget_cents", bounds)
        capacity = 0
        if constraints.get("capacity_investment_enabled") and utilization > 850_000:
            capacity = _bounded(1_000_000, "capacity_investment_cents", bounds)
        resilience = 0
        if constraints.get("resilience_investment_enabled") and context.risk_signals:
            resilience = _bounded(500_000, "resilience_budget_cents", bounds)
        shared = 0
        if (
            constraints.get("shared_resilience_contribution_enabled")
            and (
                self.shared_resilience_contribution_rounds is None
                or context.round in self.shared_resilience_contribution_rounds
            )
        ):
            shared = _bounded(
                self.shared_resilience_contribution_cents,
                "shared_resilience_contribution_cents",
                bounds,
            )
        if (
            constraints.get("shared_resilience_contribution_enabled")
            and self.honor_shared_resilience_commitments
            and context.cooperation
        ):
            proposals = {
                item["proposal_id"]: item
                for item in context.cooperation.get("proposals_received", [])
            }
            credibility = context.cooperation.get("public_credibility", {})
            for commitment in context.cooperation.get(
                "active_commitments", []
            ):
                if (
                    commitment.get("company_id") != context.company_id
                    or int(commitment.get("target_round", -1)) != context.round
                ):
                    continue
                proposal = proposals.get(commitment.get("proposal_id"))
                if proposal is None:
                    continue
                proposer_id = proposal["sender_company_id"]
                proposer_score = int(
                    credibility.get(proposer_id, {}).get(
                        "credibility_ppm", 500_000
                    )
                )
                if proposer_score >= self.minimum_proposer_credibility_ppm:
                    shared = max(
                        shared,
                        _bounded(
                            int(commitment["promised_contribution_cents"]),
                            "shared_resilience_contribution_cents",
                            bounds,
                        ),
                    )

        phase = (context.current_plan or {}).get("phase", "growth")
        if phase == "liquidity_crisis":
            advertising = service = capacity = resilience = shared = 0
        maximum_spend = int(
            plan_constraints.get(
                "maximum_discretionary_spend_cents",
                advertising + service + capacity + resilience + shared,
            )
        )
        requested_spend = advertising + service + capacity + resilience + shared
        if requested_spend > maximum_spend and requested_spend > 0:
            advertising = advertising * maximum_spend // requested_spend
            service = service * maximum_spend // requested_spend
            capacity = capacity * maximum_spend // requested_spend
            resilience = resilience * maximum_spend // requested_spend
            shared = shared * maximum_spend // requested_spend

        incident = constraints.get("active_incident")
        incident_response = IncidentIntent()
        if incident and int(constraints.get("max_useful_repair_budget_cents", 0)) > 0:
            incident_response = IncidentIntent(
                mode="full_repair",
                repair_budget_cents=int(
                    constraints["max_useful_repair_budget_cents"]
                ),
            )

        factors = ["current price", "cash and capacity utilization"]
        if context.belief_state is not None:
            factors.append("controller-computed public-action belief")
        if context.game_theory_advice is not None:
            factors.append("non-binding game-theory advisor")
        if context.risk_signals:
            factors.append("public risk signals")
        strategy = "Maintain balanced pricing and invest only where current signals justify it."
        decision = AgentDecision(
            plan=StrategyPlan(
                objective=context.objective,
                situation_summary=(
                    f"Round {context.round}: utilization is {utilization} ppm; "
                    f"{len(context.risk_signals)} risk signal(s) are visible."
                ),
                key_factors=factors,
                strategy_summary=strategy,
                expected_outcome=ExpectedOutcome(
                    profit="stable",
                    market_share="stable",
                    capacity="up" if capacity else "stable",
                    risk_exposure="down" if resilience else "stable",
                ),
                success_criteria=SuccessCriteria(
                    minimum_round_profit_cents=0,
                    minimum_cash_reserve_cents=int(
                        plan_constraints.get("minimum_cash_reserve_cents", 0)
                    ),
                    maximum_fixed_spend_cents=maximum_spend,
                ),
            ),
            requested_action=AgentRequestedAction(
                price_cents=price,
                advertising_budget_cents=advertising,
                service_budget_cents=service,
                capacity_investment_cents=capacity,
                resilience_budget_cents=resilience,
                shared_resilience_contribution_cents=(
                    shared
                    if constraints.get(
                        "shared_resilience_contribution_enabled"
                    )
                    else None
                ),
                incident_response=incident_response,
                strategy_summary=strategy,
            ),
            confidence_ppm=650_000,
            message_responses=message_responses,
        )
        parsed = decision.model_dump(mode="json")
        return ModelGeneration(
            model_name=self.model_name,
            prompt_version="mock-policy-v1.1.0",
            parsed_output=parsed,
            raw_response=json.dumps(parsed, ensure_ascii=False, sort_keys=True),
        )
