"""Reproducible uniformly random legal-intent opponent."""

from __future__ import annotations

import hashlib
import json
import random
from typing import Any

from game_theory_agent.agents.contracts import (
    AgentDecision,
    AgentRequestedAction,
    CommunicationContext,
    DecisionContext,
    ExpectedOutcome,
    IncidentIntent,
    ModelGeneration,
    StrategyPlan,
    SuccessCriteria,
)
from game_theory_agent.interaction.contracts import CommunicationSubmission


class UniformRandomIntentPolicy:
    def __init__(self, policy_seed: int) -> None:
        self.policy_seed = policy_seed

    def _rng(self, context: DecisionContext) -> random.Random:
        payload = (
            f"{self.policy_seed}|{context.episode_id}|"
            f"{context.round}|{context.company_id}"
        ).encode()
        seed = int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")
        return random.Random(seed)

    def sample(self, context: DecisionContext) -> AgentRequestedAction:
        rng = self._rng(context)
        constraints = context.action_constraints
        bounds = constraints["bounds"]
        plan_constraints = (context.current_plan or {}).get("constraints", {})

        def draw(field: str) -> int:
            raw = bounds[field]
            return rng.randint(int(raw["min"]), int(raw["max"]))

        price_low = max(
            int(bounds["price_cents"]["min"]),
            int(plan_constraints.get("minimum_price_cents", 0)),
        )
        price_high = int(bounds["price_cents"]["max"])
        price = rng.randint(min(price_low, price_high), price_high)
        available = min(
            int(constraints["cash_available_cents"]),
            int(
                plan_constraints.get(
                    "maximum_discretionary_spend_cents",
                    constraints["cash_available_cents"],
                )
            ),
        )
        fields = (
            "advertising_budget_cents",
            "service_budget_cents",
            "capacity_investment_cents",
            "resilience_budget_cents",
        )
        enabled = {
            "capacity_investment_cents": bool(
                constraints["capacity_investment_enabled"]
            ),
            "resilience_budget_cents": bool(
                constraints["resilience_investment_enabled"]
            ),
        }
        budgets = {field: 0 for field in fields}
        if available > 0:
            for _ in range(2000):
                candidate = {
                    field: draw(field) if enabled.get(field, True) else 0
                    for field in fields
                }
                if sum(candidate.values()) <= available:
                    budgets = candidate
                    break
            else:
                raw = {
                    field: draw(field) if enabled.get(field, True) else 0
                    for field in fields
                }
                total = sum(raw.values())
                if total > 0:
                    allocated = 0
                    active_fields = [field for field in fields if raw[field] > 0]
                    for field in active_fields[:-1]:
                        budgets[field] = raw[field] * available // total
                        allocated += budgets[field]
                    budgets[active_fields[-1]] = available - allocated

        incident_response = IncidentIntent()
        incident = constraints.get("active_incident")
        useful = min(
            int(constraints.get("max_useful_repair_budget_cents", 0)),
            max(0, available - sum(budgets.values())),
        )
        if incident and useful > 0:
            repair = rng.randint(0, useful)
            if repair > 0:
                incident_response = IncidentIntent(
                    mode=rng.choice(("partial_repair", "full_repair")),
                    repair_budget_cents=repair,
                )

        return AgentRequestedAction(
            price_cents=price,
            incident_response=incident_response,
            strategy_summary=(
                "reproducible uniform-random legal intent; "
                "conditioned only on execution constraints"
            ),
            **budgets,
        )


class UniformRandomModelClient:
    def __init__(self, policy_seed: int) -> None:
        self.policy = UniformRandomIntentPolicy(policy_seed)
        self.model_name = "uniform-random-intent-v1"

    async def generate_communication(
        self, context: CommunicationContext
    ) -> ModelGeneration:
        """Rule benchmark intentionally remains silent in communication phases."""

        submission = CommunicationSubmission()
        parsed = submission.model_dump(mode="json")
        return ModelGeneration(
            model_name=self.model_name,
            prompt_version="uniform-random-communication-policy-v1.0.0",
            parsed_output=parsed,
            raw_response=json.dumps(parsed, ensure_ascii=False, sort_keys=True),
        )

    async def generate_decision(self, context: DecisionContext) -> ModelGeneration:
        action = self.policy.sample(context)
        plan_constraints: dict[str, Any] = (
            (context.current_plan or {}).get("constraints", {})
        )
        decision = AgentDecision(
            plan=StrategyPlan(
                objective="sample a reproducible legal random intent",
                situation_summary=(
                    "No market strategy is used; randomness is conditioned only "
                    "on current execution feasibility."
                ),
                key_factors=["action bounds", "cash envelope", "round availability"],
                strategy_summary="uniform random benchmark",
                expected_outcome=ExpectedOutcome(),
                success_criteria=SuccessCriteria(
                    minimum_round_profit_cents=0,
                    minimum_cash_reserve_cents=int(
                        plan_constraints.get("minimum_cash_reserve_cents", 0)
                    ),
                    maximum_fixed_spend_cents=int(
                        plan_constraints.get(
                            "maximum_discretionary_spend_cents", 0
                        )
                    ),
                ),
            ),
            requested_action=action,
            confidence_ppm=0,
        )
        parsed = decision.model_dump(mode="json")
        return ModelGeneration(
            model_name=self.model_name,
            prompt_version="uniform-random-policy-v1.0.0",
            parsed_output=parsed,
            raw_response=json.dumps(parsed, ensure_ascii=False, sort_keys=True),
        )
