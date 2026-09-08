"""Submit a pre-chosen economic intent through the Agent contract."""

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
    ModelGeneration,
    StrategyPlan,
    SuccessCriteria,
)
from game_theory_agent.interaction.contracts import CommunicationSubmission


def _bounded(value: int, field: str, bounds: dict[str, Any]) -> int:
    limit = bounds[field]
    return min(max(int(value), int(limit["min"])), int(limit["max"]))


class FixedActionModelClient:
    """Human or scripted seat: communication is silence, decision is fixed."""

    def __init__(
        self,
        requested: AgentRequestedAction | None = None,
        *,
        model_name: str = "human-fixed-v1",
    ) -> None:
        self.model_name = model_name
        self.requested = requested or AgentRequestedAction(price_cents=10_000)
        self.communication = CommunicationSubmission()

    def set_communication(self, payload: dict[str, Any] | None) -> None:
        self.communication = CommunicationSubmission.model_validate(payload or {})

    def set_requested(self, requested: AgentRequestedAction) -> None:
        self.requested = requested

    async def generate_communication(
        self, context: CommunicationContext
    ) -> ModelGeneration:
        del context
        parsed = self.communication.model_dump(mode="json")
        return ModelGeneration(
            model_name=self.model_name,
            prompt_version="human-communication-silence-v1.0.0",
            parsed_output=parsed,
            raw_response=json.dumps(parsed, ensure_ascii=False, sort_keys=True),
        )

    async def generate_decision(self, context: DecisionContext) -> ModelGeneration:
        bounds = context.action_constraints["bounds"]
        requested = self.requested
        shared_enabled = bool(
            context.action_constraints.get("shared_resilience_contribution_enabled")
        )
        shared = requested.shared_resilience_contribution_cents
        action = AgentRequestedAction(
            price_cents=_bounded(requested.price_cents, "price_cents", bounds),
            advertising_budget_cents=_bounded(
                requested.advertising_budget_cents,
                "advertising_budget_cents",
                bounds,
            ),
            service_budget_cents=_bounded(
                requested.service_budget_cents, "service_budget_cents", bounds
            ),
            capacity_investment_cents=_bounded(
                requested.capacity_investment_cents,
                "capacity_investment_cents",
                bounds,
            ),
            resilience_budget_cents=_bounded(
                requested.resilience_budget_cents,
                "resilience_budget_cents",
                bounds,
            ),
            shared_resilience_contribution_cents=(
                _bounded(
                    0 if shared is None else shared,
                    "shared_resilience_contribution_cents",
                    bounds,
                )
                if shared_enabled
                else None
            ),
            **{key:getattr(requested,key) for key in ("contract_quantity_orders", "contract_duration_rounds", "contract_bid_cents")},
            primary_supplier_id=(requested.primary_supplier_id if context.action_constraints.get('supply_chain_enabled') else None),
            backup_supplier_id=(requested.backup_supplier_id if context.action_constraints.get('supply_chain_enabled') else None),
            primary_supplier_share_ppm=(requested.primary_supplier_share_ppm if context.action_constraints.get('supply_chain_enabled') else None),
            procurement_quantity_orders=(
                _bounded(requested.procurement_quantity_orders, 'procurement_quantity_orders', bounds)
                if context.action_constraints.get('procurement_quantity_enabled') and requested.procurement_quantity_orders is not None else None
            ),
            threshold_project_contribution_cents=(requested.threshold_project_contribution_cents if context.action_constraints.get('threshold_project_contribution_enabled') else None),
            mutual_aid_partner_company_id=(requested.mutual_aid_partner_company_id if context.action_constraints.get('mutual_aid_enabled') else None),
            mutual_aid_capacity_offer_orders=(requested.mutual_aid_capacity_offer_orders if context.action_constraints.get('mutual_aid_enabled') else None),
            mutual_aid_capacity_request_orders=(requested.mutual_aid_capacity_request_orders if context.action_constraints.get('mutual_aid_enabled') else None),
            price_coordination_partner_company_id=(requested.price_coordination_partner_company_id if context.action_constraints.get('price_coordination_enabled') else None),
            price_coordination_target_cents=(requested.price_coordination_target_cents if context.action_constraints.get('price_coordination_enabled') else None),
            incident_response=requested.incident_response or IncidentIntent(),
            strategy_summary=requested.strategy_summary or "human-submitted action",
        )
        decision = AgentDecision(
            plan=StrategyPlan(
                objective=context.objective,
                situation_summary="Human participant submitted this round's action.",
                key_factors=["human price", "human advertising", "human contribution"],
                strategy_summary=action.strategy_summary,
                expected_outcome=ExpectedOutcome(),
                success_criteria=SuccessCriteria(),
            ),
            requested_action=action,
            confidence_ppm=1_000_000,
        )
        parsed = decision.model_dump(mode="json")
        return ModelGeneration(
            model_name=self.model_name,
            prompt_version="human-fixed-action-v1.0.0",
            parsed_output=parsed,
            raw_response=json.dumps(parsed, ensure_ascii=False, sort_keys=True),
        )
